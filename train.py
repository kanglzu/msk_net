"""
Training Script for MSK-Net
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
import numpy as np
import random
from tqdm import tqdm
import datetime
import json
from typing import Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from models import MSKNet, build_msk_net_from_config
from datasets import create_data_loaders
from losses import get_loss_function, compute_batch_metrics


class WarmupCosineScheduler:
    """
    Cosine Annealing LR Scheduler with Linear Warmup
    
    Uses linear warmup phase followed by cosine annealing decay.
    """
    
    def __init__(self,
                 optimizer,
                 warmup_epochs: int = 10,
                 total_epochs: int = 300,
                 warmup_start_lr: float = 1e-6,
                 base_lr: float = 1e-4,
                 eta_min: float = 1e-7):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.warmup_start_lr = warmup_start_lr
        self.base_lr = base_lr
        self.eta_min = eta_min
        self.current_epoch = 0
        
        # Create cosine scheduler for post-warmup phase
        self.cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_epochs - warmup_epochs,
            eta_min=eta_min
        )
    
    def step(self):
        """Update learning rate"""
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup phase
            lr = (self.warmup_start_lr + 
                  (self.base_lr - self.warmup_start_lr) * 
                  self.current_epoch / self.warmup_epochs)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            # Cosine annealing phase
            self.cosine_scheduler.step()
        
        self.current_epoch += 1
    
    def get_last_lr(self):
        """Get current learning rate"""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]


class EarlyStopping:
    """
    Early Stopping to prevent overfitting
    """
    
    def __init__(self, patience: int = 15, mode: str = 'max', min_delta: float = 0.0):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float):
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
    
    def _is_improvement(self, score: float) -> bool:
        if self.mode == 'max':
            return score > self.best_score + self.min_delta
        else:
            return score < self.best_score - self.min_delta


def set_seed(seed: int):
    """
    Set random seeds for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def train_epoch(model,
                loader,
                optimizer,
                criterion,
                device,
                epoch,
                total_epochs,
                use_deep_supervision=True,
                ds_weights=None,
                grad_clip_norm=1.0):
    """
    Train for one epoch
    
    Args:
        model: MSK-Net model
        loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        epoch: Current epoch number
        total_epochs: Total number of epochs
        use_deep_supervision: Whether using deep supervision
        ds_weights: Deep supervision weights [0.4, 0.2, 0.1]
        grad_clip_norm: Gradient clipping max norm
    
    Returns:
        Average training loss for the epoch
    """
    model.train()
    running_loss = 0.0
    num_samples = 0
    
    pbar = tqdm(loader, desc=f"Train Epoch [{epoch+1}/{total_epochs}]")
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)
        
        batch_size = images.size(0)
        num_samples += batch_size
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        
        # Calculate loss
        if use_deep_supervision and isinstance(outputs, list):
            # Main output + auxiliary outputs
            main_out = outputs[0]
            aux_outs = outputs[1:]
            
            # Main loss
            main_loss = criterion(main_out, labels)
            
            # Auxiliary losses
            if ds_weights is None:
                ds_weights = [0.4, 0.2, 0.1][:len(aux_outs)]
            
            aux_loss = 0.0
            for weight, aux_out in zip(ds_weights, aux_outs):
                aux_loss += weight * criterion(aux_out, labels)
            
            total_loss = main_loss + aux_loss
        else:
            # Single output (evaluation mode or no deep supervision)
            if isinstance(outputs, list):
                outputs = outputs[0]
            total_loss = criterion(outputs, labels)
            main_loss = total_loss
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        
        optimizer.step()
        
        # Update running loss
        running_loss += total_loss.item() * batch_size
        
        # Calculate metrics for logging
        with torch.no_grad():
            main_out_for_metric = outputs[0] if isinstance(outputs, list) else outputs
            dice_scores = compute_batch_metrics(
                main_out_for_metric, labels, ['dice']
            )['dice'].mean().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{total_loss.item():.4f}',
            'main_loss': f'{main_loss.item():.4f}',
            'avg_loss': f'{running_loss/num_samples:.4f}',
            'dice': f'{dice_scores:.4f}'
        })
    
    avg_loss = running_loss / num_samples if num_samples > 0 else 0.0
    return avg_loss


def validate_epoch(model,
                   loader,
                   criterion,
                   device,
                   eval_type="Validation"):
    """
    Validation/Test for one epoch
    
    Args:
        model: MSK-Net model
        loader: Validation/test data loader
        criterion: Loss function
        device: Device
        eval_type: "Validation" or "Test" for logging
    
    Returns:
        Dictionary with metrics
    """
    model.eval()
    running_loss = 0.0
    num_samples = 0
    
    # Accumulate Dice scores for all classes
    total_dice = torch.zeros(3, device=device)  # For WT, TC, ET
    
    pbar = tqdm(loader, desc=f"{eval_type}")
    
    with torch.no_grad():
        for batch in pbar:
            images = batch['image'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)
            
            batch_size = images.size(0)
            num_samples += batch_size
            
            # Forward pass (returns single output in eval mode)
            outputs = model(images)
            if isinstance(outputs, list):
                outputs = outputs[0]
            
            # Calculate loss
            loss = criterion(outputs, labels)
            running_loss += loss.item() * batch_size
            
            # Calculate metrics
            metrics = compute_batch_metrics(outputs, labels, ['dice'])
            dice_batch = metrics['dice']  # [B, C]
            total_dice += dice_batch.sum(dim=0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dice': f'{dice_batch.mean().item():.4f}'
            })
    
    # Calculate averages
    avg_loss = running_loss / num_samples if num_samples > 0 else 0.0
    avg_dice_per_class = total_dice / num_samples if num_samples > 0 else torch.zeros(3, device=device)
    avg_dice = avg_dice_per_class.mean().item()
    
    return {
        'loss': avg_loss,
        'dice_mean': avg_dice,
        'dice_per_class': avg_dice_per_class.cpu().numpy(),
        'dice_wt': avg_dice_per_class[0].item(),
        'dice_tc': avg_dice_per_class[1].item(),
        'dice_et': avg_dice_per_class[2].item()
    }


def train(config_path: str, resume_from: Optional[str] = None):
    """
    Main training function
    
    Args:
        config_path: Path to configuration YAML file
        resume_from: Path to checkpoint to resume from
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set random seed
    set_seed(config['hardware']['seed'])
    
    # Setup device
    device = torch.device(config['hardware']['device'] 
                         if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directories
    output_dir = Path(config['paths']['output_dir'])
    checkpoint_dir = Path(config['paths']['checkpoint_dir'])
    log_dir = Path(config['paths']['log_dir'])
    
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_save_path = output_dir / 'config.yaml'
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Build model
    print("Building MSK-Net...")
    model = build_msk_net_from_config(config)
    model = model.to(device)
    
    # Print model info
    param_count = model.get_parameter_count()
    print(f"Model Parameters: {param_count:.2f}M (Expected: 30.61M)")
    
    # Multi-GPU training if available
    if torch.cuda.device_count() > 1 and config['hardware']['num_gpus'] > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    # Create data loaders
    print("Loading datasets...")
    train_loader, val_loader, test_loader = create_data_loaders(config)
    print(f"Train samples: {len(train_loader.dataset)}")
    if val_loader:
        print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Setup loss function
    criterion = get_loss_function(config)
    criterion = criterion.to(device)
    
    # Setup optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config['training']['optimizer']['lr'],
        betas=config['training']['optimizer']['betas'],
        weight_decay=config['training']['optimizer']['weight_decay'],
        eps=config['training']['optimizer']['eps']
    )
    
    # Setup learning rate scheduler with warmup
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=config['training']['scheduler']['warmup_epochs'],
        total_epochs=config['training']['epochs'],
        warmup_start_lr=config['training']['scheduler']['warmup_start_lr'],
        base_lr=config['training']['optimizer']['lr'],
        eta_min=config['training']['scheduler']['eta_min']
    )
    
    # Setup early stopping
    early_stopping = EarlyStopping(
        patience=config['training']['early_stopping']['patience'],
        mode=config['training']['early_stopping']['mode']
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_dice = 0.0
    
    if resume_from and os.path.exists(resume_from):
        print(f"Resuming from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        
        # Load model state
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer and scheduler
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.current_epoch = checkpoint['epoch'] + 1
        
        start_epoch = checkpoint['epoch'] + 1
        best_dice = checkpoint.get('best_dice', 0.0)
        print(f"Resumed from epoch {start_epoch}, best Dice: {best_dice:.4f}")
    
    # Training log file
    log_file = log_dir / f"{config['logging']['exp_name']}_train_log.txt"
    with open(log_file, 'a') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"Training started: {datetime.datetime.now()}\n")
        f.write(f"Configuration: {config_path}\n")
        f.write(f"{'='*80}\n\n")
    
    # Training loop
    print(f"\nStarting training for {config['training']['epochs']} epochs...")
    
    for epoch in range(start_epoch, config['training']['epochs']):
        epoch_start_time = datetime.datetime.now()
        
        # Train for one epoch
        train_loss = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            total_epochs=config['training']['epochs'],
            use_deep_supervision=config['model']['deep_supervision']['enabled'],
            ds_weights=config['model']['deep_supervision']['weights'],
            grad_clip_norm=config['training']['grad_clip']['max_norm']
        )
        
        # Validation
        if val_loader is not None:
            val_metrics = validate_epoch(model, val_loader, criterion, device, "Validation")
        else:
            # Use test set for validation if no val set
            val_metrics = validate_epoch(model, test_loader, criterion, device, "Test")
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Logging
        log_message = (
            f"Epoch [{epoch+1}/{config['training']['epochs']}] "
            f"Time: {(datetime.datetime.now() - epoch_start_time).total_seconds():.1f}s "
            f"LR: {current_lr:.2e} "
            f"Train Loss: {train_loss:.4f} "
            f"Val Dice: {val_metrics['dice_mean']:.4f} "
            f"(WT: {val_metrics['dice_wt']:.4f}, "
            f"TC: {val_metrics['dice_tc']:.4f}, "
            f"ET: {val_metrics['dice_et']:.4f})"
        )
        print(log_message)
        
        with open(log_file, 'a') as f:
            f.write(log_message + '\n')
        
        # Save checkpoint
        is_best = val_metrics['dice_mean'] > best_dice
        if is_best:
            best_dice = val_metrics['dice_mean']
        
        # Get model state dict
        if isinstance(model, nn.DataParallel):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'best_dice': best_dice,
            'config': config
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, checkpoint_dir / 'latest.pth')
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, checkpoint_dir / 'best.pth')
            print(f"New best model saved! Dice: {best_dice:.4f}")
        
        # Early stopping check
        early_stopping(val_metrics['dice_mean'])
        if early_stopping.early_stop:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
    
    print(f"\nTraining completed!")
    print(f"Best Dice Score: {best_dice:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Train MSK-Net')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration YAML file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    train(args.config, args.resume)


if __name__ == '__main__':
    main()

