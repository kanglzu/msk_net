"""
Utility Functions for MSK-Net

Helper functions for configuration loading, logging, visualization, etc.
"""

import os
import yaml
import torch
import random
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to YAML configuration file
    
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def merge_configs(base_config_path: str, override_config_path: str) -> Dict[str, Any]:
    """
    Merge two configuration files (e.g., base + dataset-specific)
    
    Args:
        base_config_path: Path to base configuration
        override_config_path: Path to override configuration
    
    Returns:
        Merged configuration dictionary
    """
    base_config = load_config(base_config_path)
    override_config = load_config(override_config_path)
    
    # Deep merge
    def deep_update(base_dict, update_dict):
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict:
                base_dict[key] = deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
        return base_dict
    
    merged = deep_update(base_config, override_config)
    return merged


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, save_path):
    """
    Save training checkpoint
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Scheduler state
        epoch: Current epoch
        metrics: Validation metrics
        save_path: Path to save checkpoint
    """
    # Handle DataParallel
    if isinstance(model, torch.nn.DataParallel):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler if scheduler is None else getattr(scheduler, 'state_dict', lambda: {})(),
        'metrics': metrics
    }
    
    torch.save(checkpoint, save_path)


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, device='cuda'):
    """
    Load training checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to load state
        scheduler: Optional scheduler to load state
        device: Device to map checkpoint to
    
    Returns:
        Loaded checkpoint dictionary
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer and scheduler if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        if hasattr(scheduler, 'load_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint


def visualize_prediction(image, prediction, ground_truth, save_path=None):
    """
    Visualize segmentation results
    
    Args:
        image: Input image [C, H, W, D]
        prediction: Predicted mask [3, H, W, D]
        ground_truth: Ground truth mask [3, H, W, D]
        save_path: Optional path to save figure
    """
    # Take middle slice
    slice_idx = image.shape[-1] // 2
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Show FLAIR modality (index 3)
    axes[0, 0].imshow(image[3, :, :, slice_idx].T, cmap='gray')
    axes[0, 0].set_title('FLAIR Input')
    axes[0, 0].axis('off')
    
    # Show predictions for each class
    class_names = ['WT', 'TC', 'ET']
    for i, class_name in enumerate(class_names):
        axes[0, i+1].imshow(prediction[i, :, :, slice_idx].T, cmap='hot')
        axes[0, i+1].set_title(f'Predicted {class_name}')
        axes[0, i+1].axis('off')
        
        axes[1, i].imshow(ground_truth[i, :, :, slice_idx].T, cmap='hot')
        axes[1, i].set_title(f'Ground Truth {class_name}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def print_model_summary(model, input_size=(4, 128, 128, 128)):
    """
    Print model summary with parameter counts
    
    Args:
        model: MSK-Net model
        input_size: Input tensor size
    """
    print("\n" + "="*80)
    print("MODEL SUMMARY")
    print("="*80)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total Parameters: {total_params/1e6:.2f}M")
    print(f"Trainable Parameters: {trainable_params/1e6:.2f}M")
    
    # Try to estimate memory
    try:
        dummy_input = torch.randn(1, *input_size)
        if torch.cuda.is_available():
            dummy_input = dummy_input.cuda()
            model = model.cuda()
            
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                _ = model(dummy_input)
            
            peak_memory = torch.cuda.max_memory_allocated() / (1024**3)
            print(f"Estimated Inference Memory: {peak_memory:.2f} GB")
    except Exception as e:
        print(f"Could not estimate memory: {e}")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    print("Utility functions loaded successfully.")

