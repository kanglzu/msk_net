"""
Loss Functions for 3D Medical Image Segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import label
from typing import Optional, List


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation
    Dice coefficient measures overlap between prediction and ground truth:
    Dice = (2 * |P ∩ G|) / (|P| + |G|)
    """
    
    def __init__(self,
                 smooth: float = 1e-5,
                 reduction: str = 'mean',
                 class_weights: Optional[List[float]] = None):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
        self.class_weights = class_weights
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Predicted probabilities [B, C, H, W, D] (after sigmoid)
            targets: Ground truth binary masks [B, C, H, W, D]
        
        Returns:
            Dice loss value
        """
        # Flatten spatial dimensions
        inputs_flat = inputs.view(inputs.size(0), inputs.size(1), -1)
        targets_flat = targets.view(targets.size(0), targets.size(1), -1)
        
        # Calculate intersection and union
        intersection = (inputs_flat * targets_flat).sum(dim=2)
        denominator = inputs_flat.sum(dim=2) + targets_flat.sum(dim=2)
        
        # Dice coefficient per class
        dice_coeff = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        dice_loss_per_class = 1.0 - dice_coeff
        
        # Apply class weights if provided
        if self.class_weights is not None:
            weights = torch.tensor(
                self.class_weights,
                device=inputs.device,
                dtype=inputs.dtype
            )
            dice_loss_per_class = dice_loss_per_class * weights.unsqueeze(0)
        
        # Reduction
        if self.reduction == 'mean':
            return dice_loss_per_class.mean()
        elif self.reduction == 'sum':
            return dice_loss_per_class.sum()
        else:
            return dice_loss_per_class


class DiceBCELoss(nn.Module):
    """
    Combined Dice and Binary Cross-Entropy Loss
    Total Loss = dice_weight * Dice + bce_weight * BCE
    """
    
    def __init__(self,
                 dice_weight: float = 0.5,
                 bce_weight: float = 0.5,
                 dice_smooth: float = 1e-5,
                 class_weights: Optional[List[float]] = None):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        
        self.bce_fn = nn.BCEWithLogitsLoss(reduction='none')
        self.dice_fn = DiceLoss(
            smooth=dice_smooth,
            reduction='none',
            class_weights=class_weights
        )
        self.class_weights = class_weights
    
    def forward(self, inputs_logits, targets):
        """
        Args:
            inputs_logits: Raw model outputs (logits) [B, C, H, W, D]
            targets: Ground truth binary masks [B, C, H, W, D]
        
        Returns:
            Combined loss value
        """
        # BCE loss
        bce_loss_raw = self.bce_fn(inputs_logits, targets)
        
        if self.class_weights is not None:
            weights = torch.tensor(
                self.class_weights,
                device=inputs_logits.device,
                dtype=inputs_logits.dtype
            ).view(1, -1, 1, 1, 1)
            bce_loss = (bce_loss_raw * weights).mean()
        else:
            bce_loss = bce_loss_raw.mean()
        
        # Dice loss (apply sigmoid to logits)
        inputs_prob = torch.sigmoid(inputs_logits)
        dice_loss = self.dice_fn(inputs_prob, targets).mean()
        
        # Combine losses
        total_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        
        return total_loss


class TopologyAwareFocalLoss(nn.Module):
    """
    Topology-Aware Focal Loss
    Combines focal loss with topology preservation term based on
    connected components analysis.
    Total Loss = Focal Loss + lambda_t * Topology Term
    """
    
    def __init__(self,
                 alpha: float = 0.25,
                 gamma: float = 2.0,
                 lambda_t: float = 0.1,
                 class_weights: Optional[List[float]] = None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_t = lambda_t
        self.class_weights = class_weights
    
    def compute_connected_components(self, tensor_binary):
        """
        Count connected components in binary mask
        """
        arr = tensor_binary.cpu().detach().numpy().astype(np.uint8)
        
        if arr.ndim == 4:  # Batch dimension present
            cc_counts = [label(arr[i])[1] for i in range(arr.shape[0])]
            return np.mean(cc_counts)
        else:
            return label(arr)[1]
    
    def focal_loss_component(self, pred_logits, target_binary):
        """
        Focal loss component
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
        """
        # Get predicted probabilities
        pt = torch.sigmoid(pred_logits)
        pt = torch.where(target_binary == 1, pt, 1 - pt)
        
        # Focal weight: (1 - pt)^gamma
        focal_weight = (1 - pt).pow(self.gamma)
        
        # BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(
            pred_logits, target_binary, reduction='none'
        )
        
        # Apply focal weight
        focal_loss = focal_weight * bce_loss
        
        # Apply alpha balancing
        if self.alpha is not None:
            alpha_t = torch.where(target_binary == 1, self.alpha, 1 - self.alpha)
            focal_loss = alpha_t * focal_loss
        
        # Apply class weights
        if self.class_weights is not None:
            weights = torch.tensor(
                self.class_weights,
                device=pred_logits.device,
                dtype=pred_logits.dtype
            ).view(1, -1, 1, 1, 1)
            focal_loss = focal_loss * weights
        
        return focal_loss.mean()
    
    def topology_term(self, pred_logits, target_binary):
        """
        Topology preservation term based on connected components
        Penalizes mismatch in number of connected components between
        prediction and ground truth
        """
        pred_binary = (torch.sigmoid(pred_logits) > 0.5)
        target_bin = (target_binary > 0.5)
        
        topology_loss = 0.0
        num_classes = pred_logits.shape[1]
        
        for c in range(num_classes):
            pred_cc = self.compute_connected_components(pred_binary[:, c])
            target_cc = self.compute_connected_components(target_bin[:, c])
            topology_loss += abs(pred_cc - target_cc)
        
        topology_loss = topology_loss / num_classes
        return torch.tensor(topology_loss, device=pred_logits.device, dtype=torch.float32)
    
    def forward(self, pred_logits, target_binary):
        """
        Args:
            pred_logits: Model output logits [B, C, H, W, D]
            target_binary: Ground truth masks [B, C, H, W, D]
        
        Returns:
            Total topology-aware focal loss
        """
        focal = self.focal_loss_component(pred_logits, target_binary)
        topo = self.topology_term(pred_logits, target_binary)
        
        return focal + self.lambda_t * topo


class HybridDiceTopoLoss(nn.Module):
    """
    Hybrid Loss: Dice-BCE + Topology-Aware Focal Loss
    Total Loss = dice_bce_weight * DiceBCE + topo_focal_weight * TopoFocal
    """
    
    def __init__(self,
                 dice_bce_weight: float = 0.7,
                 topo_focal_weight: float = 0.3,
                 dice_bce_args: Optional[dict] = None,
                 topo_focal_args: Optional[dict] = None):
        super().__init__()
        
        self.dice_bce_weight = dice_bce_weight
        self.topo_focal_weight = topo_focal_weight
        
        # Initialize component losses
        if dice_bce_args is None:
            dice_bce_args = {}
        if topo_focal_args is None:
            topo_focal_args = {}
        
        self.dice_bce_loss = DiceBCELoss(**dice_bce_args)
        self.topo_focal_loss = TopologyAwareFocalLoss(**topo_focal_args)
    
    def forward(self, pred_logits, targets):
        """
        Args:
            pred_logits: Model output logits [B, C, H, W, D]
            targets: Ground truth masks [B, C, H, W, D]
        
        Returns:
            Combined hybrid loss
        """
        dice_bce = self.dice_bce_loss(pred_logits, targets)
        topo_focal = self.topo_focal_loss(pred_logits, targets)
        
        total = (self.dice_bce_weight * dice_bce +
                self.topo_focal_weight * topo_focal)
        
        return total


def get_loss_function(config: dict):
    """
    Factory function to create loss function from config
    
    Args:
        config: Training configuration dictionary
    
    Returns:
        Loss function module
    """
    loss_cfg = config['training']['loss']
    loss_type = loss_cfg['type']
    
    if loss_type == 'dice_bce':
        return DiceBCELoss(
            dice_weight=loss_cfg['dice_bce']['dice_weight'],
            bce_weight=loss_cfg['dice_bce']['bce_weight'],
            dice_smooth=loss_cfg['dice_bce']['smooth'],
            class_weights=loss_cfg['dice_bce']['class_weights']
        )
    
    elif loss_type == 'topo_focal':
        return TopologyAwareFocalLoss(
            alpha=loss_cfg['topo_focal']['alpha'],
            gamma=loss_cfg['topo_focal']['gamma'],
            lambda_t=loss_cfg['topo_focal']['lambda_t'],
            class_weights=loss_cfg['topo_focal']['class_weights']
        )
    
    elif loss_type == 'hybrid_dice_topo':
        dice_bce_args = {
            'dice_weight': loss_cfg['dice_bce']['dice_weight'],
            'bce_weight': loss_cfg['dice_bce']['bce_weight'],
            'dice_smooth': loss_cfg['dice_bce']['smooth'],
            'class_weights': loss_cfg['dice_bce']['class_weights']
        }
        topo_focal_args = {
            'alpha': loss_cfg['topo_focal']['alpha'],
            'gamma': loss_cfg['topo_focal']['gamma'],
            'lambda_t': loss_cfg['topo_focal']['lambda_t'],
            'class_weights': loss_cfg['topo_focal']['class_weights']
        }
        return HybridDiceTopoLoss(
            dice_bce_weight=loss_cfg['hybrid']['dice_bce_weight'],
            topo_focal_weight=loss_cfg['hybrid']['topo_focal_weight'],
            dice_bce_args=dice_bce_args,
            topo_focal_args=topo_focal_args
        )
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
