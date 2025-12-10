"""
Evaluation Metrics for Segmentation

Implements standard medical image segmentation metrics:
- Dice Score (Sorensen-Dice coefficient)
- IoU (Intersection over Union / Jaccard Index)
- HD95 (95th percentile Hausdorff Distance)
- Sensitivity and Specificity
"""

import torch
import numpy as np
from scipy.ndimage import distance_transform_edt
from typing import Tuple, List


def calculate_dice_score(pred: torch.Tensor, 
                         target: torch.Tensor, 
                         smooth: float = 1e-5) -> torch.Tensor:
    """
    Calculate Dice Score (Sorensen-Dice coefficient)
    
    Dice = (2 * |P intersection G|) / (|P| + |G|)
    
    Args:
        pred: Predicted binary mask [B, C, H, W, D]
        target: Ground truth binary mask [B, C, H, W, D]
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        Dice score per class [B, C]
    """
    # Flatten spatial dimensions
    pred_flat = pred.view(pred.size(0), pred.size(1), -1)
    target_flat = target.view(target.size(0), target.size(1), -1)
    
    # Calculate intersection and sum
    intersection = (pred_flat * target_flat).sum(dim=2)
    pred_sum = pred_flat.sum(dim=2)
    target_sum = target_flat.sum(dim=2)
    
    # Dice coefficient
    dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)
    
    return dice


def calculate_iou_score(pred: torch.Tensor,
                       target: torch.Tensor,
                       smooth: float = 1e-5) -> torch.Tensor:
    """
    Calculate IoU (Jaccard Index)
    
    IoU = |P intersection G| / |P union G|
    
    Args:
        pred: Predicted binary mask [B, C, H, W, D]
        target: Ground truth binary mask [B, C, H, W, D]
        smooth: Smoothing factor
    
    Returns:
        IoU score per class [B, C]
    """
    pred_flat = pred.view(pred.size(0), pred.size(1), -1)
    target_flat = target.view(target.size(0), target.size(1), -1)
    
    intersection = (pred_flat * target_flat).sum(dim=2)
    union = pred_flat.sum(dim=2) + target_flat.sum(dim=2) - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    
    return iou


def calculate_hd95(pred: np.ndarray,
                  target: np.ndarray,
                  voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> float:
    """
    Calculate 95th Percentile Hausdorff Distance
    
    HD95 = 95th percentile of symmetric boundary distances
    
    Args:
        pred: Predicted binary mask [H, W, D]
        target: Ground truth binary mask [H, W, D]
        voxel_spacing: Physical spacing of voxels (mm)
    
    Returns:
        HD95 distance in mm
    """
    if pred.sum() == 0 or target.sum() == 0:
        # Handle empty masks
        if pred.sum() == 0 and target.sum() == 0:
            return 0.0
        else:
            # Return a large distance if one is empty
            return 373.0  # Max distance in 128^3 volume
    
    # Get surface points by computing distance transforms
    pred_dt = distance_transform_edt(1 - pred, sampling=voxel_spacing)
    target_dt = distance_transform_edt(1 - target, sampling=voxel_spacing)
    
    # Distances from pred surface to target
    pred_surface = (pred_dt == 0)
    distances_pred_to_target = target_dt[pred_surface]
    
    # Distances from target surface to pred
    target_surface = (target_dt == 0)
    distances_target_to_pred = pred_dt[target_surface]
    
    # Combine all distances
    all_distances = np.concatenate([
        distances_pred_to_target,
        distances_target_to_pred
    ])
    
    if len(all_distances) == 0:
        return 0.0
    
    # Return 95th percentile
    hd95 = np.percentile(all_distances, 95)
    
    return float(hd95)


def calculate_sensitivity_specificity(pred: torch.Tensor,
                                     target: torch.Tensor,
                                     smooth: float = 1e-5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate Sensitivity (Recall) and Specificity
    
    Sensitivity (True Positive Rate): TP / (TP + FN)
    Specificity (True Negative Rate): TN / (TN + FP)
    
    Args:
        pred: Predicted binary mask [B, C, H, W, D]
        target: Ground truth binary mask [B, C, H, W, D]
        smooth: Smoothing factor
    
    Returns:
        (sensitivity, specificity) each [B, C]
    """
    pred_flat = pred.view(pred.size(0), pred.size(1), -1)
    target_flat = target.view(target.size(0), target.size(1), -1)
    
    # True Positives, False Positives, True Negatives, False Negatives
    tp = (pred_flat * target_flat).sum(dim=2)
    fp = (pred_flat * (1 - target_flat)).sum(dim=2)
    tn = ((1 - pred_flat) * (1 - target_flat)).sum(dim=2)
    fn = ((1 - pred_flat) * target_flat).sum(dim=2)
    
    # Calculate sensitivity and specificity
    sensitivity = (tp + smooth) / (tp + fn + smooth)
    specificity = (tn + smooth) / (tn + fp + smooth)
    
    return sensitivity, specificity


def compute_batch_metrics(pred_logits: torch.Tensor,
                          target: torch.Tensor,
                          metrics: List[str] = ['dice', 'iou']) -> dict:
    """
    Compute multiple metrics for a batch
    
    Args:
        pred_logits: Model output logits [B, C, H, W, D]
        target: Ground truth masks [B, C, H, W, D]
        metrics: List of metric names to compute
    
    Returns:
        Dictionary of computed metrics
    """
    # Convert logits to binary predictions
    pred_prob = torch.sigmoid(pred_logits)
    pred_binary = (pred_prob > 0.5).float()
    
    results = {}
    
    if 'dice' in metrics:
        dice = calculate_dice_score(pred_binary, target)
        results['dice'] = dice  # [B, C]
        results['dice_mean'] = dice.mean()
    
    if 'iou' in metrics:
        iou = calculate_iou_score(pred_binary, target)
        results['iou'] = iou  # [B, C]
        results['iou_mean'] = iou.mean()
    
    if 'sensitivity' in metrics or 'specificity' in metrics:
        sens, spec = calculate_sensitivity_specificity(pred_binary, target)
        results['sensitivity'] = sens
        results['specificity'] = spec
    
    # HD95 requires numpy and is computed per-sample, not batched
    if 'hd95' in metrics:
        hd95_scores = []
        pred_np = pred_binary.cpu().numpy()
        target_np = target.cpu().numpy()
        
        for b in range(pred_np.shape[0]):
            batch_hd95 = []
            for c in range(pred_np.shape[1]):
                hd95 = calculate_hd95(pred_np[b, c], target_np[b, c])
                batch_hd95.append(hd95)
            hd95_scores.append(batch_hd95)
        
        results['hd95'] = torch.tensor(hd95_scores)  # [B, C]
        results['hd95_mean'] = results['hd95'].mean()
    
    return results


if __name__ == "__main__":
    # Test metrics
    print("Testing metrics...")
    
    B, C, H, W, D = 2, 3, 64, 64, 64
    pred = torch.randn(B, C, H, W, D)
    target = torch.randint(0, 2, (B, C, H, W, D)).float()
    
    metrics = compute_batch_metrics(pred, target, ['dice', 'iou'])
    print(f"Dice: {metrics['dice']}")
    print(f"IoU: {metrics['iou']}")
    print("Metrics test passed!")

