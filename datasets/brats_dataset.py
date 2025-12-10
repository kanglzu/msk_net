"""
BraTS Dataset Implementation
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import torch.nn.functional as F

try:
    import torchio as tio
    TORCHIO_AVAILABLE = True
except ImportError:
    print("Warning: TorchIO not available. Install with: pip install torchio")
    TORCHIO_AVAILABLE = False


class BraTSDataset(Dataset):
    """
    BraTS Dataset for 3D Brain Tumor Segmentation
    
    Preprocessing steps:
    1. Intensity normalization: Z-score per modality (non-zero voxels only)
    2. Spatial cropping: Crop to non-zero bounding box
    3. Resizing: Resize to target patch size using trilinear interpolation
    """
    
    def __init__(self,
                 data_dir: str,
                 mode: str = 'train',
                 dataset_type: str = 'brats2021',
                 patch_size: Tuple[int, int, int] = (128, 128, 128),
                 cache_dir: Optional[str] = None,
                 augment: bool = True,
                 foreground_prob: float = 0.5):
        """
        Args:
            data_dir: Path to BraTS data directory
            mode: 'train', 'val', or 'test'
            dataset_type: 'brats2018', 'brats2021', or 'brats_gli'
            patch_size: Target patch size for cropping/resizing
            cache_dir: Directory for caching preprocessed data
            augment: Enable data augmentation (only for training)
            foreground_prob: Probability of foreground-centered sampling
        """
        super().__init__()
        
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.dataset_type = dataset_type
        self.patch_size = patch_size
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.augment = augment and mode == 'train'
        self.foreground_prob = foreground_prob
        
        # Define modality names based on dataset type
        if dataset_type in ['brats2018', 'brats2021']:
            self.modalities = ['t1', 't1ce', 't2', 'flair']
            self.num_classes = 3  # WT, TC, ET
        elif dataset_type == 'brats_gli':
            self.modalities = ['t1ce', 't1n', 't2f', 't2w']
            self.num_classes = 6  # WT, TC, ET, RC, NETC, SNFH
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.patient_dirs = sorted([
            d for d in self.data_dir.glob('*')
            if d.is_dir() and not d.name.startswith('.')
        ])
        
        if len(self.patient_dirs) == 0:
            # Only raise if directory exists but is empty, avoiding error during initial setup
            if self.data_dir.exists():
                print(f"Warning: No patient directories found in {data_dir}")
        
        if self.augment and TORCHIO_AVAILABLE:
            self.transform = self._get_augmentation_transforms()
        else:
            self.transform = None
    
    def _get_augmentation_transforms(self):
        transforms = []
        transforms.append(
            tio.RandomAffine(
                scales=(0.85, 1.15),
                degrees=15,
                translation=10,
                p=0.5
            )
        )
        transforms.append(
            tio.RandomElasticDeformation(
                num_control_points=7,
                max_displacement=7.5,
                locked_borders=2,
                p=0.5
            )
        )
        transforms.append(tio.RandomFlip(axes=(0, 1, 2), p=0.5))
        transforms.append(tio.RandomGamma(log_gamma=(np.log(0.7), np.log(1.5)), p=0.5))
        transforms.append(tio.RandomNoise(mean=0, std=(0, 0.1), p=0.3))
        
        return tio.Compose(transforms)
    
    def _load_nifti_file(self, filepath: Path) -> np.ndarray:
        nii = nib.load(str(filepath))
        data = nii.get_fdata()
        return data.astype(np.float32)
    
    def _find_modality_file(self, patient_dir: Path, modality: str) -> Optional[Path]:
        patterns = [
            f"*_{modality}.nii.gz",
            f"*_{modality}.nii",
            f"*{modality}*.nii.gz",
            f"*{modality}*.nii"
        ]
        
        for pattern in patterns:
            files = list(patient_dir.glob(pattern))
            if files:
                return files[0]
        
        return None
    
    def _normalize_intensity(self, volume: np.ndarray) -> np.ndarray:
        nonzero_mask = volume > 0
        if nonzero_mask.sum() == 0:
            return volume
        
        mean = volume[nonzero_mask].mean()
        std = volume[nonzero_mask].std()
        
        if std < 1e-8:
            std = 1.0
        
        normalized = volume.copy()
        normalized[nonzero_mask] = (volume[nonzero_mask] - mean) / std
        normalized = np.clip(normalized, -5.0, 5.0)
        
        return normalized
    
    def _crop_to_nonzero(self, volumes: List[np.ndarray]) -> Tuple[List[np.ndarray], Tuple]:
        combined_mask = np.zeros_like(volumes[0], dtype=bool)
        for vol in volumes:
            combined_mask = combined_mask | (vol > 0)
        
        nonzero_indices = np.argwhere(combined_mask)
        
        if len(nonzero_indices) == 0:
            return volumes, None
        
        min_coords = nonzero_indices.min(axis=0)
        max_coords = nonzero_indices.max(axis=0) + 1
        
        cropped = [
            vol[min_coords[0]:max_coords[0],
                min_coords[1]:max_coords[1],
                min_coords[2]:max_coords[2]]
            for vol in volumes
        ]
        
        bbox = (min_coords, max_coords)
        return cropped, bbox
    
    def _resize_volume(self, volume: np.ndarray, target_size: Tuple[int, int, int]) -> np.ndarray:
        volume_tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0)
        resized = F.interpolate(
            volume_tensor,
            size=target_size,
            mode='trilinear',
            align_corners=False
        )
        resized_np = resized.squeeze().numpy()
        return resized_np
    
    def __len__(self):
        return len(self.patient_dirs)
    
    def __getitem__(self, idx):
        patient_dir = self.patient_dirs[idx]
        patient_id = patient_dir.name
        
        modality_volumes = []
        for modality in self.modalities:
            modality_file = self._find_modality_file(patient_dir, modality)
            
            if modality_file is None:
                raise FileNotFoundError(
                    f"Modality {modality} not found for patient {patient_id}"
                )
            
            volume = self._load_nifti_file(modality_file)
            volume = self._normalize_intensity(volume)
            modality_volumes.append(volume)
        
        seg_patterns = ['*seg.nii.gz', '*seg.nii', '*label*.nii.gz']
        seg_file = None
        for pattern in seg_patterns:
            files = list(patient_dir.glob(pattern))
            if files:
                seg_file = files[0]
                break
        
        if seg_file is None:
            raise FileNotFoundError(f"Segmentation file not found for {patient_id}")
        
        label_volume = self._load_nifti_file(seg_file)
        
        all_volumes = modality_volumes + [label_volume]
        cropped_volumes, bbox = self._crop_to_nonzero(all_volumes)
        
        modality_volumes = cropped_volumes[:-1]
        label_volume = cropped_volumes[-1]
        
        modality_volumes = [
            self._resize_volume(vol, self.patch_size)
            for vol in modality_volumes
        ]
        label_volume = self._resize_volume(label_volume, self.patch_size)
        
        image = np.stack(modality_volumes, axis=0)
        
        # BraTS labels: 0=background, 1=NCR/NET, 2=ED, 4=ET
        # Output classes: WT (Whole Tumor), TC (Tumor Core), ET (Enhancing Tumor)
        label_binary = self._convert_labels_to_binary(label_volume)
        
        image_tensor = torch.from_numpy(image).float()
        label_tensor = torch.from_numpy(label_binary).float()
        
        if self.augment and self.transform is not None and TORCHIO_AVAILABLE:
            subject = tio.Subject(
                image=tio.ScalarImage(tensor=image_tensor.unsqueeze(0)),
                label=tio.LabelMap(tensor=label_tensor.unsqueeze(0))
            )
            augmented = self.transform(subject)
            image_tensor = augmented['image'].data.squeeze(0)
            label_tensor = augmented['label'].data.squeeze(0)
        
        return {
            'image': image_tensor,
            'label': label_tensor,
            'patient_id': patient_id
        }
    
    def _convert_labels_to_binary(self, label: np.ndarray) -> np.ndarray:
        output = np.zeros((self.num_classes,) + label.shape, dtype=np.float32)
        
        if self.dataset_type in ['brats2018', 'brats2021']:
            # WT: All tumor regions (1, 2, 4)
            output[0] = (label > 0).astype(np.float32)
            # TC: NCR/NET + ET (1, 4)
            output[1] = ((label == 1) | (label == 4)).astype(np.float32)
            # ET: Enhancing tumor only (4)
            output[2] = (label == 4).astype(np.float32)
        
        elif self.dataset_type == 'brats_gli':
            # Placeholder for BraTS-GLI encoding
            output[0] = (label > 0).astype(np.float32)
            output[1] = ((label == 1) | (label == 4)).astype(np.float32)
            output[2] = (label == 4).astype(np.float32)
        
        return output


def create_data_loaders(config: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders from config
    """
    data_cfg = config['data']
    train_cfg = config['training']
    
    # Use preprocessing config if available, else defaults
    # In config/default_config.py we don't have detailed preprocessing section yet, 
    # so we'll check if it exists or use defaults
    foreground_prob = 0.5
    if 'preprocessing' in data_cfg and 'patch_sampling' in data_cfg['preprocessing']:
        foreground_prob = data_cfg['preprocessing']['patch_sampling']['foreground_probability']
        
    train_dataset = BraTSDataset(
        data_dir=data_cfg.get('train_data_dir', './data/train'),
        mode='train',
        dataset_type=data_cfg['dataset'],
        patch_size=tuple(config['model']['input_size']),
        cache_dir=data_cfg.get('train_cache_dir'),
        augment=True, # Config doesn't explicitly have augmentation enable/disable in default yet, assume True for train
        foreground_prob=foreground_prob
    )
    
    val_dataset = None
    if data_cfg.get('val_data_dir'):
        val_dataset = BraTSDataset(
            data_dir=data_cfg['val_data_dir'],
            mode='val',
            dataset_type=data_cfg['dataset'],
            patch_size=tuple(config['model']['input_size']),
            cache_dir=data_cfg.get('val_cache_dir'),
            augment=False
        )
    
    test_dataset = BraTSDataset(
        data_dir=data_cfg.get('test_data_dir', './data/test'),
        mode='test',
        dataset_type=data_cfg['dataset'],
        patch_size=tuple(config['model']['input_size']),
        cache_dir=data_cfg.get('test_cache_dir'),
        augment=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=True,
        num_workers=data_cfg.get('num_workers', 4),
        pin_memory=data_cfg.get('pin_memory', True),
        drop_last=True
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=train_cfg['batch_size'],
            shuffle=False,
            num_workers=data_cfg.get('num_workers', 4),
            pin_memory=data_cfg.get('pin_memory', True),
            drop_last=False
        )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1, # Inference usually batch 1
        shuffle=False,
        num_workers=data_cfg.get('num_workers', 4),
        pin_memory=data_cfg.get('pin_memory', True),
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader
