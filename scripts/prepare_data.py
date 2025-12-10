"""
Data Preparation Script

Validates BraTS dataset structure and creates dataset splits
"""

import argparse
from pathlib import Path
import json
import random


def validate_brats_structure(data_dir, dataset_type='brats2021'):
    """
    Validate BraTS dataset directory structure
    
    Args:
        data_dir: Path to BraTS data directory
        dataset_type: Type of dataset
    
    Returns:
        List of valid patient directories
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Expected modalities based on dataset type
    if dataset_type in ['brats2018', 'brats2021']:
        required_modalities = ['t1', 't1ce', 't2', 'flair']
    elif dataset_type == 'brats_gli':
        required_modalities = ['t1ce', 't1n', 't2f', 't2w']
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    valid_patients = []
    invalid_patients = []
    
    # Scan patient directories
    patient_dirs = [d for d in data_path.iterdir() if d.is_dir()]
    
    print(f"Scanning {len(patient_dirs)} patient directories...")
    
    for patient_dir in patient_dirs:
        # Check for required modalities
        found_modalities = set()
        
        for modality in required_modalities:
            pattern_matches = list(patient_dir.glob(f"*{modality}*.nii*"))
            if pattern_matches:
                found_modalities.add(modality)
        
        # Check for segmentation file
        has_seg = len(list(patient_dir.glob("*seg*.nii*"))) > 0
        
        if found_modalities == set(required_modalities) and has_seg:
            valid_patients.append(patient_dir.name)
        else:
            missing = set(required_modalities) - found_modalities
            invalid_patients.append({
                'patient': patient_dir.name,
                'missing_modalities': list(missing),
                'has_seg': has_seg
            })
    
    print(f"\nValidation Results:")
    print(f"  Valid patients: {len(valid_patients)}")
    print(f"  Invalid patients: {len(invalid_patients)}")
    
    if invalid_patients:
        print("\nInvalid patients (first 10):")
        for inv in invalid_patients[:10]:
            print(f"  {inv}")
    
    return valid_patients


def create_patient_splits(valid_patients, train_ratio=0.8, seed=42):
    """
    Create patient-wise train/test splits
    
    Paper Section 5.1: "patient-wise 5-fold cross-validation"
    
    Args:
        valid_patients: List of valid patient IDs
        train_ratio: Training set ratio
        seed: Random seed
    
    Returns:
        Dictionary with train and test patient lists
    """
    random.seed(seed)
    
    # Shuffle patients
    patients_shuffled = valid_patients.copy()
    random.shuffle(patients_shuffled)
    
    # Split
    n_train = int(len(patients_shuffled) * train_ratio)
    train_patients = sorted(patients_shuffled[:n_train])
    test_patients = sorted(patients_shuffled[n_train:])
    
    splits = {
        'train': train_patients,
        'test': test_patients,
        'seed': seed,
        'train_ratio': train_ratio
    }
    
    print(f"\nDataset Split (seed={seed}):")
    print(f"  Train: {len(train_patients)} patients")
    print(f"  Test: {len(test_patients)} patients")
    
    return splits


def main():
    parser = argparse.ArgumentParser(description='Prepare BraTS Dataset')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to BraTS data directory')
    parser.add_argument('--dataset_type', type=str, default='brats2021',
                       choices=['brats2018', 'brats2021', 'brats_gli'],
                       help='Type of BraTS dataset')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Training set ratio')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output', type=str, default='dataset_splits.json',
                       help='Output JSON file for splits')
    
    args = parser.parse_args()
    
    # Validate dataset structure
    valid_patients = validate_brats_structure(args.data_dir, args.dataset_type)
    
    if len(valid_patients) == 0:
        print("\nError: No valid patients found!")
        return
    
    # Create splits
    splits = create_patient_splits(valid_patients, args.train_ratio, args.seed)
    
    # Save splits
    with open(args.output, 'w') as f:
        json.dump(splits, f, indent=2)
    
    print(f"\nSplits saved to: {args.output}")


if __name__ == '__main__':
    main()

