#!/bin/bash
# Complete pipeline for training and testing MSK-Net

set -e

CONFIG_FILE=${1:-"configs/brats2021_config.yaml"}

echo "=========================================="
echo "MSK-Net Full Pipeline"
echo "=========================================="
echo "Configuration: $CONFIG_FILE"
echo ""

# Step 1: Validate data
echo "Step 1: Validating dataset..."
python scripts/prepare_data.py \
    --data_dir /path/to/BraTS2021/train \
    --dataset_type brats2021 \
    --output dataset_splits.json

# Step 2: Train model
echo ""
echo "Step 2: Training MSK-Net..."
python train.py --config $CONFIG_FILE

# Step 3: Test model
echo ""
echo "Step 3: Testing on test set..."
python inference.py \
    --config $CONFIG_FILE \
    --checkpoint checkpoints/best.pth \
    --tta \
    --measure-speed \
    --measure-memory

# Step 4: Explainability analysis
echo ""
echo "Step 4: Running explainability analysis..."
python explainability/eval_explainability.py \
    --config $CONFIG_FILE \
    --checkpoint checkpoints/best.pth

echo ""
echo "=========================================="
echo "Pipeline completed successfully!"
echo "=========================================="
echo "Check results in: results/"

