#!/bin/bash
# Download pre-trained MSK-Net checkpoints

echo "Downloading pre-trained MSK-Net models..."

# Create checkpoints directory
mkdir -p checkpoints/pretrained

# BraTS2021 checkpoint (placeholder - update with actual URLs when available)
echo "Downloading BraTS2021 checkpoint..."
# wget https://github.com/kanglzu/msk_net/releases/download/v1.0/msk_net_brats2021.pth \
#     -O checkpoints/pretrained/msk_net_brats2021.pth

# BraTS2018 checkpoint
echo "Downloading BraTS2018 checkpoint..."
# wget https://github.com/kanglzu/msk_net/releases/download/v1.0/msk_net_brats2018.pth \
#     -O checkpoints/pretrained/msk_net_brats2018.pth

# BraTS-GLI checkpoint
echo "Downloading BraTS-GLI checkpoint..."
# wget https://github.com/kanglzu/msk_net/releases/download/v1.0/msk_net_brats_gli.pth \
#     -O checkpoints/pretrained/msk_net_brats_gli.pth

echo "Download complete!"
echo "Checkpoints saved to: checkpoints/pretrained/"
echo ""
echo "Note: Update URLs in this script when pre-trained models are released."

