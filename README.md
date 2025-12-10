# MSK-Net: Multi-Scale Spatial KANs Enhanced U-Shaped Network For Explainable 3D Brain Tumor Segmentation

Official PyTorch implementation of **MSK-Net** for explainable 3D brain tumor segmentation.

![MSK-Net Framework](vis/framework.png)

**Paper**: MSK-Net: Multi-Scale Spatial KANs Enhanced U-Shaped Network For Explainable 3D Brain Tumor Segmentation

**Authors**: Yutong Wang, Zhongfeng Kang, et al.

**Affiliation**: Lanzhou University, China

---

## Overview

MSK-Net is a novel 3D segmentation framework that integrates **Kolmogorov-Arnold Networks (KANs)** into a U-shaped architecture for accurate and interpretable brain tumor segmentation from MRI scans.

### Key Innovations

1. **Spatial KAN Block (SKB)**: Replaces conventional linear filters with tensor products of univariate B-spline functions
2. **KAN-Boosted Attention Module (KBAM)**: Learnable spline-based channel-spatial attention mechanism
3. **Cross-Scale Gating Module (CSGM)**: Adaptive multi-scale feature fusion with KAN refinement
4. **Optimized 3D KAN Implementation**: Vectorized basis caching and spatial parameter sharing

---

## Installation

### Requirements

- Python >= 3.8
- PyTorch >= 2.1.0
- CUDA >= 12.2

### Setup

```bash
# Clone repository
git clone https://github.com/kanglzu/msk_net.git
cd msk_net

# Create virtual environment
conda create -n msk_net python=3.9
conda activate msk_net

# Install dependencies
pip install -r requirements.txt

# Install xKAN library
git clone https://github.com/mlsquare/xKAN.git
```

---

## Quick Start

### 1. Prepare Data

Download BraTS datasets
Organize data structure:
```
/path/to/BraTS2021/
├── train/
│   ├── BraTS2021_00001/
│   │   ├── BraTS2021_00001_t1.nii.gz
│   │   ├── BraTS2021_00001_t1ce.nii.gz
│   │   ├── BraTS2021_00001_t2.nii.gz
│   │   ├── BraTS2021_00001_flair.nii.gz
│   │   └── BraTS2021_00001_seg.nii.gz
│   ├── BraTS2021_00002/
│   └── ...
└── test/
    └── ...
```

### 2. Configure Experiment

Edit configuration file `configs/base_config.yaml` or create a new one.

```bash
# Example update in configs/base_config.yaml
data:
  train_data_dir: "/path/to/BraTS2021/train"
  test_data_dir: "/path/to/BraTS2021/test"
```

### 3. Train Model

```bash
# Train with default configuration
python msk_net/train.py --config msk_net/configs/base_config.yaml

# Resume from checkpoint
python msk_net/train.py --config msk_net/configs/base_config.yaml --resume output/checkpoints/latest.pth
```

### 4. Test Model

```bash
# Test with best checkpoint
python msk_net/inference.py \
    --config msk_net/configs/base_config.yaml \
    --checkpoint output/checkpoints/best.pth \
    --tta
```

---

## Project Structure

```
msk_net/
├── models/
│   ├── blocks/
│   │   ├── skb.py                 # Spatial KAN Block
│   │   ├── kbam.py                # KAN-Boosted Attention Module
│   │   ├── csgm.py                # Cross-Scale Gating Module
│   │   ├── aspp.py                # Atrous Spatial Pyramid Pooling
│   │   └── common.py              # Common layers
│   └── msk_net.py                 # Main architecture
├── datasets/
│   └── brats_dataset.py           # Data loading and preprocessing
├── losses/
│   └── losses.py                  # Loss functions
├── train.py                       # Training script
├── inference.py                   # Inference and testing
└── README.md                      # This file
```



---

## License

This project is licensed under the MIT License.

---

## Acknowledgments

- KAN implementations from [xKAN](https://github.com/mlsquare/xKAN)
- BraTS challenge organizers
- Funded by: Talent Scientific Fund of Lanzhou University (Grant 561120212)

