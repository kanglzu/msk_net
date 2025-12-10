# MSK-Net: Multi-Scale Spatial KANs Enhanced U-Shaped Network For Explainable 3D Brain Tumor Segmentation

Official PyTorch implementation of **MSK-Net** for explainable 3D brain tumor segmentation.

![MSK-Net Framework](vis/framework.png)

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
- CUDA >= 12.x

### Setup

```bash
# Clone repository
git clone https://github.com/msk-net/msk_net.git
cd msk_net

# Create virtual environment
conda create -n msk_net python=3.9
conda activate msk_net

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

### 1. Prepare Data

Download BraTS datasets and organize data structure:
```
/path/to/dataset/
├── train/
│   ├── patient_001/
│   │   ├── *_t1.nii.gz
│   │   ├── *_t1ce.nii.gz
│   │   ├── *_t2.nii.gz
│   │   ├── *_flair.nii.gz
│   │   └── *_seg.nii.gz
│   └── ...
└── test/
    └── ...
```

### 2. Configure Experiment

Copy the template configuration file and fill in your parameters:

```bash
cp configs/config_template.yaml configs/my_config.yaml
# Edit configs/my_config.yaml with your settings
```

### 3. Train Model

```bash
python train.py --config configs/my_config.yaml

# Resume from checkpoint
python train.py --config configs/my_config.yaml --resume checkpoints/latest.pth
```

### 4. Test Model

```bash
python inference.py \
    --config configs/my_config.yaml \
    --checkpoint checkpoints/best.pth \
    --tta
```

---

## Project Structure

```
msk_net/
├── configs/
│   ├── config_template.yaml      # Configuration template
│   └── ...
├── models/
│   ├── blocks/
│   │   ├── skb.py                # Spatial KAN Block
│   │   ├── kbam.py               # KAN-Boosted Attention Module
│   │   ├── csgm.py               # Cross-Scale Gating Module
│   │   ├── aspp.py               # Atrous Spatial Pyramid Pooling
│   │   └── common.py             # Common layers
│   └── msk_net.py                # Main architecture
├── datasets/
│   └── brats_dataset.py          # Data loading and preprocessing
├── losses/
│   ├── losses.py                 # Loss functions
│   └── metrics.py                # Evaluation metrics
├── explainability/
│   ├── score_cam.py              # Score-CAM implementation
│   ├── grad_cam.py               # Grad-CAM implementation
│   └── full_grad.py              # FullGrad implementation
├── train.py                      # Training script
├── inference.py                  # Inference and testing
└── README.md
```

---

## Supported KAN Basis Functions

MSK-Net supports multiple KAN basis function implementations:

- B-Spline (default)
- Fourier
- Chebyshev
- Hermite
- Gegenbauer
- Jacobi
- Bessel
- Lucas
- Fibonacci
- Gaussian RBF
- Wavelet

---

## License

This project is licensed under the MIT License.

---

## Acknowledgments

- KAN implementations from [xKAN](https://github.com/mlsquare/xKAN)
- BraTS challenge organizers
