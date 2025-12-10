"""
MSK-Net: Multi-Scale Spatial KANs Enhanced U-Shaped Network
For Explainable 3D Brain Tumor Segmentation

Official Implementation
Paper: MSK-Net: Multi-Scale Spatial KANs Enhanced U-Shaped Network 
       For Explainable 3D Brain Tumor Segmentation

Authors: Yutong Wang, Zhongfeng Kang, et al.
Affiliation: Lanzhou University, China
"""

__version__ = "1.0.0"
__author__ = "Yutong Wang, Zhongfeng Kang, et al."
__email__ = "kangzf@lzu.edu.cn"

from .models import MSKNet
from .datasets import BraTSDataset
from .losses import DiceLoss, DiceBCELoss

__all__ = ['MSKNet', 'BraTSDataset', 'DiceLoss', 'DiceBCELoss']

