"""
Common building blocks for MSK-Net.
Includes: DoubleConv, OutConv, SEBlock3D, KAN Layer Factory
"""

import torch
import torch.nn as nn
import importlib
from typing import Optional

def get_kan_layer(in_features: int, 
                  out_features: int, 
                  kan_type: str = 'bspline',
                  **kwargs):
    """
    Factory function to create KAN layers.
    
    Supported KAN types:
    - bspline: B-spline based KAN (default, used in MSK-Net)
    - fourier: Fourier basis KAN
    - chebyshev: Chebyshev polynomial KAN
    - hermite: Hermite polynomial KAN
    - gegenbauer: Gegenbauer polynomial KAN
    - jacobi: Jacobi polynomial KAN
    - bessel: Bessel function KAN
    - lucas: Lucas polynomial KAN
    - fibonacci: Fibonacci polynomial KAN
    - grbf: Gaussian RBF KAN
    - wavelet: Wavelet KAN
    """
    kan_map = {
        'bspline': ('bspine_kan', 'KANLinear'),
        'fourier': ('fourier_kan', 'FourierKANLayer'),
        'chebyshev': ('chebyshev_kan', 'ChebyKANLayer'),
        'hermite': ('hermite_kan', 'HermiteKANLayer'),
        'gegenbauer': ('gegenbauer_kan', 'GegenbauerKANLayer'),
        'jacobi': ('jacobi_kan', 'JacobiKANLayer'),
        'bessel': ('bessel_kan', 'BesselKANLayer'),
        'lucas': ('lucas_kan', 'LucasKANLayer'),
        'fibonacci': ('fibonacci_kan', 'FibonacciKANLayer'),
        'grbf': ('grbf_kan', 'GRBFKANLayer'),
        'wavelet': ('wav_kan', 'WaveletKANLayer'),
    }
    
    if kan_type not in kan_map:
        raise ValueError(f"Unknown KAN type: {kan_type}. Supported types: {list(kan_map.keys())}")
    
    module_name, class_name = kan_map[kan_type]
    
    try:
        module = importlib.import_module(f"xKAN.{module_name}")
        kan_class = getattr(module, class_name)
        
        # Build arguments based on KAN type
        if kan_type == 'bspline':
            kan_args = {
                'in_features': in_features,
                'out_features': out_features,
                'grid_size': kwargs.get('grid_size', 5),
                'spline_order': kwargs.get('spline_order', 3),
                'enable_standalone_scale_spline': kwargs.get('enable_standalone_scale_spline', True),
                'grid_range': kwargs.get('grid_range', [-1.0, 1.0]),
            }
        elif kan_type in ['chebyshev', 'hermite', 'gegenbauer', 'jacobi', 'bessel', 'lucas', 'fibonacci']:
            # Polynomial-based KANs use input_dim, output_dim, degree
            kan_args = {
                'input_dim': in_features,
                'output_dim': out_features,
                'degree': kwargs.get('degree', kwargs.get('spline_order', 3)),
            }
            # Some polynomial KANs have additional parameters
            if kan_type == 'gegenbauer' and 'alpha' in kwargs:
                kan_args['alpha'] = kwargs['alpha']
            if kan_type == 'jacobi':
                kan_args['alpha'] = kwargs.get('alpha', 1.0)
                kan_args['beta'] = kwargs.get('beta', 1.0)
        elif kan_type == 'fourier':
            kan_args = {
                'in_features': in_features,
                'out_features': out_features,
                'grid_size': kwargs.get('grid_size', 5),
            }
        elif kan_type == 'grbf':
            kan_args = {
                'in_features': in_features,
                'out_features': out_features,
                'grid_size': kwargs.get('grid_size', 5),
            }
        elif kan_type == 'wavelet':
            kan_args = {
                'in_features': in_features,
                'out_features': out_features,
                'wavelet_type': kwargs.get('wavelet_type', 'mexican_hat'),
            }
        else:
            # Fallback generic arguments
            kan_args = {
                'in_features': in_features,
                'out_features': out_features,
            }
        
        return kan_class(**kan_args)
    
    except (ImportError, AttributeError) as e:
        print(f"Warning: Could not import {class_name}: {e}. Falling back to Linear.")
        return nn.Linear(in_features, out_features)


class DoubleConv(nn.Module):
    """
    Standard double convolution block: Conv3d -> IN -> LeakyReLU -> Conv3d -> IN -> LeakyReLU
    """
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.InstanceNorm3d(mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class OutConv(nn.Module):
    """
    Output convolution: 1x1x1 Conv
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)


class SEBlock3D(nn.Module):
    """
    3D Squeeze-and-Excitation Block
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool3d(1)
        self.excitation = nn.Sequential(
            nn.Conv3d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // reduction, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        weights = self.squeeze(x)
        weights = self.excitation(weights)
        return x * weights

