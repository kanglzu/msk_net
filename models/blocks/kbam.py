"""
KAN-Boosted Attention Module (KBAM)
"""

import torch
import torch.nn as nn
from .common import get_kan_layer

class KBAM3D(nn.Module):
    """
    KAN-Boosted Attention Module.
    Uses KAN for both Channel and Spatial Attention.
    """
    def __init__(self, 
                 in_channels: int,
                 reduction: int = 16,
                 kan_type: str = 'bspline',
                 grid_size: int = 5,
                 spline_order: int = 3,
                 **kan_kwargs):
        super().__init__()
        self.in_channels = in_channels
        
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        reduced_channels = max(in_channels // reduction, 1)
        
        self.channel_kan1 = get_kan_layer(in_channels, reduced_channels, kan_type, grid_size=grid_size, spline_order=spline_order, **kan_kwargs)
        self.channel_relu = nn.ReLU(inplace=True)
        self.channel_kan2 = get_kan_layer(reduced_channels, in_channels, kan_type, grid_size=grid_size, spline_order=spline_order, **kan_kwargs)
        self.channel_sigmoid = nn.Sigmoid()
        
        self.spatial_conv = nn.Conv3d(in_channels, 1, kernel_size=1, bias=False)
        self.spatial_kan = get_kan_layer(1, 1, kan_type, grid_size=grid_size, spline_order=spline_order, **kan_kwargs)
        self.spatial_sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        B, C, H, W, D = x.size()
        
        y_avg = self.avg_pool(x).view(B, C)
        y_max = self.max_pool(x).view(B, C)
        
        out_avg = self.channel_kan2(self.channel_relu(self.channel_kan1(y_avg)))
        out_max = self.channel_kan2(self.channel_relu(self.channel_kan1(y_max)))
        
        channel_att = self.channel_sigmoid(out_avg + out_max).view(B, C, 1, 1, 1)
        x_channel = x * channel_att
        
        y_spatial = self.spatial_conv(x_channel)
        y_flat = y_spatial.view(-1, 1)
        y_processed = self.spatial_kan(y_flat)
        spatial_att = self.spatial_sigmoid(y_processed).view(B, 1, H, W, D)
        
        return x_channel * spatial_att

