"""
Cross-Scale Gating Module (CSGM)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import get_kan_layer

class CSGM(nn.Module):
    """
    Cross-Scale Gating Module for adaptive fusion.
    """
    def __init__(self,
                 encoder_channels: int,
                 decoder_channels: int,
                 kan_type: str = 'bspline',
                 grid_size: int = 5,
                 spline_order: int = 3,
                 **kan_kwargs):
        super().__init__()
        
        self.conv_gate = nn.Conv3d(encoder_channels + decoder_channels, 1, kernel_size=1)
        self.sigmoid_gate = nn.Sigmoid()
        
        self.kan_refine = get_kan_layer(1, 1, kan_type, grid_size=grid_size, spline_order=spline_order, **kan_kwargs)
        self.sigmoid_out = nn.Sigmoid()
        
        if encoder_channels != decoder_channels:
            self.proj_decoder = nn.Conv3d(decoder_channels, encoder_channels, kernel_size=1, bias=False)
        else:
            self.proj_decoder = nn.Identity()
            
    def forward(self, x_encoder, x_decoder):
        if x_decoder.size()[2:] != x_encoder.size()[2:]:
            x_decoder = F.interpolate(x_decoder, size=x_encoder.size()[2:], mode='trilinear', align_corners=False)
            
        x_decoder_proj = self.proj_decoder(x_decoder)
        
        combined = torch.cat([x_encoder, x_decoder_proj], dim=1)
        gate = self.sigmoid_gate(self.conv_gate(combined))
        
        B, _, H, W, D = gate.size()
        gate_flat = gate.view(-1, 1)
        gate_refined = self.kan_refine(gate_flat)
        gate_tilde = self.sigmoid_out(gate_refined).view(B, 1, H, W, D)
        
        return gate_tilde * x_encoder + (1 - gate_tilde) * x_decoder_proj

