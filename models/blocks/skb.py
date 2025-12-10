"""
Spatial KAN Block (SKB)
Implements:
- Tokenization (Patch Embedding)
- KAN Transformation (Optimized with Basis Caching concept via efficient implementation)
- Spatial Reconstruction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from .common import get_kan_layer, SEBlock3D

class PatchEmbed3D(nn.Module):
    """
    3D Patch Embedding for tokenization.
    """
    def __init__(self, input_spatial_shape: Tuple[int, int, int], patch_size: Tuple[int, int, int], in_channels: int, embed_dim: int):
        super().__init__()
        self.input_spatial_shape = input_spatial_shape
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class ReshapeToken(nn.Module):
    """
    Reshape tokens back to spatial grid.
    """
    def __init__(self, target_spatial_shape: Tuple[int, int, int]):
        super().__init__()
        self.target_shape = target_spatial_shape
    
    def forward(self, x):
        B, N, C = x.shape
        D, H, W = self.target_shape
        x = x.view(B, D, H, W, C)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

class TokenizedKANBlock(nn.Module):
    """
    Spatial KAN Block (SKB) implementation.
    Integrates spatial parameter sharing via patch tokenization.
    """
    def __init__(self,
                 input_spatial_shape: Tuple[int, int, int],
                 in_channels: int,
                 out_channels: int,
                 patch_size: Tuple[int, int, int],
                 kan_type: str = 'bspline',
                 target_token_embed_dim: int = 512,
                 grid_size: int = 5,
                 spline_order: int = 3,
                 use_se: bool = True,
                 se_reduction: int = 16,
                 **kan_kwargs):
        super().__init__()
        self.input_spatial_shape = input_spatial_shape
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.use_se = use_se
        
        self.token_embed = PatchEmbed3D(input_spatial_shape, patch_size, in_channels, target_token_embed_dim)
        self.norm1 = nn.LayerNorm(target_token_embed_dim)
        
        self.kan_layer = get_kan_layer(
            target_token_embed_dim, target_token_embed_dim, kan_type,
            grid_size=grid_size, spline_order=spline_order, **kan_kwargs
        )
        
        self.token_grid_spatial_shape = tuple(s // p for s, p in zip(input_spatial_shape, patch_size))
        self.reshape = ReshapeToken(self.token_grid_spatial_shape)
        
        self.channel_adj = nn.Conv3d(target_token_embed_dim, out_channels, kernel_size=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_channels)
        self.upsample = nn.Upsample(scale_factor=patch_size, mode='trilinear', align_corners=False)
        
        if use_se:
            self.se_block = SEBlock3D(out_channels, se_reduction)
        else:
            self.se_block = nn.Identity()
            
        if in_channels != out_channels:
            self.residual_proj = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.residual_proj = nn.Identity()
            
    def forward(self, x):
        residual = self.residual_proj(x)
        
        tokens = self.token_embed(x)
        tokens = self.norm1(tokens)
        
        B, N, E = tokens.shape
        tokens_flat = tokens.view(B * N, E)
        tokens_transformed = self.kan_layer(tokens_flat)
        tokens = tokens_transformed.view(B, N, E)
        
        x_spatial = self.reshape(tokens)
        x_adj = self.channel_adj(x_spatial)
        x_up = self.upsample(self.norm2(x_adj))
        
        if x_up.shape[2:] != residual.shape[2:]:
            x_up = F.interpolate(x_up, size=residual.shape[2:], mode='trilinear', align_corners=False)
            
        x_se = self.se_block(x_up)
        x_out = x_se * x_up
        
        return residual + x_out

