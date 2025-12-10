"""
Atrous Spatial Pyramid Pooling (ASPP)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class ASPP3D(nn.Module):
    """
    3D ASPP Module.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 dilation_rates: Tuple[int, ...] = (1, 2, 4, 8)):
        super().__init__()
        self.dilation_rates = dilation_rates
        self.aspp_branches = nn.ModuleList()
        
        for rate in dilation_rates:
            self.aspp_branches.append(
                nn.Sequential(
                    nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=rate, dilation=rate, bias=False),
                    nn.InstanceNorm3d(out_channels),
                    nn.LeakyReLU(inplace=True)
                )
            )
            
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
        
        num_branches = len(dilation_rates) + 1
        self.fusion = nn.Sequential(
            nn.Conv3d(out_channels * num_branches, out_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
        
    def forward(self, x):
        results = []
        for branch in self.aspp_branches:
            results.append(branch(x))
            
        pool_out = self.global_pool(x)
        pool_out = F.interpolate(pool_out, size=x.size()[2:], mode='trilinear', align_corners=False)
        results.append(pool_out)
        
        out = torch.cat(results, dim=1)
        out = self.fusion(out)
        return out

