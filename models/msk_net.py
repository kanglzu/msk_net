import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from msk_net.models.blocks import DoubleConv, OutConv, TokenizedKANBlock, KBAM3D, CSGM, ASPP3D

class MSKNet(nn.Module):
    """
    MSK-Net Architecture.
    """
    def __init__(self,
                 n_channels: int,
                 n_classes: int,
                 input_size: Tuple[int, int, int],
                 features: List[int],
                 patch_size: Tuple[int, int, int],
                 target_token_embed_dims: List[int],
                 kan_type: str,
                 skb_config: dict,
                 kbam_config: dict,
                 csgm_config: dict,
                 aspp_config: dict,
                 use_deep_supervision: bool,
                 deep_supervision_weights: List[float]):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.features = features
        self.use_deep_supervision = use_deep_supervision
        self.ds_weights = deep_supervision_weights
        
        # ENCODER
        self.inc = DoubleConv(n_channels, features[0])
        self.kbam1 = KBAM3D(features[0], kan_type=kan_type, **kbam_config)
        
        self.down1 = nn.Sequential(nn.MaxPool3d(2), DoubleConv(features[0], features[1]))
        self.kbam2 = KBAM3D(features[1], kan_type=kan_type, **kbam_config)
        
        s2_shape = tuple(s // 2 for s in input_size)
        self.down2 = nn.MaxPool3d(2)
        self.skb1 = TokenizedKANBlock(
            input_spatial_shape=tuple(s // 2 for s in s2_shape),
            in_channels=features[1], out_channels=features[2], patch_size=patch_size,
            kan_type=kan_type, target_token_embed_dim=target_token_embed_dims[2],
            grid_size=skb_config['grid_sizes'][2], spline_order=skb_config['spline_orders'][2],
            use_se=skb_config['use_se'], se_reduction=skb_config['se_reduction']
        )
        self.kbam3 = KBAM3D(features[2], kan_type=kan_type, **kbam_config)
        
        s3_shape = tuple(s // 2 for s in s2_shape)
        self.down3 = nn.MaxPool3d(2)
        self.skb2 = TokenizedKANBlock(
            input_spatial_shape=tuple(s // 2 for s in s3_shape),
            in_channels=features[2], out_channels=features[3], patch_size=patch_size,
            kan_type=kan_type, target_token_embed_dim=target_token_embed_dims[1],
            grid_size=skb_config['grid_sizes'][1], spline_order=skb_config['spline_orders'][1],
            use_se=skb_config['use_se'], se_reduction=skb_config['se_reduction']
        )
        self.kbam4 = KBAM3D(features[3], kan_type=kan_type, **kbam_config)
        
        s4_shape = tuple(s // 2 for s in s3_shape)
        self.down4 = nn.MaxPool3d(2)
        self.skb3 = TokenizedKANBlock(
            input_spatial_shape=tuple(s // 2 for s in s4_shape),
            in_channels=features[3], out_channels=features[4], patch_size=patch_size,
            kan_type=kan_type, target_token_embed_dim=target_token_embed_dims[0],
            grid_size=skb_config['grid_sizes'][0], spline_order=skb_config['spline_orders'][0],
            use_se=skb_config['use_se'], se_reduction=skb_config['se_reduction']
        )
        
        # BOTTLENECK
        num_aspp = aspp_config['num_modules']
        if num_aspp > 0:
            self.aspp_modules = nn.ModuleList([
                ASPP3D(features[4], features[4], dilation_rates=aspp_config['dilation_rates'])
                for _ in range(num_aspp)
            ])
        else:
            self.aspp_modules = None
            
        # DECODER
        self.up1 = nn.ConvTranspose3d(features[4], features[3], kernel_size=2, stride=2)
        self.csgm1 = CSGM(features[3], features[3], kan_type=kan_type, **csgm_config)
        self.decoder_block1 = TokenizedKANBlock(
            input_spatial_shape=s4_shape, in_channels=features[3], out_channels=features[3],
            patch_size=patch_size, kan_type=kan_type, target_token_embed_dim=target_token_embed_dims[1],
            grid_size=skb_config['grid_sizes'][1], spline_order=skb_config['spline_orders'][1],
            use_se=skb_config['use_se'], se_reduction=skb_config['se_reduction']
        )
        
        self.up2 = nn.ConvTranspose3d(features[3], features[2], kernel_size=2, stride=2)
        self.csgm2 = CSGM(features[2], features[2], kan_type=kan_type, **csgm_config)
        self.decoder_block2 = TokenizedKANBlock(
            input_spatial_shape=s3_shape, in_channels=features[2], out_channels=features[2],
            patch_size=patch_size, kan_type=kan_type, target_token_embed_dim=target_token_embed_dims[2],
            grid_size=skb_config['grid_sizes'][2], spline_order=skb_config['spline_orders'][2],
            use_se=skb_config['use_se'], se_reduction=skb_config['se_reduction']
        )
        
        self.up3 = nn.ConvTranspose3d(features[2], features[1], kernel_size=2, stride=2)
        self.csgm3 = CSGM(features[1], features[1], kan_type=kan_type, **csgm_config)
        self.decoder_conv3 = DoubleConv(features[1], features[1])
        
        self.up4 = nn.ConvTranspose3d(features[1], features[0], kernel_size=2, stride=2)
        self.csgm4 = CSGM(features[0], features[0], kan_type=kan_type, **csgm_config)
        self.decoder_conv4 = DoubleConv(features[0], features[0])
        
        self.outc = OutConv(features[0], n_classes)
        
        if use_deep_supervision:
            self.ds_out1 = OutConv(features[3], n_classes)
            self.ds_out2 = OutConv(features[2], n_classes)
            self.ds_out3 = OutConv(features[1], n_classes)
            
    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x1_att = self.kbam1(x1)
        
        x2 = self.down1(x1_att)
        x2_att = self.kbam2(x2)
        
        x3_in = self.down2(x2_att)
        x3 = self.skb1(x3_in)
        x3_att = self.kbam3(x3)
        
        x4_in = self.down3(x3_att)
        x4 = self.skb2(x4_in)
        x4_att = self.kbam4(x4)
        
        x5_in = self.down4(x4_att)
        x5 = self.skb3(x5_in)
        
        # Bottleneck
        x_bot = x5
        if self.aspp_modules:
            for aspp in self.aspp_modules:
                x_bot = aspp(x_bot)
                
        # Decoder
        d1 = self.up1(x_bot)
        if d1.shape[2:] != x4_att.shape[2:]:
            d1 = F.interpolate(d1, size=x4_att.shape[2:], mode='trilinear', align_corners=False)
        d1_fused = self.csgm1(x4_att, d1)
        d1_out = self.decoder_block1(d1_fused)
        
        d2 = self.up2(d1_out)
        if d2.shape[2:] != x3_att.shape[2:]:
            d2 = F.interpolate(d2, size=x3_att.shape[2:], mode='trilinear', align_corners=False)
        d2_fused = self.csgm2(x3_att, d2)
        d2_out = self.decoder_block2(d2_fused)
        
        d3 = self.up3(d2_out)
        if d3.shape[2:] != x2_att.shape[2:]:
            d3 = F.interpolate(d3, size=x2_att.shape[2:], mode='trilinear', align_corners=False)
        d3_fused = self.csgm3(x2_att, d3)
        d3_out = self.decoder_conv3(d3_fused)
        
        d4 = self.up4(d3_out)
        if d4.shape[2:] != x1_att.shape[2:]:
            d4 = F.interpolate(d4, size=x1_att.shape[2:], mode='trilinear', align_corners=False)
        d4_fused = self.csgm4(x1_att, d4)
        d4_out = self.decoder_conv4(d4_fused)
        
        main_out = self.outc(d4_out)
        
        if self.use_deep_supervision and self.training:
            aux1 = F.interpolate(self.ds_out1(d1_out), size=main_out.shape[2:], mode='trilinear', align_corners=False)
            aux2 = F.interpolate(self.ds_out2(d2_out), size=main_out.shape[2:], mode='trilinear', align_corners=False)
            aux3 = F.interpolate(self.ds_out3(d3_out), size=main_out.shape[2:], mode='trilinear', align_corners=False)
            return [main_out, aux1, aux2, aux3]
            
        return main_out
    
    def get_parameter_count(self) -> float:
        """
        Get the total number of trainable parameters in millions.
        """
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params / 1e6


def build_msk_net_from_config(config: dict):
    model_cfg = config['model']
    
    return MSKNet(
        n_channels=model_cfg['n_channels'],
        n_classes=model_cfg['n_classes'],
        input_size=tuple(model_cfg['input_size']),
        features=model_cfg['features'],
        patch_size=tuple(model_cfg['skb']['patch_size']),
        target_token_embed_dims=model_cfg['skb']['target_token_embed_dims'],
        kan_type=config['kan']['type'],
        skb_config=model_cfg['skb'],
        kbam_config=model_cfg['kbam'],
        csgm_config=model_cfg['csgm'],
        aspp_config=model_cfg['aspp'],
        use_deep_supervision=model_cfg['deep_supervision']['enabled'],
        deep_supervision_weights=model_cfg['deep_supervision']['weights']
    )
