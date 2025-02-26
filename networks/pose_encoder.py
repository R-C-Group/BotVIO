import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import DropPath
import math
import torch.cuda
import networks.depth_encoder

from torch.autograd import Variable


class LGFIP(nn.Module):
    """
    Local-Global Features Interaction
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, expan_ratio=6,
                 use_pos_emb=True, num_heads=6, qkv_bias=True, attn_drop=0., drop=0.):
        super().__init__()

        self.dim = dim
        self.expan_ration = 1
        self.pos_embd = None
        self.use_pos_emb = use_pos_emb
        if use_pos_emb:
            self.pos_embd = networks.depth_encoder.PositionalEncodingFourier(dim=self.dim)

        self.norm_xca = networks.depth_encoder.LayerNorm(self.dim, eps=1e-6)

        self.gamma_xca = nn.Parameter(layer_scale_init_value * torch.ones(self.dim),
                                      requires_grad=True) if layer_scale_init_value > 0 else None
        self.xca = networks.depth_encoder.XCAC(self.dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.pwconv0 = networks.depth_encoder.ConvNormAct(self.dim, self.dim*self.expan_ration, kernel_size=3, stride=1, dilation=1,
                                   groups=1, norm_layer='bn_2d', act_layer='silu', inplace=True)
        self.proj_drop = nn.Dropout(drop_path)

        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((self.dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        ## temproal
        self.seq_len = 2
        self.num_heads = num_heads
        self.relative_position_bias_table = nn.Parameter(torch.zeros(2 * self.seq_len - 1, self.num_heads))  # 2*T-2, nH
        self.register_buffer("relative_position_index", self.get_position_index())  # T, T

    def get_position_index(self):
        coords = torch.arange(self.seq_len)  # T
        relative_coords = coords[:, None] - coords[None, :]  # T, T
        relative_coords += self.seq_len - 1  # shift to start from 0
        relative_position_index = relative_coords  # T, T
        return relative_position_index

    def forward(self, x):
        input_ = x

        # XCA
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(0, 2, 1)

        if self.use_pos_emb:
            pos_encoding = self.pos_embd(B, H, W).reshape(B, -1, x.shape[1]).permute(0, 2, 1)
            x = x + pos_encoding

        T = self.seq_len
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.reshape(-1)].reshape(T, T, -1)  # T, T, nH
        relative_position_bias = relative_position_bias[None].permute(3, 0, 1, 2).expand(self.num_heads,
                                                                                         (C // self.num_heads) * (
                                                                                                     C // self.num_heads),
                                                                                         T, T)
        relative_position_bias = F.interpolate(relative_position_bias, scale_factor=0.5)
        relative_position_bias = F.pixel_shuffle(relative_position_bias,
                                                 upscale_factor=int(C // self.num_heads)).permute(1, 0, 2, 3)

        x = x + self.gamma_xca * self.xca(self.norm_xca(x), relative_position_bias, self.use_pos_emb)

        x = x.reshape(B, C, H, W)

        xi = self.pwconv0(x)
        x = x + (xi)

        x = x.reshape(B, H, W, C)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input_ + self.drop_path(x)

        return x

class PoseEncoder(nn.Module):
    """
    Pose_Encoder
    """
    def __init__(self, in_chans=6, height=192, width=640,
                 global_block=[1, 1, 1], global_block_type=['LGFI', 'LGFI', 'LGFI'],
                 drop_path_rate=0.2, layer_scale_init_value=1e-6, expan_ratio=6,
                 heads=[8, 8, 8], use_pos_embd_xca=[True, False, False], **kwargs):

        super().__init__()

        self.num_ch_enc = np.array([48, 80, 128])
        self.pose = [3, 3, 4]
        self.dims = [48, 80, 128]
        if height == 192 and width == 640:
            self.dilation = [[1, 2], [1, 2], [1, 2, 3]]
        elif height == 320 and width == 1024:
            self.dilation = [[1, 2, 5], [1, 2, 5], [1, 2, 5, 1, 2, 5, 2, 4, 10]]

        for g in global_block_type:
            assert g in ['None', 'LGFI']

        self.stem = nn.Sequential(
            networks.depth_encoder.Conv(in_chans, self.dims[0], kSize=3, stride=2, padding=1, bn_act=True),
            networks.depth_encoder.Conv(self.dims[0], self.dims[0], kSize=3, stride=2, padding=1, bn_act=True),
            networks.depth_encoder.Conv(self.dims[0], self.dims[0], kSize=3, stride=2, padding=1, bn_act=True),
        )

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.pose))]
        cur = 0
        for i in range(3):
            stage_blocks = []
            for j in range(self.pose[i]):
                if j > self.pose[i] - global_block[i] - 1:
                    if global_block_type[i] == 'LGFI':
                        stage_blocks.append(LGFIP(dim=self.dims[i], drop_path=dp_rates[cur + j],
                                                 expan_ratio=expan_ratio,
                                                 use_pos_emb=use_pos_embd_xca[i], num_heads=heads[i],
                                                 layer_scale_init_value=layer_scale_init_value,
                                                 ))

                    else:
                        raise NotImplementedError
                else:
                    stage_blocks.append(networks.depth_encoder.DilatedConv(dim=self.dims[i], k=3, dilation=self.dilation[i][j], drop_path=dp_rates[cur + j],
                                                    layer_scale_init_value=layer_scale_init_value,
                                                    expan_ratio=expan_ratio))

            self.stages.append(nn.Sequential(*stage_blocks))
            cur += self.pose[i]

        self.conv1 = networks.depth_encoder.Conv(self.dims[0], self.dims[1], kSize=3, stride=2, padding=1, bn_act=False)
        self.conv2 = networks.depth_encoder.Conv(self.dims[1], self.dims[2], kSize=3, stride=2, padding=1, bn_act=False)
        self.conv3 = networks.depth_encoder.Conv(self.dims[2], self.dims[2], kSize=3, stride=2, padding=1, bn_act=False)

        self.drop_out = nn.Dropout(0.2)
        
        ## only for VIO
        self.visual_headk = nn.Sequential(
            nn.Linear(3840, 512),
            )

        self.apply(self._init_weights)

    def forward_features(self, x):
        x = (x - 0.45) / 0.225
        x = self.stem(x)

        for i in range(0, 3):
            for s in range(len(self.stages[i]) - 1):
                x = self.stages[i][s](x)
            x = self.stages[i][-1](x)
                    
            if i == 0:
                x = self.conv1(x)

            elif i == 1:
                x = self.conv2(x)

            elif i == 2:
                x = self.conv3(x)

        features = x
        
        ## only for VIO
        B, C, H, W = features.shape
        features = features.view(B, -1)

        features = self.visual_headk(features)

        return features

    def forward(self, x):
        x = self.forward_features(x)

        return x

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        elif isinstance(m, (networks.depth_encoder.LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
