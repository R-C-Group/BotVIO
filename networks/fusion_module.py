import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.init import kaiming_normal_, orthogonal_
import numpy as np
from torch.distributions.utils import broadcast_all, probs_to_logits, logits_to_probs, lazy_property, clamp_probs
import torch.nn.functional as F
import networks
import networks.depth_encoder
import networks.imu_encoder
from timm.models.layers import DropPath
from networks.basic_modules import ConvNormAct1d, ConvNorm1d

class PositionalEncodingFourier1d(nn.Module):

    def __init__(self, d_hid=1, dim=256):
        super(PositionalEncodingFourier1d, self).__init__()
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(dim, d_hid))

    def _get_sinusoid_encoding_table(self, dim, d_hid):
        ''' Sinusoid position encoding table '''
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(dim)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return self.pos_table[:, :x.size(1)].clone().detach()


class Trans_Fusion(nn.Module):
    """
    Local-Global Features Interaction
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, expan_ratio=6,
                 use_pos_emb=True, num_heads=8, qkv_bias=True, attn_drop=0., drop=0.):
        super().__init__()

        self.dim = dim
        self.pos_embd = None
        self.expan_ration = 1
        if use_pos_emb:
            self.pos_embdx = PositionalEncodingFourier1d(dim=512)
            self.pos_embdy = PositionalEncodingFourier1d(dim=256)

        self.norm_xca = networks.depth_encoder.LayerNorm(self.dim, eps=1e-6)
        self.norm_xcay = networks.depth_encoder.LayerNorm(256, eps=1e-6)

        self.gamma_xca = nn.Parameter(layer_scale_init_value * torch.ones(self.dim),
                                      requires_grad=True) if layer_scale_init_value > 0 else None
        self.xca = networks.depth_encoder.XCAt(self.dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.pwconv0 = ConvNormAct1d(self.dim, self.dim, kernel_size=3, stride=1, dilation=1,
                                      groups=1, norm_layer='bn_1d', act_layer='silu', inplace=True)

        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((self.dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()


    def forward(self, x, y):
        Bv, Lv = x.shape
        x = x.view(Bv, Lv, 1)
        Bi, Li = y.shape
        y = y.view(Bi, Li, 1)   

        input_ = x

        B, L, C = x.shape

        if self.pos_embdx:
            pos_encoding = self.pos_embdx(x)
            x = x + pos_encoding

            pos_encodingy = self.pos_embdy(y)

            y = y + pos_encodingy

        x = x.permute(0, 2, 1)
        y = y.permute(0, 2, 1)

        x = x + self.gamma_xca * self.xca(self.norm_xca(x), self.norm_xcay(y))

        x = x.permute(0, 2, 1).contiguous()
        x_ = self.pwconv0(x)
        x = x + x_

        x = x.permute(0, 2, 1).contiguous()

        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 2, 1).contiguous()  # (N, L, C) -> (N, C, L)

        x = input_ + self.drop_path(x)

        return x

