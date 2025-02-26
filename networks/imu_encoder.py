import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.init import kaiming_normal_, orthogonal_
import numpy as np
from torch.distributions.utils import broadcast_all, probs_to_logits, logits_to_probs, lazy_property, clamp_probs
import torch.nn.functional as F
import networks.depth_encoder
from timm.models.layers import DropPath
import math
from networks.basic_modules import ConvNormAct1d, ConvNorm1d


class PositionalEncodingFourier1d(nn.Module):

    def __init__(self, d_hid=11, dim=256):
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

class TimeEmbeddingSine(nn.Module):
    """
    Same as below for temporal dimension
    """

    def __init__(self, max_len=200, d_model=512):
        super().__init__()
        self.max_len = max_len
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        te = torch.zeros(max_len, 1, d_model)
        te[:, 0, 0::2] = torch.sin(position * div_term)
        te[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("te", te)

    def forward(self, ln):
        pos_t = self.te[:ln]
        return pos_t

class TimeEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, num_pos_feats=200, d_model=512):
        super().__init__()
        self.time_embed = nn.Embedding(num_pos_feats, d_model)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.time_embed.weight)

    def forward(self, ln):
        return self.time_embed.weight[:ln].unsqueeze(1)


class CDilated1d(nn.Module):
    """
    This class defines the dilated convolution.
    """

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1, bias=False):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        """
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, padding=padding, bias=bias,
                              dilation=d, groups=groups)

    def forward(self, input):
        """
        :param input: input feature map
        :return: transformed feature map
        """

        output = self.conv(input)
        return output

class DilatedConv1d(nn.Module):
    """
    A single Dilated Convolution layer in the Consecutive Dilated Convolutions (CDC) module.
    """
    def __init__(self, dim, k, dilation=1, stride=1, drop_path=0.,
                 layer_scale_init_value=1e-6, expan_ratio=6):
        """
        :param dim: input dimension
        :param k: kernel size
        :param dilation: dilation rate
        :param drop_path: drop_path rate
        :param layer_scale_init_value:
        :param expan_ratio: inverted bottelneck residual
        """

        super().__init__()

        self.ddwconv = CDilated1d(dim, dim, kSize=k, stride=stride, groups=dim, d=dilation)
        self.bn1 = nn.BatchNorm1d(dim)
        self.expan_ratio = 4

        self.pwconv1 = nn.Conv1d(dim, dim*self.expan_ratio, 1)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.pwconv2 = nn.Conv1d(dim*self.expan_ratio, dim, 1)
        self.drop = nn.Dropout(drop_path)

        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        
        xt = self.ddwconv(x)
        x = self.bn1(xt)

        x = self.pwconv1(x)
        x = self.act(x)

        x = self.pwconv2(x)

        x = x.permute(0, 2, 1)  # (N, C, L) -> (N, L, C)

        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 2, 1).contiguous()  # (N, L, C) -> (N, C, L)

        x = input + self.drop_path(x)

        return x


class LGFI1dc(nn.Module):
    """
    Local-Global Features Interaction
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, expan_ratio=6,
                 use_pos_emb=True, num_heads=8, qkv_bias=True, attn_drop=0., drop=0.):
        super().__init__()

        self.dim = dim
        self.use_pos_emb = use_pos_emb
        self.pos_embd = PositionalEncodingFourier1d(dim=self.dim)

        self.norm_xca = networks.depth_encoder.LayerNorm(self.dim, eps=1e-6)

        self.gamma_xca = nn.Parameter(layer_scale_init_value * torch.ones(self.dim),
                                      requires_grad=True) if layer_scale_init_value > 0 else None
        self.xca = networks.depth_encoder.XCAC(self.dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.pwconv0 = ConvNormAct1d(self.dim, self.dim, kernel_size=3, stride=1, dilation=1,
                                      groups=1, norm_layer='bn_1d', act_layer='silu', inplace=True)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((self.dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        ## temproal
        self.seq_len = 11
        self.num_heads = num_heads
        self.relative_position_bias_table = nn.Parameter(torch.zeros(2 * self.seq_len - 1, self.num_heads))  # 2*T-1, nH
        self.register_buffer("relative_position_index", self.get_position_index())  # T, T

    def get_position_index(self):
        coords = torch.arange(self.seq_len)  # T
        relative_coords = coords[:, None] - coords[None, :]  # T, T
        relative_coords += self.seq_len - 1  # shift to start from 0
        relative_position_index = relative_coords  # T, T
        return relative_position_index

    def forward(self, x):
        input_ = x

        B, C, L = x.shape

        if self.use_pos_emb:
            pos_encoding = self.pos_embd(x)
            x = x + pos_encoding

        x = x.permute(0, 2, 1)

        T = self.seq_len
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.reshape(-1)].reshape(T, T, -1)  # T, T, nH
        relative_position_bias = relative_position_bias[None].permute(3, 0, 1, 2).expand(self.num_heads,
                                                                                         (C // self.num_heads) * (
                                                                                                     C // self.num_heads),
                                                                                         T, T)
        relative_position_bias = F.interpolate(relative_position_bias, scale_factor=1/11)
        relative_position_bias = F.pixel_shuffle(relative_position_bias,
                                                 upscale_factor=int(C // self.num_heads)).permute(1, 0, 2, 3)
        
        x = x + self.gamma_xca * self.xca(self.norm_xca(x), relative_position_bias, self.use_pos_emb)

        x = x.permute(0, 2, 1)
        
        x_ = self.pwconv0(x)
        x = x + x_

        x = x.permute(0, 2, 1)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 2, 1).contiguous()  # (N, L, C) -> (N, C, L)

        x = input_ + self.drop_path(x)

        return x


def initialization(net):
    #Initilization
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
            kaiming_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.kaiming_normal_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.kaiming_normal_(param.data)
                elif 'bias_ih' in name:
                    param.data.fill_(0)
                elif 'bias_hh' in name:
                    param.data.fill_(0)
                    n = param.size(0)
                    start, end = n//4, n//2
                    param.data[start:end].fill_(1.)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()



class InertialEncoder(nn.Module):
    def __init__(self, in_chans=6, height=192, width=640,
                 global_block=[1, 1, 1], global_block_type=['LGFI1d', 'LGFI1d', 'LGFI1d'],
                 drop_path_rate=0.2, layer_scale_init_value=1e-6, expan_ratio=6,
                 heads=[8, 8, 8], use_pos_embd_xca=[True, False, False], **kwargs):

        super().__init__()

        self.num_ch_enc = np.array([48, 80, 128])
        self.imu = [1, 1, 2]
        # self.imu = [1, 1, 2]
        self.dims = [64, 128, 224]
        if height == 192 and width == 640:
            self.dilation = [[0], [0], [0]]
            # self.dilation = [[0], [0], [0]]
        elif height == 320 and width == 1024:
            self.dilation = [[1, 2, 5], [1, 2, 5], [1, 2, 5, 1, 2, 5, 2, 4, 10]]

        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, 11)]

        self.stages = nn.ModuleList()
        cur = 0
        for i in range(3):
            stage_blocks = []
            for j in range(self.imu[i]):
                if j > self.imu[i] - global_block[i] - 1:
                    if global_block_type[i] == 'LGFI1d':
                        stage_blocks.append(LGFI1dc(dim=self.dims[i], drop_path=dp_rates[0],
                                                 expan_ratio=expan_ratio,
                                                 use_pos_emb=use_pos_embd_xca[i], num_heads=heads[i],
                                                 layer_scale_init_value=layer_scale_init_value,
                                                 ))

                    else:
                        raise NotImplementedError
                else:
                    stage_blocks.append(DilatedConv1d(dim=self.dims[i], k=3, dilation=self.dilation[i][j], drop_path=dp_rates[0],
                                                    layer_scale_init_value=layer_scale_init_value,
                                                    expan_ratio=expan_ratio))

            self.stages.append(nn.Sequential(*stage_blocks))
            cur += self.imu[i]

        self.conv1 = nn.Conv1d(self.dims[0], self.dims[1], kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv1d(self.dims[1], self.dims[2], kernel_size=3, padding=1, bias=False)
        
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(6, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0),
            )

        self.proj = nn.Linear(224 * 1 * 11, 256)
        self.drop_out = nn.Dropout(0.1)

    def forward(self, x):
        batch_size = x.shape[0]
        
        x = self.encoder_conv(x.permute(0, 2, 1))

        for i in range(0, 3):
            for s in range(len(self.stages[i]) - 1):
                x = self.stages[i][s](x)

            x = self.stages[i][-1](x)
                    
            if i == 0:
                x = self.conv1(x)

            elif i == 1:
                x = self.conv2(x)

        out = self.proj(x.view(x.shape[0], -1))           
        return out.view(batch_size, 256)
