import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import DropPath
import math
import torch.cuda
import networks
from collections import OrderedDict

import torch.nn.init as init

class Pose_CNN(nn.Module):
    def __init__(self, opt):
        super(Pose_CNN, self).__init__()

        f_len = opt.v_f_len

        stride = 1
        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv1d(f_len, 512, 1)
        self.convs[("pose", 0)] = nn.Conv1d(512, 256, 3, stride, 1)
        self.convs[("pose", 1)] = nn.Conv1d(256, 256, 1, stride, 1)
        self.convs[("pose", 2)] = nn.Conv1d(256, 6*2, 1)

        self.relu = nn.ReLU()
        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, fused):
        cat_features = self.relu(self.convs["squeeze"](fused))

        out = cat_features
        for i in range(3):
            out = self.convs[("pose", i)](out)
            if i != 2:
                out = self.relu(out)

        out = out.mean(2)

        out = 0.01 * out.view(-1, 2, 1, 6)


        axisangle = out[..., :3]
        translation = out[..., 3:]

        return axisangle, translation

