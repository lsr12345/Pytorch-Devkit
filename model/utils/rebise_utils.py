'''
# Author: Shaoran Lu
# Date: 2021/10/04
# Email: lushaoran92@gmail.com
# Description: 

example:

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

from model.utils.ops import CBA
from tools.nninit import common_init

class CatBottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super(CatBottleneck, self).__init__()
        self.conv_list = nn.ModuleList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2d(out_planes//2, out_planes//2, kernel_size=3, stride=2, padding=1, groups=out_planes//2, bias=False),
                nn.BatchNorm2d(out_planes//2),
            )
            self.skip = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(CBA(in_planes, out_planes//2, ksize=1, stride=1, act="relu"))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(CBA(out_planes//2, out_planes//2, ksize=3, stride=stride, act="relu"))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(CBA(out_planes//2, out_planes//4, ksize=3, stride=stride, act="relu"))
            elif idx < block_num - 1:
                self.conv_list.append(CBA(out_planes//int(math.pow(2, idx)), out_planes//int(math.pow(2, idx+1)),  ksize=3, stride=1))
            else:
                self.conv_list.append(CBA(out_planes//int(math.pow(2, idx)), out_planes//int(math.pow(2, idx)),  ksize=3, stride=1))
            
    def forward(self, x):
        out_list = []
        out1 = self.conv_list[0](x)

        for idx, conv in enumerate(self.conv_list[1:]):
            if idx == 0:
                if self.stride == 2:
                    out = conv(self.avd_layer(out1))
                else:
                    out = conv(out1)
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            out1 = self.skip(out1)
        out_list.insert(0, out1)

        out = torch.cat(out_list, dim=1)
        return out

class DetailHead(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes):
        super(DetailHead, self).__init__()
        self.conv = CBA(in_chan, mid_chan, ksize=3, stride=1, act="relu", pad=1)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)
        self.apply(self._init_weights)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x

    def _init_weights(self, m):
        common_init(m)

class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan, ksize=3):
        super(AttentionRefinementModule, self).__init__()
        self.conv = CBA(in_chan, out_chan, ksize=ksize, stride=1, act="relu", pad=1)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size= 1, bias=False)
        self.bn_atten = nn.BatchNorm2d(out_chan)
        self.sigmoid_atten = nn.Sigmoid()
        self.apply(self._init_weights)

    def forward(self, x):
        feat = self.conv(x)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        out = torch.mul(feat, atten)
        return out

    def _init_weights(self, m):
        common_init(m)