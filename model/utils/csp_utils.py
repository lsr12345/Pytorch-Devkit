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
import numpy as np

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

from model.utils.ops import CBA, get_activation

class ResLayer(nn.Module):
    "Residual layer with `in_channels` inputs."

    def __init__(self, in_channels: int):
        super().__init__()
        mid_channels = in_channels // 2
        self.layer1 = CBA(
            in_channels, mid_channels, ksize=1, stride=1, act="lrelu"
        )
        self.layer2 = CBA(
            mid_channels, in_channels, ksize=3, stride=1, act="lrelu"
        )

    def forward(self, x):
        out = self.layer2(self.layer1(x))
        return x + out

class Focus(nn.Module):
    """Focus width and height information into channel space."""

    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super().__init__()
        self.conv = CBA(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        return self.conv(x)

class Bottleneck(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        expansion=0.5,
        act="silu",
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = CBA(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = CBA(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y

class CSPLayer(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions
                                   x: c,w,h
                CBA(): o//2,w,h            CBA():o//2,w,h
        Bottelneck():o//2,w,h    
                                cat(): o,w,h
                                CBA():o,w,h      

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        n=1,
        shortcut=False,
        expansion=0.5,
        act="silu",
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = CBA(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = CBA(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = CBA(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        module_list = [
            Bottleneck(
                hidden_channels, hidden_channels, shortcut, 1.0,  act=act
            )
            for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)

class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP
    
                                                                        x:c,w,h
                                                                CBA():c//2,w,h
maxpool2d(5):c//2,w,h                    *:c//2,w,h                            maxpool2d(9):c//2,w,h              maxpool2d(13):c//2,w,h
                                                                cat(): c*2, w, h
                                                                CBA():o,w,h

    """

    def __init__(
        self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"
    ):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = CBA(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.m = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernel_sizes
            ]
        )
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = CBA(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x

class SPPBottleneck_1D(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP
    
                                                                        x:c,w
                                                                CBA():c//2,w
maxpool2d(5):c//2,w                    *:c//2,w                            maxpool2d(9):c//2,w              maxpool2d(13):c//2,w
                                                                cat(): c*2, w
                                                                CBA():o,w

    """

    def __init__(
        self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"
    ):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = nn.Sequential(
            nn.Conv1d(
            in_channels,
            hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=1,
            bias=False,
        ),
        nn.BatchNorm1d(hidden_channels),
        get_activation(activation, inplace=True)
        )
        self.m = nn.ModuleList(
            [
                nn.MaxPool1d(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernel_sizes
            ]
        )
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = nn.Sequential(
            nn.Conv1d(
            conv2_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=1,
            bias=False,
        ),
        nn.BatchNorm1d(out_channels),
        get_activation(activation, inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x