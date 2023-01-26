'''
# Author: Shaoran Lu
# Date: 2021/10/04
# Email: lushaoran92@gmail.com
# Description: 主干网络

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

from model.utils.ops import CBA, hswish
from tools.nninit import trunc_normal_, common_init

from model.utils.res_utils import BasicBlock, Bottleneck
from model.utils.swin_utils import PatchEmbed, PatchMerging, BasicLayer
from model.utils.csp_utils import Focus, CSPLayer, SPPBottleneck, ResLayer
from model.utils.rebise_utils import CatBottleneck
from model.utils.mobilevit_utils import MV2Block, MobileViTBlock
from model.utils.mobilenetv3_utils import SeModule, MobilenetBlock

class CrnnBackbone(nn.Module):
    def __init__(self, type_mode='basic', mode='small', nc=1,  out_feature_size=512, leakyRelu=False, act = 'silu'):
        super(CrnnBackbone, self).__init__()
        
        if type_mode == 'basic':
            backbone = nn.Sequential()
            ks = [3, 3, 3, 3, 3, 3, 2]
            ps = [1, 1, 1, 1, 1, 1, 0]
            ss = [1, 1, 1, 1, 1, 1, 1]
            nm = [64, 128, 256, 256, 512, 512, 512]
            def convRelu(i, batchNormalization=False):
                nIn = nc if i == 0 else nm[i - 1]
                nOut = nm[i]
                backbone.add_module('conv{0}'.format(i),
                            nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
                if batchNormalization:
                    backbone.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
                if leakyRelu:
                    backbone.add_module('relu{0}'.format(i),
                                nn.LeakyReLU(0.2, inplace=True))
                else:
                    backbone.add_module('relu{0}'.format(i), nn.ReLU(True))
            convRelu(0)
            backbone.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))
            convRelu(1)
            backbone.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))
            convRelu(2, True)
            convRelu(3)
            backbone.add_module('pooling{0}'.format(2),
                        nn.MaxPool2d((2, 2), (2, 1), (0, 1)))
            convRelu(4, True)
            convRelu(5)
            backbone.add_module('pooling{0}'.format(3),
                        nn.MaxPool2d((2, 2), (2, 1), (0, 1)))
            convRelu(6, True)
            self.backbone = backbone
        elif type_mode in ['dark_s', 'dark_m', 'dark_l']:
            
            if mode == 'small':
                dep_mul = 0.33
                wid_mul = 0.50
            elif mode == 'middle':
                dep_mul = 0.67
                wid_mul = 0.75          
            else:
                dep_mul = 1.0
                wid_mul = 1.0

            base_channels = int(wid_mul * 64)
            base_depth = max(round(dep_mul * 3), 1)
            self.dark1 = nn.Sequential(
                CBA(nc, base_channels , 3, 2, act=act),
                CSPLayer(
                    base_channels,
                    base_channels,
                    n=base_depth,
                    act=act,
                ),
            )
            self.dark2 = nn.Sequential(
                CBA(base_channels, base_channels * 2, 3, 2, act=act),
                SPPBottleneck(base_channels * 2, base_channels * 2, kernel_sizes=(3, 5, 7), activation=act),
                CSPLayer(
                    base_channels * 2,
                    base_channels * 2,
                    n=base_depth,
                    act=act,
                ),
            )
            self.dark3 = nn.Sequential(
                CBA(base_channels * 2, base_channels * 4, 3, 2, act=act),
                CSPLayer(
                    base_channels * 4,
                    base_channels * 4,
                    n=base_depth * 3,
                    act=act,
                ),
            )
            self.dark4 = nn.Sequential(
                CBA(base_channels * 4, base_channels * 8, 3, (2,1), act=act),
                CSPLayer(
                    base_channels * 8,
                    base_channels * 8,
                    n=base_depth * 3,
                    act=act,
                ),
            )
            self.dark5 = nn.Sequential(
                CBA(base_channels * 8, base_channels * 16, 3, (2,1), act=act),
                SPPBottleneck(base_channels * 16, base_channels * 16, kernel_sizes=(3, 5, 7), activation=act),
                CSPLayer(
                    base_channels * 16,
                    base_channels * 16,
                    n=base_depth,
                    shortcut=False,
                    act=act,
                ),
            )
            self.conv = CBA(base_channels * 16, out_feature_size, 1, 1, act=act)
            self.backbone = nn.Sequential(
                self.dark1,
                self.dark2,
                self.dark3,
                self.dark4,
                self.dark5,
                self.conv
            )
        else:
            raise NotImplementedError('crnn backbone {} not supported.'.format(type_mode))

        self.apply(self._init_weights)

    def forward(self, x):
        out = self.backbone(x)
        return out

    def _init_weights(self, m):
        common_init(m)

class ResNet(nn.Module):

    def __init__(self, type_mode='18', num_classes=100, extract_feature=False, out_features=("res2", "res3", "res4", "res5")):
        super().__init__()
        if type_mode == '18' :
            block = BasicBlock
            num_block = [2, 2, 2, 2]

        if type_mode == '34':
            block = BasicBlock
            num_block = [3, 4, 6, 3]           

        if type_mode == '50':
            block = Bottleneck
            num_block = [3, 4, 6, 3]

        if type_mode == '101':
            block = Bottleneck
            num_block = [3, 4, 23, 3]      

        if type_mode == '152':
            block = Bottleneck
            num_block = [3, 8, 36, 3]      

        self._norm_layer = nn.BatchNorm2d

        self.extract_feature = extract_feature
        self.out_features = out_features

        self.inplanes = 64
        self.dilation = 1
        self.groups = 1
        self.base_width = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, num_block[0])
        self.layer2 = self._make_layer(block, 128, num_block[1], stride=2, dilate=False)
        self.layer3 = self._make_layer(block, 256, num_block[2], stride=2, dilate=False)
        self.layer4 = self._make_layer(block, 512, num_block[3], stride=2,  dilate=False)

        if not self.extract_feature:
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.apply(self._init_weights)

    def forward(self, x):
        outputs = {}
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        outputs["stem"] = x

        x = self.layer1(x)
        outputs["res2"] = x
        x = self.layer2(x)
        outputs["res3"] = x
        x = self.layer3(x)
        outputs["res4"] = x
        x = self.layer4(x)
        outputs["res5"] = x
        if not self.extract_feature:
            output = self.avg_pool(x)
            output = output.view(output.size(0), -1)
            output = self.fc(output)
            return output

        return {k: v for k, v in outputs.items() if k in self.out_features}

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _init_weights(self, m):
        common_init(m)

class SwinTransformer(nn.Module):

    def __init__(self, img_size=None, patch_size=4, in_chans=3, num_classes=1000,
                  mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                  norm_layer=nn.LayerNorm, ape=False, patch_norm=True, type_mode='small',
                 use_checkpoint=False, extract_feature=False, out_features=("swin2", "swin3", "swin4", "swin5")):  
        super().__init__()
        self.mode_config = {'tiny':{'depths': [2,2,6,2], 'embed_dim': 96, 'window_size': 7 ,  'num_heads': [ 3, 6, 12, 24 ],  'drop_path_rate': 0.2},
                                        'small':{'depths': [2,2,18,2], 'embed_dim': 96, 'window_size': 7 ,  'num_heads': [ 3, 6, 12, 24 ],  'drop_path_rate': 0.3},
                                        'base':{'depths': [2,2,18,2], 'embed_dim': 128, 'window_size': 7 ,  'num_heads': [ 4, 8, 16, 32 ],  'drop_path_rate': 0.5},
                                        'large':{'depths': [2,2,18,2], 'embed_dim': 192, 'window_size': 7 ,  'num_heads': [ 6, 12, 24, 48 ],  'drop_path_rate': 0.5}}
        depths = self.mode_config[type_mode]['depths']
        embed_dim = self.mode_config[type_mode]['embed_dim']
        window_size = self.mode_config[type_mode]['window_size']
        num_heads = self.mode_config[type_mode]['num_heads']
        drop_path_rate = self.mode_config[type_mode]['drop_path_rate']
        self.extract_feature = extract_feature
        self.out_features = out_features

        self.img_size = img_size
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
                                                                            norm_layer=norm_layer if self.patch_norm else None)

        if self.ape and self.img_size is not None:
            img_size = (img_size, img_size)
            patch_size = (patch_size, patch_size)
            patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]

            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1]))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                                                dim=int(embed_dim * 2 ** i_layer),
                                                depth=depths[i_layer],
                                                num_heads=num_heads[i_layer],
                                                window_size=window_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                                norm_layer=norm_layer,
                                                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                                                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features
        for i_layer in range(self.num_layers):
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self.norm = norm_layer(self.num_features[-1])
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features[-1], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        """Forward function."""
        x = self.patch_embed(x)

        Wh, Ww = x.size(2), x.size(3)
        if self.ape and self.img_size is not None:
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)

            if self.extract_feature:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)
                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)
                
        if self.extract_feature:
            return tuple(outs)
        else:
            return x_out

    def forward(self, x):
        x = self.forward_features(x)
        if self.extract_feature:
            outputs = {}
            outputs["swin2"] = x[0]
            outputs["swin3"] = x[1]
            outputs["swin4"] = x[2]
            outputs["swin5"] = x[3]
            return {k: v for k, v in outputs.items() if k in self.out_features}
        else:
            x = self.norm(x)
            x = self.avgpool(x.transpose(1, 2))
            x = torch.flatten(x, 1)
            x = self.head(x)
            return x

class Darknet(nn.Module):
    depth2blocks = {21: [1, 2, 2, 1], 53: [2, 8, 8, 4]}

    def __init__(
        self,
        depth,
        in_channels=3,
        stem_out_channels=32,
        out_features=("dark3", "dark4", "dark5"),
    ):
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        self.stem = nn.Sequential(
            CBA(in_channels, stem_out_channels, ksize=3, stride=1, act="lrelu"),
            *self.make_group_layer(stem_out_channels, num_blocks=1, stride=2),
        )
        in_channels = stem_out_channels * 2

        num_blocks = Darknet.depth2blocks[depth]
        self.dark2 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[0], stride=2)
        )
        in_channels *= 2
        self.dark3 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[1], stride=2)
        )
        in_channels *= 2
        self.dark4 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[2], stride=2)
        )
        in_channels *= 2

        self.dark5 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[3], stride=2),
            *self.make_spp_block([in_channels, in_channels * 2], in_channels * 2),
        )

        self.apply(self._init_weights)

    def make_group_layer(self, in_channels: int, num_blocks: int, stride: int = 1):
        return [
            CBA(in_channels, in_channels * 2, ksize=3, stride=stride, act="lrelu"),
            *[(ResLayer(in_channels * 2)) for _ in range(num_blocks)],
        ]

    def make_spp_block(self, filters_list, in_filters):
        m = nn.Sequential(
            *[
                CBA(in_filters, filters_list[0], 1, stride=1, act="lrelu"),
                CBA(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
                SPPBottleneck(
                    in_channels=filters_list[1],
                    out_channels=filters_list[0],
                    activation="lrelu",
                ),
                CBA(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
                CBA(filters_list[1], filters_list[0], 1, stride=1, act="lrelu"),
            ]
        )
        return m

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}

    def _init_weights(self, m):
        common_init(m)

class CSPDarknet(nn.Module):
    def __init__(
        self,
        dep_mul,
        wid_mul,
        out_features=("dark3", "dark4", "dark5"),
        act="silu",
    ):
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features

        base_channels = int(wid_mul * 64)
        base_depth = max(round(dep_mul * 3), 1)

        self.stem = Focus(3, base_channels, ksize=3, act=act)
        self.dark2 = nn.Sequential(
            CBA(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(
                base_channels * 2,
                base_channels * 2,
                n=base_depth,
                act=act,
            ),
        )
        self.dark3 = nn.Sequential(
            CBA(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(
                base_channels * 4,
                base_channels * 4,
                n=base_depth * 3,
                act=act,
            ),
        )

        self.dark4 = nn.Sequential(
            CBA(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(
                base_channels * 8,
                base_channels * 8,
                n=base_depth * 3,
                act=act,
            ),
        )

        self.dark5 = nn.Sequential(
            CBA(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
            CSPLayer(
                base_channels * 16,
                base_channels * 16,
                n=base_depth,
                shortcut=False,
                act=act,
            ),
        )

        self.apply(self._init_weights)

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}

    def _init_weights(self, m):
        common_init(m)

class StdcNet(nn.Module):
    def __init__(self, type_mode='l', base_channels=64, block_num=4, num_classes=1000, extract_feature=False, out_features=("stage_1", "stage_2", "stage_3", "stage_4", "stage_5")):
        super().__init__()
        block = CatBottleneck
        if type_mode == 'l' :
            layers = [4,5,3]
        else:
            layers = [2,2,2]

        self.features = self._make_layers(base_channels, layers, block_num, block)
        self.extract_feature = extract_feature
        self.out_features = out_features

        if not extract_feature:
            self.conv_last = CBA(base_channels*16, max(1024, base_channels*16), 1, 1, act="relu")
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(max(1024, base_channels*16), max(1024, base_channels*16), bias=False)
            self.bn = nn.BatchNorm1d(max(1024, base_channels*16))
            self.relu = nn.ReLU(inplace=True)
            self.dropout = nn.Dropout(p=0.2)
            self.linear = nn.Linear(max(1024, base_channels*16), num_classes, bias=False)
        else:
            self.x2 = nn.Sequential(self.features[:1])
            self.x4 = nn.Sequential(self.features[1:2])
            self.x8 = nn.Sequential(self.features[2:6])
            self.x16 = nn.Sequential(self.features[6:11])
            self.x32 = nn.Sequential(self.features[11:])

        self.apply(self._init_weights)

    def forward(self, x):
        if not self.extract_feature:
            out = self.features(x)
            out = self.conv_last(out).pow(2)
            out = self.gap(out).flatten(1)
            out = self.fc(out)
            out = self.relu(out)
            out = self.dropout(out)
            out = self.linear(out)
            return out
        else:
            outputs = {}
            output_0 = self.x2(x)
            outputs["stage_1"] = output_0
            output_1 = self.x4(output_0)
            outputs["stage_2"] = output_1
            output_2 = self.x8(output_1)
            outputs["stage_3"] = output_2
            output_3 = self.x16(output_2)
            outputs["stage_4"] = output_3
            output_4 = self.x32(output_3)
            outputs["stage_5"] = output_4

            return {k: v for k, v in outputs.items() if k in self.out_features}

    def _make_layers(self, base, layers, block_num, block):
        features = []
        features += [CBA(3, base//2, 3, 2, act="relu")]
        features += [CBA(base//2, base, 3, 2, act="relu")]

        for i, layer in enumerate(layers):
            for j in range(layer):
                if i == 0 and j == 0:
                    features.append(block(base, base*4, block_num, 2))
                elif j == 0:
                    features.append(block(base*int(math.pow(2,i+1)), base*int(math.pow(2,i+2)), block_num, 2))
                else:
                    features.append(block(base*int(math.pow(2,i+2)), base*int(math.pow(2,i+2)), block_num, 1))

        return nn.Sequential(*features)

    def _init_weights(self, m):
        common_init(m)

class MobileViT(nn.Module):
    def __init__(self, image_size=None, type_mode='mobilevit_xs', expansion=4, kernel_size=3, patch_size=(2, 2),
                                num_classes=1000, extract_feature=False, out_features=("stage_1", "stage_2", "stage_3", "stage_4", "stage_5")):
        super().__init__()
        assert type_mode in ['mobilevit_xxs', 'mobilevit_xs', 'mobilevit_s']
        self.extract_feature = extract_feature
        self.out_features = out_features
        if type_mode == 'mobilevit_s':
            print('type_mode : mobilevit_s')
            dims = [144, 192, 240]
            channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]
        elif type_mode == 'mobilevit_xs':
            print('type_mode : mobilevit_xs')
            dims = [96, 120, 144]
            channels = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384]
        else:
            print('type_mode : mobilevit_xxs')
            dims = [64, 80, 96]
            channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320]
            expansion=2

        L = [2, 4, 3]

        self.conv1 = CBA(3, channels[0],  ksize=3, stride=2, pad=1,  bias=False, act="silu")

        self.mv2 = nn.ModuleList([])
        self.mv2.append(MV2Block(channels[0], channels[1], 1, expansion))
        self.mv2.append(MV2Block(channels[1], channels[2], 2, expansion))
        self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion))
        self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion))
        self.mv2.append(MV2Block(channels[3], channels[4], 2, expansion))
        self.mv2.append(MV2Block(channels[5], channels[6], 2, expansion))
        self.mv2.append(MV2Block(channels[7], channels[8], 2, expansion))
        
        self.mvit = nn.ModuleList([])
        self.mvit.append(MobileViTBlock(dims[0], L[0], channels[5], kernel_size, patch_size, int(dims[0]*2)))
        self.mvit.append(MobileViTBlock(dims[1], L[1], channels[7], kernel_size, patch_size, int(dims[1]*4)))
        self.mvit.append(MobileViTBlock(dims[2], L[2], channels[9], kernel_size, patch_size, int(dims[2]*4)))

        self.conv2 = CBA(channels[-2], channels[-1],  ksize=1, stride=1, pad=0,  bias=False, act="silu")
        if not self.extract_feature and image_size is not None:
            self.pool = nn.AvgPool2d(image_size//32, 1)
            self.fc = nn.Linear(channels[-1], num_classes, bias=False)

        self.apply(self._init_weights)

    def forward(self, x):
        if not self.extract_feature:
            x = self.conv1(x)
            x = self.mv2[0](x)

            x = self.mv2[1](x)
            x = self.mv2[2](x)
            x = self.mv2[3](x)

            x = self.mv2[4](x)
            x = self.mvit[0](x)

            x = self.mv2[5](x)
            x = self.mvit[1](x)

            x = self.mv2[6](x)
            x = self.mvit[2](x)
            x = self.conv2(x)

            x = self.pool(x).view(-1, x.shape[1])
            x = self.fc(x)
            return x
        else:
            outputs = {}
            x = self.conv1(x)
            output_0 = self.mv2[0](x)
            outputs["stage_1"] = output_0

            x = self.mv2[1](output_0)
            x = self.mv2[2](x)
            output_1 = self.mv2[3](x)
            outputs["stage_2"] = output_1

            x = self.mv2[4](output_1)
            output_2 = self.mvit[0](x)
            outputs["stage_3"] = output_2

            x = self.mv2[5](output_2)
            output_3 = self.mvit[1](x)
            outputs["stage_4"] = output_3

            x = self.mv2[6](output_3)
            x = self.mvit[2](x)
            output_4 = self.conv2(x)
            outputs["stage_5"] = output_4

            return {k: v for k, v in outputs.items() if k in self.out_features}

    def _init_weights(self, m):
        common_init(m)

class Mobilenetv3(nn.Module):
    def __init__(self, type_mode='mobilenetv3_s',num_classes=1000, extract_feature=False, out_features=("stage_1", "stage_2", "stage_3", "stage_4", "stage_5")):
        super().__init__()
        assert type_mode in ['mobilenetv3_s', 'mobilenetv3_l']
        self.type_mode = type_mode
        self.extract_feature = extract_feature
        self.out_features = out_features

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()
        if type_mode == 'mobilenetv3_s':
            print('type_mode : mobilenetv3 small')
            self.bneck = nn.ModuleList([])
            self.bneck.append(MobilenetBlock(3, 16, 16, 16, nn.ReLU(inplace=True), SeModule(16), 2))
            self.bneck.append(MobilenetBlock(3, 16, 72, 24, nn.ReLU(inplace=True), None, 2))
            self.bneck.append(MobilenetBlock(3, 24, 88, 24, nn.ReLU(inplace=True), None, 1))
            self.bneck.append(MobilenetBlock(5, 24, 96, 40, hswish(), SeModule(40), 2))
            self.bneck.append(MobilenetBlock(5, 40, 240, 40, hswish(), SeModule(40), 1))
            self.bneck.append(MobilenetBlock(5, 40, 240, 40, hswish(), SeModule(40), 1))
            self.bneck.append(MobilenetBlock(5, 40, 120, 48, hswish(), SeModule(48), 1))
            self.bneck.append(MobilenetBlock(5, 48, 144, 48, hswish(), SeModule(48), 1))
            self.bneck.append(MobilenetBlock(5, 48, 288, 96, hswish(), SeModule(96), 2))
            self.bneck.append(MobilenetBlock(5, 96, 576, 96, hswish(), SeModule(96), 1))
            self.bneck.append(MobilenetBlock(5, 96, 576, 96, hswish(), SeModule(96), 1))

            self.conv2 = nn.Conv2d(96, 576, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn2 = nn.BatchNorm2d(576)
            self.hs2 = hswish()
            self.linear3 = nn.Linear(576, 1280)

        else:
            print('type_mode : mobilenetv3 large')
            self.bneck = nn.ModuleList([])
            self.bneck.append(MobilenetBlock(3, 16, 16, 16, nn.ReLU(inplace=True), None, 1))
            self.bneck.append(MobilenetBlock(3, 16, 64, 24, nn.ReLU(inplace=True), None, 2))
            self.bneck.append(MobilenetBlock(3, 24, 72, 24, nn.ReLU(inplace=True), None, 1))
            self.bneck.append(MobilenetBlock(5, 24, 72, 40, nn.ReLU(inplace=True), SeModule(40), 2))
            self.bneck.append(MobilenetBlock(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1))
            self.bneck.append(MobilenetBlock(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1))
            self.bneck.append(MobilenetBlock(3, 40, 240, 80, hswish(), None, 2))
            self.bneck.append(MobilenetBlock(3, 80, 200, 80, hswish(), None, 1))
            self.bneck.append(MobilenetBlock(3, 80, 184, 80, hswish(), None, 1))
            self.bneck.append(MobilenetBlock(3, 80, 184, 80, hswish(), None, 1))
            self.bneck.append(MobilenetBlock(3, 80, 480, 112, hswish(), SeModule(112), 1))
            self.bneck.append(MobilenetBlock(3, 112, 672, 112, hswish(), SeModule(112), 1))
            self.bneck.append(MobilenetBlock(5, 112, 672, 160, hswish(), SeModule(160), 1))
            self.bneck.append(MobilenetBlock(5, 160, 672, 160, hswish(), SeModule(160), 2))
            self.bneck.append(MobilenetBlock(5, 160, 960, 160, hswish(), SeModule(160), 1))

            self.conv2 = nn.Conv2d(160, 960, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn2 = nn.BatchNorm2d(960)
            self.hs2 = hswish()
            self.linear3 = nn.Linear(960, 1280)

        self.bn3 = nn.BatchNorm1d(1280)
        self.hs3 = hswish()

        if not self.extract_feature:
            self.linear4 = nn.Linear(1280, num_classes)

        self.apply(self._init_weights)

    def forward(self, x):
        if not self.extract_feature:
            out = self.hs1(self.bn1(self.conv1(x)))

            for layer in self.bneck:
                out = layer(out)

            out = self.hs2(self.bn2(self.conv2(out)))

            out = F.avg_pool2d(out, 7)
            out = out.view(out.size(0), -1)
            out = self.hs3(self.bn3(self.linear3(out)))
            out = self.linear4(out)
            return out
        else:
            outputs = {}
            if self.type_mode == 'mobilenetv3_s':
                output = self.hs1(self.bn1(self.conv1(x)))    
                outputs["stage_1"] = output

                for i, layer in enumerate(self.bneck):              
                    if i == 0:
                        output = layer(output)
                        outputs["stage_2"] = output
                    elif i == 2:
                        output = layer(output)
                        outputs["stage_3"] = output
                    elif i == 7:
                        output = layer(output)
                        outputs["stage_4"] = output
                    else:
                        output = layer(output)
                
                output = self.hs2(self.bn2(self.conv2(output)))
                outputs["stage_5"] = output

            else :
                output = self.hs1(self.bn1(self.conv1(x)))     

                for i, layer in enumerate(self.bneck):     
                    if i == 0:
                        output = layer(output)
                        outputs["stage_1"] = output
                    elif i == 2:
                        output = layer(output)
                        outputs["stage_2"] = output
                    elif i == 5:
                        output = layer(output)
                        outputs["stage_3"] = output
                    elif i == 12:
                        output = layer(output)
                        outputs["stage_4"] = output
                    else:
                        output = layer(output)
                
                output = self.hs2(self.bn2(self.conv2(output)))
                outputs["stage_5"] = output
                    

            return {k: v for k, v in outputs.items() if k in self.out_features}

    def _init_weights(self, m):
        common_init(m)