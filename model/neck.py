'''
# Author: Shaoran Lu
# Date: 2021/10/04
# Email: lushaoran92@gmail.com
# Description: neck net

example:

'''

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

from model.utils.ops import CBA, SeparableConv
from model.utils.csp_utils import CSPLayer
from model.utils.rebise_utils import AttentionRefinementModule

from tools.nninit import  common_init

from model.utils.transformer_utils import TransformerDecoderLayer

class NECKF(nn.Module):
    def __init__(self, type_mode='FPN', feature_number=4, fusion=False, fusion_channel=256, in_channel=[64,128,256,512], act='silu', channel_norm=False):
        super().__init__()
        assert feature_number == len(in_channel)
        self.type_mode = type_mode
        self.feature_number = feature_number
        self.fusion = fusion
        self.fusion_channel = fusion_channel
        self.in_channel = in_channel
        self.channel_norm = channel_norm

        self.channle_conv_1 = CBA(in_channel[-1], in_channel[-2], ksize=1, stride=1, act=act)
        self.channle_conv_2 = CBA(in_channel[-2], in_channel[-3], ksize=1, stride=1, act=act)
        self.channle_conv_3 = CBA(in_channel[-3], in_channel[-4], ksize=1, stride=1, act=act)
        self.C4_C3 = CSPLayer(2*in_channel[-2], in_channel[-2], shortcut=False,  act=act)
        self.C3_C2 = CSPLayer(2*in_channel[-3], in_channel[-3], shortcut=False, act=act)
        self.C2_C1 = CSPLayer(2*in_channel[-4], in_channel[-4], shortcut=False, act=act)

        self.up_sampling = nn.Upsample(scale_factor=2)

        if type_mode == 'PAFPN':
            self.channle_conv_4 = CBA(in_channel[-4], in_channel[-4], ksize=3, stride=2, act=act)
            self.channle_conv_5 = CBA(in_channel[-3], in_channel[-3], ksize=3, stride=2, act=act)
            self.channle_conv_6 = CBA(in_channel[-2], in_channel[-2], ksize=3, stride=2, act=act)

            self.C1_C2 = CSPLayer(in_channel[-3], in_channel[-3], shortcut=False, act=act)
            self.C2_C3 = CSPLayer(in_channel[-2], in_channel[-2], shortcut=False, act=act)
            self.C3_C4 = CSPLayer(in_channel[-1], in_channel[-1], shortcut=False, act=act)

            if self.channel_norm:
                self.channel_norm_conv_1 = CBA(2*in_channel[-4], in_channel[-3], ksize=1, stride=1, act=act)
                self.channel_norm_conv_2 = CBA(2*in_channel[-3], in_channel[-2], ksize=1, stride=1, act=act)
                self.channel_norm_conv_3 = CBA(2*in_channel[-2], in_channel[-1], ksize=1, stride=1, act=act)

        elif type_mode == 'BiFPN':
            pass

        if fusion:
            self.up_conv_1 = CBA(in_channel[-1], fusion_channel//feature_number, ksize=3, stride=1, act=act)
            self.up_conv_2 = CBA(in_channel[-2], fusion_channel//feature_number, ksize=3, stride=1, act=act)
            self.up_conv_3 = CBA(in_channel[-3], fusion_channel//feature_number, ksize=3, stride=1, act=act)
            self.up_conv_4 = CBA(in_channel[-4], fusion_channel//feature_number, ksize=3, stride=1, act=act)

            self.fup_sampling_1 = nn.Upsample(scale_factor=8)
            self.fup_sampling_2 = nn.Upsample(scale_factor=4)
            self.fup_sampling_3 = nn.Upsample(scale_factor=2)

        self.apply(self._init_weights)

    def forward(self, *x):
        assert len(x) == self.feature_number
        x1, x2, x3, x4 = x
        if self.type_mode == 'FPN':
            x4_up = self.up_sampling(self.channle_conv_1(x4))
            x43 = torch.cat([x4_up, x3], dim=1)
            x3 = self.C4_C3(x43)

            x3_up = self.up_sampling(self.channle_conv_2(x3))
            x32 = torch.cat([x3_up, x2], dim=1)
            x2 = self.C3_C2(x32)

            x2_up = self.up_sampling(self.channle_conv_3(x2))
            x21 = torch.cat([x2_up, x1], dim=1)
            x1 = self.C2_C1(x21)

            if self.fusion:
                P4 = self.fup_sampling_1(self.up_conv_1(x4))
                P3 = self.fup_sampling_2(self.up_conv_2(x3))
                P2 = self.fup_sampling_3(self.up_conv_3(x2))
                P1 = self.up_conv_4(x1)
                out = torch.cat((P1,P2,P3,P4), dim=1)
                return out
            else:
                return x1, x2, x3, x4
        elif self.type_mode == 'PAFPN':
            x4 = self.channle_conv_1(x4)

            x4_up = self.up_sampling(x4)
            x43 = torch.cat([x4_up, x3], dim=1)
            x3 = self.C4_C3(x43)
            x3 = self.channle_conv_2(x3)

            x3_up = self.up_sampling(x3)
            x32 = torch.cat([x3_up, x2], dim=1)

            x2 = self.C3_C2(x32)
            x2 = self.channle_conv_3(x2)

            x2_up = self.up_sampling(x2)
            x21 = torch.cat([x2_up, x1], dim=1)
            x1 = self.C2_C1(x21)

            x1_down = self.channle_conv_4(x1)
            x12 = torch.cat([x1_down, x2], dim=1)
            if self.channel_norm:
                x12 = self.channel_norm_conv_1(x12)
            x2 = self.C1_C2(x12)

            x2_down = self.channle_conv_5(x2)
            x23 = torch.cat([x2_down, x3], dim=1)
            if self.channel_norm:
                x23 = self.channel_norm_conv_2(x23)
            x3 = self.C2_C3(x23)

            x3_down = self.channle_conv_6(x3)
            x34 = torch.cat([x3_down, x4], dim=1)
            if self.channel_norm:
                x34 = self.channel_norm_conv_3(x34)
            x4 = self.C3_C4(x34)

            if self.fusion:
                P4 = self.fup_sampling_1(self.up_conv_1(x4))
                P3 = self.fup_sampling_2(self.up_conv_2(x3))
                P2 = self.fup_sampling_3(self.up_conv_3(x2))
                P1 = self.up_conv_4(x1)
                out = torch.cat((P1,P2,P3,P4), dim=1)
                return out
            else:
                return x1, x2, x3, x4

    def _init_weights(self, m):
        common_init(m)

class NECKT(nn.Module):
    def __init__(self, type_mode='FPN', feature_number=3, fusion=False, fusion_channel=768, in_channel=[256, 512, 1024], act='silu',
                            enhance_feature=False, channel_norm=False, block='CSP'):
        super().__init__()
        BLOCK = {'CSP': CSPLayer, 'SeparableConv': SeparableConv}
        assert feature_number == len(in_channel)
        self.type_mode = type_mode
        self.feature_number = feature_number
        self.fusion = fusion
        self.fusion_channel = fusion_channel
        self.in_channel = in_channel

        self.enhance_feature = enhance_feature
        self.channel_norm = channel_norm

        self.channle_conv_1 = CBA(in_channel[-1], in_channel[-2], ksize=1, stride=1, act=act)
        self.channle_conv_2 = CBA(in_channel[-2], in_channel[-3], ksize=1, stride=1, act=act)
        self.C3_C2 = BLOCK[block](2*in_channel[-2], in_channel[-2],  act=act)
        self.C2_C1 = BLOCK[block](2*in_channel[-3], in_channel[-3],  act=act)

        self.up_sampling = nn.Upsample(scale_factor=2)

        if type_mode == 'PAFPN':
            self.channle_conv_3 = CBA(in_channel[-3], in_channel[-3], ksize=3, stride=2, act=act)
            self.channle_conv_4 = CBA(in_channel[-2], in_channel[-2], ksize=3, stride=2, act=act)
            self.C1_C2 = BLOCK[block](in_channel[-2], in_channel[-2], act=act)
            self.C2_C3 = BLOCK[block](in_channel[-1], in_channel[-1], act=act)
            if self.channel_norm:
                self.channel_norm_conv_1 = CBA(2*in_channel[-3], in_channel[-2], ksize=1, stride=1, act=act)
                self.channel_norm_conv_2 = CBA(2*in_channel[-2], in_channel[-1], ksize=1, stride=1, act=act)

            if self.enhance_feature:
                self.arm_1 = AttentionRefinementModule(2*in_channel[-1], in_channel[-1], ksize=1)
                self.arm_2 = AttentionRefinementModule(2*in_channel[-2], in_channel[-2], ksize=1)
                self.arm_3 = AttentionRefinementModule(2*in_channel[-3], in_channel[-3], ksize=1)

        if fusion:
            self.up_conv_1 = CBA(in_channel[-1], fusion_channel//feature_number, ksize=3, stride=1, act=act)
            self.up_conv_2 = CBA(in_channel[-2], fusion_channel//feature_number, ksize=3, stride=1, act=act)
            self.up_conv_3 = CBA(in_channel[-3], fusion_channel//feature_number, ksize=3, stride=1, act=act)

            self.fup_sampling_1 = nn.Upsample(scale_factor=4)
            self.fup_sampling_2 = nn.Upsample(scale_factor=2)

        self.apply(self._init_weights)

    def forward(self, *x):
        assert len(x) == self.feature_number
        x1, x2, x3 = x
        if self.type_mode == 'FPN':
            
            x3_up = self.up_sampling(self.channle_conv_1(x3))
            x32 = torch.cat([x3_up, x2], dim=1)
            x2 = self.C3_C2(x32)

            x2_up = self.up_sampling(self.channle_conv_2(x2))
            x21 = torch.cat([x2_up, x1], dim=1)
            x1 = self.C2_C1(x21)

            if self.fusion:
                P3 = self.fup_sampling_1(self.up_conv_1(x3))
                P2 = self.fup_sampling_2(self.up_conv_2(x2))
                P1 = self.up_conv_3(x1)
                out = torch.cat((P1,P2,P3), dim=1)
                return out
            else:
                return x1, x2, x3
        
        elif self.type_mode == 'PAFPN':
            if self.enhance_feature:
                x1_, x2_, x3_ = x1, x2, x3
            x3 = self.channle_conv_1(x3)
            x3_up = self.up_sampling(x3)
            x32 = torch.cat([x3_up, x2], dim=1)
            x2 = self.C3_C2(x32)
            x2 = self.channle_conv_2(x2)
            x2_up = self.up_sampling(x2)
            x21 = torch.cat([x2_up, x1], dim=1)
            x1 = self.C2_C1(x21)

            x1_down = self.channle_conv_3(x1)
            x12 = torch.cat([x1_down, x2], dim=1)
            if self.channel_norm:
                x12 = self.channel_norm_conv_1(x12)
            x2 = self.C1_C2(x12)

            x2_down = self.channle_conv_4(x2)
            x23 = torch.cat([x2_down, x3], dim=1)
            if self.channel_norm:
                x23 = self.channel_norm_conv_2(x23)
            x3 = self.C2_C3(x23)
            if self.enhance_feature:
                x3 = self.arm_1(torch.cat([x3_, x3], dim=1))
                x2 = self.arm_2(torch.cat([x2_, x2], dim=1))
                x1 = self.arm_3(torch.cat([x1_, x1], dim=1))

            if self.fusion:
                P3 = self.fup_sampling_1(self.up_conv_1(x3))
                P2 = self.fup_sampling_2(self.up_conv_2(x2))
                P1 = self.up_conv_3(x1)
                out = torch.cat((P1,P2,P3), dim=1)
                return out
            else:
                return x1, x2, x3

    def _init_weights(self, m):
        common_init(m)

class SOLOFPN(nn.Module):
    """
    input: [[b,64,1/4,1/4], [b,128,1/8,1/8],[b,256,1/16,1/16],[b,512,1/32,1/32]]

    return: [[b,256,1/4,1/4], [b,256,1/8,1/8],[b,256,1/16,1/16],[b,256,1/32,1/32],[b,256,1/64,1/64]]
    """
    def __init__(self, 
               in_channels,
               out_channels,
               num_outs):
        super(SOLOFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs


        
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.num_ins):
            l_conv = nn.Conv2d(in_channels[i], out_channels, kernel_size=1)
            fpn_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.apply(self._init_weights)

    def forward(self, *inputs):
        assert len(inputs) == len(self.in_channels)
        laterals = [lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)]
        
        for i in range(self.num_ins - 1, 0, -1):
            laterals[i - 1] += F.interpolate(laterals[i], scale_factor=2, mode='nearest')

        outs = [self.fpn_convs[i](laterals[i]) for i in range(self.num_ins)]
        if self.num_outs > len(outs):
            for i in range(self.num_outs - self.num_ins):
                outs.append(F.max_pool2d(outs[-1], 1, stride=2))
        return tuple(outs)

    def _init_weights(self, m):
        common_init(m)

class TransformerEncoder(nn.Module):
    """
    input: 
    x: (b, max_input_len)
    encoder_mask = torch.zeros((x.size(1), x.size(1))).type(torch.bool)      (input_seq_len, input_seq_len)
    encoder_padding_mask: (b, max_input_len)   bool

    return: 
    x: (b, max_input_len, d_model)
    """
    def __init__(self, input_vocab_size=None, d_model=512, num_heads=8, dff=2018, dropout=0.1, activation='relu', num_layers=2, pe=True, maxlen=1000):
        super(TransformerEncoder, self).__init__()
        assert isinstance(num_layers, int)
        self.num_layers = num_layers 
        self.d_model = d_model

    def forward(self, x, encoder_mask, encoder_padding_mask):
        
        return x

class TransformerDecoder(nn.Module):
    """
    input: 
    x: (b, max_input_len)
    encoder_mask = torch.zeros((x.size(1), x.size(1))).type(torch.bool)      (input_seq_len, input_seq_len)
    encoder_padding_mask: (b, max_input_len)   bool

    return: 
    x: (b, max_input_len, d_model)  or (b, max_input_len, traget_voc_size)
    """
    def __init__(self, target_vocab_size=None, d_model=512, num_heads=8, dff=2018, dropout=0.1, num_layers=2, PAD_IDX=1):
        super(TransformerDecoder, self).__init__()
        assert isinstance(num_layers, int)
 
        self.decoder_layers = TransformerDecoderLayer(num_heads, d_model, num_layers, dropout, dff, target_vocab_size, PAD_IDX=PAD_IDX)

        if target_vocab_size is not None:
            self.generator = nn.Linear(d_model, target_vocab_size)
        else:
            self.generator = None
        
        self.apply(self._init_weights)

    def forward(self, x, memory):
        x = self.decoder_layers(x, memory)
        
        if self.generator is not None:
            x = self.generator(x)
        return x

    def _init_weights(self, m):
        common_init(m)

class ContextSpatialFusion(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(ContextSpatialFusion, self).__init__()
        self.convblk = CBA(in_chan, out_chan, ksize=1, stride=1, act="relu", pad=0)
        self.conv1 = nn.Conv2d(out_chan, out_chan//4, kernel_size = 1, stride = 1, padding = 0, bias = False)
        self.conv2 = nn.Conv2d(out_chan//4, out_chan, kernel_size = 1, stride = 1, padding = 0, bias = False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.apply(self._init_weights)

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out

    def _init_weights(self, m):
        common_init(m)

class MaskFormerPixelDecoder(nn.Module):
    def __init__(self, convdims=256, maskdims=256, in_channel= [256,512,1024,2048], norm='GN', act='relu'):
        super().__init__()
        self.convdims = convdims
        self.maskdims = maskdims
        self.norm = norm
        self.in_channel = in_channel
        self.act = act

        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()

        for idx, in_channel in enumerate(self.in_channel):
            if idx == len(self.in_channel) - 1:
                output_conv = CBA(in_channel, convdims, ksize=1, stride=1, pad=1, norm=self.norm,  act=self.act)

                self.lateral_convs.append(None)
                self.output_convs.append(output_conv)
            else:
                lateral_conv = CBA(in_channel, convdims, ksize=1, stride=1, pad=1, norm=self.norm,  act=self.act)
                output_conv = CBA(convdims, convdims, ksize=1, stride=1, pad=1, norm=self.norm,  act=self.act)

                self.lateral_convs.append(lateral_conv)
                self.output_convs.append(output_conv)

        self.lateral_convs = self.lateral_convs[::-1]
        self.output_convs = self.output_convs[::-1]

        self.mask_features =  nn.Conv2d(convdims, maskdims, kernel_size=3,  stride=1, padding=1, bias=False)

        self.apply(self._init_weights)

    def forward(self, *x):
        x1, x2, x3, x4 = x
        for idx, x_ in enumerate([x4, x3, x2, x1]):
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            if lateral_conv is None:
                y = output_conv(x_)
            else:
                cur_fpn = lateral_conv(x_)
                y = cur_fpn + F.interpolate(y, size=cur_fpn.shape[-2:], mode="nearest")
                y = output_conv(y)


        return self.mask_features(y)

    def _init_weights(self, m):
        common_init(m)
