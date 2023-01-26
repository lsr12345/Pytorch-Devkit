'''
# Author: Shaoran Lu
# Date: 2021/10/04
# Email: lushaoran92@gmail.com
# Description: 算法 NET

example:

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import  Dict

import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

from model.utils.ops import CBA

from model.backbone import ResNet, SwinTransformer, Darknet, CSPDarknet, CrnnBackbone, StdcNet, MobileViT, Mobilenetv3
from model.neck import NECKF, NECKT, SOLOFPN, TransformerDecoder, ContextSpatialFusion, MaskFormerPixelDecoder
from model.head import DBHead, YOLOXHead, CrnnHead, SOLOCateKernelHead, SOLOMaskFeatHead, MaskFormerTransHead

from model.utils.transformer_utils import PositionalEncoding2D
from model.utils.rebise_utils import DetailHead, AttentionRefinementModule

class NeuralNetwork(nn.Module):
    def __init__(self, classes):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, classes),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class CRNN(nn.Module):
    def __init__(self, type_mode, num_classes, feature_size=512, hidden_feature = 256, train_model=True):
        super().__init__()
        self.train_model = train_model
        self.feature_size = feature_size
        self.num_classes = num_classes
        self.use_basic = False
        if type_mode in ['dark_s', 'dark_m', 'dark_l']:
            mode_dict = {'dark_s':'small', 'dark_m':'middle', 'dark_l':'large'}
        else:
            type_mode = 'basic'
            mode_dict = {'basic':None}
        print('----------------------------')
        print(type_mode)
        print('----------------------------')
        self.backbone = CrnnBackbone(type_mode=type_mode, mode=mode_dict[type_mode], out_feature_size=self.feature_size)
        
        self.head = CrnnHead(self.num_classes, bi_mode=True,  in_feature=self.feature_size, hidden_feature=hidden_feature, usespp=False)

    def forward(self, x):
        out_features = self.backbone(x)
        b, c, h, w = out_features.size()
        assert h ==1
        o_1 = out_features.squeeze(2)
        outputs = self.head(o_1)

        return outputs

    def load_pretrained_model(self, filename):
        state_dict_names = ["state_dict", "model"]
        match_tensor = 0
        pretrained_state_dict = torch.load(filename,  map_location="cpu")
        for name in state_dict_names:
            if name in pretrained_state_dict:
                pretrained_state_dict = pretrained_state_dict[name]
                break
        self_state_dict = self.backbone.state_dict()
        for k, v in pretrained_state_dict.items():
            if k in self_state_dict:
                self_state_dict.update({k: v})
                match_tensor += 1
        print("Pretrained Model inital tensors: {}/{}".format(match_tensor, len(pretrained_state_dict)))
        self.backbone.load_state_dict(self_state_dict, strict=False)

class DBNet(nn.Module):
    def __init__(self, type_mode, neck_mode, train_model=True, k=50):
        super().__init__()
        self.k = k
        self.train_model = train_model
        channel_norm = False
        if type_mode in ['res_18', 'res_34', 'res_50', 'res_101', 'res_152']:
            mode_dict = {'res_18':'18', 'res_34':'34', 'res_50':'50', 'res_101':'101', 'res_152':'152'}
            self.io_features = ("res2", "res3", "res4", "res5")
            self.backbone = ResNet( type_mode=mode_dict[type_mode], extract_feature=True, out_features=self.io_features)
            if mode_dict[type_mode] in ['18', '34']:
                self.in_channel = [64,128,256,512]
                self.fusion_channel = 256
            else:
                self.in_channel = [128,256,512,1024]
                self.fusion_channel = 512
        elif type_mode in ['sw_t', 'sw_s', 'sw_b', 'sw_l']:
            mode_dict = {'sw_t':'tiny', 'sw_s':'small', 'sw_b':'base', 'sw_l':'large'}
            self.io_features = ("swin2", "swin3", "swin4", "swin5")
            in_channel = {'sw_t':96, 'sw_s':96, 'sw_b':128, 'sw_l':192}
            self.backbone = SwinTransformer(type_mode=mode_dict[type_mode], extract_feature=True, out_features=self.io_features)
            self.in_channel = [in_channel[type_mode], in_channel[type_mode]*2, in_channel[type_mode]*4, in_channel[type_mode]*8]
            if mode_dict[type_mode] in ['tiny', 'small']:
                self.fusion_channel = 256
            else:
                self.fusion_channel = 512
        elif type_mode in ['dark_s', 'dark_m', 'dark_l']:
            mode_dict = {'dark_s':[0.33, 0.50], 'dark_m':[0.67, 0.75], 'dark_l':[1.0, 1.0]}
            self.io_features = ("dark2", "dark3", "dark4", "dark5")
            dep_mul = mode_dict[type_mode][0]
            wid_mul = mode_dict[type_mode][1]
            self.backbone = CSPDarknet(dep_mul=dep_mul, wid_mul=wid_mul,out_features=self.io_features)
            base_channels = int(wid_mul * 64)
            self.in_channel = [base_channels*2, base_channels*4, base_channels*8, base_channels*16]
            if type_mode in ['dark_s', 'dark_m']:
                self.fusion_channel = 256
            else:
                self.fusion_channel = 512
        elif type_mode in ['mobilevit_xxs', 'mobilevit_xs', 'mobilevit_s']:
            self.io_features = ("stage_2", "stage_3", "stage_4", "stage_5")
            self.backbone = MobileViT(type_mode=type_mode, extract_feature=True, out_features=self.io_features)
            in_channels_dict =  {'mobilevit_xxs':[24, 48, 64, 320], 'mobilevit_xs':[48, 64, 80, 384], 'mobilevit_s':[64, 96, 128, 640]}
            self.in_channel = in_channels_dict[type_mode]
            channel_norm = True
            self.fusion_channel = 256
        elif type_mode in ['mobilenetv3_s', 'mobilenetv3_l']:
            self.io_features = ("stage_2", "stage_3", "stage_4", "stage_5")
            self.backbone = Mobilenetv3(type_mode=type_mode, extract_feature=True, out_features=self.io_features)
            in_channels_dict =  {'mobilenetv3_s':[16, 24, 48, 576], 'mobilenetv3_l':[24, 40, 160, 960]}
            self.in_channel = in_channels_dict[type_mode]
            channel_norm = True
            self.fusion_channel = 256

        else:
            raise NotImplementedError('Type {} not supported.'.format(type_mode))

        self.neck = NECKF(type_mode=neck_mode, feature_number=4, fusion=True, fusion_channel=self.fusion_channel, in_channel=self.in_channel, channel_norm=channel_norm)
        self.head = DBHead(in_channel=self.fusion_channel, k=k, act="silu", train_model=self.train_model)

    def forward(self, x):
        out_features = self.backbone(x)
        features = [out_features[f] for f in self.io_features]
        o_1, o_2, o_3, o_4 = features

        fuse = self.neck(o_1, o_2, o_3, o_4)
        
        outputs = self.head(fuse)

        return outputs

    def load_pretrained_model(self, filename):
        state_dict_names = ["state_dict", "model"]
        match_tensor = 0
        pretrained_state_dict = torch.load(filename,  map_location="cpu")
        for name in state_dict_names:
            if name in pretrained_state_dict:
                pretrained_state_dict = pretrained_state_dict[name]
                break
        self_state_dict = self.backbone.state_dict()
        for k, v in pretrained_state_dict.items():
            if k in self_state_dict:
                self_state_dict.update({k: v})
                match_tensor += 1
        print("Pretrained Model inital tensors: {}/{}".format(match_tensor, len(pretrained_state_dict)))
        self.backbone.load_state_dict(self_state_dict, strict=False)

class Yolox(nn.Module):
    def __init__(self, type_mode, neck_mode, train_model=True, num_classes=80, decode_in_inference=False, act='relu'):
        super().__init__()
        self.num_classes = num_classes
        self.train_model = train_model
        channel_norm = False
        if   type_mode in ['nano', 'dark_s', 'dark_m', 'dark_l']:
            mode_dict = {'nano':[0.33, 0.25], 'dark_s':[0.33, 0.50], 'dark_m':[0.67, 0.75], 'dark_l':[1.0, 1.0]}
            self.io_features = ("dark3", "dark4", "dark5")
            dep_mul = mode_dict[type_mode][0]
            wid_mul = mode_dict[type_mode][1]
            self.backbone = CSPDarknet(dep_mul=dep_mul, wid_mul=wid_mul,out_features=self.io_features, act=act)
            base_channels = int(wid_mul * 64)
            self.in_channel = [base_channels*4, base_channels*8, base_channels*16]
        elif type_mode == 'dark':
            self.io_features = ("dark3", "dark4", "dark5")
            self.backbone = Darknet(depth=53)
            base_channels =  32
            self.in_channel = [base_channels*8, base_channels*16, base_channels*16]
        elif type_mode in ['mobilevit_xxs', 'mobilevit_xs', 'mobilevit_s']:
            self.io_features = ("stage_3", "stage_4", "stage_5")
            self.backbone = MobileViT(type_mode=type_mode, extract_feature=True, out_features=self.io_features)
            in_channels_dict =  {'mobilevit_xxs':[48, 64, 320], 'mobilevit_xs':[64, 80, 384], 'mobilevit_s':[96, 128, 640]}
            self.in_channel = in_channels_dict[type_mode]
            channel_norm = True
        elif type_mode in ['mobilenetv3_s', 'mobilenetv3_l']:
            self.io_features = ("stage_3", "stage_4", "stage_5")
            self.backbone = Mobilenetv3(type_mode=type_mode, extract_feature=True, out_features=self.io_features)
            in_channels_dict =   {'mobilenetv3_s':[24, 48, 576], 'mobilenetv3_l':[40, 160, 960]}
            self.in_channel = in_channels_dict[type_mode]
            channel_norm = True

        else:
            raise NotImplementedError('Type {} not supported.'.format(type_mode))
        self.neck = NECKT(type_mode=neck_mode, feature_number=3, fusion=False, in_channel=self.in_channel, act=act,
                                                enhance_feature=False, channel_norm=channel_norm, block='CSP')
        self.head = YOLOXHead(num_classes=self.num_classes, strides=[8, 16, 32],in_channels=self.in_channel,
                                                            train_model=self.train_model, decode_in_inference=decode_in_inference, act=act)

    def forward(self, x):
        out_features = self.backbone(x)
        features = [out_features[f] for f in self.io_features]
        o_1, o_2, o_3 = features

        o_1, o_2, o_3 = self.neck(o_1, o_2, o_3)

        outputs = self.head([o_1, o_2, o_3])
        return outputs

    def load_pretrained_model(self, filename):
        state_dict_names = ["state_dict", "model"]
        match_tensor = 0
        pretrained_state_dict = torch.load(filename,  map_location="cpu")
        for name in state_dict_names:
            if name in pretrained_state_dict:
                pretrained_state_dict = pretrained_state_dict[name]
                break
        self_state_dict = self.backbone.state_dict()
        for k, v in pretrained_state_dict.items():
            if k in self_state_dict:
                self_state_dict.update({k: v})
                match_tensor += 1
        print("Pretrained Model inital tensors: {}/{}".format(match_tensor, len(pretrained_state_dict)))
        self.backbone.load_state_dict(self_state_dict, strict=False)

class SOLO(nn.Module):
    def __init__(self, type_mode, num_classes, train_model=True):
        super().__init__()
        self.train_model = train_model
        if type_mode in ['res_18', 'res_34', 'res_50', 'res_101', 'res_152']:
            mode_dict = {'res_18':'18', 'res_34':'34', 'res_50':'50', 'res_101':'101', 'res_152':'152'}
            self.io_features = ("res2", "res3", "res4", "res5")
            self.backbone = ResNet( type_mode=mode_dict[type_mode], extract_feature=True, out_features=self.io_features)
            if mode_dict[type_mode] in ['18', '34']:
                self.in_channel = [64,128,256,512]
            else:
                self.in_channel = [256,512,1024,2048]
        elif type_mode in ['sw_t', 'sw_s', 'sw_b', 'sw_l']:
            mode_dict = {'sw_t':'tiny', 'sw_s':'small', 'sw_b':'base', 'sw_l':'large'}
            self.io_features = ("swin2", "swin3", "swin4", "swin5")
            in_channel = {'sw_t':96, 'sw_s':96, 'sw_b':128, 'sw_l':192}
            self.backbone = SwinTransformer(type_mode=mode_dict[type_mode], extract_feature=True, out_features=self.io_features)
            self.in_channel = [in_channel[type_mode], in_channel[type_mode]*2, in_channel[type_mode]*4, in_channel[type_mode]*8]

        elif type_mode in ['dark_s', 'dark_m', 'dark_l']:
            mode_dict = {'dark_s':[0.33, 0.50], 'dark_m':[0.67, 0.75], 'dark_l':[1.0, 1.0]}
            self.io_features = ("dark2", "dark3", "dark4", "dark5")
            dep_mul = mode_dict[type_mode][0]
            wid_mul = mode_dict[type_mode][1]
            self.backbone = CSPDarknet(dep_mul=dep_mul, wid_mul=wid_mul,out_features=self.io_features)
            base_channels = int(wid_mul * 64)
            self.in_channel = [base_channels*2, base_channels*4, base_channels*8, base_channels*16]
        else:
            raise NotImplementedError('Type {} not supported.'.format(type_mode))

        self.neck = SOLOFPN(in_channels=self.in_channel, out_channels=256, num_outs=5)

        self.catekernelhead = SOLOCateKernelHead(num_classes,  in_channels=256, seg_feat_channels=256, stacked_convs=2,
                                                                                                    num_grids=[40, 36, 24, 16, 12],   kernel_out_channels=128, train_mode=self.train_model)

        self.maskfeaturehead = SOLOMaskFeatHead(in_channels=256, out_channels=128, start_level=0, end_level=4, feature_size=128)

    def forward(self, x):
        out_features = self.backbone(x)

        features = [out_features[f] for f in self.io_features]
        o_1, o_2, o_3, o_4 = features

        x = self.neck(o_1, o_2, o_3, o_4)

        x_ = x[:4]
        maskfeature_outs = self.maskfeaturehead(x_)

        catekernel_outs = self.catekernelhead(x) 

        return (catekernel_outs, maskfeature_outs)

    def load_pretrained_model(self, filename):
        state_dict_names = ["state_dict", "model"]
        match_tensor = 0
        pretrained_state_dict = torch.load(filename,  map_location="cpu")
        for name in state_dict_names:
            if name in pretrained_state_dict:
                pretrained_state_dict = pretrained_state_dict[name]
                break
        self_state_dict = self.backbone.state_dict()
        for k, v in pretrained_state_dict.items():
            if k in self_state_dict:
                self_state_dict.update({k: v})
                match_tensor += 1
        print("Pretrained Model inital tensors: {}/{}".format(match_tensor, len(pretrained_state_dict)))
        self.backbone.load_state_dict(self_state_dict, strict=False)

class ImageCaptionTransformer(nn.Module):
    def __init__(self, type_mode, neck_mode, target_vocab_size=None, d_model=512, num_heads=4, dff=1024, num_layers=1, pe=True, train_model=True):
        super().__init__()
        self.train_model = train_model
        if type_mode in ['res_18', 'res_34', 'res_50', 'res_101', 'res_152']:
            mode_dict = {'res_18':'18', 'res_34':'34', 'res_50':'50', 'res_101':'101', 'res_152':'152'}
            self.io_features = ("res2", "res3", "res4", "res5")
            self.backbone = ResNet( type_mode=mode_dict[type_mode], extract_feature=True, out_features=self.io_features)
            if mode_dict[type_mode] in ['18', '34']:
                self.in_channel = [64,128,256,512]
            else:
                self.in_channel = [128,256,512,1024]
        elif type_mode in ['sw_t', 'sw_s', 'sw_b', 'sw_l']:
            mode_dict = {'sw_t':'tiny', 'sw_s':'small', 'sw_b':'base', 'sw_l':'large'}
            self.io_features = ("swin2", "swin3", "swin4", "swin5")
            in_channel = {'sw_t':96, 'sw_s':96, 'sw_b':128, 'sw_l':192}
            self.backbone = SwinTransformer(type_mode=mode_dict[type_mode], extract_feature=True, out_features=self.io_features)
            self.in_channel = [in_channel[type_mode], in_channel[type_mode]*2, in_channel[type_mode]*4, in_channel[type_mode]*8]

        elif type_mode in ['dark_s', 'dark_m', 'dark_l']:
            mode_dict = {'dark_s':[0.33, 0.50], 'dark_m':[0.67, 0.75], 'dark_l':[1.0, 1.0]}
            self.io_features = ("dark2", "dark3", "dark4", "dark5")
            dep_mul = mode_dict[type_mode][0]
            wid_mul = mode_dict[type_mode][1]
            self.backbone = CSPDarknet(dep_mul=dep_mul, wid_mul=wid_mul,out_features=self.io_features)
            base_channels = int(wid_mul * 64)
            self.in_channel = [base_channels*2, base_channels*4, base_channels*8, base_channels*16]
        else:
            raise NotImplementedError('Type {} not supported.'.format(type_mode))

        if neck_mode is not None:
            self.neck = NECKF(type_mode=neck_mode, feature_number=4, fusion=False,  in_channel=self.in_channel)
        else:
            self.neck = None

        self.position_embedding = PositionalEncoding2D(self.in_channel[-1], dropout=0.1, max_h=1000, max_w=1000)

        self.head = TransformerDecoder(target_vocab_size=target_vocab_size, d_model=d_model, num_heads=num_heads,
                                                                                dff=dff, dropout=0.1, num_layers=num_layers, PAD_IDX=1)

    def forward(self, inputs):
        if self.train_model:
            x = inputs['batch_images']
            y = inputs['batch_input_labels']
            x = x.cuda()
            y = y.cuda()
        else:
            x = inputs
        out_features = self.backbone(x)

        features = [out_features[f] for f in self.io_features]
        o_1, o_2, o_3, o_4 = features

        if self.neck is  not None:
            _, _, _, o_4 = self.neck(o_1, o_2, o_3, o_4)

        feature = self.position_embedding(o_4)
        b, c, h, w = feature.shape
        feature = feature.view(b, c, h * w)
        feature = feature.permute((0, 2, 1))
        if self.train_model:
            outs = self.head(y, feature)
            return outs.contiguous().view(-1, outs.shape[-1])
        else:
            raise NotImplementedError

class ReBiSeNet(nn.Module):
    def __init__(self, type_mode, num_classes, train_model=True):
        super().__init__()
        self.train_model = train_model

        if type_mode in ['stdc_l', 'stdc_s']:
            mode_dict = {'stdc_l':'l', 'stdc_s':'s'}
            self.io_features = ("stage_3", "stage_4", "stage_5")
            self.backbone = StdcNet( type_mode=mode_dict[type_mode], extract_feature=True, out_features=self.io_features)
            self.in_channel = [256, 512, 1024]

        elif type_mode in ['sw_t', 'sw_s', 'sw_b', 'sw_l']:
            mode_dict = {'sw_t':'tiny', 'sw_s':'small', 'sw_b':'base', 'sw_l':'large'}
            self.io_features = ("swin3", "swin4", "swin5")
            in_channel = {'sw_t':96, 'sw_s':96, 'sw_b':128, 'sw_l':192}
            self.backbone = SwinTransformer(type_mode=mode_dict[type_mode], extract_feature=True, out_features=self.io_features)
            self.in_channel = [in_channel[type_mode]*2, in_channel[type_mode]*4, in_channel[type_mode]*8]

        else:
            raise NotImplementedError('Type {} not supported.'.format(type_mode))

        if self.train_model:
            self.detailhead = DetailHead(self.in_channel[0],  64, 1)
            self.conv_out16 = DetailHead(128, 64, num_classes)
            self.conv_out32 = DetailHead(128, 64, num_classes)
        
        self.conv_out = DetailHead(256, 256, num_classes)

        self.arm16 = AttentionRefinementModule(self.in_channel[1], 128)
        self.arm32 = AttentionRefinementModule(self.in_channel[2], 128)

        self.conv_head32 = CBA(128, 128, ksize=3, stride=1, act="relu", pad=1)
        self.conv_head16 = CBA(128, 128, ksize=3, stride=1, act="relu", pad=1)
        self.conv_avg = CBA(1024, 128, ksize=1, stride=1, act="relu", pad=0)
        
        inplane = self.in_channel[0] + 128
        self.ffm = ContextSpatialFusion(inplane, 256)

    def forward(self, x):
        H, W = x.size()[2:]
        out_features = self.backbone(x)

        features = [out_features[f] for f in self.io_features]
        feat8, feat16, feat32 = features

        H8, W8 = feat8.size()[2:]
        H16, W16 = feat16.size()[2:]
        H32, W32 = feat32.size()[2:]
        
        avg = F.avg_pool2d(feat32, feat32.size()[2:])

        avg = self.conv_avg(avg)
        avg_up = F.interpolate(avg, (H32, W32), mode='nearest')

        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg_up
        feat32_up = F.interpolate(feat32_sum, (H16, W16), mode='nearest')
        feat32_up = self.conv_head32(feat32_up)

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = F.interpolate(feat16_sum, (H8, W8), mode='nearest')
        feat16_up = self.conv_head16(feat16_up)

        feat_fuse = self.ffm(feat8, feat16_up)

        feat_out = self.conv_out(feat_fuse)
        feat_out = F.interpolate(feat_out, (H, W), mode='bilinear', align_corners=True)

        if self.train_model:
            featdetail_sp8 = self.detailhead(feat8)
            feat_out8 = self.conv_out16(feat16_up)
            feat_out16 = self.conv_out32(feat32_up)
            feat_out8 = F.interpolate(feat_out8, (H, W), mode='bilinear', align_corners=True)
            feat_out16 = F.interpolate(feat_out16, (H, W), mode='bilinear', align_corners=True)

            return feat_out, feat_out8, feat_out16, featdetail_sp8

        else:
            return feat_out

    def load_pretrained_model(self, filename):
        state_dict_names = ["state_dict", "model"]
        match_tensor = 0
        pretrained_state_dict = torch.load(filename,  map_location="cpu")
        for name in state_dict_names:
            if name in pretrained_state_dict:
                pretrained_state_dict = pretrained_state_dict[name]
                break
        self_state_dict = self.backbone.state_dict()
        for k, v in pretrained_state_dict.items():
            if k in self_state_dict:
                self_state_dict.update({k: v})
                match_tensor += 1
        print("Pretrained Model inital tensors: {}/{}".format(match_tensor, len(pretrained_state_dict)))
        self.backbone.load_state_dict(self_state_dict, strict=False)

class MaskFormer(nn.Module):
    def __init__(self, type_mode, num_classes, train_model=True):
        super().__init__()
        self.train_model = train_model
        if type_mode in ['res_18', 'res_34', 'res_50', 'res_101', 'res_152']:
            mode_dict = {'res_18':'18', 'res_34':'34', 'res_50':'50', 'res_101':'101', 'res_152':'152'}
            self.io_features = ("res2", "res3", "res4", "res5")
            self.backbone = ResNet( type_mode=mode_dict[type_mode], extract_feature=True, out_features=self.io_features)
            if mode_dict[type_mode] in ['18', '34']:
                self.in_channel = [64,128,256,512]
            else:
                self.in_channel =  [256,512,1024,2048]
        elif type_mode in ['sw_t', 'sw_s', 'sw_b', 'sw_l']:
            mode_dict = {'sw_t':'tiny', 'sw_s':'small', 'sw_b':'base', 'sw_l':'large'}
            self.io_features = ("swin2", "swin3", "swin4", "swin5")
            in_channel = {'sw_t':96, 'sw_s':96, 'sw_b':128, 'sw_l':192}
            self.backbone = SwinTransformer(type_mode=mode_dict[type_mode], extract_feature=True, out_features=self.io_features)
            self.in_channel = [in_channel[type_mode], in_channel[type_mode]*2, in_channel[type_mode]*4, in_channel[type_mode]*8]
        elif type_mode in ['stdc_s', 'stdc_l']:
            mode_dict = {'stdc_s':'s', 'stdc_l':'l'}
            self.io_features = ("stage_2", "stage_3", "stage_4", "stage_5")
            self.backbone = StdcNet( type_mode=mode_dict[type_mode], extract_feature=True, out_features=self.io_features)
            self.in_channel = [64,256,512,1024]
        elif type_mode in ['mobilevit_xxs', 'mobilevit_xs', 'mobilevit_s']:
            self.io_features = ("stage_2", "stage_3", "stage_4", "stage_5")
            self.backbone = MobileViT(type_mode=type_mode, extract_feature=True, out_features=self.io_features)
            in_channels_dict =  {'mobilevit_xxs':[24, 48, 64, 320], 'mobilevit_xs':[48, 64, 80, 384], 'mobilevit_s':[64, 96, 128, 640]}
            self.in_channel = in_channels_dict[type_mode]
        elif type_mode in ['mobilenetv3_s', 'mobilenetv3_l']:
            self.io_features = ("stage_2", "stage_3", "stage_4", "stage_5")
            self.backbone = Mobilenetv3(type_mode=type_mode, extract_feature=True, out_features=self.io_features)
            in_channels_dict =  {'mobilenetv3_s':[16, 24, 48, 576], 'mobilenetv3_l':[24, 40, 160, 960]}
            self.in_channel = in_channels_dict[type_mode]

        else:
            raise NotImplementedError('Type {} not supported.'.format(type_mode))

        self.neck = MaskFormerPixelDecoder(convdims=256, maskdims=256, in_channel=self.in_channel, norm='GN', act='relu')
        self.head = MaskFormerTransHead(in_channels = self.in_channel[-1], mask_classification =True,  num_classes = num_classes,  hidden_dim = 256,  
                                                                                    num_queries = 100, nheads = 8, dropout = 0.1,  dim_feedforward = 2048, enc_layers = 0,  dec_layers = 6, 
                                                                                    pre_norm = False, deep_supervision = True,  mask_dim = 256, train_model = train_model)

    def forward(self, x):
        out_features = self.backbone(x)

        features = [out_features[f] for f in self.io_features]
        o_1, o_2, o_3, o_4 = features
        
        mask_features = self.neck(o_1, o_2, o_3, o_4)
        
        outputs = self.head(o_4, mask_features)

        return outputs

    def load_pretrained_model(self, filename):
        state_dict_names = ["state_dict", "model"]
        match_tensor = 0
        pretrained_state_dict = torch.load(filename,  map_location="cpu")
        for name in state_dict_names:
            if name in pretrained_state_dict:
                pretrained_state_dict = pretrained_state_dict[name]
                break
        self_state_dict = self.backbone.state_dict()
        for k, v in pretrained_state_dict.items():
            if k in self_state_dict:
                self_state_dict.update({k: v})
                match_tensor += 1
        print("Pretrained Model inital tensors: {}/{}".format(match_tensor, len(pretrained_state_dict)))
        self.backbone.load_state_dict(self_state_dict, strict=False)
