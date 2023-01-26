'''
# Author: Shaoran Lu
# Date: 2021/10/04
# Email: lushaoran92@gmail.com
# Description: head net

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

from utils.common import multi_apply
from tools.nms import points_nms
from tools.nninit import common_init

from model.utils.ops import CBA, MLP, get_activation, BidirectionalLSTM, FFN
from model.utils.csp_utils import SPPBottleneck_1D
from model.utils.maskformer_utils import MaskTransformer, PositionEmbeddingSine

class CrnnHead(nn.Module):
    def __init__(self, num_classes, bi_mode=True,  in_feature=512, hidden_feature=256, num_layer=2, usespp=False ):
        super().__init__()
        self.bi_mode = bi_mode
        self.in_feature = in_feature
        self.hidden_feature = hidden_feature
        self.num_classes = num_classes
        self.usespp = usespp
        
        seq = []
        for i in range(num_layer):
            if i == num_layer-1:
                if not usespp:
                    seq.append(BidirectionalLSTM(self.in_feature, self.hidden_feature, self.num_classes))
                else:
                    seq.append(SPPBottleneck_1D(self.in_feature,  self.num_classes))
            else:
                if not usespp:
                    seq.append(BidirectionalLSTM(self.in_feature, self.hidden_feature, self.hidden_feature))
                    self.in_feature = self.hidden_feature
                else:
                    seq.append(SPPBottleneck_1D(self.in_feature,  self.in_feature))
            
        self.seq = nn.Sequential(*seq)


    def forward(self, x):
        if not self.usespp:
            x = x.permute(2, 0, 1)
        x = self.seq(x)
        if self.usespp:
            x = x.permute(2, 0, 1)
        output = F.log_softmax(x, dim=2)
        return output

class DBHead(nn.Module):
    def __init__(
        self,
        in_channel=256,
        k=50,
        act="silu",
        train_model=True
        ):
        super().__init__()
        self.train_model = train_model
        self.in_channel = in_channel
        self.k = k
        self.act = act
        self.probability = nn.Sequential(
            CBA(self.in_channel, 64, 3, 1,  bias=False, act=self.act, use_bn=True),
            nn.ConvTranspose2d(64, 64,  kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            get_activation(name=self.act, inplace=True),
            nn.ConvTranspose2d(64, 1,  kernel_size=2, stride=2),
            nn.Sigmoid()
            )

        if self.train_model:
            self.threshold = nn.Sequential(
                CBA(self.in_channel, 64, 3, 1,  bias=False, act=self.act, use_bn=True),
                nn.ConvTranspose2d(64, 64,  kernel_size=2, stride=2),
                nn.BatchNorm2d(64),
                get_activation(name=self.act, inplace=True),
                nn.ConvTranspose2d(64, 1,  kernel_size=2, stride=2),
                nn.Sigmoid()
            )
        
        self.apply(self._init_weights)

    def forward(self, fuse):
        p = self.probability(fuse)
        if not self.train_model:
            return p
        else:
            t = self.threshold(fuse)
            b_hat = torch.reciprocal(1 + torch.exp(-self.k * (p - t)))
            out = torch.cat((p, b_hat, t), dim=1)
        return out

    def _init_weights(self, m):
        common_init(m)

class YOLOXHead(nn.Module):
    def __init__(
        self,
        num_classes,
        strides=[8, 16, 32],
        in_channels=[256, 512, 1024],
        act="silu",
        train_model=True,
        decode_in_inference=True
        ):
        super().__init__()
        self.n_anchors = 1
        self.num_classes = num_classes
        self.decode_in_inference = decode_in_inference
        self.train_model = train_model


        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()

        for i in range(len(in_channels)):
            self.stems.append(
                CBA(
                    in_channels=int(in_channels[i]),
                    out_channels=int(256),
                    ksize=1,
                    stride=1,
                    act=act,
                )
            )
            self.cls_convs.append(
                nn.Sequential(
                    *[
                        CBA(
                            in_channels=int(256),
                            out_channels=int(256),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        CBA(
                            in_channels=int(256),
                            out_channels=int(256),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    *[
                        CBA(
                            in_channels=int(256),
                            out_channels=int(256),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        CBA(
                            in_channels=int(256),
                            out_channels=int(256),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.cls_preds.append(
                nn.Conv2d(
                    in_channels=int(256),
                    out_channels=self.n_anchors * self.num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=int(256),
                    out_channels=4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.obj_preds.append(
                nn.Conv2d(
                    in_channels=int(256),
                    out_channels=self.n_anchors * 1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )

        self.strides = strides
        self.grids = [torch.zeros(1)] * len(in_channels)

        self._init_weights()

    def forward(self, xin):
        outputs = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(zip(self.cls_convs, self.reg_convs, self.strides, xin)):
            x = self.stems[k](x)
            cls_x = x
            reg_x = x

            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat)

            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            if self.train_model and self.training:
                output = torch.cat([reg_output, obj_output, cls_output], 1)
                output, grid = self.get_output_and_grid(
                    output, k, stride_this_level, xin[0].type()
                )
                x_shifts.append(grid[:, :, 0])
                y_shifts.append(grid[:, :, 1])
                expanded_strides.append(
                    torch.zeros(1, grid.shape[1]).fill_(stride_this_level).type_as(xin[0])
                )
            else:
                output = torch.cat([reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1)

            outputs.append(output)

        if self.train_model and self.training:
            return [torch.cat(outputs, 1), x_shifts, y_shifts, expanded_strides, xin[0].dtype]
        else:
            self.hw = [list(map(int, x.shape[-2:])) for x in outputs]


            proc_view = lambda x: x.view(-1, int(x.size(1)), int(x.size(2) * x.size(3)))
            outputs = torch.cat(
                [proc_view(x) for x in outputs], dim=2
            ).permute(0, 2, 1)

            if not self.train_model and not  self.training:
                return self.decode_outputs(outputs, dtype=xin[0].type())
            else:
                return outputs  

    def decode_outputs(self, outputs, dtype):
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))

        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)

        xy = (outputs[..., :2] + grids) * strides
        wh = torch.exp(outputs[..., 2:4]) * strides
        return torch.cat((xy, wh, outputs[..., 4:]), dim=-1)


    '''
    channel放到最后一维并为reg结果进行grid和stride还原
    return:
                 output.shape=(batchsize, total_anchors, n_ch)
    '''
    def get_output_and_grid(self, output, k, stride, dtype):
        grid = self.grids[k]

        batch_size = output.shape[0]
        n_ch = 5 + self.num_classes
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype)
            self.grids[k] = grid

        output = output.view(batch_size, self.n_anchors, n_ch, hsize, wsize)
        output = output.permute(0, 1, 3, 4, 2).reshape(
            batch_size, self.n_anchors * hsize * wsize, -1
        )
        grid = grid.view(1, -1, 2)
        output[..., :2] = (output[..., :2] + grid) * stride
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
        return output, grid

    def _init_weights(self):
        prior_prob=0.01
        for conv in self.cls_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.obj_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

class SOLOCateKernelHead(nn.Module):
    """
    input: [[b,256,1/4,1/4], [b,256,1/8,1/8],[b,256,1/16,1/16],[b,256,1/32,1/32],[b,256,1/64,1/64]]

    return:  [cate_pred, kernel_pred]
    
    cate_pred: [[b,num_classes,40,40],[b,num_classes,36,36],[b,num_classes,24,24],[b,num_classes,16,16],[b,num_classes,12,12]] 
    kernel_pred: [[b,128,40,40],[b,128,36,36],[b,128,24,24],[b,128,16,16],[b,128,12,12]] 
    """
    def __init__(self,
                 num_classes,
                 in_channels,
                 seg_feat_channels=256,
                 stacked_convs=4,
                 num_grids=[40, 36, 24, 16, 12],  
                 kernel_out_channels=128,
                 train_mode=True
                 ):
        super(SOLOCateKernelHead, self).__init__()
        self.num_classes = num_classes
        self.seg_num_grids = num_grids
        self.in_channels = in_channels
        self.seg_feat_channels = seg_feat_channels
        self.stacked_convs = stacked_convs
        self.kernel_out_channels = kernel_out_channels

        self.train_mode = train_mode

        self.cate_convs = nn.ModuleList()
        self.kernel_convs = nn.ModuleList()
        for i in range(self.stacked_convs):          
            chn = self.in_channels + 2 if i == 0 else self.seg_feat_channels
            self.kernel_convs.append(nn.Sequential(
                    nn.Conv2d(chn, self.seg_feat_channels, 3, stride=1, padding=1, bias=False),
                    nn.GroupNorm(num_channels=self.seg_feat_channels, num_groups=32),
                    nn.ReLU(inplace=True)
                    ))

            chn = self.in_channels if i == 0 else self.seg_feat_channels
            self.cate_convs.append(nn.Sequential(
                    nn.Conv2d(chn, self.seg_feat_channels, 3, stride=1, padding=1, bias=False),
                    nn.GroupNorm(num_channels=self.seg_feat_channels, num_groups=32),
                    nn.ReLU(inplace=True)
                    ))

        self.solo_cate = nn.Conv2d(
            self.seg_feat_channels, self.num_classes, 3, padding=1)

        self.solo_kernel = nn.Conv2d(
            self.seg_feat_channels, self.kernel_out_channels, 3, padding=1)

        self.apply(self._init_weights)

    def forward(self, feats):
        new_feats = self.split_feats(feats)
        cate_pred, kernel_pred = multi_apply(self.forward_single, new_feats,
                                                                                        list(range(len(self.seg_num_grids))),
                                                                                        train=self.train_mode
                                                                                        )
        return cate_pred, kernel_pred
    
    def split_feats(self, feats):
        return (feats[0], feats[1], feats[2], feats[3], feats[4])

    def forward_single(self, x, idx, train=False):
        ins_kernel_feat = x
        x_range = torch.arange(-1, 1, 2/ins_kernel_feat.shape[-1], device=ins_kernel_feat.device)
        y_range = torch.arange(-1, 1, 2/ins_kernel_feat.shape[-2], device=ins_kernel_feat.device)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([ins_kernel_feat.shape[0], 1, -1, -1])
        x = x.expand([ins_kernel_feat.shape[0], 1, -1, -1])
        
        coord_feat = torch.cat([x, y], 1)
        ins_kernel_feat = torch.cat([ins_kernel_feat, coord_feat], 1)
        
        kernel_feat = ins_kernel_feat
        seg_num_grid = self.seg_num_grids[idx]
        kernel_feat = F.interpolate(kernel_feat, size=seg_num_grid, mode='bilinear',align_corners=False)

        cate_feat = kernel_feat[:, :-2, :, :]

        kernel_feat = kernel_feat.contiguous()
        for i, kernel_layer in enumerate(self.kernel_convs):
            kernel_feat = kernel_layer(kernel_feat)
        kernel_pred = self.solo_kernel(kernel_feat)

        cate_feat = cate_feat.contiguous()
        for i, cate_layer in enumerate(self.cate_convs):
            cate_feat = cate_layer(cate_feat)
        cate_pred = self.solo_cate(cate_feat)

        if not train:
            cate_pred = points_nms(cate_pred.sigmoid(), kernel=2).permute(0, 2, 3, 1)
        return cate_pred, kernel_pred

    def _init_weights(self, m):
        common_init(m)

class SOLOMaskFeatHead(nn.Module):
    """
    input: [[b,256,1/4,1/4], [b,256,1/8,1/8],[b,256,1/16,1/16],[b,256,1/32,1/32]]

    return: [b, 128,/4, /4]

    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 start_level,
                 end_level,
                 feature_size
                ):
        super(SOLOMaskFeatHead, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.start_level = start_level
        self.end_level = end_level
        assert start_level >= 0 and end_level >= start_level
        self.feature_size = feature_size
        self.convs_all_levels = nn.ModuleList()
        for i in range(self.start_level, self.end_level):
            convs_per_level = nn.Sequential()
            if i == 0:
                one_conv = nn.Sequential(
                                        nn.Conv2d(self.in_channels, self.out_channels, 3, padding=1, bias=False),
                                        nn.GroupNorm(num_channels=self.out_channels, num_groups=32),
                                        nn.ReLU(inplace=True)
                                        )
                convs_per_level.add_module('conv' + str(i), one_conv)
                self.convs_all_levels.append(convs_per_level)
                continue

            for j in range(i):
                if j == 0:
                    chn = self.in_channels+2 if i==3 else self.in_channels
                    one_conv = nn.Sequential(
                                            nn.Conv2d(chn, self.out_channels, 3, padding=1, bias=False),
                                            nn.GroupNorm(num_channels=self.out_channels, num_groups=32),
                                            nn.ReLU(inplace=True)
                                            )
                    convs_per_level.add_module('conv' + str(j), one_conv)
                    one_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
                    convs_per_level.add_module('upsample' + str(j), one_upsample)
                    continue

                one_conv = nn.Sequential(
                                        nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1, bias=False),
                                        nn.GroupNorm(num_channels=self.out_channels, num_groups=32),
                                        nn.ReLU(inplace=True)
                                        )
                convs_per_level.add_module('conv' + str(j), one_conv)
                one_upsample = nn.Upsample(scale_factor=2,  mode='bilinear', align_corners=False)
                convs_per_level.add_module('upsample' + str(j), one_upsample)

            self.convs_all_levels.append(convs_per_level)

        self.conv_pred = nn.Sequential(
                                        nn.Conv2d(self.out_channels, self.feature_size,1,padding=0,bias = False),
                                        nn.GroupNorm(num_channels=self.out_channels, num_groups=32),
                                        nn.ReLU(inplace=True)
                                        )

        self.apply(self._init_weights)

    def forward(self, inputs):
        assert len(inputs) == (self.end_level - self.start_level)
        input_p = inputs[0]
        feature_add_all_level = self.convs_all_levels[0](input_p)
        for i in range(1, len(inputs)):
            input_p = inputs[i]
            if i == 3:
                input_feat = input_p
                x_range = torch.arange(-1, 1, 2/input_feat.shape[-1], device=input_feat.device)
                y_range = torch.arange(-1, 1, 2/input_feat.shape[-2], device=input_feat.device)
                y, x = torch.meshgrid(y_range, x_range)
                y = y.expand([input_feat.shape[0], 1, -1, -1])
                x = x.expand([input_feat.shape[0], 1, -1, -1])
                coord_feat = torch.cat([x, y], 1)
                input_p = torch.cat([input_p, coord_feat], 1)
                
            feature_add_all_level = feature_add_all_level + self.convs_all_levels[i](input_p)

        feature_pred = self.conv_pred(feature_add_all_level)
        return feature_pred

    def _init_weights(self, m):
        common_init(m)

class MaskFormerTransHead(nn.Module): 
    def __init__(self, in_channels=2048, mask_classification=True,  num_classes=171,  hidden_dim=256,  num_queries=100,  
                                nheads=8, dropout=0.1,  dim_feedforward=2048, enc_layers=0,  dec_layers=6,  pre_norm=False,   
                                deep_supervision=True,  mask_dim=256, train_model=True):  
        super().__init__()
        self.mask_classification = mask_classification

        self.train_model = train_model
        if not self.training:
            deep_supervision = False
        if not self.train_model:
            deep_supervision = False

        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        transformer = MaskTransformer(
            d_model=hidden_dim,
            dropout=dropout,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            normalize_before=pre_norm,
            return_intermediate_dec=deep_supervision,
        )

        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model

        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        if in_channels != hidden_dim:
            self.input_proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        else:
            self.input_proj = nn.Sequential()

        self.aux_loss = deep_supervision

        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)

        self.mask_embed = FFN(hidden_dim, hidden_dim, mask_dim, 3)

        self.apply(self._init_weights)

    def forward(self, x, mask_features):
        pos = self.pe_layer(x)

        src = x
        mask = None
        hs, memory = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos)

        if self.mask_classification:
            outputs_class = self.class_embed(hs)
            out = {"pred_logits": outputs_class[-1]}
        else:
            out = {}

        if self.aux_loss:
            mask_embed = self.mask_embed(hs)
            outputs_seg_masks = torch.einsum("lbqc,bchw->lbqhw", mask_embed, mask_features)
            out["pred_masks"] = outputs_seg_masks[-1]
            out["aux_outputs"] = self._set_aux_loss(
                outputs_class if self.mask_classification else None, outputs_seg_masks
            )
        else:
            mask_embed = self.mask_embed(hs[-1])
            outputs_seg_masks = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
            out["pred_masks"] = outputs_seg_masks
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]

    def _init_weights(self, m):
        common_init(m)
