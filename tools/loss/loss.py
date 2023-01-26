'''
# Author: Shaoran Lu
# Date: 2021/10/04
# Email: lushaoran92@gmail.com
# Description: loss

example:

'''

from matplotlib.pyplot import axes
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from typing import  Dict

import numpy as np
from scipy import ndimage
import cv2
import math

from loguru import logger

from ..boxes import bboxes_iou
from utils.common import multi_apply
from .loss_utils import sigmoid_focal_loss


class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        iou = (area_i) / (area_p + area_g - area_i + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "giou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_i) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss

class Diceloss(nn.Module):
    def __init__(self,  reduction="none", epsilon = 1e-3):
        super(Diceloss, self).__init__()
        self.reduction = reduction
        self.epsilon = epsilon

    def forward(self, pred, target, mask=1.):
        assert pred.shape[0] == target.shape[0]
        pred = pred.contiguous().view(pred.size()[0], -1)
        target = target.contiguous().view(target.size()[0], -1).type_as(pred)
        intersection = torch.sum(pred * target*mask, 1)
        union = torch.sum(pred * pred*mask, 1) + torch.sum(target * target*mask, 1) + 2*self.epsilon
        d = (2 * intersection) / union
        loss = 1 - d
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss

class FocalLoss_cuda(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0):
        super(FocalLoss_cuda, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            loss_cls = self.loss_weight * sigmoid_focal_loss(
                pred,
                target,
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor)
        else:
            raise NotImplementedError
        return loss_cls

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes = 3, size_average=True):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """
        super(FocalLoss,self).__init__()
        self.size_average = size_average
        if isinstance(alpha,list):
            assert len(alpha)==num_classes
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha<1
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha)

        self.gamma = gamma

    def forward(self, preds, labels):
        """
        focal_loss损失计算
        :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B,N] or [B]
        :return:
        """
        preds = preds.view(-1,preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_logsoft = F.log_softmax(preds, dim=1)
        preds_softmax = torch.exp(preds_logsoft)

        preds_softmax = preds_softmax.gather(1,labels.view(-1,1))
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))
        self.alpha = self.alpha.gather(0,labels.view(-1))
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)

        loss = torch.mul(self.alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

class OhemCELoss(nn.Module):
    def __init__(self, thresh, n_min, ignore_lb=255, reduction='mean', weight=None):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.reduction = reduction
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none', weight=weight)

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        loss = self.criteria(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.n_min] > self.thresh:
            loss = loss[loss>self.thresh]
        else:
            loss = loss[:self.n_min]
        
        if self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        return loss

class CrnnLoss(nn.Module):
    def __init__(self,  reduction="mean"):
        super(CrnnLoss, self).__init__()
        self.loss_func = torch.nn.CTCLoss(reduction=reduction)

    def forward(self, pred, target,):
        assert pred.shape[1] == target.shape[0]
        Input_lengths = target[:, 0]
        Target_lengths = target[:, 1]
        Targets = target[:, 2:]
        loss = self.loss_func(pred, Targets, Input_lengths, Target_lengths)
        return loss

class DBLoss(nn.Module):

    def __init__(self, negative_ratio=3., balance_scale=5., epsilon = 1e-9, l1_scale=10.):
        super().__init__()
        self.negative_ratio = negative_ratio
        self.balance_scale = balance_scale
        self.epsilon = epsilon
        self.l1_scale = l1_scale
        self.dice_loss = Diceloss(reduction="mean")

    def forward(self, y_pr, y_gt):
        binary = y_pr[:,0,:,:]
        thresh_binary =y_pr[:,1,:,:]
        thresh = y_pr[:,2,:,:]

        gt = y_gt[:,0,:,:]
        mask = y_gt[:,1,:,:]
        thresh_map = y_gt[:,2,:,:]
        thresh_mask = y_gt[:,3,:,:]

        l1_loss = self.l1_loss([thresh, thresh_map, thresh_mask])
        balanced_ce_loss, dice_loss_weights = self.balanced_crossentropy_loss([binary, gt, mask])
        
        dice_mask = mask * ((dice_loss_weights - torch.min(dice_loss_weights)) / (torch.max(dice_loss_weights) - torch.min(dice_loss_weights)) + 1.)
        dice_mask = dice_mask.contiguous().view(dice_mask.size()[0], -1)
        dice_loss = self.dice_loss(thresh_binary, gt, mask=dice_mask)
        
        return l1_loss + balanced_ce_loss + dice_loss

    def l1_loss(self, args):
        pred, gt, mask = args
        mask_sum = torch.sum(mask)
        if mask_sum.item() > 0:
            loss = torch.sum(torch.abs(pred - gt) * mask) / mask_sum
        else:
            loss =  torch.tensor(0, dtype=torch.float32)
        loss = loss * self.l1_scale
        return loss

    def balanced_crossentropy_loss(self, args):
        pred, gt, mask = args
        positive_mask = (gt * mask)
        negative_mask = ((1 - gt) * mask)
        positive_count = torch.sum(positive_mask)
        negative_count = torch.min(torch.tensor([torch.sum(negative_mask), positive_count * self.negative_ratio]))
        loss = F.binary_cross_entropy(pred, gt, reduction='none')
        positive_loss = loss * positive_mask
        negative_loss = loss * negative_mask
        negative_loss, _ = torch.topk(negative_loss.view(-1), negative_count.int())
        balanced_loss = (torch.sum(positive_loss) + torch.sum(negative_loss)) / ( positive_count + negative_count + self.epsilon)
        balanced_loss = balanced_loss * self.balance_scale
        return balanced_loss, loss

class YoloxLoss(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")
        self.num_classes = num_classes

    def forward(self, y_pr, y_gt):
        outputs, x_shifts, y_shifts, expanded_strides, dtype = y_pr
        labels = y_gt

        bbox_preds = outputs[:, :, :4]
        obj_preds = outputs[:, :, 4].unsqueeze(-1)
        cls_preds = outputs[:, :, 5:]

        mixup = labels.shape[2] > 5
        if mixup:
            label_cut = labels[..., :5]
        else:
            label_cut = labels
        nlabel = (label_cut.sum(dim=2) > 0).sum(dim=1)

        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)
        y_shifts = torch.cat(y_shifts, 1)
        expanded_strides = torch.cat(expanded_strides, 1)

        cls_targets = []
        reg_targets = []
        obj_targets = []
        fg_masks = []

        num_fg = 0.0
        num_gts = 0.0

        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                gt_bboxes_per_image = labels[batch_idx, :num_gt, :4]
                gt_classes = labels[batch_idx, :num_gt, 4]
                bboxes_preds_per_image = bbox_preds[batch_idx]
                
                try:
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels
                    )
                except RuntimeError:
                    logger.error(
                        "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size."
                    )
                    torch.cuda.empty_cache()
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        "cpu"
                    )

                torch.cuda.empty_cache()
                num_fg += num_fg_img

                cls_target = F.one_hot(
                    gt_matched_classes.to(torch.int64), self.num_classes
                ) * pred_ious_this_matching.unsqueeze(-1)
                obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_bboxes_per_image[matched_gt_inds]

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.to(dtype))
            fg_masks.append(fg_mask)


        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)

        num_fg = max(num_fg, 1)
        loss_iou = (
            self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)
        ).sum() / num_fg
        loss_obj = (
            self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)
        ).sum() / num_fg
        loss_cls = (
            self.bcewithlog_loss(
                cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets
            )
        ).sum() / num_fg

        loss_l1 = 0.0

        reg_weight = 5.0
        loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1

        return loss

    @torch.no_grad()
    def get_assignments(
        self,
        batch_idx,
        num_gt,
        total_num_anchors,
        gt_bboxes_per_image,
        gt_classes,
        bboxes_preds_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        cls_preds,
        bbox_preds,
        obj_preds,
        labels,
        mode="gpu"):

        if mode == "cpu":
            print("------------CPU Mode for This Batch-------------")
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            gt_classes = gt_classes.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()

        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            total_num_anchors,
            num_gt,
        )

        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        cls_preds_ = cls_preds[batch_idx][fg_mask]
        obj_preds_ = obj_preds[batch_idx][fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()

        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)

        gt_cls_per_image = (
            F.one_hot(gt_classes.to(torch.int64), self.num_classes)
            .float()
            .unsqueeze(1)
            .repeat(1, num_in_boxes_anchor, 1)
        )
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()

        cls_preds_ = (
            cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            * obj_preds_.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
        )
        pair_wise_cls_loss = F.binary_cross_entropy(
            cls_preds_.sqrt_(), gt_cls_per_image, reduction="none"
        ).sum(-1)
        del cls_preds_

        cost = (
            pair_wise_cls_loss
            + 3.0 * pair_wise_ious_loss
            + 100000.0 * (~is_in_boxes_and_center)
        )

        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
        ) = self.dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()

        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )

    def get_in_boxes_info(
        self,
        gt_bboxes_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        total_num_anchors,
        num_gt,
    ):
        expanded_strides_per_image = expanded_strides[0]
        x_shifts_per_image = x_shifts[0] * expanded_strides_per_image
        y_shifts_per_image = y_shifts[0] * expanded_strides_per_image
        x_centers_per_image = (
            (x_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0)
            .repeat(num_gt, 1)
        )
        y_centers_per_image = (
            (y_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0)
            .repeat(num_gt, 1)
        )

        gt_bboxes_per_image_l = (
            (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_r = (
            (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_t = (
            (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_b = (
            (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )

        b_l = x_centers_per_image - gt_bboxes_per_image_l
        b_r = gt_bboxes_per_image_r - x_centers_per_image
        b_t = y_centers_per_image - gt_bboxes_per_image_t
        b_b = gt_bboxes_per_image_b - y_centers_per_image
        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)

        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0

        center_radius = 2.5

        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)

        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all

        is_in_boxes_and_center = (
            is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        )
        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        matching_matrix = torch.zeros_like(cost)

        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = min(10, ious_in_boxes_matrix.size(1))
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1.0

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0.0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
        fg_mask_inboxes = matching_matrix.sum(0) > 0.0
        num_fg = fg_mask_inboxes.sum().item()

        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds

class SOLOLoss(nn.Module):

    def __init__(self, ins_weight=3.0, num_classes=80):
        super().__init__()
        self.ins_loss_weight = ins_weight
        self.dice_loss = Diceloss(reduction="none")
        self.loss_cate = FocalLoss_cuda(use_sigmoid=True,gamma=2.0, alpha=0.25, loss_weight=1.0)
        self.scale_ranges = ((1, 56), (28, 112), (56, 224), (112, 448), (224, 896))
        self.seg_num_grids = [40, 36, 24, 16, 12]
        self.sigma = 0.2
        self.cate_out_channels = num_classes

    def forward(self, y_pr, y_gt):
        catekernel_outs, maskfeature_outs = y_pr
        cate_preds, kernel_preds = catekernel_outs

        gt_bbox_list = y_gt['batch_bboxes']
        gt_label_list = y_gt['batch_labels']
        gt_mask_list = y_gt['batch_masks']
        mask_feat_size = maskfeature_outs.size()[-2:]

        ins_label_list, cate_label_list, ins_ind_label_list, grid_order_list = multi_apply(
            self.solov2_target_single,
            gt_bbox_list,
            gt_label_list,
            gt_mask_list,
            mask_feat_size=mask_feat_size)

        ins_labels = [torch.cat([ins_labels_level_img
                                    for ins_labels_level_img in ins_labels_level], 0)
                                 for ins_labels_level in zip(*ins_label_list)]

        kernel_preds = [[kernel_preds_level_img.view(kernel_preds_level_img.shape[0], -1)[:, grid_orders_level_img]
                                            for kernel_preds_level_img, grid_orders_level_img in   
                                            zip(kernel_preds_level, grid_orders_level)] 
                                            for kernel_preds_level, grid_orders_level in zip(kernel_preds, zip(*grid_order_list))]

        ins_pred_list = []
        for b_kernel_pred in kernel_preds:
            b_mask_pred = []
            for idx, kernel_pred in enumerate(b_kernel_pred):

                if kernel_pred.size()[-1] == 0:
                    continue
                cur_ins_pred = maskfeature_outs[idx, ...]
                H, W = cur_ins_pred.shape[-2:]
                N, I = kernel_pred.shape
                cur_ins_pred = cur_ins_pred.unsqueeze(0)
                kernel_pred = kernel_pred.permute(1, 0).view(I, -1, 1, 1)
                cur_ins_pred = F.conv2d(cur_ins_pred, kernel_pred, stride=1).view(-1, H, W)
                b_mask_pred.append(cur_ins_pred)
            if len(b_mask_pred) == 0:
                b_mask_pred = None
            else:
                b_mask_pred = torch.cat(b_mask_pred, 0)
            ins_pred_list.append(b_mask_pred)

        ins_ind_labels = [
                                            torch.cat([ins_ind_labels_level_img.flatten()
                                                    for ins_ind_labels_level_img in ins_ind_labels_level])
                                            for ins_ind_labels_level in zip(*ins_ind_label_list)
                                            ]

        flatten_ins_ind_labels = torch.cat(ins_ind_labels)

        num_ins = flatten_ins_ind_labels.sum()

        loss_ins = []

        for input, target in zip(ins_pred_list, ins_labels): 
            if input is None:
                continue
            input = torch.sigmoid(input)
            loss_ins.append(self.dice_loss(input, target))

        loss_ins = torch.cat(loss_ins).mean()
        loss_ins = loss_ins * self.ins_loss_weight

        cate_labels = [
                                    torch.cat([cate_labels_level_img.flatten()
                                            for cate_labels_level_img in cate_labels_level])
                                    for cate_labels_level in zip(*cate_label_list)
                                ]

        flatten_cate_labels = torch.cat(cate_labels)

        cate_preds = [
                                    cate_pred.permute(0, 2, 3, 1).reshape(-1, self.cate_out_channels) 
                                    for cate_pred in cate_preds
                                ]

        flatten_cate_preds = torch.cat(cate_preds)
        flatten_cate_labels = flatten_cate_labels.cuda()
        num_ins = num_ins.cuda()

        loss_cate = self.loss_cate(flatten_cate_preds, flatten_cate_labels, avg_factor=num_ins + 1)

        return loss_ins + loss_cate

    def imrescale(self, img, scale):
        h, w = img.shape[:2]
        new_size = (int(w * float(scale) + 0.5), int(h * float(scale) + 0.5))
        resized_img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)

        return resized_img

    def center_of_mass(self, bitmasks):
        _, h, w = bitmasks.size()
        ys = torch.arange(0, h, dtype=torch.float32, device=bitmasks.device)
        xs = torch.arange(0, w, dtype=torch.float32, device=bitmasks.device)

        m00 = bitmasks.sum(dim=-1).sum(dim=-1).clamp(min=1e-6)
        m10 = (bitmasks * xs).sum(dim=-1).sum(dim=-1)
        m01 = (bitmasks * ys[:, None]).sum(dim=-1).sum(dim=-1)
        center_x = m10 / m00
        center_y = m01 / m00
        return center_x, center_y

    def solov2_target_single(self,  gt_bboxes_raw,  gt_labels_raw,  gt_masks_raw, mask_feat_size):
        gt_bboxes_raw = gt_bboxes_raw.cuda()
        gt_labels_raw = gt_labels_raw.cuda()
        device = gt_labels_raw.device
        gt_masks_raw = gt_masks_raw.cpu().numpy()
        gt_areas = torch.sqrt((gt_bboxes_raw[:, 2] - gt_bboxes_raw[:, 0]) * (gt_bboxes_raw[:, 3] - gt_bboxes_raw[:, 1]))

        ins_label_list = []
        cate_label_list = []
        ins_ind_label_list = []
        grid_order_list = []
        for (lower_bound, upper_bound),  num_grid  in zip(self.scale_ranges, self.seg_num_grids):
            hit_indices = ((gt_areas >= lower_bound) & (gt_areas <= upper_bound)).nonzero(as_tuple=False).flatten()
            num_ins = len(hit_indices)

            ins_label = []
            grid_order = []

            cate_label = torch.zeros([num_grid, num_grid], dtype=torch.int64, device=device)


            ins_ind_label = torch.zeros([num_grid ** 2], dtype=torch.bool, device=device)

            if num_ins == 0:
                ins_label = torch.zeros([0, mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8, device=device)
                ins_label_list.append(ins_label)
                cate_label_list.append(cate_label)
                ins_ind_label_list.append(ins_ind_label)
                grid_order_list.append([])
                continue
            gt_bboxes = gt_bboxes_raw[hit_indices]
            gt_labels = gt_labels_raw[hit_indices]
            gt_masks = gt_masks_raw[hit_indices.cpu().numpy(), ...]

            half_ws = 0.5 * (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * self.sigma
            half_hs = 0.5 * (gt_bboxes[:, 3] - gt_bboxes[:, 1]) * self.sigma

            gt_masks_pt = torch.from_numpy(gt_masks).to(device=device)
            center_ws, center_hs = self.center_of_mass(gt_masks_pt)
            valid_mask_flags = gt_masks_pt.sum(dim=-1).sum(dim=-1) > 0

            output_stride = 4
            for seg_mask, gt_label, half_h, half_w, center_h, center_w, valid_mask_flag in zip(gt_masks, gt_labels, half_hs, half_ws, center_hs, center_ws, valid_mask_flags):
                if not valid_mask_flag:
                    continue
                upsampled_size = (mask_feat_size[0] * 4, mask_feat_size[1] * 4)

                coord_w = torch.div((center_w / upsampled_size[1]), (1. / num_grid), rounding_mode='trunc')
                coord_h = torch.div((center_h / upsampled_size[0]), (1. / num_grid), rounding_mode='trunc')

                top_box = max(0, torch.div(((center_h - half_h) / upsampled_size[0]), (1. / num_grid), rounding_mode='trunc'))
                down_box = min(num_grid - 1, torch.div(((center_h + half_h) / upsampled_size[0]), (1. / num_grid), rounding_mode='trunc'))
                left_box =max(0, torch.div(((center_w - half_w) / upsampled_size[1]), (1. / num_grid), rounding_mode='trunc'))
                right_box = min(num_grid - 1, torch.div(((center_w + half_w) / upsampled_size[1]), (1. / num_grid), rounding_mode='trunc'))

                top = int(max(top_box, coord_h-1))
                down =int(min(down_box, coord_h+1)) 
                left = int(max(coord_w-1, left_box))
                right = int(min(right_box, coord_w+1))

                cate_label[top:(down+1), left:(right+1)] = gt_label
                seg_mask = self.imrescale(seg_mask, scale=1. / output_stride)
                seg_mask = torch.Tensor(seg_mask)
                seg_mask = seg_mask.to(device=device)

                for i in range(top, down+1):
                    for j in range(left, right+1):
                        label = int(i * num_grid + j)

                        cur_ins_label = torch.zeros([mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8, device=device)
                        cur_ins_label[:seg_mask.shape[0], :seg_mask.shape[1]] = seg_mask

                        ins_label.append(cur_ins_label)
                        ins_ind_label[label] = True
                        grid_order.append(label)

            if len(ins_label) == 0:
                ins_label = torch.zeros([0, mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8, device=device)
            else:
                ins_label = torch.stack(ins_label, 0)
  
            ins_label_list.append(ins_label)
            cate_label_list.append(cate_label)
            ins_ind_label_list.append(ins_ind_label)
            grid_order_list.append(grid_order)
        
        return ins_label_list, cate_label_list, ins_ind_label_list, grid_order_list

class ReBiSeDetailAggregateLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ReBiSeDetailAggregateLoss, self).__init__()
        self.dice_loss = Diceloss(reduction='mean')
        self.laplacian_kernel = torch.tensor(
            [-1, -1, -1, -1, 8, -1, -1, -1, -1],
            dtype=torch.float32).reshape(1, 1, 3, 3).requires_grad_(False).type(torch.cuda.FloatTensor)
        
        self.fuse_kernel = torch.nn.Parameter(torch.tensor([[6./10], [3./10], [1./10]],
            dtype=torch.float32).reshape(1, 3, 1, 1).type(torch.cuda.FloatTensor))

    def forward(self, boundary_logits, gtmasks):

        boundary_targets = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel, padding=1)
        boundary_targets = boundary_targets.clamp(min=0)
        boundary_targets[boundary_targets > 0.1] = 1
        boundary_targets[boundary_targets <= 0.1] = 0

        boundary_targets_x2 = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel, stride=2, padding=1)
        boundary_targets_x2 = boundary_targets_x2.clamp(min=0)
        
        boundary_targets_x4 = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel, stride=4, padding=1)
        boundary_targets_x4 = boundary_targets_x4.clamp(min=0)

    
        boundary_targets_x4_up = F.interpolate(boundary_targets_x4, boundary_targets.shape[2:], mode='nearest')
        boundary_targets_x2_up = F.interpolate(boundary_targets_x2, boundary_targets.shape[2:], mode='nearest')
        
        boundary_targets_x2_up[boundary_targets_x2_up > 0.1] = 1
        boundary_targets_x2_up[boundary_targets_x2_up <= 0.1] = 0
        
        boundary_targets_x4_up[boundary_targets_x4_up > 0.1] = 1
        boundary_targets_x4_up[boundary_targets_x4_up <= 0.1] = 0
       
        
        boudary_targets_pyramids = torch.stack((boundary_targets, boundary_targets_x2_up, boundary_targets_x4_up), dim=1)
        
        boudary_targets_pyramids = boudary_targets_pyramids.squeeze(2)
        boudary_targets_pyramid = F.conv2d(boudary_targets_pyramids, self.fuse_kernel)

        boudary_targets_pyramid[boudary_targets_pyramid > 0.1] = 1
        boudary_targets_pyramid[boudary_targets_pyramid <= 0.1] = 0
        
        if boundary_logits.shape[-1] != boundary_targets.shape[-1]:
            boundary_logits = F.interpolate(
                boundary_logits, boundary_targets.shape[2:], mode='bilinear', align_corners=True)
        
        bce_loss = F.binary_cross_entropy_with_logits(boundary_logits, boudary_targets_pyramid, reduce=None, reduction='mean')
        dice_loss = self.dice_loss(torch.sigmoid(boundary_logits), boudary_targets_pyramid)

        return bce_loss,  dice_loss

class ReBiSeLoss(nn.Module):
    def __init__(self, n_min, thresh=0.7, ignore_lb=255):
        super(ReBiSeLoss, self).__init__()
        self.detail_aggre_loss = ReBiSeDetailAggregateLoss()
        self.seg_out_loss = OhemCELoss(thresh=thresh, n_min=n_min, ignore_lb=ignore_lb)
        self.supervise_feature8_loss = OhemCELoss(thresh=thresh, n_min=n_min, ignore_lb=ignore_lb)
        self.supervise_feature16_loss = OhemCELoss(thresh=thresh, n_min=n_min, ignore_lb=ignore_lb)

    def forward(self, y_pr, y_gt):
        feat_out, feat_out8, feat_out16, featdetail_sp8 = y_pr
        seg_loss = self.seg_out_loss(feat_out, y_gt)
        sup_f8_loss = self.supervise_feature8_loss(feat_out8, y_gt)
        sup_f16_loss = self.supervise_feature8_loss(feat_out16, y_gt)
        boundery_detaile_bce_loss,  boundery_detaile_dice_loss = self.detail_aggre_loss(featdetail_sp8, y_gt)

        return seg_loss + sup_f8_loss + sup_f16_loss + boundery_detaile_bce_loss + boundery_detaile_dice_loss
