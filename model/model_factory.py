'''
# Author: Shaoran Lu
# Date: 2021/10/04
# Email: lushaoran92@gmail.com
# Description: Model 类

example:

'''

import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import numpy as np
from loguru import logger
from tqdm import tqdm

import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

from model.models import CRNN, DBNet, Yolox, SOLO, ImageCaptionTransformer, ReBiSeNet, MaskFormer
from model.backbone import ResNet, SwinTransformer

from tools.evaluation_tools import ConvertCocoFormat, Coco_eval, SemanticSegmIOU
from tools.loss.loss import CrnnLoss, DBLoss, YoloxLoss, SOLOLoss, ReBiSeLoss
from tools.loss.detr_criterion import SetCriterion
from tools.loss.detr_matcher import HungarianMatcher
from tools.nms import multiclass_nms, matrix_nms

from data.data_utils import preproc

from utils.common import reduce_mean, reduce_sum
from utils.standard_tools import togpu, recursiveToTensor

import segmentation_models_pytorch as smp

class Classify_Model():
    def __init__(self, config, CLASSES_NUM=10, optimizer_name='adamW', scheduler_name='Cosine', amp_training=False, act='relu'):
        
        self.config = config
        self.CLASSES_NUM = config.get('classes_num', CLASSES_NUM)
        self.optimizer_name = config.get('optimizer', optimizer_name)
        self.scheduler_name = config.get('scheduler', scheduler_name)
    def get_model(self):
        if self.config['backbone'] in ['res18', 'res34', 'res50', 'res101', 'res152']:
            mode_dict = {'res18':'18', 'res34':'34', 'res50':'50', 'res101':'101', 'res152':'152'}
            model = ResNet( type_mode=mode_dict[self.config['backbone']], num_classes=self.CLASSES_NUM, extract_feature=False)
        elif self.config['backbone'] in ['sw_t', 'sw_s', 'sw_b', 'sw_l']:
            mode_dict = {'sw_t':'tiny', 'sw_s':'small', 'sw_b':'base', 'sw_l':'large'}
            model = SwinTransformer(type_mode=mode_dict[self.config['backbone']], num_classes=self.CLASSES_NUM, extract_feature=False)
        else:
            raise NotImplementedError('Model {} not supported.'.format(self.config['backbone']))
        return model
    
    def get_optimizer(self, model):
        if self.optimizer_name == 'adam':
            optimizer = torch.optim.Adam([ 
                                        dict(params=model.parameters(), lr=self.config['lr']),
                                    ])
        elif self.optimizer_name == 'adamW':
            optimizer = torch.optim.AdamW(params=model.parameters(), lr=self.config['lr'])

        elif self.optimizer_name == 'SGD':
            optimizer =  torch.optim.SGD(params=model.parameters(), lr=self.config['lr'], momentum=0.9, weight_decay=0.0001)

        else:
            raise NotImplementedError('Optimizer {} not supported.'.format(self.optimizer_name))
        
        return optimizer
    
    def get_lr_scheduler(self, optimizer, max_iter):
        if self.scheduler_name == 'Cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iter,
                                                            eta_min=0.00001, last_epoch=-1, verbose=False)
        elif self.scheduler_name == 'OneCycleLR':
            return torch.optim.lr_scheduler.OneCycleLR(optimizer, self.config['lr'], total_steps=max_iter, verbose=False)

        else:
            return None
                 
    def get_loss_func(self):
        return nn.CrossEntropyLoss()
    
    def eval_model(self, model, data_loader, loss_fn, local_rank, distributed=False, is_main_process=True, nprocs=1,  loss_criteria=True):
        size = len(data_loader.dataset)
        num_batches = len(data_loader)
        test_loss, correct = 0, 0
        model.eval()
        with torch.no_grad():
            for X, y in data_loader:
                X = X.cuda(local_rank, non_blocking=True)
                y = y.cuda(local_rank, non_blocking=True)
                pred = model(X)
                loss = loss_fn(pred, y)
                cr= (pred.argmax(1) == y).type(torch.float).sum()
                if distributed:
                    torch.distributed.barrier()
                    loss= reduce_mean(loss, nprocs if distributed else 1)
                    cr= reduce_sum(cr)
                correct += cr
                test_loss += loss.item()
        test_loss /= num_batches
        correct /= size
        if is_main_process:
            logger.info(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        return test_loss

class Segmentation_Model():
    def __init__(self, config, ENCODER='se_resnext50_32x4d', ENCODER_WEIGHTS='imagenet',
                                        act='sigmoid', optimizer_name='adamW', scheduler_name='Cosine', amp_training=False):
        
        self.config = config
        self.ENCODER = config.get('encoder', ENCODER)
        self.ENCODER_WEIGHTS = config.get('weights', ENCODER_WEIGHTS)
        self.ACTIVATION = config.get('activation', act)
        self.optimizer_name = config.get('optimizer', optimizer_name)
        self.scheduler_name = config.get('scheduler', scheduler_name)
    def get_model(self, CLASSES = ['visible_row', 'visible_column', 'unvisible_row', 'unvisible_column']):
        self.CLASSES = self.config.get('classes', CLASSES)
        model = smp.FPN(
                        encoder_name=self.ENCODER, 
                        encoder_weights=self.ENCODER_WEIGHTS, 
                        classes=len(self.CLASSES)+1, 
                        activation=self.ACTIVATION,
                    )
        return model
    
    def get_optimizer(self, model):
        if self.optimizer_name == 'adam':
            optimizer = torch.optim.Adam([ 
                                        dict(params=model.parameters(), lr=self.config['lr']),
                                    ])
        elif self.optimizer_name == 'adamW':
            optimizer = torch.optim.AdamW(params=model.parameters(), lr=self.config['lr'])

        elif self.optimizer_name == 'SGD':
            optimizer =  torch.optim.SGD(params=model.parameters(), lr=self.config['lr'], momentum=0.9, weight_decay=0.0001)

        else:
            raise NotImplementedError('Optimizer {} not supported.'.format(self.optimizer_name))
        
        return optimizer
    
    def get_lr_scheduler(self, optimizer, max_iter):
        if self.scheduler_name == 'Cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iter,
                                                            eta_min=0.00001, last_epoch=-1, verbose=False)
        elif self.scheduler_name == 'OneCycleLR':
            return torch.optim.lr_scheduler.OneCycleLR(optimizer, self.config['lr'], total_steps=max_iter, verbose=False)

        else:
            return None
    
    def get_loss_func(self):
        return smp.utils.losses.DiceLoss()
    
    def eval_model(self, model, data_loader, loss_fn, local_rank, distributed=False, is_main_process=True, nprocs=1,  loss_criteria=True):
        eval_nums = len(data_loader)
        model.eval()
        eval_loss = 0.
        for inps, targets in data_loader:
            with torch.no_grad():
                inps = inps.cuda(local_rank, non_blocking=True)
                targets = targets.cuda(local_rank, non_blocking=True)
                logic = model(inps)
                loss = loss_fn(logic, targets)
                if distributed:
                    torch.distributed.barrier()
                    loss= reduce_mean(loss, nprocs if distributed else 1)
                eval_loss += loss.item()
        return eval_loss/eval_nums

class Crnn_Model():
    def __init__(self, config, BACKBONE='res18', optimizer_name='adamW', scheduler_name='Cosine', amp_training=False, act='relu'):
        self.config = config
        self.num_classes = int(self.config['num_classes'])
        self.optimizer_name = config.get('optimizer', optimizer_name)
        self.scheduler_name = config.get('scheduler', scheduler_name)
        self.BACKBONE = config.get('backbone', BACKBONE)

    def get_model(self):
        model = CRNN(type_mode=self.BACKBONE, num_classes=self.num_classes, feature_size=512, hidden_feature=256, train_model=True)
        return model
    
    def get_optimizer(self, model):
        if self.optimizer_name == 'adam':
            optimizer = torch.optim.Adam([ 
                                        dict(params=model.parameters(), lr=self.config['lr']),
                                    ])
        elif self.optimizer_name == 'adamW':
            optimizer = torch.optim.AdamW(params=model.parameters(), lr=self.config['lr'])

        elif self.optimizer_name == 'SGD':
            optimizer =  torch.optim.SGD(params=model.parameters(), lr=self.config['lr'], momentum=0.9, weight_decay=0.0001)

        else:
            raise NotImplementedError('Optimizer {} not supported.'.format(self.optimizer_name))
        
        return optimizer
    
    def get_lr_scheduler(self, optimizer, max_iter):
        if self.scheduler_name == 'Cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iter,
                                                            eta_min=0.00001, last_epoch=-1, verbose=False)
        elif self.scheduler_name == 'OneCycleLR':
            return torch.optim.lr_scheduler.OneCycleLR(optimizer, self.config['lr'], total_steps=max_iter, verbose=False)

        else:
            return None
    
    def get_loss_func(self):
        return CrnnLoss(reduction='mean')
    
    def eval_model(self, model, data_loader, loss_fn, local_rank, distributed=False, is_main_process=True, nprocs=1,  loss_criteria=True):
        size = len(data_loader.dataset)
        num_batches = len(data_loader)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for inps, targets in data_loader:
                b_correct = 0
                inps = inps.cuda(local_rank, non_blocking=True)
                targets = targets.cuda(local_rank, non_blocking=True)
                logic = model(inps)
                loss = loss_fn(logic, targets)
                logic = logic.argmax(2)
                logic = logic.transpose(1, 0).contiguous()
                for pre, lab in zip(targets, logic):
                    pre = self.eval_utils(pre)
                    lab = [int(i) for i in lab.data]
                    if pre == lab:
                        b_correct += 1

                if distributed:
                    torch.distributed.barrier()
                    loss= reduce_mean(loss, nprocs if distributed else 1)
                    b_correct= reduce_sum(b_correct)
                
                correct += b_correct
                test_loss += loss.item()
        test_loss /= num_batches
        correct /= size
        if is_main_process:
            logger.info(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        return test_loss
        
    def eval_utils(self, x, ignore_index=0):
        y = []
        x = x.data
        y_ = [i for i in x if i != ignore_index]
        y.append(int(y_[0]))
        for i in range(len(y_)-1):
            if y_[i] == y_[i+1]:
                continue
            else:
                y.append(int(y_[i+1]))
    
class DB_Model():
    def __init__(self, config, BACKBONE='res18', NECK='FPN', optimizer_name='adamW', scheduler_name='Cosine', amp_training=False, act='relu'):
        self.config = config
        self.optimizer_name = config.get('optimizer', optimizer_name)
        self.scheduler_name = config.get('scheduler', scheduler_name)
        self.BACKBONE = config.get('backbone', BACKBONE)
        self.NECK = config.get('neck', NECK)
    def get_model(self):
        model = DBNet(type_mode=self.BACKBONE, neck_mode=self.NECK, train_model=True)
        return model

    def get_inference_model(self):
        model = DBNet(type_mode=self.BACKBONE, neck_mode=self.NECK, train_model=False)
        return model
    
    def get_optimizer(self, model):
        if self.optimizer_name == 'adam':
            optimizer = torch.optim.Adam([ 
                                        dict(params=model.parameters(), lr=self.config['lr']),
                                    ])
        elif self.optimizer_name == 'adamW':
            optimizer = torch.optim.AdamW(params=model.parameters(), lr=self.config['lr'])

        elif self.optimizer_name == 'SGD':
            optimizer =  torch.optim.SGD(params=model.parameters(), lr=self.config['lr'], momentum=0.9, weight_decay=0.0001)

        else:
            raise NotImplementedError('Optimizer {} not supported.'.format(self.optimizer_name))
        
        return optimizer
    
    def get_lr_scheduler(self, optimizer, max_iter):
        if self.scheduler_name == 'Cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iter,
                                                            eta_min=0.00001, last_epoch=-1, verbose=False)
        elif self.scheduler_name == 'OneCycleLR':
            return torch.optim.lr_scheduler.OneCycleLR(optimizer, self.config['lr'], total_steps=max_iter, verbose=False)

        else:
            return None
    
    def get_loss_func(self):
        return DBLoss()
    
    def eval_model(self, model, data_loader, loss_fn, local_rank, distributed=False, is_main_process=True, nprocs=1,  loss_criteria=True):
        eval_nums = len(data_loader)
        model.eval()
        eval_loss = 0.
        for inps, targets in data_loader:
            with torch.no_grad():
                inps = inps.cuda(local_rank, non_blocking=True)
                targets = targets.cuda(local_rank, non_blocking=True)
                logic = model(inps)
                loss = loss_fn(logic, targets)
                if distributed:
                    torch.distributed.barrier()
                    loss= reduce_mean(loss, nprocs if distributed else 1)
                eval_loss += loss.item()
        return eval_loss/eval_nums

class Yolox_Model():
    def __init__(self, config, BACKBONE='dark_m', NECK='PAFPN', optimizer_name='adamW', scheduler_name='Cosine', amp_training=False, act='relu'):
        self.config = config
        self.amp_training = amp_training
        self.tensor_type = torch.cuda.HalfTensor if self.amp_training else torch.cuda.FloatTensor
        self.optimizer_name = config.get('optimizer', optimizer_name)
        self.scheduler_name = config.get('scheduler', scheduler_name)
        self.BACKBONE = config.get('backbone', BACKBONE)
        self.NECK = config.get('neck', NECK)
        self.act = config.get('act', act)
        self.num_classes = config.get('num_classes', 80)

        self.rgb_mean = config.get('rgb_mean', (0.485, 0.456, 0.406))
        self.rgb_std = config.get('rgb_std', (0.229, 0.224, 0.225))

    def get_model(self):
        model = Yolox(type_mode=self.BACKBONE, neck_mode=self.NECK, train_model=True, num_classes=self.num_classes, decode_in_inference=False, act=self.act)
        return model

    def get_inference_model(self):
        model = Yolox(type_mode=self.BACKBONE, neck_mode=self.NECK, train_model=False, num_classes=self.num_classes, decode_in_inference=False, act=self.act)
        model = model.eval()
        return model
    
    def get_optimizer(self, model):
        if self.optimizer_name == 'adam':
            optimizer = torch.optim.Adam([ 
                                        dict(params=model.parameters(), lr=self.config['lr']),
                                    ])
        elif self.optimizer_name == 'adamW':
            optimizer = torch.optim.AdamW(params=model.parameters(), lr=self.config['lr'])

        elif self.optimizer_name == 'SGD':
            optimizer =  torch.optim.SGD(params=model.parameters(), lr=self.config['lr'], momentum=0.9, weight_decay=0.0001, nesterov=True)

        else:
            raise NotImplementedError('Optimizer {} not supported.'.format(self.optimizer_name))
        
        return optimizer
    
    def get_lr_scheduler(self, optimizer, max_iter):
        if self.scheduler_name == 'Cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iter,
                                                            eta_min=0.00001, last_epoch=-1, verbose=False)
        elif self.scheduler_name == 'OneCycleLR':
            return torch.optim.lr_scheduler.OneCycleLR(optimizer, self.config['lr'], total_steps=max_iter, verbose=False)

        else:
            return None
    
    def get_loss_func(self):
        return YoloxLoss(num_classes=int(self.config['num_classes']))
    
    def preprocess(self, image, input_size, swap=(2, 0, 1)):
        padded_img, r = preproc(image, input_size, self.rgb_mean, self.rgb_std, swap=swap, interpolation='bilinear')
        return padded_img, r

    def postprocess(self, outputs, ratio, nms_thr=0.45, score_thr=0.1):
        outputs = outputs[0].cpu().numpy()
        boxes = outputs[:, :4]
        scores = outputs[:, 4:5] * outputs[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
        boxes_xyxy /= ratio

        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=nms_thr, score_thr=score_thr)

        if dets is not None:
            final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
            return final_boxes, final_scores, final_cls_inds
        else:
            return None, None, None

    def postprocess_tensor(self, input_size, outputs, ratios, nms_thr, score_thr, max_boxes=100):
        b_bboxes =  [None for _ in range(outputs.shape[0])]
        b_cls = [None for _ in range(outputs.shape[0])]
        b_scores = [None for _ in range(outputs.shape[0])]

        for i, output in enumerate(outputs):
            
            class_scores, class_pred = torch.max(output[:, 5:], dim=1, keepdim=False)
            
            conf_mask = (output[:, 4] * class_scores >= score_thr).squeeze()
            scores = output[:, 4] * class_scores
            boxes = output[:, :4]

            boxes_xyxy = torch.ones_like(boxes)
            boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
            boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
            boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
            boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
            boxes_xyxy /= ratios[i]

            boxes_xyxy = boxes_xyxy[conf_mask]
            scores = scores[conf_mask]
            class_pred = class_pred[conf_mask]

            nms_out_index = torchvision.ops.batched_nms(boxes_xyxy, scores, class_pred, iou_threshold=nms_thr)
            b_bboxes[i] = boxes_xyxy[nms_out_index]
            b_cls[i] = class_pred[nms_out_index]
            b_scores[i] = scores[nms_out_index]

        return b_bboxes, b_cls, b_scores

    def eval_model(self, model, data_loader, loss_fn, local_rank, distributed=False, is_main_process=True, nprocs=1,
                                    loss_criteria=True, confthre=0.1, nmsthre=0.65):

        eval_nums = len(data_loader)
        convert_format = ConvertCocoFormat(data_loader.dataset.id2cat)
        if loss_criteria:
            model.train()
            eval_loss = 0.
            with torch.no_grad():
                for inps, targets, img_info, img_id, scale_factors in data_loader:
                    inps = inps.cuda(local_rank, non_blocking=True)
                    targets = targets.cuda(local_rank, non_blocking=True)
                    logic = model(inps)
                    loss = loss_fn(logic, targets)
                    if distributed:
                        torch.distributed.barrier()
                        loss= reduce_mean(loss, nprocs if distributed else 1)
                    eval_loss += loss.item()
            return eval_loss/eval_nums
        else:
            val_data_list = []
            model =  model.eval()
            with torch.no_grad():
                for inps, targets, img_infos, img_id, scale_factors in tqdm(data_loader):
                    inps = inps.cuda(local_rank, non_blocking=True)
                    targets = targets.cuda(local_rank, non_blocking=True)
                    scale_factors = scale_factors.cuda(local_rank, non_blocking=True)
                    if self.amp_training:
                        inps = inps.type(self.tensor_type)
                        scale_factors = scale_factors.type(self.tensor_type)

                    outputs = model(inps)
                    boxes, cls_inds, scores = self.postprocess_tensor(eval(self.config['input_size']), outputs, scale_factors, nms_thr=nmsthre, score_thr=confthre, max_boxes=100)
                    data_list_ = convert_format(boxes, cls_inds, scores, img_id)
                    val_data_list.extend(data_list_)

            mAP = Coco_eval(eval_bbox=True)(val_data_list, os.path.join(self.config['data_dir'], "annotations", self.config['json_file_val']))
            return mAP

class Solo_Model():
    def __init__(self, config, BACKBONE='res_18',  num_classes=80, optimizer_name='adamW', scheduler_name='Cosine', amp_training=False, act='relu'):
        self.config = config
        self.optimizer_name = config.get('optimizer', optimizer_name)
        self.scheduler_name = config.get('scheduler', scheduler_name)
        self.BACKBONE = config.get('backbone', BACKBONE)
        self.num_classes = int(config.get('num_classes', num_classes))

        self.rgb_mean = config.get('rgb_mean', (0.485, 0.456, 0.406))
        self.rgb_std = config.get('rgb_std', (0.229, 0.224, 0.225))

    def get_model(self):
        model = SOLO(type_mode=self.BACKBONE, train_model=True, num_classes=self.num_classes)
        return model

    def get_inference_model(self):
        model = SOLO(type_mode=self.BACKBONE, train_model=False, num_classes=self.num_classes)
        return model
    
    def get_optimizer(self, model):
        if self.optimizer_name == 'adam':
            optimizer = torch.optim.Adam([ 
                                        dict(params=model.parameters(), lr=self.config['lr']),
                                    ])
        elif self.optimizer_name == 'adamW':
            optimizer = torch.optim.AdamW(params=model.parameters(), lr=self.config['lr'])

        elif self.optimizer_name == 'SGD':
            optimizer =  torch.optim.SGD(params=model.parameters(), lr=self.config['lr'], momentum=0.9, weight_decay=0.0001)

        else:
            raise NotImplementedError('Optimizer {} not supported.'.format(self.optimizer_name))
        
        return optimizer
    
    def get_lr_scheduler(self, optimizer, max_iter):
        if self.scheduler_name == 'Cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iter,
                                                            eta_min=0.00001, last_epoch=-1, verbose=False)
        elif self.scheduler_name == 'OneCycleLR':
            return torch.optim.lr_scheduler.OneCycleLR(optimizer, self.config['lr'], total_steps=max_iter, verbose=False)

        else:
            return None
    
    def get_loss_func(self):
        return SOLOLoss(num_classes=self.num_classes)
    
    def eval_model(self, model, data_loader, loss_fn, local_rank, distributed=False, is_main_process=True, nprocs=1, loss_criteria=True):
        eval_nums = len(data_loader)
        model.eval()
        eval_loss = 0.
        for inps, targets in data_loader:
            with torch.no_grad():
                inps = inps.cuda(local_rank, non_blocking=True)
                logic = model.forward(inps)
                loss = loss_fn(logic, targets)
                if distributed:
                    torch.distributed.barrier()
                    loss= reduce_mean(loss, nprocs if distributed else 1)
                eval_loss += loss.item()
        return eval_loss/eval_nums

    def preprocess(self, image, input_size, swap=(2, 0, 1)):
        self.ori_shape = image.shape
        padded_img, r = preproc(image, input_size, self.rgb_mean, self.rgb_std, swap=swap, interpolation='bilinear')
        self.img_shape = padded_img.shape
        return padded_img, r

    def postprocess(self, outputs, ratio):
        catekernel_outs, maskfeature_outs = outputs
        cate_preds, kernel_preds = catekernel_outs

        seg_results = self.get_seg(cate_preds, kernel_preds, maskfeature_outs, self.img_shape, ratio, cate_out_channels=80, kernel_out_channels=128)
        return seg_results

    def get_seg(self, cate_preds, kernel_preds, seg_pred, img_shape, ratio, cate_out_channels=80, kernel_out_channels=128):
        num_levels = len(cate_preds)
        featmap_size = seg_pred.size()[-2:]

        cate_pred_list = [
            cate_preds[i][0].view(-1, cate_out_channels).detach() for i in range(num_levels)
        ]
        seg_pred_list = seg_pred[0, ...].unsqueeze(0)
        kernel_pred_list = [
            kernel_preds[i][0].permute(1, 2, 0).view(-1, kernel_out_channels).detach()
                            for i in range(num_levels)
        ]
        cate_pred_list = torch.cat(cate_pred_list, dim=0)
        kernel_pred_list = torch.cat(kernel_pred_list, dim=0)

        result = self.get_seg_single(cate_pred_list, seg_pred_list, kernel_pred_list,
                                        featmap_size, img_shape, ratio, score_thr=0.1, mask_thr=0.5, nms_pre=500, kernel='gaussian', sigma=2.0, update_thr=0.05, max_per_img=30)
        return result

    def get_seg_single(self,
                       cate_preds,
                       seg_preds,
                       kernel_preds,
                       featmap_size,
                       img_shape,
                       ratio,
                       score_thr, mask_thr, nms_pre, kernel, sigma, update_thr, max_per_img,
                       seg_num_grids=[40, 36, 24, 16, 12],
                       strids_ = [8, 8, 16, 32, 32]):

        assert len(cate_preds) == len(kernel_preds)

        _, h, w = img_shape
        upsampled_size_out = (featmap_size[0] * 4, featmap_size[1] * 4)

        inds = (cate_preds > score_thr)
        cate_scores = cate_preds[inds]
        if len(cate_scores) == 0:
            return None

        inds = inds.nonzero()
        cate_labels = inds[:, 1]
        kernel_preds = kernel_preds[inds[:, 0]]

        size_trans = cate_labels.new_tensor(seg_num_grids).pow(2).cumsum(0)
        strides = kernel_preds.new_ones(size_trans[-1])

        n_stage = len(seg_num_grids)
        strides[:size_trans[0]] *= strids_[0]
        for ind_ in range(1, n_stage):
            strides[size_trans[ind_-1]:size_trans[ind_]] *= strids_[ind_]
        strides = strides[inds[:, 0]]

        I, N = kernel_preds.shape
        kernel_preds = kernel_preds.view(I, N, 1, 1)
        seg_preds = F.conv2d(seg_preds, kernel_preds, stride=1).squeeze(0).sigmoid()
        seg_masks = seg_preds > mask_thr
        sum_masks = seg_masks.sum((1, 2)).float()

        keep = sum_masks > strides
        if keep.sum() == 0:
            return None

        seg_masks = seg_masks[keep, ...]
        seg_preds = seg_preds[keep, ...]
        sum_masks = sum_masks[keep]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]

        seg_scores = (seg_preds * seg_masks.float()).sum((1, 2)) / sum_masks
        cate_scores *= seg_scores

        sort_inds = torch.argsort(cate_scores, descending=True)
        if len(sort_inds) >nms_pre:
            sort_inds = sort_inds[:nms_pre]
        seg_masks = seg_masks[sort_inds, :, :]
        seg_preds = seg_preds[sort_inds, :, :]
        sum_masks = sum_masks[sort_inds]
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]

        cate_scores = matrix_nms(seg_masks, cate_labels, cate_scores,
                                    kernel=kernel,sigma=sigma, sum_masks=sum_masks)

        keep = cate_scores >= update_thr
        if keep.sum() == 0:
            return None
        seg_preds = seg_preds[keep, :, :]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]

        sort_inds = torch.argsort(cate_scores, descending=True)
        if len(sort_inds) > max_per_img:
            sort_inds = sort_inds[:max_per_img]
        seg_preds = seg_preds[sort_inds, :, :]
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]

        seg_preds = F.interpolate(seg_preds.unsqueeze(0),
                                    size=upsampled_size_out,
                                    mode='bilinear', align_corners=False)[:, :, :h, :w]
        seg_masks = F.interpolate(seg_preds,
                               size=(int(h/ratio), int(w/ratio)),
                               mode='bilinear',
                               align_corners=False).squeeze(0)
        seg_masks = seg_masks > mask_thr
        return seg_masks, cate_labels, cate_scores

class ICTransformer():
    def __init__(self, config, BACKBONE='res_18',  target_vocab_size=80, optimizer_name='adamW', scheduler_name='Cosine', amp_training=False, act='relu'):
        self.config = config
        self.optimizer_name = config.get('optimizer', optimizer_name)
        self.scheduler_name = config.get('scheduler', scheduler_name)
        self.BACKBONE = config.get('backbone', BACKBONE)
        self.target_vocab_size = int(config.get('target_vocab_size', target_vocab_size))
        self.NECK = config.get('neck', None)

        self.d_model = int(config.get('d_model', 512))
        self.num_heads = int(config.get('num_heads', 8))
        self.dff = int(config.get('dff', 1024))
        self.num_layers = int(config.get('num_layers', 2))
        
    def get_model(self):
        model = ImageCaptionTransformer(type_mode=self.BACKBONE, neck_mode=self.NECK, target_vocab_size=self.target_vocab_size,
                                                                                    d_model=self.d_model, num_heads=self.num_heads, dff=self.dff, num_layers=self.num_layers, pe=True, train_model=True)
        return model

    def get_inference_model(self):
        model = ImageCaptionTransformer(type_mode=self.BACKBONE, neck_mode=self.NECK, target_vocab_size=self.target_vocab_size,
                                                                                    d_model=self.d_model, num_heads=self.num_heads, dff=self.dff, num_layers=self.num_layers, pe=True, train_model=False)
        return model
    
    def get_optimizer(self, model):
        if self.optimizer_name == 'adam':
            optimizer = torch.optim.Adam([ 
                                        dict(params=model.parameters(), lr=self.config['lr']),
                                    ])
        elif self.optimizer_name == 'adamW':
            optimizer = torch.optim.AdamW(params=model.parameters(), lr=self.config['lr'])

        elif self.optimizer_name == 'SGD':
            optimizer =  torch.optim.SGD(params=model.parameters(), lr=self.config['lr'], momentum=0.9, weight_decay=0.0001)

        else:
            raise NotImplementedError('Optimizer {} not supported.'.format(self.optimizer_name))
        
        return optimizer
    
    def get_lr_scheduler(self, optimizer, max_iter):
        if self.scheduler_name == 'Cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iter,
                                                            eta_min=0.00001, last_epoch=-1, verbose=False)
        elif self.scheduler_name == 'OneCycleLR':
            return torch.optim.lr_scheduler.OneCycleLR(optimizer, self.config['lr'], total_steps=max_iter, verbose=False)

        else:
            return None
    
    def get_loss_func(self):
        return torch.nn.CrossEntropyLoss(ignore_index=int(self.config.get('pad_index', 1)), reduction='mean')
    
    def eval_model(self, model, data_loader, loss_fn, local_rank, distributed=False, is_main_process=True, nprocs=1, loss_criteria=True):
        eval_nums = len(data_loader)
        model.eval()
        eval_loss = 0.
        for inps, targets in data_loader:
            with torch.no_grad():
                inps = inps.cuda(local_rank, non_blocking=True)
                logic = model.forward(inps)
                loss = loss_fn(logic, targets)
                if distributed:
                    torch.distributed.barrier()
                    loss= reduce_mean(loss, nprocs if distributed else 1)
                eval_loss += loss.item()
        return eval_loss/eval_nums

class ReBiSeNet_Model():
    def __init__(self, config, BACKBONE='stdc_l', num_classes=19, optimizer_name='adamW', scheduler_name='Cosine', amp_training=False, act='relu'):
        self.config = config
        self.optimizer_name = config.get('optimizer', optimizer_name)
        self.scheduler_name = config.get('scheduler', scheduler_name)
        self.BACKBONE = config.get('backbone', BACKBONE)
        self.num_classes = config.get('num_classes', num_classes)
    def get_model(self):
        model = ReBiSeNet(type_mode=self.BACKBONE, num_classes=self.num_classes, train_model=True)
        return model

    def get_inference_model(self):
        model = ReBiSeNet(type_mode=self.BACKBONE, num_classes=self.num_classes, train_model=False)
        return model
    
    def get_optimizer(self, model):
        if self.optimizer_name == 'adam':
            optimizer = torch.optim.Adam([ 
                                        dict(params=model.parameters(), lr=self.config['lr']),
                                    ])
        elif self.optimizer_name == 'adamW':
            optimizer = torch.optim.AdamW(params=model.parameters(), lr=self.config['lr'])

        elif self.optimizer_name == 'SGD':
            optimizer =  torch.optim.SGD(params=model.parameters(), lr=self.config['lr'], momentum=0.9, weight_decay=0.0001)

        else:
            raise NotImplementedError('Optimizer {} not supported.'.format(self.optimizer_name))
        
        return optimizer
    
    def get_lr_scheduler(self, optimizer, max_iter):
        if self.scheduler_name == 'Cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iter,
                                                            eta_min=0.00001, last_epoch=-1, verbose=False)
        elif self.scheduler_name == 'OneCycleLR':
            return torch.optim.lr_scheduler.OneCycleLR(optimizer, self.config['lr'], total_steps=max_iter, verbose=False)
        else:
            return None
    
    def get_loss_func(self):
        W, H = eval(self.config['input_size'])
        img_per_gpu = self.config['batch_size']
        n_min = img_per_gpu*W*H//16
        return ReBiSeLoss(n_min, thresh=0.7, ignore_lb=255)
    
    def eval_model(self, model, data_loader, loss_fn, local_rank, distributed=False, is_main_process=True, nprocs=1, loss_criteria=True):
        eval_nums = len(data_loader)
        model.eval()
        eval_loss = 0.
        with torch.no_grad():
            if loss_criteria:
                for inps, targets in data_loader:
                    inps = inps.cuda(local_rank, non_blocking=True)
                    targets = targets.cuda(local_rank, non_blocking=True)
                    logic = model(inps)
                    loss = loss_fn(logic, targets)
                    if distributed:
                        torch.distributed.barrier()
                        loss= reduce_mean(loss, nprocs if distributed else 1)
                    eval_loss += loss.item()
                return eval_loss/eval_nums
            else:
                miou_50 = SemanticSegmIOU(scale=0.5, ignore_label=255)(model, data_loader, self.num_classes)
                return miou_50

class MaskFormer_Model():
    def __init__(self, config, BACKBONE='res_50', num_classes=171, optimizer_name='adamW', scheduler_name='Cosine', amp_training=False, act='relu'):
        self.config = config
        self.optimizer_name = config.get('optimizer', optimizer_name)
        self.scheduler_name = config.get('scheduler', scheduler_name)
        self.BACKBONE = config.get('backbone', BACKBONE)
        self.num_classes = config.get('num_classes', num_classes)

    def get_model(self):
        model = MaskFormer(type_mode=self.BACKBONE, num_classes=self.num_classes, train_model=True)
        return model

    def get_inference_model(self):
        model = MaskFormer(type_mode=self.BACKBONE, num_classes=self.num_classes, train_model=False)
        return model
    
    def get_optimizer(self, model):
        if self.optimizer_name == 'adam':
            optimizer = torch.optim.Adam([ 
                                        dict(params=model.parameters(), lr=self.config['lr']),
                                    ])
        elif self.optimizer_name == 'adamW':
            optimizer = torch.optim.AdamW(params=model.parameters(), lr=self.config['lr'])

        elif self.optimizer_name == 'SGD':
            optimizer =  torch.optim.SGD(params=model.parameters(), lr=self.config['lr'], momentum=0.9, weight_decay=0.0001)

        else:
            raise NotImplementedError('Optimizer {} not supported.'.format(self.optimizer_name))
        
        return optimizer
    
    def get_lr_scheduler(self, optimizer, max_iter):
        if self.scheduler_name == 'Cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iter,
                                                            eta_min=0.00001, last_epoch=-1, verbose=False)
        elif self.scheduler_name == 'OneCycleLR':
            return torch.optim.lr_scheduler.OneCycleLR(optimizer, self.config['lr'], total_steps=max_iter, verbose=False)

        else:
            return None
    
    def get_loss_func(self):
        dice_weight = 1.0
        mask_weight = 20.0
        no_object_weight = 0.1
        deep_supervision = True
        dec_layers = 6

        matcher = HungarianMatcher(
            cost_class=1,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
        )

        weight_dict = {"loss_ce": 1, "loss_mask": mask_weight, "loss_dice": dice_weight}
        if deep_supervision:
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses_component = ["labels", "masks"]

        return SetCriterion(
                            self.num_classes,
                            matcher=matcher,
                            weight_dict=weight_dict,
                            eos_coef=no_object_weight,
                            losses=losses_component,
                            )
    
    def eval_model(self, model, data_loader, loss_fn, local_rank, distributed=False, is_main_process=True, nprocs=1,  loss_criteria=True):
        eval_nums = len(data_loader)
        model.eval()
        eval_loss = 0.
        for inps, targets in data_loader:
            with torch.no_grad():

                if not isinstance(inps ,dict):
                    inps = inps.cuda(non_blocking=True)
                    inps.requires_grad = False
                elif isinstance(inps ,dict):
                    inps = togpu(inps)

                if not isinstance(targets ,dict) and not isinstance(targets, list):
                    targets = targets.cuda(non_blocking=True)
                    targets.requires_grad = False
                elif isinstance(targets ,dict):
                    targets = togpu(targets)

                logic = model(inps)
                loss = loss_fn(logic, targets)
                if distributed:
                    torch.distributed.barrier()
                    loss= reduce_mean(loss, nprocs if distributed else 1)
                eval_loss += loss.item()
        return eval_loss/eval_nums
