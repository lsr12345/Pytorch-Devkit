'''
# Author: Shaoran Lu
# Date: 2021/10/04
# Email: lushaoran92@gmail.com
# Description: 

example:

'''

from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import json
import tempfile

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from tools.boxes import xyxy2xywh

class Coco_eval():
    def __init__(self,  eval_bbox=False, eval_mask=False, jsonfile=None):
        self.eval_bbox = eval_bbox
        self.eval_mask = eval_mask
        self.jsonfile = jsonfile
        
    def __call__(self, data_list, ann_file):
        if self.jsonfile is not None:
            json.dump(data_list, open(self.jsonfile, "w"))
        else:
            _, self.jsonfile = tempfile.mkstemp()
            json.dump(data_list, open(self.jsonfile, "w"))
        print('Loading annotations...')
        gt_annotations = COCO(ann_file)
        test_res = gt_annotations.loadRes(self.jsonfile)

        if self.eval_bbox:
            print('\nEvaluating BBoxes:')
            bbox_eval = COCOeval(gt_annotations, test_res, 'bbox')
            bbox_eval.evaluate()
            bbox_eval.accumulate()
            bbox_eval.summarize()

        if self.eval_mask:
            print('\nEvaluating Masks:')
            bbox_eval = COCOeval(gt_annotations, test_res, 'segm')
            bbox_eval.evaluate()
            bbox_eval.accumulate()
            bbox_eval.summarize()

        return bbox_eval.stats[0]

class ConvertCocoFormat():

    def __init__(self, id2cat, mode='bbox'):
        self.id2cat = id2cat
        self.mode = mode

    def __call__(self, b_bboxes, b_cls, b_scores,  ids):
        data_list = []
        for (bboxes, cls, scores, img_id) in zip(b_bboxes, b_cls, b_scores, ids):
            bboxes, cls, scores = bboxes.cpu(), cls.cpu(), scores.cpu()

            if bboxes is None:
                continue
            if self.mode == 'bbox':
                for ind in range(bboxes.shape[0]):
                    label = self.id2cat[int(cls[ind])]
                    pred_data = {
                        "image_id": int(img_id.numpy().item()),
                        "category_id": label,
                        "bbox": bboxes[ind].numpy().tolist(),
                        "score": scores[ind].numpy().item(),
                        "segmentation": [],
                    }
                    data_list.append(pred_data)
            else:
                raise NotImplementedError
        return data_list
    
class SemanticSegmIOU():
    def __init__(self, scale=0.5, ignore_label=255):
        super().__init__()
        self.scale = scale
        self.ignore_label = ignore_label

    def __call__(self, model, dataset, n_classes):
        hist = torch.zeros(n_classes, n_classes).cuda().detach()

        for inps, targets in tqdm(dataset):

            N, H, W = targets.shape
            targets = targets.cuda()
            size = targets.size()[-2:]

            inps = inps.cuda()
            N, C, H, W = inps.size()

            new_hw = [int(H*self.scale), int(W*self.scale)]
            inps = F.interpolate(inps, new_hw, mode='bilinear', align_corners=True)

            logits = model(inps)[0]
            logits = F.interpolate(logits, size=size, mode='bilinear', align_corners=True)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            keep = targets != self.ignore_label

            hist += torch.bincount(targets[keep] * n_classes + preds[keep], minlength=n_classes ** 2).view(n_classes, n_classes).float()
            
        ious = hist.diag() / (hist.sum(dim=0) + hist.sum(dim=1) - hist.diag())
        miou = ious.mean()
        return miou.item()