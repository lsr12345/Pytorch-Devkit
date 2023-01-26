'''
# Author: Shaoran Lu
# Date: 2021/10/04
# Email: lushaoran92@gmail.com
# Description: 

example:

'''

import cv2
import numpy as np
import pycocotools.mask as maskUtils
import torch

from data.boxes import xyxy2cxcywh

import math
import random

import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

from utils.standard_tools import  recursiveToTensor

cv2_interp_codes = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
    'area': cv2.INTER_AREA,
    'lanczos': cv2.INTER_LANCZOS4
}

def augment_hsv(img, hgain=0.015, sgain=0.7, vgain=0.4):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge(
        (cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))
    ).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)

def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.2):
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))
    return (
        (w2 > wh_thr)
        & (h2 > wh_thr)
        & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr)
        & (ar < ar_thr)
    )

def random_perspective(
    img,
    targets=(),
    degrees=10,
    translate=0.1,
    scale=0.1,
    shear=10,
    perspective=0.0,
    border=(0, 0),
):
    height = img.shape[0] + border[0] * 2
    width = img.shape[1] + border[1] * 2

    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2
    C[1, 2] = -img.shape[0] / 2

    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    s = random.uniform(scale[0], scale[1])
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)

    T = np.eye(3)
    T[0, 2] = (
        random.uniform(0.5 - translate, 0.5 + translate) * width
    )
    T[1, 2] = (
        random.uniform(0.5 - translate, 0.5 + translate) * height
    )

    M = T @ S @ R @ C


    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():
        if perspective:
            img = cv2.warpPerspective(
                img, M, dsize=(width, height), borderValue=(114, 114, 114)
            )
        else:
            img = cv2.warpAffine(
                img, M[:2], dsize=(width, height), borderValue=(114, 114, 114)
            )

    n = len(targets)
    if n:
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
            n * 4, 2
        )
        xy = xy @ M.T
        if perspective:
            xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)
        else:
            xy = xy[:, :2].reshape(n, 8)

        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)

        i = box_candidates(box1=targets[:, :4].T * s, box2=xy.T)
        targets = targets[i]
        targets[:, :4] = xy[i]

    return img, targets

def _distort(image):
    def _convert(image, alpha=1, beta=0):
        tmp = image.astype(float) * alpha + beta
        tmp[tmp < 0] = 0
        tmp[tmp > 255] = 255
        image[:] = tmp

    image = image.copy()

    if random.randrange(2):
        _convert(image, beta=random.uniform(-32, 32))

    if random.randrange(2):
        _convert(image, alpha=random.uniform(0.5, 1.5))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    if random.randrange(2):
        tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
        tmp %= 180
        image[:, :, 0] = tmp

    if random.randrange(2):
        _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image

def _mirror(image, boxes):
    _, width, _ = image.shape
    if random.randrange(2):
        image = image[:, ::-1]
        boxes = boxes.copy()
        boxes[:, 0::2] = width - boxes[:, 2::-2]
    return image, boxes

def preproc(image, input_size, mean, std, swap=(2, 0, 1), interpolation='bilinear'):
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
    else:
        padded_img = np.ones(input_size) * 114.0
    img = np.array(image)
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2_interp_codes[interpolation],
    ).astype(np.float32)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img[:, :, ::-1]
    padded_img /= 255.0
    if mean is not None:
        padded_img -= mean
    if std is not None:
        padded_img /= std
    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r

def pad_fill(image, pad_size, pad_value=0):
    mask = torch.full(pad_size, pad_value, dtype=image.dtype)
    mask[:image.shape[0], :image.shape[1]] = mask
    return mask

class Collater_BaseFunc():
    def __init__(self, *params):
        self.params = params

    def __call__(self, batch_data):
        batch_data = recursiveToTensor(batch_data)
        return_batch = dict()
        for key in batch_data[0].keys():
            return_batch[key] = [x[key] for x in batch_data]
        return return_batch

class Collater_Solo():
    def __init__(self, *params):
        self.params = params

    def __call__(self, batch_data):
        batch_imgs = []
        batch_gt_bboxes = []
        batch_labels = []
        batch_gt_masks = []
            
        for data in batch_data:
            if len(data) > 2:
                img = data[0]
                target = data[1]
            else:
                img, target = data
            batch_imgs.append(img)

            batch_gt_bboxes.append(torch.from_numpy(target['bboxes']))
            batch_labels.append(torch.from_numpy(target['labels']))
            batch_gt_masks.append(torch.from_numpy(target['masks']))

        target = dict(
            batch_bboxes=batch_gt_bboxes,
            batch_labels=batch_labels,
            batch_masks=batch_gt_masks
        )
            
        return torch.from_numpy(np.stack(batch_imgs)), target

class Collater_ICT():
    def __init__(self, *params):
        self.params = params

    def __call__(self, batch_data):
        batch_images = []
        batch_input_labels = []
        batch_labels = []

            
        for data in batch_data:
            img, label = data
            batch_images.append(img)
            batch_input_labels.append(label[:-1])
            batch_labels.append(label[1:])

        batch_images =  torch.from_numpy(np.stack(batch_images))
        batch_input_labels = torch.from_numpy(np.stack(batch_input_labels))
        batch_labels = torch.from_numpy(np.stack(batch_labels))
        inputs = dict(
            batch_images = batch_images,
            batch_input_labels = batch_input_labels
        )
        return inputs, batch_labels.contiguous().view(-1)

class Collater_Maskformer():
    def __init__(self, *params):
        self.params = params

    def __call__(self, batch_data):
        batch_imgs = []
        batch_target = []
            
        for data in batch_data:
            if len(data) > 2:
                img = data[0]
                target = data[1]
            else:
                img, target = data
            batch_imgs.append(img)

            labels = torch.from_numpy(target['labels'])
            masks = torch.from_numpy(target['masks'])
            target = dict(
                labels=labels,
                masks=masks
                )
            
            batch_target.append(target)  
        return torch.from_numpy(np.stack(batch_imgs)), batch_target

class TrainTransform:
    def __init__(self, p=0.5, rgb_means=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_labels=50):
        self.means = rgb_means
        self.std = std
        self.p = p
        self.max_labels = max_labels

    def __call__(self, image, targets, input_dim, augmentation=None):
        if len(targets) == 0:
            targets = np.zeros((self.max_labels, 5), dtype=np.float32)
            image, r_o = preproc(image, input_dim, self.means, self.std)
            image = np.ascontiguousarray(image, dtype=np.float32)
            return image, targets, r_o

        if augmentation is not None:
            image, targets = augmentation.make_aug(image, targets)
            targets = np.array(targets)
            if targets.shape[0] == 0:
                targets = np.zeros((self.max_labels, 5), dtype=np.float32)
                image, r_o = preproc(image, input_dim, self.means, self.std)
                image = np.ascontiguousarray(image, dtype=np.float32)
                return image, targets, r_o

        boxes = targets[:, :4].copy()
        labels = targets[:, 4].copy()

        image_o = image.copy()
        targets_o = targets.copy()
        height_o, width_o, _ = image_o.shape
        boxes_o = targets_o[:, :4]
        labels_o = targets_o[:, 4]
        boxes_o = xyxy2cxcywh(boxes_o)

        image_t = _distort(image)
        height, width, _ = image_t.shape
        image_t, r_ = preproc(image_t, input_dim, self.means, self.std)
        boxes = xyxy2cxcywh(boxes)
        boxes *= r_

        mask_b = np.minimum(boxes[:, 2], boxes[:, 3]) > 8
        boxes_t = boxes[mask_b]
        labels_t = labels[mask_b]

        if len(boxes_t) == 0:
            image_t, r_o = preproc(image_o, input_dim, self.means, self.std)
            boxes_o *= r_o
            boxes_t = boxes_o
            labels_t = labels_o

        labels_t = np.expand_dims(labels_t, 1)

        targets_t = np.hstack((boxes_t, labels_t))
        padded_labels = np.zeros((self.max_labels, 5))
        padded_labels[range(len(targets_t))[: self.max_labels]] = targets_t[
            : self.max_labels
        ]
        padded_labels = np.ascontiguousarray(padded_labels, dtype=np.float32)
        image_t = np.ascontiguousarray(image_t, dtype=np.float32)
        return image_t, padded_labels, r_

class TrainTransform_Instance:
    def __init__(self, rgb_means=0., std=1., keep_ratio=True,  divisor=32, with_box=True):
        self.means = rgb_means
        self.std = std
        self.keep_ratio = keep_ratio
        self.divisor = divisor
        self.with_box = with_box

    def __call__(self, image, targets, input_dim, augmentation=None):
 
        bboxes = targets['bboxes']
        labels = targets['labels']
        bboxes_ignore = targets['bboxes_ignore']
        masks = targets['masks']
        seg_map = targets['seg_map']
        h, w, _ = image.shape

        gt_masks = [self._poly2mask(mask, h, w) for mask in masks]
        gt_masks = np.stack(gt_masks)
            
        if augmentation is not None and len(gt_masks) > 0:
            gt_bboxes_ = []
            labels_ = []
            gt_masks_ = []
            if self.with_box:
                _gt_bboxes = bboxes.copy()
                _gt_bboxes = _gt_bboxes.tolist()
                for i in range(len(_gt_bboxes)):
                    _gt_bboxes[i].append(int(labels[i]))
                _gt_bboxes = np.array(_gt_bboxes)

                _gt_masks = np.transpose(gt_masks, axes=(1,2,0))
                image_, _gt_masks, _gt_bboxes = augmentation.make_aug(image, _gt_masks, box_label=_gt_bboxes)
                _gt_masks = np.transpose(_gt_masks, axes=(2,0,1))

            else:
                _gt_masks = np.transpose(gt_masks, axes=(1,2,0))
                image_, _gt_masks = augmentation.make_aug(image, _gt_masks)
                _gt_masks = np.transpose(_gt_masks, axes=(2,0,1))

            for i, gt_mask in enumerate(_gt_masks):
                if gt_mask.sum() > 20:
                    if self.with_box:
                        gt_bboxes_.append(_gt_bboxes[i][:4])
                    else:
                        gt_bboxes_.append(None)
                    labels_.append(labels[i])
                    gt_masks_.append(_gt_masks[i])

            if len(gt_masks_) > 0:
                bboxes = gt_bboxes_
                labels = labels_
                gt_masks = gt_masks_
                gt_masks = np.stack(gt_masks)
                image = image_

        input_d = input_dim
        if self.keep_ratio:
            assert isinstance(input_d, tuple)
            assert input_d[0] % self.divisor ==0
            assert input_d[1] % self.divisor ==0

            resized_img, scale_factor = preproc(image, input_dim, self.means, self.std, interpolation='bilinear')
            img_shape = resized_img.shape
            if bboxes is not None:
                gt_bboxes = np.array(bboxes) * scale_factor
                gt_bboxes[:, 0::2] = np.clip(gt_bboxes[:, 0::2], 0, img_shape[2] - 1)
                gt_bboxes[:, 1::2] = np.clip(gt_bboxes[:, 1::2], 0, img_shape[1] - 1)
            else:
                gt_bboxes = [None]

            new_size = (int(w * float(scale_factor)), int(h * float(scale_factor)))
            gt_masks = [self._mask_resize(mask, new_size, input_dim) for mask in gt_masks]
            if len(gt_masks) > 0:
                gt_masks = np.stack(gt_masks)
            else:
                print('error mask')
                gt_masks =np.zeros((0, 4), dtype=np.uint8)


            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)





        new_targets = dict(
                                                bboxes=gt_bboxes,
                                                labels=labels,
                                                masks=gt_masks
                                                )

        return resized_img, new_targets, scale_factor

    def _poly2mask(self, mask_ann, img_h, img_w):
        if isinstance(mask_ann, list):
            rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
            rle = maskUtils.merge(rles)
        else:
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask

    def _mask_resize(self, mask, size, mask_size):
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
        pad_mask = np.full(mask_size,  fill_value=0, dtype=np.uint8)
        pad_mask[:mask.shape[0], :mask.shape[1]] = mask
        return pad_mask

class ValTransform:
    """
    Defines the transformations that should be applied to test PIL image
    for input into the network

    dimension -> tensorize -> color adj

    Arguments:
        resize (int): input dimension to SSD
        rgb_means ((int,int,int)): average RGB of the dataset
            (104,117,123)
        swap ((int,int,int)): final order of channels

    Returns:
        transform (transform) : callable transform to be applied to test/val
        data
    """

    def __init__(self, rgb_means=None, std=None, swap=(2, 0, 1)):
        self.means = rgb_means
        self.swap = swap
        self.std = std

    def __call__(self, img, res, input_size):
        img, _ = preproc(img, input_size, self.means, self.std, self.swap)
        return img, np.zeros((1, 5))
