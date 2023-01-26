'''
# Author: Shaoran Lu
# Date: 2021/10/04
# Email: lushaoran92@gmail.com
# Description: 

example:

'''


from torch.utils.data import Dataset
import torchvision.transforms as transforms

import cv2
import h5py
import numpy as np
import random
import math
import json
import pyclipper
from shapely.geometry import Polygon

from pycocotools.coco import COCO
from data.coco.coco_classes import COCO_CLASSES, COCO_LABEL, COCO_LABEL_MAP
from data.coco.coco_stuff_10k_classes import COCO_STUFF_10k_CATEGORIES

from loguru import logger
import os

from data_utils import Collater_Solo, Collater_ICT, Collater_Maskformer


class CustomDataset(Dataset):
    
    def __init__(
            self, 
            lists,
            shape=(512,512),
            augmentation=None,
            dtype='train'
    ):
        super().__init__()
        images_dir, labels_dir = lists
        self.img_ids = os.listdir(images_dir)
        self.lb_ids = os.listdir(labels_dir)
        self.dtype = dtype

        assert len(self.img_ids) == len(self.lb_ids), 'Check datasets'
        check_id = random.randint(0, len(self.img_ids)-1)
        assert os.path.splitext(self.img_ids[check_id])[0] == os.path.splitext(self.lb_ids[check_id])[0], 'Check datasets'
        
        self.images_pt = [os.path.join(images_dir, image_id) for image_id in self.img_ids]
        self.labels_pt = [os.path.join(labels_dir, label_id) for label_id in self.lb_ids]
        
        self.augmentation = augmentation
        self.shape = shape
    
    def __getitem__(self, i):
        
        image = cv2.imread(self.images_pt[i])
        label = self.read_label(self.labels_pt[i])
        image = self.resize(image, label, self.shape)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.augmentation:
            image, label = self.augmentation.make_aug(image, label)

        image = self.preprocessing(image)
        image = np.ascontiguousarray(image)

        return image, label
        
    def __len__(self):
        return len(self.lb_ids)
    
    def to_tensor(self, x, **kwargs):
        return x.transpose(2, 0, 1).astype('float32')

    def preprocessing(self, image):
        return self.to_tensor(image)
    
    def read_label(self, label_path):
        with open (label_path, mode='r', encoding='UTF-8') as fr:
            ff = fr.readlines()
            for line in ff:
                line = line.strip()
        return None
    
    def resize(self, image, label, size):
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        return image, label

class CustomDataset_seg(Dataset):
    CLASSES = ['visible_row', 'visible_column', 'unvisible_row', 'unvisible_column']
    
    def __init__(
            self, 
            lists,
            shape=(512,512),
            classes=CLASSES, 
            augmentation=None,
            dtype='train'
    ):
        super().__init__()
        self.dtype = dtype
        images_dir, masks_vr_dir, masks_vc_dir, masks_ur_dir, masks_uc_dir = lists
        self.ids = os.listdir(images_dir)
        logger.info("Dataset samples: {}".format(len(self.ids)))
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_vr_fps = [os.path.join(masks_vr_dir, image_id) for image_id in self.ids]
        self.masks_vc_fps = [os.path.join(masks_vc_dir, image_id) for image_id in self.ids]
        self.masks_ur_fps = [os.path.join(masks_ur_dir, image_id) for image_id in self.ids]
        self.masks_uc_fps = [os.path.join(masks_uc_dir, image_id) for image_id in self.ids]

        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        
        self.shape = shape
    
    def __getitem__(self, i):
        image = cv2.imread(self.images_fps[i])
        image = cv2.resize(image, dsize=self.shape, interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        masks_vr = cv2.imread(self.masks_vr_fps[i], 0)
        masks_vr = cv2.resize(masks_vr, dsize=self.shape, interpolation=cv2.INTER_AREA)
        masks_vr = masks_vr==0
        
        masks_vc = cv2.imread(self.masks_vc_fps[i], 0)
        masks_vc = cv2.resize(masks_vc, dsize=self.shape, interpolation=cv2.INTER_AREA)
        masks_vc = masks_vc==0
        
        masks_ur = cv2.imread(self.masks_ur_fps[i], 0)
        masks_ur = cv2.resize(masks_ur, dsize=self.shape, interpolation=cv2.INTER_AREA)
        masks_ur = masks_ur==0
        
        masks_uc = cv2.imread(self.masks_uc_fps[i], 0)
        masks_uc = cv2.resize(masks_uc, dsize=self.shape, interpolation=cv2.INTER_AREA)
        masks_uc = masks_uc==0
        
        masks_bg = 1 - masks_vr|masks_vc|masks_ur|masks_uc

        mask = np.stack([masks_vr, masks_vc, masks_ur, masks_uc, masks_bg], axis=-1).astype('bool')

        if self.augmentation:
            image, mask = self.augmentation.make_aug(image, mask)

        image, mask = self.preprocessing(image, mask)
        image = np.ascontiguousarray(image)
        mask = np.ascontiguousarray(mask)
        return image, mask
        
    def __len__(self):
        return len(self.ids)
    
    def to_tensor(self, x, **kwargs):
        return x.transpose(2, 0, 1).astype('float32')

    def preprocessing(self, image, mask):
        return self.to_tensor(image), self.to_tensor(mask)

    def get_collate_fn(self):
        return None

class OcrDetDataset(Dataset):
    
    def __init__(
            self, 
            lists,
            shape=(960,960),
            augmentation=None,
            min_text_size=8, 
            shrink_ratio=0.4,
            thresh_min=0.3,
             thresh_max=0.7,
             mean=[0.,0.,0.],
             std=[1.0,1.0,1.0],
             epsilon = 1e-9,
             dtype='train'
    ):
        super().__init__()
        images_dir, labels_dir = lists
        self.img_ids = os.listdir(images_dir)
        self.lb_ids = os.listdir(labels_dir)
        self.dtype = dtype
        assert len(self.img_ids) == len(self.lb_ids), 'Check datasets'
        
        self.images_pt = []
        self.labels_pt = []
        for n in self.lb_ids:
            if os.path.exists(os.path.join(images_dir, os.path.splitext(n)[0] + '.jpg')):
                self.images_pt.append(os.path.join(images_dir, os.path.splitext(n)[0] + '.jpg'))
                self.labels_pt.append(os.path.join(labels_dir, n))

        
        self.augmentation = augmentation
        self.shape = shape
        self.min_text_size = min_text_size
        self.shrink_ratio = shrink_ratio
        self.thresh_min = thresh_min
        self.thresh_max = thresh_max
        self.mean = mean
        self.std = std
        self.epsilon = epsilon
    
    def __getitem__(self, i):
        
        image = cv2.imread(self.images_pt[i])
        label = self.read_label(self.labels_pt[i])

        image, label = self.resize(self.shape[0], image, label)
        image = image.astype(np.uint8)
        
        if self.augmentation is not None:
            bboxes = []
            for ann in label:
                bboxes.append([ann['poly'][0], ann['poly'][1],
                            ann['poly'][2], ann['poly'][3], ann['text']])

            image, bboxes = self.augmentation.make_aug(image, bboxes)

            label = []
            for i, box in enumerate(bboxes):
                label.append({'poly': box[:-1], 'text': box[-1]})

        image, mask = self.preprocessing(image, label)
        image = np.ascontiguousarray(image)
        mask = np.ascontiguousarray(mask)

        return image, mask
        
    def __len__(self):
        return len(self.labels_pt)
    
    def to_tensor(self, x, **kwargs):
        return x.transpose(2, 0, 1).astype('float32')

    def preprocessing(self, image, label):
        label_ = []
        for lb in label:
            try:
                if Polygon(lb['poly']).is_valid:
                    label_.append(lb)
            except:
                print('*')
                continue
        label = label_

        gt = np.zeros(self.shape, dtype=np.float32)
        mask = np.ones(self.shape, dtype=np.float32)
        thresh_map = np.zeros(self.shape, dtype=np.float32)
        thresh_mask = np.zeros(self.shape, dtype=np.float32)
        for ann in label:
            poly = np.array(ann['poly'])
            height = max(poly[:, 1]) - min(poly[:, 1])
            width = max(poly[:, 0]) - min(poly[:, 0])
            polygon = Polygon(poly)
            if polygon.area < 1 or min(height, width) < self.min_text_size or ann['text'] == '###':
                cv2.fillPoly(mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
                continue
            else:
                distance = polygon.area * (1 - np.power(self.shrink_ratio, 2)) / polygon.length
                subject = [tuple(l) for l in ann['poly']]
                padding = pyclipper.PyclipperOffset()
                padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
                shrinked = padding.Execute(-distance)
                if len(shrinked) == 0:
                    cv2.fillPoly(mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
                    continue
                else:
                    shrinked = np.array(shrinked[0]).reshape(-1, 2)
                    if shrinked.shape[0] > 2 and Polygon(shrinked).is_valid:
                        cv2.fillPoly(gt, [shrinked.astype(np.int32)], 1)
                    else:
                        cv2.fillPoly(mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
                        continue
            try:
                self.draw_thresh_map(ann['poly'], thresh_map, thresh_mask, shrink_ratio=self.shrink_ratio)
            except:
                continue
        thresh_map = thresh_map * (self.thresh_max - self.thresh_min) + self.thresh_min
        gt = np.expand_dims(gt, axis=-1)
        mask = np.expand_dims(mask, axis=-1)
        thresh_map = np.expand_dims(thresh_map, axis=-1)
        thresh_mask = np.expand_dims(thresh_mask, axis=-1)
        output = np.concatenate((gt, mask, thresh_map, thresh_mask), axis=-1)


        image = (image - self.mean) / self.std
        return self.to_tensor(image),  self.to_tensor(output)
    
    def read_label(self, label_path):
        lines = []
        with open(label_path, mode='r', encoding='UTF-8') as fr:
            reader = fr.readlines()
            for line in reader:
                item = {}
                line = line.strip().split(',')
                label = line[-1]
                try:
                    i = [round(float(ii)) for ii in line[:8]]
                    poly = np.array(i).reshape((-1, 2))
                    poly = self.reorder_vertexes(poly)
                except:
                    print('wrong poly:', label_path)
                    continue
                item['poly'] = poly.astype(np.int32).tolist()
                item['text'] = label
                lines.append(item)
        return lines

    def reorder_vertexes(self, pts):
        rect = np.zeros((4, 2), dtype = "float32")

        s = pts.sum(axis = 1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis = 1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect
    
    def resize(self, size, image, anns):
        h, w, c = image.shape
        padimg = np.full(shape=(size,size,c), fill_value=255, dtype=float)
        if max(h, w) <= size:
            scale = 1
            padimg[:h, :w, :] = image
        else:
            scale_w = size / w
            scale_h = size / h
            scale = min(scale_w, scale_h)
            h = round(float(h * scale))
            w = round(float(w * scale))
            padimg[:h, :w, :] = cv2.resize(image, (w, h))
            
        new_anns = []
        for ann in anns:
            poly = np.array(ann['poly']).astype(np.float64)
            poly *= scale
            poly = self.reorder_vertexes(poly)
            new_ann = {'poly': poly.tolist(), 'text': ann['text']}
            new_anns.append(new_ann)
        return padimg, new_anns

    def draw_thresh_map(self, polygon, canvas, mask, shrink_ratio=0.4):
        polygon = np.array(polygon)
        assert polygon.ndim == 2
        assert polygon.shape[1] == 2

        polygon_shape = Polygon(polygon)
        distance = polygon_shape.area * (1 - np.power(shrink_ratio, 2)) / polygon_shape.length
        subject = [tuple(l) for l in polygon]
        padding = pyclipper.PyclipperOffset()
        padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        padded_polygon = np.array(padding.Execute(distance)[0])
        cv2.fillPoly(mask, [padded_polygon.astype(np.int32)], 1.0)

        xmin = padded_polygon[:, 0].min()
        xmax = padded_polygon[:, 0].max()
        ymin = padded_polygon[:, 1].min()
        ymax = padded_polygon[:, 1].max()
        width = xmax - xmin + 1
        height = ymax - ymin + 1

        polygon[:, 0] = polygon[:, 0] - xmin
        polygon[:, 1] = polygon[:, 1] - ymin

        xs = np.broadcast_to(np.linspace(0, width - 1, num=width).reshape(1, width), (height, width))
        ys = np.broadcast_to(np.linspace(0, height - 1, num=height).reshape(height, 1), (height, width))

        distance_map = np.zeros((polygon.shape[0], height, width), dtype=np.float32)
        for i in range(polygon.shape[0]):
            j = (i + 1) % polygon.shape[0]
            absolute_distance = self.compute_distance(xs, ys, polygon[i], polygon[j])
            distance_map[i] = np.clip(absolute_distance / distance, 0, 1)
        distance_map = np.min(distance_map, axis=0)

        xmin_valid = min(max(0, xmin), canvas.shape[1] - 1)
        xmax_valid = min(max(0, xmax), canvas.shape[1] - 1)
        ymin_valid = min(max(0, ymin), canvas.shape[0] - 1)
        ymax_valid = min(max(0, ymax), canvas.shape[0] - 1)
        canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1] = np.fmax(
            1 - distance_map[
                ymin_valid - ymin:ymax_valid - ymax + height,
                xmin_valid - xmin:xmax_valid - xmax + width],
            canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1])

    def compute_distance(self, xs, ys, point_1, point_2):
        square_distance_1 = np.square(xs - point_1[0]) + np.square(ys - point_1[1])
        square_distance_2 = np.square(xs - point_2[0]) + np.square(ys - point_2[1])
        square_distance = np.square(point_1[0] - point_2[0]) + np.square(point_1[1] - point_2[1])
        cosin = (square_distance - square_distance_1 - square_distance_2) / (2 * np.sqrt(square_distance_1 * square_distance_2) + self.epsilon)
        square_sin = 1 - np.square(cosin)
        square_sin = np.nan_to_num(square_sin)
        result = np.sqrt(square_distance_1 * square_distance_2 * square_sin / (square_distance + self.epsilon))

        result[cosin < 0] = np.sqrt(np.fmin(square_distance_1, square_distance_2))[cosin < 0]
        return result

    def get_collate_fn(self):
        return None

class OcrRecDataset(Dataset):
    
    def __init__(
            self, 
            lists,
            shape=(32,320),
            num_channel=1,
            augmentation=None,
             mean=[0.,0.,0.],
             std=[1.0,1.0,1.0],
             max_length=20,
             dtype='train'
    ):
        super().__init__()
        self.num_channel = num_channel
        images_dir, labels_dir, chars_file = lists

        self.char2id = self.read_charsFile(chars_file)
        self.img_ids = os.listdir(images_dir)
        self.lables = self.read_labelFile(labels_dir)
        self.max_length = max_length
        self.dtype = dtype
        images_ = []
        labels_ = []
        for n in self.lables:
            if os.path.exists(os.path.join(images_dir, n[0])):
                images_.append(os.path.join(images_dir, n[0]))
                labels_.append(n[1])
        assert len(labels_) == len(images_)
        assert len(labels_) > 0
        self.images = images_
        self.lables = labels_

        self.augmentation = augmentation
        self.shape = shape
        self.mean = mean
        self.std = std
    
    def __getitem__(self, i):
        
        image = cv2.imread(self.images[i])
        label = self.lables[i]
        label = np.array([int(self.char2id[char]) for char in label])

        image = self.resize(image, size=self.shape, hsize=self.shape[0], wsize=None)
        image = image.astype(np.uint8)
        
        if self.augmentation is not None:
            image, label = self.augmentation.make_aug(image, label)

        image, label = self.preprocessing(image, label)
        label_mask = np.full(self.max_length, fill_value=0,dtype=int)
    
        input_length = self.shape[1] // 8
        target_length = len(label)
        label_mask[:target_length] = label
        total_label  = np.concatenate([[input_length], [target_length], label_mask], axis=-1)
        image = np.ascontiguousarray(image)
        total_label = np.ascontiguousarray(total_label)

        return image, total_label
        
    def __len__(self):
        return len(self.lables)

    def read_charsFile(self, chars_file):
        char_list = []
        with open(chars_file, 'r', encoding='UTF-8') as f:
            ff = f.readlines()
            for i, char in enumerate(ff):
                char = char.strip()
                char_list.append(char)
        char2id = {j:i for i, j in enumerate(char_list)}
        return char2id

    def read_labelFile(self, labe_file):
        label_lists = []
        with open(labe_file, mode='r', encoding='utf-8') as fr:
            file = fr.readlines()
            for line in file:
                line = line.strip()
                line = line.split(' ')
                label_lists.append(line)
        return label_lists
    
    def to_tensor(self, x, **kwargs):
        return x.transpose(2, 0, 1).astype('float32')

    def resize(self, image, size=(32, 320), hsize=32, wsize=None):
        assert hsize <= size[0]
        h, w, c = image.shape
        hscale = hsize / h
        if wsize is None:
            wsize = int(math.ceil(w * hscale / 16) * 16)
            if wsize > size[1]:
                wsize = size[1]
        image = cv2.resize(image, (wsize, hsize))
        padimg = np.full(shape=(size[0], size[1], 3), fill_value=255, dtype=float)

        padimg[:hsize, :wsize, :] = image

        return padimg

    def preprocessing(self, image, label):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.expand_dims(image, axis=-1)
        return self.to_tensor(image), label

    def get_collate_fn(self):
        return None

class CocoDataset(Dataset):
    def __init__(self,  data_dir, json_file="instances_train2017.json", name="train2017",  shape=(416, 416), augmentation=None, preproc=None, mode='det', dtype='train'):
        super().__init__()
        self.data_dir = data_dir
        self.json_file = json_file
        self.mode = mode
        self.dtype = dtype

        self.coco = COCO(os.path.join(self.data_dir, "annotations", self.json_file))
        self.ids = self.coco.getImgIds()
        self.class_ids = sorted(self.coco.getCatIds())

        self.id2cat = {i:j for i, j in enumerate(self.class_ids)}

        cats = self.coco.loadCats(self.coco.getCatIds())
        self._classes = tuple([c["name"] for c in cats])
        self.annotations = self._load_coco_annotations()
        self.name = name
        self.shape = shape
        self.augmentation = augmentation
        self.preproc = preproc

        valid_inds = self._filter_imgs()
        self.annotations = [self.annotations[i] for i in valid_inds]
        self.ids = [self.ids[i] for i in valid_inds]

    def __len__(self):
        return len(self.annotations)

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for i, anno in enumerate(self.annotations):
            if self.ids[i] not in ids_with_ann:
                continue
            if min(anno[1][0], anno[1][1]) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _load_coco_annotations(self):
        if self.mode == 'det':
            return [self.load_anno_from_ids(_ids) for _ids in self.ids]
        elif self.mode == 'instance':
            return [self.load_anno_from_ids_instance(_ids) for _ids in self.ids]
        else:
            raise NotImplementedError

    def load_anno_from_ids(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)
        objs = []
        for obj in annotations:
            x1 = np.max((0, obj["bbox"][0]))
            y1 = np.max((0, obj["bbox"][1]))
            x2 = np.min((width, x1 + np.max((0, obj["bbox"][2]))))
            y2 = np.min((height, y1 + np.max((0, obj["bbox"][3]))))
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)

        num_objs = len(objs)
        res = np.zeros((num_objs, 5))

        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls

        img_info = [height, width]
        file_name = im_ann["file_name"]
        del im_ann, annotations
        return (res, img_info, file_name)

    def load_anno_from_ids_instance(self, id_):
        info = self.coco.loadImgs([id_])[0]
        height = info["height"]
        width = info["width"]
        img_info = [height, width]
        file_name =  info["file_name"]
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)

        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []

        for i, ann in enumerate(annotations):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(COCO_LABEL_MAP[ann["category_id"]])
                gt_masks_ann.append(ann['segmentation'])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = info['file_name'].replace('jpg', 'png')
        
        res = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)
        
        return (res, img_info, file_name)

    def pull_item(self, index):
        id_ = self.ids[index]

        res, img_info, file_name = self.annotations[index]
        img_file = os.path.join(self.data_dir, self.name, file_name)
        img = cv2.imread(img_file)
        assert img is not None
        return img, res.copy(),  np.array(img_info), np.array([id_])

    def __getitem__(self, index):
        img, target, img_info, img_id = self.pull_item(index)


        if self.preproc is not None:
            img, target, scale_factor = self.preproc(img, target, self.shape, self.augmentation)
        else:
            scale_factor = 1.
        if self.dtype == 'train':
            return img, target
        else:
            return img, target, img_info, img_id, np.array([scale_factor])

    def get_collate_fn(self):
        return Collater_Solo() if self.mode == 'instance' else None

class CityscapesDataset(Dataset):
    def __init__(self,  data_dir, json_file="./cityscapes_info.json",  shape=(416, 416), augmentation=None,  name='train', 
                                dtype='train', ignore_lb=255, scales=(0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 0.875, 1.0, 1.25, 1.5)):
        super().__init__()

        self.to_tensor = transforms.Compose([
            transforms.ToTensor()
            ])

        self.ignore_lb = ignore_lb
        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.augmentation = augmentation
        self.scales = scales

        if json_file is not None:
            self.lb_map = self.read_label_map(json_file)
        else:
            self.lb_map = None
        
        self.imgs = {}
        imgnames = []
        impth = os.path.join(data_dir, 'leftImg8bit', self.name)
        folders = os.listdir(impth)
        for fd in folders:
            fdpth = os.path.join(impth, fd)
            im_names = os.listdir(fdpth)
            names = [el.replace('_leftImg8bit.png', '') for el in im_names]
            impths = [os.path.join(fdpth, el) for el in im_names]
            imgnames.extend(names)
            self.imgs.update(dict(zip(names, impths)))

        self.imnames = imgnames

        self.labels = {}
        gtpth = os.path.join(data_dir, 'gtFine', self.name)
        folders = os.listdir(gtpth)
        for fd in folders:
            fdpth = os.path.join(gtpth, fd)
            lbnames = os.listdir(fdpth)
            lbnames = [el for el in lbnames if 'labelIds' in el]
            names = [el.replace('_gtFine_labelIds.png', '') for el in lbnames]
            lbpths = [os.path.join(fdpth, el) for el in lbnames]
            self.labels.update(dict(zip(names, lbpths)))

        assert set(self.imnames) == set(self.imgs.keys())
        assert set(self.imnames) == set(self.labels.keys())        
        
    def __len__(self):
        return len(self.imnames)

    def read_label_map(self, file_path):
        with open(file_path, 'r') as fr:
            labels_info = json.load(fr)
        return  {el['id']: el['trainId'] for el in labels_info}

    def __getitem__(self, index):
        fn  = self.imnames[index]
        impth = self.imgs[fn]
        lbpth = self.labels[fn]
        img = cv2.imread(impth)
        label = cv2.imread(lbpth, -1)
        if self.dtype == 'train':
            img, label = self.random_scale(img, label)
            img, label = self.random_crop(img, label, crop_size=self.shape)
            if self.augmentation is not None:
                img, label = self.augmentation.make_aug(img, label)

        label = np.array(label).astype(np.int64)
        label = self.convert_labels(label)

        img = self.to_tensor(img)
        return img, label

    def convert_labels(self, label):
        if self.lb_map is not None:
            for k, v in self.lb_map.items():
                label[label == k] = v
        return label

    def random_scale(self, image, label):
        H, W, _ = image.shape
        scale = random.choice(self.scales)
        w, h = int(W * scale), int(H * scale)
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)
        return image, label

    def random_crop(self, image, label, crop_size):
        H, W, _ = image.shape
        if (W, H) == crop_size: 
            return image, label
        if W < crop_size[1] or H < crop_size[0]:
            h_scale = crop_size[0] / H
            w_scale = crop_size[1] / W
            scale = max(h_scale, w_scale)

            W, H = int(scale * W + 1), int(scale * H + 1)
            image = cv2.resize(image, (W, H), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (W, H), interpolation=cv2.INTER_NEAREST)

        sw, sh = random.random() * (W - crop_size[1]), random.random() * (H - crop_size[0])
        crop = int(sw), int(sh), int(sw) + crop_size[1], int(sh) + crop_size[0]

        return image[crop[1]:crop[3], crop[0]:crop[2], :], label[crop[1]:crop[3], crop[0]:crop[2]]

    def get_collate_fn(self):
        return None

class ImageCaptionDataset(Dataset):
    
    def __init__(
            self, 
            images_dir,
            labels_dir,
            chars_file,
            shape=(64, 640),
            num_channel=3,
            augmentation=None,
             mean=[0.,0.,0.],
             std=[1.0,1.0,1.0],
             max_length=50,
             dtype='train'
    ):
        super().__init__()
        self.num_channel = num_channel
        self.images_dir = images_dir

        self.img_lists, self.label_lists  = self.read_labelFile(labels_dir)

        assert len(self.img_lists) == len(self.label_lists)
        assert len(self.img_lists) != 0, 'No file exists'

        self.voc2id, self.id2voc = self.read_charsFile(chars_file)

        self.max_length = max_length
        self.dtype = dtype

        self.augmentation = augmentation
        self.shape = shape
        self.mean = mean
        self.std = std
    
    def __getitem__(self, index):
        
        image = cv2.imread(self.img_lists[index])

        label = self.label_lists[index]
        label = np.array([int(self.voc2id.get(voc, self.voc2id['UNK_TOKEN'])) for voc in label])

        image = self.resize(image, size=self.shape)
        image = image.astype(np.uint8)
        
        if self.augmentation is not None:
            image, label = self.augmentation.make_aug(image, label)

        image, label = self.preprocessing(image, label)
        image = np.ascontiguousarray(image)
        label = np.ascontiguousarray(label)

        return image, label
        
    def __len__(self):
        return len(self.img_lists)

    def read_charsFile(self, chars_file):
        voc2id = {}
        voc2id['START_TOKEN'] = 0
        voc2id['PAD_TOKEN'] = 1
        voc2id['END_TOKEN'] = 2
        voc2id['UNK_TOKEN'] = 3

        with open(chars_file,encoding='utf-8',  mode='r') as f:
            ff = f.readlines()
            for i, voc in enumerate(ff):
                voc = voc.strip()
                voc2id[voc] = i+4

        id2voc = {j:i for i, j in enumerate(voc2id)}
        return voc2id, id2voc

    def read_labelFile(self, labe_file):
        img_lists = []
        label_lists = []
        with open(labe_file, mode='r') as fr:
            ff  = fr.readlines()
            for line in ff:
                line = line.strip()
                line = line.split(' ')
                img_name = line[0]
                if not img_name.endswith('g'):
                    img_name = img_name[:-1]
                img_lists.append(os.path.join(str(self.images_dir), img_name))
                label_lists.append(line[1:])
        return img_lists, label_lists
    
    def to_tensor(self, x, **kwargs):
        return x.transpose(2, 0, 1).astype('float32')

    def resize(self, image, size=(64, 640)):
        hsize, wsize, c = image.shape
        hscale = size[0] / hsize
        wscale = size[1] / wsize
        scale = min((hscale, wscale))

        image = cv2.resize(image, (int(scale*wsize), int(scale*hsize)))
        padimg = np.full(shape=(size[0], size[1], 3), fill_value=255, dtype=float)

        padimg[:int(scale*hsize), :int(scale*wsize), :] = image

        return padimg

    def preprocessing(self, image, label):
        label_mask = np.full(shape=(self.max_length), fill_value=self.voc2id['PAD_TOKEN'], dtype=int)
        label = [self.voc2id.get(i, self.voc2id['UNK_TOKEN']) for i in label]
        label.insert(0,self.voc2id['START_TOKEN'])
        label.append(self.voc2id['END_TOKEN'])
        label_len = len(label) if len(label) < self.max_length-1 else self.max_length-1
        label_mask[:label_len] = label[:label_len]

        if self.num_channel == 1:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = np.expand_dims(image, axis=-1)
        return self.to_tensor(image), label_mask

    def get_collate_fn(self):
        return Collater_ICT()

class CocoStuff_10kDataset(Dataset):
    
    def __init__(
            self, 
            data_dir='./coco_suff_10k',
            shape=(512,512),
            categories=COCO_STUFF_10k_CATEGORIES, 
            augmentation=None,
            dtype='train',
            ignore_label=255,
            mean=None,
            std=None
    ):
        super().__init__()
        self.mean = mean
        self.std = std

        self.dtype = dtype
        self.data_dir = data_dir

        self.ignore_label = ignore_label

        lists_file = os.path.join(self.data_dir, 'imageLists', self.dtype+'.txt')

        with open(lists_file, "r") as f:
            image_list = f.readlines()

        self.image_list = [f.strip() for f in image_list]

        logger.info("Dataset {} samples: {}".format(self.dtype, len(self.image_list)))

        self.augmentation = augmentation
        
        self.shape = shape

        stuff_ids = [k["id"] for k in categories]
        assert len(stuff_ids) == 171, len(stuff_ids)

        self.stuff_dataset_id_to_contiguous_id = {i: k for i, k in enumerate(stuff_ids)}
        self.stuff_classes = [k["name"] for k in categories]
        
    
    def __getitem__(self, i):
        fname = self.image_list[i]
        image_path =  os.path.join(self.data_dir, "images", fname + ".jpg")
        image = cv2.imread(image_path)

        matfile = h5py.File(os.path.join(self.data_dir, "annotations", fname + ".mat"))
        gt_numpy = np.array(matfile["S"]).astype(np.uint8)
        gt_numpy = np.transpose(gt_numpy)
        gt_numpy = gt_numpy - 2

        assert gt_numpy.shape == image.shape[:2], "{} vs {}".format(gt_numpy.shape, image.shape)

        image, gt_numpy = self.preprocessing(image, gt_numpy, self.shape, mean=self.mean, std=self.std)
        classes = np.unique(gt_numpy).astype(np.int)
        
        classes = classes[classes != self.ignore_label]
        masks = []
        for class_id in classes:
            masks.append(gt_numpy == class_id)


        if len(masks) == 0:
            gt_masks = np.zeros((0, gt_numpy.shape[-2], gt_numpy.shape[-1]))
        else:
            gt_masks = np.stack([np.ascontiguousarray(x.copy()) for x in masks])
            gt_masks = gt_masks.astype(np.bool)
    
        if self.augmentation:
            gt_masks_ = np.transpose(gt_masks, axes=(1,2,0)).astype(np.int32)
            image_aug, masks_aug = self.augmentation.make_aug(image.transpose((1,2,0)).astype(np.uint8), gt_masks_)
            assert len(classes) == masks_aug.shape[2], '{} != {}'.format(len(classes), masks_aug.shape[2])
            valid_idx = []
            for class_idx in range(len(classes)):
                m = masks_aug[:, :, class_idx]
                if m.sum() > 10:
                    valid_idx.append(class_idx)
                    
            classes_ = classes[valid_idx]
            gt_masks_ = masks_aug.transpose((2,0,1))[valid_idx].astype(np.bool)
            assert len(classes_) == gt_masks_.shape[0]
            if len(gt_masks_) != 0:
                classes = classes_
                gt_masks = gt_masks_
                image = image_aug.transpose((2,0,1)).astype(np.float32)

        target = dict(
            labels=classes,
            masks=gt_masks
            )
        return image, target
        
    def __len__(self):
        return len(self.image_list)
    
    def preprocessing(self, image, mask, input_size, mean=None, std=None, swap=(2, 0, 1)):
        if len(image.shape) == 3:
            padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
        else:
            padded_img = np.ones(input_size) * 114.0
        img = np.array(image)
        r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
        resized_img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

        padded_img = padded_img[:, :, ::-1]
        padded_img /= 255.0
        if mean is not None:
            padded_img -= mean
        if std is not None:
            padded_img /= std
        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)

        padded_mask = np.full(shape=(input_size[0], input_size[1]), fill_value=255, dtype=np.uint8)
        resized_mask = cv2.resize(mask, (int(mask.shape[1] * r), int(mask.shape[0] * r)), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        padded_mask[: int(mask.shape[0] * r), : int(mask.shape[1] * r)] = resized_mask

        return padded_img, padded_mask

    def get_collate_fn(self):
        return Collater_Maskformer()