'''
# Author: Shaoran Lu
# Date: 2021/10/04
# Email: lushaoran92@gmail.com
# Description: 数据增强

example:

'''

import numpy as np
from loguru import logger

import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables.polys import PolygonsOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

aug_func = {
    'affine': iaa.Affine,
    'fliplr': iaa.Fliplr,
    'flipud': iaa.Flipud,
    'addgaussiannoise': iaa.AdditiveGaussianNoise,
    'multiply': iaa.Multiply,
    'cutout': iaa.Cutout,
    'add': iaa.Add,
    'grayscale': iaa.Grayscale,
    'clouds': iaa.Clouds,
    'fog': iaa.Fog,
    'snowflakes': iaa.Snowflakes,
    'rain': iaa.Rain,
    'gaussianblur': iaa.GaussianBlur
}

class BaseAugmentation():
    def __init__(self, aug_dicts, mode='some'):
        assert isinstance(aug_dicts, dict)
        self.aug_dicts = aug_dicts
        self.mode = mode
    
    def __call__(self):
        augment_func =[aug_func[f](**self.aug_dicts[f]) for f in self.aug_dicts]
        if self.mode == 'some':
            return iaa.SomeOf((0, len(augment_func)), augment_func)
        else:
            return iaa.Sequential(augment_func)

class Augmentation():
    def __init__(self, use_aug=True, task_type='cls', aug=None):
        
        assert task_type in ['cls', 'det', 'seg', 'polygon', 'custom']
        self.use_aug = use_aug
        self.aug = iaa.SomeOf((0, 13),[
                    iaa.Affine(translate_percent=[-0.05, 0.05], scale=[0.8, 1.2], rotate=(-5, 5), mode='constant', cval=[240, 255]),
                    iaa.Fliplr(0.5),
                    iaa.Flipud(0.5),
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 12.0), per_channel=0.5),
                    iaa.Multiply((0.5, 1.5)),
                    iaa.Cutout(nb_iterations=(1, 4), size=0.1, squared=False, fill_mode="constant", cval=(0, 255), fill_per_channel=0.5),
                    iaa.Add((-40, 40), per_channel=0.5),
                    iaa.Grayscale(alpha=(0.0, 1.0)),
                    iaa.GaussianBlur(sigma=(0.0,1.4))
                ]) if aug is None else aug
        
        self.task_type = task_type
        logger.info("Augmentation_type: {}".format(self.task_type))
    def make_aug(self, img, label, box_label=None):
        if self.task_type == 'cls':
            img = self.aug(image=img)
            return img, label
        elif self.task_type == 'det':
            boxes = BoundingBoxesOnImage([BoundingBox(x1=float(ii[0]), y1=float(ii[1]), x2=float(ii[2]), y2=float(ii[3]), 
                                                                                                                label=ii[4]) for ii in label], shape=img.shape)
            new_img, new_boxes = self.aug(image=img, bounding_boxes=boxes)
            new_boxes = new_boxes.remove_out_of_image().clip_out_of_image()
            boxes_ = [[float(new_boxes.bounding_boxes[j].x1), float(new_boxes.bounding_boxes[j].y1),
                      float(new_boxes.bounding_boxes[j].x2), float(new_boxes.bounding_boxes[j].y2), new_boxes.bounding_boxes[j].label] for j in range(len(new_boxes.bounding_boxes))]
            
            return new_img, boxes_
        
        elif self.task_type == 'polygon':
            polygons = PolygonsOnImage([ia.Polygon(p[:-1], label=p[-1]) for p in label], shape=img.shape)
            new_img, new_polygons = self.aug(image=img,  polygons=polygons)
            new_polygons = new_polygons.remove_out_of_image().clip_out_of_image()
            polygons_ = [new_polygons.polygons[j].coords.tolist()+[new_polygons.polygons[j].label] for j in range(len(new_polygons.polygons))]
            
            return new_img, polygons_
            
        elif self.task_type == 'seg':
            label = np.array(label)
            if box_label is not None:
                box_label = np.array(box_label)
                box_label = BoundingBoxesOnImage([BoundingBox(x1=float(ii[0]), y1=float(ii[1]), x2=float(ii[2]), y2=float(ii[3]), 
                                                                                                                    label=ii[4]) for ii in box_label], shape=img.shape)
                seg_map = SegmentationMapsOnImage(label, shape=img.shape)
                new_img, seg_map, new_boxes = self.aug(image=img, segmentation_maps=seg_map, bounding_boxes=box_label)
                new_boxes = [[float(new_boxes.bounding_boxes[j].x1), float(new_boxes.bounding_boxes[j].y1),
                        float(new_boxes.bounding_boxes[j].x2), float(new_boxes.bounding_boxes[j].y2), new_boxes.bounding_boxes[j].label] for j in range(len(new_boxes.bounding_boxes))]
                seg_map = seg_map.get_arr()
                return new_img, seg_map, new_boxes

            else:
                seg_map = SegmentationMapsOnImage(label, shape=img.shape)
                new_img, seg_map = self.aug(image=img, segmentation_maps=seg_map)
                seg_map = seg_map.get_arr()
                
                return new_img, seg_map
        
        else:
            return self.custom_label_type(img, label)
        
    def custom_label_type(self, img, label):
        raise NotImplementedError('Custom label type not supported.')

    def reorder_vertexes(self, pts):
        pts = np.array(pts)
        rect = np.zeros((4, 2), dtype = "float32")

        s = pts.sum(axis = 1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis = 1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect.tolist()