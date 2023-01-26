'''
# Author: Shaoran Lu
# Date: 2021/10/04
# Email: lushaoran92@gmail.com
# Description: 

example:

'''
import torch
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor,  Resize

from loguru import logger
import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

from tools.augmentation import Augmentation, BaseAugmentation
from data.dataset import CustomDataset,  CustomDataset_seg, OcrDetDataset, OcrRecDataset, CocoDataset,\
    ImageCaptionDataset, CityscapesDataset, CocoStuff_10kDataset
from data.data_utils import TrainTransform, TrainTransform_Instance

trans = torchvision.transforms.Compose([ ToTensor(), Resize(size=(224, 224)) ])

class Data_loader():
    def __init__(self, config, args): 
        self.is_main_process = True if  args.rank== 0 else False
        self.config = config
        self.train_list = config.get('train_list', None)
        self.test_list = config.get('test_list', None)
        self.dataset_name = self.config['dataset_name']
        augmentation_type = self.config.get('augmentation_type', None)
        aug_dicts = config.get('aug_dicts', None)
        if aug_dicts is not None:
            aug=BaseAugmentation(config['aug_dicts'])()
            if self.is_main_process:
                print('Using config augmentation')
        else:
            aug = None
            if self.is_main_process:
                print('Using default augmentation')
        self.augmentation = Augmentation(task_type='cls' if augmentation_type is None else augmentation_type, aug=aug)
        
        
    def get_train(self, distributed=False, nprocs=1):
        if self.is_main_process:
            logger.info("Trian Dataset name: {}".format(self.dataset_name))
        if self.dataset_name == 'custom':
            train_dataset = CustomDataset(lists=self.train_list, shape=eval(self.config['input_size']),
                                          augmentation=self.augmentation if self.config['aug'] else None, dtype='train'
                                          )
        elif self.dataset_name == 'test_cls':
            train_dataset = datasets.CIFAR10(
                            root="../torch_data",
                            train=True,
                            download=True,
                            transform=trans
                            )
        elif self.dataset_name == 'test_seg':
            train_dataset = CustomDataset_seg(lists=self.train_list, shape=eval(self.config['input_size']),
                                              augmentation=self.augmentation if self.config['aug'] else None, dtype='train'
                                              )    
        elif self.dataset_name == 'OcrDet':
            train_dataset = OcrDetDataset(lists=self.train_list, shape=eval(self.config['input_size']),
                                              augmentation=self.augmentation if self.config['aug'] else None, dtype='train'
                                              )  
        elif self.dataset_name == 'OcrRec':
            train_dataset = OcrRecDataset( lists=self.train_list, shape=eval(self.config['input_size']),
                                                augmentation=self.augmentation if self.config['aug'] else None, dtype='train'
                                              )   
        elif self.dataset_name == 'Coco':
            train_dataset = CocoDataset(data_dir=self.config['data_dir'],  json_file=self.config['json_file'], name=self.config['name'],
                                                                        shape=eval(self.config['input_size']), augmentation=self.augmentation if self.config['aug'] else None,
                                                                        preproc=TrainTransform(rgb_means=(0.485, 0.456, 0.406),
                                                                                                                            std=(0.229, 0.224, 0.225),
                                                                                                                            max_labels=50
                                                                                                                            ) if self.config.get('training_mission', 'det') != 'instance' else TrainTransform_Instance(rgb_means=(0.485, 0.456, 0.406),
                                                                                                                            std=(0.229, 0.224, 0.225)),
                                                                        mode= self.config.get('training_mission', 'det'),
                                                                        dtype='train'
                                                                        )
        elif self.dataset_name == 'IC':
            train_dataset = ImageCaptionDataset(images_dir=self.config['images_dir'],  labels_dir=self.config['labels_dir'], chars_file=self.config['chars_file'],
                                                                        shape=eval(self.config['input_size']), augmentation=self.augmentation if self.config['aug'] else None,
                                                                        max_length = int(self.config.get('max_length', 50)),
                                                                        num_channel= int(self.config.get('num_channel', 3)),
                                                                        dtype='train'
                                                                        )
        elif self.dataset_name == 'Cityscapes':
            train_dataset = CityscapesDataset(data_dir=self.config['images_dir'], json_file=self.config.get('json_file', None),  shape=eval(self.config['input_size']),
                                                                                    augmentation=self.augmentation if self.config['aug'] else None,  name=self.config['name'], 
                                                                                    dtype='train', ignore_lb=int(self.config.get('ignore_lb', 255)),
                                                                                    scales=(0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 0.875, 1.0, 1.25, 1.5)
                                                                        )

        elif self.dataset_name == 'Coco_stff_10k':
            train_dataset = CocoStuff_10kDataset(data_dir=self.config['data_dir'], shape=eval(self.config['input_size']),
                                                                                    dtype='train',  mean=None, std=None
                                                                                    )

        else:
            raise NotImplementedError('{} dataset_name not supported.'.format(self.dataset_name))

        if self.is_main_process:
            logger.info("Train Dataset samples: {}".format(len(train_dataset)))

        if not distributed:
            return DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=self.config.get('shuffle', True),
                            num_workers=self.config['num_workers'], collate_fn=train_dataset.get_collate_fn())

        else:
            assert self.config['batch_size']  % nprocs == 0
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            sampler_flag = self.config.get('sampler', True)
            train_loader = DataLoader(train_dataset,
                                                    batch_size=self.config['batch_size']  // nprocs,
                                                    num_workers=max(self.config['num_workers'] // nprocs, 1),
                                                    pin_memory=True,
                                                    sampler=train_sampler if sampler_flag else None,
                                                    collate_fn=train_dataset.get_collate_fn())
            return train_loader, train_sampler

    
    def get_test(self, distributed=False, nprocs=1):
        if self.is_main_process:
            logger.info("Test Dataset name: {}".format(self.dataset_name))

        if self.dataset_name == 'custom':
            test_dataset = CustomDataset(lists=self.test_list, shape=eval(self.config['input_size']),
                                          augmentation=None, dtype='val'
                                          )
        elif self.dataset_name == 'test_cls':
            test_dataset = datasets.CIFAR10(
                           root="../torch_data",
                           train=False,
                           download=True,
                           transform=trans
                           )
        elif self.dataset_name == 'test_seg':
            test_dataset = CustomDataset_seg(lists=self.test_list, shape=eval(self.config['input_size']),
                                             augmentation=None, dtype='val'
                                             )
        elif self.dataset_name == 'OcrDet':
            test_dataset = OcrDetDataset(lists=self.test_list, shape=eval(self.config['input_size']),
                                              augmentation=None, dtype='val'
                                              ) 
        elif self.dataset_name == 'OcrRec':
            test_dataset = OcrRecDataset( lists=self.test_list, shape=eval(self.config['input_size']),
                                                augmentation=None, dtype='val'
                                              )   
        elif self.dataset_name == 'Coco':
            test_dataset = CocoDataset(data_dir=self.config['data_dir'],  json_file=self.config['json_file_val'], name=self.config['name_val'],
                                                                        shape=eval(self.config['input_size']), augmentation=None,
                                                                        preproc=TrainTransform(rgb_means=(0.485, 0.456, 0.406),
                                                                                                                            std=(0.229, 0.224, 0.225),
                                                                                                                            max_labels=50
                                                                                                                            ) if self.config.get('training_mission', 'det') != 'instance' else TrainTransform_Instance(rgb_means=(0.485, 0.456, 0.406),
                                                                                                                            std=(0.229, 0.224, 0.225), with_box=True),
                                                                        mode= self.config.get('training_mission', 'det'),
                                                                        dtype='val'
                                                                        )
        elif self.dataset_name == 'IC':
            test_dataset = ImageCaptionDataset(images_dir=self.config['images_dir_val'],  labels_dir=self.config['labels_dir_val'], chars_file=self.config['chars_file'],
                                                                        shape=eval(self.config['input_size']), augmentation= None,
                                                                        max_length = int(self.config.get('max_length', 50)),
                                                                        num_channel= int(self.config.get('num_channel', 3)),
                                                                        dtype='val'
                                                                        )

        elif self.dataset_name == 'Cityscapes':
            test_dataset = CityscapesDataset(data_dir=self.config['images_dir'], json_file=self.config.get('json_file', None),  shape=eval(self.config['input_size']),
                                                                                    augmentation=None,  name=self.config['name_val'], 
                                                                                    dtype='val', ignore_lb=int(self.config.get('ignore_lb', 255)),
                                                                                    scales=(0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 0.875, 1.0, 1.25, 1.5)
                                                                        )
        elif self.dataset_name == 'Coco_stff_10k':
            test_dataset = CocoStuff_10kDataset(data_dir=self.config['data_dir'], shape=eval(self.config['input_size']),
                                                                                    dtype='test',  mean=self.config.get('mean', None), std=self.config.get('std', None)
                                                                                    )

        else:
            raise NotImplementedError('{} dataset_name not supported.'.format(self.dataset_name))
        if self.is_main_process:
            logger.info("Test Dataset samples: {}".format(len(test_dataset)))
        
        if not distributed:
            return DataLoader(test_dataset, batch_size=self.config['batch_size'], shuffle=False,
                            num_workers=self.config['num_workers'], collate_fn=test_dataset.get_collate_fn())
        else:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
            sampler_flag = self.config.get('sampler', True)
            test_loader = DataLoader(test_dataset,
                                                    batch_size=self.config['batch_size']  // nprocs,
                                                    num_workers=max(self.config['num_workers'] // nprocs, 1),
                                                    pin_memory=True,
                                                    sampler=test_sampler if sampler_flag else None,
                                                    collate_fn=test_dataset.get_collate_fn()
                                                    )
            return test_loader, test_sampler


