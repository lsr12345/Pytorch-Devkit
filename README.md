# Pytorch-Devkit

This repository contains related algorithms in many fields: 
* object detection
* segmentation
* OCR
* image translation

## Table of Contents
- [Pytorch-Devkit](#pytorch-devkit)
  - [Table of Contents](#table-of-contents)
  - [About Details](#about-details)
  - [How to Use](#how-to-use)
  - [Requirements](#requirements)

## About Details
- Support distributed training
- Support mixed precision training
- Support multiple augments
- Backbone: Mobilenetv3 MobileViT Resnet Swintransformer DarkNet StdcNet
- NECK:FPN、PAFPN
- Character Recognition：CRNN
- Character detection：DBNET
- Object detection：YOLOX
- Segmentation：ReBiSegNet MaskFormer SOLOV2
- Image translation：ICT
## How to Use

```bash
$ python setup.py develop
$ python train_ddp.py -f ./config/Config_Yolox.yaml
```

For details about how to configure related algorithms, see examples.


## Requirements

* `requirements.txt`