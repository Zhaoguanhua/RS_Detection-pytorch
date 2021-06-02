#!usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : zhaoguanhua
@Email   : 
@Time    : 2021/6/2 17:08
@File    : fasterrcnn.py
@Software: PyCharm
"""
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_fasterrcnn(num_classes):
    model=torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features=model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor=FastRCNNPredictor(in_features,num_classes)

    return model