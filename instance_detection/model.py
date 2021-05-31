# -*- coding:utf-8 -*-
"""
@Author :zhaoguanhua
@Email   :
@Time    :2020/12/1 13:32
@File    :model.py
@Software:PyCharm
"""
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def get_instance_segmentation_model(num_classes):
    # load and instance segmentation model
    model=torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False)

    #get the number of input features for the classifier
    in_features=model.roi_heads.box_predictor.cls_score.in_features
    #replace the pre-trained head with a new one
    model.roi_heads.box_predictor=FastRCNNPredictor(in_features,num_classes)

    #now get the number of input features for the mask classifier
    in_features_mask=model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer=256
    #and replace the mask predictor with a new one
    model.roi_heads.mask_predictor=MaskRCNNPredictor(in_features_mask,hidden_layer,num_classes)

    return model



