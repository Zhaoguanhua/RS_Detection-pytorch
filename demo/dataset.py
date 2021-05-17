#!usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : zhaoguanhua
@Email   : 
@Time    : 2021/5/17 21:16
@File    : dataset.py
@Software: PyCharm
"""
import os
import torch
import torch.utils.data
import cv2
import json
import numpy as np
from pycocotools import mask as coco_mask
import transforms as T

class BuildingDataset(torch.utils.data.Dataset):
    def __init__(self,images_dir,annotation_file,transforms=None):
        self.images_dir=images_dir
        self.annotation_file=annotation_file
        self.transform=transforms
        self.images_name=os.listdir(images_dir)
        with open(annotation_file) as f:
            self.labels_json=json.load(f)

    def __getitem__(self, i):
        image=cv2.imread(os.path.join(self.images_dir,self.images_name[i]))
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        label_info=self.labels_json[self.images_name[i]+'262144']
        annos=label_info['regions']

        boxes=[]
        labels=[]
        masks=[]

        for anno in annos:
            shape_attr=anno["shape_attributes"]
            px=shape_attr["all_points_x"]
            py=shape_attr["all_points_y"]
            poly=[(x+0.5,y+0.5) for x,y in zip(px,py)]
            poly=[p for x in poly for p in x]

            #将点坐标转换为mask二值图片
            rles=coco_mask.frPyObjects([poly],512,512)
            mask=coco_mask.decode(rles)

            category_id=1

            boxes.append([np.min(px),np.min(py),np.max(px),np.max(py)])
            labels.append(category_id)
            masks.append(mask.squeeze())

        image_id=torch.tensor([i])
        iscrowd=torch.zeros((len(annos),),dtype=torch.int64)

        boxes=torch.as_tensor(boxes,dtype=torch.float32)

        labels=torch.as_tensor(labels,dtype=torch.int64)
        masks=torch.as_tensor(masks,dtype=torch.uint8)

        target={}
        target["boxes"]=boxes
        target["labels"]=labels
        target["masks"]=masks
        target["image_id"]=image_id
        target["iscrowd"]=iscrowd

        if  self.transform is not None:
            image,target=self.transform(image,target)

        return image,target


    def __len__(self):
        return len(self.images_name)


def get_transform(train):
    transforms=[]

    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFilp(0.5))

    return T.Compose(transforms)

