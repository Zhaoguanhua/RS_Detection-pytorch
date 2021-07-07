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
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
from utils_tool import transforms as T
from PIL import Image
import matplotlib.pyplot as plt

class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transform = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])

        img = Image.open(img_path)
        img = img.convert("RGB")
        mask = Image.open(mask_path)
        mask = np.array(mask)

        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]

        masks = mask == obj_ids[:, None, None]

        num_objs = len(obj_ids)

        boxes = []

        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])

            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


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

class CrowAiBuildingDataset(torch.utils.data.Dataset):

    def __init__(self,images_dir,annotation_file,use_mask=False,transforms=None):
        self.images_dir=images_dir
        self.annotation_file=annotation_file
        self.transform=transforms
        self.use_mask=use_mask
        self.coco=COCO(self.annotation_file)
        self.class_Ids=self.coco.getCatIds()
        self.image_ids=self.coco.getImgIds()

    def __getitem__(self, i):


        annos = self.coco.getAnnIds(imgIds=[self.image_ids[i]], catIds=self.class_Ids, iscrowd=None)
        anns = self.coco.loadAnns(annos)

        img_name= self.coco.imgs[self.image_ids[i]]['file_name']
        image=cv2.imread(os.path.join(self.images_dir,img_name))
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        boxes=[]
        labels=[]
        masks=[]

        for ann in anns:


            seg=ann["segmentation"]

            x_points=seg[0][::2]
            y_points=seg[0][1::2]
            x11=min(x_points)
            x22=max(x_points)
            y11=min(y_points)
            y22=max(y_points)

            category_id=1

            if x11!=x22 and y11!=y22:
                bbox=[x11,y11,x22,y22]
                boxes.append(bbox)
                labels.append(category_id)
                if self.use_mask:
                    mask = self.coco.annToMask(ann)
                    masks.append(mask)

        image_id=torch.tensor([i])
        iscrowd=torch.zeros((len(annos),),dtype=torch.int64)
        #
        # print(boxes)
        boxes=torch.as_tensor(boxes,dtype=torch.float32)

        labels=torch.as_tensor(labels,dtype=torch.int64)

        if self.use_mask:
            masks=torch.as_tensor(masks,dtype=torch.uint8)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target={}
        target["boxes"]=boxes
        target["labels"]=labels
        if self.use_mask:
            target["masks"]=masks
        target["image_id"]=image_id
        target["iscrowd"]=iscrowd
        target["area"] = area

        if  self.transform is not None:
            image,target=self.transform(image,target)

        return image,target

    def __len__(self):
        return len(self.coco.getImgIds())


class DatasetFromSemantic(torch.utils.data.Dataset):
    """
    将语义分割的标签转换为mask rcnn网络需要的格式
    """
    def __init__(self,image_dir,labels_dir,classes,transform=None):
        super(DatasetFromSemantic, self).__init__()
        self.image_dir=image_dir
        self.labels_dir=labels_dir
        self.images=os.listdir(image_dir)
        self.transform=transform
        self.classes=classes

    def __getitem__(self, i):

        image_name=os.path.basename(self.images[i])
        image_array=cv2.imread(os.path.join(self.image_dir,image_name))
        image_array=cv2.cvtColor(image_array,cv2.COLOR_BGR2RGB)

        mask_array=cv2.imread(os.path.join(image_array,cv2.COLOR_BGR2RGB))

        kernel = np.ones((5,5),np.uint8)
        mask_array=cv2.morphologyEx(mask_array,cv2.MORPH_OPEN, kernel)

        boxes=[]
        labels=[]
        masks=[]

        #将mask转换为mask rcnn网络输入的格式
        target_num=0
        for class_i in range(1,self.classes):      #逐类别转换
            mask_i =(mask_array==class_i).astype("uint8")
            if np.max(mask_i)!=0:
                contours, hierarchy = cv2.findContours(mask_i,
                                                       cv2.RETR_CCOMP,
                                                       cv2.CHAIN_APPROX_NONE)

                for contour in contours:
                    if contour.shape[0]>10:
                        contour=contour.squeeze()
                        maxx,maxy=np.max(contour,axis=0)
                        minx,miny=np.min(contour,axis=0)

                        if ((maxx-minx)*(maxy-miny))>5:
                            mask_i_j=np.zeros(image_array.shape,np.uint8)
                            cv2.fillConvexPoly(mask_i_j,contour,(1,1,1))
                            masks.append(mask_i_j[:,:,0])
                            boxes.append([minx,miny,maxx,maxy])
                            labels.append(class_i)
                            target_num+=1


        image_id=torch.tensor([i])
        iscrowd=torch.zeros((target_num,),dtype=torch.int64)
        #
        # print(boxes)
        boxes=torch.as_tensor(boxes,dtype=torch.float32)
        labels=torch.as_tensor(labels,dtype=torch.int64)
        masks=torch.as_tensor(masks,dtype=torch.uint8)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target={}
        target["boxes"]=boxes
        target["labels"]=labels
        target["masks"]=masks
        target["image_id"]=image_id
        target["iscrowd"]=iscrowd
        target["area"] = area

        if  self.transform is not None:
            image_array,target=self.transform(image_array,target)

        return image_array,target


    def __len__(self):
        return len(self.images)


def get_transform(train):
    transforms=[]

    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))

    return T.Compose(transforms)

if __name__ == '__main__':

    # image_dir=r"D:\test_data\aicrowd_building\train\images"
    # anno_json_file=r"D:\test_data\aicrowd_building\train\annotation-small.json"

    image_dir=r"D:\test_data\aicrowd_building\val\images"
    anno_json_file=r"D:\test_data\aicrowd_building\val\annotation-small.json"
    #
    ai_dataset=CrowAiBuildingDataset(images_dir=image_dir,annotation_file=anno_json_file,
                                     transforms=get_transform(train=True))
    print(len(ai_dataset))
    for i in range(len(ai_dataset)):
        print(i)
        img,target=ai_dataset[i]
    # # print(ai_dataset.class_Ids)
    # data=ai_dataset[0]
    # print(data)
    # # with open(anno_json_file) as f:
    # #     labels_json = json.load(f)
    # #     print(type(labels_json))
    # #
    # #     for i in labels_json:
    # #         print(i)
    # #         # print(labels_json[i])
    # #
    # #     # for j in labels_json["images"]:
    # #     #     print(j)
    # #
    # #     for k in labels_json["annotations"]:
    # #         print(k)
    # coco=COCO(anno_json_file)
    # #
    # image_ids=coco.getImgIds()
    # class_Ids=coco.getCatIds()
    # # # # print(class_Ids)
    # print(image_ids)
    # # print(len(image_ids))
    # #
    # # import shutil
    # # small_dir=r"D:\test_data\train\image"
    # for i in image_ids:
    #     path=os.path.join(image_dir,coco.imgs[i]['file_name'])
    #     img_array=cv2.imread(path)
    #     # small_path=os.path.join(small_dir,coco.imgs[i]['file_name'])
    #     # print(path)
    #     # shutil.copy(path,small_path)
    #     annos=coco.getAnnIds(imgIds=[i],catIds=class_Ids,iscrowd=None)
    #     # print(annos[0])
    #     anns=coco.loadAnns(annos)
    #     # coco.showAnns(ann)
    #     # print(ann)
    #     # mask=coco.annToMask(ann)
    #     # print(mask)
    #     id=0
    #     for ann in anns:
    #         temp_path=str(id)+".PNG"
    #         seg=ann["segmentation"]
    #         bbox=ann["bbox"]
    #
    #         mask=coco.annToMask(ann)
    #         # id+=1
    #         x_points=seg[0][::2]
    #         y_points=seg[0][1::2]
    #         x11=min(x_points)
    #         x22=max(x_points)
    #         y11=min(y_points)
    #         y22=max(y_points)
    #
    #         if x11==x22 or y11==y22:
    #             print((x11, y11, x22, y22))
    #             cv2.rectangle(img_array,(x11,y11),(x22,y22),(0,255,0),2)
    #             plt.imshow(img_array)
    #             plt.show()
            # x1=bbox[1]
            # x2=bbox[1]+bbox[3]
            # y1=bbox[0]
            # y2=bbox[0]+bbox[2]
            # print((x1,y1,x2,y2))
            # cv2.rectangle(img_array,(x1,y1),(x2,y2),(0,255,0),2)
            # cv2.imwrite(temp_path,img_array)

        # plt.imshow(img_array)
        # coco.showAnns(anns)
        # plt.show()
        # break