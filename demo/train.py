# -*- coding:utf-8 -*-
"""
@Author :zhaoguanhua
@Email   :
@Time    :2020/12/1 13:46
@File    :train.py
@Software:PyCharm
"""
import os
import sys
sys.path.append("..")
import torch
from utils_tool import utils
from torch.utils.data import DataLoader
from dataset import PennFudanDataset, get_transform,CrowAiBuildingDataset
from instance_detection.model import get_instance_segmentation_model
from utils_tool.engine import train_one_epoch,evaluate

# # root dir
root_dir = r"D:\test_data\aicrowd_building"
#

# data_train = os.path.join(root_dir, "train")
data_valid = os.path.join(root_dir, "val")

dataset_train=CrowAiBuildingDataset(os.path.join(data_valid,"images"),os.path.join(data_valid,"annotation-small.json"),
                                    transforms=get_transform(train=True))
dataset_valid=CrowAiBuildingDataset(os.path.join(data_valid,"images"),os.path.join(data_valid,"annotation-small.json"),
                                    transforms=get_transform(train=False))

torch.manual_seed(1)
indices=torch.randperm(len(dataset_train)).tolist()
dataset_train=torch.utils.data.Subset(dataset_train,indices[:-50])
dataset_valid=torch.utils.data.Subset(dataset_valid,indices[-50:])

# define dataloader
dataloader_train = DataLoader(dataset_train,batch_size=2,shuffle=True,num_workers=0,collate_fn=utils.collate_fn)
dataloader_valid=DataLoader(dataset_valid,batch_size=2,shuffle=False,num_workers=0,collate_fn=utils.collate_fn)

device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

num_classes=2

#model
model=get_instance_segmentation_model(num_classes=num_classes)

print(model)

#move model to the right device
model.to(device)

#construct an optimizer
params=[p for p in model.parameters() if p.requires_grad]
optimizer=torch.optim.SGD(params,lr=0.005,momentum=0.9,weight_decay=0.0005)
#add a learning rate scheduler which decreases the learning rate by 10x every 3 epoches
lr_scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=3,gamma=0.1)

#train model
num_epoches=10

for epoch in range(num_epoches):
    #train for one epoch,printing every 10 iterations
    train_one_epoch(model,optimizer,dataloader_train,device,epoch,print_freq=1)

    # if min_score > valid_logs[loss_name]:
    #     min_score = valid_logs[loss_name]
    torch.save(model.state_dict(), 'maskrcnn_building_epoch{}.pth'.format(epoch))
    # print('Model saved at epoch {}!'.format())

    # if i%20==0:
    #     torch.save(model.state_dict(),ENCODER+'_unet_building_'+str(i)+'.pth')

    #update the learning rate
    lr_scheduler.step()
    #evaluate on the validation dataset
    evaluate(model,dataloader_valid,device=device)
