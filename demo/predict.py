# -*- coding:utf-8 -*-
"""
@Author :zhaoguanhua
@Email   :
@Time    :2020/12/1 16:16
@File    :predict.py
@Software:PyCharm
"""
import numpy as np
import torch
from model import get_instance_segmentation_model
from Base import PennFudanDataset, get_transform
import cv2
from PIL import Image
import transforms as T
from torchvision.transforms import functional as F

test_file=r"D:\project\Python\Mask_RCNN_Pytorch\PennFudanPed\PNGImages\FudanPed00029.png"
model_path=r"unet_building_epoch9.pth"

device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model=get_instance_segmentation_model(num_classes=2)
# print(model)

model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()

# data_train = os.path.join(root_dir, "")
# data_valid = os.path.join(root_dir, "")

# define dataset
# converts the image, a PIL image, into a PyTorch Tensor
img = Image.open(test_file)
img = img.convert("RGB")
img=F.to_tensor(img)

print(img.size())
out_predict="test.png"
out_array=np.zeros((img.size()[1:]))
with torch.no_grad():
    res=model([img.to(device)])
    # print(res["masks"])
    for res_id in res:
        print(res_id["boxes"])
        print(res_id["labels"])
        print(res_id["scores"])
        print(res_id["masks"].shape)
        masks=res_id["masks"].cpu().numpy()
        for i in range(res_id["boxes"].shape[0]):
            out_array+=masks[i,0,:,:]*100
            # predict_other = (predict_other.squeeze().cpu().numpy())

cv2.imwrite(out_predict,out_array)
