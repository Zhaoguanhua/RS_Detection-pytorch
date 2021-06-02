# -*- coding:utf-8 -*-
"""
@Author :zhaoguanhua
@Email   :
@Time    :2020/12/1 16:16
@File    :predict.py
@Software:PyCharm
"""
import os
import numpy as np
import torch
from instance_detection.model import get_instance_segmentation_model
import cv2
from torchvision.transforms import functional as F

_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.857, 0.857, 0.857,
        1.000, 1.000, 1.000
    ]
).astype(np.float32).reshape(-1, 3)

def random_color(rgb=False,maximum=255):
    idx=np.random.randint(0,len(_COLORS))
    ret=_COLORS[idx]*maximum
    if not rgb:
        ret=ret[::-1]


    return ret


def inference(input_path,output_path,model):
    # converts the image into a PyTorch Tensor
    img2 = cv2.imread(input_path)

    img = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img = F.to_tensor(img)

    with torch.no_grad():
        res = model([img.to(device)])
        img2 = img2.astype("float64")

        for res_id in res:
            if "masks" in res_id:
                masks = res_id["masks"].round().cpu().numpy().astype("uint8")
            boxes = res_id["boxes"]
            for i in range(res_id["boxes"].shape[0]):
                if res_id["scores"][i] > 0.8:
                    color = random_color(rgb=True)
                    x1, y1, x2, y2 = boxes[i]
                    # 提取边界
                    if "masks" in res_id:
                        contours, hierarchy = cv2.findContours(masks[i, 0, :, :],
                                                               cv2.RETR_CCOMP,
                                                               cv2.CHAIN_APPROX_NONE)
                        zeros = np.zeros(img2.shape, dtype=np.uint8)
                        mask = cv2.fillPoly(zeros, contours, color=(int(color[0]), int(color[1]), int(color[2])
                                                                    ), lineType=cv2.LINE_AA)
                        img2 += 0.5 * mask
                        cv2.polylines(img2, contours, isClosed=True, thickness=1, color=(255, 255, 255))

                    img2 = cv2.rectangle(img2, (int(x1), int(y1)), (int(x2), int(y2)),
                                         (int(color[0]), int(color[1]), int(color[2])), 1)

    cv2.imwrite(output_path, img2)


if __name__ == '__main__':

    test_dir=r"D:\project\Python\Mask_RCNN_Pytorch\PennFudanPed\PNGImages"
    input_files=os.listdir(test_dir)

    model_path=r"maskrcnn_building_epoch9.pth"
    device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model=get_instance_segmentation_model(num_classes=2)
    # print(model)

    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    for file_name in input_files:
        input_path=os.path.join(test_dir,file_name)

        inference(input_path,file_name,model)




