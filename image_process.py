"""
time: 2022/04/14
author: cong
theme: image read show
"""
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np

# yolov4读入图像的两种方式

# opencv
image0 = cv2.imread('1_10.jpg')  # BGR通道
image1 = cv2.cvtColor(image0, cv2.COLOR_BGR2RGB)
image2 = Image.fromarray(np.uint8(image1)) # 转变成Image
image3 = image2.resize((416, 416), Image.BICUBIC)
image_data4  = np.expand_dims(np.array(image3, dtype='float32')/255.0, 0)
# Image
imagea = Image.open('1_10.jpg')
imageb = imagea.resize((416, 416), Image.BICUBIC)
image_datac  = np.expand_dims(np.array(imageb, dtype='float32')/255.0, 0)



# image = cv2.imread('1_10.jpg')  # BGR通道
# plt.imshow(a)
# plt.show()            # 显示不正常，cv2读进来的为BGR通道的数据
# cv2.imshow('a:', a)   # 显示正常，show的时候已经转换为RGB
# cv2.imshow('a:', cv2.cvtColor(a, cv2.COLOR_BGR2RGB)) # 多此一举，显示的为BGR通道
# cv2.waitKey(0)

