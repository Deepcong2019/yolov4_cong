"""
time: 2022/03/25
author: cong
object_detection：yolo_v4推理过程中的详细细节
"""
import datetime

import numpy as np
from model import yolo_body
from PIL import Image
import tensorflow as tf
import time
from keras import backend as K
# from keras.layers import Input
# a= Input((None, None, 3))
# #<tf.Tensor 'input_2:0' shape=(?, ?, ?, 3) dtype=float32>


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# ---------------------------------------------------#
#   获得目标类别
#---------------------------------------------------#
def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)
#---------------------------------------------------#
#   获得先验框
#---------------------------------------------------#
def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path, encoding='utf-8') as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    anchors = np.array(anchors).reshape(-1, 2)
    return anchors, len(anchors)
#---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
#---------------------------------------------------------#
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image
#---------------------------------------------------#
#   对输入图像进行resize
#---------------------------------------------------#
def resize_image(image, size, letterbox_image):
    # ---------------------------------------------------------------------#
    #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
    #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
    # ---------------------------------------------------------------------#
    iw, ih = image.size
    w, h = size
    if letterbox_image:
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)
        image = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        # 将image 贴到new_image的图像上，起始位置在((w-nw)//2, (h-nh)//2)
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image


def preprocess_input(image):
    image /= 255.0
    return image


class_path = 'model_data/my_class.txt'
model_path = 'model_data/ep100-loss15.533-val_loss15.416.h5'
img_path = '111.jpg'
anchors_path = 'model_data/yolo_anchors.txt'
max_boxes = 100   # 最大框的数量
confidence = 0.5  #   只有得分大于置信度的预测框会被保留下来
nms_iou = 0.3     #   非极大抑制所用到的nms_iou大小
#   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
#   在多次测试后，发现关闭letterbox_image直接resize的效果更好
# ---------------------------------------------------------------------#
letterbox_image  = False
input_shape = [416, 416]
anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
class_names, num_classes = get_classes(class_path)
anchors_all, num_anchors = get_anchors(anchors_path)

print('names:', class_names)
image = Image.open(img_path)
image = cvtColor(image)    # print('image_shape:', np.shape(image))# image_shape: (1080, 1920, 3)
image_data = resize_image(image, input_shape, letterbox_image=False)
#   添加上batch_size维度，并进行归一化,
#   np.expand_dims(a, axis=0)表示在0位置添加数据,
image_data = np.expand_dims(preprocess_input(np.array(image_data, dtype='float32')), 0)

def concat(a, numbers):
    resa = a
    for i in range(numbers-1):
        resa = np.concatenate((resa, a))
    return resa


image_data = concat(image_data, 10)
num = len(image_data)
#print(image_data.shape) (1, 416, 416, 3)
# 加载模型
model = yolo_body((None, None, 3), anchors_mask, num_classes)
model.load_weights(model_path)
print('model loaded')
t1 = datetime.datetime.now()
model_output = model.predict(image_data)
output = [[model_output[0][i],model_output[1][i],model_output[2][i]] for i in range(num)]
t2 = datetime.datetime.now()

print('time_consuming:', t2-t1)
# model.summary()
print('model:', model)