"""
time : 2022/04/07
author : cong
"""
import copy

from PIL import Image
import numpy as np
import cv2


#   将图像转换成RGB图像，防止灰度图在预测时报错。代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
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
# 生成随机数
def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


# 图像归一化
def preprocess_input(image):
    image = image.astype(np.float32)
    image /= 255.0
    return image
# ================================================================================ =======#
# train_lines
train_annotation_path = '2007_train.txt'
with open(train_annotation_path) as f:
    train_lines = f.readlines()

input_shape = [416, 416]
max_boxes = 100
image_datas = []
box_datas = []

# train_lines[0]: '/Users/cong/yolov4-keras-master/VOCdevkit/VOC2007/JPEGImages/007826.jpg 80,217,320,273,10 197,193,257,326,8 258,180,312,314,8 10,195,93,358,8 82,252,243,372,8\n'
annotation_line = train_lines[0]
line = annotation_line.split()  # 默认切分 空格，换行，制表符,返回列表。# ['/Users/cong/yolov4-keras-master/VOCdevkit/VOC2007/JPEGImages/007826.jpg', '80,217,320,273,10','197,193,257,326,8', '258,180,312,314,8', '10,195,93,358,8', '82,252,243,372,8']
#   读取图像并转换成RGB图像
# line[0]:'/Users/cong/yolov4-keras-master/VOCdevkit/VOC2007/JPEGImages/2009_002295.jpg'
image = Image.open(line[0])
# cvtColor函数,,比如说截图。
if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
    image = image
else:
    image = image.convert('RGB')
# image.show()

#   获得图像的高宽与目标高宽
iw, ih = image.size
h, w = input_shape
#   获得预测框, int向下取整, box.split(','): ['80', '217', '320', '273', '10']
box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])
# [[80, 217, 320, 273, 10],]
box_data = np.zeros((max_boxes, 5))
box_ori = box

# 原图像
ori = resize_image(image, input_shape, letterbox_image= False)
# ori.show()
image_ori = np.array(ori)
# 原图像的box
if len(box) > 0:
    np.random.shuffle(box)
    if len(box) > max_boxes:
        box = box[:max_boxes]
    box_ori[:, [0, 2]] = box[:, [0, 2]] * input_shape[0] / iw
    box_ori[:, [1, 3]] = box[:, [1, 3]] * input_shape[1] / ih
    box_data[:len(box)] = box_ori


image_datas.append(preprocess_input(np.array(image_ori)))
box_datas.append(box_data)

# for i in range(len(box_ori)):
#     cv2.rectangle(image_ori, (box_ori[i][0] ,box_ori[i][1]), (box_ori[i][2], box_ori[i][3]), (0, 255, 0), 2)
# cv2.imshow("fff", image_ori)
# cv2.waitKey(0)








# ------------------------------------------#
#   左右镜像
# ------------------------------------------#
flip = True
if flip: image_flip = ori.transpose(Image.FLIP_LEFT_RIGHT)  # 左右翻转
# image_flip.show()
image_flip = np.array(image_flip)
box_flip = copy.deepcopy(box_ori)
box_data = np.zeros((max_boxes, 5))
if flip: box_flip[:, [0, 2]] = w - box_ori[:, [2, 0]]
box_data[:len(box)] = box_flip
image_datas.append(preprocess_input(np.array(image_flip)))
box_datas.append(box_data)







# for i in range(len(box_flip)):
#     print((box_flip[i][0] ,box_flip[i][1]), (box_flip[i][2], box_flip[i][3]))
#     cv2.rectangle(image_flip, (box_flip[i][0], box_flip[i][1]), (box_flip[i][2], box_flip[i][3]), (0, 255, 0), 2)
# cv2.imshow("fff", image_flip)
# cv2.waitKey(0)
#







# ------------------------------------------#
#   色域扭曲 RGB==>HSV==>RGB
# ------------------------------------------#
# hue = .1
# sat = 1.5
# val = 1.5

# hue = rand(-hue, hue)
# sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
# val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
# x = cv2.cvtColor(np.array(image, np.float32) / 255, cv2.COLOR_RGB2HSV)
# x[..., 0] += hue * 360
# x[..., 0][x[..., 0] > 1] -= 1
# x[..., 0][x[..., 0] < 0] += 1
# x[..., 1] *= sat
# x[..., 2] *= val
# x[x[:, :, 0] > 360, 0] = 360
# x[:, :, 1:][x[:, :, 1:] > 1] = 1
# x[x < 0] = 0
# image_color = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) * 255  # numpy array, 0 to 1
#
# box_color = box_ori
#
#
# image_datas.append(preprocess_input(np.array(image_ori)))
# box_datas.append(box)















