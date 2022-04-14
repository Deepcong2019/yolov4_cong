"""
time : 2022/03/29
author : cong
theme: yolov4_get_random_data
"""
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


# 生成随机数
def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


# ================================================================================ #
# train_lines
train_annotation_path = '2007_train.txt'
with open(train_annotation_path) as f:
    train_lines = f.readlines()

input_shape = [416, 416]
max_boxes = 100

random = True
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
#   获得图像的高宽与目标高宽
iw, ih = image.size
h, w = input_shape
#   获得预测框, int向下取整, box.split(','): ['80', '217', '320', '273', '10']
box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])
# array([[80, 217, 320, 273, 10],
#       [197, 193, 257, 326, 8],
#       [258, 180, 312, 314, 8],
#      [10, 195, 93, 358, 8],
#      [82, 252, 243, 372, 8]])
#    not random ==================================
if not random:
    scale = min(w / iw, h / ih) # w,h:416,,iw:375, ih:500
    # nw,nh中 ，最多和w,h一样大
    nw = int(iw * scale)
    nh = int(ih * scale)
    dx = (w - nw) // 2
    dy = (h - nh) // 2

    # ---------------------------------#
    #   将图像多余的部分加上灰条
    # ---------------------------------#
    image = image.resize((nw, nh), Image.BICUBIC)
    image.show()
    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    new_image.show()
    new_image.paste(image, (dx, dy))
    new_image.show()
    image_data = np.array(new_image, np.float32)
    # 对真实框进行调整
    box_data = np.zeros((max_boxes, 5))
    if len(box) > 0:
        np.random.shuffle(box)
        box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
        box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
        box[:, 0:2][box[:, 0:2] < 0] = 0
        box[:, 2][box[:, 2] > w] = w
        box[:, 3][box[:, 3] > h] = h
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w > 1, box_h > 1)]# 从box中取出符合条件的。logical_and逻辑与。
        if len(box) > max_boxes:
            box = box[:max_boxes]
        box_data[:len(box)] = box

# random ===========================================
# 多余部分加上灰度条//翻转//色域扭曲
if random:
    jitter = .3
    hue = .1
    sat = 1.5
    val = 1.5
    new_ar = w / h * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
    scale = rand(.25, 2)
    if new_ar < 1:
        nh = int(scale * h)
        nw = int(nh * new_ar)
    else:
        nw = int(scale * w)
        nh = int(nw / new_ar)
    image = image.resize((nw, nh), Image.BICUBIC)

    # ------------------------------------------#
    #   将图像多余的部分加上灰条
    # ------------------------------------------#
    dx = int(rand(0, w - nw))
    dy = int(rand(0, h - nh))
    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    # new_image.show()
    new_image.paste(image, (dx, dy))
    image = new_image
    # image.show()

    # ------------------------------------------#
    #   翻转图像
    # ------------------------------------------#
    flip = rand() < .5
    if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)  # 左右翻转

    # ------------------------------------------#
    #   色域扭曲 RGB==>HSV==>RGB
    # ------------------------------------------#
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
    val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
    x = cv2.cvtColor(np.array(image, np.float32) / 255, cv2.COLOR_RGB2HSV)
    x[..., 0] += hue * 360
    x[..., 0][x[..., 0] > 1] -= 1
    x[..., 0][x[..., 0] < 0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x[:, :, 0] > 360, 0] = 360
    x[:, :, 1:][x[:, :, 1:] > 1] = 1
    x[x < 0] = 0
    image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) * 255  # numpy array, 0 to 1
    image_uint8 = image_data.astype(np.uint8)
    image = Image.fromarray(image_uint8)
    image.show()
    cv2.imshow('image:', cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    # ---------------------------------#
    #   对真实框进行调整
    # ---------------------------------#
    box_data = np.zeros((max_boxes, 5))
    if len(box) > 0:
        np.random.shuffle(box)
        box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
        box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
        if flip: box[:, [0, 2]] = w - box[:, [2, 0]]
        box[:, 0:2][box[:, 0:2] < 0] = 0
        box[:, 2][box[:, 2] > w] = w
        box[:, 3][box[:, 3] > h] = h
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
        if len(box) > max_boxes: box = box[:max_boxes]
        box_data[:len(box)] = box

















