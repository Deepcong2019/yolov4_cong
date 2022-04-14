"""
time: 2022/03/30
author: cong
theme： mosaic
"""
from PIL import Image
from random import shuffle, sample
import numpy as np
import cv2


# 将图像转换成RGB图像，防止灰度图在预测时报错。代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image


# 生成随机数
def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


def merge_bboxes(bboxes, cutx, cuty):
    merge_bbox = []
    for i in range(len(bboxes)):
        for box in bboxes[i]:
            tmp_box = []
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

            if i == 0:
                if y1 > cuty or x1 > cutx:
                    continue
                if y2 >= cuty and y1 <= cuty:
                    y2 = cuty
                if x2 >= cutx and x1 <= cutx:
                    x2 = cutx

            if i == 1:
                if y2 < cuty or x1 > cutx:
                    continue
                if y2 >= cuty and y1 <= cuty:
                    y1 = cuty
                if x2 >= cutx and x1 <= cutx:
                    x2 = cutx

            if i == 2:
                if y2 < cuty or x2 < cutx:
                    continue
                if y2 >= cuty and y1 <= cuty:
                    y1 = cuty
                if x2 >= cutx and x1 <= cutx:
                    x1 = cutx

            if i == 3:
                if y1 > cuty or x2 < cutx:
                    continue
                if y2 >= cuty and y1 <= cuty:
                    y2 = cuty
                if x2 >= cutx and x1 <= cutx:
                    x1 = cutx
            tmp_box.append(x1)
            tmp_box.append(y1)
            tmp_box.append(x2)
            tmp_box.append(y2)
            tmp_box.append(box[-1])
            merge_bbox.append(tmp_box)
    return merge_bbox
# ================================================================================ #
# train_lines
train_annotation_path = '2007_train.txt'
with open(train_annotation_path) as f:
    train_lines = f.readlines()

input_shape = [416, 416]
h, w = input_shape
max_boxes = 100
max_boxes = 100
hue = .1
sat = 1.5
val = 1.5
# train_lines[0]: '.../JPEGImages/007826.jpg 80,217,320,273,10 197,193,257,326,8\n'
lines = train_lines[:10]
annotation_line = sample(lines, 4)

min_offset_x = rand(0.25, 0.75)
min_offset_y = rand(0.25, 0.75)

nws = [int(w * rand(0.4, 1)), int(w * rand(0.4, 1)), int(w * rand(0.4, 1)),
       int(w * rand(0.4, 1))]
nhs = [int(h * rand(0.4, 1)), int(h * rand(0.4, 1)), int(h * rand(0.4, 1)),
       int(h * rand(0.4, 1))]

place_x = [int(w * min_offset_x) - nws[0], int(w * min_offset_x) - nws[1], int(w * min_offset_x),
           int(w * min_offset_x)]
place_y = [int(h * min_offset_y) - nhs[0], int(h * min_offset_y), int(h * min_offset_y),
           int(h * min_offset_y) - nhs[3]]

image_datas = []
box_datas = []
index = 0
for line in annotation_line:
    # 每一行进行分割
    # 默认切分 空格，换行，制表符,返回列表。
    # ['/Users/cong/yolov4-keras-master/VOCdevkit/VOC2007/JPEGImages/007826.jpg', '80,217,320,273,10','197,193,257,326,8', '258,180,312,314,8', '10,195,93,358,8', '82,252,243,372,8']
    #   读取图像并转换成RGB图像
    # line[0]:'/Users/cong/yolov4-keras-master/VOCdevkit/VOC2007/JPEGImages/2009_002295.jpg'

    line_content = line.split()
    # 打开图片
    image = Image.open(line_content[0])
    image = cvtColor(image)

    # 图片的大小
    iw, ih = image.size
    # 保存框的位置
    box = np.array([np.array(list(map(int, box.split(',')))) for box in line_content[1:]])

    # 是否翻转图片
    flip = rand() < .5
    if flip and len(box) > 0:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        box[:, [0, 2]] = iw - box[:, [2, 0]]

    nw = nws[index]
    nh = nhs[index]
    image = image.resize((nw, nh), Image.BICUBIC)

    # 将图片进行放置，分别对应四张分割图片的位置
    dx = place_x[index]
    dy = place_y[index]
    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    new_image.paste(image, (dx, dy))
    image_data = np.array(new_image)

    index = index + 1
    box_data = []
    # 对box进行重新处理
    if len(box) > 0:
        np.random.shuffle(box)
        box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
        box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
        box[:, 0:2][box[:, 0:2] < 0] = 0
        box[:, 2][box[:, 2] > w] = w
        box[:, 3][box[:, 3] > h] = h
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w > 1, box_h > 1)]
        box_data = np.zeros((len(box), 5))
        box_data[:len(box)] = box

    image_datas.append(image_data)
    box_datas.append(box_data)

# 将图片分割，放在一起
cutx = int(w * min_offset_x)
cuty = int(h * min_offset_y)

new_image = np.zeros([h, w, 3])
new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]

# 进行色域变换
hue = rand(-hue, hue)
sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
x = cv2.cvtColor(np.array(new_image / 255, np.float32), cv2.COLOR_RGB2HSV)
x[..., 0] += hue * 360
x[..., 0][x[..., 0] > 1] -= 1
x[..., 0][x[..., 0] < 0] += 1
x[..., 1] *= sat
x[..., 2] *= val
x[x[:, :, 0] > 360, 0] = 360
x[:, :, 1:][x[:, :, 1:] > 1] = 1
x[x < 0] = 0
new_image = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) * 255
image_uint8 = new_image.astype(np.uint8)
image = Image.fromarray(image_uint8)
image.show()
image.save('image_mosaic.jpg')
# 对框进行进一步的处理
new_boxes = merge_bboxes(box_datas, cutx, cuty)

# 将box进行调整
box_data = np.zeros((max_boxes, 5))
if len(new_boxes) > 0:
    if len(new_boxes) > max_boxes: new_boxes = new_boxes[:max_boxes]
    box_data[:len(new_boxes)] = new_boxes

