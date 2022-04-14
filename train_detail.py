"""
time: 2022/03/27
author: cong

训练自己的目标检测模型一定需要注意以下几点：
1、训练前仔细检查自己的格式是否满足要求，该库要求数据集格式为VOC格式，需要准备好的内容有输入图片和标签
   输入图片为.jpg图片，无需固定大小，传入训练前会自动进行resize。
   灰度图会自动转成RGB图片进行训练，无需自己修改。
   输入图片如果后缀非jpg，需要自己批量转成jpg后再开始训练。

   标签为.xml格式，文件中会有需要检测的目标信息，标签文件和输入图片文件相对应。

2、训练好的权值文件保存在logs文件夹中，每个epoch都会保存一次，如果只是训练了几个step是不会保存的，epoch和step的概念要捋清楚一下。
   在训练过程中，该代码并没有设定只保存最低损失的，因此按默认参数训练完会有100个权值，如果空间不够可以自行删除。
   这个并不是保存越少越好也不是保存越多越好，有人想要都保存、有人想只保存一点，为了满足大多数的需求，还是都保存可选择性高。

3、损失值的大小用于判断是否收敛，比较重要的是有收敛的趋势，即验证集损失不断下降，如果验证集损失基本上不改变的话，模型基本上就收敛了。
   损失值的具体大小并没有什么意义，大和小只在于损失的计算方式，并不是接近于0才好。如果想要让损失好看点，可以直接到对应的损失函数里面除上10000。
   训练过程中的损失值会保存在logs文件夹下的loss_%Y_%m_%d_%H_%M_%S文件夹中

4、调参是一门蛮重要的学问，没有什么参数是一定好的，现有的参数是我测试过可以正常训练的参数，因此我会建议用现有的参数。
   但是参数本身并不是绝对的，比如随着batch的增大学习率也可以增大，效果也会好一些；过深的网络不要用太大的学习率等等。
   这些都是经验上，只能靠各位同学多查询资料和自己试试了。
"""

from matplotlib import pyplot as plt
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
from keras.layers import Input, Lambda
from keras.models import Model
from model import yolo_body
import numpy as np
import tensorflow as tf
import math
from PIL import Image
from random import shuffle, sample
import cv2
import os
import keras
import scipy.signal
import matplotlib
matplotlib.use('Agg')
# Agg 渲染器是非交互式的后端，没有GUI界面，所以不显示图片，它是用来生成图像文件。
# Qt5Agg 是意思是Agg渲染器输出到Qt5绘图面板，它是交互式的后端，拥有在屏幕上展示的能力3。


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


#   获得目标类别
def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)


#   获得先验框
def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path, encoding='utf-8') as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    anchors = np.array(anchors).reshape(-1, 2)
    return anchors, len(anchors)


#   将图像转换成RGB图像，防止灰度图在预测时报错。代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image


# 图像归一化
def preprocess_input(image):
    image /= 255.0
    return image


# 1、__init__方法的使用和功能：
#     1、用来构造初始化函数，用来给类的实例进行初始化属性，所以不需要返回值。
#     2、在创建实例时系统自动调用
#     3、自定义类如果不定义的话，默认调用父类的，同理继承也是，子类若无，调用父类，若有，调用自己的.
class YoloDatasets(keras.utils.Sequence):
    def __init__(self, annotation_lines, input_shape, anchors, batch_size, num_classes, anchors_mask, mosaic, train):
        self.annotation_lines = annotation_lines
        self.length = len(self.annotation_lines)
        self.input_shape = input_shape
        self.anchors = anchors
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.anchors_mask = anchors_mask
        self.mosaic = mosaic
        self.train = train

    # 覆盖keras.utils.Sequence中的__len__函数, 返回类实例的长度
    def __len__(self):
        return math.ceil(len(self.annotation_lines) / float(self.batch_size))  # ceil:大于结果的整数

    # 覆盖keras.utils.Sequence中的__getitem__函数，这个是返回一个可迭代的对象，如果一个类实现了这个魔法函数，那么这个类就是可迭代对象
    def __getitem__(self, index):
        image_data = []
        box_data = []
        for i in range(index * self.batch_size, (index + 1) * self.batch_size):
            i = i % self.length
            # ---------------------------------------------------#
            #   训练时进行数据的随机增强
            #   验证时不进行数据的随机增强
            # ---------------------------------------------------#
            if self.mosaic:
                if self.rand() < 0.5:
                    lines = sample(self.annotation_lines, 3)  # sample(序列a，n)从序列a中随机抽取n个元素，并以list形式返回。
                    lines.append(self.annotation_lines[i])
                    shuffle(lines)
                    image, box = self.get_random_data_with_Mosaic(lines, self.input_shape)
                else:
                    image, box = self.get_random_data(self.annotation_lines[i], self.input_shape, random=self.train)
            else:
                image, box = self.get_random_data(self.annotation_lines[i], self.input_shape, random=self.train)
            image_data.append(preprocess_input(np.array(image)))
            box_data.append(box)

        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = self.preprocess_true_boxes(box_data, self.input_shape, self.anchors, self.num_classes)
        return [image_data, *y_true], np.zeros(self.batch_size) # 在列表前加*号，会将列表拆分成一个一个的独立元素，
        # return [image_data, *y_true]

    def on_epoch_begin(self):
        shuffle(self.annotation_lines)

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, annotation_line, input_shape, max_boxes=100, jitter=.3, hue=.1, sat=1.5, val=1.5,
                        random=True):
        line = annotation_line.split()
        # ------------------------------#
        #   读取图像并转换成RGB图像
        # ------------------------------#
        image = Image.open(line[0])
        image = cvtColor(image)
        # ------------------------------#
        #   获得图像的高宽与目标高宽
        # ------------------------------#
        iw, ih = image.size
        h, w = input_shape
        # ------------------------------#
        #   获得预测框
        # ------------------------------#
        box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

        if not random:
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2

            # ---------------------------------#
            #   将图像多余的部分加上灰条
            # ---------------------------------#
            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)

            # ---------------------------------#
            #   对真实框进行调整
            # ---------------------------------#
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
                box = box[np.logical_and(box_w > 1, box_h > 1)]
                if len(box) > max_boxes: box = box[:max_boxes]
                box_data[:len(box)] = box

            return image_data, box_data

        # ------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        # ------------------------------------------#
        new_ar = w / h * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(.25, 2)
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
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        # ------------------------------------------#
        #   翻转图像
        # ------------------------------------------#
        flip = self.rand() < .5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # ------------------------------------------#
        #   色域扭曲
        # ------------------------------------------#
        hue = self.rand(-hue, hue)
        sat = self.rand(1, sat) if self.rand() < .5 else 1 / self.rand(1, sat)
        val = self.rand(1, val) if self.rand() < .5 else 1 / self.rand(1, val)
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

        return image_data, box_data

    def merge_bboxes(self, bboxes, cutx, cuty):
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

    def get_random_data_with_Mosaic(self, annotation_line, input_shape, max_boxes=100, hue=.1, sat=1.5, val=1.5):
        h, w = input_shape
        min_offset_x = self.rand(0.25, 0.75)
        min_offset_y = self.rand(0.25, 0.75)

        nws = [int(w * self.rand(0.4, 1)), int(w * self.rand(0.4, 1)), int(w * self.rand(0.4, 1)),
               int(w * self.rand(0.4, 1))]
        nhs = [int(h * self.rand(0.4, 1)), int(h * self.rand(0.4, 1)), int(h * self.rand(0.4, 1)),
               int(h * self.rand(0.4, 1))]

        place_x = [int(w * min_offset_x) - nws[0], int(w * min_offset_x) - nws[1], int(w * min_offset_x),
                   int(w * min_offset_x)]
        place_y = [int(h * min_offset_y) - nhs[0], int(h * min_offset_y), int(h * min_offset_y),
                   int(h * min_offset_y) - nhs[3]]

        image_datas = []
        box_datas = []
        index = 0
        for line in annotation_line:
            # 每一行进行分割
            line_content = line.split()
            # 打开图片
            image = Image.open(line_content[0])
            image = cvtColor(image)

            # 图片的大小
            iw, ih = image.size
            # 保存框的位置
            box = np.array([np.array(list(map(int, box.split(',')))) for box in line_content[1:]])

            # 是否翻转图片
            flip = self.rand() < .5
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
        hue = self.rand(-hue, hue)
        sat = self.rand(1, sat) if self.rand() < .5 else 1 / self.rand(1, sat)
        val = self.rand(1, val) if self.rand() < .5 else 1 / self.rand(1, val)
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

        # 对框进行进一步的处理
        new_boxes = self.merge_bboxes(box_datas, cutx, cuty)

        # 将box进行调整
        box_data = np.zeros((max_boxes, 5))
        if len(new_boxes) > 0:
            if len(new_boxes) > max_boxes: new_boxes = new_boxes[:max_boxes]
            box_data[:len(new_boxes)] = new_boxes
        return new_image, box_data

    def preprocess_true_boxes(self, true_boxes, input_shape, anchors, num_classes):
        assert (true_boxes[..., 4] < num_classes).all(), 'class id must be less than num_classes'
        # -----------------------------------------------------------#
        #   获得框的坐标和图片的大小
        # -----------------------------------------------------------#
        true_boxes = np.array(true_boxes, dtype='float32')
        input_shape = np.array(input_shape, dtype='int32')

        # -----------------------------------------------------------#
        #   一共有三个特征层数
        # -----------------------------------------------------------#
        num_layers = len(self.anchors_mask)
        # -----------------------------------------------------------#
        #   m为图片数量，grid_shapes为网格的shape
        # -----------------------------------------------------------#
        m = true_boxes.shape[0]
        grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8}[l] for l in range(num_layers)]
        # -----------------------------------------------------------#
        #   y_true的格式为(m,13,13,3,85)(m,26,26,3,85)(m,52,52,3,85)
        # -----------------------------------------------------------#
        y_true = [np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], len(self.anchors_mask[l]), 5 + num_classes),
                           dtype='float32') for l in range(num_layers)]

        # -----------------------------------------------------------#
        #   通过计算获得真实框的中心和宽高
        #   中心点(m,n,2) 宽高(m,n,2)
        # -----------------------------------------------------------#
        boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
        boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
        # -----------------------------------------------------------#
        #   将真实框归一化到小数形式
        # -----------------------------------------------------------#
        true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
        true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

        # -----------------------------------------------------------#
        #   [9,2] -> [1,9,2]
        # -----------------------------------------------------------#
        anchors = np.expand_dims(anchors, 0)
        anchor_maxes = anchors / 2.
        anchor_mins = -anchor_maxes

        # -----------------------------------------------------------#
        #   长宽要大于0才有效
        # -----------------------------------------------------------#
        valid_mask = boxes_wh[..., 0] > 0

        for b in range(m):
            # -----------------------------------------------------------#
            #   对每一张图进行处理
            # -----------------------------------------------------------#
            wh = boxes_wh[b, valid_mask[b]]
            if len(wh) == 0: continue
            # -----------------------------------------------------------#
            #   [n,2] -> [n,1,2]
            # -----------------------------------------------------------#
            wh = np.expand_dims(wh, -2)
            box_maxes = wh / 2.
            box_mins = - box_maxes

            # -----------------------------------------------------------#
            #   计算所有真实框和先验框的交并比
            #   intersect_area  [n,9]
            #   box_area        [n,1]
            #   anchor_area     [1,9]
            #   iou             [n,9]
            # -----------------------------------------------------------#
            intersect_mins = np.maximum(box_mins, anchor_mins)
            intersect_maxes = np.minimum(box_maxes, anchor_maxes)
            intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
            intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

            box_area = wh[..., 0] * wh[..., 1]
            anchor_area = anchors[..., 0] * anchors[..., 1]

            iou = intersect_area / (box_area + anchor_area - intersect_area)
            # -----------------------------------------------------------#
            #   维度是[n,] 感谢 消尽不死鸟 的提醒
            # -----------------------------------------------------------#
            best_anchor = np.argmax(iou, axis=-1)

            for t, n in enumerate(best_anchor):
                # -----------------------------------------------------------#
                #   找到每个真实框所属的特征层
                # -----------------------------------------------------------#
                for l in range(num_layers):
                    if n in self.anchors_mask[l]:
                        # -----------------------------------------------------------#
                        #   floor用于向下取整，找到真实框所属的特征层对应的x、y轴坐标
                        # -----------------------------------------------------------#
                        i = np.floor(true_boxes[b, t, 0] * grid_shapes[l][1]).astype('int32')
                        j = np.floor(true_boxes[b, t, 1] * grid_shapes[l][0]).astype('int32')
                        # -----------------------------------------------------------#
                        #   k指的的当前这个特征点的第k个先验框
                        # -----------------------------------------------------------#
                        k = self.anchors_mask[l].index(n)
                        # -----------------------------------------------------------#
                        #   c指的是当前这个真实框的种类
                        # -----------------------------------------------------------#
                        c = true_boxes[b, t, 4].astype('int32')
                        # -----------------------------------------------------------#
                        #   y_true的shape为(m,13,13,3,85)(m,26,26,3,85)(m,52,52,3,85)
                        #   最后的85可以拆分成4+1+80，4代表的是框的中心与宽高、
                        #   1代表的是置信度、80代表的是种类
                        # -----------------------------------------------------------#
                        y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
                        y_true[l][b, j, i, k, 4] = 1
                        y_true[l][b, j, i, k, 5 + c] = 1

        return y_true


#   将预测值的每个特征层调成真实值
def get_anchors_and_decode(feats, anchors, num_classes, input_shape, calc_loss=False):
    num_anchors = len(anchors)
    # 生成grid
    grid_shape = feats.shape[1:3]
    grid_x = np.tile(np.reshape(range(grid_shape[1]), [1, -1, 1, 1]), [grid_shape[0], 1, num_anchors, 1])
    grid_y = np.tile(np.reshape(range(grid_shape[0]), [-1, 1, 1, 1]), [1, grid_shape[1], num_anchors, 1])
    grid = np.concatenate([grid_x, grid_y], axis=3)
    # 生成anchors_reshape
    anchors_reshape = np.reshape(anchors, [1, 1, num_anchors, 2])
    anchors_reshape = np.tile(anchors_reshape, [grid_shape[0], grid_shape[1], 1, 1])
    #
    feats = np.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])
    out_xy = feats[..., :2]  # 网络预测输出的x、y的偏移量
    out_wh = feats[..., 2:4] # 网络预测输出的w、h的缩放尺度
    xy = sigmoid(out_xy) + grid   # 实际的网格中x、y的坐标
    wh = np.exp(out_wh) * anchors_reshape
    # 在Python中“/”表示浮点数除法，返回浮点结果，也就是结果为浮点数，
    # 而“//”在Python中表示整数除法，返回不大于结果的一个最大的整数，意思就是除法结果向下取整。
    box_xy = xy / grid_shape[::-1]  #对应位置的元素除以13
    box_wh = wh / input_shape[::-1]  #对应位置的元素除以416
    # ------------------------------------------#
    #   获得预测框的置信度
    # ------------------------------------------#
    box_confidence = sigmoid(feats[..., 4:5])
    box_class_probs = sigmoid(feats[..., 5:])
    # ---------------------------------------------------------------------#
    #   在计算loss的时候返回grid, feats, box_xy, box_wh
    #   在预测的时候返回box_xy, box_wh, box_confidence, box_class_probs
    # ---------------------------------------------------------------------#
    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


def box_ciou(b1, b2):
    """
    输入为：
    ----------
    b1: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
    b2: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh

    返回为：
    -------
    ciou: tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
    """
    # -----------------------------------------------------------#
    #   求出预测框左上角右下角
    #   b1_mins     (batch, feat_w, feat_h, anchor_num, 2)
    #   b1_maxes    (batch, feat_w, feat_h, anchor_num, 2)
    # -----------------------------------------------------------#
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half
    # -----------------------------------------------------------#
    #   求出真实框左上角右下角
    #   b2_mins     (batch, feat_w, feat_h, anchor_num, 2)
    #   b2_maxes    (batch, feat_w, feat_h, anchor_num, 2)
    # -----------------------------------------------------------#
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    # -----------------------------------------------------------#
    #   求真实框和预测框所有的iou
    #   iou         (batch, feat_w, feat_h, anchor_num)
    # -----------------------------------------------------------#
    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    union_area = b1_area + b2_area - intersect_area
    iou = intersect_area / K.maximum(union_area, K.epsilon())

    # -----------------------------------------------------------#
    #   计算中心的差距
    #   center_distance (batch, feat_w, feat_h, anchor_num)
    # -----------------------------------------------------------#
    center_distance = K.sum(K.square(b1_xy - b2_xy), axis=-1)
    enclose_mins = K.minimum(b1_mins, b2_mins)
    enclose_maxes = K.maximum(b1_maxes, b2_maxes)
    enclose_wh = K.maximum(enclose_maxes - enclose_mins, 0.0)
    # -----------------------------------------------------------#
    #   计算对角线距离
    #   enclose_diagonal (batch, feat_w, feat_h, anchor_num)
    # -----------------------------------------------------------#
    enclose_diagonal = K.sum(K.square(enclose_wh), axis=-1)
    ciou = iou - 1.0 * (center_distance) / K.maximum(enclose_diagonal, K.epsilon())

    v = 4 * K.square(tf.math.atan2(b1_wh[..., 0], K.maximum(b1_wh[..., 1], K.epsilon())) - tf.math.atan2(b2_wh[..., 0],
                                                                                                         K.maximum(
                                                                                                             b2_wh[
                                                                                                                 ..., 1],
                                                                                                             K.epsilon()))) / (
                    math.pi * math.pi)
    alpha = v / K.maximum((1.0 - iou + v), K.epsilon())
    ciou = ciou - alpha * v

    ciou = K.expand_dims(ciou, -1)
    return ciou


# ---------------------------------------------------#
#   平滑标签
# ---------------------------------------------------#
def _smooth_labels(y_true, label_smoothing):
    num_classes = tf.cast(K.shape(y_true)[-1], dtype=K.floatx())
    label_smoothing = K.constant(label_smoothing, dtype=K.floatx())
    return y_true * (1.0 - label_smoothing) + label_smoothing / num_classes


# ---------------------------------------------------#
#   用于计算每个预测框与真实框的iou
# ---------------------------------------------------#
def box_iou(b1, b2):
    # ---------------------------------------------------#
    #   num_anchor,1,4
    #   计算左上角的坐标和右下角的坐标
    # ---------------------------------------------------#
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # ---------------------------------------------------#
    #   1,n,4
    #   计算左上角和右下角的坐标
    # ---------------------------------------------------#
    b2 = K.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    # ---------------------------------------------------#
    #   计算重合面积
    # ---------------------------------------------------#
    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou


# ---------------------------------------------------#
#   loss值计算
# ---------------------------------------------------#
def yolo_loss(args, input_shape, anchors, anchors_mask, num_classes, ignore_thresh=.5, label_smoothing=0.1,
              print_loss=False):
    num_layers = len(anchors_mask)
    # ---------------------------------------------------------------------------------------------------#
    #   将预测结果和实际ground truth分开，args是[*model_body.output, *y_true]
    #   y_true是一个列表，包含三个特征层，shape分别为:
    #   (m,13,13,3,85)
    #   (m,26,26,3,85)
    #   (m,52,52,3,85)
    #   yolo_outputs是一个列表，包含三个特征层，shape分别为:
    #   (m,13,13,3,85)
    #   (m,26,26,3,85)
    #   (m,52,52,3,85)
    # ---------------------------------------------------------------------------------------------------#
    y_true = args[num_layers:]
    yolo_outputs = args[:num_layers]

    # -----------------------------------------------------------#
    #   得到input_shpae为416,416
    # -----------------------------------------------------------#
    input_shape = K.cast(input_shape, K.dtype(y_true[0]))

    # -----------------------------------------------------------#
    #   取出每一张图片
    #   m的值就是batch_size
    # -----------------------------------------------------------#
    m = K.shape(yolo_outputs[0])[0]

    loss = 0
    num_pos = 0
    # ---------------------------------------------------------------------------------------------------#
    #   y_true是一个列表，包含三个特征层，shape分别为(m,13,13,3,85),(m,26,26,3,85),(m,52,52,3,85)。
    #   yolo_outputs是一个列表，包含三个特征层，shape分别为(m,13,13,3,85),(m,26,26,3,85),(m,52,52,3,85)。
    # ---------------------------------------------------------------------------------------------------#
    for l in range(num_layers):
        # -----------------------------------------------------------#
        #   以第一个特征层(m,13,13,3,85)为例子
        #   取出该特征层中存在目标的点的位置。(m,13,13,3,1)
        # -----------------------------------------------------------#
        object_mask = y_true[l][..., 4:5]
        # -----------------------------------------------------------#
        #   取出其对应的种类(m,13,13,3,80)
        # -----------------------------------------------------------#
        true_class_probs = y_true[l][..., 5:]
        if label_smoothing:
            true_class_probs = _smooth_labels(true_class_probs, label_smoothing)

        # -----------------------------------------------------------#
        #   将yolo_outputs的特征层输出进行处理、获得四个返回值
        #   其中：
        #   grid        (13,13,1,2) 网格坐标
        #   raw_pred    (m,13,13,3,85) 尚未处理的预测结果
        #   pred_xy     (m,13,13,3,2) 解码后的中心坐标
        #   pred_wh     (m,13,13,3,2) 解码后的宽高坐标
        # -----------------------------------------------------------#
        grid, raw_pred, pred_xy, pred_wh = get_anchors_and_decode(yolo_outputs[l],
                                                                  anchors[anchors_mask[l]], num_classes, input_shape,
                                                                  calc_loss=True)

        # -----------------------------------------------------------#
        #   pred_box是解码后的预测的box的位置
        #   (m,13,13,3,4)
        # -----------------------------------------------------------#
        pred_box = K.concatenate([pred_xy, pred_wh])

        # -----------------------------------------------------------#
        #   找到负样本群组，第一步是创建一个数组，[]
        # -----------------------------------------------------------#
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool')

        # -----------------------------------------------------------#
        #   对每一张图片计算ignore_mask
        # -----------------------------------------------------------#
        def loop_body(b, ignore_mask):
            # -----------------------------------------------------------#
            #   取出n个真实框：n,4
            # -----------------------------------------------------------#
            true_box = tf.boolean_mask(y_true[l][b, ..., 0:4], object_mask_bool[b, ..., 0])
            # -----------------------------------------------------------#
            #   计算预测框与真实框的iou
            #   pred_box    13,13,3,4 预测框的坐标
            #   true_box    n,4 真实框的坐标
            #   iou         13,13,3,n 预测框和真实框的iou
            # -----------------------------------------------------------#
            iou = box_iou(pred_box[b], true_box)

            # -----------------------------------------------------------#
            #   best_iou    13,13,3 每个特征点与真实框的最大重合程度
            # -----------------------------------------------------------#
            best_iou = K.max(iou, axis=-1)

            # -----------------------------------------------------------#
            #   判断预测框和真实框的最大iou小于ignore_thresh
            #   则认为该预测框没有与之对应的真实框
            #   该操作的目的是：
            #   忽略预测结果与真实框非常对应特征点，因为这些框已经比较准了
            #   不适合当作负样本，所以忽略掉。
            # -----------------------------------------------------------#
            ignore_mask = ignore_mask.write(b, K.cast(best_iou < ignore_thresh, K.dtype(true_box)))
            return b + 1, ignore_mask

        # -----------------------------------------------------------#
        #   在这个地方进行一个循环、循环是对每一张图片进行的
        # -----------------------------------------------------------#
        _, ignore_mask = tf.while_loop(lambda b, *args: b < m, loop_body, [0, ignore_mask])

        # -----------------------------------------------------------#
        #   ignore_mask用于提取出作为负样本的特征点
        #   (m,13,13,3)
        # -----------------------------------------------------------#
        ignore_mask = ignore_mask.stack()
        #   (m,13,13,3,1)
        ignore_mask = K.expand_dims(ignore_mask, -1)

        # -----------------------------------------------------------#
        #   真实框越大，比重越小，小框的比重更大。
        # -----------------------------------------------------------#
        box_loss_scale = 2 - y_true[l][..., 2:3] * y_true[l][..., 3:4]

        # -----------------------------------------------------------#
        #   计算Ciou loss
        # -----------------------------------------------------------#
        raw_true_box = y_true[l][..., 0:4]
        ciou = box_ciou(pred_box, raw_true_box)
        ciou_loss = object_mask * box_loss_scale * (1 - ciou)

        # ------------------------------------------------------------------------------#
        #   如果该位置本来有框，那么计算1与置信度的交叉熵
        #   如果该位置本来没有框，那么计算0与置信度的交叉熵
        #   在这其中会忽略一部分样本，这些被忽略的样本满足条件best_iou<ignore_thresh
        #   该操作的目的是：
        #   忽略预测结果与真实框非常对应特征点，因为这些框已经比较准了
        #   不适合当作负样本，所以忽略掉。
        # ------------------------------------------------------------------------------#
        confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[..., 4:5], from_logits=True) + \
                          (1 - object_mask) * K.binary_crossentropy(object_mask, raw_pred[..., 4:5],
                                                                    from_logits=True) * ignore_mask

        class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[..., 5:], from_logits=True)

        location_loss = K.sum(ciou_loss)
        confidence_loss = K.sum(confidence_loss)
        class_loss = K.sum(class_loss)
        # -----------------------------------------------------------#
        #   计算正样本数量
        # -----------------------------------------------------------#
        num_pos += tf.maximum(K.sum(K.cast(object_mask, tf.float32)), 1)
        loss += location_loss + confidence_loss + class_loss
        # if print_loss:
        #   loss = tf.Print(loss, [loss, location_loss, confidence_loss, class_loss, K.sum(ignore_mask)], message='loss: ')

    loss = loss / num_pos
    return loss


class LossHistory(keras.callbacks.Callback):
    def __init__(self, log_dir):
        import datetime
        curr_time = datetime.datetime.now()
        time_str = datetime.datetime.strftime(curr_time, '%Y_%m_%d_%H_%M_%S')
        self.log_dir = log_dir
        self.time_str = time_str
        self.save_path = os.path.join(self.log_dir, "loss_" + str(self.time_str))
        self.losses = []
        self.val_loss = []

        os.makedirs(self.save_path)

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        with open(os.path.join(self.save_path, "epoch_loss_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(logs.get('loss')))
            f.write("\n")
        with open(os.path.join(self.save_path, "epoch_val_loss_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(logs.get('val_loss')))
            f.write("\n")
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth=2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth=2, label='val loss')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15

            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle='--', linewidth=2,
                     label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle='--', linewidth=2,
                     label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('A Loss Curve')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.save_path, "epoch_loss_" + str(self.time_str) + ".png"))

        plt.cla()
        plt.close("all")


class ExponentDecayScheduler(keras.callbacks.Callback):
    def __init__(self,
                 decay_rate,
                 verbose=0):
        super(ExponentDecayScheduler, self).__init__()
        self.decay_rate = decay_rate
        self.verbose = verbose
        self.learning_rates = []

    def on_epoch_end(self, batch, logs=None):
        learning_rate = K.get_value(self.model.optimizer.lr) * self.decay_rate
        K.set_value(self.model.optimizer.lr, learning_rate)
        if self.verbose > 0:
            print('Setting learning rate to %s.' % (learning_rate))


class WarmUpCosineDecayScheduler(keras.callbacks.Callback):
    def __init__(self, T_max, eta_min=0, verbose=0):
        super(WarmUpCosineDecayScheduler, self).__init__()
        self.T_max = T_max
        self.eta_min = eta_min
        self.verbose = verbose
        self.init_lr = 0
        self.last_epoch = 0

    def on_train_begin(self, batch, logs=None):
        self.init_lr = K.get_value(self.model.optimizer.lr)

    def on_epoch_end(self, batch, logs=None):
        learning_rate = self.eta_min + (self.init_lr - self.eta_min) * (
                    1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
        self.last_epoch += 1

        K.set_value(self.model.optimizer.lr, learning_rate)
        if self.verbose > 0:
            print('Setting learning rate to %s.' % (learning_rate))


if __name__ == "__main__":
    # --------------------------------------------------------#
    #   训练前一定要修改classes_path，使其对应自己的数据集
    # --------------------------------------------------------#
    classes_path = 'model_data/voc_classes.txt'
    # ---------------------------------------------------------------------#
    #   anchors_path代表先验框对应的txt文件，一般不修改。
    #   anchors_mask用于帮助代码找到对应的先验框，一般不修改。
    # ---------------------------------------------------------------------#
    anchors_path = 'model_data/yolo_anchors.txt'
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    # ----------------------------------------------------------------------------------------------------------------------------#
    #   权值文件的下载请看README，可以通过网盘下载。模型的 预训练权重 对不同数据集是通用的，因为特征是通用的。
    #   模型的 预训练权重 比较重要的部分是 主干特征提取网络的权值部分，用于进行特征提取。
    #   预训练权重对于99%的情况都必须要用，不用的话主干部分的权值太过随机，特征提取效果不明显，网络训练的结果也不会好
    #
    #   如果训练过程中存在中断训练的操作，可以将model_path设置成logs文件夹下的权值文件，将已经训练了一部分的权值再次载入。
    #   同时修改下方的 冻结阶段 或者 解冻阶段 的参数，来保证模型epoch的连续性。
    #
    #   当model_path = ''的时候不加载整个模型的权值。
    #
    #   此处使用的是整个模型的权重，因此是在train.py进行加载的。
    #   如果想要让模型从主干的预训练权值开始训练，则设置model_path为主干网络的权值，此时仅加载主干。
    #   如果想要让模型从0开始训练，则设置model_path = ''，Freeze_Train = Fasle，此时从0开始训练，且没有冻结主干的过程。
    #   一般来讲，从0开始训练效果会很差，因为权值太过随机，特征提取效果不明显。
    #
    #   网络一般不从0开始训练，至少会使用主干部分的权值，有些论文提到可以不用预训练，主要原因是他们 数据集较大 且 调参能力优秀。
    #   如果一定要训练网络的主干部分，可以了解imagenet数据集，首先训练分类模型，分类模型的 主干部分 和该模型通用，基于此进行训练。
    # ----------------------------------------------------------------------------------------------------------------------------#
    model_path = 'model_data/yolo4_weight.h5'
    # ------------------------------------------------------#
    #   输入的shape大小，一定要是32的倍数
    # ------------------------------------------------------#
    input_shape = [416, 416]
    # ------------------------------------------------------#
    #   Yolov4的tricks应用
    #   mosaic 马赛克数据增强 True or False
    #   实际测试时mosaic数据增强并不稳定，所以默认为False
    #   Cosine_scheduler 余弦退火学习率 True or False
    #   label_smoothing 标签平滑 0.01以下一般 如0.01、0.005
    # ------------------------------------------------------#
    mosaic = False
    Cosine_scheduler = False
    label_smoothing = 0

    # ----------------------------------------------------#
    #   训练分为两个阶段，分别是冻结阶段和解冻阶段。
    #   显存不足与数据集大小无关，提示显存不足请调小batch_size。
    #   受到BatchNorm层影响，batch_size最小为2，不能为1。
    # ----------------------------------------------------#
    # ----------------------------------------------------#
    #   冻结阶段训练参数
    #   此时模型的主干被冻结了，特征提取网络不发生改变
    #   占用的显存较小，仅对网络进行微调
    # ----------------------------------------------------#
    Init_Epoch = 0
    Freeze_Epoch = 50
    Freeze_batch_size = 8
    Freeze_lr = 1e-3
    # ----------------------------------------------------#
    #   解冻阶段训练参数
    #   此时模型的主干不被冻结了，特征提取网络会发生改变
    #   占用的显存较大，网络所有的参数都会发生改变
    # ----------------------------------------------------#
    UnFreeze_Epoch = 100
    Unfreeze_batch_size = 4
    Unfreeze_lr = 1e-4
    # ------------------------------------------------------#
    #   是否进行冻结训练，默认先冻结主干训练后解冻训练。
    # ------------------------------------------------------#
    Freeze_Train = True
    # ------------------------------------------------------#
    #   用于设置是否使用多线程读取数据，1代表关闭多线程
    #   开启后会加快数据读取速度，但是会占用更多内存
    #   keras里开启多线程有些时候速度反而慢了许多
    #   在IO为瓶颈的时候再开启多线程，即GPU运算速度远大于读取图片的速度。
    # ------------------------------------------------------#
    num_workers = 1
    # ----------------------------------------------------#
    #   获得图片路径和标签
    # ----------------------------------------------------#
    train_annotation_path = '2007_train.txt'
    val_annotation_path = '2007_val.txt'

    # ----------------------------------------------------#
    #   获取classes和anchor
    # ----------------------------------------------------#
    class_names, num_classes = get_classes(classes_path)
    anchors, num_anchors = get_anchors(anchors_path)

    K.clear_session()
    # ------------------------------------------------------#
    #   创建yolo模型
    # ------------------------------------------------------#
    model_body = yolo_body((None, None, 3), anchors_mask, num_classes)
    if model_path != '':
        #  载入预训练权重
        print('Load weights {}.'.format(model_path))
        model_body.load_weights(model_path, by_name=True, skip_mismatch=True)
    # 对模型预测结果形状进行定义
    y_true = [Input(shape=(input_shape[0] // {0: 32, 1: 16, 2: 8}[l], input_shape[1] // {0: 32, 1: 16, 2: 8}[l], \
                           len(anchors_mask[l]), num_classes + 5)) for l in range(len(anchors_mask))]
    # 定义YOLO损失函数,
    # 为什么Lambda层后面还接了个([*model_body.output, *y_true])?
    # 这类似于函数表达式y=Dense(units=10)(x) 中的x
    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments={'input_shape': input_shape, 'anchors': anchors, 'anchors_mask': anchors_mask,
                                   'num_classes': num_classes, 'label_smoothing': label_smoothing})([*model_body.output, *y_true])
    # 构建Model，为训练做准备
    model = Model([model_body.input, *y_true], model_loss)   # inputs=[model_body.input, *y_true], outputs=model_loss.

    # -------------------------------------------------------------------------------#
    #   训练参数的设置
    #   logging表示tensorboard的保存地址
    #   checkpoint用于设置权值保存的细节，period用于修改多少epoch保存一次
    #   reduce_lr用于设置学习率下降的方式
    #   early_stopping用于设定早停，val_loss多次不下降自动结束训练，表示模型基本收敛
    # -------------------------------------------------------------------------------#
    logging = TensorBoard(log_dir='logs/')
    checkpoint = ModelCheckpoint('logs/ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=False, period=1)
    if Cosine_scheduler:
        reduce_lr = WarmUpCosineDecayScheduler(T_max=5, eta_min=1e-5, verbose=1)
    else:
        reduce_lr = ExponentDecayScheduler(decay_rate=0.94, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
    loss_history = LossHistory('logs/')

    # ---------------------------#
    #   读取数据集对应的txt
    # ---------------------------#
    with open(train_annotation_path) as f:
        train_lines = f.readlines()
    with open(val_annotation_path) as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    if Freeze_Train:
        freeze_layers = 249
        for i in range(freeze_layers): model_body.layers[i].trainable = False
        print('Freeze the first {} layers of total {} layers.'.format(freeze_layers, len(model_body.layers)))

    # ------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   UnFreeze_Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    # ------------------------------------------------------#
    if True:
        batch_size = Freeze_batch_size
        lr = Freeze_lr
        start_epoch = Init_Epoch
        end_epoch = Freeze_Epoch

        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError('数据集过小，无法进行训练，请扩充数据集。')
        # 对于损失函数yolo_loss，以及y_true和y_pred：
        # 把y_true当成一个输入，构成多输入模型，把loss写成层（Lambda层），作为最后的输出。
        # 这样，构建模型的时候，就只需要将模型的输出（output）定义为loss即可。
        # 而编译（compile）的时候，直接将loss设置为y_pred，因为模型的输出就是loss，即y_pred就是loss，
        # 因而无视y_true。训练的时候，随便添加一个符合形状的y_true数组即可。
        # 解释：模型compile时传递的是自定义的loss，而把loss写成一个层融合到model里面后，
        # y_pred就是loss。自定义损失函数规定要以y_true, y_pred为参数
        model.compile(optimizer=Adam(lr=lr), loss={'yolo_loss': lambda y_true, y_pred: y_pred})

        train_dataloader = YoloDatasets(train_lines, input_shape, anchors, batch_size, num_classes, anchors_mask,
                                        mosaic=mosaic, train=True)
        val_dataloader = YoloDatasets(val_lines, input_shape, anchors, batch_size, num_classes, anchors_mask,
                                      mosaic=False, train=False)

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(
            generator=train_dataloader,
            steps_per_epoch=epoch_step,
            validation_data=val_dataloader,
            validation_steps=epoch_step_val,
            epochs=end_epoch,
            initial_epoch=start_epoch,
            use_multiprocessing=True if num_workers > 1 else False,
            workers=num_workers,
            callbacks=[logging, checkpoint, reduce_lr, early_stopping, loss_history]
        )

    if Freeze_Train:
        for i in range(freeze_layers): model_body.layers[i].trainable = True

    if True:
        batch_size = Unfreeze_batch_size
        lr = Unfreeze_lr
        start_epoch = Freeze_Epoch
        end_epoch = UnFreeze_Epoch

        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError('数据集过小，无法进行训练，请扩充数据集。')

        model.compile(optimizer=Adam(lr=lr), loss={'yolo_loss': lambda y_true, y_pred: y_pred})
        # train_dataloader的生成过程 详见yolo_datasets.py
        train_dataloader = YoloDatasets(train_lines, input_shape, anchors, batch_size, num_classes, anchors_mask,
                                        mosaic=mosaic, train=True)
        val_dataloader = YoloDatasets(val_lines, input_shape, anchors, batch_size, num_classes, anchors_mask,
                                      mosaic=False, train=False)

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(
                            generator=train_dataloader,
                            steps_per_epoch=epoch_step,
                            validation_data=val_dataloader,
                            validation_steps=epoch_step_val,
                            epochs=end_epoch,
                            initial_epoch=start_epoch,
                            use_multiprocessing=True if num_workers > 1 else False,
                            workers=num_workers,
                            callbacks=[logging, checkpoint, reduce_lr, early_stopping, loss_history])
