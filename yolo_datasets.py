"""
time: 2022/03/28
author: cong
"""

import numpy as np
import math
from PIL import Image
from random import shuffle, sample
import cv2
import keras
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
#Sequence类必须重载三个私有方法__init__、__len__和__getitem__，主要是__getitem__。__init__是构造方法，用于初始化数据的，
# 只要能让样本数据集通过形参顺利传进来就行了。__len__基本上不用改写，用于计算样本数据长度。__getitem__用于生成批量数据，
# 喂给神经网络模型训练用，其输出格式是元组。元组里面有两个元素，每个元素各是一个列表，第一个元素是batch data构成的列表，
# 第二个元素是label构成的列表。在第一个列表中每个元素是一个batch，每个batch里面才是图像张量，所有的batch串成一个列表。第二个列表比较普通，
# 每个元素是个实数，表示标签。这点跟生成器不一样，生成器是在执行时通过yield关键字把每个batch数据喂给模型，一次喂一个batch。__getitem__相当于生成器的作用，
# 如同ImageDataGenerator，但注意编写方法时返回数据不要用yield，而要用return，像一个普通函数一样。至于后台怎么迭代调用这个生成器，
# 这是keras后台程序处理好的事，我们不用担心。我这里程序假定样本数据已经转成pickle的形式，是字典构成的列表，字典的键值分别是feature和label，其中feature是图像张量。
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
    # 在__getitem__中没有结束条件
    def __getitem__(self, index):
        image_data = []
        box_data = []
        for i in range(index * self.batch_size, (index + 1) * self.batch_size):
            i = i % self.length  # voc:17416
            # ---------------------------------------------------#
            #   训练时进行数据的随机增强,验证时不进行数据的随机增强
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
        print(image_data.shape)
        box_data = np.array(box_data)
        print(box_data.shape)
        y_true = self.preprocess_true_boxes(box_data, self.input_shape, self.anchors, self.num_classes)
        return [image_data, *y_true], np.zeros(self.batch_size) # 在列表前加*号，会将列表拆分成一个一个的独立元素，

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


# train_lines
train_annotation_path = '2007_train.txt'
with open(train_annotation_path) as f:
    train_lines = f.readlines()

# input_shape
input_shape = [416, 416]

# anchors
anchors_path = 'model_data/yolo_anchors.txt'
anchors, num_anchors = get_anchors(anchors_path)

# batch_size
Freeze_batch_size = 8
Unfreeze_batch_size = 4
batch_size  = Freeze_batch_size

# num_classes,
classes_path = 'model_data/voc_classes.txt'
class_names, num_classes = get_classes(classes_path)

# anchors_mask
anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

# mosaic
mosaic = False

train_dataloader = YoloDatasets(train_lines, input_shape, anchors, batch_size,
                                num_classes, anchors_mask, mosaic=mosaic, train=True)

dataset_len = len(train_dataloader)
print('dataset_len:', dataset_len)
train_lines_len = len(train_lines)
dataset_len_cal = math.ceil(train_lines_len / batch_size)
print('dataset_len_cal:', dataset_len_cal)

# a = train_dataloader[2180]
