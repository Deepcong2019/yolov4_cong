"""
time: 2022/03/29
author: cong
在 get_random_data.py的基础上，剖析preprocess_true_boxes,,
相当于在没有mosaic的时候，生成所有的image_data,boxes。

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


# 图像归一化
def preprocess_input(image):
    image /= 255.0
    return image


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

# 生成随机数
def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


# ================================================================================ #
# train_lines
train_annotation_path = '2007_train.txt'
with open(train_annotation_path) as f:
    train_lines = f.readlines()
# xml坐标为xmin, ymin,xmax,ymax
# train_lines[0]: '/Users/cong/yolov4-keras-master/VOCdevkit/VOC2007/JPEGImages/007826.jpg 80,217,320,273,10 197,193,257,326,8 258,180,312,314,8 10,195,93,358,8 82,252,243,372,8\n'
train_lines = train_lines[:1]
input_shape = [416, 416]
max_boxes = 100
random = True
# anchors_mask
anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
classes_path = 'model_data/voc_classes.txt'
anchors_path = 'model_data/yolo_anchors.txt'
class_names, num_classes = get_classes(classes_path)
anchors, num_anchors = get_anchors(anchors_path)
image_datas = []
box_datas = []
for i in range(len(train_lines)):
    annotation_line = train_lines[i]
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
    #    not random =====================================================================
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
        # new_image.show()
        new_image.paste(image, (dx, dy))
        # new_image.show()
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

        image_datas.append(np.array(image_data)/255.0)
        box_datas.append(box_data)
    # random ====================================================================================
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
        image_datas.append((np.array(image_data))/255.0)
        box_datas.append(box_data)

image_datas_array = np.array(image_datas)
box_datas_array = np.array(box_datas)

#==========================================================================================
# y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)

assert (box_datas_array[..., 4] < num_classes).all(), 'class id must be less than num_classes'
# -----------------------------------------------------------#
#   获得框的坐标和图片的大小
# -----------------------------------------------------------#
true_boxes = np.array(box_datas_array, dtype='float32')
input_shape = np.array(input_shape, dtype='int32')

# -----------------------------------------------------------#
#   一共有三个特征层数
# -----------------------------------------------------------#
num_layers = len(anchors_mask)
# -----------------------------------------------------------#
#   m为图片数量，grid_shapes为网格的shape
# -----------------------------------------------------------#
m = true_boxes.shape[0]
grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8}[l] for l in range(num_layers)]
# -----------------------------------------------------------#
#   y_true的格式为(m,13,13,3,85)(m,26,26,3,85)(m,52,52,3,85)
# -----------------------------------------------------------#
y_true = [np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], len(anchors_mask[l]), 5 + num_classes),
                   dtype='float32') for l in range(num_layers)]

# -----------------------------------------------------------#
#   通过计算获得真实框的中心和宽高
#   中心点(m,n,2) 宽高(m,n,2)
# -----------------------------------------------------------#
boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2  #  (xmax - xmin)/2 + xmin = (xmin + xmax)/2
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
valid_mask = boxes_wh[..., 0] > 0  # box_wh:(100,100,2)
# valid_mask:(100)
for b in range(m):
    # -----------------------------------------------------------#
    #   对每一张图进行处理
    # -----------------------------------------------------------#
    wh = boxes_wh[b, valid_mask[b]] # array:(5,2)
    if len(wh) == 0: continue
    # -----------------------------------------------------------#
    #   [n,2] -> [n,1,2]
    # -----------------------------------------------------------#
    wh = np.expand_dims(wh, -2)  #array:(5,1,2)
    box_maxes = wh / 2.
    box_mins = - box_maxes

    # -----------------------------------------------------------#
    #   计算所有真实框和先验框的交并比
    #   intersect_area  [n,9]
    #   box_area        [n,1]
    #   anchor_area     [1,9]
    #   iou             [n,9]
    # -----------------------------------------------------------#
    # 一张图像上 所有目标框与anchor的交并比
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
        #   找到每个真实框所属的特征层，
        #   n:best_anchor的最大索引，
        #   b：第几张图片
        #   t: 这张图片上的第几个框
        #   l：哪个特征层上的anchor
        for l in range(num_layers):
            if n in anchors_mask[l]:
                #   floor用于向下取整，找到真实框所属的特征层对应的x、y轴顶点坐标#
                i = np.floor(true_boxes[b, t, 0] * grid_shapes[l][1]).astype('int32')
                j = np.floor(true_boxes[b, t, 1] * grid_shapes[l][0]).astype('int32')
                #   k指的的当前这个特征层的第k个先验框
                k = anchors_mask[l].index(n)
                #   c指的是当前这个真实框的种类
                c = true_boxes[b, t, 4].astype('int32')#
                #   y_true的shape为(m,13,13,3,85)(m,26,26,3,85)(m,52,52,3,85)
                #   最后的85可以拆分成4+1+80，4代表的是框的中心与宽高、
                #   1代表的是置信度、80代表的是种类
                # -----------------------------------------------------------#
                y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
                y_true[l][b, j, i, k, 4] = 1
                y_true[l][b, j, i, k, 5 + c] = 1

# return [image_datas_array, *y_true], np.zeros(self.batch_size)
# tuple:2   生成器返回两个元素，，，在列表前加*号，会将列表拆分成一个一个的独立元素，




































