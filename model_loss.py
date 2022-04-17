"""
time : 2022/03/31
author: cong
"""

import math
import numpy as np
from model import yolo_body
from PIL import Image
import tensorflow as tf
from keras import backend as K
import cv2

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

# 生成随机数
def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


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

    v = 4 * K.square(tf.math.atan2(b1_wh[..., 0], K.maximum(b1_wh[..., 1],
                                        K.epsilon())) - tf.math.atan2(b2_wh[..., 0],
                                                                    K.maximum(b2_wh[..., 1],K.epsilon())))  /(math.pi * math.pi)


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
    #input_shape = K.cast(input_shape, K.dtype(y_true[0]))
    # input_shape = K.constant(input_shape)
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
        # TensorArray可以看做是具有动态size功能的Tensor数组。通常都是跟while_loop或map_fn结合使用
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
            #  tf.boolean_mask的作用是通过布尔值过滤元素
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
            #选出最大的iou
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
        # box_loss_scale:(1,52,52,3,1)
        # -----------------------------------------------------------#
        #   计算Ciou loss
        # -----------------------------------------------------------#
        raw_true_box = y_true[l][..., 0:4]
        ciou = box_ciou(pred_box, raw_true_box) # (1,52,52,3,1)
        ciou_loss = object_mask * box_loss_scale * (1 - ciou) # (1,52,52,3,1)

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











class_path = 'model_data/coco_classes.txt'
model_path = 'model_data/yolo4_weight.h5'
anchors_path = 'model_data/yolo_anchors.txt'
max_boxes = 100   # 最大框的数量
confidence = 0.5  #   只有得分大于置信度的预测框会被保留下来
nms_iou = 0.5     #   非极大抑制所用到的nms_iou大小
#   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
#   在多次测试后，发现关闭letterbox_image直接resize的效果更好
# ---------------------------------------------------------------------#
letterbox_image  = False
input_shape = [416, 416]
anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
class_names, num_classes = get_classes(class_path)
anchors_all, num_anchors = get_anchors(anchors_path)

annotation_line = '/Users/cong/yolov4-keras-master/VOCdevkit/VOC2007/JPEGImages/007826.jpg 80,217,320,273,10 197,193,257,326,8 258,180,312,314,8 10,195,93,358,8 82,252,243,372,8\n'
line = annotation_line.split()
image = Image.open(line[0])
print('names:', class_names)
image = cvtColor(image)    # print('image_shape:', np.shape(image))# image_shape: (1080, 1920, 3)
image_data = resize_image(image, input_shape, letterbox_image=False)
#   添加上batch_size维度，并进行归一化,
#   np.expand_dims(a, axis=0)表示在0位置添加数据,
image_data = np.expand_dims(preprocess_input(np.array(image_data, dtype='float32')), 0)
#print(image_data.shape) (1, 416, 416, 3)
# 加载模型
model = yolo_body((None, None, 3), anchors_mask, num_classes)
# model.summary()
model.load_weights(model_path)
model_output = model.predict(image_data)
# print('model:', model)
# 以上为模型推理的结果


###########################################################
# 以下找出该图片的y_true
iw, ih = image.size
h, w = input_shape
#   获得预测框, int向下取整, box.split(','): ['80', '217', '320', '273', '10']
box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])


# anchors_mask
anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

box_data = np.zeros((max_boxes, 5))
if len(box) > 0:
    # np.random.shuffle(box)
    if len(box) > max_boxes:
        box = box[:max_boxes]
    box_data[:len(box)] = box
box_datas_array = np.expand_dims(box_data, axis=0)


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
anchors = np.expand_dims(anchors_all, 0)
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

    iou = intersect_area / (box_area + anchor_area - intersect_area) # (n,9)
    # -----------------------------------------------------------#
    #   维度是[n,] 感谢 消尽不死鸟 的提醒
    # -----------------------------------------------------------#
    best_anchor = np.argmax(iou, axis=-1) #(6,5,5,5,6)# 索引为5的anchor在26*26的特征图上同时预测3个物体

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
                y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]#   J是第几行，所以算的是ymin的长度  i是第几列，要计算x方向的长度
                y_true[l][b, j, i, k, 4] = 1
                y_true[l][b, j, i, k, 5 + c] = 1





#   将预测结果和实际ground truth分开，args是[*model_body.output, *y_true]
args = [*model_output, *y_true]

ignore_thresh=.5,
label_smoothing=0.1,
print_loss=False
num_layers = len(anchors_mask)
y_true = [K.constant(i) for i in args[num_layers:]]
yolo_outputs = args[:num_layers]

# -----------------------------------------------------------#
#   得到input_shpae为416,416
# -----------------------------------------------------------#
# input_shape = K.constant(input_shape)

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
def loop_body(b, ignore_mask):
    #   取出n个真实框：(n,4)
    #  tf.boolean_mask的作用是通过布尔值过滤元素
    true_box = tf.boolean_mask(y_true[l][b, ..., 0:4], object_mask_bool[b, ..., 0])
    print(true_box.shape)
    # -----------------------------------------------------------#
    #   计算预测框与真实框的iou
    #   pred_box    13,13,3,4 预测框的坐标
    #   true_box    n,4 真实框的坐标
    #   iou         13,13,3,n 预测框和真实框的iou
    # -----------------------------------------------------------#
    iou = box_iou(pred_box[b], true_box)  # (13, 13, 3, ?)
    print(iou.shape)
    # -----------------------------------------------------------#
    #   best_iou    (13,13,3) 每个特征点的3个预测框与真实框的最大重合程度
    # -----------------------------------------------------------#
    best_iou = K.max(iou, axis=-1)
    print(best_iou.shape)

    ignore_mask = ignore_mask.write(b, K.cast(best_iou < ignore_thresh, K.dtype(true_box)))
    print('ignore_mask_shape:', K.cast(best_iou < ignore_thresh, K.dtype(true_box)).shape)

    return b + 1, ignore_mask


for l in range(num_layers-2):

    print('l:', l)

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
    #标签平滑的意思比如真实值编码为[1, 0, 0], 现在将真实值变为[0.95, 0.025, 0.025]
    #其实就是对训练过程进行惩罚。
    # -----------------------------------------------------------#
    #   将yolo_outputs的特征层输出进行处理、获得四个返回值
    #   其中：
    #   grid        (13,13,1,2) 网格坐标
    #   raw_pred    (m,13,13,3,85) 尚未处理的预测结果
    #   pred_xy     (m,13,13,3,2) 解码后的中心坐标
    #   pred_wh     (m,13,13,3,2) 解码后的宽高坐标
    # -----------------------------------------------------------#
    grid, raw_pred, pred_xy, pred_wh = get_anchors_and_decode(yolo_outputs[l],
                                                              anchors_all[anchors_mask[l]], num_classes, input_shape,
                                                              calc_loss=True)

    # -----------------------------------------------------------#
    #   pred_box是解码后的预测的box的位置
    #   (m,13,13,3,4)
    # -----------------------------------------------------------#
    pred_box = K.concatenate([K.constant(pred_xy), K.constant(pred_wh)])

    # y_true = K.constant(y_true)
    # -----------------------------------------------------------#
    #   找到负样本群组，第一步是创建一个数组，[]
    # dynamic_size指定数组长度可变
    # -----------------------------------------------------------#
    ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
    object_mask_bool = K.cast(object_mask, 'bool') # (1,52,52,3,1)

    # -----------------------------------------------------------#
    #   对每一张图片计算ignore_mask
    # -----------------------------------------------------------#

    # -----------------------------------------------------------#
    #   在这个地方进行一个循环、循环是对每一张图片进行的
    # -----------------------------------------------------------#
    _, ignore_mask = tf.while_loop(lambda b, *args: b < m, loop_body, [0, ignore_mask])
    #
    # -----------------------------------------------------------#
    #   ignore_mask用于提取出作为负样本的特征点
    #   (m,13,13,3)
    # -----------------------------------------------------------#
    ignore_mask = ignore_mask.stack()  # 将TensorArray中元素叠起来当做一个Tensor输出
    #   (m,13,13,3,1)
    ignore_mask = K.expand_dims(ignore_mask, -1)
    #   (?,13,13,3,1)
    # -----------------------------------------------------------#
    #   真实框越大，比重越小，小框的比重更大。
    # -----------------------------------------------------------#
    box_loss_scale = 2 - y_true[l][..., 2:3] * y_true[l][..., 3:4]  # (1,52,52,3,1)

    # -----------------------------------------------------------#
    #   计算Ciou loss
    # -----------------------------------------------------------#
    raw_true_box = y_true[l][..., 0:4]
    ciou = box_ciou(pred_box, raw_true_box)   #
    ciou_loss = object_mask * box_loss_scale * (1 - ciou)

    # ------------------------------------------------------------------------------#
    #   如果该位置本来有框，那么计算1与置信度的交叉熵
    #   如果该位置本来没有框，那么计算0与置信度的交叉熵
    #   binary_cross_entropy是二分类的交叉熵，实际是多分类softmax_cross_entropy的一种特殊情况，
    #   当多分类中，类别只有两类时，即0或者1，即为二分类，二分类也是一个逻辑回归问题，也可以套用逻辑回归的损失函数。

    # ------------------------------------------------------------------------------#
    confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[..., 4:5], from_logits=True) + \
                      (1 - object_mask) * K.binary_crossentropy(object_mask, raw_pred[..., 4:5],
                                                                from_logits=True) * ignore_mask

    class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[..., 5:], from_logits=True)

    location_loss = K.sum(ciou_loss)#沿着指定轴计算张量之和
    confidence_loss = K.sum(confidence_loss)
    class_loss = K.sum(class_loss)
    # -----------------------------------------------------------#
    #   计算正样本数量，没有正样本的时候K.sum(K.cast(object_mask, tf.float32))=0，，num_pos取值为1
    # -----------------------------------------------------------#
    num_pos += tf.maximum(K.sum(K.cast(object_mask, tf.float32)), 1) # 返回最大值

    loss += location_loss + confidence_loss + class_loss
    # if print_loss:
    #   loss = tf.Print(loss, [loss, location_loss, confidence_loss, class_loss, K.sum(ignore_mask)], message='loss: ')

    loss = loss / num_pos


sess = K.get_session()
loss = sess.run(loss)
location_loss = sess.run(location_loss)
object_mask_num = sess.run(K.sum(K.cast(object_mask, tf.float32)))
ignore_mask = sess.run(ignore_mask)
object_mask = sess.run(object_mask)
confidence_loss = sess.run(confidence_loss)
num_pos = sess.run(num_pos)
y_true = sess.run(y_true)
print('loss:', loss)
print('location_loss:', location_loss)
# print('ignore_mask',ignore_mask)

print('object_mask_num:', object_mask_num)
print('num_pos:', num_pos)