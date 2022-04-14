"""
time: 2022/03/25
author: cong
object_detection：yolo_v4推理过程中的详细细节
"""
import numpy as np
from model import yolo_body
from PIL import Image
import tensorflow as tf
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


class_path = 'model_data/predefined_classes.txt'
model_path = 'model_data/ep101-loss5.217-val_loss9.845.h5'
img_path = '1_10.jpg'
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

print('names:', class_names)
image = Image.open(img_path)
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
print('model:', model)
# keras  model输出之后计算结果

# ---------------------------------------------------#
#   将预测值的每个特征层调成真实值
# ---------------------------------------------------#
def get_anchors_and_decode(feats, anchors, num_classes, input_shape, calc_loss=False):
    num_anchors = 3
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
    # print('feats[..., 4:5]:', feats[..., 4:5])
    box_class_probs = sigmoid(feats[..., 5:])
    # print('feats[..., 5:]:', feats[..., 5:])
    # ---------------------------------------------------------------------#
    #   在计算loss的时候返回grid, feats, box_xy, box_wh
    #   在预测的时候返回box_xy, box_wh, box_confidence, box_class_probs
    # ---------------------------------------------------------------------#
    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


image_shape = [image.size[1], image.size[0]]

box_xy = []
box_wh = []
box_confidence  = []
box_class_probs = []
for i in range(len(model_output)):
    # sub_box_xy:ndarray:(1,52,52,3,2)
    anchors=anchors_all[anchors_mask[i]]
    feats = model_output[i]
    num_anchors = 3
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
    sub_box_xy = xy / grid_shape[::-1]  #对应位置的元素除以13
    sub_box_wh = wh / input_shape[::-1]  #对应位置的元素除以416
    # ------------------------------------------#
    #   获得预测框的置信度
    # ------------------------------------------#
    sub_box_confidence = sigmoid(feats[..., 4:5])
    # print('feats[..., 4:5]:', feats[..., 4:5])
    sub_box_class_probs = sigmoid(feats[..., 5:])
    # print('feats[..., 5:]:', feats[..., 5:])
    # ---------------------------------------------------------------------#
    #   在计算loss的时候返回grid, feats, box_xy, box_wh
    #   在预测的时候返回box_xy, box_wh, box_confidence, box_class_probs
    # ---------------------------------------------------------------------#



    box_xy.append(np.reshape(sub_box_xy, [-1, 2]))
    box_wh.append(np.reshape(sub_box_wh, [-1, 2]))
    box_confidence.append(np.reshape(sub_box_confidence, [-1, 1]))
    box_class_probs.append(np.reshape(sub_box_class_probs, [-1, num_classes]))


box_xy = np.concatenate(box_xy, axis = 0)  # box_xy:(10647, 2)
box_wh = np.concatenate(box_wh, axis = 0)  # box_wh:(10647, 2)
box_confidence = np.concatenate(box_confidence, axis = 0)  # boxes_confidence:(10647,1)
box_class_probs = np.concatenate(box_class_probs, axis = 0)  # box_class_probs:(10647,2)

#------------------------------------------------------------------------------------------------------------#
#   在图像传入网络预测前会进行letterbox_image给图像周围添加灰条，因此生成的box_xy, box_wh是相对于有灰条的图像的
#   我们需要对其进行修改，去除灰条的部分。 将box_xy、和box_wh调节成y_min,y_max,xmin,xmax
#   如果没有使用letterbox_image也需要将归一化后的box_xy, box_wh调整成相对于原图大小的
#------------------------------------------------------------------------------------------------------------#
box_yx = box_xy[..., ::-1]   #倒序排列
box_hw = box_wh[..., ::-1]


if letterbox_image:
    # -----------------------------------------------------------------#
    #   这里求出来的offset是图像有效区域相对于图像左上角的偏移情况
    #   new_shape指的是宽高缩放情况
    # -----------------------------------------------------------------#
    new_shape = np.round(image_shape * np.min(input_shape / image_shape))
    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape

    box_yx = (box_yx - offset) * scale
    box_hw *= scale

box_mins = box_yx - (box_hw / 2.)
box_maxes = box_yx + (box_hw / 2.)
boxes = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=1)
boxes *= np.concatenate([image_shape, image_shape])

# a=[]
# for i in range(len(box_confidence)):
#     if box_confidence[i][0]>0:
#         if i==94:
#             print(box_confidence[i][0])
#         # box_confidence[i][0]=1
#         a.append(i)
# print('a_len:', len(a))
# for i in range(len(box_class_probs)):
#
#     if i in a:
#         print(box_class_probs[i][0])
#         box_class_probs[i][0] = 1

box_scores  = box_confidence * box_class_probs
# for i in range(len(box_scores)):
#     if box_scores[i][0]>=0.1:
#         print(i)


#-----------------------------------------------------------#
#   判断得分是否大于score_threshold
#-----------------------------------------------------------#
mask = box_scores >= confidence     # （10647， 2）
# a=0
# for i in range(10647):
#     if mask[i][0]:
#         a+=1
#     print(a)




max_boxes_tensor = K.constant(max_boxes, dtype='int32')
boxes_out   = []
scores_out  = []
classes_out = []
for c in range(num_classes):
    #-----------------------------------------------------------#
    #   取出所有box_scores >= score_threshold的框，和成绩
    #-----------------------------------------------------------#
    class_boxes = tf.boolean_mask(boxes, mask[:, c])
    if c==0:
        sess = K.get_session()
        result111 = sess.run(class_boxes)

    class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])

    #-----------------------------------------------------------#
    #   非极大抑制
    #   保留一定区域内得分最大的框
    #-----------------------------------------------------------#
    nms_index = tf.image.non_max_suppression(tf.cast(class_boxes, tf.float32), class_box_scores, max_boxes_tensor, iou_threshold=nms_iou)

    #-----------------------------------------------------------#
    #   获取非极大抑制后的结果
    #   下列三个分别是：框的位置，得分与种类
    # gather(reference, indices),在给定的张量中搜索给定下标的向量。
    # 参数：reference表示被搜寻的向量；indices表示整数张量，要查询的元素的下标。
    # 返回值：一个与参数reference数据类型一致的张量。
    #-----------------------------------------------------------#
    class_boxes = K.gather(class_boxes, nms_index)
    class_box_scores = K.gather(class_box_scores, nms_index)
    classes = K.ones_like(class_box_scores, 'int32') * c

    boxes_out.append(class_boxes)
    scores_out.append(class_box_scores)
    classes_out.append(classes)
boxes_out = K.concatenate(boxes_out, axis=0)
scores_out = K.concatenate(scores_out, axis=0)
classes_out = K.concatenate(classes_out, axis=0)
sess = K.get_session()
result = sess.run([boxes_out, scores_out, classes_out])
