
import time
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from yolo import YOLO


config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)

if __name__ == "__main__":
    yolo = YOLO()

    #-------------------------------------------------------------------------#
    #   dir_origin_path指定了用于检测的图片的文件夹路径
    #   dir_save_path指定了检测完图片的保存路径
    #   dir_origin_path和dir_save_path仅在mode='dir_predict'时有效
    #-------------------------------------------------------------------------#
    dir_origin_path = "/Users/cong/yolov4-keras-master/first"
    dir_save_path   = "first_out_cong"
    #-------------------------------------------------------------------------#
    #
    #   dir_save_path指定了检测完图片的保存路径
    #   dir_origin_path和dir_save_path仅在mode='dir_predict'时有效
    #-------------------------------------------------------------------------#
    point_color = (0, 255, 0)
    thickness = 1
    lineType = 4
    font = cv2.FONT_ITALIC
    import os


    files = os.listdir(dir_origin_path)
    count = 0
    num = len(files)
    print('files numbers:', num)
    for filename in files:
        count += 1
        print('count:', count )
        image = Image.open(os.path.join(dir_origin_path,filename))
        ss = yolo.detect_image(image)
        frame = np.array(ss)
        # RGBtoBGR满足opencv显示格式
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # r_image = yolo.detect_image(image)
        cv2.imwrite(os.path.join(dir_save_path,filename), np.array(frame))


