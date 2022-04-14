#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 11:57:26 2020

@author: cong
"""
from cv2 import FONT_HERSHEY_SIMPLEX

"""
目的：将原图片(img)与其xml(xml)，合成为打标记的图片(labelled)，矩形框标记用红色即可
已有：（1）原图片文件夹(imgs_path)，（2）xml文件夹(xmls_path)
思路：
    step1: 读取（原图片文件夹中的）一张图片
    step2: 读取（xmls_path）该图片的xml文件，并获取其矩形框的两个对角顶点的位置
    step3: 依据矩形框顶点坐标，在该图片中画出该矩形框
    step4: 图片另存为'原文件名'+'_labelled'，存在‘lablled’文件夹中
"""
import os
import cv2 as cv
import xml.etree.ElementTree as ET
import re


if __name__ == '__main__':
    imgs_path = '/Users/cong/yolov4-keras-master/VOCdevkit/VOC2007/JPEGImages'
    xmls_path = '/Users/cong/yolov4-keras-master/VOCdevkit/VOC2007/Annotations'
    labelled_path = '/Users/cong/yolov4-keras-master/xmls_to_imgs/'
    imgs_list = os.listdir(imgs_path)
    imgs_lists = []
    for i in imgs_list:
        if i.endswith('jpg'):
          imgs_lists.append(i)
    imgs_lists.sort(key=lambda x:int(''.join(list(filter(str.isdigit,x)))))

    xmls_list = os.listdir(xmls_path)
    xmls_lists = []
    for i in xmls_list:
        if i.endswith('xml'):
          xmls_lists.append(i)

    xmls_lists.sort(key=lambda x:int(''.join(list(filter(str.isdigit,x)))))

    nums = len(imgs_lists)
    nums = 2
    for i in range(nums):
        img_path = os.path.join(imgs_path, imgs_lists[i])
        xml_path = os.path.join(xmls_path, xmls_lists[i])
        img = cv.imread(img_path)
        labelled = img
        root = ET.parse(xml_path).getroot()
        objects = root.findall('object')
        for obj in objects:
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text.strip())
            ymin = int(bbox.find('ymin').text.strip())
            xmax = int(bbox.find('xmax').text.strip())
            ymax = int(bbox.find('ymax').text.strip())
            bname = obj.find('name').text.strip()
            # print(bname)
            cv.rectangle(labelled, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
            cv.putText(labelled, bname, (xmin, ymin), FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv.imwrite('%s%s_labelled.jpg' % (labelled_path, imgs_lists[i]), labelled)
    print('Done...')

