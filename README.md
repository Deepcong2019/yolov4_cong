# yolov4_cong
yolov4  
1、data_aug.py:原图像直接resize获得一个样本，，然后镜像获得一张样本。  
2、get_random_data.py: 不使用mosaic时，使用的数据增强流程：随机缩放paste到416*416的灰度图上；随机确定是否镜像；色域扰动。  
3、get_random_data_mosaic.py: 使用mosaic数据增强。  
4、image_process.py: yolov4读入图像的两种处理方式。  
5、image_random_scale.py: 随机缩放paste到416*416的灰度图上.  
6、infer.py / infer_detail.py: 图像预处理---输入网络----网络输出流程的直观结果。  
7、model.py: 纯净的网络结构代码。
8、model_loss.py: 单纯推理一张图片和真实标签的loss计算直观过程。  
9、multiput.py: 预测时输入多张图片同时预测。  
10、preprocess_images_boxes.py: 制作真是标签y_true的直观流程。  
11、xmltoimg.py: 标注文件转化为带有标签的图像。  
12、yolo_datasets.py: 直观生成batch_data训练数据的过程。  


