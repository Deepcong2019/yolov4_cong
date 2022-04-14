"""
time: 2022/04/14
author: cong
theme: 对图像进行长和宽的扭曲达到缩放的目的并且多余部分加上灰度条。
"""
from PIL import Image
import numpy as np


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


w = 416
h = 416
jitter = 0.3
new_ar = w / h * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
print('new_ar:', new_ar)
scale = rand(.25, 2)
print('scale:', scale)
image = Image.open('img.png')
# 随机缩放
if new_ar < 1:
    nh = int(scale * h)
    nw = int(nh * new_ar)
    print('nw:', nw, 'nh:', nh)
else:
    nw = int(scale * w)
    nh = int(nw / new_ar)
    print('nw:', nw, 'nh:', nh)
image = image.resize((nw, nh), Image.BICUBIC)
image.show()
image.save('random_scale.jpg')
# ------------------------------------------#
#   将图像多余的部分加上灰条
# ------------------------------------------#
dx = int(rand(0, w - nw))
dy = int(rand(0, h - nh))
print('dx:', dx, 'dy:', dy)
new_image = Image.new('RGB', (w, h), (128, 128, 128))
new_image.paste(image, (dx, dy)) # 把image粘到new_image上，起始位置相对于new_image的位置（dx,dy)
new_image.show()
new_image.save('random_scale_gray.jpg')