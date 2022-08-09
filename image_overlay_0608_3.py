# -*- coding: utf-8 -*-
# @Time : 2022/6/6 22:49
# @Author : wuyx
# @File : image_overlay_0606.py
import os.path

import cv2
import numpy as np
import copy
from scipy import signal
from scipy import misc
from scipy import signal
from blend_modes import *


def img_resize(image, width_new, height_new):
    height, width = image.shape[0], image.shape[1]
    # 判断图片的长宽比率
    if width / height >= width_new / height_new:
        img_new = cv2.resize(image, (width_new, int(height * width_new / width)))
    else:
        img_new = cv2.resize(image, (int(width * height_new / height), height_new))
    tmp_zeros = np.ones((height_new, width_new, 3))
    tmp_zeros *= 255
    transparent_dims = np.zeros((height_new, width_new, 1))
    tmp_zeros = np.concatenate((tmp_zeros, transparent_dims), axis=-1)
    tmp_zeros[:img_new.shape[0], :img_new.shape[1]] = img_new
    return tmp_zeros


def image_overlay_0606(background, images, point, change_len, save_path):
    # 添加透明维度
    image_shape = list(images.shape)
    image_shape[-1] = 1
    image_transparent = np.ones(image_shape) * 255
    images = np.concatenate((images, image_transparent), axis=-1)

    # 提取褶皱的信息
    background_copy = copy.deepcopy(background)
    background_copy = cv2.cvtColor(background_copy, cv2.COLOR_BGR2GRAY).astype(np.int)
    w1, h1, w2, h2 = point
    w1 = w1 - change_len
    w2 = w2 + change_len
    h1 = h1 - change_len
    h2 = h2 + change_len
    w = w2 - w1
    h = h2 - h1
    background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    background_boxes = background_gray[h1 - 100: h2 + 100, w1: w2 + 200]
    # max_value = np.max(background_boxes)
    # min_value = np.min(background_boxes)
    max_value = 255
    min_value = 0
    mean_value = (max_value + min_value) / 2
    max_mean = max_value - mean_value
    variety_matrix = np.ones_like(background_boxes).astype(np.float)

    # affine transform
    pts1 = np.float32([[w1, h1], [w2, h1], [w1, h2]])
    pts2 = np.float32([[264, 827], [491, 815], [277, 1035]])
    M = cv2.getAffineTransform(pts1, pts2)
    images = img_resize(images, w - 2 * change_len, h - 2 * change_len)
    transformed_images = cv2.warpAffine(images, M, (w + 200, h + 200))
    images = transformed_images
    cv2.imwrite('tmp/transformed_test.png', transformed_images)

    for x in range(h + 200):
        for y in range(w + 200):
            pixel_value = background_boxes[x, y]
            variety_value = (pixel_value - mean_value) / max_mean
            variety_matrix[x, y] = variety_value

    b = np.array([[1 / 9, 1 / 9, 1 / 9],
                  [1 / 9, 1 / 9, 1 / 9],
                  [1 / 9, 1 / 9, 1 / 9]])
    variety_matrix = signal.convolve2d(variety_matrix, b, mode='same')

    # images = img_resize(images, w - 2 * change_len, h - 2 * change_len).astype(np.int)
    # images = np.pad(images, ((change_len, change_len), (change_len, change_len), (0, 0)), 'constant',
    #                 constant_values=0)
    # cv2.imwrite('tmp/test1.png', images)
    background = background.astype(np.float)
    image_test = np.ones_like(images) * 255
    for x in range(h + 200):
        for y in range(w + 200):
            variety_value = variety_matrix[x, y]
            change_value = variety_value * change_len
            change_x = x - change_value
            change_y = y - change_value
            r = change_value - int(change_value)
            change_x_1 = x - int(change_value)
            change_y_1 = y - int(change_value)
            change_x_2 = change_x_1 + 1
            change_y_2 = change_y_1 + 1
            if change_x_2 >= h + 200 or change_y_2 >= w + 200:
                continue
            pixel = images[change_x_1, change_y_1] * r + (1 - r) * images[change_x_2, change_y_2]
            image_test[x, y] = pixel
            # background[h1 + x, w1 + y] = pixel * background[h1 + x, w1 + y] / 255
    background_shape = list(background.shape)
    background_shape[-1] = 1
    background_transparent = np.ones(background_shape) * 255


    background = np.concatenate((background, background_transparent), axis=-1)
    # image_test = np.concatenate((image_test, image_transparent), axis=-1)
    image_test = image_test.astype(np.float)
    background[h1 - 100: h2 + 100, w1: w2 + 200, :] = multiply(background[h1 - 100: h2 + 100, w1: w2 + 200, :], image_test, 0.8)

    # background[h1: h2, w1: w2, :] = blended_img_float
    cv2.imwrite('tmp/test2.png', image_test)
    cv2.imwrite(save_path, background)


def expansion_img(src_img, change_len):
    Y, X, N = src_img.shape
    ex_img = np.zeros((Y + 2 * change_len, X + 2 * change_len, N))
    ex_img[change_len:-change_len, change_len:-change_len, :] = src_img  # 中心
    for i in range(change_len):  # 左右，不含四角
        ex_img[change_len:-change_len, i, :] = src_img[:, 0, :]  # 左
        ex_img[change_len:-change_len, -i - 1, :] = src_img[:, -1, :]  # 右
    for i in range(change_len):  # 上下，含四角
        ex_img[i, :, :] = ex_img[change_len, :, :]  # 上
        ex_img[-i - 1, :, :] = ex_img[-(change_len + 1), :, :]  # 下

    return ex_img


if __name__ == '__main__':
    back_path = 't8.png'
    pros_path = '20210426213945282.jpg'
    # pros_path = 'Blooming-01.png'
    # back_path = '16pic_96170_b.jpg'
    # pros_path = '123.png'
    # pros_path = 'test_2.png'
    # p = (434, 600, 776, 980)
    p = (270, 802, 533, 1030)
    back_image = cv2.imread(back_path)
    pros_image = cv2.imread(pros_path).astype(np.float64)
    # back_image = cv2.resize(back_image, (pros_image.shape[:-1]))
    # p = (14, 14, 2000, 2000)
    # for i in [4, 5, 6, 7, 8, 9, 10]:
    i = 5
    save_image = f'output_blend/output_{i}.png'
    save_dir = os.path.dirname(save_image)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    image_overlay_0606(back_image, pros_image, p, i, save_image)
