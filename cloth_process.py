# PyCharm
# -*- coding: utf-8 -*-
# Created wuyx on 2022/10/13
import numpy as np
import cv2
from blend_modes import *
import time
from mmdet.apis import inference_detector, init_detector

model = init_detector('configs/fashionformer/fashionpedia/fashionformer_r50_mlvl_feat_3x.py',
                      'ckpt/fashionformer_r50_3x.pth', device='cuda:0')


def init_cloth_status(cloth_image, draw_box_shape, affine_point, change_len=5):
    """

    :param cloth_image:
    :param draw_box_shape: 维度 0: 宽 1: 高
    :param affine_point:
    :param change_len:
    :return:
    """
    affine_point = np.array(affine_point)
    # 计算仿射变换后 最大的框。并且往外括 change_len 个像素
    x_point = affine_point[:, 0]
    y_point = affine_point[:, 1]
    x1 = min(x_point) - change_len
    x2 = max(x_point) + change_len
    y1 = min(y_point) - change_len
    y2 = max(y_point) + change_len

    # 计算 仿射变化后 的点相对于 切割出来 box 的位置
    x_box_point = x_point - x1
    y_box_point = y_point - y1

    # 计算像素变换矩阵
    cloth_image_gray = cv2.cvtColor(cloth_image, cv2.COLOR_BGR2GRAY)
    cloth_image_gray_boxes = cloth_image_gray[y1:y2, x1:x2]
    variety_matrix = (cloth_image_gray_boxes - 127.5) / 255

    # 计算仿射变换矩阵
    draw_point = np.float32([[0, 0], [draw_box_shape[0], 0], [0, draw_box_shape[1]]])
    affine_box_point = np.float32([[x_box_point[0], y_box_point[0]], [x_box_point[1], y_box_point[1]],
                                   [x_box_point[2], y_box_point[2]]])
    affine_box_point_m = cv2.getAffineTransform(draw_point, affine_box_point)

    affine_box = (x1, y1, x2, y2)

    # 提取t的mask
    result = inference_detector(model, cloth_image)
    t_shirt_mask_bool = result[1][1][0]
    t_shirt_mask = np.zeros_like(t_shirt_mask_bool)
    t_shirt_mask[t_shirt_mask_bool] = 1
    t_shirt_mask = t_shirt_mask[:, :, np.newaxis]
    return variety_matrix, affine_box_point_m, affine_box, t_shirt_mask


def image_mix(cloth_image, draw_image, variety_matrix, affine_m, affine_box, change_len, t_shirt_mask=None):
    affine_box_shape = [affine_box[2] - affine_box[0], affine_box[3] - affine_box[1]]
    # draw_image 仿射变换
    draw_trans_image = cv2.warpAffine(draw_image, affine_m, affine_box_shape, borderValue=(255, 255, 255))
    # cv2.imwrite('t2.jpg', draw_trans_image)
    # 添加透明维度
    image_shape = list(draw_trans_image.shape)
    image_shape[-1] = 1
    image_transparent = np.ones(image_shape) * 255
    draw_trans_image = np.concatenate((draw_trans_image, image_transparent), axis=-1)

    image_shape = list(cloth_image.shape)
    image_shape[-1] = 1
    image_transparent = np.ones(image_shape) * 255
    cloth_image = np.concatenate((cloth_image, image_transparent), axis=-1)

    mask_box = t_shirt_mask[affine_box[1]: affine_box[3], affine_box[0]:affine_box[2]]
    variety_image = np.ones_like(draw_trans_image) * 255
    for y in range(affine_box_shape[1]):
        for x in range(affine_box_shape[0]):
            variety_value = variety_matrix[y, x]
            change_value = variety_value * change_len
            r = change_value - int(change_value)
            change_y_1 = y - int(change_value)
            change_x_1 = x - int(change_value)
            change_y_2 = change_y_1 + 1
            change_x_2 = change_x_1 + 1
            if change_y_2 >= affine_box_shape[1] or change_x_2 >= affine_box_shape[0] or mask_box[y, x] == 0:
                continue

            pixel = draw_trans_image[change_y_1, change_x_1] * r + (1 - r) * draw_trans_image[change_y_2, change_x_2]
            variety_image[y, x] = pixel
    # cv2.imwrite('t3.jpg', variety_image)
    cloth_image[affine_box[1]: affine_box[3], affine_box[0]: affine_box[2], :] = multiply(
        cloth_image[affine_box[1]: affine_box[3], affine_box[0]: affine_box[2], :], variety_image, 0.8)


    return cloth_image


if __name__ == '__main__':
    point = (234, 687, 564, 1138)
    box_shape = [564 - 234, 1138 - 687]
    af_point = [[225, 628], [573, 551], [275, 1122]]
    ch_len = 5

    cloth = cv2.imread('t8.png')
    draw = cv2.imread('20210426213945282.jpg')

    var_matrix, aff_box_point_m, aff_box, t_shirt_mask = init_cloth_status(cloth, box_shape, af_point, ch_len)
    box_shape = box_shape[::-1]
    box_shape.append(3)
    # draw = cv2.resize(draw, (200, 200))
    draw = cv2.resize(draw, box_shape[:2])
    # draw_ones = np.ones(box_shape) * 255
    # draw_ones[50:250, 100:300, :] = draw
    draw_ones = draw
    # cv2.imwrite('t1.jpg', draw_ones)
    start_time = time.time()
    cloth_image = image_mix(cloth, draw_ones, var_matrix, aff_box_point_m, aff_box, ch_len, t_shirt_mask)
    print('image mis time is : ', time.time() - start_time)
    cv2.imwrite('cloth_process_test.jpg', cloth_image)
