# -*- coding: utf-8 -*-
# @Time : 2022/6/6 22:49
# @Author : wuyx
# @File : image_overlay_0606.py
import cv2
import numpy as np
import copy

def img_resize(image, width_new, height_new):
    height, width = image.shape[0], image.shape[1]
    # 判断图片的长宽比率
    if width / height >= width_new / height_new:
        img_new = cv2.resize(image, (width_new, int(height * width_new / width)))
    else:
        img_new = cv2.resize(image, (int(width * height_new / height), height_new))
    tmp_zeros = np.ones((height_new, width_new, 3))
    tmp_zeros *= 255
    tmp_zeros[:img_new.shape[0], :img_new.shape[1]] = img_new
    return tmp_zeros


def image_overlay_0606(background, images, point, change_len):
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
    background_boxes = background_gray[h1: h2, w1: w2]
    # max_value = np.max(background_boxes)
    # min_value = np.min(background_boxes)
    max_value = 255
    min_value = 0
    mean_value = (max_value + min_value) / 2
    max_mean = max_value - mean_value
    variety_matrix = np.ones_like(background_boxes).astype(np.float)
    for x in range(h):
        for y in range(w):
            pixel_value = background_boxes[x, y]
            variety_value = (pixel_value - mean_value) / max_mean
            variety_matrix[x, y] = variety_value
    # variety_matrix += change_len
    # variety_matrix = cv2.blur(variety_matrix, (3, 3))

    # images 对应坐标位移
    # print(w)
    # padding_images = np.zeros((h + (2 * change_len), w + (2 * change_len), 4)) * (255, 255, 255, 0)
    # padding_images = np.zeros((h + (2 * change_len), w + (2 * change_len), 4)) * (255, 255, 255, 0)
    # padding_images = np.zeros((h + (2 * change_len), w + (2 * change_len), 3)) * (255, 255, 255)
    # images = img_resize(images, w - 2 * change_len, h - 2 * change_len).astype(np.int)
    # cv2.imwrite('test1.png', images)
    images = cv2.resize(images, dsize=(w - 2 * change_len, h - 2 * change_len), interpolation=cv2.INTER_NEAREST).astype(np.int)
    # images = cv2.resize(images, (w, h))
    # images = images[0:h, 0:w]
    # images = images[0:h, 0:w]
    # background = cv2.cvtColor(background, cv2.COLOR_BGR2BGRA)
    images = np.pad(images, ((change_len, change_len), (change_len, change_len), (0, 0)), 'constant',
                    constant_values=255)
    image_test = np.ones_like(images) * 255
    background = background.astype(np.int)
    for x in range(h):
        for y in range(w):
            variety_value = variety_matrix[x, y]
            change_value = variety_value * change_len
            change_x = x - change_value
            change_y = y - change_value
            # if change_y < 0 or change_x < 0 or change_x > h or change_y > w:
            #     continue
            # background[h + x, w + y] = images[change_x, change_y]
            # if images[change_x, change_y][-1] != 0:
            # background[h1 + x, w1 + y] = images[change_x, change_y]

            r = change_value - int(change_value)
            change_x_1 = x - int(change_value)
            change_y_1 = y - int(change_value)
            change_x_2 = change_x_1 + 1
            change_y_2 = change_y_1 + 1
            if change_x_2 >= h or change_y_2 >= w:
                continue
            pixel = images[change_x_1, change_y_1] * r + (1 - r) * images[change_x_2, change_y_2]
            # print(pixel)
            # if np.all(pixel == 255):
            #     # image_test[x, y] = background[h1 + x, w1 + y]
            #     continue
            image_test[x, y] = pixel
            background[h1 + x, w1 + y] = pixel * background[h1 + x, w1 + y] / 255
            # else:
            # background[h1 + x, w1 + y] += images[change_x, change_y]
            # if images[change_x, change_y][-1] == 0:
            #     background[h1 + x, w1 + y][-1] = 0
    # bgray = cv2.cvtColor(background_copy[h1: h2, w1: w2, :], cv2.COLOR_BGR2GRAY).astype(np.int)
    # bgray = np.array([bgray, bgray, bgray]).transpose((1, 2, 0))
    # background_copy[h1: h2, w1: w2, :] = bgray * image_test / 220
    cv2.imwrite('test2.png', image_test)
    # print(padding_images.shape)
    # cv2.imwrite('padding_outputs.png', padding_images)
    # images 插值
    # print(np.where(padding_images == 0))
    # 贴合图片
    # background = cv2.cvtColor(background, cv2.COLOR_BGR2BGRA)
    #
    # background[h1 - change_len: h2 + change_len, w1 - change_len: w2 + change_len] = background[
    #                                                                                  h1 - change_len: h2 + change_len,
    #                                                                                  w1 - change_len: w2 + change_len] \
    #                                                                                  + padding_images
    # background = cv2.rectangle(background, (w1, h1), (w2, h2), (0, 255, 0), 1)
    cv2.imwrite('output_16.png', background)
    # cv2.addWeighted(background[h1 - change_len: h2 + change_len, w1 - change_len, w2 + change_len, :], 0.7, )


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
    back_path = 't1.jpg'
    pros_path = 't7.png'
    # pros_path = 'test_2.png'
    p = (434, 600, 776, 980)
    back_image = cv2.imread(back_path)
    pros_image = cv2.imread(pros_path).astype(np.float64)
    image_overlay_0606(back_image, pros_image, p, 13)
