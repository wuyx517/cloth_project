# -*- coding: utf-8 -*-
# @Time : 2022/8/3 22:20
# @Author : wuyx
# @File : test_convolve.py

import numpy as np
from scipy import signal

a = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

b = np.array([[1 / 9, 1 / 9, 1 / 9],
             [1 / 9 , 1 / 9, 1 / 9],
             [1 / 9, 1 / 9, 1 / 9]])

grad = signal.convolve2d(a, b, mode='same')

print(grad)
