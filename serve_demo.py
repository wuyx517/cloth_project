# PyCharm
# -*- coding: utf-8 -*-
# Created wuyx on 2022/10/18
import json
from api_operation.image_tools import image_to_base64, b64code_to_cv_image
import requests
import cv2

host = 'http://127.0.0.1:5555'


if __name__ == '__main__':
    point = (234, 687, 564, 1138)
    box_shape = [564 - 234, 1138 - 687]
    af_point = [[225, 628], [573, 551], [275, 1122]]
    ch_len = 5

    cloth = cv2.imread('t8.png')
    draw = cv2.imread('20210426213945282.jpg')

    cloth_b64 = image_to_base64(cloth)

    res = requests.post(url=host + '/api/add_cloth', data=json.dumps({'cloth_b64': cloth_b64,
                                                                      'box_shape': box_shape,
                                                                      'affine_point': af_point,
                                                                      'change_len': 5}))
    result = json.loads(res.text)['data']
    image_id = result['id']

    print(image_id)

    box_shape = box_shape[::-1]
    box_shape.append(3)
    # draw = cv2.resize(draw, (200, 200))
    draw = cv2.resize(draw, box_shape[:2])
    # draw_ones = np.ones(box_shape) * 255
    # draw_ones[50:250, 100:300, :] = draw
    draw_ones = draw

    draw_b64 = image_to_base64(draw_ones)

    res = requests.post(url=host + '/api/draw_cloth', data=json.dumps({'id': image_id,
                                                                      'draw_b64': draw_b64}))

    result = json.loads(res.text)['data']
    mix_image_b64 = result['result']

    mix_image = b64code_to_cv_image(mix_image_b64)

    cv2.imwrite('server_test.jpg', mix_image)





