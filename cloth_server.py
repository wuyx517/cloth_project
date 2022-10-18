# PyCharm
# -*- coding: utf-8 -*-
# Created wuyx on 2022/10/18

from fastapi import FastAPI, status
from api_operation.api_mode import *
from api_operation.image_tools import b64code_to_cv_image, image_to_base64
from api_operation.resp_body import AppResult
from fastapi.responses import JSONResponse
from cloth_process import init_cloth_status, image_mix
import uuid
import pickle as pkl

app = FastAPI()

cloth_dict = {}


@app.post("/api/add_cloth", status_code=status.HTTP_201_CREATED)
def add_cloth(request_data: ClothInfo):
    app_result = AppResult()
    try:
        # 捕获解析图像异常
        image = b64code_to_cv_image(request_data.cloth_b64)  # 解析base64图像
    except Exception as err:
        print(err)
        app_result.code = status.HTTP_400_BAD_REQUEST
        app_result.msg = '图像解析错误, 请检测图像编码格式'
        return JSONResponse(app_result.dump(), status_code=app_result.code)

    variety_matrix, affine_box_point_m, affine_box, t_shirt_mask = \
        init_cloth_status(image, request_data.box_shape, request_data.affine_point, request_data.change_len)
    uuid_str = str(uuid.uuid4())
    cloth_dict[uuid_str] = {
        'image': image,
        'variety_matrix': variety_matrix,
        'affine_box_point_m': affine_box_point_m,
        'affine_box': affine_box,
        't_shirt_mask': t_shirt_mask,
        'ch_len': request_data.change_len
    }
    app_result.err_code = '0'
    app_result.msg = 'success'
    app_result.data = {'id': uuid_str}
    return JSONResponse(app_result.dump(), status_code=app_result.code)


@app.post("/api/draw_cloth", status_code=status.HTTP_201_CREATED)
def draw_cloth(request_data: MixInfo):
    app_result = AppResult()
    try:
        # 捕获解析图像异常
        image = b64code_to_cv_image(request_data.draw_b64)  # 解析base64图像
    except Exception as err:
        print(err)
        app_result.code = status.HTTP_400_BAD_REQUEST
        app_result.msg = '图像解析错误, 请检测图像编码格式'
        return JSONResponse(app_result.dump(), status_code=app_result.code)

    cloth_info = cloth_dict[request_data.id]
    cloth = cloth_info['image']
    cloth_image = image_mix(cloth, image, cloth_info['variety_matrix'], cloth_info['affine_box_point_m'],
                            cloth_info['affine_box'], cloth_info['ch_len'], cloth_info['t_shirt_mask'])
    image_b64 = image_to_base64(cloth_image)
    app_result.err_code = 0
    app_result.msg = 'success'
    app_result.data = {'result': image_b64}
    return JSONResponse(app_result.dump(), status_code=app_result.code)


@app.get("api/inquire", status_code=status.HTTP_201_CREATED)
def inquire():
    app_result = AppResult()
    inquire_result = {}
    for key, value in cloth_dict.items():
        image = value['image']
        image_b64 = image_to_base64(image)
        inquire_result[key] = image_b64
    app_result.err_code = 0
    app_result.msg = 'success'
    app_result.data = {'result': inquire_result}
    return JSONResponse(app_result.dump(), status_code=app_result.code)


@app.get("api/save", status_code=status.HTTP_201_CREATED)
def save():
    app_result = AppResult()
    with open('cloth_dict.pkl', 'wb') as f:
        pkl.dump(cloth_dict, f)
    app_result.err_code = 0
    app_result.msg = 'success'
    return JSONResponse(app_result.dump(), status_code=app_result.code)
