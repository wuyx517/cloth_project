# PyCharm
# -*- coding: utf-8 -*-
# Created wuyx on 2022/10/18


from pydantic import BaseModel


class ClothInfo(BaseModel):
    cloth_b64: str = None
    box_shape: list = None
    affine_point: list = None
    change_len: int = 5


class MixInfo(BaseModel):
    id: str = None
    draw_b64: str = None


