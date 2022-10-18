import base64
import json
import numpy as np
import cv2
import urllib


def video_encode(b64code, out_file):
    try:
        with open(out_file, "wb") as file:
            txt = base64.b64decode(b64code)
            file.write(txt)
        return True
    except Exception as err:
        print(err)
        return False


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def b64code_to_cv_image(_bin):
    image = None
    if _bin is not None:
        if not isinstance(_bin, list):
            _bin = base64.b64decode(_bin)
            _bin = np.fromstring(_bin, np.uint8)
            image = cv2.imdecode(_bin, cv2.IMREAD_COLOR)
            # faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = []
            for __bin in _bin:
                __bin = base64.b64decode(__bin)
                __bin = np.fromstring(__bin, np.uint8)
                _image = cv2.imdecode(__bin, cv2.IMREAD_COLOR)
                image.append(image)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def image_to_base64(image_np):
    image = cv2.imencode('.jpg', image_np)[1]
    image_code = str(base64.b64encode(image))[2:-1]

    return image_code
