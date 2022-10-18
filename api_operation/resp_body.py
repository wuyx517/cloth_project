# OK = 200
# CREATED = 201
# NO_CONTENT = 204
# INVALID_REQUEST = 400
# NOT_FOUND = 404
# NOT_ACCEPTABLE = 406
from fastapi import FastAPI, status


class AppResult:
    def __init__(self):
        self.code = status.HTTP_201_CREATED
        self.err_code = None
        self.msg = None
        self.data = None

    def dump(self):
        return dict(code=self.code, err_code=self.err_code, msg=self.msg, data=self.data)
