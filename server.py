from sanic import Sanic
from sanic.response import text, html, json, file, stream, HTTPResponse, StreamingHTTPResponse
from sanic.request import Request
import filetype
import cv2
import numpy as np
import utils.utils
import torch
import model.detector
import os

app = Sanic("DeepLearnApp")
# db_settings = {
#     'DB_HOST': 'localhost',
#     'DB_NAME': 'appdb',
#     'DB_USER': 'appuser'
# }
# app.config.update(db_settings)

DATA_PATH = "data/coco.data"
WEIGHTS_PATH = "modelzoo/coco2017-0.241078ap-model.pth"


@app.listener("before_server_start")
async def setup_deep_learn(app, loop):
    assert os.path.exists(DATA_PATH), "请指定正确的模型路径"
    assert os.path.exists(WEIGHTS_PATH), "请指定正确的参数路径"
    # assert os.path.exists(opt.img), "请指定正确的测试图像路径"
    cfg = utils.utils.load_datafile(DATA_PATH)

    # 模型加载
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.detector.Detector(
        cfg["classes"], cfg["anchor_num"], True).to(device)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))

    # sets the module in eval node
    model.eval()


@app.get("/")
async def hello_world(request: Request) -> HTTPResponse:

    app.ctx.db = "fack db"
    return text("Hello, world.")


@app.post('/deep_learn_test')
async def deep_learn_test(request: Request) -> HTTPResponse:
    file = request.files.get("file")
    is_image = filetype.is_image(file.body)
    print(is_image)
    if is_image:
        img = cv2.imdecode(np.frombuffer(
            file.body, np.uint8), cv2.IMREAD_COLOR)
        return json({'code': 0})
    return json({'err': '扩展名错误'})
