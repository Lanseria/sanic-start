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
import time

app = Sanic("DeepLearnApp")
# db_settings = {
#     'DB_HOST': 'localhost',
#     'DB_NAME': 'appdb',
#     'DB_USER': 'appuser'
# }
# app.config.update(db_settings)

DATA_PATH = "data/coco.data"
WEIGHTS_PATH = "modelzoo/coco2017-0.241078ap-model.pth"


# 模型加载
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@app.listener("before_server_start")
async def setup_deep_learn(app, loop):
    assert os.path.exists(DATA_PATH), "请指定正确的模型路径"
    assert os.path.exists(WEIGHTS_PATH), "请指定正确的参数路径"
    # assert os.path.exists(opt.img), "请指定正确的测试图像路径"
    cfg = utils.utils.load_datafile(DATA_PATH)
    app.ctx.cfg = cfg
    current_model = model.detector.Detector(
        cfg["classes"], cfg["anchor_num"], True).to(device)
    current_model.load_state_dict(
        torch.load(WEIGHTS_PATH, map_location=device))

    # sets the module in eval node
    current_model.eval()
    app.ctx.current_model = current_model


async def process_img(ori_img):
    # 获取全局cfg
    cfg = app.ctx.cfg
    # 获取全局model
    current_model = app.ctx.current_model

    res_img = cv2.resize(
        ori_img, (cfg["width"], cfg["height"]), interpolation=cv2.INTER_LINEAR)
    img = res_img.reshape(1, cfg["height"], cfg["width"], 3)
    img = torch.from_numpy(img.transpose(0, 3, 1, 2))
    img = img.to(device).float() / 255.0

    # 模型推理
    start = time.perf_counter()
    preds = current_model(img)
    end = time.perf_counter()
    ptime = (end - start) * 1000.
    print("forward time:%fms" % ptime)

    # 特征图后处理
    output = utils.utils.handel_preds(preds, cfg, device)
    output_boxes = utils.utils.non_max_suppression(
        output, conf_thres=0.3, iou_thres=0.4)

    # 加载label names
    LABEL_NAMES = []
    with open(cfg["names"], 'r') as f:
        for line in f.readlines():
            LABEL_NAMES.append(line.strip())

    h, w, _ = ori_img.shape
    scale_h, scale_w = h / cfg["height"], w / cfg["width"]
    real_box = []
    # 绘制预测框
    for box in output_boxes[0]:
        box = box.tolist()

        obj_score = box[4]
        category = LABEL_NAMES[int(box[5])]

        x1, y1 = int(box[0] * scale_w), int(box[1] * scale_h)
        x2, y2 = int(box[2] * scale_w), int(box[3] * scale_h)
        real_box.append({
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "obj_score": obj_score,
            "category": category
        })
    return {
        "time": ptime,
        "output_boxes": real_box,

    }


@app.get("/")
async def hello_world(request: Request) -> HTTPResponse:

    app.ctx.db = "fack db"
    return text("Hello, world.")


@app.post('/deep_learn_test')
async def deep_learn_test(request: Request) -> HTTPResponse:
    file = request.files.get("file")
    is_image = filetype.is_image(file.body)
    if is_image:
        img = cv2.imdecode(np.frombuffer(
            file.body, np.uint8), cv2.IMREAD_COLOR)
        cv2.imwrite("2.png", img)
        data = await process_img(img)
        return json({'code': 0, 'data': data})
    return json({'err': '扩展名错误'})
