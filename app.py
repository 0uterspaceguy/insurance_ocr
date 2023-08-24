import logging
import uvicorn
from fastapi import FastAPI, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from triton_model import TritonInference

from configurations import config

import numpy as np
import cv2

from utils import preprocess_yolo, postprocess_yolo, ocr
from yolo_config import primary_config


triton_ip = config.triton_ip
triton_port = config.triton_port
fastapi_port = config.fastapi_port

logger = logging.getLogger("OCR_API")
logger.setLevel('INFO')
app = FastAPI()

primary = TritonInference(
    f"{triton_ip}:{triton_port}", "primary"#, primary_config
)


@app.post("/recognize")
async def recognize(
    request: Request,
):
    try:
        data = await request.body()
        decoded_img = cv2.imdecode(np.fromstring(data, dtype=np.uint8), 1)

    except Exception as ex:
        logger.error(f"Error with {ex}")
        return {}
    
    tensor = preprocess_yolo(decoded_img)
    yolo_output = primary.inference(tensor)
    number_box, sum_box, sign_box = postprocess_yolo(
                                    yolo_output,
                                    conf_threshold=0.2,
                                    iou_threshold=0.5,
                                    orig_shape=decoded_img.shape[:2],
                                    )
    
    num_string, sum_string = ocr(decoded_img, number_box, sum_box)

    return {
            'Number':num_string,
            'Sum':sum_string,
            'Sign_exists': bool(len(sign_box)),
    }


@app.get('/health', status_code=200)
async def health(request: Request):
    return {'client_host': request.client.host, 'response_status': 200}


def uvicorn_start():
    logging.basicConfig(level=30, format='%(asctime)s %(levelname)-8s %(module)s.%(funcName)s -> %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    uvicorn.run("app:app", host='0.0.0.0', port=config.fastapi_port,
                log_level=config.log_level, timeout_keep_alive=20)


if __name__ == "__main__":
    try:
        uvicorn_start()
    except KeyboardInterrupt:
        logger.info("See you again")
        exit()


