import numpy as np
from PIL import Image
import sys
from functools import partial
import os
import io

from azureml.contrib.services.aml_request import AMLRequest, rawhttp
from azureml.contrib.services.aml_response import AMLResponse
from utils import preprocess, postprocess, triton_init, triton_infer


def init():
    triton_init()

@rawhttp
def run(request):
    if request.method == 'POST':
        reqBody = request.get_data(False)
        img = Image.open(io.BytesIO(reqBody))
        result = triton_infer(model_name="densenet_onnx", img=img)

        return AMLResponse(result, 200)
    else:
        return AMLResponse("bad request", 500)