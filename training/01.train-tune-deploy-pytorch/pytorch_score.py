# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import os
import json
import base64
from io import BytesIO
from PIL import Image

from azureml.core.model import Model


def preprocess_image(image_file):
    """Preprocess the input image."""
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_file)
    image = data_transforms(image).float()
    image = torch.tensor(image)
    image = image.unsqueeze(0)
    return image


def base64ToImg(base64ImgString):
    base64Img = base64ImgString.encode('utf-8')
    decoded_img = base64.b64decode(base64Img)
    return BytesIO(decoded_img)


def init():
    global model
    model_path = Model.get_model_path('pytorch-hymenoptera')
    model = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.eval()


def run(input_data):
    img = base64ToImg(json.loads(input_data)['data'])
    img = preprocess_image(img)

    # get prediction
    output = model(img)

    classes = ['ants', 'bees']
    softmax = nn.Softmax(dim=1)
    pred_probs = softmax(model(img)).detach().numpy()[0]
    index = torch.argmax(output, 1)

    result = json.dumps({"label": classes[index], "probability": str(pred_probs[index])})
    return result
