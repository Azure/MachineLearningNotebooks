import cv2
import numpy as np
import torch
import time
import torchvision
from PIL import Image
from typing import Any, Dict, List


def letterbox(
    img,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=True,
    scaleFill=False,
    scaleup=True,
):
    """Resize image to a 32-pixel-multiple rectangle
    https://github.com/ultralytics/yolov3/issues/232

    :param img: an image
    :type img: <class 'numpy.ndarray'>
    :param new_shape: target shape in [height, width]
    :type new_shape: <class 'int'>
    :param color: color for pad area
    :type color: <class 'tuple'>
    :param auto: minimum rectangle
    :type auto: bool
    :param scaleFill: stretch the image without pad
    :type scaleFill: bool
    :param scaleup: scale up
    :type scaleup: bool
    :return: letterbox image, scale ratio, padded area in (width, height) in each side
    :rtype: <class 'numpy.ndarray'>, <class 'tuple'>, <class 'tuple'>
    """
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border
    return img, ratio, (dw, dh)


def clip_coords(boxes, img_shape):
    """Clip bounding xyxy bounding boxes to image shape (height, width)

    :param boxes: bbox
    :type boxes: <class 'torch.Tensor'>
    :return: img_shape: image shape
    :rtype: img_shape: <class 'tuple'>: (height, width)
    """
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


def unpad_bbox(boxes, img_shape, pad):
    """Correct bbox coordinates by removing the padded area from letterbox image

    :param boxes: bbox absolute coordinates from prediction
    :type boxes: <class 'torch.Tensor'>
    :param img_shape: image shape
    :type img_shape: <class 'tuple'>: (height, width)
    :param pad: pad used in letterbox image for inference
    :type pad: <class 'tuple'>: (width, height)
    :return: (unpadded) image height and width
    :rtype: <class 'tuple'>: (height, width)
    """
    dw, dh = pad
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    img_width = img_shape[1] - (left + right)
    img_height = img_shape[0] - (top + bottom)

    if boxes is not None:
        boxes[:, 0] -= left
        boxes[:, 1] -= top
        boxes[:, 2] -= left
        boxes[:, 3] -= top
        clip_coords(boxes, (img_height, img_width))

    return img_height, img_width


def _convert_to_rcnn_output(output, height, width, pad):
    # output: nx6 (x1, y1, x2, y2, conf, cls)
    rcnn_label: Dict[str, List[Any]] = {"boxes": [], "labels": [], "scores": []}

    # Adjust bbox to effective image bounds
    img_height, img_width = unpad_bbox(
        output[:, :4] if output is not None else None, (height, width), pad
    )

    if output is not None:
        rcnn_label["boxes"] = output[:, :4]
        rcnn_label["labels"] = output[:, 5:6].long()
        rcnn_label["scores"] = output[:, 4:5]

    return rcnn_label, (img_height, img_width)


def xywh2xyxy(x):
    """Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right

    :param x: bbox coordinates in [x center, y center, w, h]
    :type x: <class 'numpy.ndarray'> or torch.Tensor
    :return: new bbox coordinates in [x1, y1, x2, y2]
    :rtype: <class 'numpy.ndarray'> or torch.Tensor
    """
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def box_iou(box1, box2):
    """Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py

    :param box1: bbox in (Tensor[N, 4]), N for multiple bboxes and 4 for the box coordinates
    :type box1: <class 'torch.Tensor'>
    :param box2: bbox in (Tensor[M, 4]), M is for multiple bboxes
    :type box2: <class 'torch.Tensor'>
    :return: iou of box1 to box2 in (Tensor[N, M]), the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    :rtype: <class 'torch.Tensor'>
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.t())
    area2 = box_area(box2.t())

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (
        (
            torch.min(box1[:, None, 2:], box2[:, 2:])
            - torch.max(box1[:, None, :2], box2[:, :2])
        )
        .clamp(0)
        .prod(2)
    )
    return inter / (
        area1[:, None] + area2 - inter
    )  # iou = inter / (area1 + area2 - inter)


def non_max_suppression(
    prediction,
    conf_thres=0.1,
    iou_thres=0.6,
    multi_label=False,
    merge=False,
    classes=None,
    agnostic=False,
):
    """Performs per-class Non-Maximum Suppression (NMS) on inference results

    :param prediction: predictions
    :type prediction: <class 'torch.Tensor'>
    :param conf_thres: confidence threshold
    :type conf_thres: float
    :param iou_thres: IoU threshold
    :type iou_thres: float
    :param multi_label: enable to have multiple labels in each box?
    :type multi_label: bool
    :param merge: Merge NMS (boxes merged using weighted mean)
    :type merge: bool
    :param classes: specific target class
    :type classes:
    :param agnostic: enable class agnostic NMS?
    :type agnostic: bool
    :return: detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    :rtype: <class 'list'>
    """
    if prediction.dtype is torch.float16:
        prediction = prediction.float()  # to FP32

    nc = prediction[0].shape[1] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # min_wh = 2
    max_wh = 4096  # (pixels) maximum box width and height
    max_det = 300  # maximum number of detections per image
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    if multi_label and nc < 2:
        multi_label = False  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero().t()
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Sort by confidence
        # x = x[x[:, 4].argsort(descending=True)]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3e3):
            try:  # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(
                    1, keepdim=True
                )  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy
            except Exception:  # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
                print(
                    "[WARNING: possible CUDA error ({} {} {} {})]".format(
                        x, i, x.shape, i.shape
                    )
                )
                pass

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output


def _read_image(ignore_data_errors: bool, image_url: str, use_cv2: bool = False):
    try:
        if use_cv2:
            # cv2 can return None in some error cases
            img = cv2.imread(image_url)  # BGR
            if img is None:
                print("cv2.imread returned None")
            return img
        else:
            image = Image.open(image_url).convert("RGB")
            return image
    except Exception as ex:
        if ignore_data_errors:
            msg = "Exception occurred when trying to read the image. This file will be ignored."
            print(msg)
        else:
            print(str(ex), has_pii=True)
        return None


def preprocess(image_url, img_size=640):
    img0 = _read_image(
        ignore_data_errors=False, image_url=image_url, use_cv2=True
    )  # cv2.imread(image_url)  # BGR
    if img0 is None:
        return image_url, None, None

    img, ratio, pad = letterbox(img0, new_shape=img_size, auto=False, scaleup=False)

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x640x640
    img = np.ascontiguousarray(img)
    np_image = torch.from_numpy(img)
    np_image = np.expand_dims(np_image, axis=0)
    np_image = np_image.astype(np.float32) / 255.0
    return np_image, pad
