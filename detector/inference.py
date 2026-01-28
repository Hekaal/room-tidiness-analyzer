# detector/inference.py
import numpy as np
from PIL import Image
from detector.classes import COCO_CLASSES, ALLOWED_CLASSES


IMG_SIZE = 640
CONF_THRES = 0.35
IOU_THRES = 0.5


def preprocess(image):
    img = Image.fromarray(image).resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    return img


def xywh_to_xyxy(box):
    cx, cy, w, h = box
    return [
        cx - w / 2,
        cy - h / 2,
        cx + w / 2,
        cy + h / 2,
    ]


def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return inter / (area1 + area2 - inter + 1e-6)


def nms(boxes, scores):
    idxs = np.argsort(scores)[::-1]
    keep = []

    while idxs.size > 0:
        cur = idxs[0]
        keep.append(cur)
        idxs = idxs[1:]

        idxs = np.array([
            i for i in idxs
            if iou(boxes[cur], boxes[i]) < IOU_THRES
        ])

    return keep


def detect_objects(session, image):
    orig_h, orig_w = image.shape[:2]
    inp = preprocess(image)

    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: inp})[0]

    # pastikan shape (8400, 84)
    if output.shape[1] == 84:
        preds = output[0].T
    else:
        preds = output[0]

    boxes, scores, classes = [], [], []

    for pred in preds:
        box = pred[:4]
        class_scores = pred[4:]

        class_id = np.argmax(class_scores)
        conf = class_scores[class_id]

        if conf < CONF_THRES:
            continue

        class_name = COCO_CLASSES.get(class_id)
        if class_name not in ALLOWED_CLASSES:
            continue

        boxes.append(xywh_to_xyxy(box))
        scores.append(conf)
        classes.append(class_name)

    if not boxes:
        return []

    keep = nms(boxes, scores)

    detections = []
    for i in keep:
        x1, y1, x2, y2 = boxes[i]

        # scale back to original image size
        x1 = int(x1 * orig_w / IMG_SIZE)
        y1 = int(y1 * orig_h / IMG_SIZE)
        x2 = int(x2 * orig_w / IMG_SIZE)
        y2 = int(y2 * orig_h / IMG_SIZE)

        detections.append({
            "class": classes[i],
            "confidence": float(scores[i]),
            "bbox": (x1, y1, x2, y2)
        })

    return detections
