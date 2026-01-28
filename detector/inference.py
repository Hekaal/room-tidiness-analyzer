import numpy as np
from detector.classes import COCO_CLASSES, ALLOWED_CLASSES

def preprocess(img, size=640):
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))   # HWC → CHW
    img = np.expand_dims(img, axis=0)
    return img

def detect_objects(session, image_np):
    inputs = preprocess(image_np)

    # 🔴 AMBIL NAMA INPUT ONNX SECARA DINAMIS
    input_name = session.get_inputs()[0].name

    outputs = session.run(None, {input_name: inputs})[0]

    detections = []

    for det in outputs[0]:
        conf = det[4]
        if conf < 0.4:
            continue

        class_id = int(det[5])
        class_name = COCO_CLASSES.get(class_id)

        if class_name is None or class_name not in ALLOWED_CLASSES:
            continue

        x, y, w, h = det[:4]

        detections.append({
            "class": class_name,
            "confidence": float(conf),
            "bbox": (
                int(x - w / 2),
                int(y - h / 2),
                int(x + w / 2),
                int(y + h / 2),
            )
        })

    return detections
