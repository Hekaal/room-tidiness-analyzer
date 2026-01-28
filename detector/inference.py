import numpy as np

def preprocess(img, size=640):
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

def detect_objects(session, image_np):
    inp = preprocess(image_np)
    outputs = session.run(None, {"images": inp})[0]

    detections = []
    for det in outputs[0]:
        conf = det[4]
        if conf < 0.4:
            continue

        x, y, w, h = det[:4]
        detections.append({
            "class": "object",
            "confidence": float(conf),
            "bbox": (
                int(x - w/2),
                int(y - h/2),
                int(x + w/2),
                int(y + h/2)
            )
        })

    return detections
