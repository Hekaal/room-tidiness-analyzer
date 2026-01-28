ALLOWED_CLASSES = {
    "backpack", "handbag", "bottle", "cup",
    "book", "laptop",
    "chair", "bed", "couch",
    "person"
}

def detect_objects(model, image):
    results = model(image, verbose=False)
    detections = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])

            if cls_name not in ALLOWED_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            detections.append({
                "class": cls_name,
                "confidence": conf,
                "bbox": (x1, y1, x2, y2)
            })

    return detections
