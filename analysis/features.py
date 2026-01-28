import cv2
import numpy as np

SURFACE_CLASSES = {"bed", "couch", "chair"}
IGNORE_CLASSES = {"person"}

def extract_features(detections, image):
    h, w = image.shape[:2]
    area_img = h * w

    total_objects = 0
    total_bbox_area = 0
    objects_on_floor = 0
    objects_on_surface = 0

    floor_y = int(h * 0.6)

    surface_boxes = [
        d["bbox"] for d in detections if d["class"] in SURFACE_CLASSES
    ]

    for d in detections:
        if d["class"] in IGNORE_CLASSES:
            continue

        x1, y1, x2, y2 = d["bbox"]
        bbox_area = (x2 - x1) * (y2 - y1)
        total_bbox_area += bbox_area
        total_objects += 1

        center_y = (y1 + y2) / 2
        if center_y >= floor_y:
            objects_on_floor += 1

        for sx1, sy1, sx2, sy2 in surface_boxes:
            if x1 > sx1 and x2 < sx2 and y1 > sy1 and y2 < sy2:
                objects_on_surface += 1
                break

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / area_img

    return {
        "total_objects": total_objects,
        "total_bbox_area_ratio": round(total_bbox_area / area_img, 4),
        "objects_on_floor": objects_on_floor,
        "objects_on_surface": objects_on_surface,
        "edge_density": round(float(edge_density), 4)
    }
