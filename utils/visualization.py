import cv2

ZONE_COLOR = (255, 200, 0)

def draw_zones(image):
    h, w = image.shape[:2]
    floor_y = int(h * 0.6)
    cv2.rectangle(image, (0, floor_y), (w, h), ZONE_COLOR, 2)
    cv2.putText(image, "Floor Zone", (10, floor_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, ZONE_COLOR, 2)
    return image

def draw_detections(image, detections):
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        label = d["class"]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    return image
