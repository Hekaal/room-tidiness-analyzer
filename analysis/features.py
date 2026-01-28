import numpy as np
from skimage.feature import canny
from skimage.color import rgb2gray

SURFACE_CLASSES = {"bed", "couch", "chair"}
IGNORE_CLASSES = {"person"}


def extract_features(detections, image_np):
    h, w, _ = image_np.shape
    img_area = h * w

    # =========================
    # 1️⃣ OBJECT-BASED FEATURES
    # =========================
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
        area = (x2 - x1) * (y2 - y1)

        total_objects += 1
        total_bbox_area += area

        center_y = (y1 + y2) / 2
        if center_y >= floor_y:
            objects_on_floor += 1

        for sx1, sy1, sx2, sy2 in surface_boxes:
            if x1 > sx1 and x2 < sx2 and y1 > sy1 and y2 < sy2:
                objects_on_surface += 1
                break

    # =========================
    # 2️⃣ FLOOR TEXTURE ANALYSIS
    # =========================
    gray = rgb2gray(image_np)
    edges = canny(gray, sigma=2)

    floor_mask = np.zeros((h, w), dtype=np.uint8 demonstrate)
    floor_mask[floor_y:, :] = 1

    floor_edges = edges * floor_mask
    floor_edge_density = floor_edges.sum() / floor_mask.sum()

    # =========================
    # 3️⃣ FLOOR CLUTTER AREA
    # =========================
    clutter_pixels = np.logical_and(
        edges,
        floor_mask
    ).sum()

    floor_clutter_ratio = clutter_pixels / floor_mask.sum()

    # =========================
    # 4️⃣ INFERRED FLOOR CLUTTER
    # =========================
    inferred_floor_clutter = False
    if floor_edge_density > 0.07 and objects_on_floor < 2:
        inferred_floor_clutter = True

    return {
        "total_objects": total_objects,
        "total_bbox_area_ratio": round(total_bbox_area / img_area, 4),
        "objects_on_floor": objects_on_floor,
        "objects_on_surface": objects_on_surface,

        # NEW – lantai
        "floor_edge_density": round(float(floor_edge_density), 4),
        "floor_clutter_ratio": round(float(floor_clutter_ratio), 4),
        "inferred_floor_clutter": inferred_floor_clutter
    }
