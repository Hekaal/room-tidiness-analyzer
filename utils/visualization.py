# utils/visualization.py
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from utils.floor_visualization import floor_clutter_mask


def draw_visuals(image_np, detections, alpha=0.45):
    pil_img = Image.fromarray(image_np).convert("RGB")
    draw = ImageDraw.Draw(pil_img)
    w, h = pil_img.size

    # =========================
    # 🔴 FLOOR CLUTTER OVERLAY
    # =========================
    mask = floor_clutter_mask(image_np)  # boolean mask

    overlay = np.array(pil_img).copy()
    overlay[mask] = [255, 0, 0]  # merah

    blended = (
        overlay * alpha + np.array(pil_img) * (1 - alpha)
    ).astype(np.uint8)

    pil_img = Image.fromarray(blended)
    draw = ImageDraw.Draw(pil_img)

    # =========================
    # 🟠 FLOOR LINE
    # =========================
    floor_y = int(h * 0.6)
    draw.line([(0, floor_y), (w, floor_y)], fill=(255, 165, 0), width=3)

    # =========================
    # 🟢 BOUNDING BOX
    # =========================
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        cls = d["class"]

        draw.rectangle(
            [x1, y1, x2, y2],
            outline=(0, 255, 0),
            width=3
        )

        draw.text(
            (x1, max(0, y1 - 14)),
            cls,
            fill=(0, 255, 0)
        )

    return pil_img
