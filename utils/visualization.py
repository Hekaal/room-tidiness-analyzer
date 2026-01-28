from PIL import ImageDraw, ImageFont

def draw_visuals(pil_img, detections):
    draw = ImageDraw.Draw(pil_img)
    w, h = pil_img.size

    floor_y = int(h * 0.6)
    draw.rectangle([(0, floor_y), (w, h)], outline="orange", width=3)
    draw.text((10, floor_y - 20), "Floor Zone", fill="orange")

    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
        draw.text((x1, y1 - 12), d["class"], fill="green")

    return pil_img
