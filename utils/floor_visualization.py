import numpy as np
from skimage.feature import canny
from skimage.color import rgb2gray


def floor_clutter_mask(image_np, threshold=0.07):
    h, w, _ = image_np.shape
    floor_y = int(h * 0.6)

    gray = rgb2gray(image_np)
    edges = canny(gray, sigma=2)

    mask = np.zeros((h, w), dtype=bool)
    mask[floor_y:, :] = True

    clutter = edges & mask
    density = clutter.sum() / mask.sum()

    return clutter if density > threshold else np.zeros_like(mask)
