"""Small shared helpers. All images are float32 RGB in [0, 1] unless noted."""
from pathlib import Path
import cv2
import numpy as np


def to_float(img_u8):
    return img_u8.astype(np.float32) / 255.0


def to_u8(img):
    return (np.clip(img, 0, 1) * 255.0 + 0.5).astype(np.uint8)


def luminance(img):
    return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]


def box(x, r):
    return cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REFLECT)


def guided_filter(guide, src, r=8, eps=1e-3):
    """Single-channel guided filter (He et al.). guide, src: HxW float32."""
    guide, src = guide.astype(np.float32), src.astype(np.float32)
    mg, ms = box(guide, r), box(src, r)
    cov = box(guide * src, r) - mg * ms
    var = box(guide * guide, r) - mg * mg
    a = cov / (var + eps)
    b = ms - a * mg
    return box(a, r) * guide + box(b, r)


def guided_filter_rgb(img, r=8, eps=1e-3):
    """Filter each channel guided by the image luminance (edge-preserving)."""
    g = luminance(img)
    return np.stack([guided_filter(g, img[..., c], r, eps) for c in range(3)], -1)


def resize_max_side(img, max_side):
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s >= 1:
        return img
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def sanitize_image(img, max_side=0):
    """Make any user input a float32 HxWx3 RGB array in [0, 1].
    Handles grayscale, RGBA, 16-bit, float images and rejects empty / wrong-shaped input."""
    if img is None:
        raise ValueError("No image provided.")
    a = np.asarray(img)
    if a.ndim == 2:
        a = np.stack([a] * 3, -1)
    elif a.ndim == 3 and a.shape[2] == 1:
        a = np.repeat(a, 3, -1)
    elif a.ndim == 3 and a.shape[2] >= 3:
        a = a[..., :3]
    else:
        raise ValueError(f"Unsupported image shape {a.shape}.")
    if min(a.shape[:2]) < 8:
        raise ValueError("Image is too small (minimum 8 x 8 pixels).")
    if a.dtype == np.uint8:
        a = a.astype(np.float32) / 255.0
    elif a.dtype == np.uint16:
        a = a.astype(np.float32) / 65535.0
    else:
        a = a.astype(np.float32)
        if a.max() > 1.5:
            a = a / 255.0
    a = np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=0.0)
    a = np.clip(a, 0.0, 1.0)
    if max_side:
        a = resize_max_side(a, int(max_side))
    return np.ascontiguousarray(a)


EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(d):
    return sorted(p for p in Path(d).rglob("*") if p.suffix.lower() in EXT)


def read_rgb(p):
    return cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
