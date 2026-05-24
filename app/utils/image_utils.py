"""
Image preprocessing utilities shared by the image service and AI service.

These mirror the preprocessing pipeline your YOLO model expects:
  resize → normalize → convert colour space

When you integrate the real YOLOv9 weights, update _YOLO_INPUT_SIZE
and verify the normalisation values match your training config.
"""
from __future__ import annotations

import io
from typing import Tuple

import numpy as np
from PIL import Image, ImageOps

# YOLOv9 default input size (change to match your training config)
_YOLO_INPUT_SIZE: Tuple[int, int] = (640, 640)

# ImageNet normalisation (standard for transfer-learned backbones)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_image(image_bytes: bytes) -> Image.Image:
    """Open image bytes as a PIL RGB image."""
    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img)   # correct orientation
    return img.convert("RGB")


def resize_with_padding(img: Image.Image, target: Tuple[int, int] = _YOLO_INPUT_SIZE) -> Image.Image:
    """
    Letterbox-resize: fit image inside target while preserving aspect ratio.
    Fills empty space with grey (114, 114, 114) – YOLOv9 convention.
    """
    tw, th = target
    iw, ih = img.size
    scale = min(tw / iw, th / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    img = img.resize((nw, nh), Image.LANCZOS)

    padded = Image.new("RGB", target, (114, 114, 114))
    pad_x = (tw - nw) // 2
    pad_y = (th - nh) // 2
    padded.paste(img, (pad_x, pad_y))
    return padded


def to_numpy_tensor(img: Image.Image, normalise: bool = True) -> np.ndarray:
    """
    Convert PIL image → float32 numpy array shaped (1, 3, H, W).
    Optionally applies ImageNet normalisation.
    This is the format expected by most ONNX / PyTorch YOLO exports.
    """
    arr = np.array(img, dtype=np.float32) / 255.0    # H x W x C, [0,1]
    if normalise:
        arr = (arr - _MEAN) / _STD
    arr = arr.transpose(2, 0, 1)                      # C x H x W
    return np.expand_dims(arr, 0)                     # 1 x C x H x W


def preprocess_for_yolo(image_bytes: bytes) -> Tuple[np.ndarray, Image.Image]:
    """
    Full preprocessing pipeline → (tensor, original_image).
    Returns the original image so bounding boxes can be mapped back.
    """
    original = load_image(image_bytes)
    resized = resize_with_padding(original)
    tensor = to_numpy_tensor(resized)
    return tensor, original


def scale_boxes_to_original(
    boxes: list,         # list of (x1, y1, x2, y2) in _YOLO_INPUT_SIZE coords
    orig_w: int,
    orig_h: int,
) -> list:
    """
    Scale letterboxed bounding boxes back to original image coordinates.
    Returns normalised (0–1) coordinates relative to orig_w × orig_h.
    """
    tw, th = _YOLO_INPUT_SIZE
    scale = min(tw / orig_w, th / orig_h)
    pad_x = (tw - int(orig_w * scale)) // 2
    pad_y = (th - int(orig_h * scale)) // 2

    scaled = []
    for x1, y1, x2, y2 in boxes:
        sx1 = max(0.0, (x1 - pad_x) / (scale * orig_w))
        sy1 = max(0.0, (y1 - pad_y) / (scale * orig_h))
        sx2 = min(1.0, (x2 - pad_x) / (scale * orig_w))
        sy2 = min(1.0, (y2 - pad_y) / (scale * orig_h))
        scaled.append((sx1, sy1, sx2, sy2))
    return scaled


def get_image_dimensions(image_bytes: bytes) -> Tuple[int, int]:
    """Return (width, height) of an image without full decode."""
    img = Image.open(io.BytesIO(image_bytes))
    return img.size
