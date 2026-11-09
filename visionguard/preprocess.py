"""Image preprocessing shared by training (tf.data) and serving (Flask / API).

Training and serving MUST call the same function, otherwise the model sees
differently prepared pixels in production than it saw during training
(training/serving skew), and its accuracy silently drops.

Steps:
1. Decode with Pillow (handles JPEG, PNG, WEBP, GIF, BMP; TensorFlow's own
   decoder cannot read WEBP, which exists in this dataset).
2. Use the first frame of animated images and apply the EXIF orientation.
3. Convert to 3-channel RGB (drops alpha / palette / greyscale differences).
4. Resize to size x size with bilinear interpolation. The aspect ratio is NOT
   preserved (images are squashed): this keeps the whole picture (borders,
   watermarks) visible to the model, and because serving does exactly the same
   thing the distortion is consistent.
5. Return uint8 pixels in [0, 255]. Model-specific scaling (e.g. MobileNetV2's
   [-1, 1]) lives INSIDE the saved model, so callers never need to know it.
"""
from __future__ import annotations

import io
from typing import BinaryIO, Union

import numpy as np
from PIL import Image, ImageOps

# Protect against decompression bombs but allow the large photos in the dataset.
Image.MAX_IMAGE_PIXELS = 120_000_000

Source = Union[str, bytes, BinaryIO]


def _open(source: Source) -> Image.Image:
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    img = Image.open(source)
    if getattr(img, "is_animated", False):
        img.seek(0)
    img.load()
    return img


def load_image(source: Source, size: int) -> np.ndarray:
    """Decode an image (path, bytes or file object) into a uint8 (size, size, 3) array."""
    img = _open(source)
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:  # broken EXIF should not make an image unusable
        pass
    img = img.convert("RGB")
    img = img.resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def is_decodable(data: bytes) -> bool:
    """True if Pillow can fully decode the bytes as an image."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.verify()
        with Image.open(io.BytesIO(data)) as im:
            im.load()
        return True
    except Exception:
        return False
