"""
classifier/preprocess.py

Parses a serialized TFRecord byte-string into a (1, 96, 96, 3) float32
numpy array in [0, 1], matching the schema used in training.

TFRecord schema (matches optimus_stratus_imageclassifier.py safe_parse):
    "image"    : bytes_list  — raw float16 pixel data  (H * W * C * 2 bytes)
    "label"    : bytes_list  — raw uint8 segmentation mask (H * W bytes)
    "height"   : int64
    "width"    : int64
    "channels" : int64

Binary classification rule (matches Colab training):
    label = 1.0 (cloudy)      if mean(mask) > 0.2
    label = 0.0 (non-cloudy)  otherwise
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple

IMG_SIZE: int = 96   # Must match training resolution


# ─── Main entry points ────────────────────────────────────────────────────────

def parse_tfrecord_numpy(raw_bytes: bytes) -> Tuple[Optional[np.ndarray], np.float32]:
    """
    Parse one serialised TFRecord into a preprocessed image + label.

    Args:
        raw_bytes: raw bytes from a single TFRecord file (tf.train.Example format).

    Returns:
        img   : np.ndarray shape (96, 96, 3) float32 in [0, 1], or None on failure.
        label : float32  1.0 = cloudy,  0.0 = non-cloudy,  -1.0 = parse error.
    """
    try:
        # Import tensorflow only when needed so the rest of the env can run
        # with a pure numpy stack if TF is not present.
        import tensorflow as tf
        from PIL import Image as PILImage

        example = tf.train.Example()
        example.ParseFromString(raw_bytes)
        feats = example.features.feature

        h = int(feats["height"].int64_list.value[0])
        w = int(feats["width"].int64_list.value[0])
        c = int(feats["channels"].int64_list.value[0])

        img_bytes  = feats["image"].bytes_list.value[0]
        mask_bytes = feats["label"].bytes_list.value[0]

        # Validate byte counts before reshape
        expected_img  = h * w * c * 2   # float16 = 2 bytes per element
        expected_mask = h * w           # uint8   = 1 byte per element
        if len(img_bytes) != expected_img or len(mask_bytes) != expected_mask:
            return None, np.float32(-1.0)

        # Decode raw bytes → numpy
        img  = np.frombuffer(img_bytes,  dtype=np.float16).reshape(h, w, c).astype(np.float32)
        mask = np.frombuffer(mask_bytes, dtype=np.uint8  ).reshape(h, w  ).astype(np.float32)

        # Normalise image to [0, 1]
        img_max = img.max()
        if img_max > 0:
            img = img / img_max
        img = np.clip(img, 0.0, 1.0)

        # Resize to 96×96 using PIL (avoids any TF graph dependency)
        img_96 = np.stack([
            np.array(
                PILImage.fromarray(img[:, :, ch]).resize(
                    (IMG_SIZE, IMG_SIZE), PILImage.Resampling.BILINEAR
                )
            )
            for ch in range(c)
        ], axis=-1)
        img_96 = np.clip(img_96, 0.0, 1.0).astype(np.float32)

        # Derive binary label from segmentation mask  (>20% cloudy pixels ⇒ cloudy)
        label = np.float32(mask.mean() > 0.2)

        return img_96, label

    except Exception:
        return None, np.float32(-1.0)


def prepare_input(img_96: np.ndarray) -> np.ndarray:
    """
    Add the batch dimension required by TFLite.

    Args:
        img_96: np.ndarray shape (96, 96, 3) float32 in [0, 1]

    Returns:
        np.ndarray shape (1, 96, 96, 3) float32
    """
    return np.expand_dims(img_96, axis=0)
