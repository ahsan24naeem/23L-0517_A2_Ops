"""tf.data input pipeline.

Stages (in order):
  1. from_tensor_slices   - (path, label) pairs read from a split CSV
  2. map(load)            - decode + RGB + resize via the shared Pillow function
                            (tf.numpy_function, parallel with AUTOTUNE)
  3. cache                - decoded uint8 tensors cached to disk, so the slow
                            JPEG/PNG decoding happens only once per image size
  4. shuffle (train only) - full-buffer shuffle, reshuffled every epoch
  5. batch
  6. augment (train only) - random flip / rotation / zoom / contrast, applied
                            on batches so it runs vectorised; val/test are never
                            augmented so their metrics are comparable
  7. prefetch             - overlaps input preparation with model execution
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import layers

from .config import CACHE_DIR, ROOT, SEED
from .preprocess import load_image

AUTOTUNE = tf.data.AUTOTUNE


def read_split(name: str, splits_dir: Path | None = None) -> pd.DataFrame:
    splits_dir = splits_dir or (ROOT / "splits")
    df = pd.read_csv(splits_dir / f"{name}.csv")
    df["abspath"] = df["path"].apply(lambda p: str(ROOT / p))
    return df


def build_augmenter(seed: int = SEED, strength: float = 1.0) -> keras.Sequential:
    """Mild geometric/photometric augmentation.

    Deliberately NO blur, noise or JPEG re-compression: those would erase the
    high-frequency artefacts that often distinguish generated images.
    """
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal", seed=seed),
            layers.RandomRotation(0.05 * strength, seed=seed),
            layers.RandomZoom(0.10 * strength, seed=seed),
            layers.RandomContrast(0.10 * strength, seed=seed),
        ],
        name="augmentation",
    )


def _loader(size: int):
    def _np_load(path: bytes) -> np.ndarray:
        return load_image(path.decode("utf-8"), size)

    def _tf_load(path, label):
        img = tf.numpy_function(_np_load, [path], tf.uint8)
        img = tf.ensure_shape(img, (size, size, 3))
        return img, tf.cast(label, tf.float32)

    return _tf_load


def make_dataset(
    df: pd.DataFrame,
    size: int,
    batch_size: int,
    training: bool,
    augment: bool = True,
    seed: int = SEED,
    cache_name: str | None = None,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((df["abspath"].values, df["label"].values))
    ds = ds.map(_loader(size), num_parallel_calls=AUTOTUNE)
    if cache_name:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        ds = ds.cache(str(CACHE_DIR / f"{cache_name}_{size}"))
    else:
        ds = ds.cache()
    if training:
        ds = ds.shuffle(len(df), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=AUTOTUNE)
    if training and augment:
        aug = build_augmenter(seed)
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)
