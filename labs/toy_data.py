"""Fashion-MNIST loader for the tool labs.

Tries keras.datasets first; if Google Storage is unreachable it falls back to the
official Zalando GitHub copy of the same files.
"""
from __future__ import annotations

import gzip
import urllib.request
from pathlib import Path

import numpy as np

GITHUB = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
FILES = {
    "x_train": "train-images-idx3-ubyte.gz", "y_train": "train-labels-idx1-ubyte.gz",
    "x_test": "t10k-images-idx3-ubyte.gz", "y_test": "t10k-labels-idx1-ubyte.gz",
}


def _from_github():
    cache = Path.home() / ".keras" / "datasets" / "fashion-mnist-gh"
    cache.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for key, fname in FILES.items():
        p = cache / fname
        if not p.exists():
            urllib.request.urlretrieve(GITHUB + fname, p)
        with gzip.open(p, "rb") as f:
            buf = f.read()
        if key.startswith("x"):
            arrays[key] = np.frombuffer(buf, np.uint8, offset=16).reshape(-1, 28, 28)
        else:
            arrays[key] = np.frombuffer(buf, np.uint8, offset=8)
    return (arrays["x_train"], arrays["y_train"]), (arrays["x_test"], arrays["y_test"])


def load_fashion_mnist():
    try:
        import keras

        return keras.datasets.fashion_mnist.load_data()
    except Exception:
        return _from_github()
