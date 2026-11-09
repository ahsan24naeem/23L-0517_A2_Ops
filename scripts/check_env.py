"""Task 1.1 - record exact library versions and whether TensorFlow sees a GPU."""
import getpass
import platform
import socket
import sys
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version

PKGS = ["tensorflow", "tensorflow-cpu", "keras", "keras-tuner", "mlflow", "dvc", "numpy", "pandas", "pillow",
        "scikit-learn", "matplotlib", "flask", "sqlalchemy", "gunicorn"]

print(f"user@host : {getpass.getuser()}@{socket.gethostname()}   {datetime.now():%Y-%m-%d %H:%M:%S}")
print(f"python    : {sys.version.split()[0]}  ({sys.executable})")
print(f"platform  : {platform.platform()}")
for p in PKGS:
    try:
        print(f"{p:14s}: {version(p)}")
    except PackageNotFoundError:
        pass
import tensorflow as tf  # noqa: E402

gpus = tf.config.list_physical_devices("GPU")
print(f"TF devices: {[d.name for d in tf.config.list_physical_devices()]}")
print(f"GPU available: {bool(gpus)} {gpus if gpus else '(training runs on CPU)'}")
print(f"built with CUDA: {tf.test.is_built_with_cuda()}")
