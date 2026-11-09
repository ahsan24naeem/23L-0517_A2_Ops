"""Task 2.1 figure: one augmented training batch from the tf.data pipeline."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from visionguard.config import FIGURES_DIR  # noqa: E402
from visionguard.data import make_dataset, read_split  # noqa: E402

ds = make_dataset(read_split("train"), 224, 32, training=True, augment=True, cache_name="train")
x, y = next(iter(ds))
fig, axes = plt.subplots(2, 8, figsize=(18, 5.8))
for i, ax in enumerate(axes.ravel()):
    ax.imshow(np.clip(x[i].numpy(), 0, 255).astype("uint8"))
    ax.set_title("AI-generated" if y[i] == 1 else "Real", fontsize=9)
    ax.axis("off")
fig.suptitle("One augmented training batch (224x224, random flip / rotation / zoom / contrast)")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "augmented_batch.png", dpi=110)
print("saved", x.shape, y.numpy()[:16])
