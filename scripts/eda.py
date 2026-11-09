"""Task 1.3 - visual EDA: sample grid per class + width/height/file-size histograms."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from visionguard.config import FIGURES_DIR, REPORTS_DIR, ROOT, SEED  # noqa: E402
from visionguard.preprocess import load_image  # noqa: E402

COLORS = {"Real": "#2a6f97", "AI-generated": "#c8553d"}


def sample_grid(df: pd.DataFrame, n: int = 8) -> None:
    classes = ["Real", "AI-generated"]
    fig, axes = plt.subplots(len(classes), n, figsize=(2.1 * n, 2.4 * len(classes)))
    for r, cls in enumerate(classes):
        g = df[df["class"] == cls].sample(n, random_state=SEED)
        for c, (_, row) in enumerate(g.iterrows()):
            ax = axes[r, c]
            ax.imshow(load_image(str(ROOT / row.path), 256))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{int(row.width)}x{int(row.height)}", fontsize=8)
            if c == 0:
                ax.set_ylabel(cls, fontsize=11)
    fig.suptitle("Random samples per class (resized to square for display; title = original size)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_sample_grid.png", dpi=130)
    plt.close(fig)


def histograms(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    specs = [
        ("width", "Width (px)", np.linspace(0, 6000, 41)),
        ("height", "Height (px)", np.linspace(0, 6000, 41)),
        ("file_size_kb", "File size (KB, log scale)", np.logspace(0, np.log10(df.file_size_kb.max() + 1), 41)),
    ]
    for ax, (col, label, bins) in zip(axes, specs):
        for cls, g in df.groupby("class"):
            ax.hist(g[col].clip(upper=bins[-1]), bins=bins, alpha=0.55, label=f"{cls} (median {g[col].median():.0f})",
                    color=COLORS[cls])
        ax.set_xlabel(label)
        ax.set_ylabel("Images")
        ax.legend(fontsize=8)
        if col == "file_size_kb":
            ax.set_xscale("log")
    fig.suptitle("Image dimension and file-size distributions per class")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_histograms.png", dpi=130)
    plt.close(fig)


def aspect_and_formats(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for cls, g in df.groupby("class"):
        axes[0].hist((g.width / g.height).clip(0.2, 3), bins=40, alpha=0.55, label=cls, color=COLORS[cls])
    axes[0].axvline(1.0, color="k", lw=0.8, ls="--")
    axes[0].set_xlabel("Aspect ratio (width / height)")
    axes[0].set_ylabel("Images")
    axes[0].legend()
    ct = pd.crosstab(df["format"], df["class"])
    ct.plot.bar(ax=axes[1], color=[COLORS[c] for c in ct.columns])
    axes[1].set_ylabel("Images")
    axes[1].set_xlabel("Decoded format")
    axes[1].tick_params(axis="x", rotation=0)
    fig.suptitle("Aspect ratio and file format per class")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_aspect_formats.png", dpi=130)
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(REPORTS_DIR / "audit.csv")
    df = df[df.readable]
    sample_grid(df)
    histograms(df)
    aspect_and_formats(df)
    print("figures written to", FIGURES_DIR)


if __name__ == "__main__":
    main()
