"""Create stratified train/val/test split lists (70/15/15) from the audit table.

Corrupt files are dropped. For exact duplicates only the first copy is kept, so
the same picture can never sit in both train and test (data leakage). If a
duplicate group spans both classes its label is ambiguous and all copies go.
Usage: python scripts/make_splits.py  (run audit_dataset.py first)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from visionguard.config import REPORTS_DIR, SEED, SPLITS_DIR  # noqa: E402


def main() -> None:
    df = pd.read_csv(REPORTS_DIR / "audit.csv")
    n0 = len(df)
    df = df[df.readable & ~df.cross_class_dup]
    df = df.drop_duplicates("sha256", keep="first")
    print(f"kept {len(df)} of {n0} files after removing corrupt files and duplicates")
    cols = ["path", "label", "class", "sha256"]
    train, rest = train_test_split(df[cols], test_size=0.30, stratify=df.label, random_state=SEED)
    val, test = train_test_split(rest, test_size=0.50, stratify=rest.label, random_state=SEED)
    SPLITS_DIR.mkdir(exist_ok=True)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.sort_values("path").to_csv(SPLITS_DIR / f"{name}.csv", index=False)
        print(f"{name:5s} n={len(part):4d}  AI={int(part.label.sum()):4d}  Real={int((part.label == 0).sum()):4d}")


if __name__ == "__main__":
    main()
