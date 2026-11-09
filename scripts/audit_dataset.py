"""Task 1.2 - dataset audit.

Walks data/raw/<class folder>/ and builds one row per file with:
class, path, file size, width, height, colour mode, format, SHA-256, readable.
Detects unreadable files and exact duplicates (same SHA-256), and writes:
  reports/audit.csv            full per-image table
  reports/audit_summary.md     counts / min-median-max / corrupt / duplicates
  reports/audit_summary.json   same numbers, machine readable
Usage:  python scripts/audit_dataset.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from visionguard.config import CLASS_FOLDERS, CLASS_NAMES, DATA_DIR, REPORTS_DIR, ROOT  # noqa: E402

Image.MAX_IMAGE_PIXELS = 120_000_000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect(path: Path) -> dict:
    row = {"width": None, "height": None, "mode": None, "format": None, "readable": False, "error": ""}
    try:
        with Image.open(path) as im:
            row.update(width=im.width, height=im.height, mode=im.mode, format=im.format)
            im.load()  # full decode, catches truncated files
        row["readable"] = True
    except Exception as e:  # noqa: BLE001
        row["error"] = f"{type(e).__name__}: {e}"[:200]
    return row


def main() -> None:
    rows = []
    for folder, label in CLASS_FOLDERS.items():
        for p in sorted((DATA_DIR / folder).rglob("*")):
            if not p.is_file() or p.name.startswith("."):
                continue
            row = {
                "class": CLASS_NAMES[label],
                "label": label,
                "path": str(p.relative_to(ROOT)),
                "extension": p.suffix.lower() or "(none)",
                "file_size_kb": round(p.stat().st_size / 1024, 1),
                "sha256": sha256(p),
            }
            row.update(inspect(p))
            rows.append(row)
    df = pd.DataFrame(rows)
    df["dup_group_size"] = df.groupby("sha256")["sha256"].transform("size")
    df["is_duplicate"] = df.duplicated("sha256", keep="first")  # later copies
    df["cross_class_dup"] = df.groupby("sha256")["label"].transform("nunique") > 1
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(REPORTS_DIR / "audit.csv", index=False)

    ok = df[df.readable]
    summary = {"total_files": int(len(df))}
    lines = ["# Dataset audit summary", "", f"Total files: {len(df)}", ""]
    lines += ["| Class | Files | Readable | Corrupt | Duplicate copies |", "|---|---|---|---|---|"]
    per_class = {}
    for cls, g in df.groupby("class"):
        per_class[cls] = {
            "files": int(len(g)),
            "readable": int(g.readable.sum()),
            "corrupt": int((~g.readable).sum()),
            "duplicates": int(g.is_duplicate.sum()),
        }
        c = per_class[cls]
        lines.append(f"| {cls} | {c['files']} | {c['readable']} | {c['corrupt']} | {c['duplicates']} |")
    summary["per_class"] = per_class

    lines += ["", "| Class | Stat | Width (px) | Height (px) | File size (KB) |", "|---|---|---|---|---|"]
    stats = {}
    for cls, g in ok.groupby("class"):
        stats[cls] = {}
        for stat, fn in [("min", "min"), ("median", "median"), ("max", "max")]:
            vals = {c: float(getattr(g[c], fn)()) for c in ["width", "height", "file_size_kb"]}
            stats[cls][stat] = vals
            lines.append(
                f"| {cls} | {stat} | {vals['width']:.0f} | {vals['height']:.0f} | {vals['file_size_kb']:.1f} |"
            )
    summary["stats"] = stats

    lines += ["", "| Class | Format | Count |", "|---|---|---|"]
    fmt = ok.groupby(["class", "format"]).size().reset_index(name="n")
    summary["formats"] = {f"{r['class']}/{r.format}": int(r.n) for _, r in fmt.iterrows()}
    for _, r in fmt.iterrows():
        lines.append(f"| {r['class']} | {r.format} | {r.n} |")
    lines += ["", "| Class | Colour mode | Count |", "|---|---|---|"]
    modes = ok.groupby(["class", "mode"]).size().reset_index(name="n")
    summary["modes"] = {f"{r['class']}/{r['mode']}": int(r.n) for _, r in modes.iterrows()}
    for _, r in modes.iterrows():
        lines.append(f"| {r['class']} | {r['mode']} | {r.n} |")

    corrupt = df[~df.readable]
    dups = df[df.dup_group_size > 1].sort_values("sha256")
    summary["corrupt_files"] = corrupt[["path", "error"]].to_dict("records")
    summary["duplicate_groups"] = int(dups.sha256.nunique())
    summary["duplicate_copies_removed"] = int(df.is_duplicate.sum())
    summary["cross_class_duplicate_groups"] = int(df[df.cross_class_dup].sha256.nunique())
    lines += ["", f"Corrupt / unreadable files: {len(corrupt)}"]
    for _, r in corrupt.iterrows():
        lines.append(f"- `{r.path}` - {r.error}")
    lines += [
        "",
        f"Exact-duplicate groups (same SHA-256): {summary['duplicate_groups']}, "
        f"extra copies: {summary['duplicate_copies_removed']}, "
        f"groups spanning both classes: {summary['cross_class_duplicate_groups']}",
    ]
    for sha, g in dups.groupby("sha256"):
        lines.append(f"- `{sha[:12]}`: " + ", ".join(f"`{p}`" for p in g.path))

    (REPORTS_DIR / "audit_summary.md").write_text("\n".join(lines) + "\n")
    (REPORTS_DIR / "audit_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
