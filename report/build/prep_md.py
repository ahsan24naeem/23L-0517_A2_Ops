"""Pre-process the Markdown before pandoc.

* separates consecutive screenshot boxes (blockquotes) so each is its own box
* for every "> **Screenshot X.Y** ..." box whose image exists as report/shots/X.Y.png
  (ranges like 6.1-6.6 are expanded), the box is replaced by the image(s) with a caption
"""
import re
import sys
from pathlib import Path

src, dst = sys.argv[1:3]
SHOTS = Path(__file__).resolve().parents[1] / "shots"


def ids_in(line: str) -> list[str]:
    ids = []
    for a, b in re.findall(r"(\d+\.\d+)(?:\s*[–-]\s*(\d+\.\d+))?", line):
        if b and a.split(".")[0] == b.split(".")[0]:
            major, lo, hi = a.split(".")[0], int(a.split(".")[1]), int(b.split(".")[1])
            ids += [f"{major}.{i}" for i in range(lo, hi + 1)]
        else:
            ids.append(a)
    return ids


out, prev_quote = [], False
for line in Path(src).read_text().splitlines():
    q = line.startswith("> ")
    if q and "Screenshot" in line:
        segs = re.split(r"\*\*Screenshots?\s+", line)[1:]
        found, missing = [], []
        for seg in segs:
            head, _, desc = seg.partition("**")
            desc = desc.strip().lstrip("—").strip()
            for i in ids_in(head):
                (found if (SHOTS / f"{i}.png").exists() else missing).append((i, desc))
        if found:
            out.append("")
            for i, desc in found:
                out.append(f"![Screenshot {i} — {desc}](shots/{i}.png){{width=100%}}")
                out.append("")
            if missing:
                out.append("> " + "  ".join(f"**Screenshot {i}** — {d}" for i, d in missing))
                out.append("")
            prev_quote = False
            continue
    if q and prev_quote:
        out.append("")
    out.append(line)
    prev_quote = q
Path(dst).write_text("\n".join(out) + "\n")
