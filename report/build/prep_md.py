import re, sys
src, dst = sys.argv[1:3]
out = []
prev_quote = False
for line in open(src).read().splitlines():
    q = line.startswith("> ")
    if q and prev_quote:
        out.append("")
    out.append(line)
    prev_quote = q
open(dst, "w").write("\n".join(out) + "\n")
