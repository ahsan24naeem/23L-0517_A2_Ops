"""Task 4.1 demo: load the champion once, classify one known real and one known AI image."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
t0 = time.time()
from visionguard.inference import Predictor  # noqa: E402

p = Predictor("models/champion.keras")
print(f"loaded {p.model_name} v{p.model_version} (threshold {p.threshold}, input {p.img_size}px) "
      f"ONCE in {time.time() - t0:.1f}s")
for f in ["data/raw/RealArt/1032249.jpg", "data/raw/AiArtData/midjourney20images.jpg"]:
    r = p.predict(open(f, "rb").read())
    print(f"{f:45s} -> {r.label:12s} conf={r.confidence:.3f} ({r.certainty}) p_ai={r.p_ai:.3f} {r.latency_ms} ms")
