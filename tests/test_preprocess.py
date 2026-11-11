import numpy as np

from tests.conftest import make_image_bytes
from visionguard.preprocess import is_decodable, load_image


def test_load_image_shape_dtype_and_rgb():
    for fmt in ["PNG", "JPEG", "WEBP", "GIF", "BMP"]:
        arr = load_image(make_image_bytes(fmt), 224)
        assert arr.shape == (224, 224, 3)
        assert arr.dtype == np.uint8


def test_greyscale_and_rgba_become_rgb():
    import io

    from PIL import Image
    for mode in ["L", "RGBA", "P"]:
        buf = io.BytesIO()
        Image.new(mode, (30, 40)).save(buf, format="PNG")
        assert load_image(buf.getvalue(), 32).shape == (32, 32, 3)


def test_is_decodable_rejects_garbage():
    assert is_decodable(make_image_bytes())
    assert not is_decodable(b"\x89PNG\r\n\x1a\n" + b"not really a png" * 10)
    assert not is_decodable(b"")


def test_decision_confidence_is_consistent_with_label():
    from visionguard.inference import certainty_label, decision_confidence
    t = 0.71
    assert abs(decision_confidence(t, t) - 0.5) < 1e-9          # at the threshold: coin flip
    assert abs(decision_confidence(1.0, t) - 1.0) < 1e-9
    assert abs(decision_confidence(0.0, t) - 1.0) < 1e-9
    c = decision_confidence(0.571, t)                            # "Real" side
    assert 0.5 < c < 0.65 and certainty_label(c) == "uncertain"
    assert decision_confidence(0.02, t) > 0.95
