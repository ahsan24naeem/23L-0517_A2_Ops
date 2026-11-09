import pytest

from visionguard.validation import UploadError, validate_upload


def test_valid_upload(png_bytes):
    info = validate_upload("../../etc/My Photo.png", png_bytes, 10_000_000)
    assert info.safe_name == "etc_My_Photo.png"     # path components and spaces removed
    assert info.width == 64 and info.height == 48 and len(info.sha256) == 64


@pytest.mark.parametrize("name,status", [("notes.txt", 415), ("script.exe", 415), (None, 400)])
def test_rejects_bad_names(png_bytes, name, status):
    with pytest.raises(UploadError) as e:
        validate_upload(name, png_bytes, 10_000_000)
    assert e.value.status == status


def test_rejects_too_large(png_bytes):
    with pytest.raises(UploadError) as e:
        validate_upload("a.png", png_bytes, 10)
    assert e.value.status == 413


def test_rejects_fake_image():
    with pytest.raises(UploadError) as e:
        validate_upload("fake.jpg", b"this is text pretending to be a jpeg", 10_000)
    assert e.value.status == 422
