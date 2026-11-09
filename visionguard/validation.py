"""Upload validation shared by the web app and the model API."""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass

from PIL import Image
from werkzeug.utils import secure_filename

from .preprocess import is_decodable

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "gif", "bmp"}


class UploadError(Exception):
    def __init__(self, status: int, title: str, message: str):
        super().__init__(message)
        self.status, self.title, self.message = status, title, message


@dataclass
class UploadInfo:
    safe_name: str
    sha256: str
    size_bytes: int
    width: int
    height: int


def validate_upload(filename: str | None, data: bytes, max_bytes: int,
                    allowed: set[str] = ALLOWED_EXTENSIONS) -> UploadInfo:
    if not filename:
        raise UploadError(400, "No file selected", "Please choose an image file to upload.")
    # secure_filename strips paths ("../../etc/passwd"), spaces and unsafe characters
    safe = secure_filename(filename)[:200] or "upload"
    ext = safe.rsplit(".", 1)[-1].lower() if "." in safe else ""
    if ext not in allowed:
        raise UploadError(415, "Unsupported file type",
                          f"'.{ext or '?'}' files are not accepted. Allowed: {', '.join(sorted(allowed))}.")
    if len(data) == 0:
        raise UploadError(400, "Empty file", "The uploaded file is empty.")
    if len(data) > max_bytes:
        raise UploadError(413, "File too large",
                          f"The file is {len(data) / 1e6:.1f} MB; the limit is {max_bytes / 1e6:.1f} MB.")
    if not is_decodable(data):
        raise UploadError(422, "Not a valid image",
                          "The file has an image extension but could not be decoded as an image "
                          "(it may be corrupt, truncated or a different file type renamed).")
    with Image.open(io.BytesIO(data)) as im:
        w, h = im.size
    return UploadInfo(safe, hashlib.sha256(data).hexdigest(), len(data), w, h)
