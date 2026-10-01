"""Decode uploads into page images. Validates type by content (magic bytes), not extension."""
from __future__ import annotations

import io
from dataclasses import dataclass, field

import cv2
import numpy as np

MAX_UPLOAD_BYTES = 80_000_000
MAX_PDF_PAGES = 60
MAX_PIXELS = 220_000_000
PDF_MAX_DPI = 300
PDF_MAX_SIDE = 3600


class InputError(ValueError):
    """User-facing input problem. `code` lets the UI offer the right recovery."""

    def __init__(self, message: str, code: str = "invalid_input", hint: str | None = None):
        super().__init__(message)
        self.code = code
        self.hint = hint


@dataclass
class PageSource:
    index: int
    image: np.ndarray  # BGR uint8, display orientation applied (EXIF / PDF /Rotate)
    is_photo: bool
    dpi: float | None = None
    pdf_size_pt: tuple[float, float] | None = None  # displayed (rotated) page size in points
    pdf_rotation: int = 0
    has_text_layer: bool = False
    exif_orientation: int | None = None


@dataclass
class Document:
    kind: str  # "image" | "pdf"
    mime: str
    pages: list[PageSource]
    warnings: list[str] = field(default_factory=list)


def sniff(data: bytes) -> str | None:
    h = data[:32]
    if h.startswith(b"%PDF") or b"%PDF" in data[:1024]:
        return "application/pdf"
    if h.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if h.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if h[4:8] == b"ftyp" and h[8:12] in (b"heic", b"heix", b"hevc", b"hevx", b"mif1", b"msf1", b"heim", b"heis", b"avif"):
        return "image/heic" if h[8:12] != b"avif" else "image/avif"
    if h.startswith(b"RIFF") and h[8:12] == b"WEBP":
        return "image/webp"
    if h.startswith(b"II*\x00") or h.startswith(b"MM\x00*"):
        return "image/tiff"
    if h.startswith(b"BM"):
        return "image/bmp"
    return None


SUPPORTED = ["JPG/JPEG", "PNG", "HEIC/HEIF", "WEBP", "TIFF", "BMP", "PDF"]


def load_document(data: bytes, filename: str = "") -> Document:
    if not data:
        raise InputError("The file is empty.", "empty_file")
    if len(data) > MAX_UPLOAD_BYTES:
        raise InputError(f"File is too large ({len(data) / 1e6:.0f} MB). Maximum is {MAX_UPLOAD_BYTES // 1_000_000} MB.",
                         "too_large", "Compress the file or split the PDF into smaller parts.")
    mime = sniff(data)
    if mime is None:
        raise InputError(f"Unsupported file type{(' (' + filename + ')') if filename else ''}.", "unsupported_type",
                         "Upload a photo or scan (" + ", ".join(SUPPORTED) + ").")
    if mime == "application/pdf":
        return _load_pdf(data)
    return _load_image(data, mime)


def _load_image(data: bytes, mime: str) -> Document:
    from PIL import Image, ImageOps

    Image.MAX_IMAGE_PIXELS = MAX_PIXELS
    if mime in ("image/heic", "image/avif"):
        try:
            import pillow_heif

            pillow_heif.register_heif_opener()
        except Exception:
            raise InputError("HEIC images are not supported on this server.", "unsupported_type",
                             "Export the photo as JPG and upload again.")
    warnings = []
    try:
        im = Image.open(io.BytesIO(data))
        exif_orient = None
        try:
            exif_orient = im.getexif().get(0x0112)
        except Exception:
            pass
        dpi = im.info.get("dpi", (None,))[0]
        im.load()
        im = ImageOps.exif_transpose(im)
    except Image.DecompressionBombError:
        raise InputError("Image is too large to process safely.", "too_large", "Resize the image below 200 megapixels.")
    except InputError:
        raise
    except Exception as e:
        raise InputError("The image could not be decoded — the file looks corrupt or truncated.", "corrupt_file",
                         "Re-export or re-download the original and try again.") from e
    if im.width < 16 or im.height < 16:
        raise InputError("Image is too small to contain a readable document.", "too_small")
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    if im.mode == "I;16" or im.mode == "I":
        arr = np.asarray(im, dtype=np.float32)
        arr = (arr / max(1.0, arr.max()) * 255).astype(np.uint8)
        im = Image.fromarray(arr)
    im = im.convert("RGB")
    arr = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
    if min(arr.shape[:2]) < 600:
        warnings.append("Low resolution image — small text may be unreliable. A sharper photo will give better results.")
    if exif_orient and exif_orient != 1:
        warnings.append(f"Applied camera orientation (EXIF {exif_orient}).")
    page = PageSource(0, arr, is_photo=True, dpi=float(dpi) if dpi else None, exif_orientation=exif_orient)
    return Document("image", mime, [page], warnings)


def _load_pdf(data: bytes) -> Document:
    import pymupdf

    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as e:
        raise InputError("The PDF could not be opened — it looks corrupt.", "corrupt_file") from e
    if doc.needs_pass:
        raise InputError("This PDF is password-protected.", "encrypted_pdf", "Remove the password and upload again.")
    if doc.page_count == 0:
        raise InputError("The PDF has no pages.", "empty_document")
    warnings = []
    n = doc.page_count
    if n > MAX_PDF_PAGES:
        warnings.append(f"Only the first {MAX_PDF_PAGES} of {n} pages were loaded.")
        n = MAX_PDF_PAGES
    pages = []
    for i in range(n):
        p = doc[i]
        r = p.rect  # displayed (rotation applied) size in points
        dpi = min(PDF_MAX_DPI, PDF_MAX_SIDE * 72.0 / max(r.width, r.height))
        try:
            pix = p.get_pixmap(dpi=int(dpi), alpha=False, colorspace=pymupdf.csRGB)
        except Exception as e:
            raise InputError(f"Page {i + 1} of the PDF could not be rendered.", "corrupt_file") from e
        arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
        arr = cv2.cvtColor(arr[..., :3], cv2.COLOR_RGB2BGR)
        words = p.get_text("words")
        imgs = p.get_images(full=True)
        # an image-only page that is basically one big picture: treat like a photo/scan
        is_photo = (not words) and len(imgs) >= 1
        pages.append(PageSource(i, arr, is_photo=is_photo, dpi=float(int(dpi)),
                                pdf_size_pt=(float(r.width), float(r.height)), pdf_rotation=int(p.rotation),
                                has_text_layer=bool(words)))
    return Document("pdf", "application/pdf", pages, warnings)
