from pathlib import Path
import cv2
import fitz
import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}


def load_pages(path: str, dpi: int = 200):
    """Return a list of BGR page images from a PDF or image file."""
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".pdf":
        doc = fitz.open(path)
        scale = dpi / 72.0
        matrix = fitz.Matrix(scale, scale)
        pages = []
        for page in doc:
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            arr = np.frombuffer(pix.samples, dtype=np.uint8)
            arr = arr.reshape(pix.height, pix.width, pix.n)
            if pix.n == 3:
                bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            else:
                bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
            pages.append(bgr)
        return pages

    if suffix in IMAGE_EXTS:
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f"Could not read image: {path}")
        return [image]

    raise ValueError(f"Unsupported file type: {suffix}")
