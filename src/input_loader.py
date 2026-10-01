from pathlib import Path
import cv2
import numpy as np


def load_image(path: Path) -> np.ndarray:
    path = Path(path)
    # imdecode handles Windows paths/non-ASCII filenames better than cv2.imread.
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not open image: {path}")
    return image


def sharpness_score(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())
