import cv2
import numpy as np

from config import GRID_ROWS, GRID_COLS


def _trim_page(image: np.ndarray) -> np.ndarray:
    """
    Remove only a tiny outer border from the photographed/scanned page.
    """
    h, w = image.shape[:2]

    pad_x = int(w * 0.008)
    pad_y = int(h * 0.008)

    return image[
        pad_y:h - pad_y,
        pad_x:w - pad_x
    ]


def _deskew_card(card: np.ndarray) -> np.ndarray:
    """
    Correct small rotations by detecting long horizontal lines
    in the student form.
    """
    gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(
        gray,
        60,
        160
    )

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=80,
        minLineLength=max(80, card.shape[1] // 3),
        maxLineGap=18,
    )

    angles = []

    if lines is not None:
        # Normalize OpenCV output.
        # HoughLinesP may return (N, 1, 4) or (N, 4).
        lines = np.asarray(lines).reshape(-1, 4)

        for x1, y1, x2, y2 in lines:
            x1 = float(x1)
            y1 = float(y1)
            x2 = float(x2)
            y2 = float(y2)

            angle = np.degrees(
                np.arctan2(
                    y2 - y1,
                    x2 - x1
                )
            )

            # Only use almost-horizontal lines.
            if -8.0 <= angle <= 8.0:
                angles.append(angle)

    # No useful horizontal lines found.
    if not angles:
        return card

    angle = float(np.median(angles))

    # Already basically straight.
    if abs(angle) < 0.25:
        return card

    h, w = card.shape[:2]

    matrix = cv2.getRotationMatrix2D(
        (w / 2, h / 2),
        angle,
        1.0
    )

    rotated = cv2.warpAffine(
        card,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )

    return rotated


def split_cards(image: np.ndarray):
    """
    Split the photographed sheet into four student cards.

    Returns:
        [
            (1, card_image),
            (2, card_image),
            (3, card_image),
            (4, card_image)
        ]
    """
    if image is None:
        raise ValueError("split_cards received an empty image")

    page = _trim_page(image)

    h, w = page.shape[:2]

    cards = []

    # Small overlap helps avoid cutting content near the center boundaries.
    overlap_x = int(w * 0.018)
    overlap_y = int(h * 0.018)

    index = 1

    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):

            x1 = max(
                0,
                int(col * w / GRID_COLS) - overlap_x
            )

            x2 = min(
                w,
                int((col + 1) * w / GRID_COLS) + overlap_x
            )

            y1 = max(
                0,
                int(row * h / GRID_ROWS) - overlap_y
            )

            y2 = min(
                h,
                int((row + 1) * h / GRID_ROWS) + overlap_y
            )

            crop = page[y1:y2, x1:x2].copy()

            # Correct small rotation before OCR/OMR.
            crop = _deskew_card(crop)

            cards.append((index, crop))

            index += 1

    return cards