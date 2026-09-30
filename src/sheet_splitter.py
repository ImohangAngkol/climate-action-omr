import cv2
import numpy as np
from config import GRID_ROWS, GRID_COLS


def normalize_page(page):
    """Basic scan cleanup. Perspective correction can be added later."""
    gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return page, gray


def split_into_cards(page):
    """MVP splitter for the supplied 3x2 sheet layout."""
    h, w = page.shape[:2]
    cards = []
    for r in range(GRID_ROWS):
        y1 = round(r * h / GRID_ROWS)
        y2 = round((r + 1) * h / GRID_ROWS)
        for c in range(GRID_COLS):
            x1 = round(c * w / GRID_COLS)
            x2 = round((c + 1) * w / GRID_COLS)
            card = page[y1:y2, x1:x2].copy()
            cards.append(card)
    return cards


def crop_normalized(image, roi):
    x1f, y1f, x2f, y2f = roi
    h, w = image.shape[:2]
    x1, y1 = int(x1f*w), int(y1f*h)
    x2, y2 = int(x2f*w), int(y2f*h)
    return image[y1:y2, x1:x2].copy()
