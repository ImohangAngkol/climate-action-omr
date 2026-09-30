import re
import cv2
import pytesseract
from config import NAME_ROI, SCHOOL_ROI
from src.sheet_splitter import crop_normalized


def _prep_text_roi(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def _clean(text):
    text = re.sub(r"\s+", " ", text or "").strip()
    # Drop field labels if OCR picks them up.
    text = re.sub(r"^(name|school)\s*[:\-]?\s*", "", text, flags=re.I)
    return text


def read_name_and_school(card, enabled=True):
    if not enabled:
        return "", ""

    name_roi = _prep_text_roi(crop_normalized(card, NAME_ROI))
    school_roi = _prep_text_roi(crop_normalized(card, SCHOOL_ROI))

    cfg = "--psm 7"
    name = _clean(pytesseract.image_to_string(name_roi, config=cfg))
    school = _clean(pytesseract.image_to_string(school_roi, config=cfg))
    return name, school
