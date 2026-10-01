import re
import shutil
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

from config import TESSERACT_CMD, KNOWN_SCHOOLS, MIN_DATE_YEAR, MAX_DATE_YEAR


def _configure_tesseract():
    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
        return True
    found = shutil.which("tesseract")
    if found:
        pytesseract.pytesseract.tesseract_cmd = found
        return True
    for candidate in [
        Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
        Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
    ]:
        if candidate.exists():
            pytesseract.pytesseract.tesseract_cmd = str(candidate)
            return True
    return False


TESSERACT_AVAILABLE = _configure_tesseract()


def _clean(text):
    text = text.replace("\n", " ").replace("\r", " ")
    text = text.replace("|", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip(" _.,;:-")


def _norm_letters(s):
    return re.sub(r"[^a-z]", "", s.lower())


def _canonical_school(text):
    if not text:
        return "", 0.0
    scored = [(SequenceMatcher(None, _norm_letters(text), _norm_letters(s)).ratio(), s)
              for s in KNOWN_SCHOOLS]
    score, school = max(scored, default=(0.0, text))
    return (school, score * 100.0) if score >= 0.52 else (text, score * 100.0)


def _variants(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    gray = cv2.resize(gray, None, fx=2.6, fy=2.6, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    blur = cv2.GaussianBlur(clahe, (3, 3), 0)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 31, 13)
    return [gray, clahe, otsu, adaptive]


def _ocr_candidate(image, psm=7, whitelist=None):
    config = f"--oem 3 --psm {psm}"
    if whitelist:
        config += f" -c tessedit_char_whitelist={whitelist}"
    data = pytesseract.image_to_data(image, config=config, output_type=Output.DICT)
    words, confs = [], []
    for text, conf in zip(data.get("text", []), data.get("conf", [])):
        text = _clean(text)
        try:
            c = float(conf)
        except Exception:
            c = -1
        if text:
            words.append(text)
            if c >= 0:
                confs.append(c)
    return _clean(" ".join(words)), (float(np.mean(confs)) if confs else 0.0)


def _ocr_best(crop, kind="text"):
    if not TESSERACT_AVAILABLE or crop.size == 0:
        return "", 0.0

    variants = _variants(crop)
    # Fast path: CLAHE grayscale + single-line OCR. Most readable samples finish here.
    text1, conf1 = _ocr_candidate(variants[1], psm=7)
    candidates = []
    if text1:
        candidates.append((conf1, text1))

    # Only pay for a second OCR pass when the first pass is weak.
    if not text1 or conf1 < 42:
        text2, conf2 = _ocr_candidate(variants[2], psm=13)
        if text2:
            candidates.append((conf2, text2))

    if not candidates:
        return "", 0.0

    def score(item):
        conf, text = item
        alpha = sum(ch.isalpha() for ch in text)
        digits = sum(ch.isdigit() for ch in text)
        plausibility = min(16, alpha + digits) * 0.6
        if kind == "date":
            plausibility += digits * 1.1
        elif kind == "name":
            plausibility += alpha * 0.35
        return conf + plausibility

    conf, text = max(candidates, key=score)
    return text, conf


def _find_info_table(card):
    gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 31, 15)
    horizontal = cv2.morphologyEx(
        bw, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, card.shape[1]//18), 1))
    )
    vertical = cv2.morphologyEx(
        bw, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(18, card.shape[0]//35)))
    )
    grid = cv2.bitwise_or(horizontal, vertical)
    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = gray.shape
    candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > W * 0.45 and H * 0.18 < y < H * 0.60 and H * 0.065 < h < H * 0.25:
            # Prefer table-like 3-row boxes around the expected middle-upper region.
            center_penalty = abs((y + h/2) - H*0.37) / H
            candidates.append((w*h - center_penalty*w*h*0.4, x, y, w, h))
    if not candidates:
        return None
    _, x, y, w, h = max(candidates)
    return x, y, w, h


def _crop_value_rows(card, table):
    x, y, w, h = table
    # The label column is ~24%; add a small inset to avoid the vertical border.
    value_x = x + int(w * 0.245)
    right = x + w - max(2, int(w * 0.01))
    row_h = h / 3.0
    crops = []
    for i in range(3):
        y1 = int(y + i * row_h + row_h * 0.08)
        y2 = int(y + (i + 1) * row_h - row_h * 0.08)
        crops.append(card[y1:y2, value_x:right])
    return crops


def _fix_common_date_ocr(text):
    t = _clean(text)
    # Only conservative substitutions near numeric date tokens.
    t = re.sub(r"(?<=\d)[Oo](?=\d)", "0", t)
    t = re.sub(r"(?<=\d)[Il](?=\d)", "1", t)
    return t


def normalize_date(text):
    """Strict parser: never turn OCR garbage into a plausible-looking date."""
    raw = _fix_common_date_ocr(text)
    if not raw:
        return ""

    # Normalize separators but retain month words.
    t = raw.lower().replace(".", " ")
    t = re.sub(r"[,]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()

    months = {
        "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
        "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
        "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
        "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
    }

    def valid(y, m, d):
        if not (MIN_DATE_YEAR <= y <= MAX_DATE_YEAR):
            return ""
        try:
            return date(y, m, d).isoformat()
        except ValueError:
            return ""

    # YYYY-MM-DD / YYYY/MM/DD
    m = re.search(r"\b(20\d{2})\s*[-/]\s*(\d{1,2})\s*[-/]\s*(\d{1,2})\b", t)
    if m:
        return valid(int(m.group(1)), int(m.group(2)), int(m.group(3)))

    # DD-MM-YYYY or DD/MM/YYYY. For these project forms, numeric input is treated day-first.
    m = re.search(r"\b(\d{1,2})\s*[-/]\s*(\d{1,2})\s*[-/]\s*(20\d{2})\b", t)
    if m:
        d, mo, y = map(int, m.groups())
        return valid(y, mo, d)

    # Month-word forms: Oct 1 2026 / 1 October 2026 / 01-October-2026.
    month_pattern = "|".join(sorted(months, key=len, reverse=True))
    m = re.search(rf"\b({month_pattern})\s*[- ]*\s*(\d{{1,2}})\s*[- ]*\s*(20\d{{2}})\b", t)
    if m:
        return valid(int(m.group(3)), months[m.group(1)], int(m.group(2)))
    m = re.search(rf"\b(\d{{1,2}})\s*[- ]*\s*({month_pattern})\s*[- ]*\s*(20\d{{2}})\b", t)
    if m:
        return valid(int(m.group(3)), months[m.group(2)], int(m.group(1)))
    return ""


def read_fields(card):
    notes = []
    table = _find_info_table(card)
    if table is None:
        return {"name": "", "school": "", "date_raw": "", "date_normalized": "", "ocr_confidence": 0.0}, ["Could not locate Name/School/Date table"]

    name_crop, school_crop, date_crop = _crop_value_rows(card, table)
    name, name_conf = _ocr_best(name_crop, "name")
    school_raw, school_conf = _ocr_best(school_crop, "school")
    date_raw, date_conf = _ocr_best(date_crop, "date")
    school, school_match = _canonical_school(school_raw)

    if not name:
        notes.append("Name OCR uncertain/blank")
    if not school_raw:
        notes.append("School OCR uncertain/blank")
    elif school_match < 52:
        notes.append("School did not confidently match a known school")
    if not date_raw:
        notes.append("Date OCR uncertain/blank")

    date_normalized = normalize_date(date_raw)
    if date_raw and not date_normalized:
        notes.append("Date rejected by strict validator")

    # Handwriting confidence is advisory; never silently 'correct' names.
    confs = [c for c, text in ((name_conf, name), (school_conf, school_raw), (date_conf, date_raw)) if text]
    overall_conf = float(np.mean(confs)) if confs else 0.0
    if name and name_conf < 25:
        notes.append("Name OCR confidence is low")

    return {
        "name": name,
        "school": school,
        "date_raw": date_raw,
        "date_normalized": date_normalized,
        "ocr_confidence": overall_conf,
    }, notes


def read_test_type(card):
    if not TESSERACT_AVAILABLE:
        return "UNKNOWN"
    table = _find_info_table(card)
    H, W = card.shape[:2]
    if table:
        x, y, w, h = table
        # Topic/header box is immediately above the Name/School/Date table.
        top = max(0, y - int(H * 0.14))
        crop = card[top:y, max(0, x):min(W, x+w)]
    else:
        crop = card[int(H*0.13):int(H*0.36), int(W*0.05):int(W*0.95)]

    variants = _variants(crop)
    for img, psm in ((variants[1], 6), (variants[2], 11)):
        text = _clean(pytesseract.image_to_string(img, config=f"--oem 3 --psm {psm}")).upper()
        if re.search(r"\bPOST\b|POST\s*TEST", text):
            return "POST"
        if re.search(r"\bPRE\b|PRE\s*TEST", text):
            return "PRE"
    return "UNKNOWN"
