import cv2
import numpy as np
from config import MIN_FILLED_RATIO, MIN_WINNER_MARGIN

LETTERS = "ABCD"


def _cluster(values, tolerance):
    values = sorted(values)
    groups = []
    for value in values:
        if not groups or abs(value - np.mean(groups[-1])) > tolerance:
            groups.append([value])
        else:
            groups[-1].append(value)
    return [float(np.mean(g)) for g in groups]


def _find_20_circles(card):
    gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
    scale = 1200.0 / max(card.shape[:2])
    if scale < 1.0:
        work = cv2.resize(gray, None, fx=scale, fy=scale)
    else:
        work = gray
        scale = 1.0

    work = cv2.GaussianBlur(work, (5, 5), 1.1)
    h, w = work.shape

    # The answer area is in the lower ~half of each card.
    roi_y = int(h * 0.48)
    roi = work[roi_y:, :]

    min_r = max(7, int(min(h, w) * 0.014))
    max_r = max(min_r + 4, int(min(h, w) * 0.035))
    min_dist = max(18, int(min(h, w) * 0.035))

    circles = cv2.HoughCircles(
        roi,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_dist,
        param1=100,
        param2=24,
        minRadius=min_r,
        maxRadius=max_r,
    )
    if circles is None:
        return []

    raw = []
    for x, y, r in np.round(circles[0]).astype(int):
        y += roi_y
        raw.append((x / scale, y / scale, r / scale))

    # Keep candidates in the expected answer band and ignore registration marks.
    H, W = card.shape[:2]
    raw = [c for c in raw if H * 0.52 < c[1] < H * 0.91 and W * 0.04 < c[0] < W * 0.96]

    # We expect exactly 8 answer columns x 3 possible rows, but only 20 circles
    # exist because questions 4/5 have no third row.
    if len(raw) < 20:
        return []

    # If Hough finds extras, favor circles near the dominant radius.
    median_r = np.median([c[2] for c in raw])
    raw.sort(key=lambda c: abs(c[2] - median_r))
    return raw[:20]


def _dark_ratio(gray, x, y, r):
    # Measure the inside of the circle, not the printed outline.
    rr = max(3, int(r * 0.58))
    x, y = int(round(x)), int(round(y))
    y1, y2 = max(0, y-rr), min(gray.shape[0], y+rr+1)
    x1, x2 = max(0, x-rr), min(gray.shape[1], x+rr+1)
    patch = gray[y1:y2, x1:x2]
    yy, xx = np.ogrid[:patch.shape[0], :patch.shape[1]]
    cy, cx = y-y1, x-x1
    mask = (xx-cx)**2 + (yy-cy)**2 <= rr**2
    pixels = patch[mask]
    if pixels.size == 0:
        return 0.0
    # Adaptive-ish darkness: filled pen/pencil is much darker than paper.
    return float(np.mean(pixels < 150))


def read_answers(card):
    circles = _find_20_circles(card)
    if len(circles) != 20:
        return ["", "", "", "", ""], ["Could not reliably locate all 20 answer bubbles"]

    H, W = card.shape[:2]
    xs = _cluster([c[0] for c in circles], tolerance=W * 0.035)
    ys = _cluster([c[1] for c in circles], tolerance=H * 0.035)

    if len(xs) != 8 or len(ys) != 3:
        return ["", "", "", "", ""], [f"Bubble grid ambiguous (x={len(xs)}, y={len(ys)})"]

    xs = sorted(xs)
    ys = sorted(ys)
    gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
    median_r = float(np.median([c[2] for c in circles]))

    questions = [
        (xs[:4], ys[0]),
        (xs[:4], ys[1]),
        (xs[:4], ys[2]),
        (xs[4:], ys[0]),
        (xs[4:], ys[1]),
    ]

    answers = []
    notes = []
    for q_num, (qx, qy) in enumerate(questions, start=1):
        scores = [_dark_ratio(gray, x, qy, median_r) for x in qx]
        order = np.argsort(scores)[::-1]
        best_i, second_i = int(order[0]), int(order[1])
        best, second = scores[best_i], scores[second_i]

        if best < MIN_FILLED_RATIO:
            answers.append("")
            notes.append(f"Q{q_num}: no strong filled bubble")
        elif best - second < MIN_WINNER_MARGIN:
            answers.append("")
            notes.append(f"Q{q_num}: possible double/uncertain mark")
        else:
            answers.append(LETTERS[best_i])

    return answers, notes
