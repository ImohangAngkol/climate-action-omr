import cv2
import numpy as np
from config import (
    ANSWER_ROI, HOUGH_DP, HOUGH_MIN_DIST, HOUGH_PARAM1, HOUGH_PARAM2,
    MIN_RADIUS, MAX_RADIUS, FILL_THRESHOLD, MIN_DARKNESS_GAP,
    QUESTIONS, OPTIONS,
)
from src.sheet_splitter import crop_normalized


def _cluster(values, groups, tolerance):
    """Cluster 1-D coordinates, then return group centers sorted ascending."""
    values = sorted(values)
    clusters = []
    for v in values:
        if not clusters or abs(v - np.mean(clusters[-1])) > tolerance:
            clusters.append([v])
        else:
            clusters[-1].append(v)
    centers = [float(np.mean(c)) for c in clusters]
    if len(centers) <= groups:
        return centers
    # Keep the densest groups if Hough finds noise.
    ranked = sorted(zip(clusters, centers), key=lambda x: len(x[0]), reverse=True)[:groups]
    return sorted(c for _, c in ranked)


def _darkness(gray, x, y, r):
    """Mean darkness inside the inner bubble, excluding most of the ring."""
    rr = max(2, int(r * 0.55))
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.circle(mask, (int(x), int(y)), rr, 255, -1)
    pixels = gray[mask == 255]
    if pixels.size == 0:
        return 0.0
    return float(1.0 - pixels.mean() / 255.0)


def read_answers(card, debug_path=None):
    roi = crop_normalized(card, ANSWER_ROI)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT,
        dp=HOUGH_DP,
        minDist=HOUGH_MIN_DIST,
        param1=HOUGH_PARAM1,
        param2=HOUGH_PARAM2,
        minRadius=MIN_RADIUS,
        maxRadius=MAX_RADIUS,
    )

    debug = roi.copy()
    if circles is None:
        if debug_path:
            cv2.imwrite(str(debug_path), debug)
        return {}, True, "No bubbles detected"

    circles = np.round(circles[0]).astype(int)

    # Expected arrangement: 5 question rows total, 4 answer columns.
    # The printed form has Q1/Q2 on the left and Q3/Q4/Q5 on the right,
    # so we identify rows separately inside left/right halves.
    h, w = gray.shape
    left = [c for c in circles if c[0] < w * 0.48]
    right = [c for c in circles if c[0] >= w * 0.48]

    def parse_block(block, q_numbers):
        if not block:
            return {}, True
        xs = _cluster([c[0] for c in block], 4, tolerance=max(8, w*0.04))
        ys = _cluster([c[1] for c in block], len(q_numbers), tolerance=max(8, h*0.08))
        if len(xs) < 4 or len(ys) < len(q_numbers):
            return {}, True

        xs = xs[:4]
        ys = ys[:len(q_numbers)]
        out = {}
        ambiguous = False

        for q, y0 in zip(q_numbers, ys):
            scores = []
            matched = []
            for x0 in xs:
                nearest = min(block, key=lambda c: (c[0]-x0)**2 + (c[1]-y0)**2)
                s = _darkness(gray, nearest[0], nearest[1], nearest[2])
                scores.append(s)
                matched.append(nearest)

            order = np.argsort(scores)[::-1]
            best_i = int(order[0])
            best = scores[best_i]
            second = scores[int(order[1])]

            if best >= FILL_THRESHOLD and (best - second) >= MIN_DARKNESS_GAP:
                out[q] = OPTIONS[best_i]
            else:
                out[q] = ""
                ambiguous = True

            for i, c in enumerate(matched):
                cv2.circle(debug, (c[0], c[1]), c[2], (0, 255, 0) if i == best_i else (255, 0, 0), 1)
                cv2.putText(debug, f"{scores[i]:.2f}", (c[0]-10, c[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0,0,255), 1)

        return out, ambiguous

    a1, bad1 = parse_block(left, [1, 2])
    a2, bad2 = parse_block(right, [3, 4, 5])
    answers = {**a1, **a2}
    needs_review = bad1 or bad2 or len(answers) != QUESTIONS or any(not answers.get(q) for q in range(1, QUESTIONS+1))
    note = "Review one or more blank/ambiguous answers" if needs_review else ""

    if debug_path:
        cv2.imwrite(str(debug_path), debug)
    return answers, needs_review, note
