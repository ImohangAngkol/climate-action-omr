from pathlib import Path

# The supplied sheet has 6 response cards arranged as 3 rows x 2 columns.
GRID_ROWS = 3
GRID_COLS = 2

# Fractions trimmed from each card edge before circle detection.
# These are intentionally broad starter values and are easy to tune later.
ANSWER_ROI = (0.08, 0.48, 0.90, 0.88)  # x1, y1, x2, y2
NAME_ROI = (0.09, 0.17, 0.88, 0.28)
SCHOOL_ROI = (0.09, 0.28, 0.88, 0.39)

# Bubble detection / fill scoring.
HOUGH_DP = 1.2
HOUGH_MIN_DIST = 14
HOUGH_PARAM1 = 80
HOUGH_PARAM2 = 22
MIN_RADIUS = 6
MAX_RADIUS = 18
FILL_THRESHOLD = 0.34
MIN_DARKNESS_GAP = 0.06

QUESTIONS = 5
OPTIONS = ["A", "B", "C", "D"]

DEBUG_DIR = Path("debug")
