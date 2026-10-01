from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

IMAGES_DIR = BASE_DIR / "images"
UNSCANNED_DIR = IMAGES_DIR / "unscanned"
DONE_DIR = IMAGES_DIR / "done"
REVIEW_DIR = IMAGES_DIR / "review"
FAILED_DIR = IMAGES_DIR / "failed"

DATA_DIR = BASE_DIR / "data"
CSV_PATH = DATA_DIR / "results.csv"
REVIEW_CSV_PATH = DATA_DIR / "review_results.csv"
PROCESSED_DB = DATA_DIR / ".processed.json"

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

GRID_ROWS = 2
GRID_COLS = 2

# Bubble detection
MIN_FILLED_RATIO = 0.18
MIN_WINNER_MARGIN = 0.08

# Image/card quality checks
MIN_SOURCE_SHARPNESS = 55.0
MIN_CARD_SHARPNESS = 45.0

MIN_ANSWERS_FOR_USABLE_CARD = 3
MAX_EMPTY_CARDS_BEFORE_IMAGE_REVIEW_ONLY = 3

# Date validation
MIN_DATE_YEAR = 2020
MAX_DATE_YEAR = 2035

# Leave None if "tesseract --version" works.
TESSERACT_CMD = None

KNOWN_SCHOOLS = [
    "Xavier University",
    "Capitol University",
    "Liceo de Cagayan University",
    "Lourdes College",
]