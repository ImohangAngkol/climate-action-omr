import csv
from dataclasses import asdict
from pathlib import Path
from config import CSV_PATH, REVIEW_CSV_PATH

HEADERS = [
    "source_file", "card_index", "test_type", "name", "school",
    "date_raw", "date_normalized", "q1", "q2", "q3", "q4", "q5",
    "ocr_confidence", "card_sharpness", "status", "notes"
]


def _append(results, csv_path):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists() and csv_path.stat().st_size > 0
    with csv_path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        if not exists:
            writer.writeheader()
        for result in results:
            row = asdict(result)
            row["notes"] = " | ".join(row["notes"])
            row["ocr_confidence"] = f"{row['ocr_confidence']:.1f}"
            row["card_sharpness"] = f"{row['card_sharpness']:.1f}"
            writer.writerow({key: row.get(key, "") for key in HEADERS})
        f.flush()


def append_results(results, csv_path=CSV_PATH):
    _append(results, csv_path)


def append_review_results(results, csv_path=REVIEW_CSV_PATH):
    _append(results, csv_path)
