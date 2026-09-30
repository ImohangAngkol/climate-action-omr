import csv
from pathlib import Path

FIELDS = [
    "source_file", "card_index", "test_type", "name", "school",
    "q1", "q2", "q3", "q4", "q5", "needs_review", "note"
]


def append_rows(csv_path, rows):
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0

    with path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
