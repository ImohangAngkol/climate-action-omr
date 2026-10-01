import hashlib
import json
import shutil
import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from config import (
    UNSCANNED_DIR,
    DONE_DIR,
    REVIEW_DIR,
    FAILED_DIR,
    PROCESSED_DB,
    SUPPORTED_EXTENSIONS,
    MIN_SOURCE_SHARPNESS,
    MIN_CARD_SHARPNESS,
    MIN_ANSWERS_FOR_USABLE_CARD,
    MAX_EMPTY_CARDS_BEFORE_IMAGE_REVIEW_ONLY,
)
from .input_loader import load_image, sharpness_score
from .sheet_detector import split_cards
from .bubble_reader import read_answers
from .field_reader import read_fields, read_test_type
from .models import StudentResult
from .csv_writer import append_results


def _ensure_dirs():
    for path in [UNSCANNED_DIR, DONE_DIR, REVIEW_DIR, FAILED_DIR, PROCESSED_DB.parent]:
        path.mkdir(parents=True, exist_ok=True)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_db():
    if not PROCESSED_DB.exists():
        return {}
    try:
        return json.loads(PROCESSED_DB.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_db(db):
    temp = PROCESSED_DB.with_suffix(".tmp")
    temp.write_text(json.dumps(db, indent=2), encoding="utf-8")
    temp.replace(PROCESSED_DB)


def _unique_destination(folder, name):
    target = folder / name
    if not target.exists():
        return target
    stem, suffix = target.stem, target.suffix
    counter = 2
    while True:
        candidate = folder / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def _wait_until_stable(path, checks=3, delay=0.7):
    last = -1
    stable = 0
    for _ in range(20):
        if not path.exists():
            return False
        size = path.stat().st_size
        if size == last and size > 0:
            stable += 1
            if stable >= checks:
                return True
        else:
            stable = 0
        last = size
        time.sleep(delay)
    return False


def process_image(path):
    path = Path(path)
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS or not path.exists():
        return

    print(f"\n[SCAN] {path.name}")
    if not _wait_until_stable(path):
        print("[FAIL] File never became stable")
        shutil.move(str(path), str(_unique_destination(FAILED_DIR, path.name)))
        return

    file_hash = _sha256(path)
    db = _load_db()
    if file_hash in db:
        print(f"[SKIP] Duplicate of {db[file_hash].get('source_file', 'previous image')}")
        shutil.move(str(path), str(_unique_destination(DONE_DIR, path.name)))
        return

    try:
        image = load_image(path)
        overall_sharpness = sharpness_score(image)
        cards = split_cards(image)
        results = []

        for card_index, card in cards:
            result = StudentResult(source_file=path.name, card_index=card_index)
            result.test_type = read_test_type(card)

            fields, field_notes = read_fields(card)
            result.name = fields["name"]
            result.school = fields["school"]
            result.date_raw = fields["date_raw"]
            result.date_normalized = fields["date_normalized"]

            answers, bubble_notes = read_answers(card)
            result.q1, result.q2, result.q3, result.q4, result.q5 = answers

            for note in field_notes + bubble_notes:
                result.mark_review(note)

            if result.test_type == "UNKNOWN":
                result.mark_review("PRE/POST type could not be read")
            if any(not answer for answer in result.answers()):
                result.mark_review("One or more answers are blank/uncertain")

            results.append(result)
            print(f"  card {card_index}: {result.name or '[name?]'} | "
                  f"{','.join(result.answers())} | {result.status}")

        # Critical ordering: write CSV first, then record hash, then move source.
        append_results(results)
        db[file_hash] = {"source_file": path.name, "rows": len(results), "time": time.time()}
        _save_db(db)

        needs_review = any(r.status != "OK" for r in results)
        destination_folder = REVIEW_DIR if needs_review else DONE_DIR
        destination = _unique_destination(destination_folder, path.name)
        shutil.move(str(path), str(destination))
        print(f"[DONE] CSV saved -> moved to {destination_folder.name}/")

    except Exception as exc:
        print(f"[ERROR] {path.name}: {exc}")
        if path.exists():
            shutil.move(str(path), str(_unique_destination(FAILED_DIR, path.name)))


def process_existing():
    _ensure_dirs()
    files = sorted(p for p in UNSCANNED_DIR.iterdir()
                   if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)
    if not files:
        print("[INFO] No images waiting in images/unscanned/")
    for path in files:
        process_image(path)


class NewImageHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:
            process_image(Path(event.src_path))

    def on_moved(self, event):
        if not event.is_directory:
            process_image(Path(event.dest_path))


def watch_forever():
    _ensure_dirs()
    process_existing()
    observer = Observer()
    observer.schedule(NewImageHandler(), str(UNSCANNED_DIR), recursive=False)
    observer.start()
    print(f"\n[WATCHING] {UNSCANNED_DIR}")
    print("Drop JPG/PNG files there. Press Ctrl+C to stop.\n")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[STOP] Watcher stopped.")
        observer.stop()
    observer.join()
