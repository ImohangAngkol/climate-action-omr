import argparse
from pathlib import Path
import cv2

from config import DEBUG_DIR
from src.input_loader import load_pages
from src.sheet_splitter import normalize_page, split_into_cards
from src.name_reader import read_name_and_school
from src.bubble_reader import read_answers
from src.csv_writer import append_rows
from src.models import CardResult


def infer_test_type(filename: str):
    name = filename.lower()
    if "pre" in name:
        return "PRE"
    if "post" in name:
        return "POST"
    return ""


def process_file(input_path: str, csv_path: str, use_ocr: bool = True, save_debug: bool = False):
    source = Path(input_path).name
    test_type = infer_test_type(source)
    pages = load_pages(input_path)
    output_rows = []

    if save_debug:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    card_counter = 0
    for page_num, page in enumerate(pages, start=1):
        page, _ = normalize_page(page)
        cards = split_into_cards(page)

        for local_index, card in enumerate(cards, start=1):
            card_counter += 1
            name, school = read_name_and_school(card, enabled=use_ocr)

            debug_path = None
            if save_debug:
                debug_path = DEBUG_DIR / f"{Path(source).stem}_p{page_num}_card{local_index}_bubbles.jpg"
                cv2.imwrite(str(DEBUG_DIR / f"{Path(source).stem}_p{page_num}_card{local_index}.jpg"), card)

            answers, needs_review, note = read_answers(card, debug_path=debug_path)

            # Handwritten OCR is inherently less reliable than bubble OMR.
            if use_ocr and not name:
                needs_review = True
                note = (note + "; " if note else "") + "Name OCR empty/uncertain"

            result = CardResult(
                source_file=source,
                card_index=card_counter,
                test_type=test_type,
                name=name,
                school=school,
                answers=answers,
                needs_review=needs_review,
                note=note,
            )
            output_rows.append(result.as_row())

    append_rows(csv_path, output_rows)
    return output_rows


def main():
    parser = argparse.ArgumentParser(description="Read 6-up pre/post test sheets and append results to CSV.")
    parser.add_argument("inputs", nargs="+", help="PDF/image files to scan")
    parser.add_argument("--csv", default="data/results.csv", help="Output CSV path")
    parser.add_argument("--no-ocr", action="store_true", help="Skip name/school OCR")
    parser.add_argument("--debug", action="store_true", help="Save card and bubble debug images")
    args = parser.parse_args()

    total = 0
    for input_path in args.inputs:
        rows = process_file(input_path, args.csv, use_ocr=not args.no_ocr, save_debug=args.debug)
        total += len(rows)
        print(f"Processed {input_path}: {len(rows)} cards")

    print(f"Done. {total} rows appended to {args.csv}")


if __name__ == "__main__":
    main()
