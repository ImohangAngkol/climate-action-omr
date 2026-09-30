# OMR Scanner Starter

A starter Python project for your 6-up PRE/POST test sheets.

It is designed to:
1. accept a scanned PDF or image;
2. split each page into 6 response cards (3 rows x 2 columns);
3. OCR the Name and School fields;
4. detect the filled A/B/C/D bubble for Questions 1-5;
5. append one CSV row per student/card;
6. mark uncertain rows with `needs_review=YES` instead of silently guessing.

## Project structure

```text
omr_starter/
├── main.py
├── config.py
├── requirements.txt
├── README.md
├── data/
│   └── results.csv          # created automatically
├── debug/                   # optional debug crops
└── src/
    ├── input_loader.py
    ├── sheet_splitter.py
    ├── name_reader.py
    ├── bubble_reader.py
    ├── csv_writer.py
    └── models.py
```

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

For OCR, install Tesseract itself too:
- Windows: install Tesseract OCR and add it to PATH.
- Ubuntu/Debian: `sudo apt install tesseract-ocr`
- macOS: `brew install tesseract`

## Run

Start without OCR while tuning bubble detection:

```bash
python main.py ClimateAction_PRETEST_SHEET.pdf --csv data/results.csv --no-ocr --debug
```

Then enable OCR:

```bash
python main.py scanned_pretest.pdf scanned_posttest.pdf --csv data/results.csv --debug
```

## CSV columns

```text
source_file,card_index,test_type,name,school,q1,q2,q3,q4,q5,needs_review,note
```

Example:

```text
scan_001.jpg,1,PRE,Juan Dela Cruz,Example School,B,A,D,C,B,NO,
```

## Important MVP limitation

The current splitter assumes the full scanned page is already reasonably straight and that the 6 cards remain in a 3x2 grid. That matches the supplied template. The next reliability upgrade should use the printed black square registration marks to perspective-correct each card before OCR/OMR.

For handwritten names, OCR should be treated as a draft value. Keep `needs_review` and later add a review screen or a second verification step.
