import argparse
from src.watcher import process_existing, watch_forever


def main():
    parser = argparse.ArgumentParser(description="DEVCON Kids OMR/OCR scanner")
    parser.add_argument("--watch", action="store_true",
                        help="Keep watching images/unscanned for new images")
    parser.add_argument("--once", action="store_true",
                        help="Process everything currently in images/unscanned, then exit")
    args = parser.parse_args()

    if args.watch:
        watch_forever()
    else:
        process_existing()


if __name__ == "__main__":
    main()
