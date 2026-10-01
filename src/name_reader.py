# Kept as a compatibility wrapper because your project already has name_reader.py.
# Actual Name/School/Date OCR now lives together in field_reader.py.
from .field_reader import read_fields


def read_name(card):
    fields, notes = read_fields(card)
    return fields["name"], notes
