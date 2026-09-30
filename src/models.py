from dataclasses import dataclass, field
from typing import Dict, Optional

@dataclass
class CardResult:
    source_file: str
    card_index: int
    test_type: str = ""
    name: str = ""
    school: str = ""
    answers: Dict[int, str] = field(default_factory=dict)
    needs_review: bool = False
    note: str = ""

    def as_row(self) -> dict:
        row = {
            "source_file": self.source_file,
            "card_index": self.card_index,
            "test_type": self.test_type,
            "name": self.name,
            "school": self.school,
        }
        for q in range(1, 6):
            row[f"q{q}"] = self.answers.get(q, "")
        row["needs_review"] = "YES" if self.needs_review else "NO"
        row["note"] = self.note
        return row
