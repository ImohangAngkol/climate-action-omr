from dataclasses import dataclass, field
from typing import List


@dataclass
class StudentResult:
    source_file: str
    card_index: int
    test_type: str = "UNKNOWN"
    name: str = ""
    school: str = ""
    date_raw: str = ""
    date_normalized: str = ""
    q1: str = ""
    q2: str = ""
    q3: str = ""
    q4: str = ""
    q5: str = ""
    ocr_confidence: float = 0.0
    card_sharpness: float = 0.0
    status: str = "OK"
    notes: List[str] = field(default_factory=list)

    def mark_review(self, reason: str) -> None:
        self.status = "REVIEW"
        if reason and reason not in self.notes:
            self.notes.append(reason)

    def answers(self):
        return [self.q1, self.q2, self.q3, self.q4, self.q5]

    def answer_count(self):
        return sum(bool(x) for x in self.answers())
