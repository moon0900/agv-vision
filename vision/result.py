from dataclasses import dataclass
from typing import Tuple

@dataclass
class RawOCR:
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)

@dataclass
class PlateResult:
    text: str
    similarity: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)