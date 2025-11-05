from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Tuple


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    width: int
    height: int

    @property
    def xy(self) -> Tuple[int, int]:
        return self.x, self.y

    @property
    def wh(self) -> Tuple[int, int]:
        return self.width, self.height

    @property
    def center(self) -> Tuple[float, float]:
        return self.x + self.width / 2.0, self.y + self.height / 2.0

    def clamp(self, max_width: int, max_height: int) -> "Rect":
        x = max(0, min(self.x, max_width))
        y = max(0, min(self.y, max_height))
        w = max(0, min(self.width, max_width - x))
        h = max(0, min(self.height, max_height - y))
        return Rect(x, y, w, h)

    def as_slice(self) -> Tuple[slice, slice]:
        return (slice(self.y, self.y + self.height), slice(self.x, self.x + self.width))


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def rolling_mean(values: Iterable[float], window: int) -> float:
    window_values = list(values)[-window:]
    if not window_values:
        return 0.0
    return sum(window_values) / len(window_values)
