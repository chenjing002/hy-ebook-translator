"""Minimal dependency-free progress reporting."""

from __future__ import annotations

import shutil
import sys
import time


class Progress:
    def __init__(self, total: int, label: str, enabled: bool = True) -> None:
        self.total = max(total, 0)
        self.label = label
        self.done = 0
        self.enabled = enabled and sys.stderr.isatty()
        self.start = time.monotonic()
        self._last = 0.0
        self.note = ""

    def advance(self, n: int = 1, note: str = "") -> None:
        self.done += n
        if note:
            self.note = note
        now = time.monotonic()
        if self.enabled and (now - self._last > 0.1 or self.done >= self.total):
            self._last = now
            self._draw()

    def _draw(self) -> None:
        width = min(shutil.get_terminal_size((90, 20)).columns, 110)
        frac = self.done / self.total if self.total else 1.0
        elapsed = time.monotonic() - self.start
        rate = self.done / elapsed if elapsed > 0 else 0
        eta = (self.total - self.done) / rate if rate > 0 else 0
        head = f"{self.label} "
        tail = f" {self.done}/{self.total} {_hms(elapsed)}<{_hms(eta)}"
        if self.note:
            tail += f" {self.note}"
        barw = max(10, width - len(head) - len(tail) - 8)
        filled = int(barw * frac)
        bar = "█" * filled + "·" * (barw - filled)
        sys.stderr.write(f"\r{head}{bar} {frac * 100:3.0f}%{tail}\x1b[K")
        sys.stderr.flush()

    def close(self, summary: str = "") -> None:
        if self.enabled:
            sys.stderr.write("\r\x1b[K")
            sys.stderr.flush()
        if summary:
            print(summary)


def _hms(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    if seconds >= 3600:
        return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"
    return f"{seconds // 60:02d}:{seconds % 60:02d}"
