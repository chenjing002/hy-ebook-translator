"""Resumable per-document state.

Translations are cached by a hash of the *segment content*, not by position, so
editing one paragraph of a long book re-translates only that paragraph, and
reordering chapters costs nothing. State is rewritten atomically after every
batch, which makes Ctrl-C safe at any point.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path


def segment_key(kind: str, text: str) -> str:
    raw = f"{kind}\x00{text}".encode()
    return hashlib.sha1(raw).hexdigest()[:20]


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def slugify(name: str) -> str:
    slug = re.sub(r"[^\w一-鿿.-]+", "-", name.strip()).strip("-.")
    return slug or "document"


@dataclass
class DocState:
    path: Path
    source_sha: str = ""
    fingerprint: str = ""
    segments: dict[str, str] = field(default_factory=dict)
    warnings: list[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @classmethod
    def load(cls, state_dir: Path, slug: str, source_sha: str, fingerprint: str) -> "DocState":
        path = state_dir / f"{slug}.json"
        st = cls(path=path, source_sha=source_sha, fingerprint=fingerprint)
        if not path.exists():
            return st
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return st
        # A changed API or translation setting invalidates cached text; a
        # changed source does not, because keys are content-addressed.
        if data.get("fingerprint") != fingerprint:
            return st
        st.segments = {k: v for k, v in (data.get("segments") or {}).items() if isinstance(v, str)}
        st.warnings = data.get("warnings") or []
        st.stats = data.get("stats") or {}
        return st

    def get(self, kind: str, text: str) -> str | None:
        return self.segments.get(segment_key(kind, text))

    def put(self, kind: str, text: str, zh: str) -> None:
        self.segments[segment_key(kind, text)] = zh

    def prune(self, live_keys: set[str]) -> int:
        dead = set(self.segments) - live_keys
        for key in dead:
            del self.segments[key]
        return len(dead)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "source_sha": self.source_sha,
            "fingerprint": self.fingerprint,
            "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "stats": self.stats,
            "warnings": self.warnings[-200:],
            "segments": self.segments,
        }
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=0)
            os.replace(tmp, self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
