"""Markdown block parser.

The pipeline never hands raw Markdown to the model. It first splits a document
into typed blocks, so that code fences, display math, non-table HTML and
reference definitions can be passed through byte-for-byte while prose and
table cells are translated. Every block can rebuild itself from translated
segments, which keeps tables, nested lists and blockquotes structurally intact.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Iterator

# Blocks whose text is translated.
PROSE_KINDS = {"paragraph", "heading", "list", "table", "html_table", "blockquote", "footnote"}
# Blocks emitted verbatim, exactly once, with no translation.
PASSTHROUGH_KINDS = {"code", "math", "html", "hr", "front_matter", "linkdef"}

FENCE_RE = re.compile(r"^(?P<indent> {0,3})(?P<fence>`{3,}|~{3,})(?P<info>.*)$")
ATX_RE = re.compile(r"^ {0,3}(?P<hashes>#{1,6})(?:[ \t]+(?P<text>.*?))?[ \t]*(?<!\\)#*[ \t]*$")
SETEXT_RE = re.compile(r"^ {0,3}(=+|-+)[ \t]*$")
HR_RE = re.compile(r"^ {0,3}((?:\*[ \t]*){3,}|(?:-[ \t]*){3,}|(?:_[ \t]*){3,})$")
LIST_RE = re.compile(r"^(?P<indent>[ \t]*)(?P<marker>[-*+]|\d{1,9}[.)])(?P<space>[ \t]+|$)")
QUOTE_RE = re.compile(r"^ {0,3}>[ \t]?")
TABLE_DELIM_RE = re.compile(r"^ {0,3}\|?[ \t]*:?-{1,}:?[ \t]*(\|[ \t]*:?-{1,}:?[ \t]*)*\|?[ \t]*$")
MATH_OPEN_RE = re.compile(r"^ {0,3}(\$\$|\\\[)")
MATH_ENV_RE = re.compile(r"^ {0,3}\\begin\{(?P<env>[A-Za-z*]+)\}")
FOOTNOTE_RE = re.compile(r"^ {0,3}\[\^(?P<id>[^\]]+)\]:[ \t]*(?P<text>.*)$")
LINKDEF_RE = re.compile(r"^ {0,3}\[(?!\^)(?P<label>[^\]]+)\]:[ \t]*\S+.*$")
HTML_OPEN_RE = re.compile(
    r"^ {0,3}(?:<!--|<(?:/?)(?:address|article|aside|blockquote|details|dialog|div|dl|dt|dd"
    r"|fieldset|figcaption|figure|footer|form|h[1-6]|header|hgroup|hr|iframe|img|li|main|nav"
    r"|ol|p|picture|pre|script|section|style|summary|table|tbody|td|tfoot|th|thead|tr|ul|video"
    r"|center|font|sub|sup|kbd|br)\b)",
    re.IGNORECASE,
)
# Elements that never have a closing tag, so they must not open a block.
VOID_TAGS = {
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta",
    "param", "source", "track", "wbr",
}
HTML_TAG_RE = re.compile(r"<(?P<close>/?)(?P<name>[A-Za-z][A-Za-z0-9-]*)(?P<attrs>[^<>]*)>")
HTML_TOKEN_RE = re.compile(
    r"<!--.*?-->|</?[A-Za-z](?:\"[^\"]*\"|'[^']*'|[^'\">])*>|<![^>]*>", re.DOTALL
)
HTML_CELL_RE = re.compile(r"<\s*(/?)\s*(td|th|caption)\b", re.IGNORECASE)

MATH_ENVS = {
    "equation", "equation*", "align", "align*", "aligned", "gather", "gather*",
    "multline", "multline*", "eqnarray", "eqnarray*", "split", "cases", "array",
    "matrix", "pmatrix", "bmatrix", "vmatrix", "Bmatrix", "Vmatrix", "smallmatrix",
    "alignat", "alignat*", "flalign", "flalign*", "subequations",
}
CJK_RE = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
LETTER_RE = re.compile(r"[A-Za-zÀ-ɏ]")


@dataclass
class Unit:
    """One translatable span of prose."""

    uid: str
    kind: str
    text: str

    def translatable(self) -> bool:
        return bool(LETTER_RE.search(self.text) or CJK_RE.search(self.text))


@dataclass
class Block:
    index: int
    kind: str
    raw: str
    units: list[Unit] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    # ---- segments -----------------------------------------------------------
    def iter_units(self) -> Iterator[Unit]:
        yield from self.units
        for inner in self.meta.get("inner", ()):
            yield from inner.iter_units()

    # ---- rebuild ------------------------------------------------------------
    def render_target(self, zh: Callable[[str], str]) -> str:
        """Rebuild this block using translated text for each of its segments."""
        builder = _RENDERERS.get(self.kind)
        if builder is None:
            return self.raw
        return builder(self, zh)


# ─── per-kind rebuilders ─────────────────────────────────────────────────────


def _render_paragraph(block: Block, zh: Callable[[str], str]) -> str:
    return zh(block.units[0].uid) if block.units else block.raw


def _render_heading(block: Block, zh: Callable[[str], str]) -> str:
    text = zh(block.units[0].uid) if block.units else block.meta.get("text", "")
    if block.meta.get("style") == "setext":
        # Normalise setext headings to ATX so the bilingual pair stays compact.
        level = block.meta["level"]
        return f"{'#' * level} {text}".rstrip()
    hashes = "#" * block.meta["level"]
    return f"{hashes} {text}".rstrip()


def _render_footnote(block: Block, zh: Callable[[str], str]) -> str:
    text = zh(block.units[0].uid) if block.units else block.meta.get("text", "")
    return f"[^{block.meta['id']}]: {text}"


def _render_list(block: Block, zh: Callable[[str], str]) -> str:
    out: list[str] = []
    for seg in block.meta["segments"]:
        out.extend([""] * seg.get("blank_before", 0))
        if seg["type"] == "raw":
            out.extend(seg["lines"])
        elif seg["type"] == "item":
            body = zh(seg["uid"]) if seg.get("uid") else ""
            prefix = f"{seg['indent']}{seg['marker']} "
            out.append((prefix + body).rstrip() if body else prefix.rstrip())
        else:  # continuation paragraph inside an item
            out.append((seg["indent"] + zh(seg["uid"])).rstrip())
    return "\n".join(out)


def _render_table(block: Block, zh: Callable[[str], str]) -> str:
    lines: list[str] = []
    for row in block.meta["rows"]:
        if row["type"] == "delim":
            lines.append(row["raw"])
            continue
        cells = [zh(uid) if uid else text for uid, text in row["cells"]]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _render_html_table(block: Block, zh: Callable[[str], str]) -> str:
    return "".join(zh(value) if kind == "unit" else value
                   for kind, value in block.meta["parts"])


def _render_blockquote(block: Block, zh: Callable[[str], str]) -> str:
    rendered = render_blocks(block.meta["inner"], zh)
    out = []
    for line in rendered.split("\n"):
        out.append(">" if not line else f"> {line}")
    return "\n".join(out)


_RENDERERS: dict[str, Callable[[Block, Callable[[str], str]], str]] = {
    "paragraph": _render_paragraph,
    "heading": _render_heading,
    "footnote": _render_footnote,
    "list": _render_list,
    "table": _render_table,
    "html_table": _render_html_table,
    "blockquote": _render_blockquote,
}


def render_blocks(blocks: list[Block], zh: Callable[[str], str]) -> str:
    return "\n\n".join(b.render_target(zh) for b in blocks)


# ─── parser ──────────────────────────────────────────────────────────────────


class _Parser:
    def __init__(self, text: str, prefix: str = "b") -> None:
        self.lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        self.i = 0
        self.prefix = prefix
        self.blocks: list[Block] = []
        self._uid = 0

    # -- helpers
    def uid(self) -> str:
        self._uid += 1
        return f"{self.prefix}{self._uid}"

    def peek(self, offset: int = 0) -> str | None:
        j = self.i + offset
        return self.lines[j] if j < len(self.lines) else None

    def emit(self, kind: str, raw: str, units: list[Unit] | None = None, **meta) -> Block:
        block = Block(index=len(self.blocks), kind=kind, raw=raw, units=units or [], meta=meta)
        self.blocks.append(block)
        return block

    # -- entry
    def parse(self) -> list[Block]:
        if self.peek() is not None and self.peek().strip() in {"---", "+++"}:
            self._front_matter()
        while self.i < len(self.lines):
            line = self.lines[self.i]
            if not line.strip():
                self.i += 1
                continue
            for handler in (
                self._fence, self._math, self._hr, self._heading, self._html,
                self._footnote, self._linkdef, self._table, self._quote, self._list,
            ):
                if handler():
                    break
            else:
                self._paragraph()
        return self.blocks

    # -- block handlers (each returns True when it consumed something)
    def _front_matter(self) -> None:
        delim = self.lines[self.i].strip()
        start = self.i
        j = self.i + 1
        while j < len(self.lines) and self.lines[j].strip() != delim:
            j += 1
        if j >= len(self.lines):
            return  # unterminated: treat as ordinary content
        self.i = j + 1
        self.emit("front_matter", "\n".join(self.lines[start:self.i]))

    def _fence(self) -> bool:
        m = FENCE_RE.match(self.lines[self.i])
        if not m:
            return False
        fence = m.group("fence")
        char, length = fence[0], len(fence)
        start = self.i
        j = self.i + 1
        while j < len(self.lines):
            close = FENCE_RE.match(self.lines[j])
            if close and close.group("fence")[0] == char and len(close.group("fence")) >= length \
                    and not close.group("info").strip():
                j += 1
                break
            j += 1
        self.i = j
        self.emit("code", "\n".join(self.lines[start:self.i]), info=m.group("info").strip())
        return True

    def _math(self) -> bool:
        line = self.lines[self.i]
        env = MATH_ENV_RE.match(line)
        if env:
            if env.group("env") not in MATH_ENVS:
                return False
            closer = f"\\end{{{env.group('env')}}}"
        elif MATH_OPEN_RE.match(line):
            closer = "$$" if line.lstrip().startswith("$$") else "\\]"
            # A single-line `$$ x $$` closes on its own line.
            body = line.lstrip()[len(closer):] if closer == "$$" else line
            if closer == "$$" and body.rstrip().endswith("$$") and body.strip() != "":
                self.emit("math", line)
                self.i += 1
                return True
            if closer == "\\]" and "\\]" in line[line.index("\\[") + 2:]:
                self.emit("math", line)
                self.i += 1
                return True
        else:
            return False
        start = self.i
        j = self.i + 1
        while j < len(self.lines) and closer not in self.lines[j]:
            j += 1
        self.i = min(j + 1, len(self.lines))
        self.emit("math", "\n".join(self.lines[start:self.i]))
        return True

    def _hr(self) -> bool:
        if not HR_RE.match(self.lines[self.i]):
            return False
        self.emit("hr", self.lines[self.i])
        self.i += 1
        return True

    def _heading(self) -> bool:
        m = ATX_RE.match(self.lines[self.i])
        if not m:
            return False
        text = (m.group("text") or "").strip()
        uid = self.uid()
        self.emit(
            "heading", self.lines[self.i],
            units=[Unit(uid, "heading", text)] if text else [],
            level=len(m.group("hashes")), text=text, style="atx",
        )
        self.i += 1
        return True

    def _html(self) -> bool:
        line = self.lines[self.i]
        if not HTML_OPEN_RE.match(line):
            return False
        start = self.i
        if line.lstrip().startswith("<!--"):
            j = self.i
            while j < len(self.lines) and "-->" not in self.lines[j]:
                j += 1
            self.i = min(j + 1, len(self.lines))
        else:
            # CommonMark ends an HTML block at the next blank line, but
            # Markdown converted from PDF or EPUB routinely puts a whole
            # `<table>…</table>` on one line and the next paragraph of prose on
            # the line below it, with no blank line between. Closing the block
            # as soon as every element it opened is closed keeps that prose
            # translatable instead of swallowing the rest of the chapter.
            limit = self.i
            while limit < len(self.lines) and self.lines[limit].strip():
                limit += 1
            self.i = _html_end(self.lines, self.i, limit)
        raw = "\n".join(self.lines[start:self.i])
        if line.lstrip().lower().startswith("<table"):
            units, parts = _html_table_units(raw, self.uid)
            self.emit("html_table", raw, units=units, parts=parts)
        else:
            self.emit("html", raw)
        return True

    def _footnote(self) -> bool:
        m = FOOTNOTE_RE.match(self.lines[self.i])
        if not m:
            return False
        start = self.i
        text_lines = [m.group("text")]
        j = self.i + 1
        while j < len(self.lines) and self.lines[j].strip() and not FOOTNOTE_RE.match(self.lines[j]) \
                and not LIST_RE.match(self.lines[j]) and not ATX_RE.match(self.lines[j]):
            text_lines.append(self.lines[j].strip())
            j += 1
        self.i = j
        text = " ".join(t for t in text_lines if t).strip()
        uid = self.uid()
        self.emit(
            "footnote", "\n".join(self.lines[start:self.i]),
            units=[Unit(uid, "footnote", text)] if text else [],
            id=m.group("id"), text=text,
        )
        return True

    def _linkdef(self) -> bool:
        if not LINKDEF_RE.match(self.lines[self.i]):
            return False
        start = self.i
        while self.i < len(self.lines) and LINKDEF_RE.match(self.lines[self.i]):
            self.i += 1
        self.emit("linkdef", "\n".join(self.lines[start:self.i]))
        return True

    def _table(self) -> bool:
        line = self.lines[self.i]
        nxt = self.peek(1)
        if "|" not in line or not nxt or not TABLE_DELIM_RE.match(nxt) or "|" not in nxt:
            return False
        start = self.i
        j = self.i + 2
        while j < len(self.lines) and self.lines[j].strip() and "|" in self.lines[j]:
            j += 1
        self.i = j
        raw_lines = self.lines[start:self.i]

        rows: list[dict] = []
        for idx, rl in enumerate(raw_lines):
            if idx == 1:
                rows.append({"type": "delim", "raw": rl})
                continue
            cells = []
            for cell in _split_table_row(rl):
                stripped = cell.strip()
                unit = None
                if stripped and LETTER_RE.search(stripped):
                    unit = self.uid()
                    self.units_sink.append(Unit(unit, "table_cell", stripped))
                cells.append((unit, stripped))
            rows.append({"type": "row", "cells": cells})
        block = self.emit("table", "\n".join(raw_lines), rows=rows)
        block.units = [u for u in self.units_sink]
        self.units_sink = []
        return True

    units_sink: list[Unit] = []

    def _quote(self) -> bool:
        if not QUOTE_RE.match(self.lines[self.i]):
            return False
        start = self.i
        j = self.i
        while j < len(self.lines):
            line = self.lines[j]
            if QUOTE_RE.match(line):
                j += 1
                continue
            if line.strip() and not _starts_block(line):
                j += 1  # lazy continuation
                continue
            break
        self.i = j
        raw_lines = self.lines[start:self.i]
        inner_text = "\n".join(QUOTE_RE.sub("", rl, count=1) for rl in raw_lines)
        inner = _Parser(inner_text, prefix=f"{self.prefix}q{len(self.blocks)}_").parse()
        self.emit("blockquote", "\n".join(raw_lines), inner=inner)
        return True

    def _list(self) -> bool:
        m = LIST_RE.match(self.lines[self.i])
        if not m or not m.group("space"):
            return False
        base_indent = len(_expand(m.group("indent")))
        start = self.i
        j = self.i
        pending_blanks = 0
        while j < len(self.lines):
            line = self.lines[j]
            if not line.strip():
                pending_blanks += 1
                j += 1
                continue
            indent = len(_expand(line)) - len(_expand(line).lstrip())
            item = LIST_RE.match(line)
            if item and item.group("space"):
                if indent < base_indent:
                    break
                pending_blanks = 0
                j += 1
                continue
            if indent >= base_indent + 2:
                pending_blanks = 0
                j += 1
                continue
            if pending_blanks == 0 and not _starts_block(line):
                j += 1  # lazy continuation of the current item
                continue
            break
        # Trailing blank lines belong to whatever follows, not to the list.
        end = j
        while end > start and not self.lines[end - 1].strip():
            end -= 1
        self.i = end
        raw_lines = self.lines[start:end]
        segments = self._list_segments(raw_lines)
        block = self.emit("list", "\n".join(raw_lines), segments=segments)
        block.units = [u for u in self.units_sink]
        self.units_sink = []
        return True

    def _list_segments(self, raw_lines: list[str]) -> list[dict]:
        segments: list[dict] = []
        current: dict | None = None
        blanks = 0
        k = 0
        while k < len(raw_lines):
            line = raw_lines[k]
            if not line.strip():
                blanks += 1
                current = None
                k += 1
                continue
            fence = FENCE_RE.match(line)
            if fence:
                char, length = fence.group("fence")[0], len(fence.group("fence"))
                chunk = [line]
                k += 1
                while k < len(raw_lines):
                    chunk.append(raw_lines[k])
                    close = FENCE_RE.match(raw_lines[k])
                    k += 1
                    if close and close.group("fence")[0] == char and len(close.group("fence")) >= length:
                        break
                segments.append({"type": "raw", "lines": chunk, "blank_before": blanks})
                blanks, current = 0, None
                continue
            if TABLE_DELIM_RE.match(line) and "|" in line:
                # Pull the preceding header row into a verbatim table chunk.
                chunk = [segments and None] and []
                if segments and segments[-1]["type"] in {"item", "cont"}:
                    prev = segments.pop()
                    chunk.append(prev["indent"] + (prev.get("marker", "") and prev["marker"] + " ") + prev["text"])
                    self.units_sink = [u for u in self.units_sink if u.uid != prev.get("uid")]
                chunk.append(line)
                k += 1
                while k < len(raw_lines) and raw_lines[k].strip() and "|" in raw_lines[k]:
                    chunk.append(raw_lines[k])
                    k += 1
                segments.append({"type": "raw", "lines": chunk, "blank_before": blanks})
                blanks, current = 0, None
                continue
            item = LIST_RE.match(line)
            if item and item.group("space"):
                text = line[item.end():].strip()
                uid = self.uid()
                current = {
                    "type": "item", "indent": _expand(item.group("indent")),
                    "marker": item.group("marker"), "text": text,
                    "uid": uid, "blank_before": blanks,
                }
                segments.append(current)
                self.units_sink.append(Unit(uid, "list_item", text))
                blanks = 0
                k += 1
                continue
            if current is not None:
                # Soft-wrapped continuation of the same paragraph.
                current["text"] = (current["text"] + " " + line.strip()).strip()
                for u in self.units_sink:
                    if u.uid == current["uid"]:
                        u.text = current["text"]
                k += 1
                continue
            expanded = _expand(line)
            indent = len(expanded) - len(expanded.lstrip())
            uid = self.uid()
            current = {
                "type": "cont", "indent": " " * indent, "text": line.strip(),
                "uid": uid, "blank_before": blanks,
            }
            segments.append(current)
            self.units_sink.append(Unit(uid, "list_item", line.strip()))
            blanks = 0
            k += 1
        for seg in segments:
            seg["blank_before"] = min(seg.get("blank_before", 0), 1)
        if segments:
            segments[0]["blank_before"] = 0
        return segments

    def _paragraph(self) -> None:
        start = self.i
        buf: list[str] = []
        while self.i < len(self.lines):
            line = self.lines[self.i]
            if not line.strip():
                break
            if buf and SETEXT_RE.match(line):
                level = 1 if line.strip().startswith("=") else 2
                text = " ".join(x.strip() for x in buf).strip()
                uid = self.uid()
                self.i += 1
                self.emit(
                    "heading", "\n".join(self.lines[start:self.i]),
                    units=[Unit(uid, "heading", text)],
                    level=level, text=text, style="setext",
                )
                return
            if buf and _starts_block(line):
                break
            buf.append(line)
            self.i += 1
        text = " ".join(x.strip() for x in buf).strip()
        uid = self.uid()
        self.emit(
            "paragraph", "\n".join(self.lines[start:self.i]),
            units=[Unit(uid, "paragraph", text)] if text else [],
        )


def _starts_block(line: str) -> bool:
    """True when `line` cannot be a lazy continuation of a paragraph."""
    if FENCE_RE.match(line) or HR_RE.match(line) or ATX_RE.match(line):
        return True
    if QUOTE_RE.match(line) or HTML_OPEN_RE.match(line):
        return True
    if MATH_OPEN_RE.match(line) or MATH_ENV_RE.match(line):
        return True
    m = LIST_RE.match(line)
    return bool(m and m.group("space"))


def _html_end(lines: list[str], start: int, limit: int) -> int:
    """The line after the HTML block that begins at `start`.

    Only the element that *opened* the block holds it open. Converted documents
    are full of tags that never close — `<t>` standing in for an empty cell,
    a `<tbody>` without its optional end tag — and tracking overall nesting
    depth would let any one of them keep the block open to the end of the
    chapter. Matching the opening element by name is immune to all of them.

    `limit` is the usual CommonMark end of the block, the next blank line, and
    stays the answer whenever the opening element is never closed.
    """
    opener = HTML_TAG_RE.search(lines[start])
    if opener is None:
        return limit
    name = opener.group("name").lower()
    if (opener.group("close") or name in VOID_TAGS
            or opener.group("attrs").rstrip().endswith("/")):
        return start + 1  # nothing was opened, so nothing stays open

    close_re = re.compile(rf"</{re.escape(name)}\s*>", re.IGNORECASE)
    open_re = re.compile(rf"<{re.escape(name)}\b[^<>]*(?<!/)>", re.IGNORECASE)
    depth = 0
    for j in range(start, limit):
        depth += len(open_re.findall(lines[j])) - len(close_re.findall(lines[j]))
        if depth <= 0:
            return j + 1
    return limit


def _html_table_units(raw: str, uid: Callable[[], str]) -> tuple[list[Unit], list[tuple[str, str]]]:
    """Extract only visible cell text while keeping tags and attributes exact."""
    units: list[Unit] = []
    parts: list[tuple[str, str]] = []
    cell_depth = 0
    pos = 0

    def add_text(text: str) -> None:
        if not cell_depth or not text.strip():
            parts.append(("literal", text))
            return
        prefix = text[:len(text) - len(text.lstrip())]
        suffix = text[len(text.rstrip()):]
        body = text.strip()
        if prefix:
            parts.append(("literal", prefix))
        unit = Unit(uid(), "html_cell", body)
        units.append(unit)
        parts.append(("unit", unit.uid))
        if suffix:
            parts.append(("literal", suffix))

    for match in HTML_TOKEN_RE.finditer(raw):
        add_text(raw[pos:match.start()])
        tag = match.group()
        parts.append(("literal", tag))
        cell = HTML_CELL_RE.match(tag)
        if cell:
            cell_depth = max(0, cell_depth + (-1 if cell.group(1) else 1))
        pos = match.end()
    add_text(raw[pos:])
    return units, parts


def _expand(text: str) -> str:
    return text.expandtabs(4)


def _split_table_row(line: str) -> list[str]:
    """Split a pipe-table row, honouring `\\|` escapes and inline code spans."""
    body = line.strip()
    if body.startswith("|"):
        body = body[1:]
    if body.endswith("|") and not body.endswith("\\|"):
        body = body[:-1]
    cells, buf = [], []
    in_code = False
    k = 0
    while k < len(body):
        ch = body[k]
        if ch == "\\" and k + 1 < len(body):
            buf.append(body[k:k + 2])
            k += 2
            continue
        if ch == "`":
            in_code = not in_code
        if ch == "|" and not in_code:
            cells.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
        k += 1
    cells.append("".join(buf))
    return cells


def parse_markdown(text: str) -> list[Block]:
    parser = _Parser(text)
    parser.units_sink = []
    return parser.parse()


def document_title(blocks: list[Block], fallback: str) -> str:
    for block in blocks:
        if block.kind == "front_matter":
            m = re.search(r"^title:\s*(.+)$", block.raw, re.MULTILINE)
            if m:
                return m.group(1).strip().strip("'\"")
        if block.kind == "heading" and block.meta.get("level") == 1 and block.meta.get("text"):
            return block.meta["text"]
    for block in blocks:
        if block.kind == "heading" and block.meta.get("text"):
            return block.meta["text"]
    return fallback
