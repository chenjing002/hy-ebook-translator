"""EPUB → Markdown conversion, standard library only.

An EPUB is a zip archive: `META-INF/container.xml` names the package file
(OPF), whose manifest lists every resource and whose spine gives the reading
order of the XHTML documents. Each spine document is parsed into a small tree
and rendered as Markdown; images are copied out beside the Markdown file so the
references in it stay valid.

The renderer keeps what Markdown can express — headings, paragraphs, emphasis,
lists, block quotes, tables, figures with captions, code, footnotes, external
links — and drops presentational markup. Internal cross-references and index
anchors become plain text, since their targets do not survive the flattening of
a multi-file book into one document. Class names that publishers use in place
of semantic tags (`Italic`, `Semibold`, `Subscript`, `Superscript`) are honoured.
"""

from __future__ import annotations

import posixpath
import re
import time
import zipfile
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
from xml.etree import ElementTree

OPF_NS = "{http://www.idpf.org/2007/opf}"
DC_NS = "{http://purl.org/dc/elements/1.1/}"
CONTAINER_NS = "{urn:oasis:names:tc:opendocument:xmlns:container}"

VOID_TAGS = {"img", "br", "hr", "col", "meta", "link", "input", "wbr", "source", "base", "area"}
SKIP_TAGS = {"head", "script", "style", "title", "svg", "math", "video", "audio", "object",
             "iframe", "template", "noscript"}
BLOCK_TAGS = {
    "p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "ul", "ol", "li", "table",
    "thead", "tbody", "tfoot", "tr", "td", "th", "caption", "figure", "figcaption", "pre",
    "hr", "section", "article", "aside", "nav", "header", "footer", "main", "body", "html",
    "dl", "dt", "dd", "address", "details", "summary", "center", "form", "fieldset",
}
HEADING_TAGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
ITALIC_CLASSES = {"italic", "i", "em", "emphasis"}
BOLD_CLASSES = {"bold", "semibold", "b", "strong"}
SUB_CLASSES = {"subscript", "sub"}
SUP_CLASSES = {"superscript", "sup"}
UPPER_CLASSES = {"allcaps", "uppercase", "caps", "smallcaps"}
EXTERNAL_SCHEMES = {"http", "https", "mailto", "ftp", "tel"}

# Characters that would otherwise be read as Markdown or as a KaTeX delimiter.
INLINE_ESCAPE_RE = re.compile(r"([\\`*_\[\]<$])")
LINE_START_ESCAPE_RE = re.compile(r"^(\s*)(#{1,6}\s|>|[-+*]\s|\d{1,9}[.)]\s|[=-]{3,}\s*$)")
WHITESPACE_RE = re.compile(r"\s+")
EDGE_RE = re.compile(r"^(\s*)(.*?)(\s*)$", re.DOTALL)
# Emphasis is written with private-use stand-ins while a paragraph is being
# assembled, so `<i>x</i><i>i</i>` can be joined into one `*xi*` instead of
# the ambiguous `*x**i*`; the stand-ins become `*` and `**` at the end.
EM, STRONG = "\ue000", "\ue001"
MARKERS = {EM: "*", STRONG: "**"}


# ── document tree ────────────────────────────────────────────────────────────


@dataclass
class Node:
    tag: str
    attrs: dict[str, str] = field(default_factory=dict)
    children: list["Node | str"] = field(default_factory=list)

    @property
    def classes(self) -> set[str]:
        return {c.lower() for c in self.attrs.get("class", "").split()}

    def text(self) -> str:
        return "".join(c if isinstance(c, str) else c.text() for c in self.children)


class TreeBuilder(HTMLParser):
    """A tolerant tree builder: XHTML in the wild is not always well formed."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = Node("document")
        self.stack = [self.root]
        self.skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self.skip_depth:
            if tag not in VOID_TAGS:
                self.skip_depth += 1
            return
        if tag in SKIP_TAGS:
            self.skip_depth = 1
            return
        self.imply_end(tag)
        node = Node(tag, {k: (v or "") for k, v in attrs})
        self.stack[-1].children.append(node)
        if tag not in VOID_TAGS:
            self.stack.append(node)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self.skip_depth:
            return
        if tag in SKIP_TAGS:
            return
        self.stack[-1].children.append(Node(tag, {k: (v or "") for k, v in attrs}))

    def imply_end(self, tag: str) -> None:
        """HTML closes some elements for you: a new <p> ends an open one, a
        new <li> ends its sibling, and so on. Apply the same rules."""
        if tag in BLOCK_TAGS:
            self.close_open("p", stop_at=BLOCK_TAGS - {"p"})
        if tag == "li":
            self.close_open("li", stop_at={"ul", "ol"})
        if tag in {"td", "th"}:
            self.close_open("td", stop_at={"tr"})
            self.close_open("th", stop_at={"tr"})
        if tag == "tr":
            self.close_open("tr", stop_at={"table", "thead", "tbody", "tfoot"})
        if tag in {"dt", "dd"}:
            self.close_open("dt", stop_at={"dl"})
            self.close_open("dd", stop_at={"dl"})

    def close_open(self, tag: str, stop_at: set[str]) -> None:
        for i in range(len(self.stack) - 1, 0, -1):
            if self.stack[i].tag == tag:
                del self.stack[i:]
                return
            if self.stack[i].tag in stop_at:
                return

    def handle_endtag(self, tag: str) -> None:
        if self.skip_depth:
            if tag not in VOID_TAGS:
                self.skip_depth -= 1
            return
        if tag in VOID_TAGS:
            return
        for i in range(len(self.stack) - 1, 0, -1):
            if self.stack[i].tag == tag:
                del self.stack[i:]
                return

    def handle_data(self, data: str) -> None:
        if self.skip_depth or not data:
            return
        self.stack[-1].children.append(data)


def parse_html(text: str) -> Node:
    builder = TreeBuilder()
    builder.feed(text)
    builder.close()
    return builder.root


# ── package metadata ─────────────────────────────────────────────────────────


@dataclass
class Package:
    opf_path: str
    title: str
    authors: list[str]
    language: str
    manifest: dict[str, tuple[str, str, str]]   # id → (href, media type, properties)
    spine: list[tuple[str, bool]]               # (href, linear)
    nav_href: str | None


def read_package(zf: zipfile.ZipFile) -> Package:
    container = ElementTree.fromstring(zf.read("META-INF/container.xml"))
    rootfile = container.find(f".//{CONTAINER_NS}rootfile")
    if rootfile is None or not rootfile.get("full-path"):
        raise ValueError("container.xml names no package file")
    opf_path = rootfile.get("full-path", "")
    opf_dir = posixpath.dirname(opf_path)
    root = ElementTree.fromstring(zf.read(opf_path))

    titles: list[tuple[str, str, str]] = []  # (id, type, text)
    refines: dict[str, dict[str, str]] = {}
    meta = root.find(f"{OPF_NS}metadata")
    if meta is None:
        meta = root
    for m in meta.iter(f"{OPF_NS}meta"):
        target = (m.get("refines") or "").lstrip("#")
        prop = m.get("property") or ""
        if target and prop:
            refines.setdefault(target, {})[prop] = (m.text or "").strip()
    for t in meta.iter(f"{DC_NS}title"):
        tid = t.get("id") or ""
        kind = refines.get(tid, {}).get("title-type", "")
        titles.append((tid, kind, " ".join((t.text or "").split())))
    main = [t for _, k, t in titles if k == "main"] or [t for _, k, t in titles if k == ""]
    subtitle = [t for _, k, t in titles if k == "subtitle"]
    title = main[0] if main else (titles[0][2] if titles else "")
    if subtitle and subtitle[0] and subtitle[0] not in title:
        title = f"{title}: {subtitle[0]}" if title else subtitle[0]
    authors = [" ".join((c.text or "").split()) for c in meta.iter(f"{DC_NS}creator")]
    authors = [a for a in authors if a]
    lang_el = meta.find(f"{DC_NS}language")
    language = (lang_el.text or "").strip() if lang_el is not None else ""

    manifest: dict[str, tuple[str, str, str]] = {}
    nav_href = None
    for item in root.iter(f"{OPF_NS}item"):
        href = posixpath.normpath(posixpath.join(opf_dir, unquote(item.get("href", ""))))
        props = item.get("properties", "")
        manifest[item.get("id", "")] = (href, item.get("media-type", ""), props)
        if "nav" in props.split():
            nav_href = href
    spine: list[tuple[str, bool]] = []
    for ref in root.iter(f"{OPF_NS}itemref"):
        entry = manifest.get(ref.get("idref", ""))
        if entry is None:
            continue
        spine.append((entry[0], ref.get("linear", "yes").lower() != "no"))
    return Package(opf_path, title, authors, language, manifest, spine, nav_href)


# ── rendering ────────────────────────────────────────────────────────────────


@dataclass
class ConvertResult:
    markdown: str
    title: str
    authors: list[str]
    documents: int
    images: list[Path]
    warnings: list[str]


class Renderer:
    """Markdown for one spine document."""

    def __init__(self, doc_path: str, images: "ImageStore | None", demote: int = 0) -> None:
        self.doc_path = doc_path
        self.images = images
        self.demote = demote
        self.footnotes: list[tuple[str, str]] = []
        self.warnings: list[str] = []

    # blocks ────────────────────────────────────────────────────────────────

    def render(self, root: Node) -> str:
        body = _find(root, "body") or root
        blocks = self.blocks(body)
        for key, text in self.footnotes:
            blocks.append(f"[^{key}]: {text}")
        return "\n\n".join(b for b in blocks if b.strip())

    def blocks(self, node: Node) -> list[str]:
        """Render a container: block children recurse, runs of inline
        children become paragraphs of their own."""
        out: list[str] = []
        run: list[Node | str] = []

        def flush() -> None:
            if run:
                text = self.inline_run(run)
                if text:
                    out.append(text)
                run.clear()

        for child in node.children:
            if isinstance(child, str) or child.tag not in BLOCK_TAGS:
                run.append(child)
                continue
            flush()
            out.extend(self.block(child))
        flush()
        return out

    def block(self, node: Node) -> list[str]:
        tag = node.tag
        if tag in HEADING_TAGS:
            level = min(6, HEADING_TAGS[tag] + self.demote)
            text = self.inline_run(node.children, escape_lines=False).replace("\n", " ").strip()
            text = re.sub(r"\s+", " ", text)
            return [f"{'#' * level} {text}"] if text else []
        if tag == "p":
            text = self.inline_run(node.children)
            return [text] if text else []
        if tag == "blockquote":
            inner = self.blocks(node)
            if not inner:
                return []
            quoted = "\n\n".join(inner).split("\n")
            return ["\n".join(f"> {ln}".rstrip() for ln in quoted)]
        if tag in {"ul", "ol"}:
            return self.list(node)
        if tag == "li":
            # A stray item outside a list.
            return self.blocks(node)
        if tag == "pre":
            return [self.code_block(node)]
        if tag == "hr":
            return ["---"]
        if tag == "table":
            return [self.table(node)]
        if tag == "figcaption":
            text = self.inline_run(node.children)
            return [_wrap(text, "*")] if text else []
        if tag == "dt":
            text = self.inline_run(node.children)
            return [_wrap(text, "**")] if text else []
        if tag == "nav":
            kind = node.attrs.get("epub:type", "")
            if kind in {"toc", "landmarks", "page-list"} or node.attrs.get("hidden") is not None:
                return []
            return self.blocks(node)
        if tag == "aside" and _is_footnote(node):
            self.collect_footnote(node)
            return []
        if tag in {"thead", "tbody", "tfoot", "tr", "td", "th", "caption"}:
            return self.blocks(node)
        return self.blocks(node)

    def list(self, node: Node) -> list[str]:
        ordered = node.tag == "ol"
        try:
            number = int(node.attrs.get("start", "1"))
        except ValueError:
            number = 1
        lines: list[str] = []
        for child in node.children:
            if isinstance(child, str) or child.tag != "li":
                if isinstance(child, Node) and child.tag in {"ul", "ol"}:
                    # A nested list placed directly inside its parent list.
                    nested = self.list(child)
                    lines.extend("  " + ln if ln else ln for block in nested for ln in block.split("\n"))
                continue
            marker = f"{number}. " if ordered else "- "
            number += 1
            parts = self.blocks(child) or [""]
            body = _join_item(parts).split("\n")
            indent = " " * len(marker)
            lines.append(marker + body[0])
            lines.extend((indent + ln) if ln else ln for ln in body[1:])
        return ["\n".join(lines)] if lines else []

    def code_block(self, node: Node) -> str:
        raw = node.text().strip("\n")
        fence = "```"
        while fence in raw:
            fence += "`"
        lang = ""
        code = _find(node, "code")
        if code is not None:
            for cls in code.classes:
                if cls.startswith("language-"):
                    lang = cls[len("language-"):]
        return f"{fence}{lang}\n{raw}\n{fence}"

    def table(self, node: Node) -> str:
        rows: list[list[tuple[str, bool]]] = []
        complex_table = False
        for tr in _iter_rows(node):
            cells: list[tuple[str, bool]] = []
            for cell in tr.children:
                if isinstance(cell, str) or cell.tag not in {"td", "th"}:
                    continue
                if cell.attrs.get("rowspan", "1") not in {"", "1"} or \
                        cell.attrs.get("colspan", "1") not in {"", "1"}:
                    complex_table = True
                if any(isinstance(c, Node) and c.tag in BLOCK_TAGS - {"p", "div"}
                       for c in cell.children):
                    complex_table = True
                parts = self.blocks(cell)
                text = " ".join(p.replace("\n", " ") for p in parts).strip()
                cells.append((text.replace("|", "\\|"), cell.tag == "th"))
            if cells:
                rows.append(cells)
        if not rows:
            return ""
        if complex_table:
            return self.html_table(node)
        width = max(len(r) for r in rows)
        rows = [r + [("", False)] * (width - len(r)) for r in rows]
        header: list[str]
        if all(is_th for _, is_th in rows[0]):
            header = [t for t, _ in rows[0]]
            body = rows[1:]
        else:
            header = [""] * width
            body = rows
        lines = ["| " + " | ".join(header) + " |", "|" + "|".join(" --- " for _ in range(width)) + "|"]
        for row in body:
            lines.append("| " + " | ".join(t for t, _ in row) + " |")
        return "\n".join(lines)

    def html_table(self, node: Node) -> str:
        """A table with spans or block content keeps its structure as HTML."""
        lines = ["<table>"]
        for tr in _iter_rows(node):
            cells = []
            for cell in tr.children:
                if isinstance(cell, str) or cell.tag not in {"td", "th"}:
                    continue
                attrs = ""
                for name in ("rowspan", "colspan"):
                    value = cell.attrs.get(name, "")
                    if value and value != "1":
                        attrs += f' {name}="{value}"'
                text = " ".join(p.replace("\n", " ") for p in self.blocks(cell)).strip()
                cells.append(f"<{cell.tag}{attrs}>{_inline_markdown_to_html(text)}</{cell.tag}>")
            if cells:
                lines.append("<tr>" + "".join(cells) + "</tr>")
        lines.append("</table>")
        return "\n".join(lines)

    def collect_footnote(self, node: Node) -> None:
        key = node.attrs.get("id", "") or f"n{len(self.footnotes) + 1}"
        text = " ".join(self.blocks(node)).replace("\n", " ").strip()
        self.footnotes.append((key, text))

    # inline ───────────────────────────────────────────────────────────────

    def inline_run(self, nodes: list[Node | str], escape_lines: bool = True) -> str:
        text = _finish_markers("".join(self.inline(n) for n in nodes))
        text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text).strip()
        if not escape_lines:
            return text
        # "1. " or "- " opening a paragraph would start a list.
        return "\n".join(LINE_START_ESCAPE_RE.sub(r"\1\\\2", ln) for ln in text.split("\n"))

    def inline(self, node: Node | str) -> str:
        if isinstance(node, str):
            return INLINE_ESCAPE_RE.sub(r"\\\1", WHITESPACE_RE.sub(" ", node.replace("\xa0", " ")))
        tag = node.tag
        if tag == "br":
            return "  \n"
        if tag == "img":
            return self.image(node)
        if tag == "a":
            return self.link(node)
        if tag == "code" or tag == "kbd" or tag == "samp":
            raw = node.text().replace("\n", " ")
            ticks = "`"
            while ticks in raw:
                ticks += "`"
            return f"{ticks}{raw}{ticks}" if raw.strip() else ""
        if tag in BLOCK_TAGS:
            # A block inside inline context (a <div> inside a <p>): flatten it.
            return "\n\n".join(self.block(node))
        inner = "".join(self.inline(c) for c in node.children)
        classes = node.classes
        if classes & UPPER_CLASSES:
            inner = inner.upper()
        if tag == "sub" or classes & SUB_CLASSES:
            inner = _wrap_html(inner, "sub")
        if tag == "sup" or classes & SUP_CLASSES:
            inner = _wrap_html(inner, "sup")
        if tag in {"i", "em", "cite", "dfn", "var"} or classes & ITALIC_CLASSES:
            inner = _wrap(inner, EM)
        if tag in {"b", "strong"} or classes & BOLD_CLASSES:
            inner = _wrap(inner, STRONG)
        if tag in {"s", "del", "strike"}:
            inner = _wrap(inner, "~~")
        return inner

    def link(self, node: Node) -> str:
        inner = "".join(self.inline(c) for c in node.children)
        href = node.attrs.get("href", "").strip()
        kind = node.attrs.get("epub:type", "")
        if "noteref" in kind.split():
            key = urlsplit(href).fragment or inner.strip() or f"n{len(self.footnotes) + 1}"
            return f"[^{key}]"
        if urlsplit(href).scheme.lower() in EXTERNAL_SCHEMES and inner.strip():
            return f"[{inner.strip()}]({href})"
        return inner

    def image(self, node: Node) -> str:
        src = node.attrs.get("src", "").strip()
        if not src or self.images is None:
            return ""
        target = self.images.store(self.doc_path, src)
        if target is None:
            self.warnings.append(f"{self.doc_path}: image not in archive: {src}")
            return ""
        alt = WHITESPACE_RE.sub(" ", node.attrs.get("alt", "")).strip().replace("]", "")
        return f"![{alt}]({_url(target)})"


class ImageStore:
    """Copies the images a document refers to out of the archive, once each."""

    def __init__(self, zf: zipfile.ZipFile, out_dir: Path, rel_prefix: str) -> None:
        self.zf = zf
        self.out_dir = out_dir
        self.rel_prefix = rel_prefix
        self.names = {n for n in zf.namelist()}
        self.written: dict[str, str] = {}     # archive path → relative Markdown path
        self.used: set[str] = set()
        self.files: list[Path] = []

    def store(self, doc_path: str, src: str) -> str | None:
        src = unquote(urlsplit(src).path)
        archive = posixpath.normpath(posixpath.join(posixpath.dirname(doc_path), src))
        if archive in self.written:
            return self.written[archive]
        if archive not in self.names:
            return None
        name = posixpath.basename(archive)
        stem, dot, ext = name.rpartition(".")
        if not dot:
            stem, ext = name, ""
        candidate = name
        n = 1
        while candidate.lower() in self.used:
            n += 1
            candidate = f"{stem}-{n}.{ext}" if ext else f"{stem}-{n}"
        self.used.add(candidate.lower())
        self.out_dir.mkdir(parents=True, exist_ok=True)
        target = self.out_dir / candidate
        target.write_bytes(self.zf.read(archive))
        self.files.append(target)
        rel = f"{self.rel_prefix}/{candidate}" if self.rel_prefix else candidate
        self.written[archive] = rel
        return rel


# ── entry point ──────────────────────────────────────────────────────────────


def convert_epub(epub: Path, *, images_dir: Path | None, images_rel: str = "",
                 title: str | None = None, add_title: bool = True,
                 keep_nav: bool = False, include_nonlinear: bool = False,
                 provenance: bool = True) -> ConvertResult:
    """Render the whole book as one Markdown string.

    `images_dir` receives the images; `images_rel` is the path to that
    directory as the Markdown should reference it (relative to the Markdown
    file). With `images_dir=None`, images are dropped.
    """
    warnings: list[str] = []
    with zipfile.ZipFile(epub) as zf:
        pkg = read_package(zf)
        book_title = title or pkg.title or epub.stem.replace("_", " ")
        store = ImageStore(zf, images_dir, images_rel.rstrip("/")) if images_dir else None

        # If the book uses h1 for its chapters and we add a title above them,
        # every heading moves down one level so the title stays alone at the top.
        demote = 0
        if add_title:
            for href, linear in pkg.spine:
                if (linear or include_nonlinear) and href in zf.namelist():
                    if re.search(rb"<h1[\s>]", zf.read(href), re.IGNORECASE):
                        demote = 1
                        break

        parts: list[str] = []
        count = 0
        for href, linear in pkg.spine:
            if not linear and not include_nonlinear:
                continue
            if href == pkg.nav_href and not keep_nav:
                continue
            if href not in zf.namelist():
                warnings.append(f"spine item missing from archive: {href}")
                continue
            raw = zf.read(href)
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                text = raw.decode("latin-1", errors="replace")
            renderer = Renderer(href, store, demote=demote)
            md = renderer.render(parse_html(text))
            warnings.extend(renderer.warnings)
            if md.strip():
                parts.append(md)
                count += 1

    head: list[str] = []
    if provenance:
        head.append(f"<!-- Generated by hy-ebook-translator epub2md from {epub.name} on "
                    f"{time.strftime('%Y-%m-%d')} -->")
    if add_title:
        head.append(f"# {_plain_title(book_title)}")
        if pkg.authors:
            head.append(_wrap(", ".join(INLINE_ESCAPE_RE.sub(r"\\\1", a) for a in pkg.authors), "*"))
    markdown = "\n\n".join(head + parts).rstrip() + "\n"
    markdown = re.sub(r"\n{3,}", "\n\n", markdown)
    return ConvertResult(
        markdown=markdown, title=book_title, authors=pkg.authors, documents=count,
        images=store.files if store else [], warnings=warnings,
    )


# ── helpers ──────────────────────────────────────────────────────────────────


def _find(node: Node, tag: str) -> Node | None:
    for child in node.children:
        if isinstance(child, Node):
            if child.tag == tag:
                return child
            found = _find(child, tag)
            if found is not None:
                return found
    return None


def _iter_rows(table: Node):
    for child in table.children:
        if isinstance(child, str):
            continue
        if child.tag == "tr":
            yield child
        elif child.tag in {"thead", "tbody", "tfoot"}:
            yield from _iter_rows(child)


def _is_footnote(node: Node) -> bool:
    kinds = set(node.attrs.get("epub:type", "").split())
    return bool(kinds & {"footnote", "endnote", "note", "rearnote"})


LIST_START_RE = re.compile(r"^(?:[-*+] |\d{1,9}\. )")


def _join_item(parts: list[str]) -> str:
    """A nested list stays tight against the item that holds it; any other
    block inside an item is a paragraph of its own."""
    out = parts[0]
    for part in parts[1:]:
        out += ("\n" if LIST_START_RE.match(part) else "\n\n") + part
    return out


def _wrap(text: str, marker: str) -> str:
    """Emphasis markers belong against the words, not against the spaces
    publishers leave inside the tag: `<i>word </i>next` → `*word* next`."""
    lead, core, trail = EDGE_RE.match(text).groups()
    if not core:
        return lead + trail
    if core.startswith(marker) and core.endswith(marker) and len(core) > 2 * len(marker):
        return lead + core + trail
    return f"{lead}{marker}{core}{marker}{trail}"


def _finish_markers(text: str) -> str:
    """Turn emphasis stand-ins into Markdown, joining runs that touch."""
    for stand_in, marker in MARKERS.items():
        text = text.replace(stand_in * 2, "")
        # A marker followed by a space ends up outside the words it wraps
        # when `<i>a </i><i>b</i>`-style tags meet; those spaces are moved in.
        text = text.replace(stand_in, marker)
    return text


def _wrap_html(text: str, tag: str) -> str:
    lead, core, trail = EDGE_RE.match(text).groups()
    if not core:
        return lead + trail
    return f"{lead}<{tag}>{core}</{tag}>{trail}"


MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)\)")
MD_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
MD_CODE_RE = re.compile(r"(`+)(.+?)\1")
MD_STRONG_RE = re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*")
MD_EM_RE = re.compile(r"\*(?=\S)(.+?)(?<=\S)\*")
MD_ESCAPE_RE = re.compile(r"\\([\\`*_\[\]<$])")


def _inline_markdown_to_html(text: str) -> str:
    """Markdown inside a raw HTML block is not rendered, so a cell of an HTML
    table gets its emphasis, code, links and images as HTML instead."""
    literals: list[str] = []

    def hold(m: re.Match) -> str:
        ch = m.group(1)
        literals.append("&lt;" if ch == "<" else "&amp;" if ch == "&" else ch)
        return f"\ue002{len(literals) - 1}\ue003"

    text = MD_ESCAPE_RE.sub(hold, text)
    text = MD_CODE_RE.sub(lambda m: f"<code>{m.group(2).strip()}</code>", text)
    text = MD_IMAGE_RE.sub(r'<img src="\2" alt="\1">', text)
    text = MD_LINK_RE.sub(r'<a href="\2">\1</a>', text)
    text = MD_STRONG_RE.sub(r"<strong>\1</strong>", text)
    text = MD_EM_RE.sub(r"<em>\1</em>", text)
    return re.sub(r"\ue002(\d+)\ue003", lambda m: literals[int(m.group(1))], text)


def _url(path: str) -> str:
    # Spaces and parentheses would end the destination early.
    return path.replace(" ", "%20").replace("(", "%28").replace(")", "%29")


def _plain_title(title: str) -> str:
    return INLINE_ESCAPE_RE.sub(r"\\\1", " ".join(title.split()))
