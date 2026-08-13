"""Structural analysis of the PDF itself.

A text classifier only sees the string that came out of the extractor. These
checks look at *how* the text is drawn, which is where indirect prompt
injection actually hides: invisible render mode, white-on-white fill, 1pt
fonts, glyphs positioned outside the page, Unicode tag smuggling, and payloads
parked in metadata, annotations or auto-executing actions.
"""
from __future__ import annotations

import io
import re

import fitz  # PyMuPDF
from pypdf import PdfReader

from . import config

# --- tunables -------------------------------------------------------------
TINY_FONT_PT = 2.0
LOW_CONTRAST = 2.0         # WCAG-style contrast ratio below which text reads as invisible
LOW_OPACITY = 0.10
MIN_HIDDEN_CHARS = 12      # ignore stray artefacts; real payloads are wordy
SHINGLE = 5                # word n-gram width for extractor comparison
MIN_HIDDEN_WORDS = 6       # shorter divergences are extractor noise, not payloads

# Zero-width, bidi-override and other non-rendering characters.
INVISIBLE_CHARS = re.compile(
    "["
    "\u00ad"              # soft hyphen
    "\u180e"              # mongolian vowel separator
    "\u200b-\u200f"       # zero-width space/non-joiner/joiner, LRM, RLM
    "\u202a-\u202e"       # bidi embedding / override
    "\u2060-\u2064"       # word joiner, invisible operators
    "\u206a-\u206f"       # deprecated formatting controls
    "\ufeff"              # zero-width no-break space (BOM)
    "]"
)
# Unicode Tags block: a known channel for smuggling instructions past humans.
TAG_CHARS = re.compile("[\U000e0000-\U000e007f]")


def _rgb(color) -> tuple[float, float, float]:
    """Normalise PyMuPDF colour (int sRGB or float tuple) to 0..1 RGB."""
    if isinstance(color, (tuple, list)) and len(color) >= 3:
        return tuple(float(c) for c in color[:3])
    try:
        c = int(color)
    except (TypeError, ValueError):
        return (0.0, 0.0, 0.0)
    return (((c >> 16) & 255) / 255, ((c >> 8) & 255) / 255, (c & 255) / 255)


def _page_bg_color(page) -> tuple[float, float, float]:
    """Best-effort actual page background: the fill color of the largest
    vector drawing that covers (nearly) the whole page. Falls back to white
    when no such fill exists — the common case for plain text-only PDFs —
    which keeps this a no-op for pages that were already scored correctly."""
    W, H = page.rect.width, page.rect.height
    best, best_area = None, 0.0
    try:
        drawings = page.get_drawings()
    except Exception:
        return (1.0, 1.0, 1.0)
    for d in drawings:
        fill = d.get("fill")
        rect = d.get("rect")
        if fill is None or rect is None:
            continue
        if rect.width < W * 0.9 or rect.height < H * 0.9:
            continue
        area = rect.width * rect.height
        if area > best_area:
            best, best_area = _rgb(fill), area
    return best if best is not None else (1.0, 1.0, 1.0)


def _rel_luminance(rgb) -> float:
    def lin(u):
        return u / 12.92 if u <= 0.03928 else ((u + 0.055) / 1.055) ** 2.4
    r, g, b = rgb
    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)


def _contrast_ratio(rgb, bg) -> float:
    """WCAG-style contrast ratio between text color and page background."""
    l1, l2 = _rel_luminance(rgb), _rel_luminance(bg)
    hi, lo = max(l1, l2), min(l1, l2)
    return (hi + 0.05) / (lo + 0.05)


def _signal(sid, title, severity, detail, page=None, evidence=None) -> dict:
    return {
        "id": sid,
        "title": title,
        "severity": severity,
        "detail": detail,
        "page": page,
        "evidence": (evidence or "")[:300] or None,
    }


def _scan_spans(doc) -> tuple[list[dict], list[str]]:
    """Colour / size checks over every text span.

    Returns the signals plus every hidden string found, so the classifier can
    score the hidden text on its own instead of drowning it in page prose.
    """
    signals: list[dict] = []
    hidden: list[str] = []
    low_contrast_chars = 0
    tiny_chars = 0
    lc_ev = tiny_ev = ""
    lc_pg = tiny_pg = None

    for pno, page in enumerate(doc, start=1):
        if pno > config.MAX_PAGES:
            break
        bg = _page_bg_color(page)
        try:
            data = page.get_text("dict")
        except Exception:
            continue

        for block in data.get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "")
                    if not text.strip():
                        continue
                    n = len(text.strip())

                    rgb = _rgb(span.get("color", 0))
                    if _contrast_ratio(rgb, bg) < LOW_CONTRAST:
                        low_contrast_chars += n
                        lc_pg = lc_pg or pno
                        lc_ev = lc_ev or text.strip()
                        hidden.append(text)

                    if float(span.get("size", 12.0)) < TINY_FONT_PT:
                        tiny_chars += n
                        tiny_pg = tiny_pg or pno
                        tiny_ev = tiny_ev or text.strip()
                        hidden.append(text)

    if low_contrast_chars >= MIN_HIDDEN_CHARS:
        signals.append(
            _signal(
                "white-text",
                "Similar background text color",
                "high",
                f"{low_contrast_chars} characters are drawn in a color too close to the "
                "actual page background to be readable — invisible against the page "
                "(whatever its color), yet extract as text.",
                lc_pg,
                lc_ev,
            )
        )
    if tiny_chars >= MIN_HIDDEN_CHARS:
        signals.append(
            _signal(
                "tiny-font",
                "Sub-2pt font",
                "high",
                f"{tiny_chars} characters are set below {TINY_FONT_PT}pt — "
                "unreadable to a human reader but fully machine-readable.",
                tiny_pg,
                tiny_ev,
            )
        )
    return signals, hidden


def _norm(text: str) -> str:
    """Aggressive normalisation so we compare *content*, not extractor quirks
    (whitespace, hyphenation, ligatures and punctuation all differ between
    PyMuPDF and pypdf on identical input)."""
    text = INVISIBLE_CHARS.sub("", text.lower())
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _tokenise(text: str) -> tuple[list[tuple[str, int, int]], str]:
    """Return [(lowercased word, start, end)] plus the string those offsets
    index into, so a matched run can be sliced back out with original casing."""
    cleaned = INVISIBLE_CHARS.sub("", text)
    toks = [
        (m.group(0).lower(), m.start(), m.end())
        for m in re.finditer(r"[A-Za-z0-9]+", cleaned)
    ]
    return toks, cleaned


def _scan_extractor_divergence(raw: bytes, visible_text: str) -> tuple[list[dict], str]:
    """Compare a viewer-faithful extractor against a content-stream extractor.

    MuPDF clips text extraction to the page's CropBox, so it returns what a
    human actually sees. pypdf walks the content stream and returns everything
    that was *drawn*, including glyphs positioned outside the visible page.

    Anything present in the second but missing from the first is, by
    construction, content a reader cannot see but an ingestion pipeline will
    happily feed to the model. This catches off-page positioning generically,
    without having to enumerate each placement trick.
    """
    try:
        reader = PdfReader(io.BytesIO(raw))
        pages = reader.pages[: config.MAX_PAGES]
        stream_text = "\n".join((p.extract_text() or "") for p in pages)
    except Exception:
        return [], ""

    # Compare word shingles rather than whole lines: the two extractors break
    # lines at different points on identical input, so line-level matching
    # reports differences that are purely cosmetic.
    stream_tokens, stream_clean = _tokenise(stream_text)
    visible_words = [w for w, _, _ in _tokenise(visible_text)[0]]
    if not stream_tokens:
        return [], ""

    k = SHINGLE
    seen = {
        tuple(visible_words[i : i + k]) for i in range(max(len(visible_words) - k + 1, 0))
    }

    stream_words = [w for w, _, _ in stream_tokens]
    unseen = [False] * len(stream_words)
    for i in range(len(stream_words) - k + 1):
        if tuple(stream_words[i : i + k]) not in seen:
            for j in range(i, i + k):
                unseen[j] = True

    runs: list[str] = []
    start = None
    for i, flag in enumerate(unseen + [False]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start >= MIN_HIDDEN_WORDS:
                # Slice the original text so casing and punctuation survive —
                # the classifier scores this, and normalising would strip the
                # very cues it was trained on.
                runs.append(stream_clean[stream_tokens[start][1] : stream_tokens[i - 1][2]])
            start = None

    if not runs:
        return [], ""

    hidden = "\n".join(runs)
    return (
        [
            _signal(
                "extractor-divergence",
                "Text visible to parsers but not to readers",
                "high",
                f"{len(hidden)} characters are recovered by a content-stream "
                "parser (pypdf) but not by a viewer-faithful one (MuPDF). This "
                "is text drawn outside the visible page area — invisible to a "
                "human, ingested by a RAG pipeline.",
                evidence=hidden,
            )
        ],
        hidden,
    )


def _scan_render_mode(doc) -> tuple[list[dict], list[str]]:
    """Detect PDF text render mode 3 (invisible) and near-zero opacity.

    get_texttrace() is the only API that exposes the render mode; it is not
    present in every PyMuPDF build, so failure here is non-fatal.
    """
    invisible_chars = 0
    page_hit = None
    hidden: list[str] = []
    for pno, page in enumerate(doc, start=1):
        if pno > config.MAX_PAGES:
            break
        try:
            spans = page.get_texttrace()
        except Exception:
            return [], []
        for span in spans:
            n = len(span.get("chars", []) or [])
            if not n:
                continue
            if span.get("type") == 3 or float(span.get("opacity", 1.0)) < LOW_OPACITY:
                invisible_chars += n
                page_hit = page_hit or pno
                try:
                    hidden.append("".join(chr(c[0]) for c in span["chars"]))
                except Exception:
                    pass

    if invisible_chars >= MIN_HIDDEN_CHARS:
        return [
            _signal(
                "invisible-render-mode",
                "Invisible text render mode",
                "high",
                f"{invisible_chars} glyphs are drawn with an invisible render "
                "mode or near-zero opacity. This is the classic OCR-layer "
                "trick repurposed to hide instructions.",
                page_hit,
            )
        ], hidden
    return [], []


def _scan_unicode(text: str) -> list[dict]:
    signals = []

    tags = TAG_CHARS.findall(text)
    if tags:
        decoded = "".join(chr(ord(c) - 0xE0000) for c in tags if 0xE0020 <= ord(c) <= 0xE007E)
        signals.append(
            _signal(
                "unicode-tag-smuggling",
                "Unicode Tag-block smuggling",
                "high",
                f"{len(tags)} characters from the Unicode Tags block are present. "
                "They render as nothing but many tokenizers decode them into "
                "readable ASCII.",
                evidence=decoded or None,
            )
        )

    invisible = INVISIBLE_CHARS.findall(text)
    if len(invisible) >= MIN_HIDDEN_CHARS:
        signals.append(
            _signal(
                "invisible-unicode",
                "Zero-width / bidi control characters",
                "medium",
                f"{len(invisible)} zero-width or direction-override characters "
                "found. These break naive keyword filters and can reorder how "
                "text appears versus how it is read.",
            )
        )
    return signals


def _scan_metadata_and_annots(doc) -> tuple[list[dict], list[str]]:
    """Collect metadata/annotation text so the classifier scores it too."""
    signals: list[dict] = []
    extra_text: list[str] = []

    meta = doc.metadata or {}
    for key, value in meta.items():
        if value and isinstance(value, str) and len(value.strip()) > 40:
            extra_text.append(value)
            signals.append(
                _signal(
                    "verbose-metadata",
                    f"Unusually long metadata field: {key}",
                    "low",
                    "Metadata is invisible in a viewer but is often fed to an "
                    "LLM alongside the body text.",
                    evidence=value,
                )
            )

    annot_chars = 0
    annot_page = None
    for pno, page in enumerate(doc, start=1):
        if pno > config.MAX_PAGES:
            break
        try:
            annots = list(page.annots() or [])
        except Exception:
            continue
        for a in annots:
            content = (a.info or {}).get("content", "")
            if content and content.strip():
                extra_text.append(content)
                annot_chars += len(content)
                annot_page = annot_page or pno

    if annot_chars >= MIN_HIDDEN_CHARS:
        signals.append(
            _signal(
                "annotation-text",
                "Text hidden in annotations",
                "medium",
                f"{annot_chars} characters live in PDF annotations rather than "
                "page content.",
                annot_page,
            )
        )
    return signals, extra_text


def _scan_active_content(raw: bytes) -> list[dict]:
    """Look for auto-executing constructs in the resolved object tree.

    An earlier version scanned the raw bytes for markers like b"/JS". That is
    unsound: compressed streams contain arbitrary binary, and on an external
    corpus it matched inside stream data (`...F.W/JS=fhY%...`) on 14 clean
    files. Resolving the document catalog costs a parse but cannot false-match.
    """
    signals: list[dict] = []
    try:
        reader = PdfReader(io.BytesIO(raw))
        root = reader.trailer.get("/Root", {})
        names = root.get("/Names", {}) or {}
    except Exception:
        return []

    def present(container, key) -> bool:
        try:
            return key in (container or {})
        except Exception:
            return False

    if present(names, "/JavaScript"):
        signals.append(_signal(
            "pdf-javascript", "Embedded JavaScript", "high",
            "The document name tree registers a /JavaScript entry, which runs "
            "when the file is opened.",
        ))
    if present(root, "/OpenAction"):
        signals.append(_signal(
            "pdf-openaction", "Auto-executing OpenAction", "medium",
            "The catalog defines an /OpenAction triggered on open.",
        ))
    if present(root, "/AA"):
        signals.append(_signal(
            "pdf-additional-action", "Additional (event) actions", "medium",
            "The catalog defines /AA event actions.",
        ))
    if present(names, "/EmbeddedFiles"):
        signals.append(_signal(
            "pdf-embedded-file", "Embedded file attachment", "medium",
            "The document carries an embedded file attachment.",
        ))
    return signals


def analyse(raw: bytes) -> dict:
    """Parse the PDF once and return extracted text plus forensic signals."""
    doc = fitz.open(stream=raw, filetype="pdf")
    try:
        pages = doc.page_count
        parts = []
        for pno, page in enumerate(doc, start=1):
            if pno > config.MAX_PAGES:
                break
            try:
                parts.append(page.get_text())
            except Exception:
                continue
        body = "\n".join(parts)

        signals: list[dict] = []
        hidden_parts: list[str] = []

        span_signals, span_hidden = _scan_spans(doc)
        signals += span_signals
        hidden_parts += span_hidden

        rm_signals, rm_hidden = _scan_render_mode(doc)
        signals += rm_signals
        hidden_parts += rm_hidden

        meta_signals, extra = _scan_metadata_and_annots(doc)
        signals += meta_signals
        hidden_parts += extra

        # Off-page text never reaches `body`, so recover it separately and add
        # it to what the classifier sees.
        div_signals, hidden = _scan_extractor_divergence(raw, body)
        signals += div_signals
        if hidden:
            extra.append(hidden)
            hidden_parts.append(hidden)

        full_text = "\n".join([body, *extra])
        signals += _scan_unicode(full_text)
        signals += _scan_active_content(raw)

        order = {"high": 0, "medium": 1, "low": 2}
        signals.sort(key=lambda s: order.get(s["severity"], 3))

        # De-duplicate while preserving order; the same string is often found
        # by more than one check (white AND tiny, for instance).
        seen: set[str] = set()
        uniq = []
        for part in hidden_parts:
            key = " ".join(str(part).lower().split())
            if key and key not in seen:
                seen.add(key)
                uniq.append(str(part).strip())

        return {
            "pages": pages,
            "text": full_text,
            "signals": signals,
            "hidden_text": "\n".join(uniq),
        }
    finally:
        doc.close()
