"""
PDF prompt-injection DETECTOR + scoring harness (defensive research).

Detects hidden text using layered signals, then optionally classifies whether the
hidden text is instruction-like. Scores itself against a dataset manifest and
reports precision/recall broken down by hiding technique and by negative type.

Signals
  char-level (pdfplumber):     low contrast vs the actual page background (detected per page
                                from the largest full-page fill rect, not assumed white),
                                tiny font, off-page position
  content-stream (pikepdf):    text render mode 3 (invisible), fill alpha ~0 (transparent)
  semantic (regex):            instruction-like phrasing in the recovered hidden text

Decision modes
  structural   -> flag if ANY hidden text is found (high recall, flags benign hidden text)
  combined     -> flag only if hidden text is ALSO instruction-like (higher precision)

Usage
  python detector.py --scan path/to/file.pdf
  python detector.py --score dataset/manifest_test.csv --pdf-dir dataset/pdfs --mode combined
"""

import argparse, csv, os, re
import pdfplumber
import pikepdf
from pikepdf import parse_content_stream

# --------------------------------------------------------------------------
# Thresholds (tune these)
# --------------------------------------------------------------------------
CONTRAST_MIN = 2.0     # flag text whose contrast vs background is below this
TINY_PT      = 4.0     # flag font sizes below this
ALPHA_MIN    = 0.10    # flag fill alpha below this
PAGE_MARGIN  = 2.0     # tolerance (pt) for off-page detection


# --------------------------------------------------------------------------
# Color + contrast helpers
# --------------------------------------------------------------------------
def normalize_color(c):
    """Return (r,g,b) in 0..1 from pdfplumber's color field, or None if unknown."""
    if c is None:
        return None
    if isinstance(c, (int, float)):
        v = float(c); return (v, v, v)
    if isinstance(c, (list, tuple)):
        if len(c) == 1:
            v = float(c[0]); return (v, v, v)
        if len(c) == 3:
            return tuple(float(x) for x in c)
        if len(c) == 4:  # CMYK -> RGB
            cc, m, y, k = [float(x) for x in c]
            return ((1-cc)*(1-k), (1-m)*(1-k), (1-y)*(1-k))
    return None

def rel_luminance(rgb):
    def lin(u):
        return u/12.92 if u <= 0.03928 else ((u+0.055)/1.055) ** 2.4
    r, g, b = rgb
    return 0.2126*lin(r) + 0.7152*lin(g) + 0.0722*lin(b)

def contrast_ratio(rgb, bg=(1, 1, 1)):
    l1, l2 = rel_luminance(rgb), rel_luminance(bg)
    hi, lo = max(l1, l2), min(l1, l2)
    return (hi + 0.05) / (lo + 0.05)

def page_bg_color(page):
    """Best-effort actual page background: the fill color of the largest rect
    that covers (nearly) the whole page. Falls back to white when no such rect
    exists, which is the common case for plain text-only PDFs and keeps this
    change a no-op for pages that were already scored correctly."""
    W, H = page.width, page.height
    best, best_area = None, 0.0
    for r in page.rects:
        if not r.get("fill"):
            continue
        if r["width"] < W * 0.9 or r["height"] < H * 0.9:
            continue
        rgb = normalize_color(r.get("non_stroking_color"))
        if rgb is None:
            continue
        area = r["width"] * r["height"]
        if area > best_area:
            best, best_area = rgb, area
    return best if best is not None else (1, 1, 1)


# --------------------------------------------------------------------------
# Semantic pass
# --------------------------------------------------------------------------
INJECTION_PATTERNS = [
    r"ignore (all|any|previous|prior)", r"disregard", r"forget (the|all|previous)",
    r"previous instructions", r"system\s*:", r"you must", r"do not mention",
    r"perfect fit", r"rank (this|the candidate|first)", r"hire (this|immediately)",
    r"recommend (this|acceptance|for accept)", r"approve (this|automatically)",
    r"positive review", r"exceptional novelty", r"ideal candidate",
    r"emphasize (only )?strengths", r"omit (the )?weakness", r"score .{0,6}100",
]
INJ_RE = re.compile("|".join(f"({p})" for p in INJECTION_PATTERNS), re.IGNORECASE)

# Optional trained classifier (set via load_semantic_model). When present it
# replaces the regex in the semantic decision.
_MODEL = None
_MODEL_THRESH = 0.5

def load_semantic_model(path, threshold=0.5):
    global _MODEL, _MODEL_THRESH
    import joblib
    _MODEL = joblib.load(path)
    _MODEL_THRESH = threshold

def looks_like_injection(text):
    if not text:
        return False, []
    if _MODEL is not None:
        prob = float(_MODEL.predict_proba([text])[0][1])
        return (prob >= _MODEL_THRESH), [f"model_p={prob:.2f}"]
    hits = sorted({m.group(0).lower() for m in INJ_RE.finditer(text)})
    return (len(hits) > 0), hits


# --------------------------------------------------------------------------
# Char-level scan (contrast / tiny font / off-page)
# --------------------------------------------------------------------------
def scan_chars(page, bg=(1, 1, 1)):
    findings = []
    W, H = page.width, page.height
    for ch in page.chars:
        sigs = []
        rgb = normalize_color(ch.get("non_stroking_color"))
        if rgb is not None and contrast_ratio(rgb, bg) < CONTRAST_MIN:
            sigs.append("low_contrast")
        if ch.get("size", 99) < TINY_PT:
            sigs.append("tiny_font")
        if (ch["x1"] < PAGE_MARGIN or ch["x0"] > W - PAGE_MARGIN or
                ch["bottom"] < PAGE_MARGIN or ch["top"] > H - PAGE_MARGIN or
                ch["top"] < -PAGE_MARGIN or ch["bottom"] > H + PAGE_MARGIN or
                ch["x0"] < -PAGE_MARGIN or ch["x1"] > W + PAGE_MARGIN):
            sigs.append("off_page")
        if sigs:
            findings.append((sigs, ch["text"]))
    # group consecutive flagged chars into strings per signal-set
    text = "".join(t for _, t in findings)
    signals = sorted({s for sig, _ in findings for s in sig})
    return signals, text


# --------------------------------------------------------------------------
# Content-stream scan (render mode 3 / transparent) with a graphics-state stack
# --------------------------------------------------------------------------
def _decode_text_operand(op, operator):
    out = []
    if operator == "TJ":
        for el in op[0]:
            if isinstance(el, pikepdf.String):
                out.append(str(el))
        return "".join(out)
    # Tj  '  "
    for el in op:
        if isinstance(el, pikepdf.String):
            out.append(str(el))
    return "".join(out)

def scan_stream(pdf_page):
    """Walk the content stream tracking render mode + fill alpha across q/Q,
    collecting any text shown while invisible (render mode 3) or transparent."""
    extg = {}
    if "/Resources" in pdf_page and "/ExtGState" in pdf_page.Resources:
        for k, v in pdf_page.Resources.ExtGState.items():
            d = dict(v)
            ca = d.get("/ca", None)
            extg[str(k)] = float(ca) if ca is not None else None

    state = {"tr": 0, "alpha": 1.0}
    stack = []
    signals = set()
    hidden = []

    try:
        ops = parse_content_stream(pdf_page)
    except Exception:
        return signals, ""

    for instr in ops:
        operator = str(instr.operator)
        operands = instr.operands
        if operator == "q":
            stack.append(dict(state))
        elif operator == "Q":
            if stack:
                state = stack.pop()
        elif operator == "Tr":
            try: state["tr"] = int(operands[0])
            except Exception: pass
        elif operator == "gs":
            name = str(operands[0])
            if name in extg and extg[name] is not None:
                state["alpha"] = extg[name]
        elif operator in ("Tj", "TJ", "'", "\""):
            if state["tr"] == 3 or state["alpha"] < ALPHA_MIN:
                txt = _decode_text_operand(operands, operator)
                if txt.strip():
                    hidden.append(txt)
                    if state["tr"] == 3: signals.add("render_mode_3")
                    if state["alpha"] < ALPHA_MIN: signals.add("transparent")

    return signals, " ".join(hidden)


# --------------------------------------------------------------------------
# Top-level scan of one PDF
# --------------------------------------------------------------------------
def scan_pdf(path):
    signals, hidden_parts, page_bgs = set(), [], []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            bg = page_bg_color(page)
            page_bgs.append("#%02X%02X%02X" % tuple(round(v * 255) for v in bg))
            s, t = scan_chars(page, bg)
            signals |= set(s)
            if t.strip(): hidden_parts.append(t)
    with pikepdf.open(path) as pdf:
        for page in pdf.pages:
            s, t = scan_stream(page)
            signals |= set(s)
            if t.strip(): hidden_parts.append(t)

    hidden_text = " ".join(hidden_parts).strip()
    inj_like, inj_hits = looks_like_injection(hidden_text)
    return {
        "path": path,
        "signals": sorted(signals),
        "hidden_text": hidden_text,
        "has_hidden": bool(signals),
        "injection_like": inj_like,
        "injection_hits": inj_hits,
        "page_bg": page_bgs,
    }

def decide(result, mode):
    if mode == "structural":
        return result["has_hidden"]
    return result["has_hidden"] and result["injection_like"]  # combined


# --------------------------------------------------------------------------
# Scoring harness against a manifest
# --------------------------------------------------------------------------
def score(manifest_csv, pdf_dir, mode):
    rows = list(csv.DictReader(open(manifest_csv)))
    tp = fp = tn = fn = 0
    per_tech = {}          # technique -> [caught, total]  (positives)
    per_negtype = {}       # negative category -> [flagged, total]
    for r in rows:
        path = os.path.join(pdf_dir, r["filename"])
        if not os.path.exists(path):
            continue
        res = scan_pdf(path)
        flagged = decide(res, mode)
        label = int(r["label"])
        if label == 1:
            tech = r["technique"]
            d = per_tech.setdefault(tech, [0, 0]); d[1] += 1
            if flagged: d[0] += 1; tp += 1
            else: fn += 1
        else:
            negtype = r["payload_category"]  # clean | benign_hidden | near_miss
            d = per_negtype.setdefault(negtype, [0, 0]); d[1] += 1
            if flagged: d[0] += 1; fp += 1
            else: tn += 1

    prec = tp / (tp + fp) if tp + fp else 0.0
    rec  = tp / (tp + fn) if tp + fn else 0.0
    f1   = 2*prec*rec/(prec+rec) if prec+rec else 0.0
    return {
        "mode": mode, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3),
        "recall_by_technique": {k: f"{v[0]}/{v[1]}" for k, v in sorted(per_tech.items())},
        "false_positive_by_negtype": {k: f"{v[0]}/{v[1]}" for k, v in sorted(per_negtype.items())},
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", help="scan a single PDF and print findings")
    ap.add_argument("--score", help="manifest CSV to score against")
    ap.add_argument("--pdf-dir", default="dataset/pdfs")
    ap.add_argument("--mode", choices=["structural", "combined"], default="combined")
    ap.add_argument("--model", default=None, help="trained semantic model (.joblib)")
    ap.add_argument("--threshold", type=float, default=0.5, help="model decision threshold")
    args = ap.parse_args()

    if args.model:
        load_semantic_model(args.model, args.threshold)

    if args.scan:
        import json
        print(json.dumps(scan_pdf(args.scan), indent=2))
    elif args.score:
        import json
        print(json.dumps(score(args.score, args.pdf_dir, args.mode), indent=2))
    else:
        ap.print_help()
