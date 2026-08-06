"""
PDF prompt-injection DATASET builder (defensive research).

Builds on pdf_injection_gen.py. Adds:
  1. Multiple carrier-document templates (variety on the "clean content" axis).
  2. Payload loading from a JSON file (swap in public sets like deepset/prompt-injections).
  3. Cross-product SAMPLING with class-balance control (realistic base rate).
  4. GROUP-AWARE train/val/test splits (no payload or template leaks across splits).
  5. A datasheet + balance report describing the dataset.

Usage:
    python dataset_builder.py --out ./dataset --n-positive 600 --neg-ratio 4 --seed 13
    # neg-ratio 4  ->  ~1 positive : 4 negative  (tune toward the ~1% real base rate for TEST)
"""

import argparse, csv, json, os, random, hashlib
from dataclasses import dataclass, asdict, field
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4

# Reuse the validated hiding techniques + payload schema from the base generator.
from pdf_injection_gen import (TECHNIQUES, PAYLOADS, Payload, BENIGN_HIDDEN,
                               contrasting_text_color, random_bg_color, blend)

PAGE_W, PAGE_H = letter


# --------------------------------------------------------------------------
# 1. Carrier-document templates. Each draws a plausible clean document.
#    Variety here forces the detector to learn the hidden-text signal rather
#    than memorizing one layout.
# --------------------------------------------------------------------------
def _fill_bg(c, bg):
    """Paint the whole page bg. A no-op-looking white fill when bg is white,
    so old (white-background) samples render byte-for-byte the same as before."""
    c.setFillColorRGB(*bg)
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)

def _section(c, title, lines, x, y, fg, sub, body_size=10.5):
    c.setFillColorRGB(*fg); c.setFont("Helvetica-Bold", 12)
    c.drawString(x, y, title)
    c.setFont("Helvetica", body_size); c.setFillColorRGB(*sub)
    for i, ln in enumerate(lines):
        c.drawString(x, y - 16 - i * 14, ln)
    return y - 16 - len(lines) * 14 - 16

def tmpl_resume(c, bg=(1, 1, 1)):
    _fill_bg(c, bg)
    fg = contrasting_text_color(bg)
    sub = blend(fg, bg, 0.3)
    c.setFillColorRGB(*fg); c.setFont("Helvetica-Bold", 20)
    c.drawString(60, PAGE_H - 80, "Jordan Avery")
    c.setFont("Helvetica", 11); c.setFillColorRGB(*sub)
    c.drawString(60, PAGE_H - 100, "Senior Software Engineer \u2022 jordan.avery@email.com")
    y = PAGE_H - 140
    y = _section(c, "Experience", ["Backend Engineer, Acme Corp (2021\u20132024)",
                                   "Built distributed services at scale."], 60, y, fg, sub)
    _section(c, "Skills", ["Python, Go, Kubernetes, PostgreSQL"], 60, y, fg, sub)

def tmpl_paper(c, bg=(1, 1, 1)):
    _fill_bg(c, bg)
    fg = contrasting_text_color(bg)
    sub = blend(fg, bg, 0.3)
    c.setFillColorRGB(*fg); c.setFont("Helvetica-Bold", 15)
    c.drawCentredString(PAGE_W/2, PAGE_H - 90, "Efficient Retrieval for Long-Context Models")
    c.setFont("Helvetica-Oblique", 10); c.setFillColorRGB(*sub)
    c.drawCentredString(PAGE_W/2, PAGE_H - 110, "A. Researcher, B. Scholar   \u2014   Preprint")
    c.setFont("Helvetica-Bold", 11); c.setFillColorRGB(*fg)
    c.drawString(60, PAGE_H - 150, "Abstract")
    c.setFont("Helvetica", 10); c.setFillColorRGB(*sub)
    body = ("We present a method for retrieval-augmented generation that reduces "
            "latency while preserving answer quality across long documents. "
            "Experiments on standard benchmarks show consistent improvements.")
    import textwrap
    for i, ln in enumerate(textwrap.wrap(body, 92)):
        c.drawString(60, PAGE_H - 170 - i * 14, ln)

def tmpl_invoice(c, bg=(1, 1, 1)):
    _fill_bg(c, bg)
    fg = contrasting_text_color(bg)
    sub = blend(fg, bg, 0.3)
    c.setFillColorRGB(*fg); c.setFont("Helvetica-Bold", 18)
    c.drawString(60, PAGE_H - 80, "INVOICE  #2041")
    c.setFont("Helvetica", 10.5); c.setFillColorRGB(*sub)
    c.drawString(60, PAGE_H - 105, "Bill to: Northwind Traders")
    c.drawString(60, PAGE_H - 120, "Date: 2026-07-15   Due: 2026-08-15")
    _section(c, "Line items", ["Consulting services .......... $4,200",
                               "Support retainer ............. $1,000",
                               "Total ........................ $5,200"], 60, PAGE_H - 160, fg, sub)

def tmpl_cover_letter(c, bg=(1, 1, 1)):
    _fill_bg(c, bg)
    fg = contrasting_text_color(bg)
    c.setFont("Helvetica", 10.5); c.setFillColorRGB(*fg)
    c.drawString(60, PAGE_H - 80, "Dear Hiring Manager,")
    import textwrap
    body = ("I am writing to express my interest in the Senior Engineer role. "
            "Over the past decade I have led backend teams and shipped reliable, "
            "high-throughput systems. I would welcome the chance to contribute.")
    for i, ln in enumerate(textwrap.wrap(body, 90)):
        c.drawString(60, PAGE_H - 110 - i * 15, ln)
    c.drawString(60, PAGE_H - 200, "Sincerely,")
    c.drawString(60, PAGE_H - 216, "Jordan Avery")

TEMPLATES = {
    "resume": tmpl_resume,
    "paper": tmpl_paper,
    "invoice": tmpl_invoice,
    "cover_letter": tmpl_cover_letter,
}


# --------------------------------------------------------------------------
# 2. Payload loading. Falls back to the built-in set; accepts a JSON file so
#    you can drop in strings exported from public datasets.
#    Expected JSON: [{"id","text","category","severity","explicit"}, ...]
# --------------------------------------------------------------------------
def load_payloads(path=None):
    if not path:
        return list(PAYLOADS)
    with open(path) as f:
        raw = json.load(f)
    out = []
    for i, r in enumerate(raw):
        out.append(Payload(
            id=r.get("id", f"ext_{i}"),
            text=r["text"],
            category=r.get("category", "unknown"),
            severity=r.get("severity", "medium"),
            explicit=bool(r.get("explicit", False)),
        ))
    return out


# --------------------------------------------------------------------------
# 3. Placement variety — a few positions so location isn't a constant.
# --------------------------------------------------------------------------
PLACEMENTS = {
    "header":  (60, PAGE_H - 92),
    "mid":     (60, PAGE_H - 300),
    "footer":  (60, 60),
}


# --------------------------------------------------------------------------
# 4. Sample record + PDF build
# --------------------------------------------------------------------------
@dataclass
class Sample:
    filename: str
    label: int
    template: str
    technique: str
    placement: str
    payload_id: str
    payload_category: str
    severity: str
    explicit: bool
    hidden_text: str
    split: str = ""

def build(path, template_fn, technique_fn, hidden_text, pos_xy, bg=(1, 1, 1)):
    c = canvas.Canvas(path, pagesize=letter)
    template_fn(c, bg)
    if technique_fn is not None and hidden_text is not None:
        # try richest signature first: (x, y, bg) -> (x, y) -> bare (offpage self-places)
        try:
            technique_fn(c, hidden_text, x=pos_xy[0], y=pos_xy[1], bg=bg)
        except TypeError:
            try:
                technique_fn(c, hidden_text, x=pos_xy[0], y=pos_xy[1])
            except TypeError:
                technique_fn(c, hidden_text)
    c.showPage(); c.save()


# --------------------------------------------------------------------------
# 5. Build the dataset by sampling the cross-product with balance control.
# --------------------------------------------------------------------------
def make_dataset(out_dir, n_positive, neg_ratio, payload_file, seed):
    random.seed(seed)
    pdf_dir = os.path.join(out_dir, "pdfs"); os.makedirs(pdf_dir, exist_ok=True)
    payloads = load_payloads(payload_file)
    tmpl_names = list(TEMPLATES)
    tech_names = list(TECHNIQUES)
    place_names = list(PLACEMENTS)

    samples = []

    # techniques that hide text by matching an arbitrary page background need
    # that background to actually be colored; every other technique keeps the
    # plain white page it was designed against.
    bg_matching_techs = {"bg_color_match", "bg_color_match_tiny"}
    def pick_bg(tech):
        return random_bg_color() if tech in bg_matching_techs else (1, 1, 1)

    # ---- positives: sample (template, payload, technique, placement) ----
    for i in range(n_positive):
        tmpl = random.choice(tmpl_names)
        pl = random.choice(payloads)
        tech = random.choice(tech_names)
        place = random.choice(place_names)
        fn = f"pos_{i:05d}.pdf"
        build(os.path.join(pdf_dir, fn), TEMPLATES[tmpl], TECHNIQUES[tech],
              pl.text, PLACEMENTS[place], bg=pick_bg(tech))
        samples.append(Sample(fn, 1, tmpl, tech, place, pl.id, pl.category,
                              pl.severity, pl.explicit, pl.text))

    # ---- negatives: mix of clean, hard (benign hidden), near-miss ----
    n_negative = n_positive * neg_ratio
    hard_techs = ["white_on_white", "tiny_font", "invisible_render_mode",
                  "bg_color_match", "bg_color_match_tiny"]
    near_miss_visible = [
        "Please review my qualifications carefully and consider me a strong fit.",
        "I believe I am the ideal candidate for this position.",
    ]
    for i in range(n_negative):
        tmpl = random.choice(tmpl_names)
        r = random.random()
        fn = f"neg_{i:05d}.pdf"
        if r < 0.55:                        # clean
            build(os.path.join(pdf_dir, fn), TEMPLATES[tmpl], None, None, None)
            samples.append(Sample(fn, 0, tmpl, "none", "none", "none", "clean", "n/a", False, ""))
        elif r < 0.85:                      # hard negative: benign hidden text
            tech = random.choice(hard_techs); place = random.choice(place_names)
            txt = random.choice(BENIGN_HIDDEN)
            build(os.path.join(pdf_dir, fn), TEMPLATES[tmpl], TECHNIQUES[tech], txt, PLACEMENTS[place],
                  bg=pick_bg(tech))
            samples.append(Sample(fn, 0, tmpl, tech, place, "none", "benign_hidden", "n/a", False, txt))
        else:                               # near-miss: injection-like but VISIBLE
            def draw_visible(c, text, x=60, y=PAGE_H-260):
                c.setFillColorRGB(0.27,0.27,0.27); c.setFont("Helvetica",10.5); c.drawString(x,y,text)
            txt = random.choice(near_miss_visible)
            build(os.path.join(pdf_dir, fn), TEMPLATES[tmpl], draw_visible, txt, (60, PAGE_H-260))
            samples.append(Sample(fn, 0, tmpl, "visible", "mid", "none", "near_miss", "n/a", False, txt))

    # --------------------------------------------------------------------
    # 6. GROUP-AWARE split: a payload or template must not straddle splits.
    #    Group key = payload_id for positives; template for negatives with no
    #    payload. Assign each group deterministically to train/val/test.
    # --------------------------------------------------------------------
    def group_key(s):
        return s.payload_id if s.payload_id not in ("none",) else f"tmpl:{s.template}:{s.payload_category}"

    def bucket(key):
        h = int(hashlib.md5(f"{seed}:{key}".encode()).hexdigest(), 16) % 100
        if h < 70: return "train"
        if h < 85: return "val"
        return "test"

    for s in samples:
        s.split = bucket(group_key(s))

    # --------------------------------------------------------------------
    # 7. Write per-split manifests + combined + datasheet
    # --------------------------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)
    fields = list(asdict(samples[0]).keys())
    def write_csv(path, rows):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
            for s in rows: w.writerow(asdict(s))

    write_csv(os.path.join(out_dir, "manifest_all.csv"), samples)
    for sp in ("train", "val", "test"):
        write_csv(os.path.join(out_dir, f"manifest_{sp}.csv"),
                  [s for s in samples if s.split == sp])

    # balance report
    def counts(rows):
        pos = sum(1 for s in rows if s.label == 1)
        return {"n": len(rows), "pos": pos, "neg": len(rows) - pos,
                "pos_rate": round(pos / max(1, len(rows)), 3)}
    report = {
        "seed": seed, "n_positive": n_positive, "neg_ratio": neg_ratio,
        "templates": tmpl_names, "techniques": tech_names, "placements": place_names,
        "n_payloads": len(payloads),
        "overall": counts(samples),
        "splits": {sp: counts([s for s in samples if s.split == sp])
                   for sp in ("train", "val", "test")},
        "positives_by_technique": {
            t: sum(1 for s in samples if s.label == 1 and s.technique == t) for t in tech_names},
        "negatives_by_type": {
            k: sum(1 for s in samples if s.label == 0 and s.payload_category == k)
            for k in ("clean", "benign_hidden", "near_miss")},
    }
    with open(os.path.join(out_dir, "datasheet.json"), "w") as f:
        json.dump(report, f, indent=2)

    # leakage assertion: no group key appears in more than one split
    seen = {}
    leaks = 0
    for s in samples:
        k = group_key(s)
        if k in seen and seen[k] != s.split: leaks += 1
        seen[k] = s.split
    report["leakage_groups"] = leaks

    return samples, report, pdf_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./dataset")
    ap.add_argument("--n-positive", type=int, default=600)
    ap.add_argument("--neg-ratio", type=int, default=4)
    ap.add_argument("--payloads", default=None, help="JSON file of payloads (optional)")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    samples, report, pdf_dir = make_dataset(
        args.out, args.n_positive, args.neg_ratio, args.payloads, args.seed)

    print(f"built {report['overall']['n']} PDFs -> {pdf_dir}")
    print(f"  overall: {report['overall']}")
    for sp, c in report["splits"].items():
        print(f"  {sp:5s}: {c}")
    print(f"  positives by technique: {report['positives_by_technique']}")
    print(f"  negatives by type: {report['negatives_by_type']}")
    print(f"  leakage groups across splits: {report['leakage_groups']}")
