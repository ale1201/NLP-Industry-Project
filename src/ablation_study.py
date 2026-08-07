"""
Structural-signal ablation study (defensive research).

detector.py's structural pass combines five independent signals: low_contrast,
tiny_font, off_page, render_mode_3, transparent. DATASET_CREATION.md's technique
table claims each hiding technique has ONE designed detection signal (e.g.
white_on_white -> exact color match / contrast; tiny_font -> font size). This
script tests that claim directly instead of asserting it: scan every positive PDF
once, then post-process the collected signal set two ways --

  leave-one-out : would this PDF still be flagged with signal X removed?
                  (per-technique recall under each of the 5 ablations)
  keep-only-one : would this PDF be flagged using ONLY signal X, alone?
                  (which single signal is sufficient, per technique)

No re-scanning per ablation is needed -- scan_pdf() already returns the full signal
set per PDF, so both questions are answered by set arithmetic over one pass.

Usage
  python ablation_study.py --manifest ../dataset_check/manifest_all.csv \
      --pdf-dir ../dataset_check/pdfs --out ../results/ablation_study.json
"""

import argparse, csv, json, os
from detector import scan_pdf

ALL_SIGNALS = ["low_contrast", "tiny_font", "off_page", "render_mode_3", "transparent"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--pdf-dir", required=True)
    ap.add_argument("--out", default="../results/ablation_study.json")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.manifest)))
    pos_rows = [r for r in rows if int(r["label"]) == 1]
    neg_rows = [r for r in rows if int(r["label"]) == 0]
    print(f"scanning {len(pos_rows)} positive + {len(neg_rows)} negative PDFs...")

    # per-technique: list of signal sets (one per positive PDF)
    pos_signals_by_tech = {}
    for i, r in enumerate(pos_rows):
        path = os.path.join(args.pdf_dir, r["filename"])
        if not os.path.exists(path):
            continue
        result = scan_pdf(path)
        pos_signals_by_tech.setdefault(r["technique"], []).append(set(result["signals"]))
        if (i + 1) % 100 == 0:
            print(f"  positives: {i+1}/{len(pos_rows)}")

    # negatives: only benign_hidden matters here (clean PDFs have no signals to ablate)
    neg_signals_by_type = {}
    for i, r in enumerate(neg_rows):
        path = os.path.join(args.pdf_dir, r["filename"])
        if not os.path.exists(path):
            continue
        result = scan_pdf(path)
        neg_signals_by_type.setdefault(r["payload_category"], []).append(set(result["signals"]))
        if (i + 1) % 200 == 0:
            print(f"  negatives: {i+1}/{len(neg_rows)}")

    def recall_all_signals(sig_lists):
        return sum(1 for s in sig_lists if s) / len(sig_lists) if sig_lists else None

    def recall_leave_one_out(sig_lists, removed):
        hits = sum(1 for s in sig_lists if s - {removed})
        return round(hits / len(sig_lists), 3) if sig_lists else None

    def recall_keep_only_one(sig_lists, kept):
        hits = sum(1 for s in sig_lists if kept in s)
        return round(hits / len(sig_lists), 3) if sig_lists else None

    techniques = sorted(pos_signals_by_tech)

    baseline = {t: round(recall_all_signals(pos_signals_by_tech[t]), 3) for t in techniques}

    leave_one_out = {}
    for sig in ALL_SIGNALS:
        leave_one_out[sig] = {t: recall_leave_one_out(pos_signals_by_tech[t], sig) for t in techniques}

    keep_only_one = {}
    for sig in ALL_SIGNALS:
        keep_only_one[sig] = {t: recall_keep_only_one(pos_signals_by_tech[t], sig) for t in techniques}

    # false-positive side: same leave-one-out question for benign_hidden negatives
    fp_leave_one_out = {}
    if "benign_hidden" in neg_signals_by_type:
        bh = neg_signals_by_type["benign_hidden"]
        fp_leave_one_out["benign_hidden_baseline_fp_rate"] = round(recall_all_signals(bh), 3)
        for sig in ALL_SIGNALS:
            fp_leave_one_out[f"remove_{sig}"] = recall_leave_one_out(bh, sig)

    out = {
        "manifest": args.manifest,
        "n_positive": sum(len(v) for v in pos_signals_by_tech.values()),
        "n_negative": sum(len(v) for v in neg_signals_by_type.values()),
        "baseline_recall_by_technique_all_signals": baseline,
        "leave_one_out_recall": {
            "description": "recall per technique with the named signal REMOVED (other 4 kept)",
            **leave_one_out,
        },
        "keep_only_one_recall": {
            "description": "recall per technique using ONLY the named signal (other 4 ignored)",
            **keep_only_one,
        },
        "benign_hidden_false_positive_rate_leave_one_out": fp_leave_one_out,
    }
    print(json.dumps(out, indent=2))
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {args.out}")
