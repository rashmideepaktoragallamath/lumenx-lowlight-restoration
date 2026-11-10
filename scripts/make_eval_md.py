"""Build the graded evaluation file  eval_<ProbID>_<TeamID>.md  straight from evaluate.py outputs
(so every number in it is reproducible from per_image.csv - nothing is typed by hand).

  python evaluate.py --low_dir data/LOL/eval15/low --high_dir data/LOL/eval15/high --ablation --max_side 0 --out results/final/lol_eval15
  python evaluate.py --low_dir data/unseen/DICM --out results/final/unseen_DICM          (repeat for LIME, ExDark100)
  python scripts/make_eval_md.py --lol results/final/lol_eval15 --unseen results/final/unseen_DICM results/final/unseen_LIME \\
         --usns 1RVU24CSE001,1RVU24CSE002 --team T01 --prob P1 --out eval_P1_T01.md
Line 1 of the output is the comma-separated USN list, as the instruction sheet requires."""
import argparse, os
import numpy as np
import pandas as pd

NAMES = {"classical": "Classical only", "dl_only": "Deep learning only", "hybrid": "Proposed hybrid"}


def fmt(x, d=2):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def mean_std(df, col):
    v = df[col].dropna()
    return (v.mean(), v.std()) if len(v) else (np.nan, np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lol", required=True, help="evaluate.py output dir on LOL eval15 (needs --high_dir and --ablation)")
    ap.add_argument("--unseen", nargs="*", default=[], help="evaluate.py output dirs on no-reference sets")
    ap.add_argument("--usns", required=True, help="comma separated, no spaces")
    ap.add_argument("--team", required=True); ap.add_argument("--prob", default="P1")
    ap.add_argument("--out"); ap.add_argument("--note", default="")
    a = ap.parse_args()
    out = a.out or f"eval_{a.prob}_{a.team}.md"
    df = pd.read_csv(os.path.join(a.lol, "per_image.csv"))
    n = df.image.nunique()
    L = [a.usns.strip(), "", f"# Evaluation - {a.prob} / {a.team}", "",
         f"Paired test set: LOL eval15 (n={n}), leakage-safe protocol (see docs/DESIGN.md). "
         "All values are means over images from `per_image.csv`; latency is wall-clock per image on the stated hardware.", ""]
    if a.note:
        L += [a.note, ""]
    L += ["## 1. Quantitative benchmarks (metrics in the order of the report template)", "",
          "| Approach | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | NIQE ↓ | Latency (ms/frame) ↓ |", "|---|---|---|---|---|---|"]
    for k in ("classical", "dl_only", "hybrid"):
        d = df[df.method == k]
        if d.empty:
            continue
        cells = [fmt(mean_std(d, c)[0], 3 if c in ("SSIM", "LPIPS") else 2) + (f" ± {fmt(mean_std(d, c)[1], 3 if c in ('SSIM', 'LPIPS') else 2)}" if c != "latency_ms" else "")
                 for c in ("PSNR", "SSIM", "LPIPS", "NIQE")]
        L.append(f"| {NAMES[k]} | " + " | ".join(cells) + f" | {fmt(mean_std(d, 'latency_ms')[0], 0)} |")
    inp = df[df.method == "input"]
    L += ["", f"Unprocessed dark input: PSNR {fmt(mean_std(inp, 'PSNR')[0])} dB, SSIM {fmt(mean_std(inp, 'SSIM')[0], 3)}, NIQE {fmt(mean_std(inp, 'NIQE')[0])}.", ""]
    if df[df.method == "hybrid"].LPIPS.isna().all():
        L += ["> LPIPS is n/a in this run (install `lpips` and allow the AlexNet weight download, then re-run evaluate.py).", ""]

    L += ["## 2. Artefact, noise and edge measures (LOL eval15)", "",
          "| Approach | new overexposure % ↓ | colour dE vs GT ↓ | noise gain ↓ | edge corr ↑ |", "|---|---|---|---|---|"]
    for k in ("classical", "dl_only", "hybrid"):
        d = df[df.method == k]
        if not d.empty:
            L.append(f"| {NAMES[k]} | " + " | ".join(fmt(mean_std(d, c)[0], 2) for c in ("new_overexposed_%", "dE_vs_ref", "noise_gain", "edge_corr")) + " |")
    L.append("")

    ab = df[df.method.str.startswith("abl_")]
    if not ab.empty:
        full = ab[ab.method == "abl_full"]
        L += ["## 3. Ablation (leave-one-out on the hybrid pipeline, LOL eval15)", "",
              "| Variant | PSNR | SSIM | NIQE | noise gain | new overexp % | dPSNR vs full |", "|---|---|---|---|---|---|---|"]
        fp = mean_std(full, "PSNR")[0]
        for m in ["abl_full"] + sorted(set(ab.method) - {"abl_full"}):
            d = ab[ab.method == m]
            L.append(f"| {m[4:]} | " + " | ".join(fmt(mean_std(d, c)[0], 3 if c == "SSIM" else 2) for c in ("PSNR", "SSIM", "NIQE", "noise_gain", "new_overexposed_%"))
                     + f" | {fmt(mean_std(d, 'PSNR')[0] - fp, 2)} |")
        L.append("")
    else:
        L += ["## 3. Ablation", "", "Not run (use `evaluate.py --ablation`).", ""]

    if a.unseen:
        L += ["## 4. Previously unseen images (no ground truth)", "",
              "| Set | n | Mode | NIQE ↓ | new overexp % ↓ | noise gain ↓ | edge corr ↑ | mean lum |", "|---|---|---|---|---|---|---|---|"]
        for u in a.unseen:
            du = pd.read_csv(os.path.join(u, "per_image.csv"))
            for k in ("input", "classical", "dl_only", "hybrid"):
                d = du[du.method == k]
                if not d.empty:
                    L.append(f"| {os.path.basename(u.rstrip('/'))} | {d.image.nunique()} | {k} | " + " | ".join(
                        fmt(mean_std(d, c)[0], 2) for c in ("NIQE", "new_overexposed_%", "noise_gain", "edge_corr", "mean_lum")) + " |")
        L.append("")
    open(out, "w").write("\n".join(L))
    print("wrote", out)


if __name__ == "__main__":
    main()
