"""Compare Solution A (classical branch, and hybrid if weights exist) with Solution B (exposure fusion).

  python scripts/compare_solutions.py --low_dir data/LOL/eval15/low --high_dir data/LOL/eval15/high --out results/compare_AB
  python scripts/compare_solutions.py --low_dir samples --out results/compare_AB          (no ground truth)

Writes per_image.csv, summary.md (means per method) and <name>_AB.jpg strips (input | A | B [| ground truth])."""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "alternatives"))
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from exposure_fusion import enhance_fusion
from lowlight import LowLightEnhancer
from lowlight.metrics import evaluate_pair, psnr, ssim
from lowlight.utils import list_images, sanitize_image, to_u8


def rd(p):
    return sanitize_image(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--low_dir", required=True); ap.add_argument("--high_dir"); ap.add_argument("--out", default="results/compare_AB")
    ap.add_argument("--max_side", type=int, default=0)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    enh = LowLightEnhancer()
    gts = {p.name: p for p in list_images(a.high_dir)} if a.high_dir else {}
    rows = []
    for f in list_images(a.low_dir):
        x = sanitize_image(rd(f), a.max_side)
        gt = rd(gts[f.name]) if f.name in gts else None
        if gt is not None and gt.shape != x.shape:
            gt = cv2.resize(gt, (x.shape[1], x.shape[0]))
        outs = {}
        r = enh.enhance(x, "classical"); outs["A_classical"] = (r.output, r.latency_ms)
        if enh.models:
            r = enh.enhance(x, "hybrid"); outs["A_hybrid"] = (r.output, r.latency_ms)
        t = time.perf_counter(); y, _ = enhance_fusion(x); outs["B_fusion"] = (y, (time.perf_counter() - t) * 1000)
        for m, (o, ms) in outs.items():
            q = evaluate_pair(x, o)
            row = {"image": f.name, "method": m, "latency_ms": ms, "new_overexposed_%": q.get("new_overexposed_%", q.get("overexposed_%")),
                   "colour_cast": q["colour_cast"], "noise_gain": q["noise_gain"], "edge_corr": q["edge_corr"]}
            if gt is not None:
                row["PSNR"], row["SSIM"] = psnr(o, gt), ssim(o, gt)
            rows.append(row)
        panels = [x] + [o for o, _ in outs.values()] + ([gt] if gt is not None else [])
        cv2.imwrite(str(out / f"{f.stem}_AB.jpg"), cv2.cvtColor(to_u8(np.concatenate(panels, 1)), cv2.COLOR_RGB2BGR))
    if not rows:
        sys.exit("no images found")
    df = pd.DataFrame(rows); df.to_csv(out / "per_image.csv", index=False)
    summ = df.drop(columns="image").groupby("method").mean().round(3)
    (out / "summary.md").write_text(f"Images: {df.image.nunique()}\n\n" + summ.to_markdown() + "\n")
    print(summ.to_markdown())


if __name__ == "__main__":
    main()
