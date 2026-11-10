"""Batch / CLI inference (the command used in the report template).

  python run_inference.py --input_dir ./samples/ --output_dir ./results/ [--mode hybrid] [--max_side 1024]

Writes <name>_<mode>.png per image plus <name>_compare.jpg (input | classical | dl | hybrid) and
<name>_params.json (image statistics + the adaptive parameters that were chosen). Bad files are skipped and listed."""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
import cv2
import numpy as np
from lowlight import LowLightEnhancer
from lowlight.utils import list_images
from lowlight.utils import sanitize_image, to_u8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True); ap.add_argument("--output_dir", default="results")
    ap.add_argument("--mode", default="hybrid", choices=["classical", "dl", "hybrid", "auto"])
    ap.add_argument("--max_side", type=int, default=1024, help="0 = keep original size")
    ap.add_argument("--compare", action="store_true", help="also write input|classical|dl|hybrid strips")
    a = ap.parse_args()
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    enh = LowLightEnhancer()
    files = list_images(a.input_dir)
    if not files:
        sys.exit(f"no images found in {a.input_dir}")
    bad = []
    for f in files:
        try:
            raw = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
            if raw is None:
                raise ValueError("unreadable image file")
            raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB) if raw.ndim == 3 and raw.shape[2] >= 3 else raw
            x = sanitize_image(raw, a.max_side)
            r = enh.enhance(x, a.mode)
        except Exception as e:
            bad.append((f.name, str(e))); print("SKIP", f.name, "-", e); continue
        cv2.imwrite(str(out / f"{f.stem}_{a.mode}.png"), cv2.cvtColor(to_u8(r.output), cv2.COLOR_RGB2BGR))
        (out / f"{f.stem}_params.json").write_text(json.dumps(
            {"mode": r.mode, "latency_ms": round(r.latency_ms, 1), "stats": r.stats.as_dict(),
             "params": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in r.params.items()},
             "info": r.info}, indent=1))
        if a.compare:
            panels = [x] + [enh.enhance(x, m).output for m in (["classical", "dl", "hybrid"] if enh.models else ["classical"])]
            cv2.imwrite(str(out / f"{f.stem}_compare.jpg"), cv2.cvtColor(to_u8(np.concatenate(panels, 1)), cv2.COLOR_RGB2BGR))
        print(f"{f.name}: {r.mode} {r.latency_ms:.0f} ms")
    print(f"done: {len(files) - len(bad)} ok, {len(bad)} skipped")


if __name__ == "__main__":
    main()
