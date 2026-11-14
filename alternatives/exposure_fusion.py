"""Solution B: training-free multi-exposure fusion (no neural network, no weights).

Idea: a dark photo contains the scene at several 'virtual exposures'. We synthesise a stack of brighter
versions of the (denoised) image, then fuse them per pixel with Mertens weights (contrast, saturation,
well-exposedness), so every region is taken from the exposure where it looks best. Dark regions come from
strongly brightened copies, already-lit regions from the original, which avoids blown highlights.

  python alternatives/exposure_fusion.py --input_dir samples --output_dir results_B
Shares only helpers (denoise, guided filter, image statistics) with Solution A; the enhancement logic is different.
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import cv2
import numpy as np
from lowlight import analysis, classical
from lowlight.utils import list_images, sanitize_image, to_u8, to_float, luminance


def exposure_gains(img, n=5, target=0.45, max_gain=40.0):
    """Adaptive exposure ladder: the brightest virtual exposure lifts the median luminance to ~target."""
    med = max(float(np.median(luminance(img))), 0.01)
    top = float(np.clip(target / med, 1.0, max_gain))
    return np.geomspace(1.0, top, n) if top > 1.0 else np.array([1.0])


def enhance_fusion(img):
    st = analysis.analyze(img)
    p = analysis.choose_params(st)
    clean = classical.denoise(img, p["denoise_strength"], p["denoise_eps"])
    gains = exposure_gains(clean)
    stack = [to_u8(np.power(np.clip(clean * g, 0, 1), 1 / (1 + 0.15 * np.log2(g + 1e-6) if g > 1 else 1.0))) for g in gains]
    fused = cv2.createMergeMertens(1.0, 1.0, 1.0).process(stack)
    out = np.clip(fused, 0, 1).astype(np.float32)
    out = classical.shades_of_gray(out, strength=p["wb_strength"])
    return np.clip(out, 0, 1), {"stats": st, "gains": gains}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--input_dir", required=True); ap.add_argument("--output_dir", default="results_B")
    a = ap.parse_args(); out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    for f in list_images(a.input_dir):
        raw = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        if raw is None:
            print("SKIP", f.name); continue
        x = sanitize_image(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB) if raw.ndim == 3 and raw.shape[2] >= 3 else raw, 1024)
        y, info = enhance_fusion(x)
        cv2.imwrite(str(out / f"{f.stem}_fusion.png"), cv2.cvtColor(to_u8(y), cv2.COLOR_RGB2BGR))
        print(f.name, "gains", np.round(info["gains"], 1).tolist())


if __name__ == "__main__":
    main()
