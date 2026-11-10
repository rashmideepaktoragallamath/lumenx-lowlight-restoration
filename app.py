"""Gradio app.  Local: python app.py   |   Hugging Face Space (sdk: gradio) or Docker: see README.md."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2
import gradio as gr
import numpy as np
import pandas as pd
from lowlight import LowLightEnhancer
from lowlight.metrics import evaluate_pair
from lowlight.utils import sanitize_image, to_u8

MAX_UPLOAD_SIDE = 4096          # refuse absurd inputs before doing any work
enh = LowLightEnhancer()
HAS_DL = bool(enh.models)
SAMPLES = [os.path.join("samples", f) for f in ("street_before.png", "coffee_dark.png") if os.path.exists(os.path.join("samples", f))]


def _vis(x):
    x = np.asarray(x, np.float32)
    if x.ndim == 2:
        x = cv2.applyColorMap(to_u8(x), cv2.COLORMAP_INFERNO)[..., ::-1] / 255.0
    return to_u8(x)


def status_md():
    if not HAS_DL:
        return "**Deep-learning weights not found - running the classical pipeline only.**"
    if not enh.weights_trained:
        return ("**Note:** the loaded Zero-DCE++ weights are demo-only (not trained on LOL). "
                "DL-only output may over-brighten; the hybrid mode compensates.")
    ps = [m["val_psnr"] for m in enh.meta.values() if m.get("val_psnr") == m.get("val_psnr")]
    return "Zero-DCE++ weights: trained, best validation PSNR " + (f"{max(ps):.2f} dB." if ps else "n/a.")


def run(img, mode, max_side, progress=gr.Progress()):
    if img is None:
        raise gr.Error("Please upload an image first.")
    try:
        if max(img.shape[:2]) > MAX_UPLOAD_SIDE:
            raise gr.Error(f"Image is larger than {MAX_UPLOAD_SIDE}px on its longest side. Please downscale it.")
        x = sanitize_image(img, int(max_side))
    except ValueError as e:
        raise gr.Error(str(e))
    modes = ["classical", "dl", "hybrid"] if HAS_DL else ["classical"]
    results = {}
    for i, m in enumerate(modes):
        progress(i / len(modes), desc=f"Running {m} ...")
        try:
            results[m] = enh.enhance(x, m)
        except Exception as e:                       # one failing branch must not kill the demo
            results[m] = None
            print("mode failed:", m, repr(e))
    results = {k: v for k, v in results.items() if v is not None}
    if not results:
        raise gr.Error("Enhancement failed for this image. Try a smaller size or a different file.")
    chosen = "hybrid" if mode == "auto" else mode
    main = results.get(chosen) or results.get("hybrid") or results["classical"]

    rows = []
    for m, r in results.items():
        mt = evaluate_pair(x, r.output)
        rows.append({"method": m, "latency_ms": round(r.latency_ms, 1), "mean_lum": round(mt["mean_lum"], 3),
                     "overexposed_%": round(mt["overexposed_%"], 2), "colour_cast": round(mt["colour_cast"], 1),
                     "noise_gain": round(mt["noise_gain"], 2), "edge_corr": round(mt["edge_corr"], 3)})
    cmp_gallery = [(_vis(x), "input")] + [(_vis(r.output), m) for m, r in results.items()]
    inter = [(_vis(v), k) for k, v in main.intermediates.items()]
    st = main.stats
    report = (f"**Darkness:** {st.darkness_level} (mean luminance {st.mean_lum:.3f}) | noise sigma {st.noise_sigma:.4f} | "
              f"non-uniformity {st.nonuniformity:.2f}\n\n**Adaptive params:** "
              + ", ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in main.params.items())
              + ("\n\n" + "\n\n".join(main.info) if main.info else ""))
    return cmp_gallery, inter, pd.DataFrame(rows), report


with gr.Blocks(title="LUMEN-X Adaptive Low-Light Restoration") as demo:
    gr.Markdown("# LUMEN-X: adaptive low-light restoration\n"
                "Classical Retinex/CLAHE + Zero-DCE++ hybrid with quality-aware fusion. "
                "Upload a dark photo (JPG/PNG). Processing takes a few seconds on CPU.")
    gr.Markdown(status_md())
    with gr.Row():
        inp = gr.Image(type="numpy", label="Low-light image", image_mode="RGB")
        with gr.Column():
            mode = gr.Radio(["auto", "classical", "dl", "hybrid"], value="auto", label="Mode shown in detail below")
            ms = gr.Slider(256, 1536, value=768, step=64, label="Max side (px) - smaller is faster")
            btn = gr.Button("Enhance", variant="primary")
    if SAMPLES:
        gr.Examples(SAMPLES, inputs=inp, label="Sample inputs")
    report = gr.Markdown()
    g1 = gr.Gallery(label="Input vs classical / DL-only / hybrid", columns=4, height=380)
    tbl = gr.Dataframe(label="No-reference metrics and latency (no ground truth for uploads)")
    g2 = gr.Gallery(label="Intermediate maps (illumination, reflectance, blend weight, ...)", columns=4, height=380)
    btn.click(run, [inp, mode, ms], [g1, g2, tbl, report])

if __name__ == "__main__":
    demo.queue(max_size=8).launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
