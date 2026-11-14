"""Smoke / robustness tests that need no GPU and no weights:  python -m pytest -q   (or python tests/test_pipeline.py)"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pytest
from lowlight import LowLightEnhancer
from lowlight.degrade import synth_low_light
from lowlight.metrics import psnr, ssim
from lowlight.utils import sanitize_image, luminance

RNG = np.random.default_rng(0)
ENH = LowLightEnhancer(rgb_weights="__none__", hybrid_weights="__none__")   # classical-only, deterministic


def _scene(h=96, w=128):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.stack([0.5 + 0.4 * np.sin(xx / 9), 0.4 + 0.3 * np.cos(yy / 7), 0.5 + 0.2 * np.sin((xx + yy) / 11)], -1)
    img[30:50, 40:80] = [0.9, 0.8, 0.3]
    return np.clip(img, 0, 1).astype(np.float32)


def test_classical_brightens_and_improves_psnr():
    gt = _scene(); low = synth_low_light(gt, np.random.default_rng(1))
    out = ENH.enhance(low, "classical").output
    assert out.shape == low.shape and out.dtype == np.float32 and 0 <= out.min() and out.max() <= 1
    assert luminance(out).mean() > luminance(low).mean() * 1.5
    assert psnr(out, gt) > psnr(low, gt)


def test_adaptive_params_differ_between_images():
    gt = _scene()
    dark = np.clip(gt * 0.03, 0, 1); dim = np.clip(gt * 0.5, 0, 1)
    p1, p2 = ENH.enhance(dark, "classical").params, ENH.enhance(dim, "classical").params
    assert p1["gamma"] != p2["gamma"] and p1["dce_iters"] != p2["dce_iters"]


def test_hybrid_without_weights_falls_back_with_message():
    r = ENH.enhance(_scene() * 0.1, "hybrid")
    assert any("fell back" in m for m in r.info)


@pytest.mark.parametrize("bad", [
    np.zeros((64, 64), np.uint8),                         # grayscale
    np.zeros((64, 64, 4), np.uint8),                      # RGBA
    np.zeros((64, 64, 3), np.uint8),                      # pure black
    np.full((64, 64, 3), 255, np.uint8),                  # pure white
    (RNG.random((64, 64, 3)) * 65535).astype(np.uint16),  # 16-bit
    RNG.random((64, 64, 3)).astype(np.float64),           # float64
    np.zeros((9, 9, 3), np.uint8),                        # tiny
    np.full((64, 64, 3), np.nan, np.float32),             # NaNs
])
def test_bad_inputs_do_not_crash(bad):
    r = ENH.enhance(bad, "classical")
    assert np.isfinite(r.output).all() and r.output.ndim == 3 and r.output.shape[2] == 3


@pytest.mark.parametrize("bad", [None, np.zeros((4, 4, 3), np.uint8), np.zeros((5,), np.uint8)])
def test_invalid_inputs_raise_clear_error(bad):
    with pytest.raises(ValueError):
        sanitize_image(bad)


def test_noise_estimate_tracks_noise_in_very_dark_images():
    from lowlight.analysis import analyze, choose_params
    rng = np.random.default_rng(3)
    base = _scene() * 0.04
    levels = []
    for s in (0.0, 0.002, 0.005, 0.01):
        img = np.clip(base + rng.normal(0, s, base.shape), 0, 1).astype(np.float32)
        levels.append(choose_params(analyze(img))["noise_level"])
    assert levels == sorted(levels) and levels[-1] > levels[0] + 0.2   # adaptivity now responds


def test_exposure_fusion_solution_runs_and_brightens():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "alternatives"))
    from exposure_fusion import enhance_fusion
    low = _scene() * 0.05
    out, _ = enhance_fusion(low.astype(np.float32))
    assert out.shape == low.shape and np.isfinite(out).all() and luminance(out).mean() > luminance(low).mean() * 2


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
