"""Torch-free synthetic low-light degradation (used for training data and tests)."""
import numpy as np


def synth_low_light(img, rng):
    """Physically-motivated degradation: exposure drop + gamma + non-uniform illumination
    (random linear gradient / vignette) + signal-dependent shot noise + read noise + colour cast."""
    h, w = img.shape[:2]
    lin = np.power(img, 2.2)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = rng.uniform(0.2, 0.8) * w, rng.uniform(0.2, 0.8) * h
    vig = 1.0 - rng.uniform(0.0, 0.8) * np.clip(np.hypot(xx - cx, yy - cy) / np.hypot(w, h), 0, 1)
    scale = np.exp(rng.uniform(np.log(0.01), np.log(0.25)))
    cast = 1 + rng.uniform(-0.12, 0.12, 3).astype(np.float32)
    dark = lin * scale * vig[..., None] * cast
    peak = rng.uniform(30, 400)
    noisy = rng.poisson(dark * peak).astype(np.float32) / peak + rng.normal(0, rng.uniform(0.001, 0.01), dark.shape)
    return np.clip(np.power(np.clip(noisy, 0, 1), 1 / 2.2), 0, 1).astype(np.float32)
