# Two solutions to Problem P1

| | **Solution A: Hybrid (main)** | **Solution B: Exposure fusion (`alternatives/`)** |
|---|---|---|
| Core idea | Classical Retinex branch + Zero-DCE++ network, blended by illumination, quality-aware selection | Build a stack of virtual exposures from the denoised image, fuse per pixel with Mertens weights |
| Learning | Needs trained weights (Colab) | None; training-free, no weights |
| Adaptivity | Rule-based parameters, quality score | Exposure ladder sized from the image median; denoise and white balance from image statistics |
| Strength | Highest potential fidelity once trained; satisfies the deep-learning requirement | Fast, runs anywhere, no failure from missing weights, no halo-prone blending |
| Weakness | Weights must be trained; slower (~1 s CPU) | No learned prior, so heavy noise or extreme darkness is not recovered by learning; does not meet the deep-learning requirement alone |
| Code | `lowlight/`, `app.py` | `alternatives/exposure_fusion.py` |

Single-image smoke check (`samples/coffee_dark.png` against its ground truth, CPU, n = 1, so only indicative):
classical branch of A: PSNR 14.52, SSIM 0.363; Solution B: PSNR 17.40, SSIM 0.356; no new overexposure in either.
Run both on the full LOL eval15 with `evaluate.py` before drawing conclusions.

Run B: `python alternatives/exposure_fusion.py --input_dir samples --output_dir results_B`
