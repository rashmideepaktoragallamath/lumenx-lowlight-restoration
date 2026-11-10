# Failure analysis
Observed on baseline v0 (demo weights); re-check each row after the final training run and replace with your own images from `results/final`.

| Failure | Where seen | Cause | Mitigation in code | Remaining risk |
|---|---|---|---|---|
| Over-exposure / blown highlights | DL-only on DICM: 31.5% new clipped pixels, mean luminance 0.33 to 0.70 | untrained curves push already-lit pixels towards 1 | blend map keeps classical output where L is high; quality score penalises new clipping; soft tone-map | trained DL-only may still clip near lamps |
| Colour cast (cyan sky, tinted shadows) | night-street sample | input already has B/R = 1.15 so shades-of-gray gain is about 1; Retinex and saturation boost amplify sky chroma | partial WB, colour-cast term in quality score | mixed light sources are ambiguous |
| Noise amplification in extreme dark | every mode has NIQE worse than the dark input on LOL; classical noise gain 1.56 | Retinex divides by small L, which multiplies sensor noise | pre-denoise scaled by noise sigma, gentle CLAHE, damped detail transfer | very high ISO leaves blotchy chroma noise |
| Weak adaptivity on LOL | `dce_iters` is 8 for all LOL inputs | all LOL inputs fall in the dark class; Immerkaer sigma on near-black luminance underestimates noise | none yet | estimate noise on a gamma-lifted luminance (proposed) |
| Quality selector not predictive | Spearman 0.00 vs PSNR on eval15 | hand-set heuristic weights | soft (not hard) selection | fit the weights on the validation split |
| No information in the input | motion blur, pure-black regions | enhancement cannot recover missing signal | none | out of scope |
