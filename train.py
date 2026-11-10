"""Train / fine-tune Zero-DCE++ (RGB or hybrid 4-channel) with a proper train/validation protocol.

Recommended (leakage-safe LOL split from scripts/prepare_data.py -> data/splits.json):
  python train.py --data_root data/LOL --split_json data/splits.json --epochs 100 --hybrid          # 4-ch, with illumination prior
  python train.py --data_root data/LOL --split_json data/splits.json --epochs 100                   # 3-ch RGB

Stage 1 (optional, unpaired / non-reference losses only):  python train.py --low_dir data/low
Synthetic degradation of normal photos:                    python train.py --low_dir data/normal --synthetic
Resume / fine-tune:                                         --init weights/zerodce_rgb.pt

Model selection: the checkpoint written to --out is the one with the best VALIDATION PSNR (paired) or the
lowest validation loss (unpaired). The test split is never touched here. Smoke test: --epochs 1 --limit 8.
"""
import argparse, csv, json, math, os, sys, time
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lowlight.data import LowLightDataset
from lowlight.zerodce import DCENet, ZeroDCELoss


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def psnr_t(a, b):
    mse = ((a.clamp(0, 1) - b) ** 2).mean().item()
    return 10 * math.log10(1.0 / max(mse, 1e-10))


@torch.no_grad()
def validate(net, crit, dl, dev, hybrid):
    net.eval(); losses, psnrs = [], []
    for b in dl:
        low = b["low"].to(dev)
        x = torch.cat([low, b["prior"].to(dev)], 1) if hybrid else low
        gt = b["gt"].to(dev) if "gt" in b else None
        enh, A = net(x)
        loss, _ = crit(enh, low, A, gt)
        losses.append(loss.item())
        if gt is not None:
            psnrs.append(psnr_t(enh, gt))
    net.train()
    return float(np.mean(losses)), (float(np.mean(psnrs)) if psnrs else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    # data
    ap.add_argument("--data_root", help="LOL-style root containing our485/{low,high} (use with --split_json)")
    ap.add_argument("--split_json", help="data/splits.json from scripts/prepare_data.py")
    ap.add_argument("--low_dir"); ap.add_argument("--high_dir")
    ap.add_argument("--val_low_dir"); ap.add_argument("--val_high_dir", help="explicit validation folders (e.g. LOL-v2)")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N training images (smoke test)")
    # model / optimisation
    ap.add_argument("--hybrid", action="store_true"); ap.add_argument("--init"); ap.add_argument("--out")
    ap.add_argument("--epochs", type=int, default=100); ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--crop", type=int, default=256); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=2); ap.add_argument("--seed", type=int, default=0)
    # loss weights (supervised defaults differ from the Zero-DCE unpaired ones; tune on VALIDATION psnr)
    ap.add_argument("--l1_w", type=float, default=None); ap.add_argument("--ssim_w", type=float, default=None)
    ap.add_argument("--nonref_scale", type=float, default=None, help="scale of the 4 non-reference losses")
    ap.add_argument("--exp_target", type=float, default=None)
    a = ap.parse_args()
    set_seed(a.seed)

    if a.data_root:
        a.low_dir, a.high_dir = f"{a.data_root}/our485/low", f"{a.data_root}/our485/high"
    assert a.low_dir, "give --data_root (+ --split_json) or --low_dir"
    paired = bool(a.high_dir)
    tr_names = va_names = None
    if a.split_json:
        sp = json.load(open(a.split_json)); tr_names, va_names = sp["train"], sp["val"]
        assert not set(tr_names) & set(va_names), "train/val overlap"
        assert not set(tr_names) & set(sp["test"]), "train/test overlap"
    if a.limit and tr_names:
        tr_names = tr_names[:a.limit]

    # paired fine-tuning: keep the Zero-DCE exposure prior weak so it does not fight the ground truth
    l1_w = a.l1_w if a.l1_w is not None else (10.0 if paired else 0.0)
    ssim_w = a.ssim_w if a.ssim_w is not None else (2.0 if paired else 0.0)
    ns = a.nonref_scale if a.nonref_scale is not None else (0.1 if paired else 1.0)
    et = a.exp_target if a.exp_target is not None else (0.45 if paired else 0.6)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    in_ch = 4 if a.hybrid else 3
    out = a.out or f"weights/zerodce_{'hybrid' if a.hybrid else 'rgb'}.pt"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    net = DCENet(in_ch=in_ch).to(dev)
    if a.init:
        ck = torch.load(a.init, map_location=dev)
        if ck["in_ch"] == in_ch:
            net.load_state_dict(ck["state_dict"])
        else:
            print(f"--init has in_ch={ck['in_ch']} but model has {in_ch}: skipping init")

    ds = LowLightDataset(a.low_dir, a.high_dir, a.crop, with_prior=a.hybrid, synthetic=a.synthetic,
                         seed=a.seed, names=tr_names)
    if a.limit and not tr_names:
        ds.low = ds.low[:a.limit]
    dl = DataLoader(ds, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=len(ds) >= a.bs)

    vdl = None
    if va_names or a.val_low_dir:
        vds = LowLightDataset(a.val_low_dir or a.low_dir, a.val_high_dir or a.high_dir, a.crop, with_prior=a.hybrid,
                              synthetic=a.synthetic, seed=a.seed, names=None if a.val_low_dir else va_names,
                              augment=False)
        vdl = DataLoader(vds, batch_size=1, shuffle=False, num_workers=0)
    else:
        print("WARNING: no validation split given -> checkpoint = last epoch. Pass --split_json for model selection.")

    crit = ZeroDCELoss(w=(1.0, 10.0, 5.0, 200.0, l1_w, ssim_w), exp_target=et, nonref_scale=ns).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, a.epochs), eta_min=a.lr * 0.05)
    print(f"device={dev} train={len(ds)} val={len(vdl.dataset) if vdl else 0} paired={paired} "
          f"l1_w={l1_w} ssim_w={ssim_w} nonref_scale={ns} exp_target={et}")

    log_path = out.replace(".pt", "_log.csv")
    best, rows = None, []
    meta_base = {"in_ch": in_ch, "nf": 32, "n_iter": 8, "args": vars(a)}
    for ep in range(a.epochs):
        t0, tot = time.time(), 0.0
        for b in dl:
            low = b["low"].to(dev)
            x = torch.cat([low, b["prior"].to(dev)], 1) if a.hybrid else low
            enh, A = net(x)
            loss, parts = crit(enh, low, A, b["gt"].to(dev) if "gt" in b else None)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.1)
            opt.step(); tot += loss.item()
        sched.step()
        tr_loss = tot / max(1, len(dl))
        v_loss, v_psnr = validate(net, crit, vdl, dev, a.hybrid) if vdl else (float("nan"), float("nan"))
        # model selection on VALIDATION only
        score = v_psnr if (vdl and paired) else (-v_loss if vdl else -tr_loss)
        improved = best is None or score > best
        if improved:
            best = score
            torch.save({"state_dict": net.state_dict(), **meta_base, "epoch": ep + 1, "val_psnr": v_psnr,
                        "val_loss": v_loss}, out)
        torch.save({"state_dict": net.state_dict(), **meta_base, "epoch": ep + 1}, out.replace(".pt", "_last.pt"))
        rows.append({"epoch": ep + 1, "train_loss": tr_loss, "val_loss": v_loss, "val_psnr": v_psnr,
                     "best": int(improved), "sec": round(time.time() - t0, 1)})
        print(f"epoch {ep + 1}/{a.epochs} train {tr_loss:.4f} val_loss {v_loss:.4f} val_psnr {v_psnr:.2f}"
              f"{'  *saved best*' if improved else ''}  ({rows[-1]['sec']}s)")
        with open(log_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("best checkpoint:", out, "| training log:", log_path)


if __name__ == "__main__":
    main()
