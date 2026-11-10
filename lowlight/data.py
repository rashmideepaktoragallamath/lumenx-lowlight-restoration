"""Datasets: unpaired low-light folder, paired (LOL-style low/high), or synthetic degradation of
normally exposed images (useful for pre-training / when no low-light data is available)."""
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from .classical import illumination_estimate
from .degrade import synth_low_light  # noqa: F401  (re-exported)
from .utils import EXT, list_images, read_rgb  # noqa: F401  (re-exported)

class LowLightDataset(Dataset):
    def __init__(self, low_dir, high_dir=None, crop=256, with_prior=True, synthetic=False, seed=0,
                 names=None, augment=True):
        """names: optional iterable of file names to keep (a split from splits.json).
        augment=False -> full image, no random crop / flip (used for validation and test)."""
        self.low = list_images(low_dir)
        if names is not None:
            keep = set(names)
            self.low = [p for p in self.low if p.name in keep]
        self.high = {p.name: p for p in list_images(high_dir)} if high_dir else {}
        self.crop, self.prior, self.synth, self.augment = crop, with_prior, synthetic, augment
        self.seed, self._rng_seed, self._rng = seed, None, None
        assert self.low, f"no images in {low_dir} (after split filter)"

    @property
    def rng(self):
        # One RNG per DataLoader worker and epoch. A single shared RNG would be copied into every
        # worker with the same state, giving identical crops/flips across workers.
        s = (torch.initial_seed() + self.seed) % (2 ** 32)
        if self._rng_seed != s:
            self._rng_seed, self._rng = s, np.random.default_rng(s)
        return self._rng

    def __len__(self):
        return len(self.low)

    def _crop(self, *imgs):
        if not self.augment:
            return [np.ascontiguousarray(i) for i in imgs]
        h, w = imgs[0].shape[:2]
        c = self.crop
        if min(h, w) < c:
            s = c / min(h, w) + 1e-3
            imgs = [cv2.resize(i, (int(w * s) + 1, int(h * s) + 1)) for i in imgs]
            h, w = imgs[0].shape[:2]
        y, x = self.rng.integers(0, h - c + 1), self.rng.integers(0, w - c + 1)
        out = [i[y:y + c, x:x + c] for i in imgs]
        if self.rng.random() < 0.5:
            out = [i[:, ::-1] for i in out]
        return [np.ascontiguousarray(i) for i in out]

    def __getitem__(self, i):
        p = self.low[i]
        img = read_rgb(p)
        gt = None
        if self.synth:
            (img,) = self._crop(img)
            gt, img = img, synth_low_light(img, self.rng)
        elif p.name in self.high:
            img, gt = self._crop(img, read_rgb(self.high[p.name]))
        else:
            (img,) = self._crop(img)
        item = {"low": torch.from_numpy(img.transpose(2, 0, 1)).float()}
        if self.prior:
            item["prior"] = torch.from_numpy(illumination_estimate(img)[None]).float()
        if gt is not None:
            item["gt"] = torch.from_numpy(gt.transpose(2, 0, 1)).float()
        return item
