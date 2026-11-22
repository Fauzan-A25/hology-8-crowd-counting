"""Data crowd counting Hology 8.0: label titik kepala, dataset PyTorch, dan peta kepadatan target."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
STRIDE = 8          # resolusi peta kepadatan = 1/8 gambar
DENSITY_SCALE = 100.0  # target dikali konstanta agar nilai per piksel tidak terlalu kecil untuk MSE


def find_data_dir(local: str | Path = "../data/raw") -> Path:
    """Folder yang berisi train/labels dan test/images (di /kaggle/input, ikuti symlink)."""
    for root, dirs, _ in os.walk("/kaggle/input", followlinks=True):
        if "train" in dirs and "test" in dirs and (Path(root) / "train" / "labels").exists():
            return Path(root)
    return Path(local)


def load_labels(data_dir: str | Path) -> pd.DataFrame:
    rows = []
    for f in sorted((Path(data_dir) / "train" / "labels").glob("*.json")):
        d = json.loads(f.read_text())
        pts = np.array([[p["x"], p["y"]] for p in d["points"]], dtype=np.float32).reshape(-1, 2)
        rows.append({"img_id": d["img_id"], "count": int(d["human_num"]), "points": pts})
    return pd.DataFrame(rows)


def resize_keep(img: Image.Image, pts: np.ndarray | None, max_side: int):
    """Perkecil gambar agar sisi terpanjang <= max_side (titik ikut diskalakan)."""
    w, h = img.size
    s = min(1.0, max_side / max(w, h))
    if s < 1.0:
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
        if pts is not None:
            pts = pts * s
    return img, pts


def density_map(pts: np.ndarray, h: int, w: int, sigma: float = 1.0) -> torch.Tensor:
    """Hitung titik per sel grid 1/STRIDE, lalu haluskan dengan Gaussian. Jumlah total = jumlah orang."""
    gh, gw = h // STRIDE, w // STRIDE
    grid = np.zeros((gh, gw), dtype=np.float32)
    if len(pts):
        x = np.clip((pts[:, 0] / STRIDE).astype(int), 0, gw - 1)
        y = np.clip((pts[:, 1] / STRIDE).astype(int), 0, gh - 1)
        np.add.at(grid, (y, x), 1.0)
    t = torch.from_numpy(grid)[None, None]
    r = int(3 * sigma) + 1
    k = torch.arange(-r, r + 1, dtype=torch.float32)
    g = torch.exp(-(k ** 2) / (2 * sigma ** 2)); g = g / g.sum()
    t = F.conv2d(F.pad(t, (r, r, 0, 0), mode="constant"), g.view(1, 1, 1, -1))
    t = F.conv2d(F.pad(t, (0, 0, r, r), mode="constant"), g.view(1, 1, -1, 1))
    return t[0]


def to_tensor(img: Image.Image) -> torch.Tensor:
    a = (np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(a.transpose(2, 0, 1).copy())


class CrowdDataset(Dataset):
    """train=True: skala acak, crop acak, flip. train=False: gambar utuh (di-pad ke kelipatan STRIDE)."""

    def __init__(self, img_dir, ids, points=None, train=False, crop=384, max_side=1024, sigma=1.0, seed=2025):
        self.img_dir, self.ids, self.points = Path(img_dir), list(ids), points
        self.train, self.crop, self.max_side, self.sigma = train, crop, max_side, sigma
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        img = Image.open(self.img_dir / self.ids[i]).convert("RGB")
        pts = None if self.points is None else self.points[i].copy()
        img, pts = resize_keep(img, pts, self.max_side)
        if not self.train:
            w, h = img.size
            x = to_tensor(img)
            ph, pw = (-h) % STRIDE, (-w) % STRIDE
            x = F.pad(x, (0, pw, 0, ph))
            return x, (len(pts) if pts is not None else -1), self.ids[i]
        s = self.rng.uniform(0.7, 1.3)
        w, h = img.size
        nw, nh = max(self.crop, round(w * s)), max(self.crop, round(h * s))
        sx, sy = nw / w, nh / h
        img = img.resize((nw, nh), Image.BILINEAR)
        pts = pts * np.array([sx, sy], dtype=np.float32)
        x0 = int(self.rng.integers(0, nw - self.crop + 1)); y0 = int(self.rng.integers(0, nh - self.crop + 1))
        img = img.crop((x0, y0, x0 + self.crop, y0 + self.crop))
        pts = pts - np.array([x0, y0], dtype=np.float32)
        pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] < self.crop) & (pts[:, 1] >= 0) & (pts[:, 1] < self.crop)]
        if self.rng.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            pts[:, 0] = self.crop - 1 - pts[:, 0]
        return to_tensor(img), density_map(pts, self.crop, self.crop, self.sigma) * DENSITY_SCALE, float(len(pts))
