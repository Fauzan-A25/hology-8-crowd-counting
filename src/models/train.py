"""Loop training, evaluasi, dan prediksi untuk CSRNet dan CountRegressor."""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

from src.data.dataset import DENSITY_SCALE


def set_seed(seed: int = 2025):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


@torch.no_grad()
def predict_counts(model, loader, device, kind: str = "density", tta: bool = False):
    """Prediksi jumlah untuk setiap gambar (batch size 1, gambar utuh)."""
    model.eval()
    ids, preds, trues, maps = [], [], [], {}
    for x, y, img_id in loader:
        x = x.to(device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            out = model(x)
            if tta:
                out2 = model(torch.flip(x, dims=[3]))
        if kind == "density":
            c = out.float().sum().item() / DENSITY_SCALE
            if tta:
                c = (c + out2.float().sum().item() / DENSITY_SCALE) / 2
            maps[img_id[0]] = out[0, 0].float().cpu().numpy() / DENSITY_SCALE
        else:
            c = float(np.expm1(out.float().item()))
            if tta:
                c = (c + float(np.expm1(out2.float().item()))) / 2
        ids.append(img_id[0]); preds.append(max(c, 0.0)); trues.append(float(y[0]))
    return np.array(ids), np.array(preds), np.array(trues), maps


def train_model(model, train_loader, val_loader, device, epochs: int, lr: float = 1e-4, kind: str = "density",
                eval_every: int = 2, log=print):
    """Adam + cosine LR. Simpan bobot dengan MAE validasi terbaik."""
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    hist, best, best_state = [], np.inf, None
    for ep in range(1, epochs + 1):
        model.train(); t0 = time.time(); losses = []
        for x, dens, cnt in train_loader:
            x, dens, cnt = x.to(device), dens.to(device), cnt.to(device).float()
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                out = model(x)
                if kind == "density":
                    loss = F.mse_loss(out.float(), dens, reduction="sum") / x.size(0)
                else:
                    loss = F.l1_loss(out.float(), torch.log1p(cnt))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            losses.append(loss.item())
        sched.step()
        row = {"epoch": ep, "loss": float(np.mean(losses)), "detik": time.time() - t0}
        if ep % eval_every == 0 or ep == epochs:
            _, p, t, _ = predict_counts(model, val_loader, device, kind)
            row["val_mae"] = float(np.mean(np.abs(p - t)))
            if row["val_mae"] < best:
                best = row["val_mae"]; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            log(f"epoch {ep:3d} | loss {row['loss']:.4f} | MAE val {row['val_mae']:.2f} | {row['detik']:.0f}s")
        hist.append(row)
    model.load_state_dict(best_state)
    return model, hist, best
