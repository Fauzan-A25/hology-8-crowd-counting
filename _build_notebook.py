"""Generate notebook riset Penyisihan Hology 8.0 2025 Data Mining (crowd counting) dan runner Kaggle (GPU)."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).parent
NB_NAME = "hology-8-crowd-counting.ipynb"
FINDINGS = json.loads((ROOT / "_findings.json").read_text()) if (ROOT / "_findings.json").exists() else {}


def T(key: str) -> str:
    return FINDINGS.get(key, "_(temuan diisi setelah run pertama)_")


nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
    "accelerator": "GPU",
}
cells: list = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip()))


md(
    f"""
# Menghitung Kerumunan dari Satu Foto: Peta Kepadatan vs Regresi Langsung

| | |
| --- | --- |
| **Kompetisi** | [Penyisihan Hology 8.0 2025 Data Mining](https://www.kaggle.com/competitions/penyisihan-hology-8-0-2025-data-mining) (Universitas Brawijaya, Kaggle, September 2025) |
| **Tujuan** | Memprediksi jumlah orang dalam setiap gambar kerumunan (metrik MAE) |
| **Data** | 1.900 gambar latih dengan koordinat kepala setiap orang, 500 gambar uji |
| **Penulis** | Fauzan Ahsanudin Alfikri · [Kaggle: fauzan2](https://www.kaggle.com/fauzan2) |
| **Tanggal** | Disusun ulang November 2026 (notebook asli 2025 hilang) |
| **Lingkungan** | Dijalankan penuh di Kaggle (GPU), ±1 jam |

**Pertanyaan penelitian.** Jumlah orang dalam satu gambar berkisar dari 1 sampai lebih dari 3.000. Apakah lebih baik
model langsung menebak satu angka, atau belajar "di mana" setiap orang berada lalu menjumlahkannya?

**Ringkasan temuan.**
{T("summary")}

**Alur:** 1. Setup → 2. Memahami data → 3. EDA → 4. Preprocessing & target → 5. Desain validasi → 6. Baseline →
7. Regresi jumlah langsung → 8. Peta kepadatan (CSRNet) → 9. Evaluasi → 10. Submission → 11. Kesimpulan
"""
)

md(
    """
## 1. Setup

Semua import, parameter, dan path ada di sel ini. Dataset dan pembentukan peta kepadatan ada di `src/data`, arsitektur di
`src/models/networks.py`, dan loop training di `src/models/train.py`. Jika GPU tidak tersedia, notebook berjalan dalam
mode uji kecil (subset data, satu epoch) agar alurnya tetap bisa diperiksa.
"""
)

code(
    """
import os
import shutil
import sys
import time
import warnings
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from PIL import Image
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader

SRC_ROOT = Path.cwd().parent
for r, _, f in os.walk("/kaggle/input", followlinks=True):
    if Path(r).name == "models" and "networks.py" in f:
        pkg = Path(r).parent
        if pkg.name == "src":
            SRC_ROOT = pkg.parent
        else:
            SRC_ROOT = Path("/tmp/project")
            shutil.copytree(pkg, SRC_ROOT / "src", dirs_exist_ok=True)
        break
sys.path.insert(0, str(SRC_ROOT))
from src.data.dataset import STRIDE, CrowdDataset, density_map, find_data_dir, load_labels, resize_keep
from src.models.networks import CSRNet, CountRegressor
from src.models.train import predict_counts, set_seed, train_model
from src.visualization.plots import PALETTE, pct_axis, save_fig, set_style

# ── Parameter eksperimen
SEED = 2025
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SMOKE = DEVICE.type != "cuda"          # tanpa GPU: uji alur dengan subset kecil
VAL_FRAC = 0.2
CROP = 128 if SMOKE else 384
MAX_SIDE = 384 if SMOKE else 1024     # sisi terpanjang gambar setelah resize
BATCH = 8
EPOCHS_REG = 1 if SMOKE else 25
EPOCHS_DENSITY = 1 if SMOKE else 70
LR = 1e-4
SIGMA = 1.0                            # lebar Gaussian dalam sel peta kepadatan (1 sel = 8 piksel)

DATA_DIR = find_data_dir(local="../data/raw")
OUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("../reports")
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings("ignore")
set_style()
set_seed(SEED)
C1, C2, C3, CN = PALETTE["normal"], PALETTE["fraud"], PALETTE["accent"], PALETTE["neutral"]
T0 = time.time()
print("DATA_DIR:", DATA_DIR, "| DEVICE:", DEVICE, torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "", "| SMOKE:", SMOKE)
"""
)

md(
    """
## 2. Memahami data

Setiap gambar latih punya file JSON berisi `human_num` (jumlah orang) dan `points`, yaitu koordinat kepala setiap orang.
Label titik ini lebih kaya daripada sekadar angka: model bisa belajar lokasi orang, bukan hanya jumlahnya.
"""
)

code(
    """
labels = load_labels(DATA_DIR)
sizes = [Image.open(DATA_DIR / "train" / "images" / i).size for i in labels.img_id]
labels["w"], labels["h"] = [s[0] for s in sizes], [s[1] for s in sizes]
test_ids = sorted(os.listdir(DATA_DIR / "test" / "images"), key=lambda s: int(Path(s).stem))
sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
assert (labels["count"] == labels.points.apply(len)).all()
print(f"Train: {len(labels):,} gambar, {labels['count'].sum():,} orang | Test: {len(test_ids)} gambar")
labels[["count", "w", "h"]].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.99]).T.style.format("{:,.1f}")
"""
)

md(
    """
## 3. Exploratory Data Analysis

### 3.1 Seberapa beragam jumlah orang per gambar?
"""
)

code(
    """
c = labels["count"].values
srt = np.sort(c)[::-1]
fig, ax = plt.subplots(1, 2, figsize=(15, 4.5))
ax[0].hist(c, bins=np.logspace(0, np.log10(c.max() + 1), 40), color=C1)
ax[0].set_xscale("log"); ax[0].set(xlabel="jumlah orang (skala log)", ylabel="gambar", title="Distribusi jumlah orang per gambar")
ax[0].axvline(np.median(c), color=C2, ls="--", label=f"median {np.median(c):.0f}"); ax[0].axvline(c.mean(), color=CN, ls=":", label=f"rata-rata {c.mean():.0f}"); ax[0].legend()
ax[1].plot(np.arange(1, len(c) + 1) / len(c), np.cumsum(srt) / srt.sum(), color=C2)
pct_axis(ax[1], 0); ax[1].xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1, 0))
ax[1].set(xlabel="gambar terpadat (persentil)", ylabel="porsi seluruh orang", title="Seberapa terkonsentrasi orang di gambar terpadat?")
save_fig(FIG_DIR, "01_count_distribution")
top10 = srt[: len(c) // 10].sum() / srt.sum()
print(f"10% gambar terpadat memuat {top10:.0%} dari seluruh orang; {np.mean(c <= 50):.0%} gambar berisi ≤ 50 orang")
"""
)

md("**Temuan.** " + T("counts"))

md("### 3.2 Seperti apa gambarnya?")

code(
    """
picks = [labels.iloc[(labels["count"] - q).abs().argsort().iloc[0]] for q in [5, 40, 300, 1500]]
fig, axes = plt.subplots(1, 4, figsize=(20, 4.6))
for ax, row in zip(axes, picks):
    img = Image.open(DATA_DIR / "train" / "images" / row.img_id)
    ax.imshow(img); ax.scatter(row.points[:, 0], row.points[:, 1], s=4, c=C2)
    ax.set_title(f"{row.img_id}: {row['count']} orang ({img.size[0]}×{img.size[1]})", fontsize=10); ax.axis("off")
save_fig(FIG_DIR, "02_examples")
"""
)

md("**Temuan.** " + T("examples"))

md(
    """
### 3.3 Ukuran kepala: seberapa besar variasi skala?

Jarak ke tetangga terdekat antar kepala adalah proksi ukuran kepala dalam piksel. Jika jarak ini sangat bervariasi, model
harus mengenali orang pada berbagai skala.
"""
)

code(
    """
nn_med = []
for pts in labels.points:
    if len(pts) >= 2:
        d, _ = cKDTree(pts).query(pts, k=2)
        nn_med.append(np.median(d[:, 1]))
    else:
        nn_med.append(np.nan)
labels["nn_px"] = nn_med
labels["nn_rel"] = labels.nn_px / np.maximum(labels.w, labels.h)
fig, ax = plt.subplots(1, 2, figsize=(15, 4.5))
ax[0].scatter(labels["count"], labels.nn_px, s=6, alpha=0.4, color=C1)
ax[0].set(xscale="log", yscale="log", xlabel="jumlah orang", ylabel="median jarak antar kepala (px)", title="Semakin padat, semakin kecil kepala")
ax[1].scatter(labels.w * labels.h / 1e6, labels["count"], s=6, alpha=0.4, color=C1)
ax[1].set(xscale="log", yscale="log", xlabel="luas gambar (megapiksel)", ylabel="jumlah orang", title="Resolusi gambar vs jumlah orang")
save_fig(FIG_DIR, "03_scale")
rho = labels[["count", "nn_px"]].corr("spearman").iloc[0, 1]
area_rho = pd.Series(labels.w * labels.h).corr(labels["count"], method="spearman")
print(f"Spearman jumlah vs jarak antar kepala: {rho:.2f} | jumlah vs luas gambar: {area_rho:.2f}")
print("Jarak antar kepala (px), persentil 10/50/90:", np.nanpercentile(labels.nn_px, [10, 50, 90]).round(1))
"""
)

md("**Temuan.** " + T("scale"))

md("### 3.4 Di bagian mana orang biasanya berada?")

code(
    """
H = np.zeros((50, 50))
for pts, w, h in zip(labels.points, labels.w, labels.h):
    if len(pts):
        hx, _, _ = np.histogram2d(np.clip(pts[:, 1] / h, 0, 0.999), np.clip(pts[:, 0] / w, 0, 0.999), bins=50, range=[[0, 1], [0, 1]])
        H += hx / len(pts)
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(H / H.sum(), cmap="cividis", extent=[0, 1, 1, 0])
ax.set(xlabel="posisi horizontal (relatif)", ylabel="posisi vertikal (relatif)", title="Sebaran posisi kepala (rata-rata per gambar)")
plt.colorbar(im, ax=ax, label="porsi")
save_fig(FIG_DIR, "04_spatial")
upper = H[:25].sum() / H.sum()
print(f"Porsi kepala di setengah atas gambar: {upper:.0%}")
"""
)

md("**Temuan.** " + T("spatial"))

md(
    """
## 4. Preprocessing dan target

| Temuan EDA | Keputusan |
| --- | --- |
| Gambar sampai 1920×1080, kepala bisa sekecil beberapa piksel (3.3) | sisi terpanjang dibatasi 1024 px agar muat di GPU tanpa membuang terlalu banyak detail |
| Jumlah orang dari 1 sampai 3.000+, distribusi sangat miring (3.1) | regresi langsung memprediksi log(1 + jumlah); peta kepadatan tidak perlu transformasi |
| Variasi skala kepala besar (3.3) | augmentasi skala acak 0,7–1,3×, crop acak 384 px, flip horizontal |
| Label berupa titik kepala | target peta kepadatan: titik dihitung per sel 8×8 px lalu dihaluskan Gaussian; jumlah seluruh peta = jumlah orang |
"""
)

code(
    """
row = picks[2]
img, pts = resize_keep(Image.open(DATA_DIR / "train" / "images" / row.img_id), row.points, MAX_SIDE)
w, h = img.size
dm = density_map(pts, h - h % STRIDE, w - w % STRIDE, SIGMA)[0].numpy()
fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
ax[0].imshow(img); ax[0].scatter(pts[:, 0], pts[:, 1], s=4, c=C2); ax[0].set_title(f"Titik kepala: {len(pts)} orang"); ax[0].axis("off")
ax[1].imshow(dm, cmap="cividis"); ax[1].set_title(f"Peta kepadatan target 1/8 resolusi (jumlah = {dm.sum():.1f})"); ax[1].axis("off")
save_fig(FIG_DIR, "05_density_target")
"""
)

md(
    """
## 5. Desain validasi

20% gambar latih disisihkan sebagai validasi, dibagi secara stratified berdasarkan kelompok jumlah orang agar gambar sangat
padat ada di kedua bagian. Semua model dibandingkan pada validasi yang sama. Model akhir untuk data uji adalah bobot dengan
MAE validasi terbaik.
"""
)

code(
    """
rng = np.random.default_rng(SEED)
bins = pd.qcut(labels["count"].rank(method="first"), 10, labels=False)
val_mask = np.zeros(len(labels), bool)
for b in range(10):
    idx = np.where(bins == b)[0]
    val_mask[rng.choice(idx, int(round(len(idx) * VAL_FRAC)), replace=False)] = True
tr_df, va_df = labels[~val_mask].reset_index(drop=True), labels[val_mask].reset_index(drop=True)
if SMOKE:
    tr_df, va_df = tr_df.sample(40, random_state=SEED).reset_index(drop=True), va_df.sample(20, random_state=SEED).reset_index(drop=True)
IMG_TR = DATA_DIR / "train" / "images"
ds_tr = CrowdDataset(IMG_TR, tr_df.img_id, list(tr_df.points), train=True, crop=CROP, max_side=MAX_SIDE, sigma=SIGMA, seed=SEED)
ds_va = CrowdDataset(IMG_TR, va_df.img_id, list(va_df.points), train=False, max_side=MAX_SIDE)
g = torch.Generator().manual_seed(SEED)
dl_tr = DataLoader(ds_tr, batch_size=BATCH, shuffle=True, num_workers=2, drop_last=True, generator=g)
dl_va = DataLoader(ds_va, batch_size=1, shuffle=False, num_workers=2)
pd.DataFrame({"gambar": [len(tr_df), len(va_df)], "median jumlah": [tr_df["count"].median(), va_df["count"].median()],
              "rata-rata jumlah": [tr_df["count"].mean(), va_df["count"].mean()], "maks": [tr_df["count"].max(), va_df["count"].max()]},
             index=["latih", "validasi"]).style.format("{:,.1f}")
"""
)

md(
    """
## 6. Baseline

Menebak angka yang sama untuk semua gambar. Untuk MAE, konstanta terbaik adalah **median** jumlah orang di data latih.
"""
)

code(
    """
yv = va_df["count"].values.astype(float)
RESULTS = {
    "konstanta: rata-rata": np.full(len(yv), tr_df["count"].mean()),
    "konstanta: median": np.full(len(yv), tr_df["count"].median()),
}
pd.Series({k: np.mean(np.abs(p - yv)) for k, p in RESULTS.items()}).to_frame("MAE validasi").style.format("{:.2f}")
"""
)

md("**Temuan.** " + T("baseline"))

md(
    """
## 7. Regresi jumlah langsung

Backbone VGG16-BN pra-latih ImageNet, global average pooling, lalu MLP yang memprediksi log(1 + jumlah). Model ini hanya
diberi tahu "berapa", tanpa informasi lokasi. Karena crop latih hanya memuat sebagian gambar, targetnya adalah jumlah orang
di dalam crop.
"""
)

code(
    """
set_seed(SEED)
t = time.time()
reg, hist_reg, _ = train_model(CountRegressor(pretrained=not SMOKE), dl_tr, dl_va, DEVICE, EPOCHS_REG, LR, kind="regression", eval_every=1 if SMOKE else 5)
_, p_reg, _, _ = predict_counts(reg, dl_va, DEVICE, "regression")
RESULTS["regresi jumlah langsung"] = p_reg
print(f"Regresi: {(time.time() - t) / 60:.1f} menit | MAE validasi {np.mean(np.abs(p_reg - yv)):.2f}")
del reg; torch.cuda.empty_cache()
"""
)

md("**Temuan.** " + T("regression"))

md(
    """
## 8. Peta kepadatan: CSRNet

Arsitektur CSRNet (Li dkk., 2018): 10 lapis konvolusi pertama VGG16-BN pra-latih, lalu 6 lapis konvolusi dilasi yang
memperluas *receptive field* tanpa mengecilkan resolusi. Keluarannya peta kepadatan 1/8 resolusi gambar, dan jumlah orang
adalah jumlah seluruh nilainya. Model belajar dari setiap titik kepala, sehingga satu gambar memberi ratusan sinyal latih,
bukan satu angka.
"""
)

code(
    """
set_seed(SEED)
t = time.time()
csr, hist_csr, best_csr = train_model(CSRNet(pretrained=not SMOKE), dl_tr, dl_va, DEVICE, EPOCHS_DENSITY, LR, kind="density", eval_every=1 if SMOKE else 2)
ids_va, p_csr, _, maps_va = predict_counts(csr, dl_va, DEVICE, "density")
_, p_csr_tta, _, _ = predict_counts(csr, dl_va, DEVICE, "density", tta=True)
RESULTS["CSRNet peta kepadatan"] = p_csr
RESULTS["CSRNet + flip TTA"] = p_csr_tta
TRAIN_MIN = (time.time() - t) / 60
print(f"CSRNet: {TRAIN_MIN:.1f} menit | MAE validasi terbaik {best_csr:.2f}")
"""
)

code(
    """
hc = pd.DataFrame(hist_csr); hr = pd.DataFrame(hist_reg)
fig, ax = plt.subplots(1, 2, figsize=(15, 4.3))
ax[0].plot(hc.epoch, hc.loss, color=C2); ax[0].set(xlabel="epoch", ylabel="loss MSE peta kepadatan", title="CSRNet: loss latih")
ax[1].plot(hc.dropna().epoch, hc.dropna().val_mae, marker="o", ms=3, color=C2, label="CSRNet")
ax[1].plot(hr.dropna().epoch, hr.dropna().val_mae, marker="s", ms=3, color=C1, label="regresi langsung")
ax[1].axhline(np.mean(np.abs(RESULTS["konstanta: median"] - yv)), color=CN, ls="--", label="konstanta median")
ax[1].set(xlabel="epoch", ylabel="MAE validasi", title="MAE validasi selama training", yscale="log"); ax[1].legend()
save_fig(FIG_DIR, "06_training_curves")
"""
)

md("**Temuan.** " + T("training"))

md("## 9. Evaluasi\n\n### 9.1 Perbandingan model")

code(
    """
res = pd.DataFrame({k: {"MAE": np.mean(np.abs(p - yv)), "RMSE": np.sqrt(np.mean((p - yv) ** 2)),
                        "median error relatif": np.median(np.abs(p - yv) / yv)} for k, p in RESULTS.items()}).T.sort_values("MAE")
FINAL_NAME = [k for k in res.index if k.startswith("CSRNet")][0]   # varian CSRNet terbaik dipakai untuk submission
res.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "median error relatif": "{:.1%}"}).highlight_min(subset=["MAE"], color="#CDE7F0")
"""
)

md("**Temuan.** " + T("results"))

md("### 9.2 Di mana error terjadi?")

code(
    """
p_fin = RESULTS[FINAL_NAME]
grp = pd.cut(yv, [0, 10, 50, 200, 1000, np.inf], labels=["1–10", "11–50", "51–200", "201–1000", ">1000"])
err = pd.DataFrame({"kelompok": grp, "abs": np.abs(p_fin - yv), "rel": np.abs(p_fin - yv) / yv, "reg": np.abs(RESULTS["regresi jumlah langsung"] - yv)})
by = err.groupby("kelompok", observed=True).agg(gambar=("abs", "size"), MAE=("abs", "mean"), MAE_regresi=("reg", "mean"), error_relatif=("rel", "median"), total_error=("abs", "sum"))
by["porsi total error"] = by.total_error / by.total_error.sum()
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
lim = [1, yv.max() * 1.3]
ax[0].scatter(yv, np.maximum(RESULTS["regresi jumlah langsung"], 0.5), s=10, alpha=0.5, color=C1, label="regresi langsung")
ax[0].scatter(yv, np.maximum(p_fin, 0.5), s=10, alpha=0.6, color=C2, label=FINAL_NAME)
ax[0].plot(lim, lim, color="black", lw=0.8); ax[0].set(xscale="log", yscale="log", xlabel="jumlah aktual", ylabel="jumlah prediksi", title="Prediksi vs aktual (validasi)"); ax[0].legend()
ax[1].bar(by.index.astype(str), by["porsi total error"], color=C2)
pct_axis(ax[1], 0); ax[1].set(xlabel="jumlah orang aktual", ylabel="porsi total error absolut", title="Dari mana MAE berasal?")
save_fig(FIG_DIR, "07_error_breakdown")
by.drop(columns="total_error").style.format({"MAE": "{:.1f}", "MAE_regresi": "{:.1f}", "error_relatif": "{:.1%}", "porsi total error": "{:.0%}"})
"""
)

md("**Temuan.** " + T("breakdown"))

md("### 9.3 Apa yang dilihat model?")

code(
    """
va_df["pred"] = p_fin
va_df["err"] = p_fin - yv
show = [va_df.iloc[(va_df["count"] - q).abs().argsort().iloc[0]] for q in [20, 150, 800]]
show.append(va_df.loc[va_df.err.abs().idxmax()])
fig, axes = plt.subplots(2, 4, figsize=(20, 8.5))
for j, row in enumerate(show):
    img, pts = resize_keep(Image.open(IMG_TR / row.img_id), row.points, MAX_SIDE)
    axes[0, j].imshow(img); axes[0, j].scatter(pts[:, 0], pts[:, 1], s=3, c=C2); axes[0, j].axis("off")
    axes[0, j].set_title(f"{row.img_id}: aktual {row['count']}" + (" (error terbesar)" if j == 3 else ""), fontsize=10)
    axes[1, j].imshow(maps_va[row.img_id], cmap="cividis"); axes[1, j].axis("off")
    axes[1, j].set_title(f"peta kepadatan prediksi: {row.pred:.0f}", fontsize=10)
save_fig(FIG_DIR, "08_density_examples")
"""
)

md("**Temuan.** " + T("density_examples"))

md(
    """
## 10. Submission

Varian CSRNet dengan MAE validasi terendah dipakai untuk memprediksi 500 gambar uji (gambar utuh, resize sisi terpanjang 1024 px).
"""
)

code(
    """
ds_te = CrowdDataset(DATA_DIR / "test" / "images", test_ids, None, train=False, max_side=MAX_SIDE)
dl_te = DataLoader(ds_te, batch_size=1, shuffle=False, num_workers=2)
ids_te, p_te, _, _ = predict_counts(csr, dl_te, DEVICE, "density", tta="TTA" in FINAL_NAME)
sub = sample[["image_id"]].merge(pd.DataFrame({"image_id": ids_te, "predicted_count": p_te}), on="image_id", how="left")
assert sub.predicted_count.notna().all() and len(sub) == len(sample)
sub.to_csv(OUT_DIR / "submission.csv", index=False)
res.to_csv(OUT_DIR / "validation_results.csv"); by.to_csv(OUT_DIR / "error_by_group.csv")
pd.DataFrame(hist_csr).to_csv(OUT_DIR / "history_csrnet.csv", index=False); pd.DataFrame(hist_reg).to_csv(OUT_DIR / "history_regression.csv", index=False)
fig, ax = plt.subplots(figsize=(9, 4))
bins_ = np.logspace(0, np.log10(max(c.max(), p_te.max()) + 1), 35)
ax.hist(c, bins=bins_, density=True, alpha=0.5, color=C1, label="train (aktual)")
ax.hist(np.maximum(p_te, 1), bins=bins_, density=True, alpha=0.5, color=C2, label="test (prediksi)")
ax.set(xscale="log", xlabel="jumlah orang", ylabel="densitas", title="Distribusi jumlah: train vs prediksi test"); ax.legend()
save_fig(FIG_DIR, "09_test_distribution")
print(f"Model: {FINAL_NAME} | prediksi test: median {np.median(p_te):.0f}, rata-rata {p_te.mean():.0f}, maks {p_te.max():.0f}")
"""
)

md("**Temuan.** " + T("test"))

md("## 11. Kesimpulan\n\n" + T("conclusion") + "\n\n### Jejak lingkungan")

code(
    """
import platform
pkgs = ["numpy", "pandas", "torch", "torchvision", "matplotlib", "seaborn"]
print(" | ".join([f"python {platform.python_version()}"] + [f"{p} {version(p)}" for p in pkgs]))
print("GPU:", torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "tidak ada")
print(f"Waktu eksekusi total: {(time.time() - T0) / 60:.1f} menit")
"""
)

nb["cells"] = cells
nb_path = ROOT / "kaggle-notebook" / NB_NAME
nb_path.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, nb_path)
print("ditulis:", nb_path)

src_files = {str(p.relative_to(ROOT)): p.read_text() for p in sorted((ROOT / "src").rglob("*.py"))}
runner = f'''"""Runner Kaggle untuk {NB_NAME} (di-generate oleh _build_notebook.py; jangan diedit manual)."""
import json, shutil, subprocess, sys
from pathlib import Path

SRC_FILES = {src_files!r}
NOTEBOOK = json.loads({nbf.writes(nb)!r})

proj = Path("/kaggle/working/project")
for rel, text in SRC_FILES.items():
    p = proj / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
(proj / "notebooks").mkdir(parents=True, exist_ok=True)
nb_in = proj / "notebooks" / "{NB_NAME}"
nb_in.write_text(json.dumps(NOTEBOOK))
cmd = [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", str(nb_in),
       "--output", "{NB_NAME}", "--output-dir", "/kaggle/working", "--ExecutePreprocessor.timeout=-1"]
ret = subprocess.run(cmd, cwd=proj / "notebooks")
shutil.rmtree(proj, ignore_errors=True)
sys.exit(ret.returncode)
'''
(ROOT / "kaggle" / "run_on_kaggle.py").write_text(runner)
print("ditulis:", ROOT / "kaggle" / "run_on_kaggle.py")
