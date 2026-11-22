"""Gaya grafik yang konsisten dan ramah buta warna (palet Okabe–Ito)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

PALETTE = {"normal": "#0072B2", "fraud": "#D55E00", "accent": "#009E73", "neutral": "#8C8C8C"}


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook", palette="colorblind")
    plt.rcParams.update({"figure.dpi": 110, "axes.titleweight": "bold", "axes.titlesize": 12})


def save_fig(fig_dir: Path, name: str) -> None:
    """Rapikan layout, simpan PNG ke `fig_dir`, lalu tampilkan."""
    plt.tight_layout()
    plt.savefig(Path(fig_dir) / f"{name}.png", dpi=130, bbox_inches="tight")
    plt.show()


def pct_axis(ax, decimals: int = 1, axis: str = "y") -> None:
    fmt = mtick.PercentFormatter(1, decimals=decimals)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def ref_line(ax, value: float, label: str = "rata-rata") -> None:
    ax.axhline(value, color=PALETTE["neutral"], ls="--", lw=1.2, label=label)
