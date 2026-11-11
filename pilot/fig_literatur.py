"""Sekil: EMGLab literaturunde bolme duzeyine gore raporlanan dogruluk + bizim deneylerimiz.

Cikti: literatur/sekil_bolme_dogruluk.png (300 dpi) ve .svg
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LIT = ROOT / "literatur" / "literatur_tablosu.csv"
RES = Path(__file__).resolve().parent / "results"

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"

ROWS = [("P", "Segment / pencere\n(rastgele)"), ("K", "Kayıt düzeyi"),
        ("H*", "Hasta-ayrık iddiası\n(çelişkili ya da doğrulanamıyor)"), ("H", "Hasta-ayrık"),
        ("?", "Bölme anlaşılamıyor")]


def best_acc(s: str) -> float | None:
    s = str(s)
    m = re.search(r"test\s*([0-9]+(?:\.[0-9]+)?)", s)
    if m:
        return float(m.group(1))
    nums = [float(x) for x in re.findall(r"[0-9]+(?:\.[0-9]+)?", s)]
    nums = [x for x in nums if 50 <= x <= 100]
    return max(nums) if nums else None


def main():
    df = pd.read_csv(LIT, sep=";", encoding="utf-8-sig")
    df = df[df.grup.str.startswith("EMGLab")].copy()
    counts = df.bolme_kodu.value_counts()
    df["acc"] = df.bildirilen_dogruluk.map(best_acc)
    df = df.dropna(subset=["acc"])

    ours = []  # (kod, dogruluk, etiket) -- nihai 3 tohumlu sonuclar (Tablo 2 ile ayni)
    t2 = ROOT / "makale" / "tablo2_deney_ozeti.csv"
    if t2.exists():
        t = pd.read_csv(t2, sep=";", encoding="utf-8-sig").set_index(["kurgu", "protokol"]).pencere_ort
        ours += [("P", 100 * t["1D-CNN: 3 sınıf", "pencere_rastgele"], "1D-CNN, 3 sınıf"),
                 ("H", 100 * t["1D-CNN: 3 sınıf", "hasta_grup"], "1D-CNN, 3 sınıf"),
                 ("P", 100 * t["ALSNet kurgusu: ALS–Sağlıklı", "rastgele_80_20 (ALSNet bolmesi)"], "ALSNet kurgusu"),
                 ("H", 100 * t["ALSNet kurgusu: ALS–Sağlıklı", "hasta_grup"], "ALSNet kurgusu")]

    ypos = {k: len(ROWS) - 1 - i for i, (k, _) in enumerate(ROWS)}
    fig, ax = plt.subplots(figsize=(8.6, 5.0), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    rng = np.random.default_rng(7)
    for k, _ in ROWS:
        sub = df[df.bolme_kodu == k]
        jit = rng.uniform(-0.16, 0.16, len(sub))
        ax.scatter(sub.acc, ypos[k] + 0.12 + jit * 0.6, s=46, color=BLUE, edgecolor=SURFACE, linewidth=1.5,
                   zorder=3, label="Yayımlanmış EMGLab çalışması" if k == "P" else None)
        if len(sub):
            med = sub.acc.median()
            ax.plot([med, med], [ypos[k] - 0.05, ypos[k] + 0.32], color=INK2, lw=1.5, zorder=2)

    # dikkat cekici calismalar
    notes = {"ALSNet: A Dilated": ("ALSNet 2022", -34, 11), "Revisiting the K-Fold": ("Putro 2023", 4, 23),
             "ALSGate": ("ALSGate 2026", 40, 11), "Automated Diagnosis of Neuromuscular Disorders using EMG": ("Cherifi 2023", 0, 11)}
    for key, (lab, dx, dy) in notes.items():
        hit = df[df.baslik.str.contains(key, regex=False)]
        if len(hit):
            r = hit.iloc[0]
            ax.annotate(lab, (r.acc, ypos[r.bolme_kodu] + 0.12), xytext=(dx, dy), textcoords="offset points",
                        ha="center", fontsize=8, color=INK2)

    # bizim deneyler: rastgele -> hasta-ayrik oklari
    for i in range(0, len(ours), 2):
        (k1, a1, lab), (k2, a2, _) = ours[i], ours[i + 1]
        off = -0.2 - 0.14 * (i // 2)
        ax.annotate("", xy=(a2, ypos[k2] + off), xytext=(a1, ypos[k1] + off),
                    arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.4, alpha=0.55,
                                    connectionstyle="arc3,rad=0.12"), zorder=2)
        for k, a in [(k1, a1), (k2, a2)]:
            ax.scatter(a, ypos[k] + off, marker="D", s=58, color=ORANGE, edgecolor=SURFACE, linewidth=1.5, zorder=4,
                       label="Bu çalışma (aynı model, aynı veri)" if i == 0 and k == k1 else None)
        right = i >= 2
        ax.annotate(f"{lab}: %{a1:.1f} → %{a2:.1f}", (a2, ypos[k2] + off), xytext=(8 if right else -8, -1),
                    textcoords="offset points", ha="left" if right else "right", va="center", fontsize=8, color=INK)

    ax.set_yticks([ypos[k] for k, _ in ROWS])
    ax.set_yticklabels([f"{lab}  (n={counts.get(k, 0)})" for k, lab in ROWS], fontsize=9, color=INK)
    ax.set_xlim(55, 106)
    ax.set_xticks(range(60, 101, 10))
    ax.spines["bottom"].set_bounds(55, 101)
    ax.set_ylim(-0.75, len(ROWS) - 0.45)
    ax.set_xlabel("Raporlanan doğruluk (%)", fontsize=9, color=INK2)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="x", colors=INK2, labelsize=8)
    ax.tick_params(axis="y", length=0)
    ax.set_title("EMGLab N2001 literatürü: bölme düzeyine göre raporlanan doğruluk", loc="left", fontsize=11, color=INK, pad=20)
    ax.text(0, 1.012, "n: kodlanan çalışma sayısı; noktalar sayısal doğruluk raporlayanlar (en yüksek değer). Gri çizgi: medyan.",
            transform=ax.transAxes, fontsize=7.5, color=INK2, va="bottom")
    leg = ax.legend(loc="lower left", fontsize=8, frameon=False)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    out = ROOT / "literatur" / "sekil_bolme_dogruluk"
    fig.savefig(f"{out}.png", dpi=300, facecolor=SURFACE)
    fig.savefig(f"{out}.svg", facecolor=SURFACE)
    print("yazildi:", out, "| nokta:", len(df), "| bizim:", len(ours))
    print(df.groupby("bolme_kodu").acc.describe().round(1))


if __name__ == "__main__":
    main()
