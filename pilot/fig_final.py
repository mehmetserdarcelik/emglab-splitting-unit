"""Makale Sekil 2, Sekil 3 ve Tablo 2.

Sekil 2: ayni model ve gorevde bolme duzeyi -> pencere dogrulugu (3 tohum ort. +/- sd)
Sekil 3: hasta-ayrik LOSO'da hasta basina pencere dogrulugu
Tablo 2: tum kurgular; pencere / dengeli / kayit-oyu / hasta-oyu dogruluklari + Clopper-Pearson %95 GA
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
OUT = HERE.parent / "makale"
OUT.mkdir(exist_ok=True)

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
MARKERS = ["o", "s", "D", "^", "v", "P"]
PROTO = ["pencere_rastgele", "kayit_grup", "hasta_grup"]
PROTO_LAB = {"pencere_rastgele": "Segment düzeyi\n(rastgele)", "kayit_grup": "Kayıt gruplu",
             "hasta_grup": "Hasta gruplu", "hasta_LOSO": "Hasta LOSO"}
CLS_TR = {"ALS": "ALS", "Myopathy": "Miyopati", "Normal": "Sağlıklı", 0: "ALS", 1: "Miyopati", 2: "Sağlıklı"}


def cp(k, n, a=0.05):
    lo = beta.ppf(a / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.0
    return lo, hi


def scores(d: pd.DataFrame) -> dict:
    acc = (d.y == d.p).mean()
    bal = np.mean([(g.p == c).mean() for c, g in d.groupby("y")])
    out = {"pencere": acc, "dengeli": bal}
    for lvl, col in [("kayit", "kayit"), ("hasta", "hasta")]:
        v = d.groupby(col).agg(y=("y", "first"), p=("p", lambda s: s.value_counts().idxmax()))
        out[f"{lvl}_k"], out[f"{lvl}_n"] = int((v.y == v.p).sum()), len(v)
    return out


def load_all() -> pd.DataFrame:
    parts = []
    k = pd.read_parquet(RES / "final_tahminler_klasik.parquet")
    k = k[k.model == "DVM-RTF"].copy()
    k["kurgu"] = "Klasik öznitelik + DVM: " + k.gorev.str.replace("Saglikli", "Sağlıklı").str.replace("-sinif", " sınıf")
    parts.append(k)
    c = RES / "final_tahminler_cnn3sinif.parquet"
    if c.exists():
        c = pd.read_parquet(c); c["kurgu"] = "1D-CNN: 3 sınıf"; c["gorev"] = "3-sinif"; parts.append(c)
    a = RES / "final_tahminler_alsnet.parquet"
    if a.exists():
        a = pd.read_parquet(a); a["kurgu"] = "ALSNet kurgusu: ALS–Sağlıklı"; a["gorev"] = "ALS-Saglikli"; parts.append(a)
    return pd.concat(parts, ignore_index=True)


def table(df: pd.DataFrame, d4: pd.DataFrame | None) -> pd.DataFrame:
    rows = []
    for (kurgu, proto), g in df.groupby(["kurgu", "protokol"]):
        per_seed = [scores(gg) for _, gg in g.groupby("tohum")]
        s = pd.DataFrame(per_seed)
        hk, hn = int(s.hasta_k.iloc[0]), int(s.hasta_n.iloc[0])
        lo, hi = cp(hk, hn)
        rows.append({"kurgu": kurgu, "protokol": proto, "tohum_sayisi": len(s),
                     "pencere_ort": s.pencere.mean(), "pencere_sd": s.pencere.std(ddof=1) if len(s) > 1 else np.nan,
                     "dengeli_ort": s.dengeli.mean(),
                     "kayit_oy": (s.kayit_k / s.kayit_n).mean(),
                     "hasta_oy": f"{hk}/{hn}", "hasta_oy_GA95": f"{lo:.2f}–{hi:.2f}"})
    if d4 is not None:  # ALSNet kurgusu: rastgele/kayit/hasta grup protokolleri Deney 4'ten
        m = {"segment_5kat": "pencere_rastgele", "kayit_grup_5kat": "kayit_grup", "hasta_grup_5kat": "hasta_grup",
             "rastgele_80_20": "rastgele_80_20 (ALSNet bolmesi)"}
        for p_old, p_new in m.items():
            g = d4[d4.protokol == p_old]
            hk, hn = map(int, g["hasta_dogru/toplam"].iloc[0].split("/"))
            lo, hi = cp(hk, hn)
            rows.append({"kurgu": "ALSNet kurgusu: ALS–Sağlıklı", "protokol": p_new, "tohum_sayisi": len(g),
                         "pencere_ort": g.segment_dogruluk.mean(), "pencere_sd": g.segment_dogruluk.std(ddof=1),
                         "dengeli_ort": g.dengeli_dogruluk.mean(), "kayit_oy": g.kayit_dogruluk.mean(),
                         "hasta_oy": f"{hk}/{hn}", "hasta_oy_GA95": f"{lo:.2f}–{hi:.2f}"})
    t = pd.DataFrame(rows)
    order = {p: i for i, p in enumerate(["rastgele_80_20 (ALSNet bolmesi)"] + PROTO + ["hasta_LOSO"])}
    return t.sort_values(["kurgu", "protokol"], key=lambda s: s.map(order) if s.name == "protokol" else s)


def fig2(t: pd.DataFrame):
    kurgular = ["ALSNet kurgusu: ALS–Sağlıklı", "Klasik öznitelik + DVM: ALS-Sağlıklı", "Klasik öznitelik + DVM: ALS-Miyopati",
                "1D-CNN: 3 sınıf", "Klasik öznitelik + DVM: 3 sınıf", "Klasik öznitelik + DVM: Miyopati-Sağlıklı"]
    fig, ax = plt.subplots(figsize=(8.8, 6.0), dpi=150)
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    xs = np.arange(len(PROTO))
    ends = []
    for i, k in enumerate(kurgular):
        g = t[(t.kurgu == k) & (t.protokol.isin(PROTO))].set_index("protokol").reindex(PROTO)
        if g.pencere_ort.isna().all():
            continue
        y = 100 * g.pencere_ort.values
        e = 100 * g.pencere_sd.fillna(0).values
        ax.plot(xs, y, color=COLORS[i], lw=2, zorder=2)
        ax.errorbar(xs, y, yerr=e, fmt="none", ecolor=COLORS[i], elinewidth=1.2, capsize=3, zorder=2)
        ax.scatter(xs, y, s=60, marker=MARKERS[i], color=COLORS[i], edgecolor=SURFACE, linewidth=1.5, zorder=3,
                   label=k)
        short = k.replace("Klasik öznitelik + DVM: ", "Klasik+DVM, ").replace("ALSNet kurgusu: ", "ALSNet kurgusu, ").replace("1D-CNN: ", "1D-CNN, ")
        ends.append([y[-1], f"{short}  ({y[0]:.1f} → {y[-1]:.1f}; −{y[0] - y[-1]:.1f})", COLORS[i]])
    ends.sort(key=lambda e: e[0])
    for j in range(1, len(ends)):  # etiket cakismasini onle
        ends[j][0] = max(ends[j][0], ends[j - 1][0] + 2.6)
    for yv, lab, col in ends:
        ax.text(xs[-1] + 0.12, yv, lab, va="center", fontsize=8, color=INK)
    ax.set_xticks(xs); ax.set_xticklabels([PROTO_LAB[p] for p in PROTO], fontsize=9, color=INK)
    ax.set_xlim(-0.25, xs[-1] + 2.3)
    ax.set_ylabel("Pencere düzeyinde doğruluk (%)", fontsize=9, color=INK2)
    ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.set_title("Aynı veri ve model: bölme düzeyi değiştikçe doğruluk", loc="left", fontsize=11, color=INK, pad=20)
    ax.text(0, 1.012, "EMGLab N2001; 5-kat, 3 tohum ortalaması ± SS. Etiketler: segment → hasta gruplu doğruluk.",
            transform=ax.transAxes, fontsize=7.5, color=INK2, va="bottom")
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize=7.5, frameon=False, ncol=2)
    for tx in leg.get_texts():
        tx.set_color(INK)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"sekil2_bolme_duzeyi.{ext}", dpi=300, facecolor=SURFACE)


def fig3(df: pd.DataFrame):
    sel = [("1D-CNN: 3 sınıf", "1D-CNN, 3 sınıf"), ("ALSNet kurgusu: ALS–Sağlıklı", "ALSNet kurgusu, ALS–Sağlıklı"),
           ("Klasik öznitelik + DVM: 3 sınıf", "Klasik + DVM, 3 sınıf")]
    sel = [s for s in sel if ((df.kurgu == s[0]) & (df.protokol == "hasta_LOSO")).any()]
    fig, axes = plt.subplots(1, len(sel), figsize=(3.2 * len(sel), 4.4), dpi=150, sharey=True)
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor(SURFACE)
    cls_col = {"ALS": COLORS[1], "Miyopati": COLORS[5], "Sağlıklı": COLORS[0]}
    for ax, (k, title) in zip(axes, sel):
        ax.set_facecolor(SURFACE)
        g = df[(df.kurgu == k) & (df.protokol == "hasta_LOSO")]
        per = g.groupby("hasta").apply(lambda d: pd.Series({"acc": (d.y == d.p).mean(), "y": d.y.iat[0]}),
                                       include_groups=False).reset_index()
        if k.startswith("ALSNet"):
            per["sinif"] = per.y.map({1: "ALS", 0: "Sağlıklı"})
        else:
            per["sinif"] = per.y.map(CLS_TR)
        order = [c for c in ["ALS", "Miyopati", "Sağlıklı"] if c in per.sinif.values]
        for i, c in enumerate(order):
            v = per[per.sinif == c].sort_values("acc")
            jit = np.linspace(-0.18, 0.18, len(v)) if len(v) > 1 else [0]
            ax.scatter(i + np.asarray(jit), 100 * v.acc, s=46, color=cls_col[c], edgecolor=SURFACE, linewidth=1.5, zorder=3)
            ax.hlines(100 * v.acc.median(), i - 0.28, i + 0.28, color=INK2, lw=1.5, zorder=2)
            for _, r in v[v.acc < 0.5].iterrows():
                ax.annotate(r.hasta, (i + jit[list(v.index).index(_)], 100 * r.acc), xytext=(5, 0),
                            textcoords="offset points", fontsize=7, color=INK2, va="center")
        ax.axhline(50, color=GRID, lw=1, ls="--", zorder=1)
        ax.set_xticks(range(len(order))); ax.set_xticklabels(order, fontsize=9, color=INK)
        ax.set_title(title, fontsize=9.5, color=INK, loc="left")
        ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        for sp in ["left", "bottom"]:
            ax.spines[sp].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=8)
    axes[0].set_ylabel("Dışarıda bırakılan hastada pencere doğruluğu (%)", fontsize=8.5, color=INK2)
    fig.suptitle("Hasta-ayrık değerlendirme (LOSO): hasta başına doğruluk", x=0.01, ha="left", fontsize=11, color=INK)
    fig.text(0.01, 0.905, "Her nokta bir hasta; gri çizgi: medyan; kesikli çizgi: %50. %50'nin altındaki hastalar etiketli.",
             fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"sekil3_hasta_basina.{ext}", dpi=300, facecolor=SURFACE)


if __name__ == "__main__":
    df = load_all()
    d4p = RES / "deney4_alsnet.csv"
    d4 = pd.read_csv(d4p) if d4p.exists() else None
    t = table(df, d4)
    t.to_csv(OUT / "tablo2_deney_ozeti.csv", index=False, sep=";", encoding="utf-8-sig")
    with open(OUT / "tablo2_deney_ozeti.md", "w", encoding="utf-8") as f:
        f.write("| Kurgu | Protokol | Tohum | Pencere doğruluğu | Dengeli | Kayıt (oy) | Hasta (oy) | Hasta %95 GA |\n|---|---|---|---|---|---|---|---|\n")
        for _, r in t.iterrows():
            sd = "" if pd.isna(r.pencere_sd) else f" ± {100 * r.pencere_sd:.1f}"
            f.write(f"| {r.kurgu} | {r.protokol} | {r.tohum_sayisi} | {100 * r.pencere_ort:.1f}{sd} | {100 * r.dengeli_ort:.1f} | "
                    f"{100 * r.kayit_oy:.1f} | {r.hasta_oy} | {r.hasta_oy_GA95} |\n")
    fig2(t)
    fig3(df)
    print(t.to_string())


def fig4():
    """Model siralamasinin tersine donmesi: gorev basina 3 model, segment-rastgele vs hasta-grup."""
    k = pd.read_parquet(RES / "final_tahminler_klasik.parquet")
    k["ok"] = k.y == k.p
    t = (k.groupby(["gorev", "model", "protokol", "tohum"]).ok.mean()
         .groupby(["gorev", "model", "protokol"]).agg(["mean", "std"]).reset_index())
    gorevler = [("ALS-Saglikli", "ALS–Sağlıklı"), ("ALS-Miyopati", "ALS–Miyopati"),
                ("Miyopati-Saglikli", "Miyopati–Sağlıklı"), ("3-sinif", "3 sınıf")]
    models = [("K-EYK(K=1)", "1-NN", COLORS[1], "s"), ("DVM-RTF", "DVM-RTF", COLORS[0], "o"), ("DDA", "DDA (doğrusal)", COLORS[2], "D")]
    fig, axes = plt.subplots(1, 4, figsize=(11, 4.3), dpi=150, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, (g, gl) in zip(axes, gorevler):
        ax.set_facecolor(SURFACE)
        for m, ml, col, mk in models:
            r = t[(t.gorev == g) & (t.model == m)].set_index("protokol")
            y = [100 * r.loc["pencere_rastgele", "mean"], 100 * r.loc["hasta_grup", "mean"]]
            ax.plot([0, 1], y, color=col, lw=2, zorder=2)
            ax.scatter([0, 1], y, s=55, marker=mk, color=col, edgecolor=SURFACE, linewidth=1.5, zorder=3,
                       label=ml if g == gorevler[0][0] else None)
        ranks = []
        for p in ("pencere_rastgele", "hasta_grup"):
            rr = t[(t.gorev == g) & (t.protokol == p)].sort_values("mean", ascending=False)
            ranks.append(" > ".join({"K-EYK(K=1)": "1-NN", "DVM-RTF": "DVM", "DDA": "DDA"}[x] for x in rr.model))
        ax.text(0.5, -0.24, f"Rastgele: {ranks[0]}\nHasta-ayrık: {ranks[1]}", transform=ax.transAxes,
                ha="center", va="top", fontsize=7.5, color=INK)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Segment\n(rastgele)", "Hasta\ngruplu"], fontsize=8.5, color=INK)
        ax.set_xlim(-0.3, 1.3)
        ax.set_title(gl, fontsize=9.5, color=INK, loc="left")
        ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        for sp in ["left", "bottom"]:
            ax.spines[sp].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=8)
    axes[0].set_ylabel("Pencere düzeyinde doğruluk (%)", fontsize=8.5, color=INK2)
    leg = axes[0].legend(loc="lower left", fontsize=8, frameon=False)
    for tx in leg.get_texts():
        tx.set_color(INK)
    fig.suptitle("Sızıntılı değerlendirme model sıralamasını değiştiriyor", x=0.01, ha="left", fontsize=11, color=INK)
    fig.text(0.01, 0.905, "Klasik pencere öznitelikleri; 5-kat, 3 tohum ortalaması. Sızıntılı bölmede en iyi görünen 1-NN, hasta-ayrık bölmede en kötü; doğrusal DDA neredeyse hiç şişmiyor.",
             fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.06, 1, 0.9))
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"sekil4_model_siralamasi.{ext}", dpi=300, facecolor=SURFACE)


if __name__ == "__main__":
    fig4()
