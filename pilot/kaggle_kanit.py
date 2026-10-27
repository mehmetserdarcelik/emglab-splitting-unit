"""ALSNet Kaggle bolmesi icin ayrintili kanit dosyalari.

Cikti (pilot/results/kaggle_kanit/):
  eslesme_tablosu.csv   : her Kaggle segmenti -> kume (egitim/test), eslesen kayit, hasta, pencere sirasi,
                          en iyi korelasyon, ikinci en iyi, marj
  kayit_ozeti.csv       : test segmenti iceren her kayit icin egitim/test segment siralari
  sekil_marj.png        : en iyi korelasyon ve marj dagilimi
  sekil_ornek.png       : ornek test segmenti ile eslesen ham pencere ve ayni kaydin egitimdeki komsu penceresi

Kullanim: python -I kaggle_kanit.py <kaggle_zip>
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_kontrol as kk  # noqa: E402
from n2001 import list_records  # noqa: E402

OUT = Path(__file__).resolve().parent / "results" / "kaggle_kanit"
OUT.mkdir(parents=True, exist_ok=True)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"

raw = kk.read_files(Path(sys.argv[1]))
Xtr = kk.safe_load(raw["X_train.pkl"]); Xte = kk.safe_load(raw["X_test.pkl"])
ytr = kk.safe_load(raw["y_train.pkl"]).ravel(); yte = kk.safe_load(raw["y_test.pkl"]).ravel()
L = Xtr.shape[1]

recs = [r for r in list_records() if r.group in ("ALS", "Normal")]  # tum kaslar (ALSNet verisinde BB/VM disi kaslar da var)
win, meta = [], []
for r in recs:
    x = r.signal()
    for k in range(len(x) // L):
        win.append(x[k * L:(k + 1) * L]); meta.append((r.name, r.subject, r.group, k))
W = np.asarray(win)
Wz = kk.zfp(W, n=L)


def ident(X):
    F = kk.zfp(X, n=L); res = []
    for i in range(0, len(F), 128):
        C = F[i:i + 128] @ Wz.T
        o = np.argsort(C, 1)[:, -2:]
        for t in range(len(C)):
            b, s = o[t, 1], o[t, 0]
            res.append((b, C[t, b], C[t, s]))
    return res


rows = []
for kume, X, y in [("egitim", Xtr, ytr), ("test", Xte, yte)]:
    for i, (b, c1, c2) in enumerate(ident(X)):
        name, subj, grp, k = meta[b]
        rows.append({"kume": kume, "kaggle_sira": i, "kaggle_etiket": int(y[i]), "kayit": name, "hasta": subj,
                     "grup": grp, "pencere_sira": k, "en_iyi_kor": round(float(c1), 4),
                     "ikinci_kor": round(float(c2), 4), "marj": round(float(c1 - c2), 4)})
df = pd.DataFrame(rows)
df["guvenli"] = df.marj > 0.5
df.to_csv(OUT / "eslesme_tablosu.csv", index=False, sep=";", encoding="utf-8-sig")

g = df[df.guvenli]
tr_rec = g[g.kume == "egitim"].groupby("kayit").pencere_sira.apply(lambda s: sorted(set(s)))
te_rec = g[g.kume == "test"].groupby("kayit").pencere_sira.apply(lambda s: sorted(set(s)))
ozet = pd.DataFrame({"test_pencereleri": te_rec}).join(pd.DataFrame({"egitim_pencereleri": tr_rec}), how="left")
ozet["hasta"] = ozet.index.map(lambda n: g[g.kayit == n].hasta.iat[0])
ozet["egitimde_ayni_kayit_var"] = ozet.egitim_pencereleri.apply(lambda v: isinstance(v, list) and len(v) > 0)
ozet.to_csv(OUT / "kayit_ozeti.csv", sep=";", encoding="utf-8-sig")

# Etiket tutarliligi: Kaggle etiketi (1/0) ile eslesen kaydin grubu ayni mi?
lab = g.assign(beklenen=g.grup.map({"ALS": None, "Normal": None}))
grp_by_label = g.groupby("kaggle_etiket").grup.agg(lambda s: s.value_counts().to_dict())

te = g[g.kume == "test"]
print("Kaggle egitim/test:", len(Xtr), len(Xte))
print("guvenli eslesen: egitim", int((g.kume == "egitim").sum()), "test", len(te))
print("test segmentinin kaydi egitimde de var:", int(te.kayit.isin(tr_rec.index).sum()), "/", len(te))
print("test segmentinin hastasi egitimde de var:", int(te.hasta.isin(set(g[g.kume == 'egitim'].hasta)).sum()), "/", len(te))
print("test kayit sayisi:", te.kayit.nunique(), "| tamami testte olan kayit:",
      int((~ozet.egitimde_ayni_kayit_var).sum()))
print("ayni (kayit, pencere) hem egitim hem test:", len(set(zip(te.kayit, te.pencere_sira)) &
                                                       set(zip(g[g.kume == 'egitim'].kayit, g[g.kume == 'egitim'].pencere_sira))))
print("Kaggle etiketi -> eslesen grup:", grp_by_label.to_dict())
print("en iyi kor medyan (tum):", df.en_iyi_kor.median(), "| marj medyan:", df.marj.median(),
      "| ikinci en iyi kor medyan:", df.ikinci_kor.median())
print("Kaggle deger araligi:", float(Xte.min()), float(Xte.max()))

# Sekil: korelasyon ve marj dagilimi
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), dpi=150)
fig.patch.set_facecolor(SURFACE)
bins = np.linspace(0, 1, 41)
for ax, col, title in [(axes[0], None, "Korelasyon: en iyi ve ikinci en iyi aday"), (axes[1], "marj", "Marj (en iyi − ikinci en iyi)")]:
    ax.set_facecolor(SURFACE)
    if col is None:
        ax.hist(df.en_iyi_kor, bins=bins, color=BLUE, alpha=0.85, label="en iyi eşleşme")
        ax.hist(df.ikinci_kor, bins=bins, color=ORANGE, alpha=0.75, label="ikinci en iyi aday")
        ax.legend(fontsize=8, frameon=False)
    else:
        ax.hist(df.marj, bins=bins, color=GREEN, alpha=0.9)
        ax.axvline(0.5, color=INK2, ls="--", lw=1)
        ax.text(0.52, ax.get_ylim()[1] * 0.9, "güvenli eşik 0,5", fontsize=8, color=INK2)
    ax.set_title(title, fontsize=9.5, loc="left", color=INK)
    ax.set_ylabel("Segment sayısı", fontsize=8.5, color=INK2)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=8, colors=INK2)
fig.tight_layout()
fig.savefig(OUT / "sekil_marj.png", dpi=200, facecolor=SURFACE)

# Sekil: ornek - ayni kaydin bir penceresi testte, bir baska penceresi egitimde
cand = ozet[ozet.egitimde_ayni_kayit_var].copy()
# Ornek: korelasyonu tipik degere (medyan) en yakin, biceps kaydindan bir test segmenti
med = te.en_iyi_kor.median()
pool = te[te.kayit.isin(cand.index) & te.kayit.str.contains("BB")].copy()
row = pool.iloc[(pool.en_iyi_kor - med).abs().argsort().iloc[0]]
ex_rec, te_k, ti = row.kayit, int(row.pencere_sira), int(row.kaggle_sira)
tr_k = cand.loc[ex_rec, "egitim_pencereleri"][0]
raw_idx = [j for j, m in enumerate(meta) if m[0] == ex_rec and m[3] == te_k][0]
seg = Xte.reshape(len(Xte), -1)[ti].astype(float)
ref = W[raw_idx]
z = lambda a: (a - a.mean()) / (a.std() + 1e-12)
t = np.arange(L) / 23437.5 * 1000
sl = slice(0, int(0.25 * 23437.5))  # ilk 250 ms
fig, ax = plt.subplots(figsize=(9, 3.2), dpi=150)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
ax.plot(t[sl], z(ref)[sl], color=BLUE, lw=1.2, label=f"EMGLab ham kayıt {ex_rec}, pencere {te_k}")
ax.plot(t[sl], z(seg)[sl] + 0.0, color=ORANGE, lw=0.8, ls="--", label=f"Kaggle TEST segmenti #{ti}")
c = te[(te.kaggle_sira == ti)].en_iyi_kor.iat[0]
ax.set_title(f"Kaggle test segmenti ile ham kayıt birebir örtüşüyor (r = {c:.3f}); aynı kaydın diğer saniyeleri {cand.loc[ex_rec, 'egitim_pencereleri']} EĞİTİM kümesinde",
             fontsize=9, loc="left", color=INK)
ax.set_xlabel("Zaman (ms)", fontsize=8.5, color=INK2); ax.set_ylabel("z-skoru", fontsize=8.5, color=INK2)
ax.legend(fontsize=8, frameon=False, loc="upper right")
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
ax.tick_params(labelsize=8, colors=INK2)
fig.tight_layout()
fig.savefig(OUT / "sekil_ornek.png", dpi=200, facecolor=SURFACE)
print("ornek kayit:", ex_rec, "test penceresi", te_k, "egitim pencereleri", cand.loc[ex_rec, "egitim_pencereleri"])
print(ozet.head(12).to_string())
