"""kaggle_kontrol.py'nin devami: eslesme guveni (en iyi - ikinci en iyi) ve on isleme etkisi."""
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_kontrol as kk  # noqa: E402
from hanife_pipeline import preprocess  # noqa: E402
from n2001 import list_records  # noqa: E402

raw = kk.read_files(Path(sys.argv[1]))
Xtr = kk.safe_load(raw["X_train.pkl"]); Xte = kk.safe_load(raw["X_test.pkl"])
L = Xtr.shape[1]
recs = [r for r in list_records() if r.group in ("ALS", "Normal") and r.muscle in ("BB", "VM")]
print("Kaggle segment ornek istatistik: ort %.2f, std %.2f, min %.1f, max %.1f" % (Xte.mean(), Xte.std(), Xte.min(), Xte.max()))

for label, fn in [("ham", lambda x, fs: x), ("filtreli(10-1500)", preprocess)]:
    win, meta = [], []
    for r in recs:
        x = fn(r.signal(), r.fs)
        for k in range(len(x) // L):
            win.append(x[k * L:(k + 1) * L]); meta.append((r.name, r.subject, k))
    W = kk.zfp(np.asarray(win), n=L)

    def ident(X):
        F = kk.zfp(X, n=L); res = []
        for i in range(0, len(F), 128):
            C = F[i:i + 128] @ W.T
            o = np.argsort(C, 1)[:, -2:]
            for t in range(len(C)):
                b, s = o[t, 1], o[t, 0]
                res.append((meta[b], C[t, b], C[t, b] - C[t, s]))
        return res
    tr, te = ident(Xtr), ident(Xte)
    best = np.array([c for _, c, _ in te]); marg = np.array([m for _, _, m in te])
    print(f"\n[{label}] test: en iyi kor medyan {np.median(best):.3f}; marj medyan {np.median(marg):.3f}; "
          f"marj>0.3 olan {np.mean(marg > 0.3):.1%}")
    for thr in (0.3, 0.5):
        ok_tr = [m for m, c, g in tr if g > thr]; ok_te = [m for m, c, g in te if g > thr]
        rec_tr = {m[0] for m in ok_tr}; sub_tr = {m[1] for m in ok_tr}
        shared = sum(m[0] in rec_tr for m in ok_te)
        per = Counter(Counter(m[0] for m in ok_te).values())
        print(f"  marj>{thr}: eslesen egitim {len(ok_tr)}/{len(tr)}, test {len(ok_te)}/{len(te)} | "
              f"test segmentinin kaydi egitimde de: {shared}/{len(ok_te)} ({shared / max(len(ok_te), 1):.1%}) | "
              f"test hastasi egitimde de: {sum(m[1] in sub_tr for m in ok_te)}/{len(ok_te)} | test kayitlarinda segment/kayit: {dict(per)}")
    # ayni pencerenin hem egitimde hem testte olmasi (birebir kopya)?
    trw = {(m[0], m[2]) for m, c, g in tr if g > 0.3}
    dup = sum((m[0], m[2]) in trw for m, c, g in te if g > 0.3)
    print(f"  ayni pencere (kayit+sira) hem egitim hem testte: {dup}")
