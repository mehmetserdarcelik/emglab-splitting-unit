"""ALSNet Kaggle verisinde (naimulhassan/processed-emglab-dataset-for-als-detection) egitim/test
segmentlerinin hangi EMGLab kaydina ve hastaya ait oldugunu bulur; kayit/hasta ortusmesini olcer.

Kullanim:  python -I kaggle_kontrol.py <kaggle_zip_veya_klasor>

Guvenlik: .pkl dosyalari ucuncu tarafa ait. Once pickletools ile (calistirmadan) icerdikleri
GLOBAL referanslar listelenir; yalnizca izin listesindeki numpy/pandas siniflari yuklenir.
"""
from __future__ import annotations

import io
import pickle
import pickletools
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from n2001 import list_records  # noqa: E402

ALLOWED_PREFIXES = ("numpy.core.multiarray", "numpy._core.multiarray", "numpy.core.numeric", "numpy._core.numeric",
                    "pandas.core.frame", "pandas.core.series", "pandas.core.internals", "pandas.core.indexes",
                    "pandas.core.arrays", "pandas._libs.internals", "pandas._libs.index", "pandas._libs.tslibs")
ALLOWED_EXACT = {("builtins", "slice"), ("builtins", "object"), ("collections", "OrderedDict"),
                 ("copyreg", "_reconstructor"), ("numpy", "ndarray"), ("numpy", "dtype")}


def globals_in(data: bytes) -> set[tuple[str, str]]:
    """Pickle'i calistirmadan GLOBAL/STACK_GLOBAL referanslarini cikarir."""
    out, strings = set(), []
    for op, arg, _ in pickletools.genops(io.BytesIO(data)):
        if op.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE"):
            strings.append(arg)
        elif op.name == "GLOBAL":
            mod, name = arg.split(" ", 1)
            out.add((mod, name))
        elif op.name == "STACK_GLOBAL" and len(strings) >= 2:
            out.add((strings[-2], strings[-1]))
    return out


def allowed(mod: str, name: str) -> bool:
    return (mod, name) in ALLOWED_EXACT or mod.startswith(ALLOWED_PREFIXES)


class SafeUnpickler(pickle.Unpickler):
    """Asil denetim burada: gercek modul/isim cifti izin listesinde degilse yukleme durur."""

    def find_class(self, module, name):
        if not allowed(module, name):
            raise pickle.UnpicklingError(f"izin verilmeyen referans: {module}.{name}")
        if module == "pandas.core.indexes.numeric" and name in ("Int64Index", "Float64Index", "UInt64Index"):
            import pandas as pd  # eski pandas surumlerindeki siniflar -> Index
            return pd.Index
        return super().find_class(module, name)


def safe_load(data: bytes):
    # Bilgi amacli on tarama (calistirmadan); STACK_GLOBAL ciftleri yaklasik okunur.
    print("  pickle referanslari (yaklasik):", sorted(globals_in(data)))
    obj = SafeUnpickler(io.BytesIO(data)).load()
    return np.asarray(obj.values if hasattr(obj, "values") else obj)


def read_files(src: Path) -> dict[str, bytes]:
    names = ["X_train.pkl", "X_test.pkl", "y_train.pkl", "y_test.pkl"]
    if src.is_file():
        with zipfile.ZipFile(src) as z:
            idx = {Path(n).name: n for n in z.namelist()}
            return {n: z.read(idx[n]) for n in names}
    return {n: (next(src.rglob(n))).read_bytes() for n in names}


def zfp(a: np.ndarray, n: int = 4000) -> np.ndarray:
    """Olcekten bagimsiz parmak izi: ilk n ornegin z-skoru (satir bazinda)."""
    a = a[:, :n].astype(np.float64)
    a = a - a.mean(1, keepdims=True)
    return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)


def main(src: Path):
    raw = read_files(src)
    data = {}
    for k, v in raw.items():
        print(k, f"{len(v) / 1e6:.1f} MB")
        data[k] = safe_load(v)
    Xtr, Xte = data["X_train.pkl"].reshape(len(data["X_train.pkl"]), -1), data["X_test.pkl"].reshape(len(data["X_test.pkl"]), -1)
    ytr, yte = data["y_train.pkl"].ravel(), data["y_test.pkl"].ravel()
    print("egitim", Xtr.shape, Counter(ytr.tolist()), "| test", Xte.shape, Counter(yte.tolist()))
    L = Xtr.shape[1]

    # aday EMGLab pencereleri: ALS + saglikli, BB + VM, kayit basina 11 adet ardisik pencere
    recs = [r for r in list_records() if r.group in ("ALS", "Normal") and r.muscle in ("BB", "VM")]
    win, meta = [], []
    for r in recs:
        x = r.signal()
        for k in range(len(x) // L):
            win.append(x[k * L:(k + 1) * L])
            meta.append((r.name, r.subject, r.group, k))
    W = zfp(np.asarray(win))
    print("aday pencere:", len(meta))

    def identify(X):
        F = zfp(X)
        out = []
        for i in range(0, len(F), 256):
            C = F[i:i + 256] @ W.T
            j = C.argmax(1)
            out += [(meta[jj], float(C[t, jj])) for t, jj in enumerate(j)]
        return out

    tr, te = identify(Xtr), identify(Xte)
    for name, res in [("egitim", tr), ("test", te)]:
        c = np.array([s for _, s in res])
        print(f"{name}: eslesme korelasyonu medyan {np.median(c):.4f}, min {c.min():.4f}, >0.99 olan {np.mean(c > 0.99):.1%}")

    ok_tr = [m for m, s in tr if s > 0.99]
    ok_te = [m for m, s in te if s > 0.99]
    rec_tr, rec_te = {m[0] for m in ok_tr}, {m[0] for m in ok_te}
    sub_tr, sub_te = {m[1] for m in ok_tr}, {m[1] for m in ok_te}
    seg_shared = sum(1 for m in ok_te if m[0] in rec_tr)
    print("\n=== SONUC ===")
    print(f"test segmentlerinden kaydi egitimde de bulunan: {seg_shared}/{len(ok_te)} ({seg_shared / max(len(ok_te), 1):.1%})")
    print(f"test kayitlari: {len(rec_te)}, bunlardan egitimde de segmenti olan: {len(rec_te & rec_tr)}")
    print(f"test hastalari: {sorted(sub_te)}")
    print(f"egitim hastalari: {sorted(sub_tr)}")
    print(f"ortak hasta: {sorted(sub_te & sub_tr)}")
    per_rec = Counter(m[0] for m in ok_te)
    print("test kayitlarinda segment sayisi dagilimi (11 = tum kayit testte):", Counter(per_rec.values()))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
