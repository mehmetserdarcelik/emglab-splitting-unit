"""Tum N2001 kayitlari icin oznitelikleri cikarir ve CSV'ye yazar.

Cikti:
  results/features_records.csv  : tez yapisi (MUAP -> melez kumeleme -> sablon oznitelikleri), kayit basina 1 satir
  results/features_windows.csv  : filtrelenmis sinyalin 1 sn'lik pencerelerinden klasik oznitelikler
"""
from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch

from hanife_pipeline import preprocess, record_features
from n2001 import list_records

OUT = Path(__file__).resolve().parent / "results"
WIN_FEATS = ["IEMG", "MAV", "VAR", "RMS", "STD", "SGS", "WAMP", "MF", "MDF", "PKF", "MNP", "SM1", "SM2", "SM3"]


def window_features(w: np.ndarray, fs: float, thr_uv=10.0) -> dict:
    f = {}
    f["IEMG"] = np.sum(np.abs(w))
    f["MAV"] = np.mean(np.abs(w))
    f["VAR"] = np.var(w, ddof=1)
    f["RMS"] = np.sqrt(np.mean(w ** 2))
    f["STD"] = np.std(w)
    s = np.sign(w)
    f["SGS"] = np.sum((s[:-1] * s[1:] < 0) & (np.abs(w[:-1] - w[1:]) >= thr_uv))
    f["WAMP"] = np.sum(np.abs(np.diff(w)) >= thr_uv)
    fr, P = welch(w, fs=fs, nperseg=2048)
    band = (fr >= 10) & (fr <= 1500)
    fr, P = fr[band], P[band]
    Ps = P.sum()
    f["MF"] = np.sum(fr * P) / Ps
    f["MDF"] = fr[np.searchsorted(np.cumsum(P), Ps / 2)]
    f["PKF"] = fr[np.argmax(P)]
    f["MNP"] = np.mean(P)
    f["SM1"] = np.sum(P * fr)
    f["SM2"] = np.sum(P * fr ** 2)
    f["SM3"] = np.sum(P * fr ** 3)
    return f


def process(rec):
    base = {"record": rec.name, "group": rec.group, "subject": rec.subject, "muscle": rec.muscle,
            "age": rec.meta.get("age"), "gender": rec.meta.get("gender")}
    try:
        x = rec.signal()
        feats, info = record_features(x, rec.fs)
        row = {**base, **info, "ok": feats is not None, **(feats or {})}

        y = preprocess(x, rec.fs)
        n = int(rec.fs)  # 1 sn
        wins = []
        for k in range(len(y) // n):
            wins.append({**base, "window": k, **window_features(y[k * n:(k + 1) * n], rec.fs)})
        return row, wins
    except Exception as e:  # kayit bazinda hata, toplu calismayi durdurmasin
        return {**base, "ok": False, "error": repr(e)}, []


if __name__ == "__main__":
    recs = list_records()
    print(f"{len(recs)} kayit bulundu", flush=True)
    OUT.mkdir(exist_ok=True)
    rows, wins = [], []
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 28) as pool:
        for i, (r, w) in enumerate(pool.imap_unordered(process, recs, chunksize=2), 1):
            rows.append(r)
            wins.extend(w)
            if i % 50 == 0:
                print(f"{i}/{len(recs)}", flush=True)
    pd.DataFrame(rows).sort_values("record").to_csv(OUT / "features_records.csv", index=False)
    pd.DataFrame(wins).sort_values(["record", "window"]).to_csv(OUT / "features_windows.csv", index=False)
    print("bitti", flush=True)
