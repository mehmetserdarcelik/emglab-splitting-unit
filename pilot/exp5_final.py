"""Makale icin nihai deneyler: tekrarli tohumlar ve pencere duzeyinde tahminlerin kaydi.

A) 1D-CNN, 3 sinif, tum kayitlar (exp3 modeli): pencere_rastgele / kayit_grup / hasta_grup (5-kat, 3 tohum) + hasta_LOSO
B) ALSNet benzeri CNN, ALS-S, BB+VM 302 kayit (exp4 kurgusu): hasta_LOSO (tahminler hasta basina sekil icin)
C) Klasik pencere oznitelikleri + DVM-RTF / K-EYK(1) / DDA, 4 gorev: ayni protokoller, 3 tohum

Cikti: results/final_tahminler_*.parquet (her pencere icin y, p, kayit, hasta, protokol, tohum, kat)
"""
from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold, StratifiedKFold

import evaluate as ev
import exp3_cnn as e3
import exp4_alsnet as e4
from n2001 import list_records

RES = Path(__file__).resolve().parent / "results"
SEEDS = [0, 1, 2]
dev = "cuda" if torch.cuda.is_available() else "cpu"


def protocols(X, y, rec, subj, seed, loso):
    P = {"pencere_rastgele": list(StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y)),
         "kayit_grup": list(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, rec)),
         "hasta_grup": list(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, subj))}
    if loso:
        P["hasta_LOSO"] = list(LeaveOneGroupOut().split(X, y, subj))
    return P


def run_cnn(X, y, rec, subj, fit, tag, only=None):
    rows = []
    for seed in SEEDS:
        for name, splits in protocols(X, y, rec, subj, seed, loso=(seed == SEEDS[0])).items():
            if only and name not in only:
                continue
            for k, (a, b) in enumerate(splits):
                p = fit(X[a], y[a], X[b], seed * 100 + k)
                rows.append(pd.DataFrame({"y": y[b], "p": p, "kayit": rec[b], "hasta": subj[b],
                                          "protokol": name, "tohum": seed, "kat": k}))
            d = pd.concat([r for r in rows if r.protokol.iat[0] == name and r.tohum.iat[0] == seed])
            print(tag, seed, name, "pencere dogruluk", round(float((d.y == d.p).mean()), 4), flush=True)
    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(RES / f"final_tahminler_{tag}.parquet")
    return out


def part_a():
    recs = list_records()
    with Pool(24) as pool:
        data = pool.map(e3.load, recs)
    X = np.concatenate([d[0] for d in data])
    y = np.concatenate([[e3.CLASSES.index(d[1])] * len(d[0]) for d in data])
    s = np.concatenate([[d[2]] * len(d[0]) for d in data])
    r = np.concatenate([[d[3]] * len(d[0]) for d in data])
    run_cnn(X, y, r, s, lambda a, b, c, sd: e3.train_eval(a, b, c, dev, seed=sd), "cnn3sinif")


def part_b():
    recs = [x for x in list_records() if x.group in ("ALS", "Normal") and x.muscle in ("BB", "VM")]
    rng = np.random.default_rng(2022)
    sel = []
    for g in ("ALS", "Normal"):
        c = [x for x in recs if x.group == g]
        sel += [c[i] for i in sorted(rng.choice(len(c), 151, replace=False))]
    with Pool(16) as pool:
        data = pool.map(e4.load, sel)
    X = np.concatenate([d[0] for d in data])
    y = np.concatenate([[int(d[1] == "ALS")] * 11 for d in data])
    s = np.concatenate([[d[2]] * 11 for d in data])
    r = np.concatenate([[d[3]] * 11 for d in data])
    run_cnn(X, y, r, s, lambda a, b, c, sd: e4.train_predict(a, b, c, dev, seed=sd), "alsnet",
            only={"hasta_LOSO"})


def part_c():
    dw = pd.read_csv(RES / "features_windows.csv")
    cols = ["IEMG", "SGS", "WAMP", "MF", "MDF", "RMS", "MNP"]
    rows = []
    for task, (cls, _) in ev.TASKS.items():
        d = dw[dw.group.isin(cls)].reset_index(drop=True)
        X, y, s, r = d[cols].values.astype(float), d.group.values, d.subject.values, d.record.values
        for cname in ["DVM-RTF", "K-EYK(K=1)", "DDA"]:
            for seed in SEEDS:
                for name, splits in protocols(X, y, r, s, seed, loso=(seed == SEEDS[0])).items():
                    for k, (a, b) in enumerate(splits):
                        clf = ev.classifiers()[cname]
                        clf.fit(X[a], y[a])
                        rows.append(pd.DataFrame({"y": y[b], "p": clf.predict(X[b]), "kayit": r[b], "hasta": s[b],
                                                  "protokol": name, "tohum": seed, "kat": k,
                                                  "gorev": task, "model": cname}))
            print("klasik", task, cname, "tamam", flush=True)
    pd.concat(rows, ignore_index=True).to_parquet(RES / "final_tahminler_klasik.parquet")


def part_d():
    """Tez yapisi (kayit basina 1 vektor, IEMG+SGS+WAMP+MF): tezdeki kayit rastgele 10-kat / hasta grup 5-kat / LOSO."""
    df = pd.read_csv(RES / "features_records.csv")
    df = df[df.ok].reset_index(drop=True)
    cols = ev.FEATURE_SETS[ev.BEST_SET]
    rows = []
    for task, (cls, _) in ev.TASKS.items():
        d = df[df.group.isin(cls)].reset_index(drop=True)
        X, y, s, r = d[cols].values.astype(float), d.group.values, d.subject.values, d.record.values
        for cname in ["DVM-RTF", "K-EYK(K=1)", "DDA"]:
            for seed in SEEDS:
                P = {"kayit_rastgele_10kat": list(StratifiedKFold(10, shuffle=True, random_state=seed).split(X, y)),
                     "hasta_grup": list(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, s))}
                if seed == SEEDS[0]:
                    P["hasta_LOSO"] = list(LeaveOneGroupOut().split(X, y, s))
                for name, splits in P.items():
                    for k, (a, b) in enumerate(splits):
                        clf = ev.classifiers()[cname]
                        clf.fit(X[a], y[a])
                        rows.append(pd.DataFrame({"y": y[b], "p": clf.predict(X[b]), "kayit": r[b], "hasta": s[b],
                                                  "protokol": name, "tohum": seed, "kat": k,
                                                  "gorev": task, "model": cname}))
            print("tez", task, cname, "tamam", flush=True)
    pd.concat(rows, ignore_index=True).to_parquet(RES / "final_tahminler_tez.parquet")


if __name__ == "__main__":
    parts = sys.argv[1:] or ["c", "b", "a"]
    for p in parts:
        {"a": part_a, "b": part_b, "c": part_c, "d": part_d}[p]()
    print("bitti", flush=True)
