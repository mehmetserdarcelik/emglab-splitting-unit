"""Hakem degerlendirmesi sonrasi ek deneyler (pencere oznitelikleri, exp5 part_c ile ayni oznitelikler).

A) Yalnizca biceps brachii (BB) kayitlari: kas-sinif karistiricisi olmadan ayni protokoller (3 tohum)
B) Ic ice capraz dogrulama: RBF-SVM'nin C ve gamma'si egitim katlari icinde secilir.
   Ic bolme, dis protokolle ayni birimde yapilir (segment-rastgele dis -> segment-rastgele ic;
   hasta-grup dis -> hasta-grup ic). 1 tohum.
C) 1D-CNN, 3 sinif, yalnizca BB (exp5 part_a ile ayni model): pencere_rastgele / kayit_grup / hasta_grup (1 tohum)

Cikti: results/rev_bb_klasik.parquet, results/rev_nested.csv, results/rev_bb_cnn.parquet
Kullanim: python exp6_revizyon.py [a] [b] [c]
"""
from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import evaluate as ev
from exp5_final import SEEDS, protocols

RES = Path(__file__).resolve().parent / "results"
COLS = ["IEMG", "SGS", "WAMP", "MF", "MDF", "RMS", "MNP"]


def part_a():
    dw = pd.read_csv(RES / "features_windows.csv")
    dw = dw[dw.muscle == "BB"]
    rows = []
    for task, (cls, _) in ev.TASKS.items():
        d = dw[dw.group.isin(cls)].reset_index(drop=True)
        X, y, s, r = d[COLS].values.astype(float), d.group.values, d.subject.values, d.record.values
        for cname in ["DVM-RTF", "K-EYK(K=1)", "DDA"]:
            for seed in SEEDS:
                for name, splits in protocols(X, y, r, s, seed, loso=False).items():
                    for k, (a, b) in enumerate(splits):
                        clf = ev.classifiers()[cname]
                        clf.fit(X[a], y[a])
                        rows.append(pd.DataFrame({"y": y[b], "p": clf.predict(X[b]), "kayit": r[b], "hasta": s[b],
                                                  "protokol": name, "tohum": seed, "kat": k, "gorev": task, "model": cname}))
            print("BB", task, cname, "tamam", flush=True)
    pd.concat(rows, ignore_index=True).to_parquet(RES / "rev_bb_klasik.parquet")


GRID = {"svc__C": [0.1, 1, 10, 100, 400], "svc__gamma": [1 / (2 * s ** 2) for s in (0.5, 1.5, 5.0)]}


def nested_one(args):
    task, proto = args
    cls, _ = ev.TASKS[task]
    dw = pd.read_csv(RES / "features_windows.csv")
    d = dw[dw.group.isin(cls)].reset_index(drop=True)
    X, y, s = d[COLS].values.astype(float), d.group.to_numpy(dtype=str), d.subject.to_numpy(dtype=str)
    base = make_pipeline(ev._log, StandardScaler(), SVC(kernel="rbf"))
    if proto == "pencere_rastgele":
        outer = list(StratifiedKFold(5, shuffle=True, random_state=0).split(X, y))
    else:
        outer = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, s))
    p = np.empty_like(y)
    chosen = []
    for a, b in outer:
        if proto == "pencere_rastgele":
            inner = StratifiedKFold(3, shuffle=True, random_state=1)
            gs = GridSearchCV(base, GRID, cv=inner, n_jobs=1).fit(X[a], y[a])
        else:
            inner = StratifiedGroupKFold(3, shuffle=True, random_state=1)
            gs = GridSearchCV(base, GRID, cv=inner, n_jobs=1).fit(X[a], y[a], groups=s[a])
        p[b] = gs.predict(X[b])
        chosen.append(f"C={gs.best_params_['svc__C']},g={gs.best_params_['svc__gamma']:.3g}")
    acc = float(np.mean(p == y))
    print("nested", task, proto, round(acc, 4), chosen, flush=True)
    return {"gorev": task, "protokol": proto, "dogruluk": acc, "secilen": ";".join(chosen)}


def part_b():
    jobs = [(t, p) for t in ev.TASKS for p in ("pencere_rastgele", "hasta_grup")]
    with Pool(len(jobs)) as pool:
        rows = pool.map(nested_one, jobs)
    pd.DataFrame(rows).to_csv(RES / "rev_nested.csv", index=False)


def part_c():
    import torch

    import exp3_cnn as e3
    from n2001 import list_records
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    recs = [x for x in list_records() if x.muscle == "BB"]
    with Pool(24) as pool:
        data = pool.map(e3.load, recs)
    X = np.concatenate([d[0] for d in data])
    y = np.concatenate([[e3.CLASSES.index(d[1])] * len(d[0]) for d in data])
    s = np.concatenate([[d[2]] * len(d[0]) for d in data])
    r = np.concatenate([[d[3]] * len(d[0]) for d in data])
    rows = []
    for name, splits in protocols(X, y, r, s, 0, loso=False).items():
        for k, (a, b) in enumerate(splits):
            p = e3.train_eval(X[a], y[a], X[b], dev, seed=k)
            rows.append(pd.DataFrame({"y": y[b], "p": p, "kayit": r[b], "hasta": s[b], "protokol": name, "tohum": 0, "kat": k}))
        d = pd.concat([x for x in rows if x.protokol.iat[0] == name])
        print("BB cnn", name, round(float((d.y == d.p).mean()), 4), flush=True)
    pd.concat(rows, ignore_index=True).to_parquet(RES / "rev_bb_cnn.parquet")


if __name__ == "__main__":
    for p in sys.argv[1:] or ["a", "b", "c"]:
        {"a": part_a, "b": part_b, "c": part_c}[p]()
    print("bitti", flush=True)
