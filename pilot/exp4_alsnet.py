"""Deney 4: ALSNet (Hassan ve ark., ICASSP 2022) kurgusunun yeniden uretimi ve bolme duzeyinin etkisi.

Kurgu (ALSNet ile ayni): EMGLab N2001, ALS ve saglikli; biceps brachii + vastus medialis;
her siniftan 151 kayit -> 302 kayit x 11 adet 1 sn'lik segment = 3322 segment; ikili siniflandirma.

Protokoller:
  rastgele_80_20     : segmentler rastgele %80/%20 (ALSNet'teki 665'lik test kumesiyle ayni boyut)
  segment_5kat       : segment duzeyinde tabakali 5-kat
  kayit_grup_5kat    : ayni kaydin segmentleri ayni katta
  hasta_grup_5kat    : ayni hastanin tum segmentleri ayni katta (StratifiedGroupKFold)
  hasta_LOSO         : her seferinde bir hasta test
Model: ALSNet tarifine yakin genisletilmis 1D-CNN (dilation 1,2,3,2,1; BN; global max pool; dense 128).
"""
from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.signal import decimate
from sklearn.model_selection import (LeaveOneGroupOut, StratifiedGroupKFold, StratifiedKFold,
                                     train_test_split)

from hanife_pipeline import preprocess
from n2001 import list_records

RES = Path(__file__).resolve().parent / "results"
DEC = 4
SEEDS = [0, 1, 2]


def load(rec):
    y = decimate(preprocess(rec.signal(), rec.fs), DEC, zero_phase=True)
    n = int(round(rec.fs / DEC))
    W = np.stack([y[k * n:(k + 1) * n] for k in range(11)]).astype(np.float32) / 1000.0
    return W, rec.group, rec.subject, rec.name


class ALSNetLike(nn.Module):
    def __init__(self):
        super().__init__()
        layers, c = [], 1
        for co, d in zip([32, 64, 64, 128, 128], [1, 2, 3, 2, 1]):
            layers += [nn.Conv1d(c, co, 7, stride=2, dilation=d, padding=3 * d), nn.BatchNorm1d(co), nn.ReLU()]
            c = co
        self.f = nn.Sequential(*layers)
        self.h = nn.Sequential(nn.Linear(c, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.3), nn.Linear(128, 1))

    def forward(self, x):
        return self.h(self.f(x.unsqueeze(1)).amax(-1)).squeeze(-1)


def train_predict(Xtr, ytr, Xte, dev, seed, epochs=30, bs=64):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = ALSNetLike().to(dev)
    opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=epochs * int(np.ceil(len(Xtr) / bs)))
    pos_w = torch.tensor((ytr == 0).sum() / max((ytr == 1).sum(), 1), dtype=torch.float32, device=dev)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    Xt, yt = torch.from_numpy(Xtr).to(dev), torch.from_numpy(ytr.astype(np.float32)).to(dev)
    for _ in range(epochs):
        m.train()
        perm = torch.randperm(len(Xt), device=dev)
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            if len(b) < 2:
                continue
            xb = -Xt[b] if rng.random() < 0.5 else Xt[b]
            opt.zero_grad()
            lossf(m(xb), yt[b]).backward()
            opt.step()
            sched.step()
    m.eval()
    with torch.no_grad():
        return np.concatenate([(m(torch.from_numpy(Xte[i:i + 512]).to(dev)) > 0).int().cpu().numpy()
                               for i in range(0, len(Xte), 512)])


def summarize(y, p, rec, subj):
    df = pd.DataFrame({"y": y, "p": p, "r": rec, "s": subj})
    out = {"segment_dogruluk": np.mean(y == p),
           "duyarlilik": np.mean(p[y == 1] == 1), "ozgulluk": np.mean(p[y == 0] == 0)}
    out["dengeli_dogruluk"] = (out["duyarlilik"] + out["ozgulluk"]) / 2
    for lvl, col in [("kayit", "r"), ("hasta", "s")]:
        v = df.groupby(col).agg(y=("y", "first"), p=("p", "mean"))
        ok = (v.y == (v.p >= 0.5)).astype(int)
        out[f"{lvl}_dogruluk"] = ok.mean()
        out[f"{lvl}_dogru/toplam"] = f"{ok.sum()}/{len(ok)}"
    return out


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    recs = [r for r in list_records() if r.group in ("ALS", "Normal") and r.muscle in ("BB", "VM")]
    rng = np.random.default_rng(2022)
    sel = []
    for g in ("ALS", "Normal"):
        cand = [r for r in recs if r.group == g]
        sel += [cand[i] for i in sorted(rng.choice(len(cand), 151, replace=False))]
    with Pool(16) as pool:
        data = pool.map(load, sel)
    X = np.concatenate([d[0] for d in data])
    y = np.concatenate([[int(d[1] == "ALS")] * 11 for d in data])
    s = np.concatenate([[d[2]] * 11 for d in data])
    r = np.concatenate([[d[3]] * 11 for d in data])
    print("segment:", X.shape, "ALS hasta:", sorted(set(s[y == 1])), "saglikli:", len(set(s[y == 0])), flush=True)

    idx = np.arange(len(y))
    rows = []
    for seed in SEEDS:
        protos = {}
        tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=seed)
        protos["rastgele_80_20"] = [(tr, te)]
        protos["segment_5kat"] = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y))
        protos["kayit_grup_5kat"] = list(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, r))
        protos["hasta_grup_5kat"] = list(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, s))
        if seed == SEEDS[0]:
            protos["hasta_LOSO"] = list(LeaveOneGroupOut().split(X, y, s))
        for name, splits in protos.items():
            p = np.full_like(y, -1)
            for k, (a, b) in enumerate(splits):
                p[b] = train_predict(X[a], y[a], X[b], dev, seed=seed * 100 + k)
            m = p >= 0
            res = summarize(y[m], p[m], r[m], s[m])
            ntest = int(m.sum())
            rows.append({"tohum": seed, "protokol": name, "test_segment": ntest, **res})
            print(seed, name, f"test={ntest}", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()}, flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(RES / "deney4_alsnet.csv", index=False)
    num = out.select_dtypes("number").columns.drop(["tohum", "test_segment"])
    print(out.groupby("protokol", sort=False)[list(num)].agg(["mean", "std"]).round(3).to_string())
