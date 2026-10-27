"""Deney 3: ham EMG pencereleri uzerinde 1D-CNN; pencere duzeyinde rastgele bolme ile
hasta duzeyinde gruplu bolmenin karsilastirilmasi (3 sinif).

Sinyal: 10-1500 Hz filtrelenmis, 4 kat seyreltilmis (~5.86 kHz), 1 sn'lik ortusmeyen pencereler.
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
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from hanife_pipeline import preprocess
from n2001 import list_records

RES = Path(__file__).resolve().parent / "results"
CLASSES = ["ALS", "Myopathy", "Normal"]
DEC = 4


def load(rec):
    y = decimate(preprocess(rec.signal(), rec.fs), DEC, zero_phase=True)
    n = int(round(rec.fs / DEC))
    W = np.stack([y[k * n:(k + 1) * n] for k in range(len(y) // n)]).astype(np.float32) / 1000.0
    return W, rec.group, rec.subject, rec.name


class Net(nn.Module):
    def __init__(self, n_cls=3):
        super().__init__()
        layers, c = [], 1
        for co in [16, 32, 64, 64, 128, 128]:
            layers += [nn.Conv1d(c, co, 7, padding=3), nn.BatchNorm1d(co), nn.ReLU(), nn.MaxPool1d(3)]
            c = co
        self.f = nn.Sequential(*layers)
        self.h = nn.Sequential(nn.Dropout(0.3), nn.Linear(c, n_cls))

    def forward(self, x):
        return self.h(self.f(x.unsqueeze(1)).mean(-1))


def train_eval(Xtr, ytr, Xte, dev, epochs=30, seed=0, bs=64):
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = Net().to(dev)
    opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-4)
    steps = epochs * int(np.ceil(len(Xtr) / bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps)
    w = torch.tensor(len(ytr) / (3 * np.bincount(ytr, minlength=3)), dtype=torch.float32, device=dev)
    lossf = nn.CrossEntropyLoss(weight=w)
    Xt, yt = torch.from_numpy(Xtr).to(dev), torch.from_numpy(ytr).to(dev)
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(len(Xt), device=dev)
        correct = 0
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            xb = Xt[b]
            if np.random.rand() < 0.5:   # basit veri artirma: isaret cevirme
                xb = -xb
            opt.zero_grad()
            out = m(xb)
            lossf(out, yt[b]).backward()
            opt.step()
            sched.step()
            correct += (out.argmax(1) == yt[b]).sum().item()
    pass
    m.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(Xte), 512):
            out.append(m(torch.from_numpy(Xte[i:i + 512]).to(dev)).argmax(1).cpu().numpy())
    return np.concatenate(out)


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    with Pool(24) as p:
        data = p.map(load, list_records())
    X = np.concatenate([d[0] for d in data])
    y = np.concatenate([[CLASSES.index(d[1])] * len(d[0]) for d in data])
    s = np.concatenate([[d[2]] * len(d[0]) for d in data])
    print("pencere:", X.shape, "cihaz:", dev, flush=True)

    rows = []
    protocols = {
        "pencere_rastgele_5kat": StratifiedKFold(5, shuffle=True, random_state=0).split(X, y),
        "hasta_grup_5kat": StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, s),
    }
    for pname, splits in protocols.items():
        p = np.empty_like(y)
        for k, (tr, te) in enumerate(splits):
            p[te] = train_eval(X[tr], y[tr], X[te], dev, seed=k)
            print(pname, "kat", k, "dogruluk", np.mean(p[te] == y[te]).round(3),
                  "test hastalari", sorted(map(str, set(s[te]))) if "hasta" in pname else "-", flush=True)
        df = pd.DataFrame({"y": y, "p": p, "s": s})
        vote = df.groupby("s").agg(y=("y", "first"), p=("p", lambda v: v.value_counts().idxmax()))
        rows.append({"protokol": pname, "pencere_dogruluk": np.mean(p == y),
                     "dengeli_dogruluk": np.mean([np.mean(p[y == c] == c) for c in range(3)]),
                     **{f"duyarlilik_{c}": np.mean(p[y == i] == i) for i, c in enumerate(CLASSES)},
                     "hasta_dogruluk": np.mean(vote.y == vote.p),
                     "hasta_dogru/toplam": f"{int((vote.y == vote.p).sum())}/{len(vote)}"})
    out = pd.DataFrame(rows)
    out.to_csv(RES / "deney3_cnn.csv", index=False)
    print(out.to_string())
