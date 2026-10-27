"""Pilot deney: dogrulama protokolunun basarim uzerindeki etkisi.

Deney 1 (tez yapisi, kayit basina 1 oznitelik vektoru):
  - kayit_rastgele_10kat : tezdeki gibi kayitlar rastgele 10 parcaya bolunur (10 farkli tohum)
  - hasta_LOSO           : her seferinde bir hastanin TUM kayitlari test edilir

Deney 2 (1 sn'lik pencereler, derin ogrenme makalelerindeki yaygin kurgu):
  - pencere_rastgele_10kat : pencereler rastgele bolunur
  - kayit_grup_10kat       : ayni kaydin pencereleri ayni parcada
  - hasta_LOSO             : ayni hastanin tum pencereleri ayni parcada
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC

RES = Path(__file__).resolve().parent / "results"

TASKS = {  # gorev: (siniflar, pozitif sinif)
    "ALS-Saglikli": (["ALS", "Normal"], "ALS"),
    "ALS-Miyopati": (["ALS", "Myopathy"], "ALS"),
    "Miyopati-Saglikli": (["Myopathy", "Normal"], "Myopathy"),
    "3-sinif": (["ALS", "Myopathy", "Normal"], None),
}

FEATURE_SETS = {
    **{f: [f] for f in ["IEMG", "MAV", "VAR", "RMS", "STD", "SGS", "WAMP",
                        "MF", "MDF", "PKF", "MNP", "SM1", "SM2", "SM3", "Sure", "Genlik", "Faz"]},
    "IEMG+WAMP+MF": ["IEMG", "WAMP", "MF"],
    "IEMG+SGS+WAMP+MF": ["IEMG", "SGS", "WAMP", "MF"],
    "IEMG+RMS+WAMP+MF": ["IEMG", "RMS", "WAMP", "MF"],
    "IEMG+MNP+WAMP+MF": ["IEMG", "MNP", "WAMP", "MF"],
    "IEMG+Sure+Genlik+SGS": ["IEMG", "Sure", "Genlik", "SGS"],
    "IEMG+Sure+Genlik+Faz": ["IEMG", "Sure", "Genlik", "Faz"],
    "IEMG+SGS+Genlik+Faz": ["IEMG", "SGS", "Genlik", "Faz"],
    "Sure+Genlik+Faz": ["Sure", "Genlik", "Faz"],
}
BEST_SET = "IEMG+SGS+WAMP+MF"  # tezde en iyi sonuc veren coklu oznitelik vektoru

# Guc tabanli oznitelikler cok genis aralikta; log donusumu tum siniflandiricilara ayni uygulanir.
_log = FunctionTransformer(lambda X: np.sign(X) * np.log1p(np.abs(X)))


def classifiers():
    # Tez: DVM-RTF sigma=1.5 (gamma = 1/(2*sigma^2)), C=400; Matlab svmtrain autoscale=varsayilan acik
    return {
        "DVM-Dogrusal": make_pipeline(_log, StandardScaler(), SVC(kernel="linear", C=1.0)),
        "DVM-RTF": make_pipeline(_log, StandardScaler(), SVC(kernel="rbf", C=400, gamma=1 / (2 * 1.5 ** 2))),
        "K-EYK(K=1)": make_pipeline(_log, StandardScaler(), KNeighborsClassifier(1)),
        "K-EYK(K=11)": make_pipeline(_log, StandardScaler(), KNeighborsClassifier(11)),
        "DDA": make_pipeline(_log, StandardScaler(), LinearDiscriminantAnalysis()),
        "KDA": make_pipeline(_log, StandardScaler(), QuadraticDiscriminantAnalysis(reg_param=1e-3)),
    }


def metrics(y, p, subj, pos):
    out = {"dogruluk": np.mean(y == p)}
    cls = np.unique(y)
    out["dengeli_dogruluk"] = np.mean([np.mean(p[y == c] == c) for c in cls])
    if pos is not None:
        out["duyarlilik"] = np.mean(p[y == pos] == pos)
        out["ozgulluk"] = np.mean(p[y != pos] != pos)
    # hasta duzeyi: hastanin kayitlarinin/pencerelerinin cogunluk oyu
    df = pd.DataFrame({"y": y, "p": p, "s": subj})
    vote = df.groupby("s").agg(y=("y", "first"), p=("p", lambda v: v.value_counts().idxmax()))
    out["hasta_dogruluk"] = np.mean(vote.y == vote.p)
    out["hasta_dogru/toplam"] = f"{int((vote.y == vote.p).sum())}/{len(vote)}"
    return out


def run_cv(X, y, groups_subj, splitter, clf, groups=None):
    p = np.empty_like(y)
    for tr, te in splitter.split(X, y, groups):
        clf.fit(X[tr], y[tr])
        p[te] = clf.predict(X[te])
    return p


def exp1(df: pd.DataFrame, seeds=range(10)) -> pd.DataFrame:
    rows = []
    for task, (cls, pos) in TASKS.items():
        d = df[df.group.isin(cls)].reset_index(drop=True)
        y, s = d.group.values, d.subject.values
        for fs_name, cols in FEATURE_SETS.items():
            X = d[cols].values.astype(float)
            clfs = classifiers() if fs_name == BEST_SET else {"DVM-RTF": classifiers()["DVM-RTF"]}
            for cname, clf in clfs.items():
                res = []
                for sd in seeds:
                    p = run_cv(X, y, s, StratifiedKFold(10, shuffle=True, random_state=sd), clf)
                    res.append(metrics(y, p, s, pos))
                r = pd.DataFrame(res).drop(columns="hasta_dogru/toplam").mean().to_dict()
                rows.append({"gorev": task, "oznitelik": fs_name, "siniflandirici": cname,
                             "protokol": "kayit_rastgele_10kat", "n_kayit": len(d), **r})
                p = run_cv(X, y, s, LeaveOneGroupOut(), clf, groups=s)
                rows.append({"gorev": task, "oznitelik": fs_name, "siniflandirici": cname,
                             "protokol": "hasta_LOSO", "n_kayit": len(d), **metrics(y, p, s, pos)})
        print("deney1", task, "tamam", flush=True)
    return pd.DataFrame(rows)


def exp2(dw: pd.DataFrame, cols=("IEMG", "SGS", "WAMP", "MF", "MDF", "RMS", "MNP")) -> pd.DataFrame:
    rows = []
    cols = list(cols)
    for task, (cls, pos) in TASKS.items():
        d = dw[dw.group.isin(cls)].reset_index(drop=True)
        X, y, s, r = d[cols].values.astype(float), d.group.values, d.subject.values, d.record.values
        protocols = {
            "pencere_rastgele_10kat": (StratifiedKFold(10, shuffle=True, random_state=0), None),
            "kayit_grup_10kat": (GroupKFold(10), r),
            "hasta_LOSO": (LeaveOneGroupOut(), s),
        }
        for cname in ["DVM-RTF", "K-EYK(K=1)", "DDA"]:
            for pname, (spl, g) in protocols.items():
                p = run_cv(X, y, s, spl, classifiers()[cname], groups=g)
                rows.append({"gorev": task, "siniflandirici": cname, "protokol": pname,
                             "n_pencere": len(d), **metrics(y, p, s, pos)})
        print("deney2", task, "tamam", flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = pd.read_csv(RES / "features_records.csv")
    print("kayit:", len(df), "| oznitelik cikarilabilen:", int(df.ok.sum()))
    print(df[df.ok].groupby("group").agg(kayit=("record", "size"), hasta=("subject", "nunique")))
    df = df[df.ok].reset_index(drop=True)

    r1 = exp1(df)
    r1.to_csv(RES / "deney1_tez_yapisi.csv", index=False)

    dw = pd.read_csv(RES / "features_windows.csv")
    r2 = exp2(dw)
    r2.to_csv(RES / "deney2_pencereler.csv", index=False)
    print("bitti")
