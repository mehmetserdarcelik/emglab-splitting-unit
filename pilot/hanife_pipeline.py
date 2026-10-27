"""Kucuk (2015) tezindeki 5 asamali yapinin Python ile yeniden kurulumu.

Asamalar: on isleme -> MUAP bolutleme -> melez (Kohonen + BCO) kumeleme
          -> oznitelik cikarma -> (siniflandirma: run_pilot.py)

Tezdeki EK A Matlab kodlari esas alinmistir. Belirsiz kalan yerlerde yapilan
varsayimlar ASSUMPTION etiketiyle isaretlenmistir.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import cheby1, find_peaks, sosfiltfilt

# ---------------------------------------------------------------- 1) On isleme
def preprocess(x: np.ndarray, fs: float, band=(10.0, 1500.0), order=4, ripple=0.05) -> np.ndarray:
    """Chebyshev-I bant geciren (10-1500 Hz), sifir fazli. Tezde cheby1(4, 0.05, Wn) + filtfilt.

    Not: Tez kodunda fs=10000 kullanilmis; veri 23437.5 Hz. Burada gercek fs kullanilir.
    """
    sos = cheby1(order, ripple, band, btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


# ---------------------------------------------------------------- 2) Bolutleme
def segment_muaps(x: np.ndarray, fs: float, thr_uv=50.0, tol_uv=40.0, win_ms=1.0,
                  max_len=2000) -> list[tuple[int, int, int]]:
    """MUAP baslangic/bitis noktalarini bulur (tez Adim 1-3).

    Adim 1: esigi (50 uV) asan tepeler.
    Adim 2-3: 1 ms kayan pencere; pencerenin tum noktalari medyan +/- 40 uV icindeyse
              pencere ortasi baslangic/bitis adayi olur (deger 50 uV ustundeyse alinmaz).
    Bir tepe grubunun solundaki ilk aday baslangic, sagindaki ilk aday bitistir.
    Donus: (baslangic, bitis, en buyuk tepe indisi) listesi.
    """
    w = max(int(round(win_ms * 1e-3 * fs)), 2)
    nwin = len(x) // w
    X = x[: nwin * w].reshape(nwin, w)
    med = np.median(X, axis=1)
    quiet = np.all(np.abs(X - med[:, None]) <= tol_uv, axis=1)
    cand = np.arange(nwin) * w + w // 2
    cand = cand[quiet & (x[cand] <= thr_uv)]
    peaks, _ = find_peaks(x, height=thr_uv)
    if len(peaks) == 0 or len(cand) < 2:
        return []

    idx = np.concatenate([cand, peaks])
    flag = np.concatenate([np.zeros(len(cand), int), np.ones(len(peaks), int)])
    o = np.argsort(idx, kind="stable")
    idx, flag = idx[o], flag[o]

    segs = []
    start = None
    for i in range(len(idx) - 1):
        if flag[i] == 0 and flag[i + 1] == 1:
            start = idx[i]
        elif flag[i] == 1 and flag[i + 1] == 0 and start is not None:
            end = idx[i + 1]
            if 2 < end - start <= max_len:
                pk = start + int(np.argmax(x[start:end + 1]))
                segs.append((int(start), int(end), pk))
            start = None
    return segs


def align_segments(x: np.ndarray, segs) -> np.ndarray:
    """MUAP'lari en buyuk tepelerine gore hizalar; disarida kalan kisimlar sifir (tezdeki gibi)."""
    left = max(pk - s for s, e, pk in segs)
    right = max(e - pk for s, e, pk in segs)
    out = np.zeros((len(segs), left + right + 1))
    for k, (s, e, pk) in enumerate(segs):
        seg = x[s:e + 1]
        off = left - (pk - s)
        out[k, off:off + len(seg)] = seg
    return out


# ---------------------------------------------------------------- 3) Melez kumeleme
def kohonen_leader(P: np.ndarray, delta: float, max_units=15, passes=1):
    """Tezdeki 'selfalt': yaricap (delta) tabanli, birim yaratan rekabetci ogrenme.

    ASSUMPTION: hesap suresi icin birim sayisi max_units ile sinirlandi (tezde sinir yok).
    """
    W = [P[0].copy()]
    ns = [1]
    for _ in range(passes):
        for p in P:
            d = np.linalg.norm(np.asarray(W) - p, axis=1)
            j = int(np.argmin(d))
            if d[j] < delta or len(W) >= max_units:
                ns[j] += 1
                W[j] += (0.6 / ns[j]) * (p - W[j])
            else:
                W.append(p.copy())
                ns.append(1)
    return np.asarray(W)


def fcm_validity(P: np.ndarray, init: np.ndarray, m=2.0, iters=10, tol=1e-6):
    """c = 1..cm icin BCO; her c icin degistirilmis bolum entropisi H (tez Es. 7).

    Optimal c: H(c-1) < H(c) ve H(c-1) < H(c-2) kosulunu saglayanlar arasinda en kucuk H.
    Hicbiri saglamazsa Kohonen merkezlerinin tamami kullanilir (tez kodundaki davranis).
    """
    cm = len(init)
    n = len(P)
    p2 = np.sum(P ** 2, axis=1)
    U_all, C_all, H = {}, {}, np.zeros(cm + 1)
    for c in range(1, cm + 1):
        C = init[:c].copy()
        for _ in range(iters):
            # ||p - c||, matris carpimiyla (c x n)
            D = np.sqrt(np.maximum(np.sum(C ** 2, axis=1)[:, None] + p2[None, :] - 2 * C @ P.T, 0))
            D[D == 0] = 1e-6
            inv = D ** (-2.0 / (m - 1))
            U = inv / inv.sum(axis=0, keepdims=True)
            Um = U ** m
            Cn = (Um @ P) / Um.sum(axis=1, keepdims=True)
            shift = np.linalg.norm(Cn - C)
            C = Cn
            if shift < tol:
                break
        mu = U.max(axis=0)
        H[c] = -np.sum(mu * np.log(mu)) / n
        U_all[c], C_all[c] = U, C

    cands = [i - 1 for i in range(cm, 2, -1) if H[i - 1] < H[i] and H[i - 1] < H[i - 2]]
    if not cands:
        c_opt = cm
    else:
        c_opt = min(cands, key=lambda c: H[c])
    return c_opt, U_all[c_opt], C_all[c_opt]


def zero_crossings(t: np.ndarray, thr: float) -> int:
    """Esikli sifir gecis sayisi (|x| < esik olan noktalar yok sayilir)."""
    s = np.sign(t[np.abs(t) >= thr])
    return int(np.sum(s[1:] != s[:-1]))


def hybrid_templates(aligned: np.ndarray, delta=2.0, max_units=15):
    """Melez kumeleme; tek fazli sablonlar atilir (tez 'tekfaz_kurtul').

    Donus: uV cinsinden sablonlar (k x L) ve kume buyuklukleri.
    """
    h = np.max(np.abs(aligned))
    P = aligned / h
    W = kohonen_leader(P, delta, max_units=max_units)
    c, U, _ = fcm_validity(P, W)
    lab = np.argmax(U, axis=0) if c > 1 else np.zeros(len(P), int)
    temps, sizes = [], []
    for k in range(c):
        members = P[lab == k]
        if len(members) == 0:
            continue
        t = members.mean(axis=0)
        # ASSUMPTION: tez kodunda 'zn > 1'; zc fonksiyonu verilmedigi icin tek fazli
        # (hic sifir gecisi olmayan) sablonlar atilir, iki ve daha fazla fazlilar tutulur.
        if zero_crossings(t, 0.1 * np.max(t)) >= 1:
            temps.append(t * h)
            sizes.append(len(members))
    return np.asarray(temps), np.asarray(sizes)


# ---------------------------------------------------------------- 4) Oznitelikler
TIME_FEATS = ["IEMG", "MAV", "VAR", "RMS", "STD", "SGS", "WAMP"]
FREQ_FEATS = ["MF", "MDF", "PKF", "MNP", "SM1", "SM2", "SM3"]
MORPH_FEATS = ["Sure", "Genlik", "Faz"]
ALL_FEATS = TIME_FEATS + FREQ_FEATS + MORPH_FEATS


def trim_template(t: np.ndarray, frac=0.02) -> np.ndarray:
    """ASSUMPTION: sablon, hizalama sirasinda eklenen sifirlardan arindirilir."""
    nz = np.flatnonzero(np.abs(t) >= frac * np.max(np.abs(t)))
    return t[nz[0]: nz[-1] + 1] if len(nz) else t


def template_features(t: np.ndarray, fs: float, thr_uv=10.0, nfft=4096) -> dict:
    """ASSUMPTION: SGS/WAMP esigi 10 uV; sure/faz esigi 10 uV (tezde sayisal deger verilmemis)."""
    x = trim_template(t)
    N = len(x)
    f = {}
    f["IEMG"] = np.sum(np.abs(x))
    f["MAV"] = np.mean(np.abs(x))
    f["VAR"] = np.sum(x ** 2) / max(N - 1, 1)
    f["RMS"] = np.sqrt(np.mean(x ** 2))
    f["STD"] = np.std(x)
    s = np.sign(x)
    f["SGS"] = np.sum((s[:-1] * s[1:] < 0) & (np.abs(x[:-1] - x[1:]) >= thr_uv))
    f["WAMP"] = np.sum(np.abs(np.diff(x)) >= thr_uv)

    X = np.fft.rfft(x, n=max(nfft, N))
    P = np.abs(X) ** 2 / len(X)
    fr = np.fft.rfftfreq(max(nfft, N), 1 / fs)
    band = (fr >= 10) & (fr <= 1500)
    P, fr = P[band], fr[band]
    Ps = P.sum()
    f["MF"] = np.sum(fr * P) / Ps
    f["MDF"] = fr[np.searchsorted(np.cumsum(P), Ps / 2)]
    f["PKF"] = fr[np.argmax(P)]
    f["MNP"] = np.mean(P)
    f["SM1"] = np.sum(P * fr)
    f["SM2"] = np.sum(P * fr ** 2)
    f["SM3"] = np.sum(P * fr ** 3)

    above = np.flatnonzero(np.abs(x) >= thr_uv)
    f["Sure"] = (above[-1] - above[0] + 1) / fs * 1e3 if len(above) else 0.0   # ms
    f["Genlik"] = np.max(x) - np.min(x)                                        # tepe-tepe uV
    f["Faz"] = zero_crossings(x, thr_uv) + 1
    return f


def record_features(x_uv: np.ndarray, fs: float) -> tuple[dict | None, dict]:
    """Tek kayit icin tum yapi. ASSUMPTION: kayit oznitelik vektoru = sablon ozniteliklerinin ortalamasi."""
    y = preprocess(x_uv, fs)
    segs = segment_muaps(y, fs)
    info = {"n_muap": len(segs)}
    if len(segs) < 3:
        return None, info
    aligned = align_segments(y, segs)
    temps, sizes = hybrid_templates(aligned)
    info.update(n_templates=len(temps), muap_len=aligned.shape[1])
    if len(temps) == 0:
        return None, info
    F = [template_features(t, fs) for t in temps]
    return {k: float(np.mean([d[k] for d in F])) for k in ALL_FEATS}, info
