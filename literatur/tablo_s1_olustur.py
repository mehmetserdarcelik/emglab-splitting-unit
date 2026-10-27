"""Supplementary Table S1 (English): literatur_tablosu.csv -> Table_S1_coded_studies.csv / .xlsx

Turkce serbest metin hucreleri ceviri_tablo_en.json ile cevrilir (benzersiz deger sirasi, asagidaki COLS sirasi).
Kor kodlanan 25 calisma icin kanit: kodu ortak kararla ayni olan modelin birebir alintisi (oncelik: Sonnet, Astra, Argon).
"""
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
MAK = HERE.parent / "makale"
sys.stdout.reconfigure(encoding="utf-8")

df = pd.read_csv(HERE / "literatur_tablosu.csv", sep=";", encoding="utf-8-sig", dtype=str).fillna("")
COLS = ["grup", "veri", "gorev", "ornek_birimi", "bolme", "bildirilen_dogruluk", "durum", "not_"]
vals = []
for c in COLS:
    for v in df[c].unique():
        if v and v not in vals:
            vals.append(v)
en = json.loads((HERE / "ceviri_tablo_en.json").read_text(encoding="utf-8"))
assert len(en) == len(vals), (len(en), len(vals))
tr = dict(zip(vals, en))
T = lambda v: tr.get(v, v)

CODE = {"P": "S", "K": "R", "H": "P", "H*": "P*", "?": "U"}
SEMG = {"W1993132926", "W2010824802", "W2229307046", "W4292656326"}
PHYSIONET = {"W2201436378"}
CONTRADICTED = {"W4205486194", "W4313527089", "W4392255538", "W7207759719"}  # Hao, Wang, Putro, ALSGate (+ ALSNet asagida)
NOT_VERIFIABLE = {"W2787914842", "W4386907835", "W4386908193"}  # Hazarika 2018, Rahman x2


def group(r):
    if r.grup.startswith("EMGLab"):
        return "EMGLab N2001"
    if r.grup.startswith("dislandi"):
        return "Excluded (retracted)"
    if r.id in SEMG:
        return "Other data: surface EMG"
    if r.id in PHYSIONET:
        return "Other data: PhysioNet"
    return "Other data: needle EMG"


def pstar(r):
    if CODE[r.bolme_kodu] != "P*":
        return ""
    if r.id in NOT_VERIFIABLE:
        return "not verifiable"
    return "contradicted"


LOC = [("Özetçe", "Abstract"), ("Özet", "Abstract"), ("Giriş", "Introduction"), ("Sonuç", "Conclusion"), ("son paragraf", "last paragraph"),
       ("Bölüm", "Section"), ("Tablo", "Table"), ("Şekil", "Fig."), ("Sayfa", "p."), ("Ek ", "Appendix "), ("s. ", "p. "), (" ve ", " and ")]


def loc_en(v):
    for a, b in LOC:
        v = v.replace(a, b)
    return v


# kor kodlama kanitlari
key = pd.read_csv(HERE / "kor_kodlama_anahtar.csv", sep=";", encoding="utf-8-sig", dtype=str)
models = {m: pd.read_csv(HERE / "kor_kodlama_cevaplar" / f"KOR_{m}.csv", sep=";", encoding="utf-8-sig", dtype=str)
          .assign(no=lambda d: d.no.astype(str)).set_index("no") for m in ("SONNET", "ASTRA", "ARGON")}
evidence = {}
for _, k in key.iterrows():
    final = CODE[df.loc[df.id == k.id, "bolme_kodu"].iat[0]]
    for m, d in models.items():
        x = d.loc[k.no]
        if x.bolme_kodu == final and isinstance(x.kanit_alinti, str) and x.kanit_alinti.strip():
            evidence[k.id] = f'{x.kanit_alinti.strip()} [{loc_en(x.sayfa_bolum)}]' if isinstance(x.sayfa_bolum, str) else x.kanit_alinti.strip()
            break

out = pd.DataFrame({
    "OpenAlex ID": df.id,
    "Year": df.yil,
    "First author": df.ilk_yazar.str.replace("Kucuk", "Küçük"),
    "Title": df.baslik,
    "Venue": df.dergi_kongre,
    "DOI": df.doi,
    "Group": df.apply(group, axis=1),
    "Data": df.veri.map(T),
    "Task": df.gorev.map(T),
    "Unit of classification": df.ornek_birimi.map(T),
    "Split procedure (as described)": df.bolme.map(T),
    "Split code": df.bolme_kodu.map(CODE),
    "P* subtype": df.apply(pstar, axis=1),
    "Highest reported accuracy (%)": df.bildirilen_dogruluk.map(T),
    "Coded from": df.durum.map(T),
    "Blind re-coded": df.id.isin(set(key.id)).map({True: "yes", False: ""}),
    "Verbatim evidence (blind re-coding)": df.id.map(evidence).fillna(""),
    "Notes": df.not_.map(T),
})
order = {"EMGLab N2001": 0, "Other data: needle EMG": 1, "Other data: surface EMG": 2, "Other data: PhysioNet": 3, "Excluded (retracted)": 4}
out = out.sort_values(["Group", "Year", "First author"], key=lambda s: s.map(order) if s.name == "Group" else s).reset_index(drop=True)

# kalan Turkce karakter / kelime kontrolu
left = out[["Data", "Task", "Unit of classification", "Split procedure (as described)", "Highest reported accuracy (%)", "Coded from",
            "Verbatim evidence (blind re-coding)", "Notes"]] \
    .apply(lambda s: s.str.contains(r"\b(?:kayit|ornek|sinif|egitim|belirsiz|ve|Bölüm|Tablo|Sayfa|Giriş|Özet)\b", regex=True)).any(axis=1)
assert not left.any(), out[left].iloc[:, 7:].to_string()

out.to_csv(MAK / "Table_S1_coded_studies.csv", index=False, encoding="utf-8-sig")

legend = pd.DataFrame({"Item": [
    "Split code S", "Split code R", "Split code P", "Split code P*", "Split code U",
    "P* subtype", "Group", "Data abbreviations", "Highest reported accuracy", "Blind re-coded", "Verbatim evidence"],
    "Definition": [
    "Segment-level random split: segments, frames or MUAPs of the same recording may appear in both training and test sets.",
    "Recording-level split: whole recordings are assigned to folds, but different recordings of the same participant may appear in both sets.",
    "Participant-disjoint split, stated explicitly and consistent with the reported numbers.",
    "Participant-disjoint split claimed but not supported: contradicted by released data, code or reported numbers, or not verifiable from the information given.",
    "Unit of classification or split procedure cannot be determined from the paper.",
    "'contradicted' or 'not verifiable' (see Section 2.2).",
    "EMGLab N2001; other data (needle EMG: used for comparison; surface EMG and PhysioNet: listed but not used for comparison); excluded (retracted).",
    "C = healthy controls, M = myopathy, N = normal, BB = biceps brachii, VM = vastus medialis, DE = deltoid, CV = cross-validation, LOOCV = leave-one-out cross-validation.",
    "As published; several values for different tasks are separated by '/'. Test results were used rather than training results.",
    "One of 25 EMGLab studies re-coded blind by two authors (consensus) and three language models (Section 2.2); the consensus code is shown.",
    "Verbatim quotation supplied by an independent blind coder whose code agreed with the consensus.",
]})
with pd.ExcelWriter(MAK / "Table_S1_coded_studies.xlsx", engine="openpyxl") as xw:
    out.to_excel(xw, sheet_name="Table S1", index=False)
    legend.to_excel(xw, sheet_name="Legend", index=False)
    ws = xw.sheets["Table S1"]
    from openpyxl.styles import Alignment, Font
    widths = [14, 6, 22, 50, 30, 26, 22, 34, 16, 26, 34, 8, 14, 22, 18, 9, 60, 50]
    for i, w in enumerate(widths):
        ws.column_dimensions[chr(65 + i)].width = w
    for c in ws[1]:
        c.font = Font(bold=True)
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "D2"
    xw.sheets["Legend"].column_dimensions["A"].width = 26
    xw.sheets["Legend"].column_dimensions["B"].width = 120

print(out.Group.value_counts().to_dict())
print(out[out.Group == "EMGLab N2001"]["Split code"].value_counts().to_dict())
print("kanitli satir:", (out["Verbatim evidence (blind re-coding)"] != "").sum())
