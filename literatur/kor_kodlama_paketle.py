"""Kor kodlama paketi: 25 calisma (tum P* ve P + tabakali rastgele 17).

Cikti:
  kor_kodlama/                  modellere ve insan kodlayicilara verilecek paket (kod ipucu yok)
  kor_kodlama_anahtar.csv       bizim kodlarimiz (pakete KONMAZ; kodlama bitmeden acilmamali)
"""
import shutil
import sys
from pathlib import Path

import pandas as pd
from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
PREP = HERE / "kor_kodlama_hazirlik"
OUT = HERE / "kor_kodlama"
sys.stdout.reconfigure(encoding="utf-8")

lit = pd.read_csv(HERE / "literatur_tablosu.csv", sep=";", encoding="utf-8-sig", dtype=str).fillna("")
pdfmap = pd.read_csv(HERE / "pdf_eslesme.csv", sep=";", encoding="utf-8-sig", dtype=str).fillna("").set_index("id").pdf
audited = set(pd.read_csv(HERE / "denetim_paketi" / "DENETIM_TABLOSU.csv", sep=";", encoding="utf-8-sig", dtype=str).id)

e = lit[lit.grup.str.startswith("EMGLab")].copy()
e["kaynak"] = e.id.map(lambda i: pdfmap.get(i, "") or (str(PREP / f"{i}.txt") if (PREP / f"{i}.txt").exists() else ""))
e = e[e.kaynak != ""]

must = e[e.bolme_kodu.isin(["H*", "H"])]
pool = e[~e.id.isin(must.id) & ~e.id.isin(audited)]
rand = pd.concat([pool[pool.bolme_kodu == k].sample(n, random_state=2026) for k, n in [("P", 6), ("K", 6), ("?", 5)]])
sel = pd.concat([must, rand]).sample(frac=1, random_state=7).reset_index(drop=True)  # sira karistirilir
assert len(sel) == 25, len(sel)

if OUT.exists():
    shutil.rmtree(OUT)
(OUT / "pdf").mkdir(parents=True)
(OUT / "txt").mkdir()
rows, key = [], []
for n, x in sel.iterrows():
    sur = x.ilk_yazar.split()[-1].replace("ı", "i").replace("ş", "s").replace("ü", "u").replace("ç", "c").replace("ğ", "g").replace("ö", "o")
    stem = f"{n + 1:02d}_{x.yil}_{sur}"
    src = Path(x.kaynak)
    if src.suffix.lower() == ".pdf":
        shutil.copy(src, OUT / "pdf" / f"{stem}.pdf")
        txt = "\n".join((p.extract_text() or "") for p in PdfReader(src).pages)
        dosya = f"{stem}.pdf"
    else:
        txt = src.read_text(encoding="utf-8")
        dosya = f"{stem}.txt (yalnizca txt)"
    (OUT / "txt" / f"{stem}.txt").write_text(txt, encoding="utf-8")
    rows.append({"no": n + 1, "dosya": dosya, "yil": x.yil, "ilk_yazar": x.ilk_yazar, "baslik": x.baslik,
                 "veri": "", "gorev": "", "ornek_birimi": "", "bolme_tarifi": "", "bolme_kodu": "", "pstar_turu": "",
                 "kanit_alinti": "", "sayfa_bolum": "", "dogruluk": "", "emin_misiniz": "", "not": ""})
    kod = {"P": "S", "K": "R", "H": "P", "H*": "P*", "?": "U"}[x.bolme_kodu]
    key.append({"no": n + 1, "id": x.id, "ilk_yazar": x.ilk_yazar, "yil": x.yil, "bizim_kod": kod,
                "secim": "P*/P hepsi" if x.id in set(must.id) else "rastgele"})

form = pd.DataFrame(rows)
form.to_csv(OUT / "KODLAMA_FORMU.csv", sep=";", index=False, encoding="utf-8-sig")
pd.DataFrame(key).to_csv(HERE / "kor_kodlama_anahtar.csv", sep=";", index=False, encoding="utf-8-sig")
for f in ["YONERGE.md", "GOREV_MODEL.md"]:
    shutil.copy(PREP / f, OUT / f)

# Veri ozeti: eski denetim paketinden yalnizca veri bilgileri (karar kurallari ve eski kod harfleri yok)
oz = (HERE / "denetim_paketi" / "EMGLAB_VERI_OZETI.md").read_text(encoding="utf-8").split("## Karar verirken")[0]
oz = oz.replace(", ALSNet kurgusu", "")
oz += "\nKarar kuralları için `YONERGE.md` dosyasına bakın.\n"
(OUT / "EMGLAB_VERI_OZETI.md").write_text(oz, encoding="utf-8")

# Insan kodlayicilar icin Excel formu (acilir listeli)
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

wb = Workbook()
ws = wb.active
ws.title = "Kodlama"
ws.append(list(form.columns))
for r in form.itertuples(index=False):
    ws.append(list(r))
widths = {"A": 5, "B": 26, "C": 7, "D": 20, "E": 45, "F": 18, "G": 16, "H": 14, "I": 35, "J": 11, "K": 13, "L": 50, "M": 12, "N": 10, "O": 11, "P": 30}
for c, w in widths.items():
    ws.column_dimensions[c].width = w
for cell in ws[1]:
    cell.font = Font(bold=True)
    cell.fill = PatternFill("solid", fgColor="EEF3F9")
for row in ws.iter_rows(min_row=2):
    for cell in row:
        cell.alignment = Alignment(wrap_text=True, vertical="top")
dv = DataValidation(type="list", formula1='"S,R,P,P*,U"', allow_blank=True)
dv2 = DataValidation(type="list", formula1='"celiski,dogrulanamaz"', allow_blank=True)
dv3 = DataValidation(type="list", formula1='"1,2,3"', allow_blank=True)
for d, col in [(dv, "J"), (dv2, "K"), (dv3, "O")]:
    ws.add_data_validation(d)
    d.add(f"{col}2:{col}26")
ws.freeze_panes = "F2"
wb.save(OUT / "KODLAMA_FORMU.xlsx")

print("secilen:", len(sel), "| zorunlu:", len(must), "| rastgele:", len(rand))
print(pd.DataFrame(key).bizim_kod.value_counts().to_dict())
print(sorted(p.name for p in OUT.iterdir()))
