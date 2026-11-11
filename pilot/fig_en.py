"""Makale sekillerinin Ingilizce surumleri.

Turkce cizim kodlari (fig_final, fig_literatur, kaggle_kanit) aynen calistirilir; savefig aninda
sekildeki tum metinler TR -> EN cevrilir ve dosya makale/figures_en/ altina Ingilizce adla yazilir.
Turkce sekiller (Ilyas hoca raporu) degismez.

Kullanim: python -I fig_en.py <kaggle_zip>
"""
from __future__ import annotations

import re
import runpy
import sys
from pathlib import Path

import matplotlib
import matplotlib.ticker
from matplotlib.figure import Figure
from matplotlib.text import Text

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = HERE.parent / "makale" / "figures_en"
OUT.mkdir(parents=True, exist_ok=True)

NAMES = {"sekil_bolme_dogruluk": "fig1_literature", "sekil2_bolme_duzeyi": "fig2_splitting_unit",
         "sekil3_hasta_basina": "fig3_per_participant", "sekil4_model_siralamasi": "fig4_model_ranking",
         "sekil_marj": "figS1a_match_margin", "sekil_ornek": "figS1b_match_example"}

RELAYOUT = {"sekil_bolme_dogruluk", "sekil_marj", "sekil_ornek"}
NO_TITLE = {"sekil_bolme_dogruluk", "sekil2_bolme_duzeyi", "sekil3_hasta_basina", "sekil4_model_siralamasi"}
TITLE_STARTS = ("EMGLab N2001 literature:", "n: number of coded studies", "Same data and model:", "EMGLab N2001; 5-fold",
                "Participant-disjoint evaluation (LOSO):", "Each dot is one participant", "Leaky evaluation changes",
                "Classical window features; 5-fold")

# Sira onemli: uzun ifadeler once
TR_EN = [
    # Sekil 1
    ("EMGLab N2001 literatürü: bölme düzeyine göre raporlanan doğruluk", "EMGLab N2001 literature: reported accuracy by splitting unit"),
    ("n: kodlanan çalışma sayısı; noktalar sayısal doğruluk raporlayanlar (en yüksek değer). Gri çizgi: medyan.",
     "n: number of coded studies; dots: studies reporting a numerical accuracy (highest value). Grey line: median."),
    ("Segment / pencere\n(rastgele)", "S: segment-level\n(random)"),
    ("Kayıt düzeyi", "R: recording-level"),
    ("Hasta-ayrık iddiası\n(çelişkili ya da doğrulanamıyor)", "P*: participant-disjoint claimed,\ncontradicted or not verifiable"),
    ("Bölme anlaşılamıyor", "U: split unclear"),
    ("Hasta-ayrık  (n=", "P: participant-disjoint  (n="),
    ("Yayımlanmış EMGLab çalışması", "Published EMGLab study"),
    ("Bu çalışma (aynı model, aynı veri)", "This study (same model, same data)"),
    ("Raporlanan doğruluk (%)", "Reported accuracy (%)"),
    # Sekil 2
    ("Aynı veri ve model: bölme düzeyi değiştikçe doğruluk", "Same data and model: accuracy as the splitting unit changes"),
    ("EMGLab N2001; 5-kat, 3 tohum ortalaması ± SS. Etiketler: segment → hasta gruplu doğruluk.",
     "EMGLab N2001; 5-fold, mean ± SD over 3 seeds. Labels: segment-level → participant-grouped accuracy."),
    ("Segment düzeyi\n(rastgele)", "Segment-level\n(random)"),
    ("Kayıt gruplu", "Recording-grouped"),
    ("Hasta gruplu", "Participant-grouped"),
    # Sekil 3
    ("Hasta-ayrık değerlendirme (LOSO): hasta başına doğruluk", "Participant-disjoint evaluation (LOSO): accuracy per held-out participant"),
    ("Her nokta bir hasta; gri çizgi: medyan; kesikli çizgi: %50. %50'nin altındaki hastalar etiketli.",
     "Each dot is one participant; grey line: median; dashed line: 50%. Participants below 50% are labelled."),
    ("Dışarıda bırakılan hastada pencere doğruluğu (%)", "Window accuracy in the held-out participant (%)"),
    # Sekil 4
    ("Sızıntılı değerlendirme model sıralamasını değiştiriyor", "Leaky evaluation changes the model ranking"),
    ("Klasik pencere öznitelikleri; 5-kat, 3 tohum ortalaması. Sızıntılı bölmede en iyi görünen 1-NN, hasta-ayrık bölmede en kötü; doğrusal DDA neredeyse hiç şişmiyor.",
     "Classical window features; 5-fold, mean over 3 seeds. 1-NN looks best under segment-level splitting and worst under participant-grouped splitting; linear LDA is barely inflated."),
    ("Segment\n(rastgele)", "Segment\n(random)"),
    ("Hasta\ngruplu", "Participant-\ngrouped"),
    ("Rastgele:", "Random:"),
    ("Hasta-ayrık:", "Participant-disjoint:"),
    ("DDA (doğrusal)", "LDA (linear)"),
    ("Pencere düzeyinde doğruluk (%)", "Window-level accuracy (%)"),
    # Ek sekil S1
    ("Korelasyon: en iyi ve ikinci en iyi aday", "Correlation: best and second-best candidate window"),
    ("Marj (en iyi − ikinci en iyi)", "Margin (best − second best)"),
    ("en iyi eşleşme", "best match"),
    ("ikinci en iyi aday", "second-best candidate"),
    ("güvenli eşik 0,5", "confidence threshold 0.5"),
    ("Segment sayısı", "Number of segments"),
    ("Kaggle test segmenti ile ham kayıt birebir örtüşüyor", "Released test segment matches the raw EMGLab recording"),
    ("; aynı kaydın diğer saniyeleri", ";\nother seconds of the same recording"),
    ("EĞİTİM kümesinde", "are in the TRAINING set"),
    ("EMGLab ham kayıt", "EMGLab raw recording"),
    (", pencere ", ", window "),
    ("Kaggle TEST segmenti", "Released TEST segment"),
    ("Zaman (ms)", "Time (ms)"),
    ("z-skoru", "z-score"),
    # Kurgu ve sinif adlari
    ("Klasik öznitelik + DVM", "Window features + SVM"),
    ("Klasik + DVM", "Features + SVM"),
    ("Klasik+DVM", "Features+SVM"),
    ("ALSNet kurgusu", "ALSNet-like setup"),
    ("3 sınıf", "3 classes"),
    ("Sağlıklı", "Control"),
    ("Miyopati", "Myopathy"),
    ("Hasta-ayrık", "Participant-disjoint"),
    ("DVM-RTF", "SVM-RBF"),
    ("DVM", "SVM"),
    ("DDA", "LDA"),
]
PCT = re.compile(r"%(\d+(?:\.\d+)?)")  # Turkce "%95.5" -> "95.5%"


def tr2en(s: str) -> str:
    for a, b in TR_EN:
        s = s.replace(a, b)
    return PCT.sub(r"\1%", s)


_orig = Figure.savefig


def savefig_en(self, fname, *a, **k):
    p = Path(str(fname))
    if p.stem not in NAMES:
        return _orig(self, fname, *a, **k)
    # Eksen etiketleri cizimde formatter'dan yeniden uretilir; sabit konumlu eksenlerde etiketleri yeniden ver
    self.canvas.draw()
    for ax in self.axes:
        for axis in (ax.xaxis, ax.yaxis):
            if not isinstance(axis.get_major_locator(), matplotlib.ticker.FixedLocator):
                continue
            labs = axis.get_ticklabels()
            if labs and any(tr2en(t.get_text()) != t.get_text() for t in labs):
                axis.set_ticklabels([tr2en(t.get_text()) for t in labs],
                                    fontsize=labs[0].get_fontsize(), color=labs[0].get_color())
    for t in self.findobj(Text):
        s = t.get_text()
        if s:
            s = tr2en(s)
            # Dergi kurali: baslik ve aciklama sekil uzerinde degil, sekil yazisinda olur (ana metin sekilleri)
            t.set_text("" if p.stem in NO_TITLE and s.startswith(TITLE_STARTS) else s)
    if p.stem in RELAYOUT:  # duz tight_layout kullanan sekiller: Ingilizce metne gore yeniden yerlestir
        self.tight_layout()
    self.canvas.draw()
    left = [t.get_text() for t in self.findobj(Text) if re.search("[ğüşıöçĞÜŞİÖÇ]", t.get_text())]
    if left:
        print("CEVRILMEMIS:", p.stem, left)
    target = OUT / f"{NAMES[p.stem]}{p.suffix}"
    res = _orig(self, target, *a, **k)
    if p.stem in NO_TITLE and p.suffix == ".png":  # baslik kalkinca kalan bos kenari kirp
        from PIL import Image, ImageChops
        im = Image.open(target).convert("RGB")
        box = ImageChops.difference(im, Image.new("RGB", im.size, im.getpixel((2, 2)))).getbbox()
        if box:
            pad = 30
            im.crop((max(0, box[0] - pad), max(0, box[1] - pad), min(im.width, box[2] + pad), min(im.height, box[3] + pad))) \
              .save(target, dpi=(300, 300))
    return res


Figure.savefig = savefig_en

if __name__ == "__main__":
    import pandas as pd

    import fig_final as ff
    import fig_literatur as fl

    fl.main()
    df = ff.load_all()
    d4 = pd.read_csv(ff.RES / "deney4_alsnet.csv")
    t = ff.table(df, d4)
    ff.fig2(t)
    ff.fig3(df)
    ff.fig4()
    if len(sys.argv) > 1:
        runpy.run_path(str(HERE / "kaggle_kanit.py"), run_name="__main__")
    print("Ingilizce sekiller:", sorted(x.name for x in OUT.iterdir()))
