# Görev: 25 çalışmanın bağımsız (kör) kodlanması

EMG ile ALS ve miyopati sınıflandırması yapan 25 çalışmayı, **eğitim ve test verisini hangi düzeyde ayırdıklarına** göre kodlamanı istiyoruz. Bu bir denetim değil: kimsenin kararını görmüyorsun, kodları kendin veriyorsun. Aynı çalışmaları başka kodlayıcılar da bağımsız olarak kodluyor; sonra uyum hesaplanacak.

## Dosyalar
| Dosya | İçerik |
|---|---|
| `GOREV_MODEL.md` | Bu talimat |
| `YONERGE.md` | Kod tanımları (S / R / P / P\* / U) ve karar kuralları. **Önce bunu oku.** |
| `EMGLAB_VERI_OZETI.md` | Veri setinin kişi ve kayıt sayıları. **Kararlar bunlara dayanıyor.** |
| `KODLAMA_FORMU.csv` | `;` ayraçlı, 25 satır. İlk sütunlar dolu (no, dosya, yıl, yazar, başlık), diğerleri boş. |
| `pdf/` | Makalelerin tam metinleri (`dosya` sütunundaki adlarla; bazıları yalnızca `.txt` olarak var) |
| `txt/` | Aynı makalelerin düz metin sürümleri |

## Ne yapacaksın
1. `YONERGE.md` ve `EMGLAB_VERI_OZETI.md` dosyalarını oku.
2. Formdaki **her satır** için makalenin tam metnini oku. Özellikle veri seti, ön işleme, deney ve sonuç bölümlerine bak: sınıflandırılan birim ne, eğitim ve test nasıl ayrılmış, kümelerdeki örnek sayıları ne?
3. Boş sütunları yönergeye göre doldur. `kanit_alinti` sütununa kararını destekleyen **birebir** alıntıyı koy; alıntı yoksa kodu U yap.
4. Bir satırda kararsızsan `emin_misiniz` = 1 yaz ve nedenini `not` sütununa kısaca yaz. Tahmin yapma.

## Çıktı
- Formun doldurulmuş hâlini bu klasöre **verilen dosya adıyla** kaydet. Aynı sütunlar, aynı sıra, `;` ayraç, UTF-8.
- Başka hiçbir dosyayı oluşturma ya da değiştirme. İnternete bakma.
