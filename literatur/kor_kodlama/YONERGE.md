# Kodlama Yönergesi: Eğitim/Test Bölme Birimi

Bu formda 25 çalışma var. Her çalışmayı **tam metninden** okuyup, verilerini eğitim ve test kümelerine **hangi düzeyde** ayırdığını kodlayacaksınız. Önce `EMGLAB_VERI_OZETI.md` dosyasını okuyun; birçok karar oradaki kişi ve kayıt sayılarına dayanıyor.

Başka birinin bu çalışmalar için verdiği kodlara bakmayın; formu tek başınıza doldurun.

## Neden önemli?
EMGLab'da bir kişinin onlarca kaydı var ve kayıtlar çoğu zaman kısa parçalara (segment, pencere, çerçeve, MUAP) bölünüyor. Aynı kişinin ya da aynı kaydın parçaları hem eğitimde hem testte bulunursa model, hastalığı değil o kişiyi ya da kaydı tanıyabilir. Yeni bir hastaya genellemeyi ancak **kişiye (hastaya) göre ayrılmış** bir test gösterebilir.

## Doldurulacak sütunlar
| Sütun | Ne yazılacak |
|---|---|
| `veri` | Kullanılan veri: "EMGLab N2001", "EMGLab (alt küme: ...)", "kendi verisi", "belirsiz" |
| `gorev` | Örn. ALS–sağlıklı, 3 sınıf, miyopati–sağlıklı |
| `ornek_birimi` | Sınıflandırılan birim: segment / pencere / çerçeve / MUAP / kayıt / kas / kişi |
| `bolme_tarifi` | Makale eğitim/test ayrımını nasıl anlatıyor (kısa özet) |
| `bolme_kodu` | **S / R / P / P\* / U** (aşağıdaki kurallara göre, tek kod) |
| `pstar_turu` | Yalnızca P\* verdiyseniz: **celiski** ya da **dogrulanamaz** |
| `kanit_alinti` | Kararınızı destekleyen, makaleden **birebir** alıntı (İngilizce kalabilir) |
| `sayfa_bolum` | Alıntının yeri (sayfa ya da bölüm) |
| `dogruluk` | Makalenin raporladığı en yüksek **test** doğruluğu (%) |
| `emin_misiniz` | 1 (emin değilim) – 3 (eminim) |
| `not` | Kararsız kaldığınız nokta, varsa |

## Bölme kodları
- **S (segment düzeyi, rastgele):** Kayıtlar parçalara bölünmüş (segment, pencere, çerçeve, MUAP, alt sinyal) ve bu parçalar eğitim/test kümelerine **rastgele** dağıtılmış. Aynı kaydın parçaları hem eğitimde hem testte olabilir.
- **R (kayıt düzeyi):** Sınıflandırılan birim kaydın tamamı ya da bir kaydın bütün parçaları aynı kümede; ama **aynı kişinin farklı kayıtları** eğitimde ve testte olabilir.
- **P (kişiye göre ayrılmış):** Bir kişinin bütün verisi yalnızca eğitimde ya da yalnızca testte. Makale bunu açıkça söylüyor **ve** verdiği sayılar bununla tutarlı.
- **P\* (kişiye göre ayrıldığı iddia ediliyor, ama desteklenmiyor):** Makale kişiye göre ayırdığını söylüyor, fakat:
  - **celiski:** verdiği sayılar ya da birimler bununla çelişiyor. Örneğin "birey" diye aslında kayıtları sayıyor ya da test kümesinin boyutu tam kayıtlardan veya kişilerden oluşamaz;
  - **dogrulanamaz:** iddiayı doğrulayacak bilgi yok (hangi kişinin hangi kümede olduğu, katların nasıl kurulduğu anlatılmıyor) ya da anlatılan bölme kişiye göre ayrılmış olduğunu göstermiyor.
- **U (belirlenemiyor):** Sınıflandırma birimi ya da bölme yöntemi makaleden çıkarılamıyor.

## Karar ipuçları
- Yalnızca "10-fold cross-validation" yazıyor ve birim segment ise → **S**.
- "N sinyal / kayıt" sınıflandırılıyor ve çapraz doğrulama var, kişi ayrımından söz yok → **R**.
- "Subject / individual / birey" sayısı EMGLab'daki kişi sayısını aşıyorsa, makale kayıtları birey saymıştır. Kişiye göre ayırdığını iddia ediyorsa → **P\*** (celiski); iddia etmiyor, bunu açıkça kabul ediyorsa → **R**.
- Kişiye göre bölmede test kümesi bütün kayıtlardan oluşur. Her kayıttan k parça alınıyorsa test parça sayısı k'nın katı olmalıdır. Değilse ve kişiye göre ayırdığını iddia ediyorsa → **P\*** (celiski).
- Birim ya da yöntem hiç belli değilse → **U**. Tahmin yapmayın.
- Bir çalışmada birden fazla deney varsa, **en yüksek doğruluğun raporlandığı** EMGLab deneyini kodlayın ve notta belirtin.
