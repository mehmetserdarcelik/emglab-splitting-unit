# Literatür tablosu değişiklik günlüğü

## 23 Ekim 2026: iç hakem değerlendirmesi sonrası
Yedek: `literatur_tablosu_v1_hakem_oncesi.csv`

| id | Çalışma | Sütun | Değişiklik | Gerekçe |
|---|---|---|---|---|
| W4225343804 | A. Bakiya 2022 | grup | EMGLab (kesin/olasi) → dislandi (geri cekildi) | 2023te geri cekilmis makale; analizden cikarildi (Sonnet, Argon) |
| W4386907835 | Md. Toufiqur Rahman 2023 | bolme_kodu | K → H* | Tam metin "subject-independent" iddia ediyor, bolme kayit duzeyinde tarif ediliyor; kod kuralina gore H* (Sonnet, Argon) |
| W4386908193 | Minhajur Rahman 2023 | bolme_kodu | K → H* | Tam metin "subject-independent" iddia ediyor, bolme kayit duzeyinde tarif ediliyor; kod kuralina gore H* (Sonnet, Argon) |
| W2897127292 | Abdülhamit Subaşı 2018 | grup | EMGLab (kesin/olasi) → ilgili (klinik/kendi verisi) | Tam metin: Subasi grubunun 27 denekli kendi verisi (20 kHz, 5 s, normal/miyopati/noropati); EMGLab degil (Argon) |
| W2982488225 | Emine Yaman 2019 | grup | EMGLab (kesin/olasi) → ilgili (klinik/kendi verisi) | Normal/miyopati/noropati, 800 ornek/sinif: Subasi 2018 ile ayni kurgu; EMGLab kullanildigina dair kanit yok (Argon) |
| W3082931328 | Kaniska Samanta 2020 | bildirilen_dogruluk | 00 → 100 | Tam metin: "highest mean classification accuracy of 100%"; deger dosyada "00" olarak bozulmustu |
| W4220961616 | Ines Abdennaji 2022 | bildirilen_dogruluk | 00 → 100 (ozetten) | Ozette %100; deger "00" olarak bozulmustu |
| W4313527089 | Xujian Wang 2022 | bildirilen_dogruluk | 11,18 →  | "11,18" anlamsiz bir deger; sayisal dogruluk cikarilamadi |

## 26 Ekim 2026: kör kodlama sonrası (yazarların ortak kararı)
Yedek: `literatur_tablosu_v2_kor_kodlama_oncesi.csv`. Kodlayıcılar: M. S. Çelik, İ. Eminoğlu (bağımsız, sonra ortak karar), Claude Sonnet, Google Gemini (Argon), OpenAI GPT (Astra).

| id | Çalışma | Sütun | Değişiklik | Gerekçe |
|---|---|---|---|---|
| W7207759719 | Sandipa Chowdhury 2026 | bolme_kodu | H → H* | Kor kodlama ortak karari (Celik, Eminoglu; Sonnet, Astra). GitHub not defteri ALSNet Kaggle bolmesini yukluyor (VAL_WINDOWS_TARGET=532; test ALS=310, ALSNet test ALS sayisiyla ayni); dogrulama ayrimi pencere duzeyinde rastgele. Iddia kodla celisiyor. |
| W2159725448 | Shaikh Anowarul Fattah 2012 | bolme_kodu | P → K | Kor kodlama ortak karari; 3 model ayni. Kayit duzeyi (dataset) LOOCV. |
| W4399039701 | Sarra Ouamri 2024 | bolme_kodu | ? → P | Kor kodlama ortak karari; 3 model ayni. %80 ortusmeli pencerelerden uretilen goruntuler rastgele %75/%25. |
| W2510104717 | Vipin K Mishra 2016 | bolme_kodu | ? → K | Kor kodlama ortak karari (Sonnet, Argon ayni). Not: karisiklik matrisinin kosegeni 220/320 = %68,75, makale %88 raporluyor. |
| W4382935278 | Dalila Cherifi 2023 | bolme_kodu | ? → K | Kor kodlama ortak karari (Sonnet, Astra ayni); birim acikca belirtilmemis. |
| W2593000810 | Hanife Küçük 2016 | bolme_kodu | K → ? | Kor kodlama ortak karari; 3 model ayni. Bildiri yalnizca on isleme/bolutleme/kumeleme anlatiyor, siniflandirma yok. |
