| Kurgu | Protokol | Tohum | Pencere doğruluğu | Dengeli | Kayıt (oy) | Hasta (oy) | Hasta %95 GA |
|---|---|---|---|---|---|---|---|
| 1D-CNN: 3 sınıf | pencere_rastgele | 3 | 96.4 ± 0.1 | 96.4 | 98.7 | 25/25 | 0.86–1.00 |
| 1D-CNN: 3 sınıf | kayit_grup | 3 | 89.9 ± 0.5 | 89.9 | 92.1 | 25/25 | 0.86–1.00 |
| 1D-CNN: 3 sınıf | hasta_grup | 3 | 84.6 ± 0.8 | 84.6 | 87.0 | 25/25 | 0.86–1.00 |
| 1D-CNN: 3 sınıf | hasta_LOSO | 1 | 84.6 | 84.6 | 86.8 | 25/25 | 0.86–1.00 |
| ALSNet kurgusu: ALS–Sağlıklı | rastgele_80_20 (ALSNet bolmesi) | 3 | 95.5 ± 0.8 | 95.5 | 96.5 | 17/17 | 0.80–1.00 |
| ALSNet kurgusu: ALS–Sağlıklı | pencere_rastgele | 3 | 95.6 ± 0.3 | 95.6 | 97.8 | 17/17 | 0.80–1.00 |
| ALSNet kurgusu: ALS–Sağlıklı | kayit_grup | 3 | 91.6 ± 0.6 | 91.6 | 94.3 | 17/17 | 0.80–1.00 |
| ALSNet kurgusu: ALS–Sağlıklı | hasta_grup | 3 | 89.7 ± 0.4 | 89.7 | 92.5 | 17/17 | 0.80–1.00 |
| ALSNet kurgusu: ALS–Sağlıklı | hasta_LOSO | 1 | 89.9 | 89.9 | 92.4 | 17/17 | 0.80–1.00 |
| Klasik öznitelik + DVM: 3 sınıf | pencere_rastgele | 3 | 85.5 ± 0.3 | 85.4 | 89.1 | 25/25 | 0.86–1.00 |
| Klasik öznitelik + DVM: 3 sınıf | kayit_grup | 3 | 73.5 ± 0.2 | 73.3 | 75.6 | 24/25 | 0.80–1.00 |
| Klasik öznitelik + DVM: 3 sınıf | hasta_grup | 3 | 69.7 ± 0.4 | 69.5 | 71.8 | 24/25 | 0.80–1.00 |
| Klasik öznitelik + DVM: 3 sınıf | hasta_LOSO | 1 | 68.3 | 68.2 | 71.2 | 24/25 | 0.80–1.00 |
| Klasik öznitelik + DVM: ALS-Miyopati | pencere_rastgele | 3 | 94.1 ± 0.1 | 94.0 | 95.4 | 15/15 | 0.78–1.00 |
| Klasik öznitelik + DVM: ALS-Miyopati | kayit_grup | 3 | 88.6 ± 0.4 | 88.6 | 89.2 | 15/15 | 0.78–1.00 |
| Klasik öznitelik + DVM: ALS-Miyopati | hasta_grup | 3 | 88.0 ± 0.0 | 88.0 | 89.3 | 15/15 | 0.78–1.00 |
| Klasik öznitelik + DVM: ALS-Miyopati | hasta_LOSO | 1 | 86.2 | 86.2 | 87.0 | 15/15 | 0.78–1.00 |
| Klasik öznitelik + DVM: ALS-Sağlıklı | pencere_rastgele | 3 | 91.1 ± 0.1 | 91.2 | 93.8 | 18/18 | 0.81–1.00 |
| Klasik öznitelik + DVM: ALS-Sağlıklı | kayit_grup | 3 | 82.0 ± 1.9 | 82.0 | 84.2 | 18/18 | 0.81–1.00 |
| Klasik öznitelik + DVM: ALS-Sağlıklı | hasta_grup | 3 | 80.6 ± 0.4 | 80.6 | 82.9 | 18/18 | 0.81–1.00 |
| Klasik öznitelik + DVM: ALS-Sağlıklı | hasta_LOSO | 1 | 79.6 | 79.6 | 82.0 | 18/18 | 0.81–1.00 |
| Klasik öznitelik + DVM: Miyopati-Sağlıklı | pencere_rastgele | 3 | 90.1 ± 0.1 | 90.2 | 93.4 | 17/17 | 0.80–1.00 |
| Klasik öznitelik + DVM: Miyopati-Sağlıklı | kayit_grup | 3 | 78.0 ± 0.4 | 77.9 | 80.3 | 17/17 | 0.80–1.00 |
| Klasik öznitelik + DVM: Miyopati-Sağlıklı | hasta_grup | 3 | 74.9 ± 0.7 | 74.9 | 75.8 | 16/17 | 0.71–1.00 |
| Klasik öznitelik + DVM: Miyopati-Sağlıklı | hasta_LOSO | 1 | 74.0 | 74.0 | 75.8 | 17/17 | 0.80–1.00 |

*Notlar:*
- Pencere doğruluğu: 5-kat çapraz doğrulama, 3 tohum ortalaması ± SS; LOSO tek koşu.
- Kayıt (oy) ve Hasta (oy): pencere tahminlerinin çoğunluk oyu. Hasta sütunu ilk tohum için verilmiştir.
- %95 GA: Clopper–Pearson kesin güven aralığı.
- ALSNet kurgusunun LOSO dışı satırları Deney 4'ten (`pilot/results/deney4_alsnet.csv`).
- "Dengeli" = sınıf duyarlılıklarının ortalaması.
