# Segment, recording or participant? Data-splitting unit and reported accuracy in needle-EMG classification (EMGLab N2001)

Code, coded literature table and results for the manuscript

> Çelik MS, Eminoğlu İ. *Segment, recording or participant? The data-splitting unit and reported accuracy in needle-EMG classification of ALS and myopathy on the EMGLab N2001 dataset.* (submitted)

Department of Electrical and Electronics Engineering, Ondokuz Mayıs University, Samsun, Türkiye.
Contact: Mehmet Serdar Çelik (mehmetserdar.celik@omu.edu.tr)

## What is here

| Folder | Content |
|---|---|
| `pilot/` | Data loading, the four model families, all experiments, the audit of the released ALSNet split and the figure scripts |
| `pilot/results/` | Features, per-window predictions of every experiment (`final_tahminler_*.parquet`, `rev_*`), summary tables and logs |
| `pilot/results/kaggle_kanit/` | Audit of the released ALSNet split: match of every released segment to the raw EMGLab recordings (`eslesme_tablosu.csv`) and per-recording summary |
| `literatur/` | Coded literature table (`literatur_tablosu.csv`, working version), change log, translation and the script that builds Supplementary Table S1 |
| `literatur/kor_kodlama*` | Blind re-coding of 25 studies: codebook and form (in Turkish), the three language-model coders' answers with verbatim evidence, and the comparison with the authors' consensus |
| `makale/` | Supplementary Table S1 (English), Table 2 and the English figures |

Comments and some file names are in Turkish (e.g. *kayit* = recording, *hasta* = participant, *tohum* = seed, *pencere* = window, *gorev* = task).

## Data (not included)

- **EMGLab N2001**: http://www.emglab.net (Nikolic, 2001). Download the N2001 recordings into `data/N2001_zip/`; `pilot/fetch_n2001.py` unpacks them into `data/N2001/` and fetches any recording missing from a damaged archive (see the script header). Nine recordings (N2001A05BB55–63) are empty at the source and are skipped.
- **ALSNet release** (pre-split training and test segments): Kaggle, `naimulhassan/processed-emglab-dataset-for-als-detection`. Download the zip and pass its path to `pilot/kaggle_kanit.py`.

The full texts of the coded studies are not redistributed; DOIs are listed in Table S1.

## Reproducing the results

```bash
pip install -r requirements.txt
cd pilot
python fetch_n2001.py                 # EMGLab N2001 -> ../data/N2001
python extract_features.py            # record- and window-level features
python exp5_final.py a b c d          # main experiments (Table 2, Table 3, Figs 2-4); part a/b need a GPU for reasonable run time
python exp6_revizyon.py a b c         # biceps-only and nested cross-validation (Section 3.5-3.6)
python -I kaggle_kanit.py <kaggle.zip> # audit of the released ALSNet split (Section 3.3)
python fig_final.py && python -I fig_en.py <kaggle.zip>   # figures
```

The released pickle files are loaded with a restricted unpickler that only permits NumPy and pandas container classes (`kaggle_kontrol.py`).

Splitting codes used throughout: **S** segment-level random, **R** recording-level, **P** participant-disjoint, **P\*** participant-disjoint claimed but contradicted or not verifiable, **U** undetermined (the working table uses the Turkish letters P/K/H/H\*/? for S/R/P/P\*/U).

## Licence

Code: MIT (see `LICENSE`). Tables, coded literature data and figures: CC BY 4.0.
