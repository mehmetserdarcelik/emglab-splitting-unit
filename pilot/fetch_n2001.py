"""N2001 verisini EMGLab'dan tamamlar.

1) data/N2001_zip altindaki saglam zip'leri data/N2001'e acar.
2) EMGLab dizin listesindeki kayitlardan eksik olanlari tek tek (.hea + .bin) indirir.
"""
from __future__ import annotations

import re
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "data"
BASE = "https://www.emglab.net/emglab/Signals/N2001/files/N2001/"


def get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def main():
    dst = ROOT / "N2001"
    dst.mkdir(parents=True, exist_ok=True)
    bad = []
    for z in sorted((ROOT / "N2001_zip").glob("*.zip")):
        try:
            zipfile.ZipFile(z).extractall(dst)
        except zipfile.BadZipFile:
            bad.append(z.name)
    print("bozuk zip:", bad)

    listing = get(BASE).decode("utf-8", "ignore")
    names = sorted(set(re.findall(r"/N2001/(N2001[ACM]\d{2}[A-Z]{2}\d+)/", listing)))
    missing = [n for n in names
               if not ((dst / n / f"{n}.bin").exists() and (dst / n / f"{n}.hea").exists())]
    print(f"listede {len(names)} kayit, eksik {len(missing)}")
    for n in missing:
        (dst / n).mkdir(exist_ok=True)
        for ext in (".hea", ".bin"):
            (dst / n / f"{n}{ext}").write_bytes(get(f"{BASE}{n}/{n}{ext}"))
        print("indirildi", n, flush=True)

    # boyut kontrolu: .bin, .hea'daki ornek sayisini karsilamali
    short = []
    for n in names:
        hea = (dst / n / f"{n}.hea").read_text(errors="ignore").splitlines()
        need = int(hea[0].split()[3]) * 2 + 20
        if (dst / n / f"{n}.bin").stat().st_size < need:
            short.append(n)
    print("eksik boyutlu .bin:", short)


if __name__ == "__main__":
    main()
