"""EMGLab N2001 veri seti okuyucu.

Kayit yapisi: N2001<grup><hasta no><kas><kayit no>, ornegin N2001A01AP51
  grup: A=ALS, C=Kontrol (saglikli), M=Miyopati
Her kayit bir .hea (WFDB benzeri baslik) ve bir .bin (int16, little-endian) dosyasindan olusur.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "N2001"
GROUPS = {"A": "ALS", "C": "Normal", "M": "Myopathy"}
NAME_RE = re.compile(r"^N2001([ACM])(\d{2})([A-Z]{2})(\d+)$")


@dataclass
class Record:
    name: str
    group: str          # ALS / Normal / Myopathy
    subject: str        # ornegin A01
    muscle: str         # ornegin BB
    fs: float
    gain: float         # adu / uV
    meta: dict = field(default_factory=dict)
    path: Path | None = None

    def signal(self) -> np.ndarray:
        """Sinyali uV cinsinden dondurur."""
        hea = self.path.with_suffix(".hea").read_text(errors="ignore").splitlines()
        nsamp = int(hea[0].split()[3])
        fmt = hea[1].split()[1]                    # "16+20" -> format 16, 20 bayt ofset
        offset = int(fmt.split("+")[1]) if "+" in fmt else 0
        raw = np.fromfile(self.path.with_suffix(".bin"), dtype="<i2", offset=offset, count=nsamp)
        return raw.astype(np.float64) / self.gain


def parse_header(hea_path: Path) -> Record:
    lines = hea_path.read_text(errors="ignore").splitlines()
    name, _, fs, _ = lines[0].split()[:4]
    gain = float(lines[1].split()[2].split("/")[0])
    meta = {}
    for ln in lines[2:]:
        if ln.startswith("#") and ":" in ln:
            k, v = ln[1:].split(":", 1)
            meta[k.strip()] = v.strip()
    g, sid, mus, _ = NAME_RE.match(name).groups()
    return Record(name=name, group=GROUPS[g], subject=f"{g}{sid}", muscle=mus,
                  fs=float(fs), gain=gain, meta=meta, path=hea_path)


def list_records(data_dir: Path = DATA_DIR, min_sec: float = 5.0) -> list[Record]:
    """Kaynakta bos/kesik olan .bin dosyalari (orn. N2001A05BB55-63) min_sec'ten kisaysa atlanir."""
    recs = [parse_header(p) for p in sorted(data_dir.rglob("*.hea"))]
    out = []
    for r in recs:
        b = r.path.with_suffix(".bin")
        if b.exists() and (b.stat().st_size - 20) / 2 >= min_sec * r.fs:
            out.append(r)
    return out
