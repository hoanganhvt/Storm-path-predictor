#!/usr/bin/env python3
"""
Parser for JMA RSMC Tokyo-Typhoon Center Best Track Data
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

# ----------------------------------------------------------------------
DATA_PATH = Path("../../Datasets/bst_all.txt")          
OUTPUT_PATH = Path("../../Datasets/data.csv")
# ----------------------------------------------------------------------

GRADE_MAP = {
    2: "Tropical Depression (TD)",
    3: "Tropical Storm (TS)",
    4: "Severe Tropical Storm (STS)",
    5: "Typhoon (TY)",
    6: "Extra-tropical Cyclone (L)",
    7: "Entering RSMC area",
    9: "TC of TS intensity or higher",
}

DIRECTION_MAP = {
    0: "No direction",
    1: "NE",
    2: "E",
    3: "SE",
    4: "S",
    5: "SW",
    6: "W",
    7: "NW",
    8: "N",
    9: "Symmetric",
}


def _safe_int(s: str) -> Optional[int]:
    s = s.strip()
    return int(s) if s.isdigit() else None


def _safe_float(s: str) -> Optional[float]:
    s = s.strip()
    try:
        return float(s) if s else None
    except ValueError:
        return None


def parse_bst_text(text: str) -> pd.DataFrame:
    records = []
    current_header = None

    for raw in text.splitlines():
        line = raw.rstrip("\n\r")
        if not line.strip():
            continue

        # ----- Header line -----
        if line.startswith("66666"):
            # Format: 66666 BBBB  CCC DDDD EEEE F G HHHHHHHHHHHHHHHHHHHH              IIIIIIII
            line = line.ljust(72)
            international_id = line[6:10].strip()
            n_lines          = _safe_int(line[12:15])
            tc_number        = line[16:20].strip()
            flag_last        = _safe_int(line[26:27])
            time_diff        = _safe_int(line[28:29])
            name             = line[30:50].strip()
            revision_date    = line[64:72].strip()

            current_header = {
                "international_id": international_id if international_id else None,
                "tc_number": tc_number if tc_number else None,
                "name": name if name else None,
                "n_data_lines": n_lines,
                "flag_last": flag_last,
                "time_diff_hours": time_diff,
                "revision_date": revision_date if revision_date else None,
            }
            continue

        if current_header is None:
            continue

        # ----- Data line (still fixed-width because it is more regular) -----
        # Pad to at least 72 characters so short lines don't crash
        line = line.ljust(72)

        time_str     = line[0:8]
        grade        = line[13:14]
        lat_raw      = line[15:18]
        lon_raw      = line[19:23]
        pressure     = line[24:28]
        wind_max     = line[33:36]
        dir_50kt     = line[41:42]
        rad_50_long  = line[42:46]
        rad_50_short = line[47:51]
        dir_30kt     = line[52:53]
        rad_30_long  = line[53:57]
        rad_30_short = line[58:62]
        landfall     = line[71:72].strip()

        # time
        try:
            yy = int(time_str[0:2])
            year = 1900 + yy if yy >= 50 else 2000 + yy
            dt = datetime(year, int(time_str[2:4]), int(time_str[4:6]), int(time_str[6:8]))
        except (ValueError, IndexError):
            dt = None

        record = {
            **current_header,
            "time": dt,
            "grade": _safe_int(grade),
            "grade_name": GRADE_MAP.get(_safe_int(grade)),
            "lat": (_safe_float(lat_raw) / 10.0) if lat_raw.strip() else None,
            "lon": (_safe_float(lon_raw) / 10.0) if lon_raw.strip() else None,
            "pressure_hpa": _safe_float(pressure),
            "max_wind_kt": _safe_float(wind_max),
            "dir_50kt": _safe_int(dir_50kt),
            "dir_50kt_name": DIRECTION_MAP.get(_safe_int(dir_50kt)),
            "rad_50kt_long_nm": _safe_float(rad_50_long),
            "rad_50kt_short_nm": _safe_float(rad_50_short),
            "dir_30kt": _safe_int(dir_30kt),
            "dir_30kt_name": DIRECTION_MAP.get(_safe_int(dir_30kt)),
            "rad_30kt_long_nm": _safe_float(rad_30_long),
            "rad_30kt_short_nm": _safe_float(rad_30_short),
            "landfall_flag": landfall if landfall else None,
        }
        records.append(record)

    df = pd.DataFrame(records)

    preferred = [
        "international_id", "tc_number", "name", "time",
        "grade", "grade_name", "lat", "lon",
        "pressure_hpa", "max_wind_kt",
        "dir_50kt", "dir_50kt_name", "rad_50kt_long_nm", "rad_50kt_short_nm",
        "dir_30kt", "dir_30kt_name", "rad_30kt_long_nm", "rad_30kt_short_nm",
        "landfall_flag", "flag_last", "time_diff_hours", "revision_date",
    ]
    return df[[c for c in preferred if c in df.columns]]


def load_besttrack(path: Optional[Path] = None) -> pd.DataFrame:
    path = Path(path) if path is not None else DATA_PATH
    text = path.read_text(encoding="ascii", errors="ignore")
    return parse_bst_text(text)


if __name__ == "__main__":
    df = load_besttrack()
    df.to_csv(OUTPUT_PATH,index=False)
    print(df.head(5))
    print(f"\nShape: {df.shape}")
    print(f"Storms: {df['international_id'].nunique()}")