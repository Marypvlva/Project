import re
from pathlib import Path

import pandas as pd

EXCEL_PATH = Path("resistance.xlsx")
OUTPUT_PATH = Path("metadata.csv")

POWER_RE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*mW\s*$", re.IGNORECASE)
VIDEO_ID_RE = re.compile(r"(S\d{3})")
CELL_POS_RE = re.compile(r"R(\d+)_C(\d+)")
NO_DATA_MARKERS = {"", "-", "no data"}


def find_power(df):
    for i in range(len(df)):
        for j in range(len(df.columns)):
            label = df.iat[i, j]
            if not isinstance(label, str) or "Optical power" not in label:
                continue
            if j + 1 >= len(df.columns):
                continue
            val = df.iat[i, j + 1]
            if isinstance(val, str):
                m = POWER_RE.match(val)
                if m:
                    return float(m.group(1))
    return None


def find_main_table(df):
    for i in range(len(df)):
        for j in range(len(df.columns)):
            v = df.iat[i, j]
            if isinstance(v, str) and v.strip().startswith("Speed, mm*s^(-1)"):
                speeds = []
                k = j + 1
                while k < len(df.columns):
                    val = df.iat[i, k]
                    if pd.isna(val) or str(val).strip() == "":
                        break
                    try:
                        speeds.append(float(val))
                    except (TypeError, ValueError):
                        break
                    k += 1
                if speeds:
                    return i, j, speeds
    return None, None, None


def parse_schema(df):
    schema = {}
    for i in range(len(df)):
        for j in range(len(df.columns)):
            v = df.iat[i, j]
            if not isinstance(v, str) or not v.strip().startswith("Speed, mm*s^(-1)"):
                continue
            speeds = []
            k = j + 1
            while k < len(df.columns):
                val = df.iat[i, k]
                if pd.isna(val) or str(val).strip() == "":
                    break
                try:
                    speeds.append(float(val))
                except (TypeError, ValueError):
                    break
                k += 1
            dcol = j - 1
            for r in range(i + 1, len(df)):
                dist = df.iat[r, dcol]
                if pd.isna(dist):
                    continue
                try:
                    dist = float(dist)
                except (TypeError, ValueError):
                    continue
                for s_idx, speed in enumerate(speeds):
                    c = j + 1 + s_idx
                    if c >= len(df.columns):
                        break
                    cell = df.iat[r, c]
                    if not isinstance(cell, str) or not cell.strip():
                        continue
                    m = VIDEO_ID_RE.search(cell)
                    if not m:
                        continue
                    pos = CELL_POS_RE.search(cell)
                    row, col = pos.groups() if pos else (None, None)
                    schema[(speed, dist)] = (m.group(1), row, col)
    return schema


def parse_sheet(name, df, schema, next_seq):
    power = find_power(df)
    srow, scol, speeds = find_main_table(df)
    if srow is None or power is None:
        print(f"Пропускаем {name}: не найдены параметры")
        return [], next_seq

    dcol = scol - 1
    records = []
    last_dist = None
    replicate = 0

    for i in range(srow + 1, len(df)):
        dist = df.iat[i, dcol]
        if pd.notna(dist):
            try:
                last_dist = float(dist)
                replicate = 0
            except (TypeError, ValueError):
                pass
        if last_dist is None:
            continue

        row_vals = []
        for k in range(len(speeds)):
            c = scol + 1 + k
            row_vals.append(df.iat[i, c] if c < len(df.columns) else None)

        has_data = False
        for v in row_vals:
            if v is None or pd.isna(v):
                continue
            if isinstance(v, str) and v.strip() in NO_DATA_MARKERS:
                continue
            has_data = True
            break
        if not has_data:
            continue

        replicate += 1
        for speed, v in zip(speeds, row_vals):
            if v is None or pd.isna(v):
                continue
            if isinstance(v, str) and v.strip() in NO_DATA_MARKERS:
                continue
            try:
                resist = float(v)
            except (TypeError, ValueError):
                continue

            cell = schema.get((speed, last_dist))
            if cell is None:
                cell = (f"S{next_seq:03d}", None, None)
                next_seq += 1

            video_id, video_row, video_col = cell
            records.append({
                "sample": name,
                "video_id": video_id,
                "video_row": video_row,
                "video_col": video_col,
                "power_mW": power,
                "speed_mm_s": speed,
                "distance_um": last_dist,
                "replicate": replicate,
                "resistance_kOhm_sq": resist,
            })

    return records, next_seq


def main():
    xl = pd.ExcelFile(EXCEL_PATH)
    schema = parse_schema(pd.read_excel(xl, sheet_name=xl.sheet_names[0], header=None))
    records = []
    next_seq = 31
    for sheet_name in xl.sheet_names:
        df = pd.read_excel(xl, sheet_name=sheet_name, header=None)
        recs, next_seq = parse_sheet(sheet_name, df, schema, next_seq)
        records.extend(recs)

    df_out = pd.DataFrame(records)
    df_out.to_csv(OUTPUT_PATH, index=False)
    print(f"Сохранено {len(df_out)} записей в {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
