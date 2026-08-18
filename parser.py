import re
from datetime import datetime
from pathlib import Path

import pandas as pd

EXCEL_PATH = Path("resistance.xlsx")
OUTPUT_PATH = Path("metadata.csv")

POWER_RE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*mW\s*$", re.IGNORECASE)
VIDEO_ID_RE = re.compile(r"(S\d{3})")
CELL_POS_RE = re.compile(r"R(\d+)_C(\d+)")
FILENAME_RE = re.compile(
    r"MiroC110_(S\d{3})_(R\d+)_(C\d+)_(\d+x\d+)x?_(\d+)fps_"
    r"(\d{4})[-_](\d{1,2})[-_](\d{1,2})[-_](\d{1,2})[-_](\d{1,2})[-_](\d{1,2})\.(?:cine|mp4)",
    re.IGNORECASE,
)
INDEX_SUFFIXES = {".cine", ".mp4"}
NO_DATA_MARKERS = {"", "-", "no data"}


def build_cloud_index(cloud_dir, sheet_names):
    cloud_dir = Path(cloud_dir)
    sheet_norm = {}
    for s in sheet_names:
        sheet_norm.setdefault(" ".join(s.lower().split()), s)
    index = {}
    for f in cloud_dir.rglob("*"):
        if not f.is_file() or f.suffix.lower() not in INDEX_SUFFIXES:
            continue
        sample = None
        for p in f.parents:
            sample = sheet_norm.get(" ".join(p.name.lower().split()))
            if sample:
                break
        if sample is None:
            continue
        m = FILENAME_RE.match(f.name)
        if not m:
            continue
        video_id = m.group(1)
        row = m.group(2)
        col = m.group(3)
        resolution = m.group(4)
        fps = m.group(5)
        ts = datetime(
            int(m.group(6)), int(m.group(7)), int(m.group(8)),
            int(m.group(9)), int(m.group(10)), int(m.group(11)),
        )
        entry = {
            "filename": f.name,
            "video_row": row,
            "video_col": col,
            "resolution": resolution,
            "fps": fps,
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
        }
        existing = index.get(sample, {}).get(video_id)
        if existing is None or f.suffix.lower() == ".mp4":
            index.setdefault(sample, {})[video_id] = entry
    return index


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


def parse_sheet(name, df, schema, file_queue, cloud_data=None):
    if cloud_data is None:
        cloud_data = {}
    power = find_power(df)
    srow, scol, speeds = find_main_table(df)
    if srow is None or power is None:
        print(f"Пропускаем {name}: не найдены параметры")
        return []

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
                if not file_queue:
                    continue
                vid, file_info = file_queue.pop(0)
                cell = (vid, file_info["video_row"], file_info["video_col"])

            video_id, video_row, video_col = cell

            cloud_info = cloud_data.get(video_id, {})
            if not cloud_info.get("filename"):
                continue
            if cloud_info.get("video_row"):
                video_row = cloud_info["video_row"]
            if cloud_info.get("video_col"):
                video_col = cloud_info["video_col"]

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
                "filename": cloud_info["filename"],
                "resolution": cloud_info.get("resolution", ""),
                "fps": cloud_info.get("fps", ""),
                "timestamp": cloud_info.get("timestamp", ""),
            })

    return records


def main():
    xl = pd.ExcelFile(EXCEL_PATH)
    schema = parse_schema(pd.read_excel(xl, sheet_name=xl.sheet_names[0], header=None))

    cloud_index = build_cloud_index(Path("."), xl.sheet_names)
    total = sum(len(v) for v in cloud_index.values())
    print(f"Сканируем {Path('.').resolve()}")
    print(f"Найдено {total} видеофайлов для {len(cloud_index)} листов")
    for s, v in cloud_index.items():
        print(f"  {s}: {len(v)} файлов")
    missing = [s for s in xl.sheet_names if s not in cloud_index]
    if missing:
        print("Нет подходящих видеофайлов для листов:", ", ".join(missing))

    records = []
    for sheet_name in xl.sheet_names:
        df = pd.read_excel(xl, sheet_name=sheet_name, header=None)
        cloud_data = cloud_index.get(sheet_name, {})
        file_queue = [(k, cloud_data[k]) for k in sorted(cloud_data, key=lambda s: int(s[1:]))]
        recs = parse_sheet(sheet_name, df, schema, file_queue, cloud_data)
        records.extend(recs)

    df_out = pd.DataFrame(records, columns=[
        "sample", "video_id", "video_row", "video_col", "power_mW",
        "speed_mm_s", "distance_um", "replicate", "resistance_kOhm_sq",
        "filename", "resolution", "fps", "timestamp",
    ])
    df_out.to_csv(OUTPUT_PATH, index=False)
    print(f"Сохранено {len(df_out)} записей в {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
