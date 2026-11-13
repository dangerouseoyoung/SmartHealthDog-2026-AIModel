from pathlib import Path
import re

import pandas as pd


VIDEO_PATTERN = re.compile(
    r"^(?P<pet_id>[^_]+)_"
    r"(?P<date>\d{8})_"
    r"(?P<sequence>\d+)_"
    r"(?P<quality>[A-Z])_"
    r"(?P<video_id>\d+)$"
)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

REQUIRED_COLUMNS = {
    "video_id",
    "counted_b",
    "observation",
    "reference_b",
    "notes",
}

def load_annotations(annotation_path: Path) -> pd.DataFrame:
    annotation_path = Path(annotation_path)

    if not annotation_path.exists():
        raise FileNotFoundError(
            f"Annotation file not found: {annotation_path}"
        )

    # CSV / XLSX 자동 구분
    suffix = annotation_path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(
            annotation_path,
            dtype={"video_id": str}
        )
    elif suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(
            annotation_path,
            dtype={"video_id": str},
            engine="openpyxl"
        )
    else:
        raise ValueError(f"Unsupported annotation format: {suffix}")

    required_columns = [
        "video_id",
        "counted_breaths",
        "observation_sec",
        "reference_bpm",
        "notes",
    ]

    missing = set(required_columns) - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["video_id"] = df["video_id"].str.strip().str.zfill(3)

    if df["video_id"].duplicated().any():
        duplicates = df.loc[
            df["video_id"].duplicated(),
            "video_id"
        ].tolist()
        raise ValueError(f"Duplicate video_id: {duplicates}")

    numeric_columns = [
        "counted_breaths",
        "observation_sec",
        "reference_bpm",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="raise")

    if (df["counted_breaths"] < 0).any():
        raise ValueError("counted_breaths cannot be negative")

    if (df["observation_sec"] <= 0).any():
        raise ValueError("observation_sec must be positive")

    if (df["reference_bpm"] < 0).any():
        raise ValueError("reference_bpm cannot be negative")

    # 계산값은 별도 컬럼으로 저장하고 reference_b는 유지
    df["calculated_bpm"] = (
        df["counted_breaths"] / df["observation_sec"] * 60
    )

    df["is_measurable"] = (
        (df["counted_breaths"] > 0) &
        (df["reference_bpm"] > 0)
    )

    return df


def parse_video_filename(video_path):
    path = Path(video_path)

    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        return None

    match = VIDEO_PATTERN.match(path.stem)

    if match is None:
        return None

    metadata = match.groupdict()
    metadata["video_id"] = metadata["video_id"].zfill(3)

    return metadata


def build_video_index(video_root):
    video_index = {}

    for path in Path(video_root).rglob("*"):
        if not path.is_file():
            continue

        metadata = parse_video_filename(path)

        if metadata is None:
            continue

        video_id = metadata["video_id"]

        if video_id in video_index:
            raise ValueError(
                f"Duplicate video_id {video_id}: "
                f"{video_index[video_id]['path']} and {path}"
            )

        metadata["path"] = str(path)
        video_index[video_id] = metadata

    return video_index


def build_dataset(csv_path, video_root):
    annotations = load_annotations(csv_path)
    video_index = build_video_index(video_root)

    annotations["video_path"] = annotations["video_id"].map(
        lambda vid: video_index.get(vid, {}).get("path")
    )

    annotations["pet_id"] = annotations["video_id"].map(
        lambda vid: video_index.get(vid, {}).get("pet_id")
    )

    annotations["quality"] = annotations["video_id"].map(
        lambda vid: video_index.get(vid, {}).get("quality")
    )

    annotations["recording_date"] = annotations["video_id"].map(
        lambda vid: video_index.get(vid, {}).get("date")
    )

    annotations["sequence"] = annotations["video_id"].map(
        lambda vid: video_index.get(vid, {}).get("sequence")
    )

    return annotations, video_index
