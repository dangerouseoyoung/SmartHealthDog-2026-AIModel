
from pathlib import Path
import re

import pandas as pd


# ============================================================
# Configuration
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT / "data" / "raw"
METADATA_PATH = ROOT / "data" / "metadata" / "video_metadatas.xlsx"

EXCLUDED_VIDEO_IDS = {"020"}

SUPPORTED_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
}

REQUIRED_COLUMNS = [
    "video_id",
    "pet_id",
    "file_path",
    "recorded_at",
    "duration_sec",
    "body_visibility",
    "lighting",
    "noise_condition",
    "quality_grade",
    "split",
]


# ============================================================
# Utility
# ============================================================

def normalize_video_id(value) -> str:
    """video_id를 3자리 문자열로 정규화한다."""

    if pd.isna(value):
        raise ValueError("video_id contains an empty value.")

    value = str(value).strip()

    # Excel에서 1.0으로 읽히는 경우 처리
    if value.endswith(".0"):
        value = value[:-2]

    if not value.isdigit():
        raise ValueError(f"Invalid video_id: {value}")

    return value.zfill(3)


def normalize_pet_id(value) -> str:
    """pet_id를 표준 형식으로 정규화한다."""

    if pd.isna(value):
        raise ValueError("pet_id contains an empty value.")

    return str(value).strip().upper()


def extract_video_id(file_path: Path) -> str | None:
    """
    파일명 마지막 토큰에서 video_id를 추출한다.

    지원 예시:
    CAT001_20261014_001_A_001.mp4
    001.mp4

    마지막 토큰이 3자리 숫자가 아니면 None 반환.
    """

    stem = file_path.stem
    match = re.search(r"(?:^|[_-])(\d+)$", stem)

    if match is None:
        return None

    return match.group(1).zfill(3)


# ============================================================
# Metadata Loading
# ============================================================

def load_metadata(metadata_path: Path = METADATA_PATH) -> pd.DataFrame:
    """XLSX 메타데이터를 읽고 기본 유효성을 검사한다."""

    metadata_path = Path(metadata_path)

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}"
        )

    df = pd.read_excel(
        metadata_path,
        engine="openpyxl",
    )

    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing metadata columns: {sorted(missing_columns)}"
        )

    df = df[REQUIRED_COLUMNS].copy()

    df["video_id"] = df["video_id"].apply(normalize_video_id)
    df["pet_id"] = df["pet_id"].apply(normalize_pet_id)

    # video_id + pet_id 조합 중복 검사
    duplicate_mask = df.duplicated(
        subset=["pet_id", "video_id"],
        keep=False,
    )

    if duplicate_mask.any():
        duplicates = df.loc[
            duplicate_mask,
            ["pet_id", "video_id"],
        ]

        raise ValueError(
            "Duplicate metadata entries:\n"
            f"{duplicates.to_string(index=False)}"
        )

    df["duration_sec"] = pd.to_numeric(
        df["duration_sec"],
        errors="raise",
    )

    if (df["duration_sec"] <= 0).any():
        raise ValueError("duration_sec must be positive.")

    return df


# ============================================================
# Actual Video Index
# ============================================================

def build_video_index(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """
    실제 raw 디렉터리를 탐색해 영상 파일 인덱스를 생성한다.

    엑셀의 file_path는 사용하지 않는다.
    실제 경로는 raw_dir / pet_id를 기준으로 탐색한다.
    """

    raw_dir = Path(raw_dir)

    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw video directory not found: {raw_dir}"
        )

    records = []

    for pet_dir in sorted(raw_dir.iterdir()):

        if not pet_dir.is_dir():
            continue

        pet_id = pet_dir.name.upper()

        for video_path in pet_dir.rglob("*"):

            if not video_path.is_file():
                continue

            if video_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            video_id = extract_video_id(video_path)

            if video_id is None:
                continue

            records.append({
                "pet_id": pet_id,
                "video_id": video_id,
                "resolved_file_path": str(video_path.resolve()),
                "file_name": video_path.name,
            })

    columns = [
        "pet_id",
        "video_id",
        "resolved_file_path",
        "file_name",
    ]

    return pd.DataFrame(records, columns=columns)


# ============================================================
# Metadata + Video Matching
# ============================================================

def build_metadata_dataset(
    metadata_path: Path = METADATA_PATH,
    raw_dir: Path = RAW_DIR,
    excluded_video_ids: set[str] | None = None,
) -> pd.DataFrame:
    """
    메타데이터와 실제 영상 파일을 pet_id + video_id 기준으로 연결한다.

    상태:
        MATCHED
        MISSING
        AMBIGUOUS
        EXCLUDED
    """

    if excluded_video_ids is None:
        excluded_video_ids = EXCLUDED_VIDEO_IDS

    excluded_video_ids = {
        normalize_video_id(v)
        for v in excluded_video_ids
    }

    metadata = load_metadata(metadata_path)
    video_index = build_video_index(raw_dir)

    # 중복 파일 확인
    duplicate_mask = video_index.duplicated(
        subset=["pet_id", "video_id"],
        keep=False,
    )

    duplicate_keys = set()

    if duplicate_mask.any():
        duplicate_keys = set(
            zip(
                video_index.loc[duplicate_mask, "pet_id"],
                video_index.loc[duplicate_mask, "video_id"],
            )
        )

    # 실제 영상 파일 연결
    merged = metadata.merge(
        video_index,
        on=["pet_id", "video_id"],
        how="left",
        indicator=True,
    )

    statuses = []

    for _, row in merged.iterrows():

        key = (row["pet_id"], row["video_id"])

        if row["video_id"] in excluded_video_ids:
            statuses.append("EXCLUDED")

        elif key in duplicate_keys:
            statuses.append("AMBIGUOUS")

        elif row["_merge"] == "both":
            statuses.append("MATCHED")

        else:
            statuses.append("MISSING")

    merged["file_status"] = statuses

    # 제외된 영상 및 매칭되지 않은 파일의 경로는 사용하지 않는다.
    merged.loc[
        merged["file_status"] != "MATCHED",
        "resolved_file_path",
    ] = None

    merged = merged.drop(columns=["_merge"])

    return merged


# ============================================================
# Validation Report
# ============================================================

def print_dataset_summary(df: pd.DataFrame) -> None:
    """데이터셋 매칭 결과를 출력한다."""

    print("\n===== Metadata Dataset Summary =====")

    print(f"Total metadata rows: {len(df)}")

    print("\n===== File Matching Status =====")
    print(df["file_status"].value_counts().to_string())

    active = df[df["file_status"] == "MATCHED"]

    print("\n===== Active Dataset =====")
    print(f"Active videos: {len(active)}")
    print(f"Animal count: {active['pet_id'].nunique()}")

    print("\n===== Per-animal Summary =====")

    if not active.empty:
        print(
            active.groupby("pet_id")
            .agg(
                video_count=("video_id", "count"),
                avg_duration_sec=("duration_sec", "mean"),
                quality_types=(
                    "quality_grade",
                    lambda x: ",".join(sorted(x.dropna().unique())),
                ),
            )
            .round(2)
            .to_string()
        )

    print("\n===== Problematic Videos =====")

    problematic = df[df["file_status"].isin(
        ["MISSING", "AMBIGUOUS"]
    )]

    if problematic.empty:
        print("No problematic videos.")
    else:
        print(
            problematic[
                ["pet_id", "video_id", "file_status"]
            ].to_string(index=False)
        )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    dataset = build_metadata_dataset()

    print_dataset_summary(dataset)

    output_dir = ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "metadata_inventory.csv"

    dataset.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\nSaved: {output_path}")
