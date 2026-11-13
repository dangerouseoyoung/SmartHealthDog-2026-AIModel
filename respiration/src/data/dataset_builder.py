from pathlib import Path

import pandas as pd

from src.data.metadata_loader import (
    ROOT,
    build_metadata_dataset,
)
from src.data.annotation_loader import load_annotations


# ============================================================
# Configuration
# ============================================================

ANNOTATION_PATH = (
    ROOT / "data" / "annotations" / "manual_annotations.xlsx"
)

OUTPUT_DIR = ROOT / "outputs"

# 재수집 전까지 분석에서 제외
EXCLUDED_VIDEO_IDS = {"020"}


# ============================================================
# Annotation Validation
# ============================================================

def validate_annotations(df: pd.DataFrame) -> pd.DataFrame:
    """어노테이션 데이터의 필수 값과 범위를 검사한다."""

    required_columns = [
        "video_id",
        "counted_breaths",
        "observation_sec",
        "reference_bpm",
        "notes",
    ]

    missing = set(required_columns) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing annotation columns: {sorted(missing)}"
        )

    df = df.copy()

    df["video_id"] = (
        df["video_id"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(3)
    )

    if df["video_id"].duplicated().any():
        duplicates = df.loc[
            df["video_id"].duplicated(keep=False),
            "video_id",
        ].tolist()

        raise ValueError(
            f"Duplicate annotation video_id: {duplicates}"
        )

    numeric_columns = [
        "counted_breaths",
        "observation_sec",
        "reference_bpm",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="raise")

    if df[numeric_columns].isna().any().any():
        raise ValueError("Annotation contains missing numeric values.")

    if (df["counted_breaths"] < 0).any():
        raise ValueError("counted_breaths cannot be negative.")

    if (df["observation_sec"] <= 0).any():
        raise ValueError("observation_sec must be positive.")

    if (df["reference_bpm"] < 0).any():
        raise ValueError("reference_bpm cannot be negative.")

    return df


# ============================================================
# Dataset Construction
# ============================================================

def build_dataset() -> pd.DataFrame:
    """
    메타데이터, 실제 영상 경로, 호흡 어노테이션을 통합한다.

    기준 키:
        video_id

    주요 상태:
        MATCHED: 영상 파일과 메타데이터 연결 성공
        EXCLUDED: 분석 제외 영상
        MISSING: 실제 영상 파일 없음
        AMBIGUOUS: 동일 개체 내 중복 영상
    """

    # 1. 메타데이터 + 실제 영상 연결
    metadata_df = build_metadata_dataset(
        excluded_video_ids=EXCLUDED_VIDEO_IDS
    )

    # 2. 어노테이션 로딩
    annotation_df = load_annotations(ANNOTATION_PATH)
    annotation_df = validate_annotations(annotation_df)

    # 3. video_id 기준 결합
    dataset = metadata_df.merge(
        annotation_df,
        on="video_id",
        how="outer",
        suffixes=("", "_annotation"),
        indicator="annotation_merge",
        validate="one_to_one",
    )

    # 4. 어노테이션 매칭 상태
    dataset["annotation_status"] = dataset[
        "annotation_merge"
    ].map({
        "both": "MATCHED",
        "left_only": "MISSING",
        "right_only": "ORPHAN",
    })

    dataset = dataset.drop(columns=["annotation_merge"])

    # 5. 영상 분석 가능 여부
    dataset["is_active"] = (
        dataset["file_status"] == "MATCHED"
    )

    # 6. 호흡 관찰 가능 여부
    dataset["is_measurable"] = (
        dataset["is_active"]
        & dataset["annotation_status"].eq("MATCHED")
        & dataset["counted_breaths"].gt(0)
        & dataset["reference_bpm"].gt(0)
    )

    # 7. 계산 BPM
    dataset["calculated_bpm"] = pd.NA

    measurable = dataset["is_measurable"]

    dataset.loc[measurable, "calculated_bpm"] = (
        dataset.loc[measurable, "counted_breaths"]
        / dataset.loc[measurable, "observation_sec"]
        * 60
    )

    # 8. 기준 BPM과 계산 BPM의 차이
    dataset["bpm_difference"] = pd.NA

    dataset.loc[measurable, "bpm_difference"] = (
        dataset.loc[measurable, "calculated_bpm"]
        - dataset.loc[measurable, "reference_bpm"]
    )

    return dataset


# ============================================================
# Dataset Summary
# ============================================================

def print_dataset_summary(df: pd.DataFrame) -> None:

    print("\n===== Integrated Dataset Summary =====")

    print(f"Total rows: {len(df)}")

    print("\n===== Video Status =====")
    print(df["file_status"].value_counts(dropna=False).to_string())

    print("\n===== Annotation Status =====")
    print(
        df["annotation_status"]
        .value_counts(dropna=False)
        .to_string()
    )

    active = df[df["is_active"]].copy()

    measurable = active[active["is_measurable"]]
    unmeasurable = active[~active["is_measurable"]]

    print("\n===== Active Dataset =====")
    print(f"Active videos: {len(active)}")
    print(f"Animal count: {active['pet_id'].nunique()}")

    print("\n===== Measurement Summary =====")
    print(f"Measurable videos: {len(measurable)}")
    print(f"Unmeasurable videos: {len(unmeasurable)}")

    if len(active) > 0:
        success_rate = len(measurable) / len(active) * 100
        print(f"Measurement availability: {success_rate:.2f}%")

    print("\n===== Per-animal Summary =====")

    if not active.empty:
        summary = active.groupby("pet_id").agg(
            total_videos=("video_id", "count"),
            measurable_videos=("is_measurable", "sum"),
            avg_duration_sec=("duration_sec", "mean"),
            quality_types=(
                "quality_grade",
                lambda x: ",".join(sorted(x.dropna().unique())),
            ),
        )

        summary["unmeasurable_videos"] = (
            summary["total_videos"]
            - summary["measurable_videos"]
        )

        print(summary.round(2).to_string())

    print("\n===== Unmeasurable Videos =====")

    if unmeasurable.empty:
        print("None")
    else:
        print(
            unmeasurable[
                ["video_id", "pet_id", "counted_breaths", "reference_bpm", "notes"]
            ].to_string(index=False)
        )

    print("\n===== BPM Difference =====")

    if not measurable.empty:
        print(
            measurable["bpm_difference"]
            .astype(float)
            .describe()
            .round(3)
            .to_string()
        )


# ============================================================
# Save Dataset
# ============================================================

def save_dataset(df: pd.DataFrame) -> None:

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 전체 데이터: 제외 및 매칭 실패 기록 보존
    df.to_csv(
        OUTPUT_DIR / "full_dataset_inventory.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 실제 영상 파일이 연결된 분석 대상
    active = df[df["is_active"]].copy()

    active.to_csv(
        OUTPUT_DIR / "active_dataset.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 호흡 측정이 가능한 영상
    measurable = active[active["is_measurable"]]

    measurable.to_csv(
        OUTPUT_DIR / "measurable_dataset.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 호흡 관찰 불가능 영상
    unmeasurable = active[~active["is_measurable"]]

    unmeasurable.to_csv(
        OUTPUT_DIR / "unmeasurable_dataset.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n===== Saved Files =====")

    for filename in [
        "full_dataset_inventory.csv",
        "active_dataset.csv",
        "measurable_dataset.csv",
        "unmeasurable_dataset.csv",
    ]:
        print(OUTPUT_DIR / filename)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    dataset = build_dataset()

    print_dataset_summary(dataset)

    save_dataset(dataset)
