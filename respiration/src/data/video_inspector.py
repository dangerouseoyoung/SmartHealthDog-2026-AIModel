from pathlib import Path

import cv2
import pandas as pd

from src.data.annotation_loader import (
    parse_video_filename,
    load_annotations,
)


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def inspect_video(video_path):
    path = Path(video_path)
    metadata = parse_video_filename(path)

    if metadata is None:
        return None

    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        return {
            **metadata,
            "path": str(path),
            "readable": False,
            "fps": None,
            "frame_count": None,
            "width": None,
            "height": None,
            "duration_sec": None,
            "file_size_mb": path.stat().st_size / (1024 ** 2),
        }

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    duration = frame_count / fps if fps > 0 else None

    cap.release()

    return {
        **metadata,
        "path": str(path),
        "readable": True,
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_sec": duration,
        "file_size_mb": path.stat().st_size / (1024 ** 2),
    }


def inspect_dataset(video_root, annotation_csv, output_dir):
    video_root = Path(video_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_paths = sorted(
        p for p in video_root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in VIDEO_EXTENSIONS
    )

    results = []

    for path in video_paths:
        result = inspect_video(path)

        if result is not None:
            results.append(result)

    video_df = pd.DataFrame(results)

    if video_df.empty:
        raise RuntimeError("No valid video files found.")

    annotations = load_annotations(annotation_csv)

    # Join annotation labels to video metadata.
    merged = video_df.merge(
        annotations,
        on="video_id",
        how="outer",
        suffixes=("_video", "_annotation"),
        indicator=True,
    )

    video_df.to_csv(
        output_dir / "video_audit.csv",
        index=False,
    )

    merged.to_csv(
        output_dir / "dataset_inventory.csv",
        index=False,
    )
    
    # 매칭되지 않은 어노테이션 확인
    unmatched_annotations = merged[
        merged["_merge"] == "right_only"
    ]

    print("\n===== Unmatched Annotations =====")

    if not unmatched_annotations.empty:
        print(
            unmatched_annotations[
                ["video_id", "counted_breaths", "observation_sec", "reference_bpm", "notes"]
            ].to_string(index=False)
        )
    else:
        print("No unmatched annotations.")

    # Per-animal summary.
    animal_summary = (
        video_df.groupby("pet_id")
        .agg(
            video_count=("video_id", "count"),
            avg_fps=("fps", "mean"),
            avg_duration_sec=("duration_sec", "mean"),
            avg_file_size_mb=("file_size_mb", "mean"),
            quality_types=("quality", lambda x: ",".join(sorted(set(x)))),
        )
        .reset_index()
    )

    animal_summary.to_csv(
        output_dir / "animal_summary.csv",
        index=False,
    )

    print("\n===== Video Dataset Summary =====")
    print(f"Total videos: {len(video_df)}")
    print(f"Animal count: {video_df['pet_id'].nunique()}")
    print(f"Readable videos: {video_df['readable'].sum()}")

    print("\n===== Per-animal Summary =====")
    print(animal_summary.to_string(index=False))

    print("\n===== Annotation Matching =====")
    print(merged["_merge"].value_counts().to_string())

    print(f"\nReports saved to: {output_dir}")


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[2]

    inspect_dataset(
        video_root=ROOT / "data/raw",
        annotation_csv=ROOT / "data/annotations/manual_annotations.xlsx",
        output_dir=ROOT / "outputs",
    )