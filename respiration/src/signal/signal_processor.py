
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, detrend, sosfiltfilt


FEATURES = [
    "median_flow_x",
    "median_flow_y",
    "mean_flow_x",
    "mean_flow_y",
    "mean_magnitude",
]


def interpolate_short_gaps(
    signal: np.ndarray,
    max_gap_frames: int = 3,
) -> tuple[np.ndarray, int]:
    """Interpolate only short internal NaN gaps."""
    result = signal.astype(float).copy()
    missing = ~np.isfinite(result)

    filled_count = 0
    i = 0

    while i < len(result):
        if not missing[i]:
            i += 1
            continue

        start = i
        while i < len(result) and missing[i]:
            i += 1
        end = i

        gap_length = end - start

        # Do not extrapolate leading/trailing gaps.
        if (
            gap_length <= max_gap_frames
            and start > 0
            and end < len(result)
            and np.isfinite(result[start - 1])
            and np.isfinite(result[end])
        ):
            left = result[start - 1]
            right = result[end]

            result[start:end] = np.linspace(
                left,
                right,
                gap_length + 2,
            )[1:-1]

            filled_count += gap_length

    return result, filled_count


def bandpass_filter(
    signal: np.ndarray,
    fps: float,
    low_hz: float = 0.1,
    high_hz: float = 1.0,
    order: int = 3,
) -> np.ndarray:
    nyquist = fps / 2.0

    if not 0 < low_hz < high_hz < nyquist:
        raise ValueError(
            f"Invalid filter range: {low_hz}-{high_hz} Hz "
            f"for sampling rate {fps} FPS"
        )

    sos = butter(
        order,
        [low_hz, high_hz],
        btype="bandpass",
        fs=fps,
        output="sos",
    )

    return sosfiltfilt(sos, signal)


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    mean = np.mean(signal)
    std = np.std(signal)

    if std < 1e-8:
        raise ValueError("Signal variance is too small.")

    return (signal - mean) / std


def process_track_feature(
    track_df: pd.DataFrame,
    feature: str,
    fps: float,
    min_duration_sec: float,
    min_coverage: float,
    max_gap_frames: int,
) -> tuple[pd.DataFrame | None, dict]:
    track_id = int(track_df["track_id"].iloc[0])

    # One observation per track per frame.
    grouped = (
        track_df.groupby("frame_idx")[feature]
        .mean()
        .sort_index()
    )

    first_frame = int(grouped.index.min())
    last_frame = int(grouped.index.max())

    frame_index = np.arange(
        first_frame,
        last_frame + 1,
        dtype=int,
    )

    raw = (
        grouped.reindex(frame_index)
        .to_numpy(dtype=float)
    )

    observed_count = int(np.isfinite(raw).sum())
    span = len(raw)
    coverage = observed_count / span if span else 0.0
    duration_sec = span / fps

    base_info = {
        "track_id": track_id,
        "feature": feature,
        "start_frame": first_frame,
        "end_frame": last_frame,
        "duration_sec": duration_sec,
        "observed_frames": observed_count,
        "coverage": coverage,
        "interpolated_frames": 0,
    }

    if duration_sec < min_duration_sec:
        return None, {
            **base_info,
            "status": "INSUFFICIENT_DURATION",
        }

    if coverage < min_coverage:
        return None, {
            **base_info,
            "status": "LOW_COVERAGE",
        }

    interpolated, filled_count = interpolate_short_gaps(
        raw,
        max_gap_frames=max_gap_frames,
    )

    base_info["interpolated_frames"] = filled_count

    if not np.all(np.isfinite(interpolated)):
        return None, {
            **base_info,
            "status": "UNFILLED_GAPS",
        }

    if np.std(interpolated) < 1e-8:
        return None, {
            **base_info,
            "status": "LOW_VARIANCE",
        }

    try:
        detrended = detrend(interpolated, type="linear")
        filtered = bandpass_filter(detrended, fps)
        normalized = normalize_signal(filtered)

    except (ValueError, np.linalg.LinAlgError) as exc:
        return None, {
            **base_info,
            "status": "FILTER_ERROR",
            "error": str(exc),
        }

    # Keep original ROI slot for diagnostics only.
    roi_slots = (
        track_df.groupby("frame_idx")["roi_slot"]
        .first()
        .reindex(frame_index)
        if "roi_slot" in track_df.columns
        else pd.Series(np.nan, index=frame_index)
    )

    output = pd.DataFrame({
        "frame_idx": frame_index,
        "timestamp_sec": frame_index / fps,
        "track_id": track_id,
        "roi_slot": roi_slots.to_numpy(),
        "feature": feature,
        "raw_signal": raw,
        "interpolated_signal": interpolated,
        "detrended_signal": detrended,
        "filtered_signal": filtered,
        "normalized_signal": normalized,
    })

    return output, {
        **base_info,
        "status": "OK",
        "filtered_std": float(np.std(filtered)),
    }


def load_frame_count(
    input_path: Path,
    signals: pd.DataFrame,
) -> int:
    metadata_path = input_path.parent / "optical_flow_metadata.json"

    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

        return int(metadata["frame_count"])

    return int(signals["frame_idx"].max()) + 1


def run_signal_processing(
    input_path: Path,
    output_dir: Path,
    fps: float = 30.0,
    min_duration_sec: float = 10.0,
    min_coverage: float = 0.90,
    max_gap_frames: int = 3,
) -> None:
    signals = pd.read_csv(input_path)

    required_columns = {
        "frame_idx",
        "track_id",
        *FEATURES,
    }

    missing_columns = required_columns - set(signals.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    signals = signals.dropna(subset=["track_id"]).copy()
    signals["frame_idx"] = signals["frame_idx"].astype(int)
    signals["track_id"] = signals["track_id"].astype(int)

    frame_count = load_frame_count(input_path, signals)

    if signals["frame_idx"].min() < 0:
        raise ValueError("Negative frame_idx detected.")

    if signals["frame_idx"].max() >= frame_count:
        raise ValueError(
            "frame_idx exceeds metadata frame_count."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    processed_results = []
    summary_results = []

    for track_id, track_df in signals.groupby("track_id"):
        track_df = track_df.sort_values("frame_idx")

        for feature in FEATURES:
            processed, summary = process_track_feature(
                track_df=track_df,
                feature=feature,
                fps=fps,
                min_duration_sec=min_duration_sec,
                min_coverage=min_coverage,
                max_gap_frames=max_gap_frames,
            )

            summary_results.append(summary)

            if processed is not None:
                processed_results.append(processed)

    if processed_results:
        processed_df = pd.concat(
            processed_results,
            ignore_index=True,
        )
    else:
        processed_df = pd.DataFrame()

    summary_df = pd.DataFrame(summary_results)

    processed_path = output_dir / "processed_signals.csv"
    summary_path = output_dir / "signal_processing_summary.csv"

    processed_df.to_csv(processed_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("\n=== Signal Processing Summary ===")
    print(f"Input: {input_path}")
    print(f"Frame count: {frame_count}")
    print(f"FPS: {fps}")
    print(f"Unique tracks: {signals['track_id'].nunique()}")
    print(f"Processed signals: {len(processed_results)}")
    print(f"Output: {output_dir}")

    if not summary_df.empty:
        print("\nStatus by feature:")
        print(
            summary_df.groupby(
                ["feature", "status"]
            ).size().to_string()
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--min-duration",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.90,
    )
    parser.add_argument(
        "--max-gap-frames",
        type=int,
        default=3,
    )

    args = parser.parse_args()

    run_signal_processing(
        input_path=args.input,
        output_dir=args.output_dir,
        fps=args.fps,
        min_duration_sec=args.min_duration,
        min_coverage=args.min_coverage,
        max_gap_frames=args.max_gap_frames,
    )


if __name__ == "__main__":
    main()
