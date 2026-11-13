
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, detrend, find_peaks, sosfiltfilt


LOGGER = logging.getLogger("respiratory_analyzer")


# ============================================================
# 1. Configuration
# ============================================================

@dataclass
class AnalyzerConfig:
    # Respiratory frequency range
    min_bpm: float = 15.0
    max_bpm: float = 45.0

    # Segment constraints
    min_segment_sec: float = 16.0

    # Track constraints
    min_track_coverage: float = 0.80
    min_tracks: int = 4
    min_valid_tracks: int = 3
    max_interpolation_gap: int = 3

    # Track termination event detection
    track_end_tolerance_frames: int = 3
    min_simultaneous_ends: int = 4
    min_simultaneous_end_ratio: float = 0.50

    # FFT cluster constraints
    fft_cluster_tolerance_bpm: float = 3.0
    min_fft_cluster_fraction: float = 0.60

    # Auxiliary method agreement
    max_auxiliary_disagreement_bpm: float = 5.0

    # Signal filtering
    filter_order: int = 3

    # Analysis features
    primary_feature: str = "mean_flow_y"
    diagnostic_features: tuple[str, ...] = (
        "mean_flow_x",
    )


# ============================================================
# 2. Input loading
# ============================================================

def load_signals(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {
        "frame_idx",
        "timestamp_sec",
        "track_id",
        "feature",
        "interpolated_signal",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"processed_signals.csv missing columns: "
            f"{sorted(missing)}"
        )

    df = df.copy()

    for column in (
        "frame_idx",
        "timestamp_sec",
        "track_id",
        "interpolated_signal",
    ):
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=[
            "frame_idx",
            "track_id",
            "interpolated_signal",
        ]
    )

    df["frame_idx"] = df["frame_idx"].astype(int)
    df["track_id"] = df["track_id"].astype(int)

    return df


# ============================================================
# 3. Track termination event detection
# ============================================================

def detect_track_boundaries(
    tracking_path: Path | None,
    fps: float,
    config: AnalyzerConfig,
) -> list[int]:

    """
    Detect synchronized Track termination events.

    A boundary is generated when multiple Track IDs terminate
    within a short frame interval.

    The final video termination event is excluded.
    """

    if tracking_path is None or not tracking_path.exists():
        LOGGER.warning(
            "Tracking CSV unavailable; "
            "skip event-based splitting."
        )
        return []

    df = pd.read_csv(tracking_path)

    required = {"frame_idx", "track_id"}

    if not required.issubset(df.columns):
        LOGGER.warning(
            "Tracking CSV requires frame_idx and track_id."
        )
        return []

    df = df.dropna(
        subset=["frame_idx", "track_id"]
    ).copy()

    df["frame_idx"] = df["frame_idx"].astype(int)
    df["track_id"] = df["track_id"].astype(int)

    if df.empty:
        return []

    track_ranges = (
        df.groupby("track_id")["frame_idx"]
        .agg(["min", "max"])
        .sort_values("max")
    )

    end_events = sorted(
        int(row["max"]) + 1
        for _, row in track_ranges.iterrows()
    )

    if len(end_events) < config.min_simultaneous_ends:
        return []

    # Cluster nearby termination events.
    clusters = []
    current = [end_events[0]]

    for event in end_events[1:]:

        if (
            event - current[-1]
            <= config.track_end_tolerance_frames
        ):
            current.append(event)

        else:
            clusters.append(current)
            current = [event]

    clusters.append(current)

    boundaries = []

    for cluster in clusters:

        boundary = int(
            round(float(np.median(cluster)))
        )

        # Tracks that were active immediately before boundary.
        active_tracks = track_ranges[
            (track_ranges["min"] < boundary)
            & (track_ranges["max"] >= boundary - 1)
        ]

        active_count = len(active_tracks)

        ended_count = sum(
            1
            for end_frame in active_tracks["max"]
            if abs(
                (int(end_frame) + 1) - boundary
            ) <= config.track_end_tolerance_frames
        )

        ended_ratio = ended_count / max(active_count, 1)

        if ended_count < config.min_simultaneous_ends:
            continue

        if ended_ratio < config.min_simultaneous_end_ratio:
            continue

        boundaries.append(boundary)

    # Exclude the final video termination event.
    video_end = int(df["frame_idx"].max())

    boundaries = [
        boundary
        for boundary in boundaries
        if boundary
        < video_end - config.track_end_tolerance_frames
    ]

    boundaries = sorted(set(boundaries))

    LOGGER.info(
        "Detected internal boundaries: %s",
        boundaries,
    )

    return boundaries


# ============================================================
# 4. Video segmentation
# ============================================================

def make_segments(
    start_frame: int,
    end_frame: int,
    boundaries: list[int],
    fps: float,
    min_segment_sec: float,
) -> list[tuple[int, int]]:

    points = sorted({
        boundary
        for boundary in boundaries
        if start_frame < boundary <= end_frame
    })

    edges = [start_frame] + points + [end_frame + 1]

    segments = [
        (left, right - 1)
        for left, right in zip(
            edges[:-1],
            edges[1:],
        )
        if right > left
    ]

    return segments


# ============================================================
# 5. Track signal alignment
# ============================================================

def prepare_track_signal(
    group: pd.DataFrame,
    start_frame: int,
    end_frame: int,
    config: AnalyzerConfig,
) -> tuple[np.ndarray | None, float, int]:

    """
    Align one Track signal to a shared frame axis.

    Only short internal gaps are interpolated.
    No extrapolation is performed.
    """

    indexed = (
        group.groupby("frame_idx")["interpolated_signal"]
        .mean()
        .sort_index()
    )

    target_index = pd.RangeIndex(
        start_frame,
        end_frame + 1,
    )

    aligned = indexed.reindex(target_index)

    observed = int(aligned.notna().sum())

    coverage = observed / max(
        len(target_index),
        1,
    )

    if coverage < config.min_track_coverage:
        return None, coverage, 0

    missing_before = int(aligned.isna().sum())

    aligned = aligned.interpolate(
        method="linear",
        limit=config.max_interpolation_gap,
        limit_area="inside",
    )

    # Reject remaining missing values.
    if aligned.isna().any():
        return None, coverage, missing_before

    signal = aligned.to_numpy(dtype=float)

    if not np.all(np.isfinite(signal)):
        return None, coverage, missing_before

    return signal, coverage, missing_before


# ============================================================
# 6. Signal filtering
# ============================================================

def bandpass_signal(
    signal: np.ndarray,
    fps: float,
    config: AnalyzerConfig,
) -> np.ndarray | None:

    low_hz = config.min_bpm / 60.0
    high_hz = config.max_bpm / 60.0

    nyquist = fps / 2.0

    if not (
        0 < low_hz < high_hz < nyquist
    ):
        return None

    x = np.asarray(
        signal,
        dtype=float,
    )

    min_samples = int(
        config.min_segment_sec * fps
    )

    if len(x) < min_samples:
        return None

    x = detrend(
        x,
        type="linear",
    )

    sos = butter(
        config.filter_order,
        [low_hz, high_hz],
        btype="bandpass",
        fs=fps,
        output="sos",
    )

    try:
        filtered = sosfiltfilt(
            sos,
            x,
        )

    except ValueError:
        return None

    if not np.all(np.isfinite(filtered)):
        return None

    if np.std(filtered) < 1e-8:
        return None

    return filtered


# ============================================================
# 7. Frequency estimation methods
# ============================================================

def estimate_fft(
    signal: np.ndarray,
    fps: float,
    config: AnalyzerConfig,
) -> float | None:

    x = signal - np.mean(signal)

    n = len(x)

    # Zero-padding improves frequency peak interpolation.
    # It does not improve the true frequency resolution.
    n_fft = max(
        n * 4,
        2048,
    )

    spectrum = np.abs(
        np.fft.rfft(
            x * np.hanning(n),
            n=n_fft,
        )
    ) ** 2

    frequencies = np.fft.rfftfreq(
        n_fft,
        d=1.0 / fps,
    )

    low_hz = config.min_bpm / 60.0
    high_hz = config.max_bpm / 60.0

    valid = (
        (frequencies >= low_hz)
        & (frequencies <= high_hz)
    )

    if not np.any(valid):
        return None

    indices = np.flatnonzero(valid)

    peak_idx = indices[
        np.argmax(spectrum[valid])
    ]

    return float(
        frequencies[peak_idx] * 60.0
    )


def estimate_autocorrelation(
    signal: np.ndarray,
    fps: float,
    config: AnalyzerConfig,
) -> float | None:

    x = signal - np.mean(signal)

    denominator = np.dot(x, x)

    if denominator < 1e-10:
        return None

    acf = np.correlate(
        x,
        x,
        mode="full",
    )[len(x) - 1:]

    acf = acf / denominator

    min_lag = max(
        1,
        int(fps * 60.0 / config.max_bpm),
    )

    max_lag = min(
        len(acf) - 1,
        int(fps * 60.0 / config.min_bpm),
    )

    if max_lag <= min_lag:
        return None

    region = acf[
        min_lag:max_lag + 1
    ]

    peaks, _ = find_peaks(region)

    if len(peaks) == 0:
        return None

    peak_lags = peaks + min_lag

    best_lag = int(
        peak_lags[
            np.argmax(acf[peak_lags])
        ]
    )

    return float(
        60.0 * fps / best_lag
    )


def estimate_peak_detection(
    signal: np.ndarray,
    fps: float,
    config: AnalyzerConfig,
) -> float | None:

    min_distance = max(
        1,
        int(fps * 60.0 / config.max_bpm),
    )

    prominence = max(
        0.5 * float(np.std(signal)),
        1e-8,
    )

    peaks, _ = find_peaks(
        signal,
        distance=min_distance,
        prominence=prominence,
    )

    if len(peaks) < 3:
        return None

    intervals = np.diff(peaks) / fps

    min_period = 60.0 / config.max_bpm
    max_period = 60.0 / config.min_bpm

    intervals = intervals[
        (intervals >= min_period)
        & (intervals <= max_period)
    ]

    if len(intervals) < 2:
        return None

    period = float(
        np.median(intervals)
    )

    return 60.0 / period


def estimate_methods(
    signal: np.ndarray,
    fps: float,
    config: AnalyzerConfig,
) -> dict:

    return {
        "fft_bpm": estimate_fft(
            signal,
            fps,
            config,
        ),
        "autocorrelation_bpm": estimate_autocorrelation(
            signal,
            fps,
            config,
        ),
        "peak_detection_bpm": estimate_peak_detection(
            signal,
            fps,
            config,
        ),
    }


# ============================================================
# 8. FFT cluster selection
# ============================================================

def find_largest_fft_cluster(
    rows: list[dict],
    config: AnalyzerConfig,
) -> list[dict]:

    """
    Find the largest group of Track FFT estimates
    whose BPM range is within fft_cluster_tolerance_bpm.
    """

    candidates = [
        row
        for row in rows
        if row["fft_bpm"] is not None
        and np.isfinite(row["fft_bpm"])
    ]

    candidates.sort(
        key=lambda row: row["fft_bpm"]
    )

    if len(candidates) < config.min_valid_tracks:
        return []

    best_cluster = []

    # Sliding-window search.
    for left in range(len(candidates)):

        cluster = []

        for right in range(left, len(candidates)):

            spread = (
                candidates[right]["fft_bpm"]
                - candidates[left]["fft_bpm"]
            )

            if spread > config.fft_cluster_tolerance_bpm:
                break

            cluster.append(
                candidates[right]
            )

        if len(cluster) > len(best_cluster):
            best_cluster = cluster

        elif (
            len(cluster) == len(best_cluster)
            and cluster
            and best_cluster
        ):
            # Tie-break: prefer the cluster with
            # greater auxiliary-method support.
            current_support = sum(
                row["auxiliary_support_count"]
                for row in cluster
            )

            best_support = sum(
                row["auxiliary_support_count"]
                for row in best_cluster
            )

            if current_support > best_support:
                best_cluster = cluster

    min_required = max(
        config.min_valid_tracks,
        int(
            np.ceil(
                len(candidates)
                * config.min_fft_cluster_fraction
            )
        ),
    )

    if len(best_cluster) < min_required:
        return []

    return best_cluster


# ============================================================
# 9. Feature-level analysis
# ============================================================

def analyze_segment_feature(
    df: pd.DataFrame,
    feature: str,
    start_frame: int,
    end_frame: int,
    fps: float,
    config: AnalyzerConfig,
) -> tuple[list[dict], dict]:

    """
    Analyze one motion feature.

    1. Align Track signals.
    2. Remove the across-Track common component.
    3. Apply bandpass filtering.
    4. Estimate FFT and auxiliary BPM.
    5. Select the largest FFT cluster.
    """

    segment = df[
        (df["feature"] == feature)
        & (df["frame_idx"] >= start_frame)
        & (df["frame_idx"] <= end_frame)
    ]

    track_signals = {}
    track_coverage = {}

    for track_id, group in segment.groupby("track_id"):

        signal, coverage, _ = prepare_track_signal(
            group,
            start_frame,
            end_frame,
            config,
        )

        if signal is not None:

            track_signals[int(track_id)] = signal

            track_coverage[int(track_id)] = coverage

    if len(track_signals) < config.min_tracks:

        return [], {
            "feature": feature,
            "status": "INSUFFICIENT_TRACKS",
            "valid_track_count": len(track_signals),
            "estimated_track_count": 0,
            "cluster_track_count": 0,
            "cluster_bpm": None,
            "cluster_track_ids": [],
        }

    matrix = np.vstack(
        list(track_signals.values())
    )

    # Robust common component removal.
    common_component = np.median(
        matrix,
        axis=0,
    )

    rows = []

    for track_id, original in track_signals.items():

        residual = (
            original - common_component
        )

        filtered = bandpass_signal(
            residual,
            fps,
            config,
        )

        if filtered is None:
            continue

        methods = estimate_methods(
            filtered,
            fps,
            config,
        )

        fft_bpm = methods["fft_bpm"]

        if (
            fft_bpm is None
            or not np.isfinite(fft_bpm)
        ):
            continue

        auxiliary_estimates = [
            methods["autocorrelation_bpm"],
            methods["peak_detection_bpm"],
        ]

        auxiliary_support_count = sum(
            1
            for value in auxiliary_estimates
            if (
                value is not None
                and np.isfinite(value)
                and abs(value - fft_bpm)
                <= config.max_auxiliary_disagreement_bpm
            )
        )

        available_estimates = [
            value
            for value in methods.values()
            if (
                value is not None
                and np.isfinite(value)
            )
        ]

        method_spread = (
            max(available_estimates)
            - min(available_estimates)
            if len(available_estimates) >= 2
            else None
        )

        rows.append({
            "feature": feature,
            "track_id": track_id,
            "coverage": track_coverage[track_id],

            **methods,

            # FFT is the primary Track estimate.
            "track_bpm": float(fft_bpm),

            "method_spread_bpm": method_spread,

            "auxiliary_support_count": (
                auxiliary_support_count
            ),

            "method_agreement": (
                "OK"
                if auxiliary_support_count >= 1
                else "DISAGREE"
            ),

            "cluster_member": False,
        })

    cluster = find_largest_fft_cluster(
        rows,
        config,
    )

    cluster_track_ids = {
        row["track_id"]
        for row in cluster
    }

    for row in rows:

        row["cluster_member"] = (
            row["track_id"] in cluster_track_ids
        )

    cluster_bpm = (
        float(
            np.median([
                row["fft_bpm"]
                for row in cluster
            ])
        )
        if cluster
        else None
    )

    summary = {
        "feature": feature,

        "status": (
            "OK"
            if cluster
            else "NO_RELIABLE_CLUSTER"
        ),

        "valid_track_count": len(track_signals),

        "estimated_track_count": len(rows),

        "cluster_track_count": len(cluster),

        "cluster_bpm": cluster_bpm,

        "cluster_track_ids": sorted(
            cluster_track_ids
        ),

        "auxiliary_supported_count": sum(
            row["auxiliary_support_count"] >= 1
            for row in cluster
        ),
    }

    return rows, summary


# ============================================================
# 10. Segment-level analysis
# ============================================================

def analyze_segment(
    df: pd.DataFrame,
    start_frame: int,
    end_frame: int,
    fps: float,
    config: AnalyzerConfig,
) -> tuple[list[dict], dict]:

    duration_sec = (
        end_frame - start_frame + 1
    ) / fps

    if duration_sec < config.min_segment_sec:

        return [], {
            "start_frame": start_frame,
            "end_frame": end_frame,
            "duration_sec": duration_sec,
            "status": "MEASUREMENT_FAILED",
            "failure_reason": (
                "INSUFFICIENT_SEGMENT_DURATION"
            ),
            "respiratory_rate": None,
            "feature_estimates": {},
            "feature_summaries": [],
        }

    feature_rows = []
    feature_summaries = []

    features = (
        config.primary_feature,
        *config.diagnostic_features,
    )

    for feature in features:

        rows, feature_summary = analyze_segment_feature(
            df,
            feature,
            start_frame,
            end_frame,
            fps,
            config,
        )

        feature_rows.extend(rows)
        feature_summaries.append(
            feature_summary
        )

    # Primary feature determines the measurement.
    primary_summary = next(
        (
            item
            for item in feature_summaries
            if item["feature"] == config.primary_feature
        ),
        None,
    )

    feature_estimates = {}

    if (
        primary_summary is not None
        and primary_summary["status"] == "OK"
    ):

        bpm = primary_summary["cluster_bpm"]

        feature_estimates[
            config.primary_feature
        ] = bpm

        status = "OK"
        reason = None

    else:

        bpm = None
        status = "MEASUREMENT_FAILED"
        reason = "NO_RELIABLE_FFT_CLUSTER"

    summary = {
        "start_frame": start_frame,
        "end_frame": end_frame,
        "duration_sec": duration_sec,

        "status": status,
        "failure_reason": reason,

        "respiratory_rate": bpm,

        "feature_estimates": feature_estimates,

        "feature_summaries": feature_summaries,
    }

    return feature_rows, summary


# ============================================================
# 11. Full analysis
# ============================================================

def analyze(
    input_path: Path,
    output_dir: Path,
    fps: float,
    tracking_path: Path | None,
    config: AnalyzerConfig,
) -> dict:

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if fps <= 0:
        raise ValueError("FPS must be positive.")

    df = load_signals(input_path)

    target_features = {
        config.primary_feature,
        *config.diagnostic_features,
    }

    df = df[
        df["feature"].isin(target_features)
    ].copy()

    if df.empty:
        raise ValueError(
            f"No target features found: {target_features}"
        )

    start_frame = int(
        df["frame_idx"].min()
    )

    end_frame = int(
        df["frame_idx"].max()
    )

    # Detect internal posture / tracking termination events.
    boundaries = detect_track_boundaries(
        tracking_path,
        fps,
        config,
    )

    segments = make_segments(
        start_frame,
        end_frame,
        boundaries,
        fps,
        config.min_segment_sec,
    )

    if not segments:
        segments = [
            (start_frame, end_frame)
        ]

    all_track_rows = []
    segment_summaries = []

    for segment_idx, (left, right) in enumerate(segments):

        rows, summary = analyze_segment(
            df,
            left,
            right,
            fps,
            config,
        )

        summary["segment_id"] = segment_idx

        all_track_rows.extend(
            [
                {
                    **row,
                    "segment_id": segment_idx,
                }
                for row in rows
            ]
        )

        segment_summaries.append(
            summary
        )

    # --------------------------------------------------------
    # Segment-level validation
    # --------------------------------------------------------

    valid_segments = [
        item
        for item in segment_summaries
        if (
            item["status"] == "OK"
            and item["respiratory_rate"] is not None
        )
    ]

    if not valid_segments:

        result = {
            "status": "measurement_failed",
            "reason_code": "NO_RELIABLE_SEGMENT",
            "respiratory_rate": None,
        }

    else:

        segment_bpms = [
            item["respiratory_rate"]
            for item in valid_segments
        ]

        if (
            len(segment_bpms) >= 2
            and max(segment_bpms) - min(segment_bpms)
            > config.max_segment_disagreement_bpm
        ):

            result = {
                "status": "measurement_failed",
                "reason_code": "SEGMENT_DISAGREEMENT",
                "respiratory_rate": None,
            }

        elif len(valid_segments) != len(segment_summaries):

            result = {
                "status": "measurement_failed",
                "reason_code": "PARTIAL_SEGMENT_FAILURE",
                "respiratory_rate": None,
            }

        else:

            bpm = float(
                np.median(segment_bpms)
            )

            result = {
                "status": "success",
                "respiratory_rate": int(
                    round(bpm)
                ),
                "reason_code": None,
            }

    # ========================================================
    # 12. Save outputs
    # ========================================================

    track_columns = [
        "feature",
        "track_id",
        "coverage",
        "fft_bpm",
        "autocorrelation_bpm",
        "peak_detection_bpm",
        "track_bpm",
        "method_spread_bpm",
        "auxiliary_support_count",
        "method_agreement",
        "cluster_member",
        "segment_id",
    ]

    track_df = pd.DataFrame(
        all_track_rows,
        columns=track_columns,
    )

    track_df.to_csv(
        output_dir / "track_estimates.csv",
        index=False,
    )

    with open(
        output_dir / "segment_analysis.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            segment_summaries,
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(
        output_dir / "measurement_result.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            result,
            f,
            ensure_ascii=False,
            indent=2,
        )

    metadata = {
        "input": str(input_path),

        "tracking_input": (
            str(tracking_path)
            if tracking_path
            else None
        ),

        "fps": fps,

        "config": asdict(config),

        "detected_boundaries": boundaries,

        "segments": len(segments),

        "primary_feature": config.primary_feature,
    }

    with open(
        output_dir / "analysis_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2,
        )

    LOGGER.info(
        "Analysis completed: %s",
        result,
    )

    return result


# ============================================================
# 13. CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Estimate respiratory rate "
            "from processed ROI motion signals."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to processed_signals.csv",
    )

    parser.add_argument(
        "--tracking",
        type=Path,
        default=None,
        help="Optional path to roi_tracking.csv",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
    )

    parser.add_argument(
        "--min-bpm",
        type=float,
        default=15.0,
    )

    parser.add_argument(
        "--max-bpm",
        type=float,
        default=45.0,
    )

    parser.add_argument(
        "--min-segment-sec",
        type=float,
        default=16.0,
    )

    parser.add_argument(
        "--primary-feature",
        type=str,
        default="mean_flow_y",
        choices=[
            "mean_flow_y",
            "mean_flow_x",
        ],
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s "
            "[%(levelname)s] %(message)s"
        ),
    )

    config = AnalyzerConfig(
        min_bpm=args.min_bpm,
        max_bpm=args.max_bpm,
        min_segment_sec=args.min_segment_sec,
        primary_feature=args.primary_feature,
    )

    result = analyze(
        input_path=args.input,
        output_dir=args.output_dir,
        fps=args.fps,
        tracking_path=args.tracking,
        config=config,
    )

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()