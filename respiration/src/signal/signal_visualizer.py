
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


FEATURES = [
    "median_flow_x",
    "median_flow_y",
    "mean_flow_x",
    "mean_flow_y",
    "mean_magnitude",
]

SIGNAL_COLUMNS = [
    ("raw_signal", "Raw Signal"),
    ("detrended_signal", "Detrended"),
    ("filtered_signal", "Band-pass Filtered"),
    ("normalized_signal", "Normalized"),
]


def load_processed_signals(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)

    required = {
        "frame_idx",
        "timestamp_sec",
        "track_id",
        "feature",
        *[column for column, _ in SIGNAL_COLUMNS],
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = df.dropna(subset=["track_id", "feature"]).copy()
    df["track_id"] = df["track_id"].astype(int)
    df["frame_idx"] = df["frame_idx"].astype(int)

    df = df.sort_values(
        ["track_id", "feature", "frame_idx"]
    )

    return df


def plot_track_details(
    track_id: int,
    track_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(
        nrows=len(FEATURES),
        ncols=len(SIGNAL_COLUMNS),
        figsize=(20, 15),
        sharex="row",
    )

    fig.suptitle(
        f"Respiratory Signal Processing - Track {track_id}",
        fontsize=16,
    )

    for row_idx, feature in enumerate(FEATURES):
        feature_df = track_df[
            track_df["feature"] == feature
        ].sort_values("timestamp_sec")

        if feature_df.empty:
            for col_idx in range(len(SIGNAL_COLUMNS)):
                axes[row_idx, col_idx].text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=axes[row_idx, col_idx].transAxes,
                )
            continue

        time = feature_df["timestamp_sec"]

        for col_idx, (column, title) in enumerate(SIGNAL_COLUMNS):
            ax = axes[row_idx, col_idx]

            ax.plot(
                time,
                feature_df[column],
                linewidth=0.8,
            )

            if row_idx == 0:
                ax.set_title(title)

            if col_idx == 0:
                ax.set_ylabel(feature)

            ax.grid(True, alpha=0.3)

            if row_idx == len(FEATURES) - 1:
                ax.set_xlabel("Time (sec)")

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    output_path = output_dir / f"track_{track_id:02d}.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_feature_comparison(
    feature: str,
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    feature_df = df[df["feature"] == feature]

    fig, ax = plt.subplots(figsize=(15, 6))

    for track_id, track_df in feature_df.groupby("track_id"):
        track_df = track_df.sort_values("timestamp_sec")

        duration = (
            track_df["timestamp_sec"].max()
            - track_df["timestamp_sec"].min()
        )

        ax.plot(
            track_df["timestamp_sec"],
            track_df["normalized_signal"],
            linewidth=0.9,
            label=f"Track {track_id} ({duration:.1f}s)",
        )

    ax.set_title(
        f"Normalized Signal Comparison - {feature}"
    )
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Normalized amplitude (Z-score)")
    ax.grid(True, alpha=0.3)
    ax.legend(
        loc="upper right",
        fontsize=8,
        ncol=2,
    )

    fig.tight_layout()

    output_path = output_dir / f"comparison_{feature}.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_visualization(
    input_path: Path,
    output_dir: Path,
) -> None:
    df = load_processed_signals(input_path)

    track_output_dir = output_dir / "tracks"
    comparison_output_dir = output_dir / "comparisons"

    track_output_dir.mkdir(parents=True, exist_ok=True)
    comparison_output_dir.mkdir(parents=True, exist_ok=True)

    valid_features = set(FEATURES) & set(df["feature"].unique())

    track_ids = sorted(df["track_id"].unique())

    print("=== Signal Visualization ===")
    print(f"Input: {input_path}")
    print(f"Tracks: {len(track_ids)}")
    print(f"Features: {len(valid_features)}")

    # A. Detailed plots for each track.
    for track_id, track_df in df.groupby("track_id"):
        plot_track_details(
            track_id=int(track_id),
            track_df=track_df,
            output_dir=track_output_dir,
        )

    # B. Compare normalized signals across tracks.
    for feature in FEATURES:
        if feature not in valid_features:
            continue

        plot_feature_comparison(
            feature=feature,
            df=df,
            output_dir=comparison_output_dir,
        )

    print("\nVisualization completed.")
    print(f"Track plots: {track_output_dir}")
    print(f"Feature comparisons: {comparison_output_dir}")


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

    args = parser.parse_args()

    run_visualization(
        input_path=args.input,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
