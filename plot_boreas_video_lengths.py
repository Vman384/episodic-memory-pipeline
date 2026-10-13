#!/usr/bin/env python3
"""Plot the frequency of Boreas video lengths multiplied by three."""

import argparse
import math
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_DATASET_ROOT = Path(
    "/g/data/pg06/FYP2026S1_3473/boreas_dataset"
)


def get_video_duration_seconds(video_path: Path) -> float:
    """Read a video's duration using ffprobe, which is included with ffmpeg."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    duration = float(result.stdout.strip())
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"invalid duration returned by ffprobe: {duration}")
    return duration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read each Boreas video.mp4 duration, multiply it by three, "
            "and save a frequency histogram."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=f"Boreas dataset directory (default: {DEFAULT_DATASET_ROOT})",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=8,
        help="Number of histogram bins (6-10; default: 8)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("boreas_video_length_frequency.png"),
        help="Path for the output plot (default: boreas_video_length_frequency.png)",
    )
    args = parser.parse_args()
    if not 6 <= args.bins <= 10:
        parser.error("--bins must be between 6 and 10")
    return args


def main() -> int:
    args = parse_args()
    if shutil.which("ffprobe") is None:
        print(
            "Error: ffprobe was not found; install ffmpeg and ensure it is on PATH.",
            file=sys.stderr,
        )
        return 1
    if not args.dataset_root.is_dir():
        print(f"Error: dataset directory does not exist: {args.dataset_root}", file=sys.stderr)
        return 1

    # Each item is (dataset name, video duration in seconds multiplied by 3).
    video_lengths = []
    datasets = sorted(
        path for path in args.dataset_root.glob("boreas-*") if path.is_dir()
    )
    for dataset in datasets:
        video_path = dataset / "video.mp4"
        if not video_path.is_file():
            print(f"Skipping {dataset.name}: video.mp4 not found", file=sys.stderr)
            continue
        try:
            duration_seconds = get_video_duration_seconds(video_path)
        except (subprocess.CalledProcessError, ValueError) as error:
            print(
                f"Skipping {dataset.name}: could not read video duration ({error})",
                file=sys.stderr,
            )
            continue

        scaled_duration_seconds = duration_seconds * 3
        video_lengths.append((dataset.name, scaled_duration_seconds))
        print(
            f"{dataset.name}: {duration_seconds / 60:.2f} min "
            f"(x3: {scaled_duration_seconds / 60:.2f} min)"
        )

    if not video_lengths:
        print(f"No readable video.mp4 files found under {args.dataset_root}", file=sys.stderr)
        return 1

    scaled_lengths_minutes = [duration / 60 for _, duration in video_lengths]

    figure, axis = plt.subplots()
    _, bin_edges, _ = axis.hist(
        scaled_lengths_minutes, bins=args.bins, edgecolor="black"
    )
    axis.set_title("Frequency of Boreas Video Lengths (Duration × 3)")
    axis.set_xlabel("Video length × 3 (minutes)")
    axis.set_ylabel("Number of videos")
    axis.set_xticks(bin_edges)
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=150)
    plt.close(figure)
    print(f"Plotted {len(video_lengths)} videos; saved frequency plot to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
