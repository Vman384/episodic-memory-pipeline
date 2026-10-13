#!/usr/bin/env python3
"""Generate a duration-stratified LaTeX table from the benchmark source data."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from pipeline.timeframe_converter import parse_frame_timestamp


DEFAULT_DATA_ROOT = Path("/g/data/pg06/FYP2026S1_3473")
DEFAULT_SEQUENCES = (
    "boreas-2025-07-18-14-55",
    "boreas-2024-12-03-13-13",
    "boreas-2024-12-03-13-34",
    "boreas-2024-12-04-11-45",
    "boreas-2024-12-04-11-56",
    "boreas-2025-07-18-15-12",
    "boreas-2024-12-04-14-44",
    "boreas-2025-02-15-16-58",
    "boreas-2025-07-18-11-53",
    "boreas-2024-12-04-14-34",
    "boreas-2024-12-04-14-38",
    "boreas-2025-02-21-14-51",
    "boreas-2025-02-15-17-19",
    "boreas-2024-12-04-14-59",
    "boreas-2025-07-18-10-33",
    "boreas-2025-02-22-12-26",
    "boreas-2025-08-13-10-36",
)
FRAME_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
MICROSECONDS_PER_SECOND = 1_000_000


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    frames_dir: Path
    sparse_questions: Path
    temporal_questions: Path


@dataclass(frozen=True)
class SequenceStats:
    name: str
    duration_seconds: float
    sparse_question_count: int
    temporal_question_count: int

    @property
    def total_question_count(self) -> int:
        return self.sparse_question_count + self.temporal_question_count


@dataclass(frozen=True)
class SummaryRow:
    group: str
    video_count: int
    mean_duration_minutes: float | None
    sparse_question_count: int
    temporal_question_count: int

    @property
    def total_question_count(self) -> int:
        return self.sparse_question_count + self.temporal_question_count


def boreas_spec(data_root: Path, sequence: str) -> DatasetSpec:
    """Build input paths for one Boreas sequence."""
    sequence_root = data_root / sequence
    return DatasetSpec(
        name=sequence,
        frames_dir=data_root / "boreas_dataset" / sequence / "camera",
        sparse_questions=(
            sequence_root / "sparse_outputs" / "events" / "questions.json"
        ),
        temporal_questions=(
            sequence_root
            / "temporal_outputs"
            / "narratives"
            / "questions.json"
        ),
    )


def forest_k01_spec(data_root: Path) -> DatasetSpec:
    """Build input paths for the optional CSIRO K-01 forest sequence."""
    forest_root = data_root / "forest_dataset"
    return DatasetSpec(
        name="CSIRO forest K-01",
        frames_dir=forest_root / "K-01_data",
        sparse_questions=(
            forest_root / "K-01_sparse_outputs" / "events" / "questions.json"
        ),
        temporal_questions=(
            forest_root
            / "K-01_temporal_outputs"
            / "narratives"
            / "questions.json"
        ),
    )


def frame_duration_seconds(frames_dir: Path) -> float:
    """Measure elapsed duration from the earliest/latest timestamped frame."""
    if not frames_dir.is_dir():
        raise FileNotFoundError(f"camera frame directory not found: {frames_dir}")

    timestamps = []
    for frame_path in frames_dir.iterdir():
        if (
            not frame_path.is_file()
            or frame_path.suffix.lower() not in FRAME_EXTENSIONS
        ):
            continue
        try:
            timestamps.append(parse_frame_timestamp(frame_path))
        except ValueError:
            # Ignore non-timestamped image files; the duration is based only on
            # frame names that the benchmark's timestamp parser recognizes.
            continue

    if len(timestamps) < 2:
        raise ValueError(
            f"expected at least two timestamped frames in {frames_dir}; "
            f"found {len(timestamps)}"
        )

    duration_seconds = (
        max(timestamps) - min(timestamps)
    ) / MICROSECONDS_PER_SECOND
    if duration_seconds <= 0:
        raise ValueError(f"non-positive frame duration in {frames_dir}")
    return duration_seconds


def question_count(questions_path: Path) -> int:
    """Return the number of question objects in a generated questions file."""
    if not questions_path.is_file():
        raise FileNotFoundError(f"question file not found: {questions_path}")

    try:
        document = json.loads(questions_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {questions_path}: {error}") from error

    if isinstance(document, list):
        questions = document
    elif isinstance(document, dict):
        questions = document.get("questions")
    else:
        questions = None

    if not isinstance(questions, list):
        raise ValueError(
            f"expected a top-level 'questions' list in {questions_path}"
        )
    if any(not isinstance(question, dict) for question in questions):
        raise ValueError(f"non-object entry in questions list: {questions_path}")
    return len(questions)


def collect_sequence_stats(spec: DatasetSpec) -> SequenceStats:
    """Read one sequence's frame duration and sparse/temporal QA counts."""
    return SequenceStats(
        name=spec.name,
        duration_seconds=frame_duration_seconds(spec.frames_dir),
        sparse_question_count=question_count(spec.sparse_questions),
        temporal_question_count=question_count(spec.temporal_questions),
    )


def duration_group(
    duration_seconds: float,
    short_max_minutes: float = 10,
    medium_max_minutes: float = 15,
) -> str:
    """Assign a sequence to a duration bin using the configured cutoffs."""
    duration_minutes = duration_seconds / 60
    if duration_minutes <= short_max_minutes:
        return "Short"
    if duration_minutes <= medium_max_minutes:
        return "Medium"
    return "Long"


def summarize_sequences(
    sequences: list[SequenceStats],
    short_max_minutes: float = 10,
    medium_max_minutes: float = 15,
) -> list[SummaryRow]:
    """Aggregate videos and question counts into duration groups."""
    grouped: dict[str, list[SequenceStats]] = {
        "Short": [],
        "Medium": [],
        "Long": [],
    }
    for sequence in sequences:
        grouped[
            duration_group(
                sequence.duration_seconds,
                short_max_minutes=short_max_minutes,
                medium_max_minutes=medium_max_minutes,
            )
        ].append(sequence)

    rows = []
    for group_name in ("Short", "Medium", "Long"):
        group_sequences = grouped[group_name]
        durations_minutes = [
            item.duration_seconds / 60 for item in group_sequences
        ]
        rows.append(
            SummaryRow(
                group=group_name,
                video_count=len(group_sequences),
                mean_duration_minutes=(
                    sum(durations_minutes) / len(durations_minutes)
                    if durations_minutes
                    else None
                ),
                sparse_question_count=sum(
                    item.sparse_question_count for item in group_sequences
                ),
                temporal_question_count=sum(
                    item.temporal_question_count for item in group_sequences
                ),
            )
        )

    durations_minutes = [item.duration_seconds / 60 for item in sequences]
    rows.append(
        SummaryRow(
            group="All",
            video_count=len(sequences),
            mean_duration_minutes=(
                sum(durations_minutes) / len(durations_minutes)
                if durations_minutes
                else None
            ),
            sparse_question_count=sum(
                item.sparse_question_count for item in sequences
            ),
            temporal_question_count=sum(
                item.temporal_question_count for item in sequences
            ),
        )
    )
    return rows


def group_label(
    group: str,
    short_max_minutes: float,
    medium_max_minutes: float,
    latex: bool = False,
) -> str:
    """Format a duration group label for LaTeX or CSV output."""
    if group == "Short":
        if latex:
            relation = rf"$t \leq {short_max_minutes:g}$ min"
        else:
            relation = f"<= {short_max_minutes:g} min"
        return f"Short ({relation})"
    if group == "Medium":
        if latex:
            relation = (
                rf"${short_max_minutes:g} < t \leq {medium_max_minutes:g}$ min"
            )
        else:
            relation = f"> {short_max_minutes:g} and <= {medium_max_minutes:g} min"
        return f"Medium ({relation})"
    if group == "Long":
        if latex:
            relation = rf"$t > {medium_max_minutes:g}$ min"
        else:
            relation = f"> {medium_max_minutes:g} min"
        return f"Long ({relation})"
    return group


def render_latex_table(
    sequences: list[SequenceStats],
    short_max_minutes: float = 10,
    medium_max_minutes: float = 15,
) -> str:
    """Render an IEEE two-column LaTeX table using booktabs."""
    rows = summarize_sequences(
        sequences,
        short_max_minutes=short_max_minutes,
        medium_max_minutes=medium_max_minutes,
    )
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"\textbf{Duration group} & \textbf{\# Videos} & "
        r"\textbf{Mean duration (min)} & \textbf{Sparse-event QA} & "
        r"\textbf{Temporal-order QA} & \textbf{Total QA} "
        r"\\",
        r"\midrule",
    ]

    for row in rows:
        label = group_label(
            row.group,
            short_max_minutes,
            medium_max_minutes,
            latex=True,
        )
        mean_duration = (
            f"{row.mean_duration_minutes:.2f}"
            if row.mean_duration_minutes is not None
            else "--"
        )
        lines.append(
            f"{label} & {row.video_count} & {mean_duration} & "
            f"{row.sparse_question_count} & {row.temporal_question_count} & "
            f"{row.total_question_count} "
            + r"\\"
        )
        if row.group == "Long":
            lines.append(r"\midrule")

    caption = (
        f"Duration-stratified composition of {len(sequences)} video sequences. "
        "Drive durations are computed from the earliest and latest timestamped "
        "camera frames. Duration bins are "
        f"$t \\leq {short_max_minutes:g}$ min, "
        f"${short_max_minutes:g} < t \\leq {medium_max_minutes:g}$ min, and "
        f"$t > {medium_max_minutes:g}$ min."
    )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            f"\\caption{{{caption}}}",
            r"\label{tab:duration_composition}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def write_csv(
    output_path: Path,
    rows: list[SummaryRow],
    short_max_minutes: float,
    medium_max_minutes: float,
) -> None:
    """Write the aggregated table data as CSV for inspection."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "duration_group",
                "number_of_videos",
                "mean_duration_minutes",
                "sparse_event_questions",
                "temporal_order_questions",
                "total_questions",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    group_label(
                        row.group,
                        short_max_minutes,
                        medium_max_minutes,
                    ),
                    row.video_count,
                    (
                        f"{row.mean_duration_minutes:.2f}"
                        if row.mean_duration_minutes is not None
                        else ""
                    ),
                    row.sparse_question_count,
                    row.temporal_question_count,
                    row.total_question_count,
                ]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate per-drive durations from timestamped camera frames, "
            "count sparse/temporal questions, and write a LaTeX table plus CSV."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Project data directory (default: {DEFAULT_DATA_ROOT})",
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=None,
        help=(
            "Boreas sequence directory names to include; defaults to the 17 "
            "sequences listed in run_all_eval.sh"
        ),
    )
    parser.add_argument(
        "--include-forest-k01",
        action="store_true",
        help="Also include CSIRO forest K-01 (requires both question files)",
    )
    parser.add_argument(
        "--short-max-minutes",
        type=float,
        default=10,
        help="Upper boundary of the short-video bin in minutes (default: 10)",
    )
    parser.add_argument(
        "--medium-max-minutes",
        type=float,
        default=15,
        help="Upper boundary of the medium-video bin in minutes (default: 15)",
    )
    parser.add_argument(
        "--tex-output",
        type=Path,
        default=Path("duration_stratified_table.tex"),
        help="Output LaTeX table path",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=Path("duration_stratified_stats.csv"),
        help="Output CSV statistics path",
    )
    args = parser.parse_args()
    if args.short_max_minutes <= 0:
        parser.error("--short-max-minutes must be positive")
    if args.medium_max_minutes <= args.short_max_minutes:
        parser.error("--medium-max-minutes must exceed --short-max-minutes")
    return args


def main() -> int:
    args = parse_args()
    sequence_names = args.sequences or list(DEFAULT_SEQUENCES)
    if len(set(sequence_names)) != len(sequence_names):
        print("Error: --sequences contains duplicate names", file=sys.stderr)
        return 1

    specs = [boreas_spec(args.data_root, name) for name in sequence_names]
    if args.include_forest_k01:
        specs.append(forest_k01_spec(args.data_root))
    if not specs:
        print("Error: no sequences selected", file=sys.stderr)
        return 1

    sequence_stats = []
    try:
        for spec in specs:
            stats = collect_sequence_stats(spec)
            sequence_stats.append(stats)
            minutes = stats.duration_seconds / 60
            group = duration_group(
                stats.duration_seconds,
                short_max_minutes=args.short_max_minutes,
                medium_max_minutes=args.medium_max_minutes,
            )
            print(
                f"{stats.name}: {minutes:.2f} min ({group}); "
                f"sparse QA={stats.sparse_question_count}, "
                f"temporal QA={stats.temporal_question_count}"
            )
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    rows = summarize_sequences(
        sequence_stats,
        short_max_minutes=args.short_max_minutes,
        medium_max_minutes=args.medium_max_minutes,
    )
    try:
        args.tex_output.parent.mkdir(parents=True, exist_ok=True)
        args.tex_output.write_text(
            render_latex_table(
                sequence_stats,
                short_max_minutes=args.short_max_minutes,
                medium_max_minutes=args.medium_max_minutes,
            ),
            encoding="utf-8",
        )
        write_csv(
            args.csv_output,
            rows,
            short_max_minutes=args.short_max_minutes,
            medium_max_minutes=args.medium_max_minutes,
        )
    except OSError as error:
        print(f"Error writing table outputs: {error}", file=sys.stderr)
        return 1

    all_row = rows[-1]
    print(
        f"Total: {all_row.video_count} videos, "
        f"{all_row.sparse_question_count} sparse-event QA pairs, "
        f"{all_row.temporal_question_count} temporal-order QA pairs, "
        f"{all_row.total_question_count} QA pairs"
    )
    print(f"LaTeX table written to {args.tex_output}")
    print(f"CSV summary written to {args.csv_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
