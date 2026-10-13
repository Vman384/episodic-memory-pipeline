#!/usr/bin/env python3
"""Print a model's arithmetic mean dataset mark."""

import argparse
import re
import sys
from pathlib import Path


MARK_PATTERN = re.compile(r"\s*(\d+)\s*/\s*(\d+)\s*\(\s*[\d.]+%\s*\)\s*")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Average a model's final marks across completed datasets."
    )
    parser.add_argument("model", help="Model folder under model_answers/")
    parser.add_argument(
        "--benchmark",
        default="temporal",
        help="Benchmark subfolder to read (default: temporal)",
    )
    parser.add_argument(
        "--answers-root",
        type=Path,
        default=Path("model_answers"),
        help="Root containing model output folders (default: model_answers)",
    )
    args = parser.parse_args()

    model_dir = args.answers_root / args.model
    if not model_dir.is_dir():
        print(f"Model output folder not found: {model_dir}", file=sys.stderr)
        return 1

    dataset_dirs = sorted(path for path in model_dir.iterdir() if path.is_dir())
    marks = []
    for dataset_dir in dataset_dirs:
        mark_path = dataset_dir / args.benchmark / "final_mark.txt"
        if not mark_path.is_file():
            continue

        match = MARK_PATTERN.fullmatch(mark_path.read_text())
        if match is None:
            print(f"Invalid mark format in {mark_path}", file=sys.stderr)
            return 1

        correct, total = map(int, match.groups())
        if total == 0 or correct > total:
            print(f"Invalid score in {mark_path}", file=sys.stderr)
            return 1
        marks.append(correct / total * 100)

    if not marks:
        print(
            f"No {args.benchmark} final marks found for {args.model} in {model_dir}",
            file=sys.stderr,
        )
        return 1

    average = sum(marks) / len(marks)
    print(
        f"Average mark for {args.model} ({args.benchmark}) over "
        f"{len(marks)} dataset marks: {average:.2f}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
