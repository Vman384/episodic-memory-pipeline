"""Sample and stage one dataset's camera frames for local Vertex AI inference."""

import argparse
import json
import shutil
from pathlib import Path

from QA.VideoAIParser import VideoAIParser


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frame-budget", type=int, default=512)
    parser.add_argument("--max-image-size", type=int, default=512)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir()

    _, metadata = VideoAIParser._write_sampled_frame_directory(
        Path(args.frames_dir),
        frames_dir,
        args.frame_budget,
        args.max_image_size,
    )
    metadata["dataset"] = args.dataset
    (output_dir / "frame_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    shutil.copy2(args.questions, output_dir / "questions.json")
    print(
        f"Staged {metadata['frames_used']} frames for {args.dataset} "
        f"in {output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
