"""Ask a configured VLM the questions in a JSON file about an MP4 video.

Run from the repository root:
    python -m QA.run_questionnaire --video drive.mp4 --questions questions.json
    python -m QA.run_questionnaire --frames-dir camera --questions questions.json
"""

import argparse
import json
from pathlib import Path

from pipeline.ConfigLoader import ConfigLoader
from QA.QuestionnaireParser import QuestionnaireParser
from QA.VideoAIParser import VideoAIParser


QA_DIR = Path(__file__).resolve().parent
PROMPT_FILE = QA_DIR / "prompts" / "answer_questions.txt"
DEFAULT_CONFIG = QA_DIR.parent / "configs" / "qa_gpt6_luna.json"


def build_prompt(prompt_template: str, questionnaire: dict) -> str:
    """Add only the question text and options to the short answer prompt."""
    questions = [
        {
            "type": question.get("type"),
            "question": question["question"],
            "options": question["options"],
        }
        for question in questionnaire["questions"]
    ]
    return (
        f"{prompt_template.rstrip()}\n\n"
        "Questions:\n"
        f"{json.dumps(questions, indent=2, ensure_ascii=False)}"
    )


def parse_response(response: str, questions: list[dict]) -> list[dict]:
    """Map positional answer-index lists back to the shuffled question IDs."""
    text = response.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    parsed = None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass

    answers = []
    answer_lists = parsed if isinstance(parsed, list) else []
    for position, question in enumerate(questions):
        indices = answer_lists[position] if position < len(answer_lists) else None
        if isinstance(indices, int) and not isinstance(indices, bool):
            indices = [indices]

        valid = (
            isinstance(indices, list)
            and bool(indices)
            and all(
                isinstance(index, int)
                and not isinstance(index, bool)
                and 0 <= index < len(question["options"])
                for index in indices
            )
            and len(set(indices)) == len(indices)
        )
        if valid and question.get("type") == "sequence_order":
            valid = set(indices) == set(range(len(question["options"])))
        elif valid and question.get("type") in {
            "existence",
            "deceptive",
            "noteworthy",
            "pairwise_order",
            "before_after",
        }:
            valid = len(indices) == 1

        answers.append(
            {
                "question_id": question["question_id"],
                "answer_indices": indices if valid else None,
            }
        )
    return answers


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask a VLM the benchmark questions about a video"
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--video", help="Path to the source .mp4 video")
    input_group.add_argument(
        "--frames-dir",
        help="Folder of timestamp-named camera frames (frame input mode only)",
    )
    parser.add_argument(
        "--questions",
        required=True,
        help="Path to a benchmark questions.json",
    )
    parser.add_argument(
        "--output",
        help="Path to write model answers (default: <video-stem>_qa_answers.json)",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Model configuration JSON (default: configs/qa_gpt6_luna.json)",
    )
    parser.add_argument(
        "--input-mode",
        choices=("frames", "native_video"),
        help="Override input_mode in the model config",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=100,
        help="Frame mode only: max uniformly sampled frames; 0 uses all (default: 100)",
    )
    parser.add_argument(
        "--max-image-size",
        type=int,
        default=768,
        help="Frame mode only: max image width/height sent to model (default: 768)",
    )
    parser.add_argument(
        "--api-batch-size",
        type=int,
        help=(
            "Override the model config's frame/API images per request "
            "(config fallback: 5)"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Optional random seed for reproducible question ordering",
    )
    args = parser.parse_args()

    if not PROMPT_FILE.is_file():
        raise SystemExit(f"Prompt file not found: {PROMPT_FILE}")

    questionnaire = QuestionnaireParser(seed=args.seed).load(args.questions)
    questions = questionnaire["questions"]
    prompt = build_prompt(PROMPT_FILE.read_text(), questionnaire)
    config = ConfigLoader(args.config).load()

    video_path = Path(args.video).expanduser() if args.video else None
    frames_dir = Path(args.frames_dir).expanduser() if args.frames_dir else None
    if video_path is not None and not video_path.is_file():
        raise SystemExit(f"Video file not found: {video_path}")
    if frames_dir is not None and not frames_dir.is_dir():
        raise SystemExit(f"Frames folder not found: {frames_dir}")

    input_mode = args.input_mode or config.get("input_mode", "frames")
    api_batch_size = (
        args.api_batch_size
        if args.api_batch_size is not None
        else config.get("api_batch_size", 5)
    )
    if frames_dir is not None and input_mode != "frames":
        raise SystemExit("--frames-dir can only be used with --input-mode frames")
    if input_mode == "native_video" and video_path is None:
        raise SystemExit("--input-mode native_video requires --video")
    if input_mode == "frames":
        if args.max_frames < 0:
            raise SystemExit("--max-frames must be 0 (all frames) or greater")
        if args.max_image_size < 1:
            raise SystemExit("--max-image-size must be at least 1")
        if api_batch_size < 1:
            raise SystemExit("--api-batch-size must be at least 1")

    input_path = video_path if video_path is not None else frames_dir
    output_stem = video_path.stem if video_path is not None else frames_dir.name
    output_path = (
        Path(args.output).expanduser()
        if args.output
        else Path(f"{output_stem}_qa_answers.json")
    )

    video_parser = VideoAIParser(config, input_mode=input_mode)
    if video_parser.input_mode == "native_video":
        input_summary = f"native MP4 input from {video_path}"
    elif frames_dir is not None:
        input_summary = f"up to {args.max_frames} sampled frames from {frames_dir}"
    else:
        input_summary = f"up to {args.max_frames} uniformly sampled frames"
    print(
        f"Asking {len(questions)} questions using {input_summary}",
        flush=True,
    )
    if frames_dir is not None:
        raw_response, input_metadata = video_parser.call_frames_dir(
            frames_dir,
            prompt,
            max_frames=args.max_frames,
            max_image_size=args.max_image_size,
            api_batch_size=api_batch_size,
        )
    else:
        raw_response, input_metadata = video_parser.call_video(
            video_path,
            prompt,
            max_frames=args.max_frames,
            max_image_size=args.max_image_size,
            api_batch_size=api_batch_size,
        )

    result = {
        "model": config["model"],
        "backend": config.get("backend"),
        "input_source": str(input_path.resolve()),
        "questions_file": str(Path(args.questions).expanduser().resolve()),
        "input_mode": video_parser.input_mode,
        "question_order": [question["question_id"] for question in questions],
        "seed": args.seed,
        "input_metadata": input_metadata,
        "answers": parse_response(raw_response, questions),
        "raw_response": raw_response,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Answers written to {output_path}", flush=True)


if __name__ == "__main__":
    main()
