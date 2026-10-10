"""Ask a configured VLM the questions in a JSON file about an MP4 video.

Run from the repository root:
    python -m QA.run_questionnaire --video drive.mp4 --questions questions.json
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
    """Add the blinded questions to the shared video-answer instructions."""
    return (
        f"{prompt_template.rstrip()}\n\n"
        "Questions JSON:\n"
        f"{json.dumps(questionnaire, indent=2, ensure_ascii=False)}"
    )


def parse_response(response: str, questions: list[dict]) -> list[dict]:
    """Extract one valid answer-index list per question from model JSON."""
    text = response.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    start = text.find("{")
    end = text.rfind("}")
    parsed = None
    if start != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    valid_questions = {question["question_id"]: question for question in questions}
    parsed_answers = {}
    if isinstance(parsed, dict) and isinstance(parsed.get("answers"), list):
        for answer in parsed["answers"]:
            if not isinstance(answer, dict):
                continue
            question_id = answer.get("question_id")
            if isinstance(question_id, bool) or not isinstance(question_id, (int, str)):
                continue
            question = valid_questions.get(question_id)
            indices = answer.get("answer_indices")
            if question is None or not isinstance(indices, list) or not indices:
                continue
            if not all(
                isinstance(index, int)
                and not isinstance(index, bool)
                and 0 <= index < len(question["options"])
                for index in indices
            ):
                continue
            if len(set(indices)) != len(indices):
                continue
            if question.get("type") == "sequence_order":
                if set(indices) != set(range(len(question["options"]))):
                    continue
            elif question.get("type") in {
                "existence",
                "deceptive",
                "noteworthy",
                "pairwise_order",
                "before_after",
            } and len(indices) != 1:
                continue
            parsed_answers[question_id] = indices

    return [
        {
            "question_id": question["question_id"],
            "answer_indices": parsed_answers.get(question["question_id"]),
        }
        for question in questions
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask a VLM the benchmark questions about a video"
    )
    parser.add_argument("--video", required=True, help="Path to the source .mp4 video")
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
        help="Frame mode only: maximum uniformly sampled frames (default: 100)",
    )
    parser.add_argument(
        "--max-image-size",
        type=int,
        default=768,
        help="Frame mode only: max image width/height sent to model (default: 768)",
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

    video_path = Path(args.video).expanduser()
    if not video_path.is_file():
        raise SystemExit(f"Video file not found: {video_path}")

    input_mode = args.input_mode or config.get("input_mode", "frames")
    if input_mode == "frames":
        if args.max_frames < 1:
            raise SystemExit("--max-frames must be at least 1")
        if args.max_image_size < 1:
            raise SystemExit("--max-image-size must be at least 1")

    output_path = (
        Path(args.output).expanduser()
        if args.output
        else Path(f"{video_path.stem}_qa_answers.json")
    )

    video_parser = VideoAIParser(config, input_mode=input_mode)
    if video_parser.input_mode == "frames":
        input_summary = f"up to {args.max_frames} uniformly sampled frames"
    else:
        input_summary = "native MP4 input"
    print(
        f"Asking {len(questions)} questions about {video_path} using {input_summary}",
        flush=True,
    )
    raw_response, input_metadata = video_parser.call_video(
        video_path,
        prompt,
        max_frames=args.max_frames,
        max_image_size=args.max_image_size,
    )

    result = {
        "model": config["model"],
        "backend": config.get("backend"),
        "video": str(video_path.resolve()),
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
