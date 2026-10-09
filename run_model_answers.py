"""Ask a VLM every question in a question JSON file and save its answers.

Usage:
    python run_model_answers.py --frames_dir <camera folder> --questions <questions.json> --output <answers.json>
"""

import argparse
import json
from pathlib import Path

from pipeline.AIParser import AIParser
from pipeline.ConfigLoader import ConfigLoader
from pipeline.timeframe_converter import parse_frame_timestamp

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
PROMPT_FILE = Path("pipeline/prompts/benchmark_answer.txt")


def sample_frames(frames_dir: Path, max_frames: int) -> list[Path]:
    """Return up to max_frames frames spread evenly over the whole drive."""
    frames = [
        path
        for path in frames_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    frames.sort(key=lambda path: parse_frame_timestamp(path.name))
    if not frames:
        raise ValueError(f"No image frames found in folder: {frames_dir}")

    if len(frames) <= max_frames:
        return frames
    last = len(frames) - 1
    indices = sorted({round(i * last / (max_frames - 1)) for i in range(max_frames)})
    return [frames[index] for index in indices]


def build_question_prompt(prompt_template: str, question: dict) -> str:
    """Append one question and its options to the shared answer prompt."""
    option_lines = "\n".join(
        f"{index}: {option}" for index, option in enumerate(question["options"])
    )
    return (
        f"{prompt_template}\n\n"
        f"Question: {question['question']}\n"
        f"Options:\n{option_lines}"
    )


def parse_answer_indices(response: str) -> list[int] | None:
    """Return the answer_indices list from a model response, or None if invalid."""
    start = response.find("{")
    end = response.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        answer = json.loads(response[start : end + 1])
    except json.JSONDecodeError:
        return None

    indices = answer.get("answer_indices") if isinstance(answer, dict) else None
    if not isinstance(indices, list) or not all(
        isinstance(index, int) and not isinstance(index, bool) for index in indices
    ):
        return None
    return indices


def main():
    parser = argparse.ArgumentParser(description="Get model answers to benchmark questions")
    parser.add_argument("--frames_dir", required=True, help="Camera frames folder for one drive")
    parser.add_argument("--questions", required=True, help="Question JSON file (questions.json)")
    parser.add_argument("--output", required=True, help="Path to write the answers JSON")
    parser.add_argument(
        "--config",
        default="configs/answer_eval.json",
        help="Model configuration JSON (backend, model, API settings)",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=100,
        help="Maximum frames sent per question, sampled evenly across the drive",
    )
    args = parser.parse_args()

    if not PROMPT_FILE.is_file():
        raise SystemExit(f"Prompt file not found: {PROMPT_FILE}")
    if args.max_frames < 2:
        raise SystemExit("--max_frames must be at least 2")

    frames_dir = Path(args.frames_dir)
    if not frames_dir.is_dir():
        raise SystemExit(f"Frames folder not found: {frames_dir}")

    questions_data = json.loads(Path(args.questions).read_text())
    questions = questions_data.get("questions", [])
    if not questions:
        raise SystemExit(f"No questions found in: {args.questions}")

    config = ConfigLoader(args.config).load()
    ai_parser = AIParser(config)
    prompt_template = PROMPT_FILE.read_text()
    frames = sample_frames(frames_dir, args.max_frames)
    print(f"Using {len(frames)} frames for {len(questions)} questions", flush=True)

    answers = []
    for question in questions:
        question_id = question["question_id"]
        prompt = build_question_prompt(prompt_template, question)
        response = ai_parser.call_vlm_images(prompt, frames)
        answer_indices = parse_answer_indices(response)
        if answer_indices is None:
            print(f"Question {question_id}: could not parse a valid answer", flush=True)
        else:
            print(f"Question {question_id}: answered {answer_indices}", flush=True)

        answers.append(
            {
                "question_id": question_id,
                "type": question.get("type"),
                "answer_indices": answer_indices,
                "raw_response": response,
            }
        )

    output = {
        "model": config["model"],
        "frames_dir": str(frames_dir),
        "frames_used": len(frames),
        "answers": answers,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=4))
    print(f"Answers written to {output_path}", flush=True)


if __name__ == "__main__":
    main()
