"""Ask a VLM every question in a question JSON file and save its answers.

Usage:
    python run_model_answers.py --frames_dir <camera folder> --questions <questions.json> --output <answers.json>

With --batch_size (API backend only), every frame of the drive is sent in
order in batches of that size, all in one stored conversation. The questions
are asked after the last batch, so the model answers from what it has seen.
"""

import argparse
import json
from pathlib import Path

from pipeline.AIParser import AIParser
from pipeline.ConfigLoader import ConfigLoader
from pipeline.timeframe_converter import parse_frame_timestamp

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
PROMPT_FILE = Path("pipeline/prompts/benchmark_answer.txt")


def list_frames(frames_dir: Path) -> list[Path]:
    """Return every frame in the folder, sorted by parsed timestamp."""
    frames = [
        path
        for path in frames_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    frames.sort(key=lambda path: parse_frame_timestamp(path.name))
    if not frames:
        raise ValueError(f"No image frames found in folder: {frames_dir}")
    return frames


def sample_frames(frames_dir: Path, max_frames: int) -> list[Path]:
    """Return up to max_frames frames spread evenly over the whole drive."""
    frames = list_frames(frames_dir)
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


def send_frame_batches(ai_parser: AIParser, frames: list[Path], batch_size: int):
    """
    Send frames in order, batch by batch, in one stored API conversation.

    Stops at the first batch the API rejects, for example when the context is
    full. Returns the last accepted response ID, the number of frames the model
    received, and a log entry for each batch.
    """
    previous_response_id = None
    frames_received = 0
    batches = []
    total = len(frames)

    for start in range(0, total, batch_size):
        batch = frames[start : start + batch_size]
        first_number = start + 1
        last_number = start + len(batch)
        manifest = "\n".join(
            f"Frame {number}: {path.name}"
            for number, path in enumerate(batch, start=first_number)
        )
        text = (
            f"Frames {first_number} to {last_number} of {total} from the drive, "
            "in chronological order. Remember them for later questions. "
            "Reply with OK only.\n"
            f"{manifest}"
        )

        try:
            response = ai_parser.call_api_conversation(
                ai_parser.build_api_message(text, batch),
                previous_response_id=previous_response_id,
            )
        except RuntimeError as error:
            print(f"Stopped before frame {first_number}: {error}", flush=True)
            batches.append(
                {
                    "frames_from": first_number,
                    "frames_to": last_number,
                    "status": "failed",
                    "error": str(error),
                }
            )
            break

        previous_response_id = response.id
        frames_received = last_number
        input_tokens = response.usage.input_tokens
        print(
            f"Frames {first_number}-{last_number} accepted; "
            f"input tokens so far: {input_tokens}",
            flush=True,
        )
        batches.append(
            {
                "frames_from": first_number,
                "frames_to": last_number,
                "status": "ok",
                "input_tokens": input_tokens,
            }
        )

    return previous_response_id, frames_received, batches


def answer_in_conversation(
    ai_parser: AIParser,
    questions: list[dict],
    prompt_template: str,
    previous_response_id: str | None,
    frames_received: int,
) -> list[dict]:
    """Ask each question from the end of the frame conversation.

    Each question is a separate response linked to the last frame batch, so
    one answer does not change the context of the next question.
    """
    answers = []
    for question in questions:
        question_id = question["question_id"]
        answer = {
            "question_id": question_id,
            "type": question.get("type"),
            "answer_indices": None,
            "raw_response": None,
            "frames_received": frames_received,
        }

        if previous_response_id is None:
            answer["error"] = "No frames were accepted by the API"
            print(f"Question {question_id}: skipped, no frames accepted", flush=True)
            answers.append(answer)
            continue

        prompt = build_question_prompt(prompt_template, question)
        try:
            response = ai_parser.call_api_conversation(
                prompt,
                previous_response_id=previous_response_id,
            )
        except RuntimeError as error:
            answer["error"] = str(error)
            print(f"Question {question_id}: request failed: {error}", flush=True)
            answers.append(answer)
            continue

        answer["raw_response"] = response.output_text
        answer["input_tokens"] = response.usage.input_tokens
        answer["answer_indices"] = parse_answer_indices(response.output_text)
        if answer["answer_indices"] is None:
            print(f"Question {question_id}: could not parse a valid answer", flush=True)
        else:
            print(f"Question {question_id}: answered {answer['answer_indices']}", flush=True)
        answers.append(answer)

    return answers


def answer_independently(
    ai_parser: AIParser,
    questions: list[dict],
    prompt_template: str,
    frames: list[Path],
) -> list[dict]:
    """Ask each question with the same sampled frames in a separate request."""
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
    return answers


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
        help=(
            "Maximum frames sent per question, sampled evenly across the drive. "
            "Ignored when --batch_size is set, which sends every frame."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help=(
            "API backend only. Send every frame in batches of this size in one "
            "stored conversation, then ask the questions."
        ),
    )
    args = parser.parse_args()

    if not PROMPT_FILE.is_file():
        raise SystemExit(f"Prompt file not found: {PROMPT_FILE}")
    if args.max_frames < 2:
        raise SystemExit("--max_frames must be at least 2")
    if args.batch_size is not None and args.batch_size < 1:
        raise SystemExit("--batch_size must be at least 1")

    frames_dir = Path(args.frames_dir)
    if not frames_dir.is_dir():
        raise SystemExit(f"Frames folder not found: {frames_dir}")

    questions_data = json.loads(Path(args.questions).read_text())
    questions = questions_data.get("questions", [])
    if not questions:
        raise SystemExit(f"No questions found in: {args.questions}")

    config = ConfigLoader(args.config).load()
    if args.batch_size is not None and config.get("backend", "local").lower() != "api":
        raise SystemExit("--batch_size requires backend \"api\" in the config")

    ai_parser = AIParser(config)
    prompt_template = PROMPT_FILE.read_text()

    output = {
        "model": config["model"],
        "frames_dir": str(frames_dir),
    }

    if args.batch_size is None:
        frames = sample_frames(frames_dir, args.max_frames)
        print(f"Using {len(frames)} frames for {len(questions)} questions", flush=True)
        answers = answer_independently(ai_parser, questions, prompt_template, frames)
        output["frames_used"] = len(frames)
    else:
        frames = list_frames(frames_dir)
        print(
            f"Sending {len(frames)} frames in batches of {args.batch_size}",
            flush=True,
        )
        previous_response_id, frames_received, batches = send_frame_batches(
            ai_parser, frames, args.batch_size
        )
        print(
            f"Model received {frames_received} of {len(frames)} frames; "
            f"asking {len(questions)} questions",
            flush=True,
        )
        answers = answer_in_conversation(
            ai_parser,
            questions,
            prompt_template,
            previous_response_id,
            frames_received,
        )
        output["mode"] = "conversation"
        output["batch_size"] = args.batch_size
        output["frames_total"] = len(frames)
        output["frames_used"] = frames_received
        output["batches"] = batches

    output["answers"] = answers
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=4))
    print(f"Answers written to {output_path}", flush=True)


if __name__ == "__main__":
    main()
