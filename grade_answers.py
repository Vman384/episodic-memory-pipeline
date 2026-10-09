"""Grade model answers against a question JSON file.

Usage:
    python grade_answers.py --questions <questions.json> --answers <answers.json> --output <graded.json>
"""

import argparse
import json
from pathlib import Path


def option_texts(options: list[str], indices: list[int] | None) -> list[str | None] | None:
    """Return the option text for each index, or None when an index is invalid."""
    if indices is None:
        return None
    return [
        options[index] if isinstance(index, int) and 0 <= index < len(options) else None
        for index in indices
    ]


def grade(questions: list[dict], answers: list[dict]) -> dict:
    """Compare each question's selected answer with its correct answer."""
    answers_by_id = {answer["question_id"]: answer for answer in answers}

    results = []
    for question in questions:
        question_id = question["question_id"]
        answer = answers_by_id.get(question_id, {})
        selected = answer.get("answer_indices")
        correct = question["answer_indices"]
        options = question["options"]

        results.append(
            {
                "question_id": question_id,
                "type": question.get("type"),
                "question": question["question"],
                "correct": selected is not None and selected == correct,
                "selected_indices": selected,
                "selected_answers": option_texts(options, selected),
                "correct_indices": correct,
                "correct_answers": option_texts(options, correct),
            }
        )

    correct_count = sum(result["correct"] for result in results)
    total = len(results)
    return {
        "score": {
            "correct": correct_count,
            "total": total,
            "accuracy": round(correct_count / total, 4) if total else 0.0,
        },
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Grade model answers to benchmark questions")
    parser.add_argument("--questions", required=True, help="Question JSON file (questions.json)")
    parser.add_argument("--answers", required=True, help="Answers JSON from run_model_answers.py")
    parser.add_argument(
        "--output",
        default="graded_results.json",
        help="Path to write the graded results JSON",
    )
    args = parser.parse_args()

    questions = json.loads(Path(args.questions).read_text()).get("questions", [])
    answers = json.loads(Path(args.answers).read_text()).get("answers", [])
    if not questions:
        raise SystemExit(f"No questions found in: {args.questions}")

    graded = grade(questions, answers)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(graded, indent=4))

    score = graded["score"]
    print(
        f"Score: {score['correct']}/{score['total']} "
        f"({score['accuracy'] * 100:.2f}%)"
    )
    print(f"Graded results written to {output_path}")


if __name__ == "__main__":
    main()
