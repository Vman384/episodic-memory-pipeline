"""Load benchmark questions and prepare a blinded, shuffled version for a model."""

import json
import random
from pathlib import Path


MODEL_QUESTION_FIELDS = ("question_id", "type", "question", "options")


class QuestionnaireParser:
    """Read a generated questions.json and remove benchmark-only information."""

    def __init__(self, seed: int | None = None):
        self.random = random.Random(seed)

    def load(self, path: str | Path) -> dict:
        """Load a question file and return shuffled questions without answer data."""
        question_path = Path(path).expanduser()
        if not question_path.is_file():
            raise FileNotFoundError(f"Question file not found: {question_path}")

        try:
            document = json.loads(question_path.read_text())
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON in question file: {question_path}") from error

        if not isinstance(document, dict):
            raise ValueError("Question file must contain a JSON object")

        questions = document.get("questions")
        if not isinstance(questions, list) or not questions:
            raise ValueError("Question file must contain a non-empty 'questions' list")

        model_questions = []
        seen_ids = set()
        for number, question in enumerate(questions, start=1):
            if not isinstance(question, dict):
                raise ValueError(f"Question {number} must be a JSON object")

            question_id = question.get("question_id")
            if isinstance(question_id, bool) or not isinstance(question_id, (int, str)):
                raise ValueError(
                    f"Question {number} must have a string or integer question_id"
                )
            id_key = (type(question_id), question_id)
            if id_key in seen_ids:
                raise ValueError(f"Duplicate question_id: {question_id}")
            seen_ids.add(id_key)

            question_type = question.get("type")
            if question_type is not None and not isinstance(question_type, str):
                raise ValueError(f"Question {question_id} type must be a string")

            text = question.get("question")
            options = question.get("options")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"Question {question_id} must have non-empty question text"
                )
            if (
                not isinstance(options, list)
                or not options
                or not all(isinstance(option, str) for option in options)
            ):
                raise ValueError(
                    f"Question {question_id} must have a non-empty string options list"
                )

            model_question = {
                field: question[field]
                for field in MODEL_QUESTION_FIELDS
                if field in question
            }
            model_questions.append(model_question)

        self.random.shuffle(model_questions)
        # Only pass fields needed to answer questions. This also removes
        # answer_indices, event/frame evidence, false-event flags, and the
        # top-level false_events list used during human review.
        return {"questions": model_questions}
