"""Pull staged frame datasets one at a time and evaluate them with Vertex AI."""

import argparse
import json
import posixpath
import re
import shlex
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from QA.QuestionnaireParser import QuestionnaireParser
from QA.run_questionnaire import PROMPT_FILE, build_prompt, parse_response
from grade_answers import grade


DEFAULT_PROJECT = "episodic-memory-benchmark"
DEFAULT_LOCATION = "us-central1"
DEFAULT_MODEL = "gemini-2.5-flash"
MAX_INLINE_MEDIA_BYTES = 14 * 1024 * 1024


def is_complete(output_dir: Path) -> bool:
    answers = output_dir / "question_answer.json"
    results = output_dir / "graded_results.json"
    mark = output_dir / "final_mark.txt"
    return (
        answers.is_file()
        and answers.stat().st_size > 0
        and results.is_file()
        and results.stat().st_size > 0
        and mark.is_file()
        and mark.stat().st_size > 0
        and mark.stat().st_mtime > answers.stat().st_mtime
        and mark.stat().st_mtime > results.stat().st_mtime
    )


def scp_from_remote(target: str, remote_path: str, destination: Path, recursive=False):
    command = ["scp"]
    if recursive:
        command.append("-r")
    command.extend([f"{target}:{remote_path}", str(destination)])
    subprocess.run(command, check=True)


def cleanup_remote_dataset(target: str, remote_path: str) -> None:
    command = f"rm -rf -- {shlex.quote(remote_path)}"
    subprocess.run(["ssh", target, command], check=True)


def compress_for_vertex(frame_paths: list[Path]) -> int | None:
    """Reduce JPEG quality only if the inline image payload approaches 20 MB."""
    from io import BytesIO

    from PIL import Image

    total_bytes = sum(path.stat().st_size for path in frame_paths)
    if total_bytes <= MAX_INLINE_MEDIA_BYTES:
        return None

    for quality in (75, 60, 45):
        for path in frame_paths:
            with Image.open(path) as source:
                image = source.convert("RGB")
                buffer = BytesIO()
                image.save(buffer, format="JPEG", quality=quality, optimize=True)
            path.write_bytes(buffer.getvalue())

        total_bytes = sum(path.stat().st_size for path in frame_paths)
        if total_bytes <= MAX_INLINE_MEDIA_BYTES:
            print(
                f"Compressed sampled JPEGs to quality {quality} "
                f"({total_bytes / (1024 * 1024):.1f} MiB total)",
                flush=True,
            )
            return quality

    raise RuntimeError(
        "The sampled images exceed Vertex's inline request limit after JPEG "
        "compression; rerun frame preparation with a lower MAX_FRAMES or "
        "MAX_IMAGE_SIZE"
    )


def call_vertex(
    client,
    types,
    model: str,
    prompt: str,
    frame_paths: list[Path],
    max_output_tokens: int,
) -> str:
    image_parts = [
        types.Part.from_bytes(data=path.read_bytes(), mime_type="image/jpeg")
        for path in frame_paths
    ]
    content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=prompt), *image_parts],
    )
    response = client.models.generate_content(
        model=model,
        contents=[content],
        config=types.GenerateContentConfig(max_output_tokens=max_output_tokens),
    )
    return response.text or ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ssh-target",
        required=True,
        help="SSH config alias or user@host for Gadi",
    )
    parser.add_argument(
        "--remote-stage",
        required=True,
        help="Absolute staging directory printed by the PBS job",
    )
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--location", default=DEFAULT_LOCATION)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-output-tokens", type=int, default=4000)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Local directory for answers and graded results",
    )
    args = parser.parse_args()

    remote_stage = args.remote_stage.rstrip("/")
    if not remote_stage.startswith("/") or "gemini_vertex_staging" not in remote_stage.split("/"):
        parser.error("--remote-stage must be an absolute gemini_vertex_staging job directory")
    if args.max_output_tokens < 1:
        parser.error("--max-output-tokens must be at least 1")

    try:
        from google import genai
        from google.genai import types
    except ImportError as error:
        raise SystemExit(
            "Install google-genai in the local Python environment before running "
            "the Vertex worker"
        ) from error

    client = genai.Client(
        vertexai=True,
        project=args.project,
        location=args.location,
    )
    output_root = Path(
        args.output_dir or f"model_answers/{args.model}-vertex"
    ).expanduser()
    failures = []

    with TemporaryDirectory(prefix="vertex_frame_eval_manifest_") as manifest_temp:
        manifest_temp = Path(manifest_temp)
        scp_from_remote(
            args.ssh_target,
            posixpath.join(remote_stage, "manifest.json"),
            manifest_temp,
        )
        manifest = json.loads((manifest_temp / "manifest.json").read_text())

    for dataset_name in manifest["datasets"]:
        if not re.fullmatch(r"[A-Za-z0-9._-]+", dataset_name):
            raise SystemExit(f"Unsafe dataset name in staging manifest: {dataset_name!r}")

        output_dir = output_root / dataset_name / manifest["benchmark"]
        if is_complete(output_dir):
            print(f"Already complete; skipping {dataset_name}", flush=True)
            try:
                cleanup_remote_dataset(
                    args.ssh_target,
                    posixpath.join(remote_stage, dataset_name),
                )
            except (subprocess.CalledProcessError, OSError) as error:
                print(f"Could not remove staged {dataset_name}: {error}", file=sys.stderr)
            continue

        print(f"Pulling staged frames for {dataset_name}", flush=True)
        try:
            with TemporaryDirectory(prefix=f"vertex_frames_{dataset_name}_") as temp_dir:
                temp_dir = Path(temp_dir)
                scp_from_remote(
                    args.ssh_target,
                    posixpath.join(remote_stage, dataset_name),
                    temp_dir,
                    recursive=True,
                )
                dataset_dir = temp_dir / dataset_name
                frame_paths = sorted((dataset_dir / "frames").glob("*.jpg"))
                if not frame_paths:
                    raise RuntimeError(f"No sampled JPEGs staged for {dataset_name}")

                frame_metadata = json.loads(
                    (dataset_dir / "frame_metadata.json").read_text()
                )
                compression_quality = compress_for_vertex(frame_paths)
                question_path = dataset_dir / "questions.json"
                source_questions = json.loads(question_path.read_text())
                questionnaire = QuestionnaireParser(seed=manifest["seed"]).load(
                    question_path
                )
                prompt = (
                    "These sampled frames are from the video in chronological "
                    "order. Consider them together as one sequence.\n\n"
                    + build_prompt(PROMPT_FILE.read_text(), questionnaire)
                )
                response_text = call_vertex(
                    client,
                    types,
                    args.model,
                    prompt,
                    frame_paths,
                    args.max_output_tokens,
                )

            # The temporary directory (including every transferred image) has
            # been deleted before grading or moving to the next dataset.
            answers = parse_response(response_text, questionnaire["questions"])
            metadata = {
                **frame_metadata,
                "provider": "vertex_ai",
                "api_protocol": "vertex_generate_content",
                "frames_sampled": len(frame_paths),
                "frames_received": len(frame_paths),
                "frames_used": len(frame_paths),
            }
            if compression_quality is not None:
                metadata["vertex_jpeg_quality"] = compression_quality
            result = {
                "model": args.model,
                "backend": "vertex_ai",
                "input_mode": "frames",
                "input_source": posixpath.join(remote_stage, dataset_name, "frames"),
                "questions_file": posixpath.join(remote_stage, dataset_name, "questions.json"),
                "question_order": [
                    question["question_id"] for question in questionnaire["questions"]
                ],
                "seed": manifest["seed"],
                "input_metadata": metadata,
                "answers": answers,
                "raw_response": response_text,
            }

            output_dir.mkdir(parents=True, exist_ok=True)
            answers_path = output_dir / "question_answer.json"
            graded_path = output_dir / "graded_results.json"
            mark_path = output_dir / "final_mark.txt"
            answers_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
            graded = grade(source_questions["questions"], answers)
            graded_path.write_text(json.dumps(graded, indent=2, ensure_ascii=False) + "\n")
            score = graded["score"]
            mark = f"{score['correct']}/{score['total']} ({score['accuracy'] * 100:.2f}%)"
            mark_path.write_text(mark + "\n")
            print(f"{dataset_name}: {mark}", flush=True)

            try:
                cleanup_remote_dataset(
                    args.ssh_target,
                    posixpath.join(remote_stage, dataset_name),
                )
            except (subprocess.CalledProcessError, OSError) as error:
                print(f"Could not remove staged {dataset_name}: {error}", file=sys.stderr)
        except Exception as error:
            failures.append(dataset_name)
            print(f"Vertex evaluation failed for {dataset_name}: {error}", file=sys.stderr)

    if failures:
        print(f"Datasets with failures: {', '.join(failures)}", file=sys.stderr)
        raise SystemExit(1)

    try:
        manifest_path = posixpath.join(remote_stage, "manifest.json")
        remote_cleanup = (
            f"rm -f -- {shlex.quote(manifest_path)} && "
            f"rmdir -- {shlex.quote(remote_stage)}"
        )
        subprocess.run(["ssh", args.ssh_target, remote_cleanup], check=True)
    except (subprocess.CalledProcessError, OSError) as error:
        print(f"Completed all datasets, but could not remove staging manifest: {error}")

    print(f"All Vertex evaluations saved under {output_root}")


if __name__ == "__main__":
    main()
