"""Pull staged frame datasets one at a time and evaluate them with Vertex AI."""

import argparse
import json
import os
import posixpath
import pty
import re
import shlex
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from QA.QuestionnaireParser import QuestionnaireParser
from QA.VideoAIParser import _uniform_timestamp_indices
from QA.run_questionnaire import PROMPT_FILE, build_prompt, parse_response
from grade_answers import grade


DEFAULT_PROJECT = "episodic-memory-benchmark"
DEFAULT_LOCATION = "us-central1"
DEFAULT_MODEL = "gemini-2.5-flash"
MAX_INLINE_MEDIA_BYTES = 14 * 1024 * 1024
FRAME_REDUCTION_STEP = 30
# Local-only override: replace "empty" with your password; never commit it.
ssh_pass = "UpperReal270204"
SSH_OPTIONS = (
    "-o",
    "ControlMaster=auto",
    "-o",
    "ControlPersist=1h",
    "-o",
    "ControlPath=~/.ssh/cm-%C",
)


class InlinePayloadTooLarge(RuntimeError):
    """Raised when the sampled images do not fit in one Vertex request."""


def fit_vertex_request_frames(
    all_frame_paths: list[Path],
) -> tuple[list[Path], list[int], int | None, list[dict]]:
    """Reduce only an oversized dataset's uniform frame sample, by 30 per try."""
    selected_indices = list(range(len(all_frame_paths)))
    reductions = []

    while True:
        selected_paths = [all_frame_paths[index] for index in selected_indices]
        try:
            quality = compress_for_vertex(selected_paths)
            return selected_paths, selected_indices, quality, reductions
        except InlinePayloadTooLarge:
            current_count = len(selected_indices)
            if current_count <= 1:
                raise

            next_count = max(1, current_count - FRAME_REDUCTION_STEP)
            positions = _uniform_timestamp_indices(range(current_count), next_count)
            selected_indices = [selected_indices[position] for position in positions]
            reductions.append(
                {"frames_before": current_count, "frames_after": next_count}
            )
            print(
                f"Vertex payload too large; retrying this dataset with "
                f"{next_count} frames",
                flush=True,
            )


def run_ssh_command(command: list[str]) -> None:
    if ssh_pass == "empty":
        subprocess.run(command, check=True)
        return

    prompt_tail = bytearray()

    def read_ssh_output(master_fd: int) -> bytes:
        try:
            output = os.read(master_fd, 1024)
        except OSError:
            return b""
        if b"password:" in (prompt_tail + output).lower():
            prompt_tail.clear()
            os.write(master_fd, ssh_pass.encode() + b"\n")
        else:
            prompt_tail.extend(output.lower())
            del prompt_tail[:-32]
        return output

    status = pty.spawn(
        command,
        master_read=read_ssh_output,
        stdin_read=lambda _fd: b"",
    )
    exit_code = os.waitstatus_to_exitcode(status)
    if exit_code != 0:
        raise subprocess.CalledProcessError(exit_code, command)


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
    command = ["scp", *SSH_OPTIONS]
    if recursive:
        command.append("-r")
    command.extend([f"{target}:{remote_path}", str(destination)])
    run_ssh_command(command)


def cleanup_remote_dataset(target: str, remote_path: str) -> None:
    command = f"rm -rf -- {shlex.quote(remote_path)}"
    run_ssh_command(["ssh", *SSH_OPTIONS, target, command])


def compress_for_vertex(frame_paths: list[Path]) -> int | None:
    """Reduce JPEG quality only if the inline image payload approaches 20 MB."""
    from io import BytesIO

    from PIL import Image

    total_bytes = sum(path.stat().st_size for path in frame_paths)
    if total_bytes <= MAX_INLINE_MEDIA_BYTES:
        return None

    originals = {path: path.read_bytes() for path in frame_paths}
    try:
        for quality in (75, 60, 45):
            for path, original in originals.items():
                with Image.open(BytesIO(original)) as source:
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
    except Exception:
        for path, original in originals.items():
            path.write_bytes(original)
        raise

    for path, original in originals.items():
        path.write_bytes(original)
    raise InlinePayloadTooLarge(
        "The sampled images exceed Vertex's inline request limit after JPEG "
        "compression"
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
                all_frame_paths = sorted((dataset_dir / "frames").glob("*.jpg"))
                if not all_frame_paths:
                    raise RuntimeError(f"No sampled JPEGs staged for {dataset_name}")

                frame_metadata = json.loads(
                    (dataset_dir / "frame_metadata.json").read_text()
                )
                (
                    frame_paths,
                    selected_indices,
                    compression_quality,
                    frame_reductions,
                ) = fit_vertex_request_frames(all_frame_paths)
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
            effective_frame_metadata = dict(frame_metadata)
            for field in (
                "sampled_frame_indices",
                "sampled_frame_names",
                "sampled_timestamps_seconds",
            ):
                values = frame_metadata.get(field)
                if isinstance(values, list) and len(values) == len(all_frame_paths):
                    effective_frame_metadata[field] = [
                        values[index] for index in selected_indices
                    ]
            effective_frame_metadata["frames_used"] = len(frame_paths)
            if frame_reductions:
                effective_frame_metadata["frame_budget_initial"] = frame_metadata.get(
                    "frame_budget"
                )
                effective_frame_metadata["frame_budget"] = len(frame_paths)
                effective_frame_metadata["vertex_frame_reductions"] = frame_reductions
            metadata = {
                **effective_frame_metadata,
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
        run_ssh_command(["ssh", *SSH_OPTIONS, args.ssh_target, remote_cleanup])
    except (subprocess.CalledProcessError, OSError) as error:
        print(f"Completed all datasets, but could not remove staging manifest: {error}")

    print(f"All Vertex evaluations saved under {output_root}")


if __name__ == "__main__":
    main()
