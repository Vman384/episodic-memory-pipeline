"""Single entry point — run the full pipeline from one JSON config file."""

import argparse
import asyncio
import json
from pathlib import Path

from AIParser import AIEventParser
from frame_parser import FrameParser

# Each task maps to the prompt file the VLM should use.
TASK_PROMPTS = {
    1: "prompts/sparse_event_prompt.txt",
    2: "prompts/temporal.txt",
}


def _resolve_prompt_file(task: int, prompts_root: Path) -> Path:
    """Return the absolute path to the prompt file for *task*."""
    relative = TASK_PROMPTS.get(task)
    if relative is None:
        raise SystemExit(
            f"Unknown task {task}. Supported tasks: {sorted(TASK_PROMPTS)}"
        )
    prompt_path = prompts_root / relative
    if not prompt_path.is_file():
        raise SystemExit(f"Prompt file not found: {prompt_path}")
    return prompt_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full episodic memory pipeline."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a JSON config file",
    )
    args = parser.parse_args()

    # Everything comes from the config file; no CLI overrides.
    with open(args.config) as fh:
        config = json.load(fh)

    task = config["task"]
    frames_dir = config["frames_dir"]
    sections_dir = config.get("sections_dir", "./sections")
    output = config["output"]
    frames_per_section = config.get("frames_per_section", 100)
    step = config.get("step", 1)
    model = config.get("model", "qwen3.7-plus")
    max_concurrent = config.get("max_concurrent", 3)
    base_url = config.get("base_url", "https://opencode.ai/zen/go/v1")

    # ----- Stage 1: split frames into sections -----
    print(f"[1/2] Splitting frames ({frames_dir}) into sections ...")
    section_dirs = FrameParser(
        frames_dir=frames_dir,
        output_dir=sections_dir,
        frames_per_section=frames_per_section,
        step_size=step,
    ).create_section_dir()
    print(f"  Created {len(section_dirs)} sections in {sections_dir}")

    # ----- Stage 2: run VLM over every section -----
    prompts_root = Path(__file__).resolve().parent
    prompt_path = _resolve_prompt_file(task, prompts_root)
    prompt = prompt_path.read_text()

    print(f"[2/2] Querying VLM (task={task}, model={model}) ...")
    ai_parser = AIEventParser(
        prompt=prompt,
        model=model,
        base_url=base_url,
        max_concurrent=max_concurrent,
    )
    result_paths = asyncio.run(
        ai_parser.process_subsections(sections_dir, output)
    )
    print(f"  Finished {len(result_paths)} sections, results in {output}")


if __name__ == "__main__":
    main()
