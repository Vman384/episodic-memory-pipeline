"""Sparse event localisation pipeline."""

import json
from pathlib import Path

from pipeline.AIParser import AIParser
from pipeline.ConfigLoader import ConfigLoader
from pipeline.frame_parser import FrameParser

# Each task maps to the prompt file the VLM should use.
TASK_PROMPTS = {
    1: "prompts/sparse_event_prompt.txt",
    2: "prompts/temporal.txt",
}


def _resolve_prompt_file(task: int, prompts_root: Path) -> Path:
    """
    Return the absolute path to the prompt file for *task*.
    """
    relative = TASK_PROMPTS.get(task)
    if relative is None:
        raise SystemExit(
            f"Unknown task {task}. Supported tasks: {sorted(TASK_PROMPTS)}"
        )

    prompt_path = prompts_root / relative

    if not prompt_path.is_file():
        raise SystemExit(f"Prompt file not found: {prompt_path}")
    return prompt_path


class SparseEventPipeline:
    """Coordinate configuration, frame parsing, and sparse-event VLM calls."""

    def __init__(self, config_path: str | Path):
        self.config_path = config_path
        self.config = None
        self.frame_parser = None
        self.ai_parser = None

    def load_config(self) -> dict:
        """Load the config and create the pipeline's helper classes."""
        self.config = ConfigLoader(self.config_path).load()

        # Initialise Frame parser from config
        self.frame_parser = FrameParser(
            frames_dir=self.config["frames_dir"],
            output_dir=self.config["sections_dir"],
            frames_per_section=self.config["frames_per_section"],
            step_size=self.config["step"],
            move=self.config.get("move", False),
        )

        # Initialise AI parser from config
        self.ai_parser = AIParser(
            model=self.config["model"],
            temperature=self.config.get("temperature", 0.2),
            max_tokens=self.config.get("max_tokens", 100),
            enforce_eager=self.config.get("enforce_eager", True),
            dtype=self.config.get("dtype", "half"),
            max_model_len=self.config.get("max_model_len", 4096),
            gpu_memory_utilization=self.config.get(
                "gpu_memory_utilization", 0.9
            ),
        )
        return self.config

    def run(self) -> None:
        """Run the sparse-event pipeline and write its results."""
        config = self.load_config()

        print(f"[1/2] Splitting frames ({config['frames_dir']}) into sections ...")

        # Create each section directories for segregated frames
        sections = self.frame_parser.create_section_dir()

        print(f"  Created {len(sections)} sections in {config['sections_dir']}")

        prompts_root = Path(__file__).resolve().parent
        prompt_path = _resolve_prompt_file(config["task"], prompts_root)
        prompt = prompt_path.read_text()

        print(f"[2/2] Querying VLM (task={config['task']}, "f"model={config['model']}) ...")

        output_dir = Path(config["output"])
        output_dir.mkdir(parents=True, exist_ok=True)
        all_results = []

        for curr_section in sections:

            # call vlm
            response = self.ai_parser.call_vlm(prompt, curr_section)

            # get the result and response
            section_result = {
                "section": curr_section.name,
                "response": response,
            }
            section_output_dir = output_dir / f"{curr_section.name}_output"
            section_output_dir.mkdir(parents=True, exist_ok=True)
            with open(section_output_dir / "result.json", "w") as result_file:
                json.dump(section_result, result_file, indent=2)
            all_results.append(section_result)

        with open(output_dir / "all_results.json", "w") as result_file:
            json.dump(all_results, result_file, indent=2)

        print(f"  Finished {len(all_results)} sections, results in {config['output']}")
