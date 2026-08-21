"""Sparse event localisation pipeline."""

import json
from pathlib import Path

from pipeline.AIParser import AIParser
from pipeline.ConfigLoader import ConfigLoader
from pipeline.frame_parser import FrameParser

class SparseEventPipeline:
    """Coordinate configuration, frame parsing, and sparse-event VLM calls."""

    def __init__(self, config_path: str | Path, prompt: str):
        self.config_path = config_path
        self.prompt = prompt
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
        self.ai_parser = AIParser(config=self.config)
        return self.config

    def run(self) -> None:
        """Run the sparse-event pipeline and write its results."""
        config = self.load_config()

        print(f"[1/2] Splitting frames ({config['frames_dir']}) into sections ...")

        sections_dir = Path(config["sections_dir"])
        sections = sorted(
            section for section in sections_dir.glob("section_*") if section.is_dir()
        )
        if sections:
            print(
                f"  Reusing {len(sections)} existing sections in {config['sections_dir']}"
            )
        else:
            # Create each section directory for segregated frames.
            sections = self.frame_parser.create_section_dir()
            print(f"  Created {len(sections)} sections in {config['sections_dir']}")

        print(
            f"[2/2] Querying VLM (backend={config.get('backend', 'local')}, "
            f"model={config['model']}) ..."
        )

        output_dir = Path(config["output"])
        output_dir.mkdir(parents=True, exist_ok=True)
        all_results = []

        for curr_section in sections:

            # call vlm
            response = self.ai_parser.call_vlm(self.prompt, curr_section)

            # get the result and response
            section_result = {
                "section": curr_section.name,
                "response": response,
            }

            # Write each output to a directory
            section_output_dir = output_dir / f"{curr_section.name}_output"
            section_output_dir.mkdir(parents=True, exist_ok=True)
            with open(section_output_dir / "result.json", "w") as result_file:
                json.dump(section_result, result_file, indent=2)
            all_results.append(section_result)

        with open(output_dir / "all_results.json", "w") as result_file:
            json.dump(all_results, result_file, indent=2)

        print(f"  Finished {len(all_results)} sections, results in {config['output']}")
