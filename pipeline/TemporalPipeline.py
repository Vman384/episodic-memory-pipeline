import json
from pathlib import Path

from pipeline.AIParser import AIParser
from pipeline.ConfigLoader import ConfigLoader
from pipeline.frame_parser import FrameParser



class TemporalPipeline:
    """
    Pipeline to asssess temporal events
    """

    def __init__(self, config_path: str | Path, prompt: str):
        self.config_path = config_path
        self.prompt = prompt
        self.config = None
        self.frame_parser = None
        self.ai_parser = None

    def load_config(self) -> dict:
        """
        Load config from ConfigLoader and create instances of 
        helper classes
        """
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
        """
        Run Temporal pipeline and write it's result to file
        """

        # load the configs
        config = self.load_config()

        print(f"[1/2] Splitting frames ({config['frames_dir']}) into sections ...")

        # split directories
        sections = self.frame_parser.create_section_dir()

        print(f"  Created {len(sections)} sections in {config['sections_dir']}")

        # Make directory for output
        output_dir = Path(config["output"])
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"[2/2] Querying VLM (model={config['model']}) ...")

        all_results = []

        # For every single subsections, we parse it into vlm to get a response
        for curr_section in sections:
            response = self.ai_parser.call_vlm(self.prompt, curr_section)
            section_result = {
                "section": curr_section.name,
                "response": response,
            }

            # Make directory for current section output
            section_output_dir = output_dir / f"{curr_section.name}_output"
            section_output_dir.mkdir(parents=True, exist_ok=True)

            # Store the result into a json
            with open(section_output_dir / "result.json", "w") as result_file:
                json.dump(section_result, result_file, indent=2)
            
            all_results.append(section_result)

        # Stores into our global result
        with open(output_dir / "all_results.json", "w") as result_file:
            json.dump(all_results, result_file, indent=2)

        print(f"  Saved {len(all_results)} section responses to {config['output']}")
