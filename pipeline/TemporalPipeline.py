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
            tensor_parallel_size=self.config.get("tensor_parallel_size", 1),
        )
        return self.config

    def run(self) -> None:
        """
        Run Temporal pipeline and write it's result to file
        """

        config = self.load_config()

        print(f"[1/3] Splitting frames ({config['frames_dir']}) into sections ...")

        sections = self.frame_parser.create_section_dir()

        print(f"  Created {len(sections)} sections in {config['sections_dir']}")
