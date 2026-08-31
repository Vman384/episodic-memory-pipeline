"""Load pipeline settings from JSON configuration files."""

import json
from pathlib import Path
from typing import NotRequired, Optional, TypedDict

class PipelineConfig(TypedDict):
    """Configuration required by the counting pipeline."""

    frames_dir: str
    sampled_frames_dir: str
    sampled_indices: list[int]
    model: str
    output: str
    backend: NotRequired[str]
    task: int
    frames_dir: str
    sections_dir: str
    output: str
    frames_per_section: int
    step: int
    model: str
    backend: str
    api_base_url: str
    api_key_env: str
    temperature: Optional[float]
    reasoning_effort: str
    max_tokens: int
    merge_window: int
    storyline_max_tokens: int
    question_max_tokens: int
    enforce_eager: bool
    dtype: str
    max_model_len: int
    gpu_memory_utilization: float

class CountingPipelineConfig(PipelineConfig):
    """Configuration required by the counting pipeline."""

    CLIPModelmodel: str
    CLIPProcessormodel: str
    sam3_model: str

class ConfigLoader:
    """Read one JSON configuration file for a pipeline entry point."""

    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path).expanduser()

    def load(self):
        """Return the configuration as a dictionary."""
        if not self.config_path.is_file():
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}"
            )

        with self.config_path.open() as config_file:
            config = json.load(config_file)

        if not isinstance(config, dict):
            raise ValueError("Configuration file must contain a JSON object")

        return config

