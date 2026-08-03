"""Load pipeline settings from JSON configuration files."""

import json
from pathlib import Path


class ConfigLoader:
    """Read one JSON configuration file for a pipeline entry point."""

    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path).expanduser()

    def load(self) -> dict:
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
