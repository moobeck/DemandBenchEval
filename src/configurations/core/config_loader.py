import os
import yaml
import logging
from typing import Dict, Any
from pathlib import Path


class ConfigLoader:
    """
    Object-oriented config loader that loads configuration from multiple YAML files.
    """

    def __init__(self, config_dir: str = "config"):
        """
        Initialize the ConfigLoader with the config directory.

        Args:
            config_dir: Path to the config directory (default: "config")
        """
        self.config_dir = Path(config_dir)
        self.public_dir = self.config_dir / "public"
        self.private_dir = self.config_dir / "private"

    def _load_public_config(self) -> Dict[str, Any]:
        """
        Load and merge all public config files.

        Returns:
            Dict containing merged public configuration
        """
        public_config = {}

        # Load each public config file
        config_files = {
            "system": "system.yaml",
            "filepaths": "filepaths.yaml",
            "forecast": "forecast.yaml",
            "metrics": "metrics.yaml",
            "tasks": "task.yaml",
        }

        for key, filename in config_files.items():
            file_path = self.public_dir / filename
            config_data = self._load_yaml_file(file_path)
            if config_data:
                public_config.update(config_data)
            else:
                logging.warning(f"Failed to load or empty config file: {file_path}")

        return public_config

    def _load_private_config(self) -> Dict[str, Any]:
        """
        Load the private config file.

        Returns:
            Dict containing private configuration
        """
        private_config = {}
        private_file = self.private_dir / "config.yaml"

        config_data = self._load_yaml_file(private_file)
        if config_data:
            private_config.update(config_data)
        else:
            logging.warning(
                f"Failed to load or empty private config file: {private_file}"
            )

        return private_config

    def load_all_configs(self) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Load both public and private configurations.

        Returns:
            Tuple of (public_config, private_config)
        """
        public_config = self._load_public_config()
        private_config = self._load_private_config()

        return public_config, private_config

    def _load_yaml_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Load a single YAML file.

        Args:
            file_path: Path to the YAML file

        Returns:
            Dict containing the loaded configuration, or empty dict if failed
        """
        try:
            if not file_path.exists():
                logging.warning(f"Config file not found: {file_path}")
                return {}

            with open(file_path, "r") as f:
                data = yaml.safe_load(f)
                return data if data is not None else {}

        except Exception as e:
            logging.error(f"Error loading config file {file_path}: {e}")
            return {}
