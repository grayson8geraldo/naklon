"""Configuration loader for Naklon strategy."""

import os
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | None = None) -> dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config file. Defaults to config.yaml in project root.

    Returns:
        Configuration dictionary.
    """
    if config_path is None:
        config_path = str(Path(__file__).parent.parent.parent / "config.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Override with environment variables if present
    if os.getenv("NAKLON_EXCHANGE"):
        config["exchange"]["name"] = os.getenv("NAKLON_EXCHANGE")
    if os.getenv("NAKLON_CAPITAL"):
        config["capital"]["initial"] = float(os.getenv("NAKLON_CAPITAL"))
    if os.getenv("NAKLON_TESTNET"):
        config["exchange"]["testnet"] = os.getenv("NAKLON_TESTNET").lower() == "true"

    return config
