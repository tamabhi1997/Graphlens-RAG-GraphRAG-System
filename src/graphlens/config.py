"""Configuration helpers."""

from pathlib import Path

import yaml


def load_config(path: str = "config/app.yaml") -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)
