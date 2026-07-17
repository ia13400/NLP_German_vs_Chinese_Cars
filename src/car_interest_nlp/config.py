from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from .schemas import SourceConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a relative path (e.g. from configs/*.yaml, or a notebook literal) against
    `PROJECT_ROOT` rather than the current working directory.

    A Jupyter kernel's cwd defaults to the notebook's own directory, not the project root,
    so a bare relative path used directly in a notebook silently writes/reads under
    `notebooks/` instead of the intended project-root location -- confirmed directly (real
    scraped article files ended up under `notebooks/data/raw/gdelt_articles/`). Already-
    absolute paths are returned unchanged.
    """
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


@lru_cache(maxsize=8)
def load_yaml_config(config_name: str) -> dict:
    """Load a YAML configuration file and cache it in memory."""
    config_path = CONFIG_DIR / f"{config_name}.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_project_config() -> dict:
    """Load the project configuration bundle."""
    project = load_yaml_config("project")
    brands = load_yaml_config("brands")
    sources = load_yaml_config("sources")
    return {
        "project": project.get("project", {}),
        "brands": brands.get("brands", {}),
        "sources": sources.get("sources", {}),
    }


def load_source_configs() -> dict[str, SourceConfig]:
    """Load and validate every entry in configs/sources.yaml."""
    raw_sources = load_yaml_config("sources").get("sources", {})
    return {name: SourceConfig(**settings) for name, settings in raw_sources.items()}
