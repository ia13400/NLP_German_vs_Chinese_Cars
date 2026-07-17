from __future__ import annotations

from .config import load_project_config
from .data.dataset_builder import build_analysis_dataset


def main() -> int:
    """Build the real KBA registration-share series and print a short summary."""
    config = load_project_config()
    dataset = build_analysis_dataset()
    print(config["project"]["title"])
    print(f"Loaded registration-share rows: {len(dataset)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
