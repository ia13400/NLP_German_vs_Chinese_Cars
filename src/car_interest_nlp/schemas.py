from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SourceConfig(BaseModel):
    """Input source configuration for collection and caching.

    Every source in configs/sources.yaml is validated through this model. Extra keys
    (e.g. `listing_url`, `use_structured_downloads`) are source-specific and simply
    passed through rather than enumerated here.
    """

    model_config = ConfigDict(extra="allow")

    enabled: bool = True
    source_type: str = "registration"
    access_method: str | None = None
    base_url: str | None = None
    raw_directory: str | None = None
    path: str | None = None
    url: str | None = None
    rate_limit: float = 0.0
    rate_limit_seconds: float = 0.0
    cache_policy: str = "reuse"
    force_refresh: bool = False
    cache_duration_hours: float = 24.0
    user_agent: str = "car-interest-nlp-research-bot/1.0"
    start_date: str | None = None
    end_date: str | None = None
    validation_rules: dict[str, Any] = Field(default_factory=dict)
