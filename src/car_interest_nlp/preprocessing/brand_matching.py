from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class BrandAlias:
    canonical_name: str
    origin_group: str
    ambiguous: bool = False
    case_sensitive: bool = False
    pattern: str | None = None
    vehicle_models: tuple[str, ...] = ()


def build_brand_alias_map(brand_config: dict) -> dict[str, BrandAlias]:
    """Create a lookup that maps aliases to canonical brand names."""
    alias_map: dict[str, BrandAlias] = {}
    for group_name, entries in brand_config.items():
        for entry in entries:
            ambiguous = {value.casefold() for value in entry.get("ambiguous_aliases", [])}
            for alias in entry.get("aliases", []):
                alias_map[alias.casefold()] = BrandAlias(
                    canonical_name=entry["canonical_name"],
                    origin_group=entry.get("origin_group", group_name),
                    ambiguous=alias.casefold() in ambiguous,
                    case_sensitive=entry.get("case_sensitive", False),
                    pattern=entry.get("regex_patterns", {}).get(alias),
                    vehicle_models=tuple(entry.get("vehicle_models", [])),
                )
    return alias_map


# Real, legitimate brands that are neither German nor Chinese (Toyota, Ford, Hyundai, ...)
# are genuinely present in every one of these datasets and must be counted somewhere --
# otherwise market-share percentages would be computed against only the tracked brands'
# total instead of the true total of all registered/sold cars, making German/Chinese shares
# look far larger than they really are. By default, any raw brand string that doesn't match
# a known German/Chinese/explicit-"other" alias is bucketed into this same canonical entry
# (matching configs/brands.yaml's "Other/Miscellaneous", which already covers KBA's
# "SONSTIGE" catch-all label).
DEFAULT_OTHER_CANONICAL_NAME = "Other/Miscellaneous"
DEFAULT_OTHER_ORIGIN_GROUP = "other"


def resolve_brands(
    frame: pd.DataFrame,
    alias_map: dict[str, BrandAlias],
    *,
    brand_column: str = "brand",
    spelling_overrides: dict[str, str] | None = None,
    fallback_canonical_name: str | None = DEFAULT_OTHER_CANONICAL_NAME,
    fallback_origin_group: str | None = DEFAULT_OTHER_ORIGIN_GROUP,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map raw brand strings in `frame[brand_column]` to canonical brands via `alias_map`.

    Shared by every raw-data adapter that needs to resolve brand strings against
    `configs/brands.yaml` (KBA, Switzerland). `spelling_overrides` remaps a raw
    casefolded string to another alias key already present in `alias_map` (e.g. a
    source-specific spelling quirk), applied before the alias lookup.

    By default, any raw brand string not found in `alias_map` is bucketed into
    `fallback_canonical_name`/`fallback_origin_group` (see `DEFAULT_OTHER_CANONICAL_NAME`)
    rather than dropped -- this is a deliberate, deterministic classification rule ("not
    German/Chinese/other-explicit => Other/Miscellaneous"), not a guess about which
    specific brand it is. Returns `(resolved, unresolved)`; with the default fallback
    enabled, `unresolved` is always empty. Pass `fallback_canonical_name=None` to instead
    return unmatched rows as `unresolved` and exclude them from `resolved` entirely (the
    historical strict behavior).
    """
    overrides = spelling_overrides or {}
    result = frame.copy()

    def _resolve(raw_value: object) -> BrandAlias | None:
        key = str(raw_value).strip().casefold()
        key = overrides.get(key, key)
        return alias_map.get(key)

    resolved_alias = result[brand_column].map(_resolve)
    if fallback_canonical_name is not None:
        result["canonical_brand"] = resolved_alias.map(
            lambda alias: alias.canonical_name if alias is not None else fallback_canonical_name
        )
        result["origin_group"] = resolved_alias.map(
            lambda alias: alias.origin_group if alias is not None else fallback_origin_group
        )
    else:
        result["canonical_brand"] = resolved_alias.map(
            lambda alias: alias.canonical_name if alias is not None else None
        )
        result["origin_group"] = resolved_alias.map(
            lambda alias: alias.origin_group if alias is not None else None
        )
    resolved = result[result["canonical_brand"].notna()].copy()
    unresolved = result[result["canonical_brand"].isna()].copy()
    return resolved, unresolved


def match_brand_mentions(text: str, alias_map: dict[str, BrandAlias | str]) -> list[str]:
    """Return a list of canonical brand names found in text."""
    matches: set[str] = set()
    value = text or ""
    automotive_context = re.search(
        r"\b(auto|car|vehicle|fahrzeug|modell|motor|ev|elektro|battery|batterie|"
        r"reichweite|dealer|händler|kaufen|buy|lease|leasing)\b",
        value,
        re.IGNORECASE,
    )
    for alias, details in alias_map.items():
        if isinstance(details, str):
            details = BrandAlias(details, "unknown")
        flags = 0 if details.case_sensitive else re.IGNORECASE
        expression = details.pattern or rf"(?<!\w){re.escape(alias)}(?!\w)"
        found = re.search(expression, value, flags)
        if not found:
            continue
        exact_uppercase = found.group(0).isupper() and len(found.group(0)) <= 3
        model_context = any(
            re.search(rf"(?<!\w){re.escape(model)}(?!\w)", value, re.IGNORECASE)
            for model in details.vehicle_models
        )
        if details.ambiguous and not (exact_uppercase or automotive_context or model_context):
            continue
        matches.add(details.canonical_name)
    return sorted(matches)


def normalize_brand_field(value: Iterable[str] | str | None) -> list[str]:
    """Coerce brand values into a stable list for downstream analytics."""
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]
