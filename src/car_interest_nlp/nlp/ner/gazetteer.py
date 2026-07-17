from __future__ import annotations

from ...config import load_project_config, load_yaml_config

# The custom automotive NER labels this project defines, beyond spaCy's generic built-in
# ones (PERSON, ORG, GPE/LOCATION already exist in en_core_web_sm and are reused as-is
# rather than redefined -- see pipeline.py).
CUSTOM_NER_LABELS: tuple[str, ...] = (
    "CAR_BRAND",
    "CAR_MODEL",
    "SUPPLIER",
    "TECHNOLOGY",
    "COMPONENT",
    "FACTORY",
    "REGULATION",
)


def build_entity_ruler_patterns() -> list[dict[str, object]]:
    """Build spaCy `EntityRuler` patterns from real, curated gazetteers.

    `CAR_BRAND` patterns come from `configs/brands.yaml` (every German/Chinese canonical
    name and alias -- the single source of truth already used by the KBA/Switzerland/
    Trends chapters, not duplicated). `CAR_MODEL`/`SUPPLIER`/`TECHNOLOGY`/`COMPONENT`/
    `FACTORY`/`REGULATION` come from `configs/ner_gazetteer.yaml`. Every pattern is an
    exact real phrase (not a syntactic/token-attribute pattern), since the point of Stage A
    is precise, deterministic matching of known real-world names/terms, not generalization.
    Typed `dict[str, object]` (not `dict[str, str]`) only because `EntityRuler.add_patterns`
    itself accepts that broader shape (a pattern's value can also be a token-attribute list,
    unused here since every pattern is a plain string).
    """
    patterns: list[dict[str, object]] = []

    brands_config = load_project_config()["brands"]
    for group in ("german", "chinese"):
        for entry in brands_config.get(group, []):
            for alias in entry.get("aliases") or [entry["canonical_name"]]:
                patterns.append({"label": "CAR_BRAND", "pattern": alias})

    gazetteer = load_yaml_config("ner_gazetteer").get("gazetteer", {})
    for label, entries in gazetteer.items():
        if isinstance(entries, dict):
            # CAR_MODEL is brand -> [models]; the brand itself is already a CAR_BRAND
            # pattern above, only the model names are added here.
            for models in entries.values():
                for model in models:
                    patterns.append({"label": label, "pattern": model})
        else:
            for term in entries:
                patterns.append({"label": label, "pattern": term})
    return patterns
