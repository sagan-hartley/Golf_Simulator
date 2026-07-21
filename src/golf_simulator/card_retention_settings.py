"""card_retention_settings.py.

Loads and validates ``config/card_retention.yaml`` — the file a
non-coder edits to run the "card retention" analysis: under a fixed
eligible pool of players (a "card pool") competing in a season of
mostly-120-player events plus four 156-player majors (topped up from
a second "outside qualifiers" pool), what's the probability each
player finishes well enough to keep their card for next season.

Follows the same merge-with-defaults + validate pattern as
:mod:`golf_simulator.settings` and
:mod:`golf_simulator.monday_chase_settings`, reusing `DataConfig` for
the two player pools and `DynamicWeightConfig` for optional weight
nudging.
"""

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from golf_simulator.settings import DataConfig, DynamicWeightConfig

DEFAULT_CARD_RETENTION_SETTINGS_PATH = "config/card_retention.yaml"


class CardRetentionSettingsError(ValueError):
    """Raised when config/card_retention.yaml is missing, malformed, or out of range."""


@dataclass
class ScheduleConfig:
    """Points at the season schedule CSV to use for this analysis."""

    path: str = "config/alignment_schedule.csv"


@dataclass
class RetentionConfig:
    """Controls the card-retention cutoff and how many seasons/simulations to run."""

    cutoff: int = 90
    n_simulations: int = 200
    base_seed: int = 0
    season_seed: int = 123


@dataclass
class CardRetentionOutputConfig:
    """Controls where the retention-results CSV is written."""

    output_dir: str = "outputs"
    filename: str = "card_retention_results.csv"


@dataclass
class CardRetentionSettings:
    """Top-level settings object assembled from config/card_retention.yaml."""

    card_pool: DataConfig
    outside_pool: DataConfig
    schedule: ScheduleConfig
    dynamic_weights: DynamicWeightConfig
    retention: RetentionConfig
    output: CardRetentionOutputConfig


_SECTION_TYPES = {
    "card_pool": DataConfig,
    "outside_pool": DataConfig,
    "schedule": ScheduleConfig,
    "dynamic_weights": DynamicWeightConfig,
    "retention": RetentionConfig,
    "output": CardRetentionOutputConfig,
}


def _build_section(section_name: str, raw: dict[str, Any] | None):
    """Merge a raw YAML mapping onto a section dataclass's defaults, then validate."""
    section_cls = _SECTION_TYPES[section_name]
    raw = raw or {}

    if not isinstance(raw, dict):
        raise CardRetentionSettingsError(
            f"config/card_retention.yaml: '{section_name}' must be a mapping of "
            f"key: value pairs, got {type(raw).__name__}."
        )

    valid_keys = {f.name for f in fields(section_cls)}
    unknown = set(raw) - valid_keys
    if unknown:
        raise CardRetentionSettingsError(
            f"config/card_retention.yaml: unknown key(s) under '{section_name}': "
            f"{', '.join(sorted(unknown))}. Valid keys are: {', '.join(sorted(valid_keys))}."
        )

    section = section_cls(**raw)

    if section_name == "dynamic_weights":
        for field_name in ("nudge_amount", "top_pct", "bot_pct"):
            value = getattr(section, field_name)
            if not 0.0 <= value <= 1.0:
                raise CardRetentionSettingsError(
                    f"config/card_retention.yaml: dynamic_weights.{field_name} must be "
                    f"between 0 and 1 (got {value})."
                )
        if section.min_weight <= 0.0:
            raise CardRetentionSettingsError(
                f"config/card_retention.yaml: dynamic_weights.min_weight must be "
                f"greater than 0 (got {section.min_weight})."
            )
        if section.max_weight_multiplier < 1.0:
            raise CardRetentionSettingsError(
                f"config/card_retention.yaml: dynamic_weights.max_weight_multiplier "
                f"must be at least 1.0 (got {section.max_weight_multiplier})."
            )

    if section_name == "retention":
        if section.cutoff < 1:
            raise CardRetentionSettingsError(
                f"config/card_retention.yaml: retention.cutoff must be at least 1 "
                f"(got {section.cutoff})."
            )
        if section.n_simulations < 1:
            raise CardRetentionSettingsError(
                f"config/card_retention.yaml: retention.n_simulations must be at "
                f"least 1 (got {section.n_simulations})."
            )

    return section


def load_card_retention_settings(
    path: str | Path = DEFAULT_CARD_RETENTION_SETTINGS_PATH,
) -> CardRetentionSettings:
    """
    Load, merge with defaults, and validate ``config/card_retention.yaml``.

    Parameters
    ----------
    path : str or Path
        Location of the YAML settings file.

    Returns
    -------
    CardRetentionSettings

    Raises
    ------
    CardRetentionSettingsError
        If the file is missing, isn't valid YAML, or contains an
        out-of-range value. The message names the offending
        ``section.key`` and the file path.
    """
    path = Path(path)
    if not path.exists():
        raise CardRetentionSettingsError(f"Settings file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise CardRetentionSettingsError(
            f"{path}: could not parse YAML — check indentation and colons. Details: {e}"
        ) from e

    if not isinstance(raw, dict):
        raise CardRetentionSettingsError(
            f"{path}: top level of the file must be a mapping of sections."
        )

    unknown_sections = set(raw) - set(_SECTION_TYPES)
    if unknown_sections:
        raise CardRetentionSettingsError(
            f"{path}: unknown section(s): {', '.join(sorted(unknown_sections))}. "
            f"Valid sections are: {', '.join(sorted(_SECTION_TYPES))}."
        )

    return CardRetentionSettings(
        card_pool=_build_section("card_pool", raw.get("card_pool")),
        outside_pool=_build_section("outside_pool", raw.get("outside_pool")),
        schedule=_build_section("schedule", raw.get("schedule")),
        dynamic_weights=_build_section("dynamic_weights", raw.get("dynamic_weights")),
        retention=_build_section("retention", raw.get("retention")),
        output=_build_section("output", raw.get("output")),
    )
