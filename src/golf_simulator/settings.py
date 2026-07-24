"""settings.py.

Loads and validates ``config/settings.yaml`` — the single file a
non-coder should edit to change simulation inputs (dynamic-weight
tuning, participation-weight calculation, Monte Carlo run size, data
locations, and output file names).

Any key omitted from the YAML file silently falls back to the
dataclass default below, so a beginner can edit just the one value
they care about without retyping the whole file.
"""

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

DEFAULT_SETTINGS_PATH = "config/settings.yaml"


class SettingsError(ValueError):
    """Raised when config/settings.yaml is missing, malformed, or has an out-of-range value."""


@dataclass
class DynamicWeightConfig:
    """
    Controls rank-based weight nudging after each event.

    Attributes
    ----------
    enabled : bool
        Master switch. False reproduces original static behaviour.
    nudge_amount : float
        How much to adjust weight after each event (e.g. 0.05 = 5%).
        Applied as a multiplier: weight *= (1 + nudge) or (1 - nudge).
    top_pct : float
        Top finishing percentile that earns a positive nudge.
        e.g. 0.25 = top quarter of finishers get a boost.
    bot_pct : float
        Bottom finishing percentile that earns a negative nudge.
        e.g. 0.25 = bottom quarter of finishers get a penalty.
    min_weight : float
        Absolute safety floor applied after the baseline-relative bounds,
        before re-normalisation.
    max_weight_multiplier : float
        Bounds a player's weight symmetrically relative to baseline: it can
        grow to at most ``baseline * max_weight_multiplier`` and shrink to at
        most ``baseline / max_weight_multiplier``.
    """

    enabled: bool = True
    nudge_amount: float = 0.05
    top_pct: float = 0.25
    bot_pct: float = 0.25
    min_weight: float = 0.001
    max_weight_multiplier: float = 2.0


@dataclass
class ParticipationWeightConfig:
    """Controls how a player's historical event count becomes a field-selection weight."""

    min_avg_rounds: int = 20
    weight_power: float = 1.0
    weight_floor: float = 0.05


@dataclass
class MonteCarloConfig:
    """Controls how many seasons are simulated and which random seeds are used."""

    n_simulations: int = 10
    base_seed: int = 0
    season_seed: int = 123


@dataclass
class DataConfig:
    """Points at the historical round-score data used to fit player distributions."""

    season_dir: str = "data/seasons"
    season_files: list | None = None
    player_column: str = "player"
    score_column: str = "score"
    field_file: str | None = None


@dataclass
class OutputConfig:
    """Controls where result CSVs are written."""

    output_dir: str = "outputs"
    season_static_filename: str = "season_static.csv"
    season_dynamic_filename: str = "season_dynamic.csv"
    mc_results_filename: str = "mc_results.csv"


@dataclass
class Settings:
    """Top-level settings object assembled from config/settings.yaml."""

    dynamic_weights: DynamicWeightConfig
    participation_weights: ParticipationWeightConfig
    monte_carlo: MonteCarloConfig
    data: DataConfig
    output: OutputConfig


_SECTION_TYPES = {
    "dynamic_weights": DynamicWeightConfig,
    "participation_weights": ParticipationWeightConfig,
    "monte_carlo": MonteCarloConfig,
    "data": DataConfig,
    "output": OutputConfig,
}

_RATIO_FIELDS = {
    "dynamic_weights": ("nudge_amount", "top_pct", "bot_pct"),
    "participation_weights": ("weight_floor",),
}
_POSITIVE_FIELDS = {
    "dynamic_weights": ("min_weight",),
}
_AT_LEAST_ONE_FIELDS = {
    "dynamic_weights": ("max_weight_multiplier",),
}
_NON_NEGATIVE_FIELDS = {
    "participation_weights": ("min_avg_rounds",),
}
_AT_LEAST_ONE_INT_FIELDS = {
    "monte_carlo": ("n_simulations",),
}


def _build_section(section_name: str, raw: dict[str, Any] | None):
    """Merge a raw YAML mapping onto a section dataclass's defaults, then validate."""
    section_cls = _SECTION_TYPES[section_name]
    raw = raw or {}

    if not isinstance(raw, dict):
        raise SettingsError(
            f"config/settings.yaml: '{section_name}' must be a mapping of key: value pairs, "
            f"got {type(raw).__name__}."
        )

    valid_keys = {f.name for f in fields(section_cls)}
    unknown = set(raw) - valid_keys
    if unknown:
        raise SettingsError(
            f"config/settings.yaml: unknown key(s) under '{section_name}': "
            f"{', '.join(sorted(unknown))}. Valid keys are: {', '.join(sorted(valid_keys))}."
        )

    section = section_cls(**raw)

    for field_name in _RATIO_FIELDS.get(section_name, ()):
        value = getattr(section, field_name)
        if not 0.0 <= value <= 1.0:
            raise SettingsError(
                f"config/settings.yaml: {section_name}.{field_name} must be between 0 and 1 "
                f"(got {value})."
            )
    for field_name in _POSITIVE_FIELDS.get(section_name, ()):
        value = getattr(section, field_name)
        if value <= 0.0:
            raise SettingsError(
                f"config/settings.yaml: {section_name}.{field_name} must be greater than 0 "
                f"(got {value})."
            )
    for field_name in _AT_LEAST_ONE_FIELDS.get(section_name, ()):
        value = getattr(section, field_name)
        if value < 1.0:
            raise SettingsError(
                f"config/settings.yaml: {section_name}.{field_name} must be at least 1.0 "
                f"(got {value})."
            )
    for field_name in _NON_NEGATIVE_FIELDS.get(section_name, ()):
        value = getattr(section, field_name)
        if value < 0:
            raise SettingsError(
                f"config/settings.yaml: {section_name}.{field_name} must be 0 or greater "
                f"(got {value})."
            )
    for field_name in _AT_LEAST_ONE_INT_FIELDS.get(section_name, ()):
        value = getattr(section, field_name)
        if value < 1:
            raise SettingsError(
                f"config/settings.yaml: {section_name}.{field_name} must be at least 1 "
                f"(got {value})."
            )

    return section


def load_settings(path: str | Path = DEFAULT_SETTINGS_PATH) -> Settings:
    """
    Load, merge with defaults, and validate ``config/settings.yaml``.

    Parameters
    ----------
    path : str or Path
        Location of the YAML settings file.

    Returns
    -------
    Settings

    Raises
    ------
    SettingsError
        If the file is missing, isn't valid YAML, or contains an
        out-of-range value. The message names the offending
        ``section.key`` and the file path.
    """
    path = Path(path)
    if not path.exists():
        raise SettingsError(f"Settings file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise SettingsError(
            f"{path}: could not parse YAML — check indentation and colons. Details: {e}"
        ) from e

    if not isinstance(raw, dict):
        raise SettingsError(f"{path}: top level of the file must be a mapping of sections.")

    unknown_sections = set(raw) - set(_SECTION_TYPES)
    if unknown_sections:
        raise SettingsError(
            f"{path}: unknown section(s): {', '.join(sorted(unknown_sections))}. "
            f"Valid sections are: {', '.join(sorted(_SECTION_TYPES))}."
        )

    return Settings(
        dynamic_weights=_build_section("dynamic_weights", raw.get("dynamic_weights")),
        participation_weights=_build_section(
            "participation_weights", raw.get("participation_weights")
        ),
        monte_carlo=_build_section("monte_carlo", raw.get("monte_carlo")),
        data=_build_section("data", raw.get("data")),
        output=_build_section("output", raw.get("output")),
    )
