"""monday_chase_settings.py.

Loads and validates ``config/monday_chase.yaml`` — the file a non-coder
edits to run the "chasing Mondays" analysis: how often a player can
qualify for a Tour event through Monday qualifying, and then parlay a
good finish into an exemption from having to qualify again the
following week.

Follows the same merge-with-defaults + validate pattern as
:mod:`golf_simulator.settings`, reusing its `DataConfig` for the two
player pools this analysis needs (the Monday-qualifier aspirants, and
the field they're trying to crash).
"""

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from golf_simulator.settings import DataConfig

DEFAULT_MONDAY_CHASE_SETTINGS_PATH = "config/monday_chase.yaml"


class MondayChaseSettingsError(ValueError):
    """Raised when config/monday_chase.yaml is missing, malformed, or out of range."""


@dataclass
class ChaseConfig:
    """Controls the Monday cutoff, parlay threshold, and weeks/simulations to run."""

    advance_pct: float = 0.04
    parlay_top_n: int = 25
    n_weeks: int = 8
    n_simulations: int = 200
    base_seed: int = 0


@dataclass
class MondayChaseOutputConfig:
    """Controls where the chase-results CSV is written."""

    output_dir: str = "outputs"
    filename: str = "monday_chase_results.csv"


@dataclass
class MondayChaseSettings:
    """Top-level settings object assembled from config/monday_chase.yaml."""

    aspirant_pool: DataConfig
    main_event_pool: DataConfig
    chase: ChaseConfig
    output: MondayChaseOutputConfig


_SECTION_TYPES = {
    "aspirant_pool": DataConfig,
    "main_event_pool": DataConfig,
    "chase": ChaseConfig,
    "output": MondayChaseOutputConfig,
}


def _build_section(section_name: str, raw: dict[str, Any] | None):
    """Merge a raw YAML mapping onto a section dataclass's defaults, then validate."""
    section_cls = _SECTION_TYPES[section_name]
    raw = raw or {}

    if not isinstance(raw, dict):
        raise MondayChaseSettingsError(
            f"config/monday_chase.yaml: '{section_name}' must be a mapping of "
            f"key: value pairs, got {type(raw).__name__}."
        )

    valid_keys = {f.name for f in fields(section_cls)}
    unknown = set(raw) - valid_keys
    if unknown:
        raise MondayChaseSettingsError(
            f"config/monday_chase.yaml: unknown key(s) under '{section_name}': "
            f"{', '.join(sorted(unknown))}. Valid keys are: {', '.join(sorted(valid_keys))}."
        )

    section = section_cls(**raw)

    if section_name == "chase":
        if not 0.0 < section.advance_pct <= 1.0:
            raise MondayChaseSettingsError(
                f"config/monday_chase.yaml: chase.advance_pct must be greater than 0 "
                f"and at most 1 (got {section.advance_pct})."
            )
        if section.parlay_top_n < 1:
            raise MondayChaseSettingsError(
                f"config/monday_chase.yaml: chase.parlay_top_n must be at least 1 "
                f"(got {section.parlay_top_n})."
            )
        if section.n_weeks < 1:
            raise MondayChaseSettingsError(
                f"config/monday_chase.yaml: chase.n_weeks must be at least 1 "
                f"(got {section.n_weeks})."
            )
        if section.n_simulations < 1:
            raise MondayChaseSettingsError(
                f"config/monday_chase.yaml: chase.n_simulations must be at least 1 "
                f"(got {section.n_simulations})."
            )

    return section


def load_monday_chase_settings(
    path: str | Path = DEFAULT_MONDAY_CHASE_SETTINGS_PATH,
) -> MondayChaseSettings:
    """
    Load, merge with defaults, and validate ``config/monday_chase.yaml``.

    Parameters
    ----------
    path : str or Path
        Location of the YAML settings file.

    Returns
    -------
    MondayChaseSettings

    Raises
    ------
    MondayChaseSettingsError
        If the file is missing, isn't valid YAML, or contains an
        out-of-range value. The message names the offending
        ``section.key`` and the file path.
    """
    path = Path(path)
    if not path.exists():
        raise MondayChaseSettingsError(f"Settings file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise MondayChaseSettingsError(
            f"{path}: could not parse YAML — check indentation and colons. Details: {e}"
        ) from e

    if not isinstance(raw, dict):
        raise MondayChaseSettingsError(
            f"{path}: top level of the file must be a mapping of sections."
        )

    unknown_sections = set(raw) - set(_SECTION_TYPES)
    if unknown_sections:
        raise MondayChaseSettingsError(
            f"{path}: unknown section(s): {', '.join(sorted(unknown_sections))}. "
            f"Valid sections are: {', '.join(sorted(_SECTION_TYPES))}."
        )

    return MondayChaseSettings(
        aspirant_pool=_build_section("aspirant_pool", raw.get("aspirant_pool")),
        main_event_pool=_build_section("main_event_pool", raw.get("main_event_pool")),
        chase=_build_section("chase", raw.get("chase")),
        output=_build_section("output", raw.get("output")),
    )
