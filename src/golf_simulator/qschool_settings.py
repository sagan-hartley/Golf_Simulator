"""qschool_settings.py.

Loads and validates ``config/qschool.yaml`` — the file a non-coder edits
to run the "Q-School" analysis: for a player of a given caliber, how
likely are they to earn playing status through the 4-stage qualifying
gauntlet, and how much does their entry point (stage 1, 2, or 3) matter.

Follows the same merge-with-defaults + validate pattern as the other
settings modules, reusing `DataConfig` for the two player pools (the
aspirants whose odds we compute, and the competition they face).
"""

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from golf_simulator.settings import DataConfig

DEFAULT_QSCHOOL_SETTINGS_PATH = "config/qschool.yaml"


class QSchoolSettingsError(ValueError):
    """Raised when config/qschool.yaml is missing, malformed, or out of range."""


@dataclass
class QSchoolConfig:
    """Controls the Q-School gauntlet: advance gates, field strength, and status tiers."""

    advance_pct: float = 0.20
    strength_step: float = 0.5
    rounds_per_stage: int = 4
    stage_field_size: int = 78
    pga_card_spots: int = 5
    full_kf_pct: float = 0.25
    conditional_pct: float = 0.25
    n_simulations: int = 500
    base_seed: int = 0


@dataclass
class QSchoolOutputConfig:
    """Controls where the Q-School results CSV is written."""

    output_dir: str = "outputs"
    filename: str = "qschool_results.csv"


@dataclass
class QSchoolSettings:
    """Top-level settings object assembled from config/qschool.yaml."""

    aspirant_pool: DataConfig
    competition_pool: DataConfig
    qschool: QSchoolConfig
    output: QSchoolOutputConfig


_SECTION_TYPES = {
    "aspirant_pool": DataConfig,
    "competition_pool": DataConfig,
    "qschool": QSchoolConfig,
    "output": QSchoolOutputConfig,
}


def _build_section(section_name: str, raw: dict[str, Any] | None):
    """Merge a raw YAML mapping onto a section dataclass's defaults, then validate."""
    section_cls = _SECTION_TYPES[section_name]
    raw = raw or {}

    if not isinstance(raw, dict):
        raise QSchoolSettingsError(
            f"config/qschool.yaml: '{section_name}' must be a mapping of key: value "
            f"pairs, got {type(raw).__name__}."
        )

    valid_keys = {f.name for f in fields(section_cls)}
    unknown = set(raw) - valid_keys
    if unknown:
        raise QSchoolSettingsError(
            f"config/qschool.yaml: unknown key(s) under '{section_name}': "
            f"{', '.join(sorted(unknown))}. Valid keys are: {', '.join(sorted(valid_keys))}."
        )

    section = section_cls(**raw)

    if section_name == "qschool":
        for field_name in ("advance_pct", "full_kf_pct", "conditional_pct"):
            value = getattr(section, field_name)
            if not 0.0 < value <= 1.0:
                raise QSchoolSettingsError(
                    f"config/qschool.yaml: qschool.{field_name} must be greater than 0 "
                    f"and at most 1 (got {value})."
                )
        for field_name in ("rounds_per_stage", "stage_field_size", "pga_card_spots",
                           "n_simulations"):
            value = getattr(section, field_name)
            if value < 1:
                raise QSchoolSettingsError(
                    f"config/qschool.yaml: qschool.{field_name} must be at least 1 "
                    f"(got {value})."
                )
        if section.strength_step < 0:
            raise QSchoolSettingsError(
                f"config/qschool.yaml: qschool.strength_step must be 0 or greater "
                f"(got {section.strength_step})."
            )

    return section


def load_qschool_settings(
    path: str | Path = DEFAULT_QSCHOOL_SETTINGS_PATH,
) -> QSchoolSettings:
    """
    Load, merge with defaults, and validate ``config/qschool.yaml``.

    Parameters
    ----------
    path : str or Path
        Location of the YAML settings file.

    Returns
    -------
    QSchoolSettings

    Raises
    ------
    QSchoolSettingsError
        If the file is missing, isn't valid YAML, or contains an
        out-of-range value. The message names the offending
        ``section.key`` and the file path.
    """
    path = Path(path)
    if not path.exists():
        raise QSchoolSettingsError(f"Settings file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise QSchoolSettingsError(
            f"{path}: could not parse YAML — check indentation and colons. Details: {e}"
        ) from e

    if not isinstance(raw, dict):
        raise QSchoolSettingsError(
            f"{path}: top level of the file must be a mapping of sections."
        )

    unknown_sections = set(raw) - set(_SECTION_TYPES)
    if unknown_sections:
        raise QSchoolSettingsError(
            f"{path}: unknown section(s): {', '.join(sorted(unknown_sections))}. "
            f"Valid sections are: {', '.join(sorted(_SECTION_TYPES))}."
        )

    return QSchoolSettings(
        aspirant_pool=_build_section("aspirant_pool", raw.get("aspirant_pool")),
        competition_pool=_build_section("competition_pool", raw.get("competition_pool")),
        qschool=_build_section("qschool", raw.get("qschool")),
        output=_build_section("output", raw.get("output")),
    )
