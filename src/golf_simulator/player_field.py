"""player_field.py.

Loads a custom, user-specified field of players directly from a CSV of
statistical moments — an alternative to imputing player distributions
from historical round-score data (see :mod:`golf_simulator.data_loading`).
Useful for hypothetical or synthetic fields (a Monday qualifier, a
Q-school stage, a fixed eligibility pool) that don't exist in
``data/seasons/``.
"""

from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

from golf_simulator.data_loading import compute_player_stats
from golf_simulator.distributions import build_player_generators, skewnorm_params_from_moments

DEFAULT_ID_COL = "player_id"
DEFAULT_MEAN_COL = "mean"
DEFAULT_VAR_COL = "variance"
DEFAULT_SKEW_COL = "skew"
DEFAULT_WEIGHT_COL = "weight"


class FieldError(ValueError):
    """Raised when a custom field file is missing, malformed, or has an invalid row."""


def load_custom_field(
    path: str | Path,
    id_col: str = DEFAULT_ID_COL,
    mean_col: str = DEFAULT_MEAN_COL,
    var_col: str = DEFAULT_VAR_COL,
    skew_col: str = DEFAULT_SKEW_COL,
    weight_col: str = DEFAULT_WEIGHT_COL,
) -> dict:
    """
    Load a custom field of players from a CSV of statistical moments.

    Parameters
    ----------
    path : str or Path
        Location of the field CSV. Must have columns ``player_id``,
        ``mean``, and ``variance``. ``skew`` and ``weight`` are optional
        and default to 0 and 1 respectively when absent (column names
        configurable via the ``*_col`` arguments).
    id_col, mean_col, var_col, skew_col, weight_col : str
        Column names to read.

    Returns
    -------
    dict
        pid -> (a, loc, scale, weight), the same shape
        `golf_simulator.distributions.build_player_generators` returns —
        a drop-in replacement wherever `player_params` is consumed.

    Raises
    ------
    FieldError
        If the file is missing, a required column is absent, a player id
        is duplicated, or a row has an invalid `variance` or `weight`.
    """
    path = Path(path)
    if not path.exists():
        raise FieldError(f"Field file not found: {path}")

    df = pd.read_csv(path)

    required_cols = (id_col, mean_col, var_col)
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise FieldError(
            f"{path}: missing required column(s): {', '.join(missing_cols)}. "
            f"Required columns: {', '.join(required_cols)} "
            f"(optional: {skew_col} defaults to 0, {weight_col} defaults to 1)."
        )

    if df.empty:
        raise FieldError(f"{path}: field file has no rows.")

    # skew and weight are optional -- fill sensible defaults when absent.
    if skew_col not in df.columns:
        df[skew_col] = 0.0
    if weight_col not in df.columns:
        df[weight_col] = 1.0

    duplicate_ids = df[id_col][df[id_col].duplicated()].unique().tolist()
    if duplicate_ids:
        raise FieldError(
            f"{path}: duplicate player id(s): {', '.join(str(d) for d in duplicate_ids)}."
        )

    for row in df.itertuples(index=False):
        pid = getattr(row, id_col)
        variance = getattr(row, var_col)
        weight = getattr(row, weight_col)

        if not np.isfinite(variance) or variance <= 0:
            raise FieldError(
                f"{path}: player id '{pid}' has invalid {var_col}={variance} "
                f"(must be a positive, finite number)."
            )
        if not np.isfinite(weight) or weight <= 0:
            raise FieldError(
                f"{path}: player id '{pid}' has invalid {weight_col}={weight} "
                f"(must be a positive, finite number)."
            )

    # Pre-validate each row so a bad (mean, variance, skew) combination raises a
    # FieldError naming the specific player id, instead of an opaque error from
    # deep inside the skew-normal root-finder.
    for row in df.itertuples(index=False):
        pid = getattr(row, id_col)
        try:
            skewnorm_params_from_moments(
                float(getattr(row, mean_col)),
                float(getattr(row, var_col)),
                float(getattr(row, skew_col)),
            )
        except (ValueError, RuntimeError) as e:
            raise FieldError(
                f"{path}: player id '{pid}' has an invalid mean/variance/skew "
                f"combination: {e}"
            ) from e

    renamed = df.rename(columns={
        id_col: "Player",
        mean_col: "Mean",
        var_col: "Variance",
        skew_col: "Skew",
        weight_col: "Weight",
    })
    return build_player_generators(renamed)


def _resolve_season_files(data_config) -> list:
    """
    Resolve which historical CSVs to use for a `DataConfig`.

    If `data_config.season_files` is set, use exactly those files (each
    resolved as-is if it exists, otherwise relative to `season_dir`) so a
    caller can pick specific years. Otherwise use every ``*.csv`` in
    `season_dir`.

    Raises
    ------
    FieldError
        If a listed file can't be found, or no CSVs are found in the folder.
    """
    listed = getattr(data_config, "season_files", None)
    if listed:
        resolved = []
        for name in listed:
            p = Path(name)
            if not p.exists():
                p = Path(data_config.season_dir) / name
            if not p.exists():
                raise FieldError(
                    f"Season file not found: '{name}' (looked in {data_config.season_dir})."
                )
            resolved.append(str(p))
        return sorted(resolved)

    csv_paths = sorted(glob(str(Path(data_config.season_dir) / "*.csv")))
    if not csv_paths:
        raise FieldError(f"No CSV files found in {data_config.season_dir}")
    return csv_paths


def load_player_pool(data_config, participation_config) -> dict:
    """
    Build a player pool from a `DataConfig`, custom field or historical data.

    Uses `load_custom_field` if `data_config.field_file` is set. Otherwise
    fits player distributions from historical season CSVs -- either the
    specific files named in `data_config.season_files`, or every CSV in
    `data_config.season_dir`. Shared by all four analyses so each pool loads
    identically.

    Parameters
    ----------
    data_config : golf_simulator.settings.DataConfig
    participation_config : golf_simulator.settings.ParticipationWeightConfig
        Only used for the historical-data path.

    Returns
    -------
    dict
        pid -> (a, loc, scale, weight)

    Raises
    ------
    FieldError
        If a custom field file is invalid, a listed season file is missing,
        or no historical CSVs are found.
    """
    if data_config.field_file:
        return load_custom_field(data_config.field_file)

    csv_paths = _resolve_season_files(data_config)

    moments = compute_player_stats(
        csv_paths,
        data_config.player_column,
        data_config.score_column,
        min_avg_rounds=participation_config.min_avg_rounds,
        weight_power=participation_config.weight_power,
        weight_floor=participation_config.weight_floor,
    )
    return build_player_generators(moments)
