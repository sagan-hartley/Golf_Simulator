"""player_field.py.

Loads a custom, user-specified field of players directly from a CSV of
statistical moments — an alternative to imputing player distributions
from historical round-score data (see :mod:`golf_simulator.data_loading`).
Useful for hypothetical or synthetic fields (a Monday qualifier, a
Q-school stage, a fixed eligibility pool) that don't exist in
``data/seasons/``.
"""

from pathlib import Path

import numpy as np
import pandas as pd

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
        ``mean``, ``variance``, ``skew``, and ``weight`` (column names
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

    required_cols = (id_col, mean_col, var_col, skew_col, weight_col)
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise FieldError(
            f"{path}: missing required column(s): {', '.join(missing_cols)}. "
            f"Expected columns: {', '.join(required_cols)}."
        )

    if df.empty:
        raise FieldError(f"{path}: field file has no rows.")

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
