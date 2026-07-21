"""distributions.py.

Fits a skew-normal score distribution to each player's historical
(mean, variance, skew) and draws simulated round scores from it.
"""

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import skew, skewnorm

from golf_simulator.domain import PI

# ── Skew-normal helpers ────────────────────────────────────────────────────────


def _skew_from_delta(delta, eps=1e-12):
    """Skewness of a skew-normal distribution as a function of delta."""
    base = 1.0 - (2.0 * delta**2) / PI
    if base <= eps:
        raise ZeroDivisionError(
            f"Invalid delta={delta}: denominator approaches zero."
        )
    num = ((4.0 - PI) / 2.0) * (delta * np.sqrt(2.0 / PI))**3
    den = base**1.5
    return num / den


def skewnorm_params_from_moments(mean, variance, skew_val, eps=1e-8):
    """
    Convert (mean, variance, skewness) into scipy.stats.skewnorm parameters.

    Parameters
    ----------
    mean : float
    variance : float
    skew_val : float
    eps : float

    Returns
    -------
    tuple
        (a, loc, scale)
    """
    if variance <= 0 or not np.isfinite(variance):
        raise ValueError(f"Invalid variance: {variance}")
    if not np.isfinite(skew_val):
        raise ValueError(f"Invalid skew: {skew_val}")

    if abs(skew_val) < eps:
        return 0.0, mean, np.sqrt(variance)

    delta_min, delta_max = -0.999, 0.999

    try:
        skew_min = _skew_from_delta(delta_min)
        skew_max = _skew_from_delta(delta_max)
    except ZeroDivisionError as e:
        raise RuntimeError(
            f"Skew-normal boundary failure for mean={mean}, var={variance}, skew={skew_val}"
        ) from e

    target_skew = max(min(float(skew_val), skew_max), skew_min)

    def root_fn(delta):
        return _skew_from_delta(delta) - target_skew

    try:
        delta = brentq(root_fn, delta_min, delta_max)
    except Exception as e:
        raise RuntimeError(
            f"Root-finding failed for mean={mean}, var={variance}, skew={skew_val}"
        ) from e

    denom_a = 1.0 - delta**2
    denom_scale = 1.0 - (2.0 * delta**2) / PI

    if denom_a <= eps or denom_scale <= eps:
        raise ZeroDivisionError(
            f"Degenerate skew-normal parameters for delta={delta}"
        )

    a = delta / np.sqrt(denom_a)
    scale = np.sqrt(variance / denom_scale)
    loc = mean - scale * delta * np.sqrt(2.0 / PI)

    return a, loc, scale

# ── Player generators ──────────────────────────────────────────────────────────


def build_player_generators(player_stats_df, id_col="Player", mean_col="Mean",
                             var_col="Variance", skew_col="Skew", weight_col="Weight"):
    """
    Build skew-normal parameters for every player.

    Returns
    -------
    dict
        player_id -> (a, loc, scale, weight)
    """
    params = {}
    for _, row in player_stats_df.iterrows():
        pid = row[id_col]
        m = float(row[mean_col])
        v = float(row[var_col])
        s = float(row[skew_col])
        a, loc, scale = skewnorm_params_from_moments(m, v, s)
        w = float(row[weight_col])
        params[pid] = (a, loc, scale, w)
    return params


def sample_round_scores_for_players(player_params, n_rounds, player_ids=None, random_state=None):
    """
    Draw round-by-round integer scores for multiple players.

    Parameters
    ----------
    player_params : dict
    n_rounds : int
    player_ids : list or None
    random_state : numpy.random.Generator, int, or None
        Passed through to `scipy.stats.skewnorm.rvs`. Provide the same
        Generator (or seed) used elsewhere in a simulation run to make
        that run fully reproducible; None falls back to numpy's global
        random state.

    Returns
    -------
    numpy.ndarray
        Shape (num_players, n_rounds).
    """
    if player_ids is None:
        player_ids = list(player_params.keys())

    num_players = len(player_ids)
    scores = np.zeros((num_players, n_rounds), dtype=int)

    for i, pid in enumerate(player_ids):
        a, loc, scale, _ = player_params[pid]
        raw_scores = skewnorm.rvs(a, loc=loc, scale=scale, size=n_rounds, random_state=random_state)
        scores[i, :] = np.rint(raw_scores).astype(int)

    return scores


def simulate_and_compare_player(player_moments, player_params, player_id,
                                  n_rounds=200000, seed=123):
    """
    Simulate many rounds for one player and compare moments to targets.

    Parameters
    ----------
    player_moments : pd.DataFrame
    player_params : dict
    player_id : str
    n_rounds : int
    seed : int

    Returns
    -------
    pd.DataFrame
    """
    rng = np.random.default_rng(seed)
    a, loc, scale, _ = player_params[player_id]
    x = skewnorm.rvs(a, loc=loc, scale=scale, size=n_rounds, random_state=rng)

    sim_mean = float(np.mean(x))
    sim_var = float(np.var(x, ddof=1))
    sim_skew = float(skew(x, bias=False))

    row = player_moments[player_moments["Player"] == player_id].iloc[0]

    out = pd.DataFrame({
        "moment": ["mean", "variance", "skew"],
        "target": [float(row["Mean"]), float(row["Variance"]), float(row["Skew"])],
        "simulated": [sim_mean, sim_var, sim_skew],
    })
    out["diff"] = out["simulated"] - out["target"]
    out["pct_diff"] = out["diff"] / out["target"] * 100.0
    return out
