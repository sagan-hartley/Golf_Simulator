"""weights.py.

Rank-based nudging of player participation weights after each event,
used by the "dynamic weights" mode of the season simulation.
"""

import numpy as np
import pandas as pd

from golf_simulator.settings import DynamicWeightConfig

# ── Dynamic weight nudging ─────────────────────────────────────────────────────


def nudge_weights(
    current_weights: dict,
    baseline_weights: dict,
    results_df: pd.DataFrame,
    config: DynamicWeightConfig,
) -> dict:
    """
    Apply a small rank-based nudge to participation weights after an event.

    Players finishing in the top bucket get a small weight increase.
    Players finishing in the bottom bucket or missing the cut get a decrease.
    Players who did not participate or finished in the middle are unchanged.

    A player's weight is bounded symmetrically relative to their baseline:
    it can grow to at most ``baseline * max_weight_multiplier`` and shrink to
    at most ``baseline / max_weight_multiplier``, so a strong player on a cold
    streak decays no more aggressively than a weak player on a hot streak
    rises. ``min_weight`` is a final absolute safety floor beneath that.

    Parameters
    ----------
    current_weights : dict
        Current normalised weights {pid: weight}.
    baseline_weights : dict
        Original static weights {pid: weight}, used for the symmetric
        ceiling/floor bounds.
    results_df : pd.DataFrame
        Must contain columns 'Player' and 'FinalRank'.
        Missed-cut players should have FinalRank == NaN.
    config : DynamicWeightConfig

    Returns
    -------
    dict
        New normalised weights {pid: weight}.
    """
    finished = results_df.dropna(subset=["FinalRank"]).sort_values("FinalRank")
    finished_pids = finished["Player"].astype(str).tolist()
    missed_cut = results_df[results_df["FinalRank"].isna()]["Player"].astype(str).tolist()

    n_finished = len(finished_pids)
    top_cutoff = max(1, int(np.ceil(n_finished * config.top_pct)))
    bot_start = n_finished - int(np.ceil(n_finished * config.bot_pct))
    # Guard against overlapping top/bottom slices (aggressive top_pct + bot_pct):
    # a player can never be in both buckets; the top finish takes precedence.
    bot_start = max(bot_start, top_cutoff)

    top_pids = set(finished_pids[:top_cutoff])
    bottom_pids = set(finished_pids[bot_start:]) | set(missed_cut)

    pids = list(current_weights.keys())
    weights = np.array([current_weights[pid] for pid in pids], dtype=float)
    baseline = np.array([baseline_weights[pid] for pid in pids], dtype=float)

    multiplier = np.ones(len(pids), dtype=float)
    for i, pid in enumerate(pids):
        if pid in top_pids:
            multiplier[i] = 1.0 + config.nudge_amount
        elif pid in bottom_pids:
            multiplier[i] = 1.0 - config.nudge_amount
    nudged = weights * multiplier

    # Symmetric baseline-relative bounds, then an absolute safety floor.
    ceiling = baseline * config.max_weight_multiplier
    floor = baseline / config.max_weight_multiplier
    bounded = np.clip(nudged, floor, ceiling)
    bounded = np.maximum(bounded, config.min_weight)

    bounded = bounded / bounded.sum()
    return {pid: float(bounded[i]) for i, pid in enumerate(pids)}
