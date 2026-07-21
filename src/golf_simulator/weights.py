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

    Parameters
    ----------
    current_weights : dict
        Current normalised weights {pid: weight}.
    baseline_weights : dict
        Original static weights {pid: weight}, used to enforce ceiling.
    results_df : pd.DataFrame
        Must contain columns 'Player' and 'FinalRank'.
        Missed-cut players should have FinalRank == NaN.
    config : DynamicWeightConfig

    Returns
    -------
    dict
        New normalised weights {pid: weight}.
    """
    finished = results_df.dropna(subset=["FinalRank"]).copy()
    missed_cut = (
        results_df[results_df["FinalRank"].isna()]["Player"]
        .astype(str)
        .tolist()
    )

    n_finished = len(finished)
    top_cutoff = max(1, int(np.ceil(n_finished * config.top_pct)))
    bot_cutoff = n_finished - int(np.ceil(n_finished * config.bot_pct))

    finished = finished.sort_values("FinalRank").reset_index(drop=True)
    finished["_bucket"] = "middle"
    finished.loc[finished.index < top_cutoff, "_bucket"] = "top"
    finished.loc[finished.index >= bot_cutoff, "_bucket"] = "bottom"

    bucket_map = dict(zip(
        finished["Player"].astype(str),
        finished["_bucket"],
    ))
    for pid in missed_cut:
        bucket_map[pid] = "bottom"

    new_weights = {}
    for pid, w in current_weights.items():
        bucket = bucket_map.get(pid)

        if bucket == "top":
            nudged = w * (1.0 + config.nudge_amount)
        elif bucket == "bottom":
            nudged = w * (1.0 - config.nudge_amount)
        else:
            nudged = w

        ceiling = baseline_weights[pid] * config.max_weight_multiplier
        new_weights[pid] = min(nudged, ceiling)

    # Floor then re-normalise
    new_weights = {pid: max(w, config.min_weight) for pid, w in new_weights.items()}
    total = sum(new_weights.values())
    return {pid: w / total for pid, w in new_weights.items()}
