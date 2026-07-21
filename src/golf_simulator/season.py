"""season.py.

Simulates one full season: for each scheduled event, samples a field,
plays two rounds, applies the cut, plays the remaining rounds for
survivors, and assigns points — optionally nudging participation
weights after each event.
"""

from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import skew

from golf_simulator.distributions import sample_round_scores_for_players
from golf_simulator.domain import (
    CUT_AFTER_ROUND,
    PLAYER_ID_COL,
    POINTS_TABLES,
    ROUNDS_PER_EVENT,
    TOTAL_2R_COL,
)
from golf_simulator.points import apply_cut, assign_points_with_ties
from golf_simulator.settings import DynamicWeightConfig
from golf_simulator.weights import nudge_weights

# ── Season simulation ──────────────────────────────────────────────────────────


def simulate_season(
    player_params: dict,
    schedule: list,
    seed: int = 123,
    dynamic_weight_config: Optional[DynamicWeightConfig] = None,
    output_csv_path: Optional[str] = None,
) -> tuple:
    """
    Simulate a full season with optional rank-based dynamic weights.

    Parameters
    ----------
    player_params : dict
        pid -> (a, loc, scale, weight)
    schedule : list[TournamentType]
    seed : int
    dynamic_weight_config : DynamicWeightConfig or None
        None disables dynamic weights, reproducing original behaviour.
    output_csv_path : str or None
        If provided, the season summary CSV is written to this path.

    Returns
    -------
    season_summary : pd.DataFrame
        Ranked season points table with columns:
        Player, PreSeason_a, PreSeason_loc, PreSeason_scale, PreSeason_weight,
        SeasonPoints, SeasonRank.
        Written to output_csv_path if provided.
    event_results : list[pd.DataFrame]
        Per-event result frames.
    weight_history : pd.DataFrame or None
        One row per event showing each player's weight after updating.
        None when dynamic weights are disabled.
    """
    pids = [str(pid) for pid in player_params.keys()]
    baseline_weights = {pid: float(player_params[pid][3]) for pid in pids}
    current_weights = dict(baseline_weights)

    # Capture pre-season parameters before any nudging occurs
    pre_season_params = {
        pid: {
            "PreSeason_a":      float(player_params[pid][0]),
            "PreSeason_loc":    float(player_params[pid][1]),
            "PreSeason_scale":  float(player_params[pid][2]),
            "PreSeason_weight": float(player_params[pid][3]),
        }
        for pid in pids
    }

    if dynamic_weight_config is None:
        dynamic_weight_config = DynamicWeightConfig(enabled=False)

    season_points = {pid: 0.0 for pid in pids}
    season_round_scores = {pid: [] for pid in pids}   # accumulate every simulated round score
    event_results = []
    weight_snapshots = []
    rng = np.random.default_rng(seed)

    def run_event(tournament_type):
        cfg = tournament_type.value

        current_p = np.array([current_weights[pid] for pid in pids], dtype=float)
        current_p /= current_p.sum()

        field = rng.choice(
            pids,
            size=cfg.field_size,
            replace=False,
            p=current_p,
        ).tolist()

        scores_pre = sample_round_scores_for_players(player_params, CUT_AFTER_ROUND, field)
        totals_2r = scores_pre.sum(axis=1)

        # Record pre-cut round scores for every player in the field
        for i, pid in enumerate(field):
            season_round_scores[pid].extend(scores_pre[i, :].tolist())

        df2 = pd.DataFrame({
            PLAYER_ID_COL: field,
            TOTAL_2R_COL: totals_2r,
        })

        made_cut = apply_cut(df2, cfg.cut_rule)
        survivors = [pid for pid in field if pid in made_cut]

        remaining_rounds = ROUNDS_PER_EVENT - CUT_AFTER_ROUND
        scores_post = sample_round_scores_for_players(player_params, remaining_rounds, survivors)
        totals_post = scores_post.sum(axis=1)

        # Record post-cut round scores for survivors
        for i, pid in enumerate(survivors):
            season_round_scores[pid].extend(scores_post[i, :].tolist())

        surv_totals_2r = df2.set_index(PLAYER_ID_COL).loc[survivors, TOTAL_2R_COL].values
        final_totals = surv_totals_2r + totals_post

        results_cut = pd.DataFrame({
            "Player": survivors,
            "TotalScore": final_totals,
        })

        results_cut = assign_points_with_ties(
            results_cut,
            event_type=cfg.points_type,
            points_tables=POINTS_TABLES,
            score_col="TotalScore",
        )

        missed = [pid for pid in field if pid not in made_cut]
        if missed:
            missed_df = pd.DataFrame({
                "Player": missed,
                "TotalScore": np.nan,
                "FinalRank": np.nan,
                "Points": 0.0,
            })
            results = pd.concat([results_cut, missed_df], ignore_index=True)
        else:
            results = results_cut.copy()

        for row in results.itertuples(index=False):
            season_points[str(row.Player)] += float(row.Points)

        results["TournamentType"] = tournament_type.name
        return results

    for tournament_type in schedule:
        results = run_event(tournament_type)
        event_results.append(results)

        if dynamic_weight_config.enabled:
            current_weights = nudge_weights(
                current_weights,
                baseline_weights,
                results,
                dynamic_weight_config,
            )
            weight_snapshots.append(dict(current_weights))

    # Compute empirical moments from the rounds each player actually played this season
    post_season_fitted = {}
    for pid in pids:
        rounds = season_round_scores[pid]
        if len(rounds) >= 3:
            arr = np.array(rounds, dtype=float)
            ps_mean = float(np.mean(arr))
            ps_var  = float(np.var(arr, ddof=1))
            ps_skew = float(skew(arr, bias=False))
        else:
            ps_mean, ps_var, ps_skew = np.nan, np.nan, np.nan
        post_season_fitted[pid] = (ps_mean, ps_var, ps_skew)

    season_summary = (
        pd.DataFrame(
            [(pid, season_points[pid]) for pid in pids],
            columns=["Player", "SeasonPoints"],
        )
        .sort_values("SeasonPoints", ascending=False)
        .reset_index(drop=True)
    )
    season_summary["SeasonRank"] = np.arange(1, len(season_summary) + 1)

    # Merge pre-season parameters into the summary
    pre_season_df = pd.DataFrame.from_dict(pre_season_params, orient="index")
    pre_season_df.index.name = "Player"
    pre_season_df = pre_season_df.reset_index()

    season_summary = season_summary.merge(pre_season_df, on="Player", how="left")

    # Merge post-season weight (final value of current_weights after all nudging)
    post_season_df = pd.DataFrame(
        [(pid, current_weights[pid],
          post_season_fitted[pid][0],
          post_season_fitted[pid][1],
          post_season_fitted[pid][2],
          len(season_round_scores[pid])) for pid in pids],
        columns=["Player", "PostSeason_weight",
                 "PostSeason_mean", "PostSeason_var", "PostSeason_skew",
                 "PostSeason_rounds_played"],
    )
    season_summary = season_summary.merge(post_season_df, on="Player", how="left")

    # Reorder columns: identity, then pre-season params, then post-season params, then results
    col_order = [
        "Player",
        "PreSeason_a", "PreSeason_loc", "PreSeason_scale", "PreSeason_weight",
        "PostSeason_mean", "PostSeason_var", "PostSeason_skew", "PostSeason_weight",
        "PostSeason_rounds_played",
        "SeasonPoints", "SeasonRank",
    ]
    season_summary = season_summary[col_order]

    if output_csv_path is not None:
        season_summary.to_csv(output_csv_path, index=False)

    weight_history = None
    if weight_snapshots:
        weight_history = pd.DataFrame(weight_snapshots)
        weight_history.index.name = "event_index"

    return season_summary, event_results, weight_history
