"""
simulation.py
-------------
All simulation logic: data loading, player stats, skew-normal fitting,
season simulation, dynamic weights, and diagnostics.

All configuration inputs are imported from config.py.
"""

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import skewnorm, skew
from typing import Optional
import matplotlib.pyplot as plt

from config import (
    PI,
    ROUNDS_PER_EVENT,
    CUT_AFTER_ROUND,
    CUT_TOP_50,
    CUT_TOP_60,
    CUT_TOP_65,
    CUT_TOP_70,
    CUT_PLUS_SHOTS,
    PLAYER_ID_COL,
    TOTAL_2R_COL,
    DynamicWeightConfig,
    EventType,
    CutRule,
    TournamentType,
    POINTS_TABLES,
    SEASON_SCHEDULE,
)

# ── Data loading ───────────────────────────────────────────────────────────────

def load_and_standardize_round_data(csv_paths, player_col, value_col):
    """
    Load round-level data from multiple CSV files and standardize schema.

    Parameters
    ----------
    csv_paths : list
        List of CSV file paths, each representing one season.
    player_col : str
        Column name identifying the player.
    value_col : str
        Column name containing the numeric round-level value.

    Returns
    -------
    pandas.DataFrame
        Standardized DataFrame with columns:
        - player_col
        - value_col
        - "_season"
    """
    if not csv_paths:
        raise ValueError("csv_paths cannot be empty.")

    frames = []
    for path in csv_paths:
        df = pd.read_csv(path)

        if player_col not in df.columns:
            raise ValueError("Missing column '{}' in file: {}".format(player_col, path))
        if value_col not in df.columns:
            raise ValueError("Missing column '{}' in file: {}".format(value_col, path))

        df = df[[player_col, value_col]].copy()
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
        df = df.dropna(subset=[player_col, value_col])
        df["_season"] = path
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def build_participation_weights(avg_events, weight_power=1.0, weight_floor=0.0):
    """
    Construct normalized player selection weights from participation data.

    Parameters
    ----------
    avg_events : pandas.Series
        Average number of events played per season for each player.
    weight_power : float
        Exponent applied to AvgEvents before normalization.
    weight_floor : float
        Minimum raw weight applied before normalization.

    Returns
    -------
    pandas.Series
        Normalized weights summing to 1.0.
    """
    raw = avg_events.astype(float) ** float(weight_power)

    if weight_floor > 0.0:
        raw = np.maximum(raw, float(weight_floor))

    total = float(raw.sum())
    if total <= 0.0:
        raise ValueError("Weight normalization failed: sum of raw weights is <= 0")

    return raw / total


def compute_player_stats(
    csv_paths,
    player_col,
    value_col,
    min_avg_rounds=20,
    weight_power=1.0,
    weight_floor=0.03,
):
    """
    Compute per-player statistical moments and participation-based weights.

    Parameters
    ----------
    csv_paths : list
    player_col : str
    value_col : str
    min_avg_rounds : int
    weight_power : float
    weight_floor : float

    Returns
    -------
    pandas.DataFrame
        Columns: Player, AvgRounds, AvgEvents, Mean, Variance, Skew, Weight
    """
    all_data = load_and_standardize_round_data(csv_paths, player_col, value_col)
    num_seasons = len(csv_paths)

    grouped = all_data.groupby(player_col)[value_col]
    total_rounds = grouped.size()
    avg_rounds = total_rounds / float(num_seasons)

    events_per_season = (
        all_data
        .groupby(["_season", player_col])
        .size()
        .groupby(player_col)
        .mean()
    )

    out = pd.DataFrame({
        "AvgRounds": avg_rounds,
        "AvgEvents": events_per_season,
        "Mean": grouped.mean(),
        "Variance": grouped.var(ddof=1),
        "Skew": grouped.skew(),
    }).reset_index().rename(columns={player_col: "Player"})

    out = out[out["AvgRounds"] >= float(min_avg_rounds)].copy()

    out["Weight"] = build_participation_weights(
        out["AvgEvents"],
        weight_power=weight_power,
        weight_floor=weight_floor,
    )

    out = out.sort_values(
        ["Mean", "AvgEvents"],
        ascending=[True, False]
    ).reset_index(drop=True)

    return out

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


def sample_round_scores_for_players(player_params, n_rounds, player_ids=None):
    """
    Draw round-by-round integer scores for multiple players.

    Parameters
    ----------
    player_params : dict
    n_rounds : int
    player_ids : list or None

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
        raw_scores = skewnorm.rvs(a, loc=loc, scale=scale, size=n_rounds)
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

# ── Points helpers ─────────────────────────────────────────────────────────────

def get_points_for_rank(event_type, rank, points_tables):
    """
    Return points for a given rank in a given event type.

    Parameters
    ----------
    event_type : EventType
    rank : int
    points_tables : dict

    Returns
    -------
    float
    """
    if rank < 1:
        raise ValueError("rank must be >= 1")
    if event_type not in points_tables:
        raise ValueError("Unknown event_type: " + str(event_type))

    table = points_tables[event_type]
    idx = rank - 1
    if idx >= len(table):
        return 0.0
    return float(table[idx])


def assign_points_with_ties(results_df, event_type, points_tables, score_col="TotalScore"):
    """
    Assign FinalRank and Points using PGA-style tie averaging.

    Parameters
    ----------
    results_df : pd.DataFrame
    event_type : EventType
    points_tables : dict
    score_col : str

    Returns
    -------
    pd.DataFrame
    """
    table = points_tables[event_type]
    df = results_df.sort_values(score_col, ascending=True).reset_index(drop=True).copy()

    df["_pos0"] = np.arange(len(df))
    df["FinalRank"] = df.groupby(score_col)["_pos0"].transform("min") + 1

    grp_min = df.groupby(score_col)["_pos0"].transform("min")
    grp_max = df.groupby(score_col)["_pos0"].transform("max")

    points = []
    for start, end in zip(grp_min, grp_max):
        pts = [float(table[pos]) if pos < len(table) else 0.0
               for pos in range(int(start), int(end) + 1)]
        points.append(float(np.mean(pts)))

    df["Points"] = points
    return df.drop(columns=["_pos0"])


def apply_cut(scores_after_two_rounds, rule):
    """
    Determine which players survive the cut.

    Parameters
    ----------
    scores_after_two_rounds : pd.DataFrame
        Columns: [PLAYER_ID_COL, TOTAL_2R_COL]
    rule : CutRule

    Returns
    -------
    set
        Player IDs who made the cut.
    """
    if rule == CutRule.NONE:
        return set(scores_after_two_rounds[PLAYER_ID_COL].tolist())

    df = scores_after_two_rounds.sort_values(TOTAL_2R_COL, ascending=True).reset_index(drop=True)

    top_n_map = {
        CutRule.TOP_50_TIES: CUT_TOP_50,
        CutRule.TOP_60_TIES: CUT_TOP_60,
        CutRule.TOP_65_TIES: CUT_TOP_65,
        CutRule.TOP_70_TIES: CUT_TOP_70,
    }

    if rule in top_n_map:
        n = top_n_map[rule]
        if len(df) <= n:
            return set(df[PLAYER_ID_COL].tolist())
        cut_score = float(df.loc[n - 1, TOTAL_2R_COL])
        return set(df.loc[df[TOTAL_2R_COL] <= cut_score, PLAYER_ID_COL].tolist())

    if rule == CutRule.TOP_50_PLUS_10_SHOTS:
        n = CUT_TOP_50
        if len(df) <= n:
            return set(df[PLAYER_ID_COL].tolist())
        leader = float(df.loc[0, TOTAL_2R_COL])
        base_cut = float(df.loc[n - 1, TOTAL_2R_COL])
        cut_score = max(base_cut, leader + CUT_PLUS_SHOTS)
        return set(df.loc[df[TOTAL_2R_COL] <= cut_score, PLAYER_ID_COL].tolist())

    raise ValueError("Unknown cut rule: " + str(rule))

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

    # Reorder columns: player identity first, then pre-season params, then post-season params, then season results
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


# ── Monte Carlo wrapper ────────────────────────────────────────────────────────

def run_n_simulations(
    player_params: dict,
    schedule: list,
    n: int = 100,
    base_seed: int = 0,
    dynamic_weight_config=None,
    output_csv_path=None,
) -> "pd.DataFrame":
    """
    Run n independent season simulations and summarise finish-position probabilities.

    Each simulation uses a distinct seed (base_seed + i) so results are
    independent but reproducible.

    Parameters
    ----------
    player_params : dict
        pid -> (a, loc, scale, weight)
    schedule : list[TournamentType]
    n : int
        Number of simulations to run.
    base_seed : int
        Seeds run from base_seed to base_seed + n - 1.
    dynamic_weight_config : DynamicWeightConfig or None
    output_csv_path : str or None
        If provided, the summary DataFrame is written to this path.

    Returns
    -------
    pd.DataFrame
        One row per player, sorted by Win_pct descending. Columns:
        Player, Win_pct, Top10_pct, Top20_pct, Top50_pct, Avg_rank
    """
    import pandas as pd

    pids = list(player_params.keys())
    counts = {pid: {"win": 0, "top10": 0, "top20": 0, "top50": 0, "rank_sum": 0}
              for pid in pids}

    for i in range(n):
        season_summary, _, _ = simulate_season(
            player_params,
            schedule,
            seed=base_seed + i,
            dynamic_weight_config=dynamic_weight_config,
        )

        rank_lookup = season_summary.set_index("Player")["SeasonRank"].to_dict()

        for pid in pids:
            rank = rank_lookup.get(pid)
            if rank is None:
                continue
            counts[pid]["rank_sum"] += rank
            if rank == 1:
                counts[pid]["win"]   += 1
            if rank <= 10:
                counts[pid]["top10"] += 1
            if rank <= 20:
                counts[pid]["top20"] += 1
            if rank <= 50:
                counts[pid]["top50"] += 1

    rows = []
    for pid in pids:
        c = counts[pid]
        rows.append({
            "Player":    pid,
            "Win_pct":   round(c["win"]   / n * 100, 2),
            "Top10_pct": round(c["top10"] / n * 100, 2),
            "Top20_pct": round(c["top20"] / n * 100, 2),
            "Top50_pct": round(c["top50"] / n * 100, 2),
            "Avg_rank":  round(c["rank_sum"] / n, 2),
        })

    results = (
        pd.DataFrame(rows)
        .sort_values("Win_pct", ascending=False)
        .reset_index(drop=True)
    )

    if output_csv_path is not None:
        results.to_csv(output_csv_path, index=False)

    return results

# ── Diagnostics ────────────────────────────────────────────────────────────────

def plot_weight_trajectory(weight_history: pd.DataFrame, player_id: str,
                            baseline_weights: dict):
    """
    Plot how a single player's participation weight evolved across the season.

    Parameters
    ----------
    weight_history : pd.DataFrame
        Returned by simulate_season when dynamic weights are enabled.
    player_id : str
    baseline_weights : dict
    """
    if player_id not in weight_history.columns:
        print(f"Player '{player_id}' not found in weight history.")
        return

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(weight_history.index, weight_history[player_id],
            marker="o", label="dynamic weight")
    ax.axhline(baseline_weights[player_id], linestyle="--",
               color="grey", label="baseline weight")
    ax.set_title(f"Participation weight over season — {player_id}")
    ax.set_xlabel("Event index")
    ax.set_ylabel("Weight (normalised)")
    ax.legend()
    plt.tight_layout()
    plt.show()


def compare_appearance_counts(events_a: list, events_b: list,
                               label_a="static", label_b="dynamic",
                               top_n=20) -> pd.DataFrame:
    """
    Compare how many events each player appeared in across two simulations.

    Parameters
    ----------
    events_a : list[pd.DataFrame]
    events_b : list[pd.DataFrame]
    label_a : str
    label_b : str
    top_n : int

    Returns
    -------
    pd.DataFrame
    """
    def count(event_list):
        return (
            pd.concat([df[["Player"]] for df in event_list], ignore_index=True)
            ["Player"].value_counts()
        )

    df = pd.DataFrame({
        label_a: count(events_a),
        label_b: count(events_b),
    }).fillna(0).astype(int)

    df["diff"] = df[label_b] - df[label_a]
    return df.sort_values("diff", ascending=False).head(top_n)

# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    files = [
        "golf_data/yr2021.csv",
        "golf_data/yr2022.csv",
        "golf_data/yr2023.csv",
        "golf_data/yr2024.csv",
        "golf_data/yr2025.csv",
    ]

    moments = compute_player_stats(
        csv_paths=files,
        player_col="player",
        value_col="score",
        min_avg_rounds=20,
        weight_floor=0.05,
    )

    player_params = build_player_generators(moments)

    # -- moment check --
    comparison = simulate_and_compare_player(
        moments, player_params, player_id="Scottie Scheffler",
        n_rounds=300000, seed=7,
    )
    #print(comparison)

    # -- static season --
    season_static, events_static, _ = simulate_season(
        player_params,
        SEASON_SCHEDULE,
        seed=123,
    )
    #print("\n=== Static weights ===")
    #print(season_static.head(26))

    # -- dynamic season --
    dw_cfg = DynamicWeightConfig(
        enabled=True,
        nudge_amount=0.05,
        top_pct=0.25,
        bot_pct=0.25,
        min_weight=0.001,
        max_weight_multiplier=2.0,
    )

    season_dynamic, events_dynamic, weight_history = simulate_season(
        player_params,
        SEASON_SCHEDULE,
        seed=123,
        dynamic_weight_config=dw_cfg,
    )
    #print("\n=== Dynamic weights ===")
    #print(season_dynamic.head(26))

    # -- weight trajectory plot --
    baseline_weights = {pid: float(player_params[pid][3]) for pid in player_params}
    #plot_weight_trajectory(weight_history, "Scottie Scheffler", baseline_weights)

    # -- appearance count comparison --
    #print("\n=== Appearance count comparison ===")
    #print(compare_appearance_counts(events_static, events_dynamic))

    mc_results = run_n_simulations(
        player_params,
        SEASON_SCHEDULE,
        n=10,
        base_seed=0,
        dynamic_weight_config=dw_cfg,
        output_csv_path="mc_results.csv",
        )   
    print(mc_results.head(20))