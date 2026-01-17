import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import skewnorm, skew
from enum import Enum
import matplotlib.pyplot as plt

PI = np.pi

N_REGULAR_EVENTS = 20
N_ELEVATED_EVENTS = 5

ROUNDS_PER_EVENT = 4
CUT_AFTER_ROUND = 2

FIELD_SIZE_REGULAR = 156
FIELD_SIZE_ELEVATED = 70

ELEVATED_POINTS_MULTIPLIER = 1.0

BASE_POINTS_FIRST_REG = 500.0
BASE_POINTS_FIRST_ELEV = 700.0
BASE_POINTS_DECAY = 0.93

class EventType(Enum):
    REGULAR = "regular"
    ELEVATED = "elevated"

class CutRule(Enum):
    NONE = "none"
    TOP_65_TIES = "top65_ties"
    TOP_50_PLUS_10_SHOTS = "top50_plus_10shots"

CUT_TOP_65 = 65
CUT_TOP_50 = 50
CUT_PLUS_SHOTS = 10.0

PLAYER_ID_COL = "player_id"
TOTAL_2R_COL = "total_2r"

def compute_player_stats(
    csv_paths,
    player_col,
    value_col,
    min_avg_rounds=20,
    weight_power=1.0,
    weight_floor=0.0,
):
    """
    Compute per-player mean, variance, skew, and participation-based weights
    across multiple season CSVs.

    Key details
    -----------
    - AvgRounds = total rounds across all CSVs / number of CSVs
    - AvgEvents = average number of events per season
    - Weight is derived from AvgEvents and normalized to sum to 1

    Parameters
    ----------
    csv_paths : list
        List of CSV file paths (e.g., multiple seasons).
    player_col : str
        Column name that identifies the player.
    value_col : str
        Column name for the numeric value to analyze (e.g., score).
    min_avg_rounds : int
        Minimum average rounds per season required to keep a player.
    weight_power : float
        Exponent applied to AvgEvents when building weights.
        1.0 = literal participation, <1 flattens, >1 amplifies.
    weight_floor : float
        Minimum raw weight to assign before normalization (prevents zero-prob players).

    Returns
    -------
    pandas.DataFrame
        Columns:
        Player, AvgRounds, AvgEvents, Mean, Variance, Skew, Weight
    """
    if not csv_paths:
        raise ValueError("csv_paths cannot be empty.")

    frames = []
    for path in csv_paths:
        df = pd.read_csv(path)

        if player_col not in df.columns:
            raise ValueError(f"Missing column '{player_col}' in file: {path}")
        if value_col not in df.columns:
            raise ValueError(f"Missing column '{value_col}' in file: {path}")

        df = df[[player_col, value_col]].copy()
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
        df = df.dropna(subset=[player_col, value_col])

        # Tag season so we can count events per season
        df["_season"] = path
        frames.append(df)

    all_data = pd.concat(frames, ignore_index=True)
    num_seasons = len(csv_paths)

    # --- round-level stats ---
    grouped = all_data.groupby(player_col)[value_col]

    total_rounds = grouped.size()
    avg_rounds = total_rounds / float(num_seasons)

    # --- event-level participation ---
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
    }).reset_index()

    out = out.rename(columns={player_col: "Player"})

    # Filter by AVERAGE rounds per season
    out = out[out["AvgRounds"] >= float(min_avg_rounds)].copy()

    # --- build participation-based weights ---
    raw_weights = out["AvgEvents"].astype(float) ** float(weight_power)

    if weight_floor > 0.0:
        raw_weights = np.maximum(raw_weights, weight_floor)

    out["Weight"] = raw_weights / raw_weights.sum()

    # Sort: best players first (lower mean score), then higher participation
    out = out.sort_values(
        ["Mean", "AvgEvents"],
        ascending=[True, False]
    ).reset_index(drop=True)

    return out

def _skew_from_delta(delta, eps=1e-12):
    """
    Skewness of a skew-normal distribution as a function of delta.
    """
    base = 1.0 - (2.0 * delta**2) / PI

    if base <= eps:
        raise ZeroDivisionError(
            f"Invalid delta={delta}: denominator approaches zero in skew calculation."
        )

    num = ((4.0 - PI) / 2.0) * (delta * np.sqrt(2.0 / PI))**3
    den = base**1.5

    return num / den

def skewnorm_params_from_moments(mean, variance, skew, eps=1e-8):
    """
    Convert (mean, variance, skewness) into scipy.stats.skewnorm parameters.
    """
    if variance <= 0 or not np.isfinite(variance):
        raise ValueError(f"Invalid variance: {variance}")

    if not np.isfinite(skew):
        raise ValueError(f"Invalid skew: {skew}")

    if abs(skew) < eps:
        # symmetric → normal
        return 0.0, mean, np.sqrt(variance)

    delta_min, delta_max = -0.999, 0.999

    try:
        skew_min = _skew_from_delta(delta_min)
        skew_max = _skew_from_delta(delta_max)
    except ZeroDivisionError as e:
        raise RuntimeError(
            f"Skew-normal boundary failure for mean={mean}, var={variance}, skew={skew}"
        ) from e

    # The use of target skew here is somewhat contreversial, as there are scenarios where skew falls outside of
    # the bounds downstream. Another solution could be just to fall back to normal when outside skew bounds.
    target_skew = max(min(float(skew), skew_max), skew_min)

    def root_fn(delta):
        return _skew_from_delta(delta) - target_skew

    # The use of a search instead of a close formed solution comes down to the difference between population
    # vs sample sknewness. Further explanations of this differnce can be found on wikipedia or sci py documentation.
    try:
        delta = brentq(root_fn, delta_min, delta_max)
    except Exception as e:
        raise RuntimeError(
            f"Root-finding failed for mean={mean}, var={variance}, skew={skew}"
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

def build_player_generators(player_stats_df, id_col="Player", mean_col="Mean", var_col="Variance", skew_col="Skew", weight_col = "Weight"):
    """
    Returns dict: player_id -> (a, loc, scale, w)
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
    Draw round-by-round scores for multiple players.

    Parameters
    ----------
    player_params : dict
        Mapping: player_id -> (a, loc, scale)
    n_rounds : int
        Number of rounds to simulate.
    player_ids : list or array
        Player IDs to simulate (field for this tournament).

    Returns
    -------
    numpy.ndarray
        Array of shape (num_players, n_rounds),
        where each row corresponds to one player.
    """
    if player_ids == None:
        player_ids = player_params.keys()

    num_players = len(player_ids)
    scores = np.zeros((num_players, n_rounds))

    for i, pid in enumerate(player_ids):
        a, loc, scale, _ = player_params[pid]
        scores[i, :] = skewnorm.rvs(
            a,
            loc=loc,
            scale=scale,
            size=n_rounds
        )

    return scores

def simulate_and_compare_player(player_moments, player_params, player_id, n_rounds=200000, seed=123):
    rng = np.random.default_rng(seed)

    a, loc, scale, _ = player_params[player_id]
    x = skewnorm.rvs(a, loc=loc, scale=scale, size=n_rounds, random_state=rng)

    sim_mean = float(np.mean(x))
    sim_var = float(np.var(x, ddof=1))
    sim_skew = float(skew(x, bias=False))

    row = player_moments[player_moments["Player"] == player_id].iloc[0]

    target_mean = float(row["Mean"])
    target_var = float(row["Variance"])
    target_skew = float(row["Skew"])

    out = pd.DataFrame(
        {
            "moment": ["mean", "variance", "skew"],
            "target": [target_mean, target_var, target_skew],
            "simulated": [sim_mean, sim_var, sim_skew],
        }
    )
    out["diff"] = out["simulated"] - out["target"]
    out["pct_diff"] = out["diff"] / out["target"] * 100.0

    return out

def build_simple_points_table(n_finishers, first_points):
    pts = []
    p = float(first_points)
    for _ in range(n_finishers):
        pts.append(float(p))
        p = p * BASE_POINTS_DECAY
    return pts

POINTS_TABLE_REGULAR = build_simple_points_table(FIELD_SIZE_REGULAR, BASE_POINTS_FIRST_REG)
POINTS_TABLE_ELEVATED = build_simple_points_table(FIELD_SIZE_ELEVATED, BASE_POINTS_FIRST_ELEV)

def get_points_for_rank(event_type, rank, points_tables):
    """
    rank: 1 = winner, 2 = second, ...
    returns 0 if rank is out of range

    Parameters
    ----------
    event_type : EventType
    rank : int
    points_tables : dict
        Mapping: EventType -> list of points by rank (index 0 is rank 1)
        Example:
            {
                EventType.REGULAR: POINTS_TABLE_REGULAR,
                EventType.ELEVATED: POINTS_TABLE_ELEVATED,
            }
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


def apply_cut(scores_after_two_rounds, rule):
    """
    scores_after_two_rounds: DataFrame with columns [player_id, total_2r]
    Returns a set of player_ids who make the cut.

    Parameters
    ----------
    scores_after_two_rounds : pandas.DataFrame
    rule : CutRule
    """
    if rule == CutRule.NONE:
        return set(scores_after_two_rounds[PLAYER_ID_COL].tolist())

    df = scores_after_two_rounds.sort_values(TOTAL_2R_COL, ascending=True).reset_index(drop=True)

    if rule == CutRule.TOP_65_TIES:
        if len(df) <= CUT_TOP_65:
            return set(df[PLAYER_ID_COL].tolist())

        cut_idx = CUT_TOP_65 - 1
        cut_score = float(df.loc[cut_idx, TOTAL_2R_COL])

        return set(df.loc[df[TOTAL_2R_COL] <= cut_score, PLAYER_ID_COL].tolist())

    if rule == CutRule.TOP_50_PLUS_10_SHOTS:
        if len(df) <= CUT_TOP_50:
            return set(df[PLAYER_ID_COL].tolist())

        leader = float(df.loc[0, TOTAL_2R_COL])
        base_cut_idx = CUT_TOP_50 - 1
        base_cut = float(df.loc[base_cut_idx, TOTAL_2R_COL])

        cut_score = max(base_cut, leader + CUT_PLUS_SHOTS)

        return set(df.loc[df[TOTAL_2R_COL] <= cut_score, PLAYER_ID_COL].tolist())

    raise ValueError("Unknown cut rule: " + str(rule))

def simulate_season(player_params):
    # Stable ordering so weights align with pids
    pids = [str(pid) for pid in list(player_params.keys())]

    # Optional sanity check
    missing = [pid for pid in pids if pid not in player_params]
    if missing:
        raise KeyError(
            "Some pids are missing from player_params. Example missing: " + str(missing[:10])
        )

    # Weights assumed normalized and stored as 4th element
    weights = [float(player_params[pid][3]) for pid in pids]

    points_tables = {
        EventType.REGULAR: POINTS_TABLE_REGULAR,
        EventType.ELEVATED: POINTS_TABLE_ELEVATED,
    }

    season_points = {pid: 0.0 for pid in pids}
    event_results = []

    rng = np.random.default_rng()

    def run_event(event_type, field_size, cut_rule):
        field = rng.choice(pids, size=field_size, replace=False, p=weights).tolist()

        # --- simulate first two rounds ---
        scores_pre = sample_round_scores_for_players(player_params, CUT_AFTER_ROUND, field)
        totals_2r = scores_pre.sum(axis=1)

        df2 = pd.DataFrame({
            PLAYER_ID_COL: field,
            TOTAL_2R_COL: totals_2r,
        })

        made_cut = apply_cut(df2, cut_rule)
        survivors = [pid for pid in field if pid in made_cut]

        # --- simulate remaining rounds for survivors ---
        remaining_rounds = ROUNDS_PER_EVENT - CUT_AFTER_ROUND
        scores_post = sample_round_scores_for_players(player_params, remaining_rounds, survivors)
        totals_post = scores_post.sum(axis=1)

        # Build final results
        surv_totals_2r = df2.set_index(PLAYER_ID_COL).loc[survivors, TOTAL_2R_COL].values
        final_totals = surv_totals_2r + totals_post

        results_cut = pd.DataFrame({
            "Player": survivors,
            "TotalScore": final_totals,
        }).sort_values("TotalScore", ascending=True).reset_index(drop=True)

        results_cut["FinalRank"] = np.arange(1, len(results_cut) + 1)

        results_cut["Points"] = [
            get_points_for_rank(event_type, int(r), points_tables)
            for r in results_cut["FinalRank"]
        ]

        # Players missing the cut get 0 points
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
            results = results_cut

        # Update season totals
        for row in results.itertuples(index=False):
            season_points[str(row.Player)] += float(row.Points)

        return results

    # regular season
    for _ in range(N_REGULAR_EVENTS):
        res = run_event(EventType.REGULAR, FIELD_SIZE_REGULAR, CutRule.TOP_65_TIES)
        event_results.append(res)

    # elevated season
    for _ in range(N_ELEVATED_EVENTS):
        res = run_event(EventType.ELEVATED, FIELD_SIZE_ELEVATED, CutRule.TOP_50_PLUS_10_SHOTS)
        event_results.append(res)

    season_summary = pd.DataFrame(
        [(pid, season_points[pid]) for pid in pids],
        columns=["Player", "SeasonPoints"],
    ).sort_values("SeasonPoints", ascending=False).reset_index(drop=True)

    season_summary["SeasonRank"] = np.arange(1, len(season_summary) + 1)

    return season_summary, event_results

files = [
    "golf_data\yr2021.csv",
    "golf_data\yr2022.csv",
    "golf_data\yr2023.csv",
    "golf_data\yr2024.csv",
    "golf_data\yr2025.csv"
]

moments = compute_player_stats(
    csv_paths=files,
    player_col="player",
    value_col="score",
    min_avg_rounds=20,
    weight_floor=0.05
)

#print(moments.head(20))

player_params = build_player_generators(moments)

comparison = simulate_and_compare_player(moments, player_params, player_id="Scottie Scheffler", n_rounds=300000, seed=7)
print(comparison)

season = simulate_season(player_params)

print(season[0][:26])
