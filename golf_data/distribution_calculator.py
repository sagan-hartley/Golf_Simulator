import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import skewnorm, skew
import matplotlib.pyplot as plt

PI = np.pi

N_REGULAR_EVENTS = 20
N_ELEVATED_EVENTS = 5

ROUNDS_PER_EVENT = 4
CUT_AFTER_ROUND = 2

FIELD_SIZE_REGULAR = 156
FIELD_SIZE_ELEVATED = 70

CUT_RULE_REGULAR = "top65_ties"          # "top65_ties" or "none"
CUT_RULE_ELEVATED = "top50_plus_10shots" # "top50_plus_10shots" or "none"

ELEVATED_POINTS_MULTIPLIER = 1.0

BASE_POINTS_FIRST_REG = 500.0
BASE_POINTS_FIRST_ELEV = 700.0
BASE_POINTS_DECAY = 0.93

def compute_player_stats(csv_paths, player_col, value_col, min_avg_rounds=20):
    """
    Compute per-player mean, variance, and skew across multiple season CSVs.

    Key detail:
    - AvgRounds is total rounds across all CSVs divided by number of CSVs.
    - Filtering is done using AvgRounds >= min_avg_rounds.

    Parameters
    ----------
    csv_paths : list
        List of CSV file paths (e.g., 5 season files).
    player_col : str
        Column name that identifies the player.
    value_col : str
        Column name for the numeric value to analyze (e.g., score, strokes gained, etc.).
    min_avg_rounds : int, optional
        Minimum average rounds per year required to keep a player.

    Returns
    -------
    pandas.DataFrame
        Columns: Player, AvgRounds, Mean, Variance, Skew
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

        # Coerce the value column to numeric; non-numeric becomes NaN
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

        # Drop rows missing player id or value
        df = df.dropna(subset=[player_col, value_col])

        frames.append(df)

    all_data = pd.concat(frames, ignore_index=True)

    num_seasons = len(csv_paths)

    grouped = all_data.groupby(player_col)[value_col]

    total_rounds = grouped.size()
    avg_rounds = total_rounds / float(num_seasons)

    out = pd.DataFrame({
        "AvgRounds": avg_rounds,
        "Mean": grouped.mean(),
        "Variance": grouped.var(ddof=1),
        "Skew": grouped.skew(),
    }).reset_index()

    out = out.rename(columns={player_col: "Player"})

    # Filter by AVERAGE rounds per season
    out = out[out["AvgRounds"] >= float(min_avg_rounds)].copy()

    # Optional: sort so highest participation first
    out = out.sort_values(["Mean", "AvgRounds"], ascending=[True, False]).reset_index(drop=True)

    return out

def _skew_from_delta(delta):
    """
    Skewness of a skew-normal distribution as a function of delta.
    delta must be in (-1, 1).
    """
    num = ((4.0 - PI) / 2.0) * (delta * np.sqrt(2.0 / PI))**3
    den = (1.0 - (2.0 * delta**2) / PI)**1.5
    return num / den

def skewnorm_params_from_moments(mean, variance, skew):
    """
    Convert (mean, variance, skewness) into scipy.stats.skewnorm parameters.

    Returns
    -------
    a, loc, scale
    """
    # Defensive handling
    if variance <= 0 or np.isnan(variance):
        variance = 1e-6

    if np.isnan(skew) or abs(skew) < 1e-8:
        # Essentially symmetric → normal
        a = 0.0
        scale = np.sqrt(variance)
        loc = mean
        return a, loc, scale

    # Clip target skew to feasible range
    delta_min, delta_max = -0.9999, 0.9999
    skew_min = _skew_from_delta(delta_min)
    skew_max = _skew_from_delta(delta_max)

    target_skew = float(skew)
    target_skew = max(min(target_skew, skew_max), skew_min)

    # Root-finding function
    def root_fn(delta):
        return _skew_from_delta(delta) - target_skew

    # Solve for delta
    delta = brentq(root_fn, delta_min, delta_max)

    # Convert delta → skewnorm params
    a = delta / np.sqrt(1.0 - delta**2)
    scale = np.sqrt(variance / (1.0 - (2.0 * delta**2) / PI))
    loc = mean - scale * delta * np.sqrt(2.0 / PI)

    return a, loc, scale

def build_player_generators(player_stats_df, mean_col="Mean", var_col="Variance", skew_col="Skew"):
    """
    Returns dict: player_id -> (a, loc, scale)
    """
    params = {}
    for _, row in player_stats_df.iterrows():
        pid = row["Player"]
        m = float(row[mean_col])
        v = float(row[var_col])
        s = float(row[skew_col])
        a, loc, scale = skewnorm_params_from_moments(m, v, s)
        params[pid] = (a, loc, scale)
        
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
        a, loc, scale = player_params[pid]
        scores[i, :] = skewnorm.rvs(
            a,
            loc=loc,
            scale=scale,
            size=n_rounds
        )

    return scores

def simulate_and_compare_player(player_moments, player_params, player_id, n_rounds=200000, seed=123):
    rng = np.random.default_rng(seed)

    a, loc, scale = player_params[player_id]
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

def get_points_for_rank(event_type, rank):
    """
    rank: 1 = winner, 2 = second, ...
    returns 0 if rank is out of range
    """
    if rank < 1:
        raise ValueError("rank must be >= 1")

    if event_type == "regular":
        table = POINTS_TABLE_REGULAR
    elif event_type == "elevated":
        table = POINTS_TABLE_ELEVATED
    else:
        raise ValueError("Unknown event_type: " + str(event_type))

    idx = rank - 1
    if idx >= len(table):
        return 0.0
    return float(table[idx])

def apply_cut(scores_after_two_rounds, rule):
    """
    scores_after_two_rounds: DataFrame with columns [player_id, total_2r]
    Returns a set of player_ids who make the cut.
    """
    if rule == "none":
        return set(scores_after_two_rounds["player_id"].tolist())

    df = scores_after_two_rounds.sort_values("total_2r", ascending=True).reset_index(drop=True)

    if rule == "top65_ties":
        if len(df) <= 65:
            return set(df["player_id"].tolist())
        cut_score = float(df.loc[64, "total_2r"])
        return set(df.loc[df["total_2r"] <= cut_score, "player_id"].tolist())

    if rule == "top50_plus_10shots":
        if len(df) <= 50:
            return set(df["player_id"].tolist())
        leader = float(df.loc[0, "total_2r"])
        base_cut = float(df.loc[49, "total_2r"])
        cut_score = max(base_cut, leader + 10.0)
        return set(df.loc[df["total_2r"] <= cut_score, "player_id"].tolist())

    raise ValueError("Unknown cut rule: " + str(rule))

def simulate_season(player_params):
    pids = player_params.keys()

    # Normalize PID types ONCE
    pids = [str(pid) for pid in pids]

    missing = [pid for pid in pids if pid not in player_params]
    if missing:
        raise KeyError(
            "Some pids are missing from player_params. Example missing: " + str(missing[:10])
        )

    season_points = {pid: 0.0 for pid in pids}
    event_results = []

    def run_event(event_type, field_size, cut_rule):
        #field = rng.choice(pids, size=field_size, replace=False).tolist()
        field = pids

        # --- simulate first two rounds ---
        scores_pre = sample_round_scores_for_players(player_params, CUT_AFTER_ROUND, field)
        totals_2r = scores_pre.sum(axis=1)

        df2 = pd.DataFrame({
            "player_id": field,
            "total_2r": totals_2r,
        })

        made_cut = apply_cut(df2, cut_rule)
        survivors = [pid for pid in field if pid in made_cut]

        # --- simulate remaining rounds for survivors ---
        remaining_rounds = ROUNDS_PER_EVENT - CUT_AFTER_ROUND
        scores_post = sample_round_scores_for_players(player_params, remaining_rounds, survivors)
        totals_post = scores_post.sum(axis=1)

        # Build final results
        surv_totals_2r = df2.set_index("player_id").loc[survivors, "total_2r"].values
        final_totals = surv_totals_2r + totals_post

        results_cut = pd.DataFrame({
            "Player": survivors,
            "TotalScore": final_totals,
        }).sort_values("TotalScore", ascending=True).reset_index(drop=True)

        results_cut["FinalRank"] = np.arange(1, len(results_cut) + 1)
        results_cut["Points"] = [get_points_for_rank(event_type, int(r)) for r in results_cut["FinalRank"]]

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
        res = run_event("regular", FIELD_SIZE_REGULAR, CUT_RULE_REGULAR)
        event_results.append(res)

    # elevated season
    for _ in range(N_ELEVATED_EVENTS):
        res = run_event("elevated", FIELD_SIZE_ELEVATED, CUT_RULE_ELEVATED)
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
)

#print(moments.head(20))

player_params = build_player_generators(moments)

comparison = simulate_and_compare_player(moments, player_params, player_id="Scottie Scheffler", n_rounds=300000, seed=7)
print(comparison)

season = simulate_season(player_params)

print(season[0][:26])
