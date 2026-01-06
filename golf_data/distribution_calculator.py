import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import skewnorm, skew

PI = np.pi

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
    pids = []
    for _, row in player_stats_df.iterrows():
        pid = row["Player"]
        m = float(row[mean_col])
        v = float(row[var_col])
        s = float(row[skew_col])
        a, loc, scale = skewnorm_params_from_moments(m, v, s)
        params[pid] = (a, loc, scale)
        pids.append(pid)
    return params, pids

def sample_round_scores_for_players(player_params, player_ids, n_rounds, rng=None):
    """
    Draw round-by-round scores for multiple players.

    Parameters
    ----------
    player_params : dict
        Mapping: player_id -> (a, loc, scale)
    player_ids : list or array
        Player IDs to simulate (field for this tournament).
    n_rounds : int
        Number of rounds to simulate.
    rng : numpy.random.Generator, optional
        Random number generator.

    Returns
    -------
    numpy.ndarray
        Array of shape (num_players, n_rounds),
        where each row corresponds to one player.
    """
    if rng is None:
        rng = np.random.default_rng()

    num_players = len(player_ids)
    scores = np.zeros((num_players, n_rounds))

    for i, pid in enumerate(player_ids):
        a, loc, scale = player_params[pid]
        scores[i, :] = skewnorm.rvs(
            a,
            loc=loc,
            scale=scale,
            size=n_rounds,
            random_state=rng,
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

files = [
    "yr2021.csv",
    "yr2022.csv",
    "yr2023.csv",
    "yr2024.csv",
    "yr2025.csv"
]

moments = compute_player_stats(
    csv_paths=files,
    player_col="player",
    value_col="score",
    min_avg_rounds=20,
)

print(moments.head(20))

player_params, pids = build_player_generators(moments)

comparison = simulate_and_compare_player(moments, player_params, player_id="Scottie Scheffler", n_rounds=300000, seed=7)
print(comparison)
