"""data_loading.py.

Loads round-level score data from season CSVs and computes per-player
statistical moments and participation-based sampling weights.
"""

import numpy as np
import pandas as pd


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
