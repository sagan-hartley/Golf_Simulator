import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import skewnorm, skew
from enum import Enum
from dataclasses import dataclass
import matplotlib.pyplot as plt

PI = np.pi

N_REGULAR_EVENTS = 20
N_ELEVATED_EVENTS = 5

ROUNDS_PER_EVENT = 4
CUT_AFTER_ROUND = 2

ELEVATED_POINTS_MULTIPLIER = 1.0

CUT_TOP_70 = 70
CUT_TOP_65 = 65
CUT_TOP_60 = 60
CUT_TOP_50 = 50
CUT_PLUS_SHOTS = 10.0

class EventType(Enum):
    REGULAR = "regular"
    SIGNATURE = "signature"
    MAJOR_PLAYERS = "major_players"
    ADDITIONAL = "additional"
    PLAYOFFS_2026_APPROX_750 = "playoffs_2026_approx_750"
    ZURICH_TEAM_EACH_PLAYER = "zurich_team_each_player" # code not worked to simulate this yet

class CutRule(Enum):
    NONE = "none"
    TOP_70_TIES = "top70_ties"
    TOP_65_TIES = "top65_ties"
    TOP_60_TIES = "top60_ties"
    TOP_50_TIES = "top50_ties"
    TOP_50_PLUS_10_SHOTS = "top50_plus_10shots"

@dataclass()
class TournamentConfig:
    points_type: EventType
    cut_rule: CutRule
    field_size: int

class TournamentType(Enum):
    REGULAR = TournamentConfig(
        points_type=EventType.REGULAR,
        cut_rule=CutRule.TOP_65_TIES,
        field_size=156,
    )

    SIGNATURE_NO_CUT = TournamentConfig(
        points_type=EventType.SIGNATURE,
        cut_rule=CutRule.NONE,
        field_size=70,
    )

    SIGNATURE_CUT = TournamentConfig(
        points_type=EventType.SIGNATURE,
        cut_rule=CutRule.TOP_50_PLUS_10_SHOTS,
        field_size=70,
    )

    MAJOR_MASTERS = TournamentConfig(
        points_type=EventType.MAJOR_PLAYERS,
        cut_rule=CutRule.TOP_50_TIES,
        field_size=156,
    )

    MAJOR_US_OPEN = TournamentConfig(
        points_type=EventType.MAJOR_PLAYERS,
        cut_rule=CutRule.TOP_60_TIES,
        field_size=156,
    )

    MAJOR_PGA = TournamentConfig(
        points_type=EventType.MAJOR_PLAYERS,
        cut_rule=CutRule.TOP_70_TIES,
        field_size=156,
    )

    MAJOR_OPEN = TournamentConfig(
        points_type=EventType.MAJOR_PLAYERS,
        cut_rule=CutRule.TOP_70_TIES,
        field_size=156,
    )

    PLAYERS = TournamentConfig(
        points_type=EventType.MAJOR_PLAYERS,
        cut_rule=CutRule.TOP_65_TIES,
        field_size=156,
    )

    PLAYOFF = TournamentConfig(
        points_type=EventType.MAJOR_PLAYERS,  # your current simplification
        cut_rule=CutRule.NONE,
        field_size=70,
    )

PLAYER_ID_COL = "player_id"
TOTAL_2R_COL = "total_2r"

# ---- Regular / Signature / Majors+Players / Additional tables (positions 1..85) ----

POINTS_TABLE_REGULAR_500 = [
    500, 300, 190, 135, 110, 100, 90, 85, 80, 75,
    70, 65, 60, 57, 55, 53, 51, 49, 47, 45,
    43, 41, 39, 37, 35.5, 34, 32.5, 31, 29.5, 28,
    26.5, 25, 23.5, 22, 21, 20, 19, 18, 17, 16,
    15, 14, 13, 12, 11, 10.5, 10, 9.5, 9, 8.5,
    8, 7.5, 7, 6.5, 6, 5.8, 5.6, 5.4, 5.2, 5.0,
    4.8, 4.6, 4.4, 4.2, 4.0, 3.8, 3.6, 3.4, 3.2, 3.0,
    2.9, 2.8, 2.7, 2.6, 2.5, 2.4, 2.3, 2.2, 2.1, 2.0,
    1.9, 1.8, 1.7, 1.6, 1.5
]

POINTS_TABLE_MAJOR_PLAYERS_750 = [
    750, 500, 350, 325, 300, 270, 250, 225, 200, 175,
    155, 135, 115, 105, 95, 85, 75, 70, 65, 60,
    55, 53, 51, 49, 47, 45, 43, 41, 39, 37,
    35, 33, 31, 29, 27, 26, 25, 24, 23, 22,
    21, 20.25, 19.5, 18.75, 18, 17.25, 16.5, 15.75, 15, 14.25,
    13.5, 13, 12.5, 12, 11.5, 11, 10.5, 10, 9.5, 9,
    8.5, 8.25, 8, 7.75, 7.5, 7.25, 7, 6.75, 6.5, 6.25,
    6, 5.75, 5.5, 5.25, 5, 4.75, 4.5, 4.25, 4, 3.75,
    3.5, 3.25, 3, 2.75, 2.5
]

POINTS_TABLE_SIGNATURE_700 = [
    700, 400, 350, 325, 300, 275, 225, 200, 175, 150,
    130, 120, 110, 100, 90, 80, 70, 65, 60, 55,
    50, 48, 46, 44, 42, 40, 38, 36, 34, 32.5,
    31, 29.5, 28, 26.5, 25, 24, 23, 22, 21, 20.25,
    19.5, 18.75, 18, 17.25, 16.5, 15.75, 15, 14.25, 13.5, 13,
    12.5, 12, 11.5, 11, 10.5, 10, 9.5, 9, 8.5, 8.25,
    8, 7.75, 7.5, 7.25, 7, 6.75, 6.5, 6.25, 6, 5.75,
    5.5, 5.25, 5, 4.75, 4.5, 4.25, 4, 3.75, 3.5, 3.25,
    3, 2.75, 2.5, 2.25, 2
]

POINTS_TABLE_ADDITIONAL_300 = [
    300, 165, 105, 80, 65, 60, 55, 50, 45, 40,
    37.5, 35.0, 32.5, 31.0, 30.5, 30.0, 29.5, 29.0, 28.5, 28.0,
    26.76, 25.51, 24.27, 23.02, 22.09, 21.16, 20.22, 19.29, 18.36, 17.42,
    16.49, 15.56, 14.62, 13.69, 13.07, 12.44, 11.82, 11.2, 10.58, 9.96,
    9.33, 8.71, 8.09, 7.47, 6.84, 6.53, 6.22, 5.91, 5.6, 5.29,
    4.98, 4.67, 4.36, 4.04, 3.73, 3.61, 3.48, 3.36, 3.24, 3.11,
    2.99, 2.86, 2.74, 2.61, 2.49, 2.36, 2.24, 2.12, 1.99, 1.87,
    1.8, 1.74, 1.68, 1.62, 1.56, 1.49, 1.43, 1.37, 1.31, 1.24,
    1.18, 1.12, 1.06, 1.00, 0.93
]

# ---- Zurich special case ----
# Winner points are 400 *per player* (team event).
# There isn't a simple single-player “position table” that matches standard events because payout/points are computed on
# a team basis; most simulators treat this as its own custom scoring function.
ZURICH_WINNER_POINTS_EACH_PLAYER = 400.0

POINTS_TABLES = {
    EventType.REGULAR: POINTS_TABLE_REGULAR_500,
    EventType.SIGNATURE: POINTS_TABLE_SIGNATURE_700,
    EventType.MAJOR_PLAYERS: POINTS_TABLE_MAJOR_PLAYERS_750,
    EventType.ADDITIONAL: POINTS_TABLE_ADDITIONAL_300,
    EventType.PLAYOFFS_2026_APPROX_750: POINTS_TABLE_MAJOR_PLAYERS_750
}

SEASON_SCHEDULE = [
    # Early season / West Coast swing (mostly regular)
    TournamentType.REGULAR,  # Sony
    TournamentType.REGULAR,  # AmEx
    TournamentType.REGULAR,  # Farmers
    TournamentType.REGULAR,  # WM Phoenix

    # Signatures start
    TournamentType.SIGNATURE_NO_CUT,  # Pebble (Signature, no cut)
    TournamentType.SIGNATURE_CUT,     # Genesis (Signature, cut rule)
    TournamentType.REGULAR,           # (e.g., Cognizant / regular filler)
    TournamentType.SIGNATURE_CUT,     # Arnold Palmer (Signature, cut rule)

    # Spring
    TournamentType.PLAYERS,           # THE PLAYERS
    TournamentType.REGULAR,           # Valspar
    TournamentType.REGULAR,           # Houston
    TournamentType.REGULAR,           # Valero
    TournamentType.MAJOR_MASTERS,     # Masters
    TournamentType.SIGNATURE_NO_CUT,  # RBC Heritage (Signature, no cut)

    # Mid-season signature block
    TournamentType.SIGNATURE_NO_CUT,  # (e.g., new Doral Signature)
    TournamentType.SIGNATURE_NO_CUT,  # Truist (Signature, no cut)
    TournamentType.MAJOR_PGA,         # PGA Championship

    # Summer into majors + Signature
    TournamentType.REGULAR,           # Byron Nelson
    TournamentType.REGULAR,           # Charles Schwab
    TournamentType.SIGNATURE_CUT,     # Memorial (Signature, cut rule)
    TournamentType.REGULAR,           # Canadian Open
    TournamentType.MAJOR_US_OPEN,     # U.S. Open
    TournamentType.SIGNATURE_NO_CUT,  # Travelers (Signature, no cut)

    # Late season / lead-in to playoffs
    TournamentType.REGULAR,           # John Deere
    TournamentType.REGULAR,           # Scottish Open
    TournamentType.MAJOR_OPEN,        # The Open Championship
    TournamentType.REGULAR,           # 3M
    TournamentType.REGULAR,           # Rocket
    TournamentType.REGULAR,           # Wyndham (regular season finale)

    # FedExCup Playoffs (simplified: no cut, 70 field)
    TournamentType.PLAYOFF,           # FedEx St. Jude (70)
    TournamentType.PLAYOFF,           # BMW (50)  -> will refine later
    TournamentType.PLAYOFF,           # TOUR Championship (30) -> will refine later
]

def load_and_standardize_round_data(csv_paths, player_col, value_col):
    """
    Load round-level data from multiple CSV files and standardize schema.

    This function is intentionally limited to:
    - reading CSVs
    - validating required columns
    - coercing numeric values
    - tagging each row with a season identifier

    It performs NO aggregation or filtering beyond basic cleaning,
    making it easy to unit-test in isolation.

    Parameters
    ----------
    csv_paths : list
        List of CSV file paths, each representing one season.
    player_col : str
        Column name identifying the player (e.g., player name or ID).
    value_col : str
        Column name containing the numeric round-level value
        (e.g., score, strokes gained).

    Returns
    -------
    pandas.DataFrame
        Standardized DataFrame with columns:
        - player_col
        - value_col
        - "_season" (identifier for the source CSV)
    """
    if not csv_paths:
        raise ValueError("csv_paths cannot be empty.")

    frames = []

    for path in csv_paths:
        # Load CSV
        df = pd.read_csv(path)

        # Validate schema early so errors are explicit
        if player_col not in df.columns:
            raise ValueError(
                "Missing column '{}' in file: {}".format(player_col, path)
            )
        if value_col not in df.columns:
            raise ValueError(
                "Missing column '{}' in file: {}".format(value_col, path)
            )

        # Keep only the columns we care about
        df = df[[player_col, value_col]].copy()

        # Force numeric values; invalid entries become NaN
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

        # Drop rows with missing player IDs or values
        df = df.dropna(subset=[player_col, value_col])

        # Tag rows with season identifier
        # (path is sufficient; could be replaced with year later)
        df["_season"] = path

        frames.append(df)

    # Combine all seasons into one long DataFrame
    return pd.concat(frames, ignore_index=True)

def build_participation_weights(avg_events, weight_power=1.0, weight_floor=0.0):
    """
    Construct normalized player selection weights from participation data.

    This function converts average event participation into sampling weights
    used when selecting tournament fields.

    Design goals:
    - Higher participation ⇒ higher probability of selection
    - Optional power transform to flatten or amplify differences
    - Optional floor to prevent players from having zero probability
    - Guaranteed normalization (sum of weights = 1)

    Parameters
    ----------
    avg_events : pandas.Series
        Average number of events played per season for each player.
    weight_power : float, optional
        Exponent applied to AvgEvents before normalization.
        - 1.0 = linear
        - <1.0 = flatten differences
        - >1.0 = emphasize frequent players
    weight_floor : float, optional
        Minimum raw weight applied before normalization.
        Useful to prevent rare players from being impossible to select.

    Returns
    -------
    pandas.Series
        Normalized weights summing to 1.0, aligned with avg_events index.
    """
    # Convert participation to raw weights
    raw = avg_events.astype(float) ** float(weight_power)

    # Enforce a minimum weight if requested
    if weight_floor > 0.0:
        raw = np.maximum(raw, float(weight_floor))

    # Normalize so weights sum to 1 (required by np.random.choice)
    total = float(raw.sum())
    if total <= 0.0:
        raise ValueError(
            "Weight normalization failed: sum of raw weights is <= 0"
        )

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
    Compute per-player statistical moments and participation-based weights
    across multiple seasons of round-level data.

    This function orchestrates the full pipeline:
    1) Load and standardize raw round data
    2) Aggregate round-level statistics
    3) Compute average rounds and events per season
    4) Filter out players with insufficient data
    5) Build normalized participation weights

    Parameters
    ----------
    csv_paths : list
        List of CSV file paths (one per season).
    player_col : str
        Column identifying the player.
    value_col : str
        Column containing the numeric round-level value.
    min_avg_rounds : int, optional
        Minimum average number of rounds per season required to
        include a player in the output.
    weight_power : float, optional
        Exponent applied to average event participation when building weights.
    weight_floor : float, optional
        Minimum raw weight before normalization.

    Returns
    -------
    pandas.DataFrame
        Columns:
        - Player
        - AvgRounds
        - AvgEvents
        - Mean
        - Variance
        - Skew
        - Weight

        Sorted by increasing Mean (better players first), then AvgEvents.
    """
    # Load and clean raw round-level data
    all_data = load_and_standardize_round_data(
        csv_paths, player_col, value_col
    )

    num_seasons = len(csv_paths)

    # --- round-level aggregation ---
    grouped = all_data.groupby(player_col)[value_col]

    # Total rounds played across all seasons
    total_rounds = grouped.size()

    # Average rounds per season
    avg_rounds = total_rounds / float(num_seasons)

    # --- event-level participation ---
    # Count events per player per season, then average across seasons
    events_per_season = (
        all_data
        .groupby(["_season", player_col])
        .size()
        .groupby(player_col)
        .mean()
    )

    # Assemble statistics table
    out = pd.DataFrame({
        "AvgRounds": avg_rounds,
        "AvgEvents": events_per_season,
        "Mean": grouped.mean(),
        "Variance": grouped.var(ddof=1),
        "Skew": grouped.skew(),
    }).reset_index().rename(columns={player_col: "Player"})

    # Filter players with insufficient participation
    out = out[out["AvgRounds"] >= float(min_avg_rounds)].copy()

    # Build normalized sampling weights
    out["Weight"] = build_participation_weights(
        out["AvgEvents"],
        weight_power=weight_power,
        weight_floor=weight_floor,
    )

    # Sort so better players (lower mean score) appear first
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

    # Check to make sure denominators wont blow up skew or cause ZeroDivisionError
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
    Draw round-by-round integer scores for multiple players.

    Parameters
    ----------
    player_params : dict
        Mapping: player_id -> (a, loc, scale, weight)
    n_rounds : int
        Number of rounds to simulate.
    player_ids : list or array, optional
        Player IDs to simulate (field for this tournament).
        If None, all players in player_params are used.

    Returns
    -------
    numpy.ndarray
        Array of shape (num_players, n_rounds),
        where each row corresponds to one player
        and values are integer scores.
    """
    if player_ids is None:
        player_ids = list(player_params.keys())

    num_players = len(player_ids)
    scores = np.zeros((num_players, n_rounds), dtype=int)

    for i, pid in enumerate(player_ids):
        a, loc, scale, _ = player_params[pid]

        raw_scores = skewnorm.rvs(
            a,
            loc=loc,
            scale=scale,
            size=n_rounds,
        )

        # Round to nearest integer (unbiased)
        scores[i, :] = np.rint(raw_scores).astype(int)

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

def assign_points_with_ties(results_df, event_type, points_tables, score_col="TotalScore"):
    """
    Assign FinalRank and Points using PGA-style tie averaging.

    - Competition ranking (1,1,3...)
    - Points averaged over occupied positions
    """
    table = points_tables[event_type]
    df = results_df.sort_values(score_col, ascending=True).reset_index(drop=True).copy()

    # 0-based finishing positions
    df["_pos0"] = np.arange(len(df))

    # Competition rank: first index in tie group + 1
    df["FinalRank"] = df.groupby(score_col)["_pos0"].transform("min") + 1

    # Tie group bounds
    grp_min = df.groupby(score_col)["_pos0"].transform("min")
    grp_max = df.groupby(score_col)["_pos0"].transform("max")

    points = []
    for start, end in zip(grp_min, grp_max):
        pts = []
        for pos in range(int(start), int(end) + 1):
            pts.append(float(table[pos]) if pos < len(table) else 0.0)
        points.append(float(np.mean(pts)))

    df["Points"] = points

    return df.drop(columns=["_pos0"])

def apply_cut(scores_after_two_rounds, rule):
    """
    scores_after_two_rounds: DataFrame with columns [PLAYER_ID_COL, TOTAL_2R_COL]
    Returns a set of player_ids who make the cut.
    """
    if rule == CutRule.NONE:
        return set(scores_after_two_rounds[PLAYER_ID_COL].tolist())

    df = scores_after_two_rounds.sort_values(TOTAL_2R_COL, ascending=True).reset_index(drop=True)

    # --- Generic "top N and ties" rules ---
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

    # --- "Top 50 + within 10 shots of lead" rule ---
    if rule == CutRule.TOP_50_PLUS_10_SHOTS:
        n = CUT_TOP_50
        if len(df) <= n:
            return set(df[PLAYER_ID_COL].tolist())

        leader = float(df.loc[0, TOTAL_2R_COL])
        base_cut = float(df.loc[n - 1, TOTAL_2R_COL])

        cut_score = max(base_cut, leader + CUT_PLUS_SHOTS)
        return set(df.loc[df[TOTAL_2R_COL] <= cut_score, PLAYER_ID_COL].tolist())

    raise ValueError("Unknown cut rule: " + str(rule))

def simulate_season(player_params, schedule, seed=123):
    """
    Simulate a season using TournamentType schedule.

    player_params[pid] = (a, loc, scale, weight)
    weights assumed normalized already.
    """
    pids = [str(pid) for pid in list(player_params.keys())]

    weights = [float(player_params[pid][3]) for pid in pids]

    season_points = {pid: 0.0 for pid in pids}
    event_results = []

    rng = np.random.default_rng(seed)

    def run_event(tournament_type):
        cfg = tournament_type.value  # TournamentConfig

        field = rng.choice(
            pids,
            size=cfg.field_size,
            replace=False,
            p=weights,
        ).tolist()

        # --- simulate first two rounds ---
        scores_pre = sample_round_scores_for_players(player_params, CUT_AFTER_ROUND, field)
        totals_2r = scores_pre.sum(axis=1)

        df2 = pd.DataFrame({
            PLAYER_ID_COL: field,
            TOTAL_2R_COL: totals_2r,
        })

        made_cut = apply_cut(df2, cfg.cut_rule)
        survivors = [pid for pid in field if pid in made_cut]

        # --- simulate remaining rounds for survivors ---
        remaining_rounds = ROUNDS_PER_EVENT - CUT_AFTER_ROUND
        scores_post = sample_round_scores_for_players(player_params, remaining_rounds, survivors)
        totals_post = scores_post.sum(axis=1)

        surv_totals_2r = df2.set_index(PLAYER_ID_COL).loc[survivors, TOTAL_2R_COL].values
        final_totals = surv_totals_2r + totals_post

        results_cut = pd.DataFrame({
            "Player": survivors,
            "TotalScore": final_totals,
        })

        # tie-aware ranking + points
        results_cut = assign_points_with_ties(
            results_cut,
            event_type=cfg.points_type,
            points_tables=POINTS_TABLES,
            score_col="TotalScore",
        )

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

        # Keep track of tournament type
        results["TournamentType"] = tournament_type.name

        return results

    for tournament_type in schedule:
        event_results.append(run_event(tournament_type))

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

season = simulate_season(player_params, SEASON_SCHEDULE)

print(season[0][:26])
