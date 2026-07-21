"""points.py.

Converts finishing rank into tournament points (with PGA-style tie
averaging) and determines who survives a 36-hole cut.
"""

import numpy as np

from golf_simulator.domain import (
    CUT_PLUS_SHOTS,
    CUT_TOP_50,
    CUT_TOP_60,
    CUT_TOP_65,
    CUT_TOP_70,
    PLAYER_ID_COL,
    TOTAL_2R_COL,
    CutRule,
)


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


def top_n_with_ties(df, id_col, score_col, n):
    """Return the ids of the top `n` rows by `score_col`, ties included.

    `score_col` is ascending = better; any row tied with the row at the
    cutoff is also included.

    Parameters
    ----------
    df : pd.DataFrame
    id_col : str
    score_col : str
    n : int
        Cutoff rank. If `len(df) <= n`, every row is returned.

    Returns
    -------
    set
        Ids of every row that made the cutoff.
    """
    sorted_df = df.sort_values(score_col, ascending=True).reset_index(drop=True)
    if len(sorted_df) <= n:
        return set(sorted_df[id_col].tolist())
    cut_score = float(sorted_df.loc[n - 1, score_col])
    return set(sorted_df.loc[sorted_df[score_col] <= cut_score, id_col].tolist())


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

    top_n_map = {
        CutRule.TOP_50_TIES: CUT_TOP_50,
        CutRule.TOP_60_TIES: CUT_TOP_60,
        CutRule.TOP_65_TIES: CUT_TOP_65,
        CutRule.TOP_70_TIES: CUT_TOP_70,
    }

    if rule in top_n_map:
        return top_n_with_ties(
            scores_after_two_rounds, PLAYER_ID_COL, TOTAL_2R_COL, top_n_map[rule]
        )

    df = scores_after_two_rounds.sort_values(TOTAL_2R_COL, ascending=True).reset_index(drop=True)

    if rule == CutRule.TOP_50_PLUS_10_SHOTS:
        n = CUT_TOP_50
        if len(df) <= n:
            return set(df[PLAYER_ID_COL].tolist())
        leader = float(df.loc[0, TOTAL_2R_COL])
        base_cut = float(df.loc[n - 1, TOTAL_2R_COL])
        cut_score = max(base_cut, leader + CUT_PLUS_SHOTS)
        return set(df.loc[df[TOTAL_2R_COL] <= cut_score, PLAYER_ID_COL].tolist())

    raise ValueError("Unknown cut rule: " + str(rule))
