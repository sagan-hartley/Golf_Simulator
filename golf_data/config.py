"""
config.py
---------
All simulation inputs: constants, enums, dataclasses, points tables,
tournament definitions, and the season schedule.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np

PI = np.pi

# ── Simulation constants ───────────────────────────────────────────────────────

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

PLAYER_ID_COL = "player_id"
TOTAL_2R_COL = "total_2r"

# ── Dynamic weight configuration ───────────────────────────────────────────────

@dataclass
class DynamicWeightConfig:
    """
    Controls rank-based weight nudging after each event.

    Attributes
    ----------
    enabled : bool
        Master switch. False reproduces original static behaviour.
    nudge_amount : float
        How much to adjust weight after each event (e.g. 0.05 = 5%).
        Applied as a multiplier: weight *= (1 + nudge) or (1 - nudge).
    top_pct : float
        Top finishing percentile that earns a positive nudge.
        e.g. 0.25 = top quarter of finishers get a boost.
    bot_pct : float
        Bottom finishing percentile that earns a negative nudge.
        e.g. 0.25 = bottom quarter of finishers get a penalty.
    min_weight : float
        Hard floor before re-normalisation.
    max_weight_multiplier : float
        Maximum a player's weight can grow relative to their baseline.
    """
    enabled: bool = True
    nudge_amount: float = 0.05
    top_pct: float = 0.25
    bot_pct: float = 0.25
    min_weight: float = 0.001
    max_weight_multiplier: float = 2.0


# ── Enums ──────────────────────────────────────────────────────────────────────

class EventType(Enum):
    REGULAR = "regular"
    SIGNATURE = "signature"
    MAJOR_PLAYERS = "major_players"
    ADDITIONAL = "additional"
    PLAYOFFS_2026_APPROX_750 = "playoffs_2026_approx_750"
    ZURICH_TEAM_EACH_PLAYER = "zurich_team_each_player"


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
        points_type=EventType.MAJOR_PLAYERS,
        cut_rule=CutRule.NONE,
        field_size=70,
    )


# ── Points tables ──────────────────────────────────────────────────────────────

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

ZURICH_WINNER_POINTS_EACH_PLAYER = 400.0

POINTS_TABLES = {
    EventType.REGULAR: POINTS_TABLE_REGULAR_500,
    EventType.SIGNATURE: POINTS_TABLE_SIGNATURE_700,
    EventType.MAJOR_PLAYERS: POINTS_TABLE_MAJOR_PLAYERS_750,
    EventType.ADDITIONAL: POINTS_TABLE_ADDITIONAL_300,
    EventType.PLAYOFFS_2026_APPROX_750: POINTS_TABLE_MAJOR_PLAYERS_750,
}

# ── Season schedule ────────────────────────────────────────────────────────────

SEASON_SCHEDULE = [
    TournamentType.REGULAR,
    TournamentType.REGULAR,
    TournamentType.REGULAR,
    TournamentType.REGULAR,
    TournamentType.SIGNATURE_NO_CUT,
    TournamentType.SIGNATURE_CUT,
    TournamentType.REGULAR,
    TournamentType.SIGNATURE_CUT,
    TournamentType.PLAYERS,
    TournamentType.REGULAR,
    TournamentType.REGULAR,
    TournamentType.REGULAR,
    TournamentType.MAJOR_MASTERS,
    TournamentType.SIGNATURE_NO_CUT,
    TournamentType.SIGNATURE_NO_CUT,
    TournamentType.SIGNATURE_NO_CUT,
    TournamentType.MAJOR_PGA,
    TournamentType.REGULAR,
    TournamentType.REGULAR,
    TournamentType.SIGNATURE_CUT,
    TournamentType.REGULAR,
    TournamentType.MAJOR_US_OPEN,
    TournamentType.SIGNATURE_NO_CUT,
    TournamentType.REGULAR,
    TournamentType.REGULAR,
    TournamentType.MAJOR_OPEN,
    TournamentType.REGULAR,
    TournamentType.REGULAR,
    TournamentType.REGULAR,
    TournamentType.PLAYOFF,
    TournamentType.PLAYOFF,
    TournamentType.PLAYOFF,
]