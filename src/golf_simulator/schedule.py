"""schedule.py.

Loads and validates ``config/season_schedule.csv`` — the file a
non-coder should edit to change which tournaments happen, and in what
order, over the course of a simulated season.
"""

from pathlib import Path

import pandas as pd

from golf_simulator.domain import TournamentType

DEFAULT_SCHEDULE_PATH = "config/season_schedule.csv"

_EVENT_NUMBER_COL = "event_number"
_TOURNAMENT_TYPE_COL = "tournament_type"
_REQUIRED_COLUMNS = (_EVENT_NUMBER_COL, _TOURNAMENT_TYPE_COL)

_VALID_TOURNAMENT_TYPES = sorted(TournamentType.__members__)


class ScheduleError(ValueError):
    """Raised when config/season_schedule.csv is missing, malformed, or invalid."""


def load_season_schedule(path: str | Path = DEFAULT_SCHEDULE_PATH) -> list[TournamentType]:
    """
    Load and validate the season schedule from a CSV file.

    Parameters
    ----------
    path : str or Path
        Location of the schedule CSV. Must have columns
        ``event_number`` and ``tournament_type``. Row order in the
        file doesn't matter — events are ordered by ``event_number``.

    Returns
    -------
    list[TournamentType]
        The season schedule, ordered by ``event_number``.

    Raises
    ------
    ScheduleError
        If the file is missing, a required column is absent, an
        ``event_number`` is duplicated or the sequence has a gap, or a
        ``tournament_type`` value doesn't match a known tournament type.
    """
    path = Path(path)
    if not path.exists():
        raise ScheduleError(f"Season schedule file not found: {path}")

    df = pd.read_csv(path)

    missing_cols = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ScheduleError(
            f"{path}: missing required column(s): {', '.join(missing_cols)}. "
            f"Expected columns: {', '.join(_REQUIRED_COLUMNS)}."
        )

    if df.empty:
        raise ScheduleError(f"{path}: schedule has no rows.")

    df = df.sort_values(_EVENT_NUMBER_COL).reset_index(drop=True)

    event_numbers = df[_EVENT_NUMBER_COL].tolist()
    expected = list(range(1, len(df) + 1))
    if sorted(event_numbers) != expected:
        raise ScheduleError(
            f"{path}: '{_EVENT_NUMBER_COL}' values must be unique and run from 1 to "
            f"{len(df)} with no gaps. Got: {sorted(event_numbers)}."
        )

    schedule = []
    for row in df.itertuples(index=False):
        raw_type = str(getattr(row, _TOURNAMENT_TYPE_COL)).strip()
        lookup_key = raw_type.upper()
        if lookup_key not in TournamentType.__members__:
            event_number = getattr(row, _EVENT_NUMBER_COL)
            raise ScheduleError(
                f"{path}: row with event_number={event_number} has unknown "
                f"tournament_type '{raw_type}'. Valid values are: "
                f"{', '.join(_VALID_TOURNAMENT_TYPES)}."
            )
        schedule.append(TournamentType[lookup_key])

    return schedule
