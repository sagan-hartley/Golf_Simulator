"""Tests for golf_simulator.schedule."""

import pytest

from golf_simulator.domain import TournamentType
from golf_simulator.schedule import ScheduleError, load_season_schedule


def _write(tmp_path, content):
    path = tmp_path / "season_schedule.csv"
    path.write_text(content)
    return path


def test_valid_schedule_loads_in_order(tmp_path):
    path = _write(
        tmp_path,
        "event_number,tournament_type\n3,PLAYOFF\n1,regular\n2,Major_Masters\n",
    )
    schedule = load_season_schedule(path)

    assert schedule == [
        TournamentType.REGULAR,
        TournamentType.MAJOR_MASTERS,
        TournamentType.PLAYOFF,
    ]


def test_unknown_tournament_type_raises(tmp_path):
    path = _write(tmp_path, "event_number,tournament_type\n1,NOT_A_REAL_EVENT\n")
    with pytest.raises(ScheduleError, match="event_number=1"):
        load_season_schedule(path)


def test_duplicate_event_number_raises(tmp_path):
    path = _write(
        tmp_path, "event_number,tournament_type\n1,REGULAR\n1,PLAYOFF\n"
    )
    with pytest.raises(ScheduleError):
        load_season_schedule(path)


def test_gap_in_event_numbers_raises(tmp_path):
    path = _write(
        tmp_path, "event_number,tournament_type\n1,REGULAR\n3,PLAYOFF\n"
    )
    with pytest.raises(ScheduleError):
        load_season_schedule(path)


def test_missing_column_raises(tmp_path):
    path = _write(tmp_path, "event_number\n1\n")
    with pytest.raises(ScheduleError, match="tournament_type"):
        load_season_schedule(path)


def test_missing_file_raises(tmp_path):
    with pytest.raises(ScheduleError):
        load_season_schedule(tmp_path / "does_not_exist.csv")
