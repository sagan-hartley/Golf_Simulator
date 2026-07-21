"""Tests for golf_simulator.season."""

import numpy as np
import pytest

from golf_simulator import points as points_mod
from golf_simulator.domain import TournamentType
from golf_simulator.season import play_event, validate_field_size


def _fake_player_params(n):
    return {f"Player{i}": (0.0, 70.0, 3.0, 1.0) for i in range(n)}


def _fake_params(loc, scale=0.01, weight=1.0):
    # a=0.0 (no skew), tiny scale -> scores land essentially exactly at loc,
    # making relative ranking deterministic for test purposes.
    return (0.0, loc, scale, weight)


def test_validate_field_size_passes_when_large_enough():
    # PLAYOFF has the smallest field_size (70); REGULAR has the largest (156).
    player_params = _fake_player_params(156)
    validate_field_size(player_params, [TournamentType.REGULAR, TournamentType.PLAYOFF])


def test_validate_field_size_raises_when_too_small():
    player_params = _fake_player_params(50)
    with pytest.raises(ValueError, match="50.*156"):
        validate_field_size(player_params, [TournamentType.REGULAR])


def test_validate_field_size_empty_schedule_never_raises():
    player_params = _fake_player_params(1)
    validate_field_size(player_params, [])


def test_play_event_missed_cut_gets_nan_rank(monkeypatch):
    # MAJOR_MASTERS uses CutRule.TOP_50_TIES, keyed off CUT_TOP_50; shrink it
    # so a 4-player field produces a real cut.
    monkeypatch.setattr(points_mod, "CUT_TOP_50", 2)

    player_params = {
        "A": _fake_params(60.0),
        "B": _fake_params(65.0),
        "C": _fake_params(90.0),
        "D": _fake_params(95.0),
    }
    field = ["A", "B", "C", "D"]
    rng = np.random.default_rng(0)

    results, scores_pre, survivors, scores_post = play_event(
        player_params, field, TournamentType.MAJOR_MASTERS, rng
    )

    assert set(survivors) == {"A", "B"}

    missed = results[results["Player"].isin(["C", "D"])]
    assert missed["FinalRank"].isna().all()
    assert (missed["Points"] == 0.0).all()

    made = results[results["Player"].isin(["A", "B"])]
    assert not made["FinalRank"].isna().any()
    assert set(results["TournamentType"]) == {"MAJOR_MASTERS"}
    assert scores_pre.shape == (4, 2)  # CUT_AFTER_ROUND
    assert scores_post.shape == (2, 2)  # survivors x remaining rounds
