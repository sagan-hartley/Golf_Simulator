"""Tests for golf_simulator.data_loading."""

import pandas as pd
import pytest

from golf_simulator.data_loading import (
    build_participation_weights,
    compute_player_stats,
    load_and_standardize_round_data,
)


def test_build_participation_weights_normalizes_to_one():
    avg_events = pd.Series({"A": 10.0, "B": 20.0, "C": 30.0})
    weights = build_participation_weights(avg_events)
    assert weights.sum() == pytest.approx(1.0)
    assert weights["C"] > weights["A"]


def test_build_participation_weights_floor_lifts_low_players():
    # weight_floor is a fraction of the uniform 1/N weight; a strong floor
    # lifts under-weighted players toward the uniform share, compressing the
    # gap (without capping the top).
    avg_events = pd.Series({"A": 1.0, "B": 9.0})
    no_floor = build_participation_weights(avg_events, weight_floor=0.0)
    floored = build_participation_weights(avg_events, weight_floor=1.0)

    assert floored["A"] > no_floor["A"]
    assert (floored["B"] - floored["A"]) < (no_floor["B"] - no_floor["A"])


def test_build_participation_weights_no_floor_preserves_proportions():
    avg_events = pd.Series({"A": 1.0, "B": 3.0})
    weights = build_participation_weights(avg_events, weight_floor=0.0)
    # No floor -> weights strictly proportional to participation.
    assert weights["B"] == pytest.approx(3.0 * weights["A"])


def test_build_participation_weights_weak_floor_is_noop():
    # A small floor (fraction of 1/N) below every player's actual share
    # should leave the proportional weights unchanged.
    avg_events = pd.Series({"A": 10.0, "B": 20.0, "C": 30.0})
    no_floor = build_participation_weights(avg_events, weight_floor=0.0)
    weak_floor = build_participation_weights(avg_events, weight_floor=0.05)
    assert (weak_floor.values == pytest.approx(no_floor.values))


def test_build_participation_weights_raises_on_nonpositive_sum():
    avg_events = pd.Series({"A": 0.0, "B": 0.0})
    with pytest.raises(ValueError):
        build_participation_weights(avg_events)


def test_load_and_standardize_round_data_missing_column_raises(tmp_path):
    csv_path = tmp_path / "season.csv"
    csv_path.write_text("player,round\nA,1\n")

    with pytest.raises(ValueError):
        load_and_standardize_round_data([str(csv_path)], player_col="player", value_col="score")


def test_compute_player_stats_filters_by_min_avg_rounds(tmp_path):
    csv_path = tmp_path / "season.csv"
    csv_path.write_text(
        "player,score\n"
        + "\n".join([f"Frequent,{70 + i % 5}" for i in range(30)])
        + "\n"
        + "\n".join([f"Occasional,{72 + i % 3}" for i in range(3)])
        + "\n"
    )

    stats = compute_player_stats(
        [str(csv_path)], player_col="player", value_col="score", min_avg_rounds=10
    )

    assert "Frequent" in stats["Player"].tolist()
    assert "Occasional" not in stats["Player"].tolist()
    assert stats["Weight"].sum() == pytest.approx(1.0)


def test_compute_player_stats_avg_rounds_is_per_active_season(tmp_path):
    # A player who appears in only one of two season files should be judged on
    # the rounds they actually played, not diluted by the season they missed.
    s1 = tmp_path / "s1.csv"
    s2 = tmp_path / "s2.csv"
    s1.write_text("player,score\n" + "\n".join(f"X,{70 + i % 5}" for i in range(25)) + "\n")
    s2.write_text("player,score\n" + "\n".join(f"Y,{70 + i % 5}" for i in range(25)) + "\n")

    stats = compute_player_stats(
        [str(s1), str(s2)], player_col="player", value_col="score", min_avg_rounds=20
    )

    # X played 25 rounds in its one active season -> AvgRounds 25 >= 20, kept.
    # Under the old total/num_seasons rule that would be 25/2 = 12.5 < 20,
    # wrongly dropping X.
    assert "X" in stats["Player"].tolist()
    assert stats.set_index("Player").loc["X", "AvgRounds"] == pytest.approx(25.0)
