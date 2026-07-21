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
