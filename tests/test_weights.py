"""Tests for golf_simulator.weights."""

import numpy as np
import pandas as pd
import pytest

from golf_simulator.settings import DynamicWeightConfig
from golf_simulator.weights import nudge_weights


def _config(**overrides):
    defaults = dict(
        enabled=True,
        nudge_amount=0.10,
        top_pct=0.25,
        bot_pct=0.25,
        min_weight=0.0001,
        max_weight_multiplier=10.0,
    )
    defaults.update(overrides)
    return DynamicWeightConfig(**defaults)


def test_top_bucket_gets_boosted_bottom_gets_penalized():
    current = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}
    baseline = dict(current)
    results = pd.DataFrame({
        "Player": ["A", "B", "C", "D"],
        "FinalRank": [1, 2, 3, 4],
    })

    new_weights = nudge_weights(current, baseline, results, _config())

    assert new_weights["A"] > current["A"] / sum(current.values())
    assert new_weights["D"] < current["D"] / sum(current.values())
    assert sum(new_weights.values()) == pytest.approx(1.0)


def test_missed_cut_treated_as_bottom_bucket():
    # Five finishers (so top/bottom buckets don't collide at n=1) plus one
    # player who missed the cut entirely.
    players = ["A", "B", "C", "D", "E", "Missed"]
    current = {p: 1.0 / len(players) for p in players}
    baseline = dict(current)
    results = pd.DataFrame({
        "Player": ["A", "B", "C", "D", "E", "Missed"],
        "FinalRank": [1, 2, 3, 4, 5, np.nan],
    })

    new_weights = nudge_weights(current, baseline, results, _config())

    # A finished 1st (top bucket); Missed didn't survive the cut (bottom bucket).
    assert new_weights["A"] > new_weights["Missed"]


def test_ceiling_enforced_relative_to_baseline():
    current = {"A": 0.5, "B": 0.5}
    baseline = {"A": 0.5, "B": 0.5}
    results = pd.DataFrame({"Player": ["A", "B"], "FinalRank": [1, 2]})

    tight = nudge_weights(
        current, baseline, results, _config(nudge_amount=5.0, max_weight_multiplier=1.2)
    )
    loose = nudge_weights(
        current, baseline, results, _config(nudge_amount=5.0, max_weight_multiplier=100.0)
    )

    # With a tight ceiling, A's growth is capped at baseline * 1.2; with a loose
    # ceiling, A's raw nudge (a 5x multiplier) is allowed to dominate the
    # normalized share far more.
    assert tight["A"] < loose["A"]


def test_weights_sum_to_one():
    current = {"A": 0.4, "B": 0.35, "C": 0.25}
    baseline = dict(current)
    results = pd.DataFrame({"Player": ["A", "B", "C"], "FinalRank": [1, 2, 3]})

    new_weights = nudge_weights(current, baseline, results, _config())

    assert sum(new_weights.values()) == pytest.approx(1.0)
