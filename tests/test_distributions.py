"""Tests for golf_simulator.distributions."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import skewnorm

from golf_simulator.distributions import (
    _skew_from_delta,
    add_skill_columns,
    player_means,
    skewnorm_params_from_moments,
)


def test_zero_skew_returns_normal_params():
    a, loc, scale = skewnorm_params_from_moments(mean=72.0, variance=9.0, skew_val=0.0)
    assert a == 0.0
    assert loc == pytest.approx(72.0)
    assert scale == pytest.approx(3.0)


def test_moment_round_trip_for_positive_skew():
    mean, variance, skew_val = 71.5, 6.25, 0.6
    a, loc, scale = skewnorm_params_from_moments(mean, variance, skew_val)

    sim_mean, sim_var, sim_skew = skewnorm.stats(a, loc=loc, scale=scale, moments="mvs")

    assert float(sim_mean) == pytest.approx(mean, abs=1e-6)
    assert float(sim_var) == pytest.approx(variance, abs=1e-6)
    assert float(sim_skew) == pytest.approx(skew_val, abs=1e-3)


def test_moment_round_trip_for_negative_skew():
    mean, variance, skew_val = 70.0, 4.0, -0.4
    a, loc, scale = skewnorm_params_from_moments(mean, variance, skew_val)

    sim_mean, sim_var, sim_skew = skewnorm.stats(a, loc=loc, scale=scale, moments="mvs")

    assert float(sim_mean) == pytest.approx(mean, abs=1e-6)
    assert float(sim_var) == pytest.approx(variance, abs=1e-6)
    assert float(sim_skew) == pytest.approx(skew_val, abs=1e-3)


def test_invalid_variance_raises():
    with pytest.raises(ValueError):
        skewnorm_params_from_moments(mean=70.0, variance=0.0, skew_val=0.1)

    with pytest.raises(ValueError):
        skewnorm_params_from_moments(mean=70.0, variance=-1.0, skew_val=0.1)


def test_invalid_skew_raises():
    with pytest.raises(ValueError):
        skewnorm_params_from_moments(mean=70.0, variance=4.0, skew_val=np.nan)


def test_skew_from_delta_boundary_raises_zero_division():
    with pytest.raises(ZeroDivisionError):
        _skew_from_delta(1.26)


def test_player_means_recovers_input_mean():
    # a=0 skew-normal is just Normal(loc, scale), so its mean is loc.
    params = {"A": (0.0, 70.0, 3.0, 1.0), "B": (0.0, 68.0, 3.0, 1.0)}
    means = player_means(params)
    assert means["A"] == pytest.approx(70.0)
    assert means["B"] == pytest.approx(68.0)


def test_add_skill_columns_edge_is_positive_for_better_players():
    # Field mean is 70; A (68) is 2 strokes better -> +2 edge; C (72) is -2.
    params = {"A": (0.0, 68.0, 3.0, 1.0), "B": (0.0, 70.0, 3.0, 1.0), "C": (0.0, 72.0, 3.0, 1.0)}
    df = pd.DataFrame({"Player": ["A", "B", "C"], "Win_pct": [50.0, 30.0, 20.0]})

    out = add_skill_columns(df, params)

    # columns inserted right after Player
    assert list(out.columns) == ["Player", "Mean", "Edge_vs_Field", "Win_pct"]
    row = out.set_index("Player")
    assert row.loc["A", "Mean"] == pytest.approx(68.0)
    assert row.loc["A", "Edge_vs_Field"] == pytest.approx(2.0)   # better than field
    assert row.loc["C", "Edge_vs_Field"] == pytest.approx(-2.0)  # worse than field
