"""Tests for golf_simulator.distributions."""

import numpy as np
import pytest
from scipy.stats import skewnorm

from golf_simulator.distributions import _skew_from_delta, skewnorm_params_from_moments


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
