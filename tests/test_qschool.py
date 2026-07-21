"""Tests for golf_simulator.qschool."""

import numpy as np
import pytest

from golf_simulator.qschool import (
    _shift_params,
    run_qschool,
    validate_qschool_pools,
)
from golf_simulator.qschool_settings import QSchoolConfig


def _params(loc, scale=4.0, weight=1.0):
    return (0.0, loc, scale, weight)  # a=0 -> symmetric skew-normal


def _competition(n=100, loc=71.0, seed=0):
    rng = np.random.default_rng(seed)
    return {
        f"c{i}": _params(float(loc + rng.uniform(0, 1.5)), float(rng.uniform(4, 6)))
        for i in range(n)
    }


def test_shift_params_makes_player_better():
    a, loc, scale, w = _shift_params((0.3, 70.0, 4.0, 1.0), 1.5)
    assert loc == pytest.approx(68.5)  # lower score = better
    assert (a, scale, w) == (0.3, 4.0, 1.0)


def test_disjoint_pool_ids_required():
    asp = {"shared": _params(69.0)}
    comp = _competition()
    comp["shared"] = _params(71.0)
    with pytest.raises(ValueError, match="shared"):
        validate_qschool_pools(asp, comp, QSchoolConfig(stage_field_size=78))


def test_undersized_competition_pool_raises():
    asp = {"a": _params(69.0)}
    comp = _competition(n=10)  # < stage_field_size - 1
    with pytest.raises(ValueError, match="Not enough competition"):
        validate_qschool_pools(asp, comp, QSchoolConfig(stage_field_size=78))


def test_output_shape_and_tier_nesting():
    asp = {"elite": _params(68.5), "journeyman": _params(71.0)}
    comp = _competition(n=100)
    cfg = QSchoolConfig(n_simulations=120, stage_field_size=60, strength_step=0.5)

    res = run_qschool(asp, comp, cfg)

    # one row per (aspirant, start-stage in {1,2,3})
    assert len(res) == 2 * 3
    assert set(res["Start_Stage"]) == {1, 2, 3}
    for col in ("PGA_Card_pct", "Full_KF_pct", "Conditional_pct", "Any_Status_pct"):
        assert res[col].between(0, 100).all()
    # Any_Status is the sum of the three mutually-exclusive tiers (each column
    # is independently rounded to 2 dp, so allow a small rounding tolerance).
    tier_sum = res["PGA_Card_pct"] + res["Full_KF_pct"] + res["Conditional_pct"]
    assert (res["Any_Status_pct"] - tier_sum).abs().max() < 0.05
    assert (res["Any_Status_pct"] >= res["PGA_Card_pct"]).all()


def test_stronger_aspirant_earns_status_more_often():
    asp = {"elite": _params(68.5), "average": _params(71.5)}
    comp = _competition(n=100)
    cfg = QSchoolConfig(n_simulations=200, stage_field_size=60, strength_step=0.5)

    res = run_qschool(asp, comp, cfg).set_index(["Player", "Start_Stage"])
    for stage in (1, 2, 3):
        elite = res.loc[("elite", stage), "Any_Status_pct"]
        average = res.loc[("average", stage), "Any_Status_pct"]
        assert elite > average


def test_starting_later_dominates_when_fields_are_equal():
    # With strength_step=0 every stage is equally tough, so starting later is
    # purely "fewer 20% survival gates" -> strictly better odds of status.
    asp = {"player": _params(70.0)}
    comp = _competition(n=100)
    cfg = QSchoolConfig(n_simulations=400, stage_field_size=60, strength_step=0.0)

    res = run_qschool(asp, comp, cfg).set_index("Start_Stage")
    s1 = res.loc[1, "Any_Status_pct"]
    s2 = res.loc[2, "Any_Status_pct"]
    s3 = res.loc[3, "Any_Status_pct"]
    assert s3 > s2 > s1
