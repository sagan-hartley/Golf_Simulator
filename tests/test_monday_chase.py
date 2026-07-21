"""Tests for golf_simulator.monday_chase."""

import numpy as np
import pandas as pd
import pytest

from golf_simulator.monday_chase import run_n_monday_chases, simulate_monday_chase
from golf_simulator.monday_chase_settings import ChaseConfig


def _params(loc, scale=0.01, weight=1.0):
    # a=0.0 (no skew); tiny scale makes scores land essentially exactly at
    # loc, so relative ranking is deterministic for test purposes.
    return (0.0, loc, scale, weight)


def _main_pool(n=160, loc=70.0, scale=3.0, seed=0):
    rng = np.random.default_rng(seed)
    return {
        f"M{i}": (0.0, float(loc + rng.uniform(-2, 2)), scale, 1.0)
        for i in range(n)
    }


def test_monday_advance_includes_ties_at_cutoff():
    # P0 alone at 60; P1,P2,P3 tied at 61; rest at 65+. advance_pct picks a
    # raw cutoff of 3, but the tie at rank 2-4 means 4 total should advance.
    aspirants = {
        "P0": _params(60.0),
        "P1": _params(61.0),
        "P2": _params(61.0),
        "P3": _params(61.0),
        "P4": _params(65.0),
        "P5": _params(66.0),
        "P6": _params(67.0),
        "P7": _params(68.0),
        "P8": _params(69.0),
        "P9": _params(70.0),
    }
    main_pool = _main_pool()
    config = ChaseConfig(advance_pct=0.3, parlay_top_n=1, n_weeks=1, n_simulations=1)

    weekly = simulate_monday_chase(aspirants, main_pool, config, seed=0)
    week1 = weekly[weekly["Week"] == 1]

    advanced = set(week1[week1["AdvancedMonday"]]["Player"])
    assert advanced == {"P0", "P1", "P2", "P3"}


def test_parlay_grants_exemption_next_week_and_missed_cut_does_not():
    # P0 has a dominant Monday score, guaranteeing they're the sole Monday
    # qualifier; non-P0 aspirants get a terrible main-event score too (in
    # case they somehow advance Monday), so only P0 can possibly finish
    # inside parlay_top_n.
    aspirants_main_score = {
        "P0": _params(10.0),
        "P1": _params(200.0),
        "P2": _params(201.0),
        "P3": _params(202.0),
        "P4": _params(203.0),
    }
    main_pool = _main_pool()
    config = ChaseConfig(advance_pct=0.2, parlay_top_n=5, n_weeks=2, n_simulations=1)

    weekly = simulate_monday_chase(aspirants_main_score, main_pool, config, seed=0)

    week1_p0 = weekly[(weekly["Week"] == 1) & (weekly["Player"] == "P0")].iloc[0]
    assert week1_p0["AdvancedMonday"]
    assert week1_p0["PlayedMainEvent"]
    assert week1_p0["FinalRank"] == 1
    assert week1_p0["Parlayed"]

    week2_p0 = weekly[(weekly["Week"] == 2) & (weekly["Player"] == "P0")].iloc[0]
    assert not week2_p0["AttemptedMonday"]  # skipped Monday via exemption
    assert week2_p0["PlayedMainEvent"]

    # P1 never has a shot at Monday (P0 always wins it) or the main event.
    week1_p1 = weekly[(weekly["Week"] == 1) & (weekly["Player"] == "P1")].iloc[0]
    assert week1_p1["AttemptedMonday"]
    assert not week1_p1["AdvancedMonday"]
    assert not week1_p1["PlayedMainEvent"]
    assert pd.isna(week1_p1["FinalRank"])
    assert not week1_p1["Parlayed"]


def test_overlapping_pool_ids_raise_clear_error():
    aspirants = {"Shared": _params(65.0)}
    main_pool = _main_pool()
    main_pool["Shared"] = _params(70.0)
    config = ChaseConfig(n_weeks=1, n_simulations=1)

    with pytest.raises(ValueError, match="Shared"):
        simulate_monday_chase(aspirants, main_pool, config, seed=0)


def test_oversubscribed_field_raises_instead_of_truncating():
    # 200 aspirants, all tied at the exact same score, advance_pct=1.0 ->
    # every one of them "advances" (with ties), exceeding the 156-player
    # main event field on their own.
    aspirants = {f"P{i}": _params(65.0) for i in range(200)}
    main_pool = _main_pool()
    config = ChaseConfig(advance_pct=1.0, n_weeks=1, n_simulations=1)

    with pytest.raises(ValueError, match="exceed"):
        simulate_monday_chase(aspirants, main_pool, config, seed=0)


def test_run_n_monday_chases_output_ranges():
    aspirants = {f"P{i}": _params(70.0 + i, scale=3.0) for i in range(20)}
    main_pool = _main_pool()
    config = ChaseConfig(advance_pct=0.1, parlay_top_n=25, n_weeks=4, n_simulations=15, base_seed=0)

    results = run_n_monday_chases(aspirants, main_pool, config)

    assert len(results) == 20
    for col in ("Any_Monday_Advance_pct", "Any_Parlay_pct"):
        assert results[col].between(0, 100).all()
    assert (results["Avg_Weeks_Played"] <= config.n_weeks).all()
    assert (results["Avg_Monday_Attempts"] <= config.n_weeks).all()


def test_run_n_monday_chases_skill_sensitivity():
    strong = {f"S{i}": _params(66.0, scale=2.0) for i in range(15)}
    weak = {f"W{i}": _params(78.0, scale=2.0) for i in range(15)}
    aspirants = {**strong, **weak}
    main_pool = _main_pool(n=160, loc=71.0, scale=3.0)
    config = ChaseConfig(
        advance_pct=0.15, parlay_top_n=25, n_weeks=5, n_simulations=25, base_seed=1
    )

    results = run_n_monday_chases(aspirants, main_pool, config)
    results["group"] = results["Player"].str[0]
    avg_by_group = results.groupby("group")["Advance_Rate_Per_Monday_Attempt"].mean()

    assert avg_by_group["S"] > avg_by_group["W"]


def test_run_n_monday_chases_writes_output_csv(tmp_path):
    aspirants = {f"P{i}": _params(70.0) for i in range(10)}
    main_pool = _main_pool()
    config = ChaseConfig(n_weeks=2, n_simulations=3)
    out_path = tmp_path / "chase_results.csv"

    run_n_monday_chases(aspirants, main_pool, config, output_csv_path=out_path)

    assert out_path.exists()
    loaded = pd.read_csv(out_path)
    assert len(loaded) == 10
