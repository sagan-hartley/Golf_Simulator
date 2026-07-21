"""Tests for golf_simulator.card_retention."""

import pytest

import golf_simulator.card_retention as cr_mod
from golf_simulator.card_retention import (
    run_n_alignment_seasons,
    simulate_alignment_season,
    validate_alignment_pools,
)
from golf_simulator.domain import TournamentType


def _params(loc, scale=1.0, weight=1.0):
    # a=0.0 (no skew); tiny scale (default overridden per-test) controls
    # how deterministic relative rankings are.
    return (0.0, loc, scale, weight)


def _card_pool(n=130, loc=70.0, scale=3.0, prefix="card"):
    return {f"{prefix}_{i}": _params(loc, scale) for i in range(n)}


def _outside_pool(n=40, loc=69.0, scale=3.0, prefix="outside"):
    return {f"{prefix}_{i}": _params(loc, scale) for i in range(n)}


def _spy_on_play_event(monkeypatch):
    """Capture every (tournament_type, field) play_event is called with."""
    captured = []
    original = cr_mod.play_event

    def spy(player_params, field, tournament_type, rng):
        captured.append((tournament_type, list(field)))
        return original(player_params, field, tournament_type, rng)

    monkeypatch.setattr(cr_mod, "play_event", spy)
    return captured


def test_non_major_field_excludes_outside_pool(monkeypatch):
    card_pool = _card_pool(n=120)
    outside_pool = _outside_pool(n=40)
    captured = _spy_on_play_event(monkeypatch)

    simulate_alignment_season(card_pool, outside_pool, [TournamentType.ALIGNMENT_REGULAR], seed=0)

    assert len(captured) == 1
    tournament_type, field = captured[0]
    assert tournament_type is TournamentType.ALIGNMENT_REGULAR
    assert len(field) == 120
    assert all(pid.startswith("card_") for pid in field)


def test_major_field_mixes_both_pools_with_correct_split(monkeypatch):
    card_pool = _card_pool(n=130)
    outside_pool = _outside_pool(n=40)
    captured = _spy_on_play_event(monkeypatch)

    simulate_alignment_season(card_pool, outside_pool, [TournamentType.MAJOR_MASTERS], seed=0)

    assert len(captured) == 1
    tournament_type, field = captured[0]
    assert len(field) == 156  # MAJOR_MASTERS field_size

    card_in_field = [pid for pid in field if pid.startswith("card_")]
    outside_in_field = [pid for pid in field if pid.startswith("outside_")]

    # card_slots = min(130, 156) = 130 -> every card-pool player is in the field
    assert len(card_in_field) == 130
    assert set(card_in_field) == set(card_pool.keys())
    # outside_slots = 156 - 130 = 26
    assert len(outside_in_field) == 26


def test_outside_pool_never_appears_in_season_summary():
    card_pool = _card_pool(n=130)
    outside_pool = _outside_pool(n=40)

    schedule = [TournamentType.ALIGNMENT_REGULAR, TournamentType.MAJOR_MASTERS]
    summary = simulate_alignment_season(card_pool, outside_pool, schedule, seed=0)

    assert set(summary["Player"]) == set(card_pool.keys())
    assert len(summary) == 130


def test_overlapping_pool_ids_raise_clear_error():
    card_pool = _card_pool(n=130)
    outside_pool = _outside_pool(n=40)
    outside_pool["card_0"] = _params(69.0)  # collide with a card_pool id

    with pytest.raises(ValueError, match="card_0"):
        validate_alignment_pools(card_pool, outside_pool, [TournamentType.ALIGNMENT_REGULAR])


def test_undersized_card_pool_for_non_major_raises():
    card_pool = _card_pool(n=50)  # too small for ALIGNMENT_REGULAR's 120
    outside_pool = _outside_pool(n=40)

    with pytest.raises(ValueError, match="50.*120"):
        validate_alignment_pools(card_pool, outside_pool, [TournamentType.ALIGNMENT_REGULAR])


def test_undersized_combined_pool_for_major_raises():
    card_pool = _card_pool(n=130)
    outside_pool = _outside_pool(n=5)  # 130 + 5 = 135 < 156 needed for a major

    with pytest.raises(ValueError, match="135.*156"):
        validate_alignment_pools(card_pool, outside_pool, [TournamentType.MAJOR_MASTERS])


def test_run_n_alignment_seasons_output_ranges():
    card_pool = _card_pool(n=130)
    outside_pool = _outside_pool(n=40)
    schedule = [TournamentType.ALIGNMENT_REGULAR, TournamentType.MAJOR_MASTERS]

    results = run_n_alignment_seasons(
        card_pool, outside_pool, schedule, n=10, base_seed=0, retention_cutoff=90
    )

    assert len(results) == 130
    assert results["Retained_Card_pct"].between(0, 100).all()
    assert results["Avg_SeasonRank"].between(1, 130).all()


def test_run_n_alignment_seasons_skill_sensitivity():
    strong = {f"strong_{i}": _params(66.0, scale=2.0) for i in range(65)}
    weak = {f"weak_{i}": _params(78.0, scale=2.0) for i in range(65)}
    card_pool = {**strong, **weak}
    outside_pool = _outside_pool(n=40, loc=71.0, scale=3.0)
    schedule = [
        TournamentType.ALIGNMENT_REGULAR,
        TournamentType.ALIGNMENT_REGULAR,
        TournamentType.MAJOR_MASTERS,
    ]

    results = run_n_alignment_seasons(
        card_pool, outside_pool, schedule, n=15, base_seed=1, retention_cutoff=90
    )
    results["group"] = results["Player"].str.split("_").str[0]
    avg_by_group = results.groupby("group")["Retained_Card_pct"].mean()

    assert avg_by_group["strong"] > avg_by_group["weak"]
