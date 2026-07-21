"""Tests for golf_simulator.points."""

import pandas as pd
import pytest

from golf_simulator import points as points_mod
from golf_simulator.domain import CutRule, EventType


def test_get_points_for_rank_in_range():
    tables = {EventType.REGULAR: [500, 300, 190]}
    assert points_mod.get_points_for_rank(EventType.REGULAR, 1, tables) == 500.0
    assert points_mod.get_points_for_rank(EventType.REGULAR, 3, tables) == 190.0


def test_get_points_for_rank_beyond_table_returns_zero():
    tables = {EventType.REGULAR: [500, 300, 190]}
    assert points_mod.get_points_for_rank(EventType.REGULAR, 4, tables) == 0.0


def test_get_points_for_rank_invalid_rank_raises():
    tables = {EventType.REGULAR: [500, 300, 190]}
    with pytest.raises(ValueError):
        points_mod.get_points_for_rank(EventType.REGULAR, 0, tables)


def test_get_points_for_rank_unknown_event_type_raises():
    with pytest.raises(ValueError):
        points_mod.get_points_for_rank(EventType.REGULAR, 1, {})


def test_assign_points_with_ties_no_ties():
    tables = {EventType.REGULAR: [500, 300, 190, 135]}
    df = pd.DataFrame({"Player": ["A", "B", "C", "D"], "TotalScore": [270, 271, 272, 273]})

    out = points_mod.assign_points_with_ties(df, EventType.REGULAR, tables)

    row_a = out[out["Player"] == "A"].iloc[0]
    assert row_a["FinalRank"] == 1
    assert row_a["Points"] == 500.0

    row_d = out[out["Player"] == "D"].iloc[0]
    assert row_d["FinalRank"] == 4
    assert row_d["Points"] == 135.0


def test_assign_points_with_ties_tied_top_two():
    tables = {EventType.REGULAR: [500, 300, 190, 135]}
    df = pd.DataFrame({"Player": ["A", "B", "C"], "TotalScore": [270, 270, 272]})

    out = points_mod.assign_points_with_ties(df, EventType.REGULAR, tables)

    tied = out[out["Player"].isin(["A", "B"])]
    assert (tied["FinalRank"] == 1).all()
    assert (tied["Points"] == 400.0).all()  # mean of 500 and 300

    third = out[out["Player"] == "C"].iloc[0]
    assert third["FinalRank"] == 3
    assert third["Points"] == 190.0


def test_apply_cut_none_keeps_everyone():
    df = pd.DataFrame({"player_id": ["A", "B"], "total_2r": [140, 145]})
    made_cut = points_mod.apply_cut(df, CutRule.NONE)
    assert made_cut == {"A", "B"}


def test_apply_cut_top_n_ties(monkeypatch):
    monkeypatch.setattr(points_mod, "CUT_TOP_50", 2)
    df = pd.DataFrame({
        "player_id": ["A", "B", "C", "D"],
        "total_2r": [140, 141, 141, 150],
    })
    made_cut = points_mod.apply_cut(df, CutRule.TOP_50_TIES)
    # cut line is rank 2 (score 141); C ties at 141 so also makes it, D does not
    assert made_cut == {"A", "B", "C"}


def test_apply_cut_top_n_ties_field_smaller_than_n(monkeypatch):
    monkeypatch.setattr(points_mod, "CUT_TOP_65", 65)
    df = pd.DataFrame({"player_id": ["A", "B"], "total_2r": [140, 145]})
    made_cut = points_mod.apply_cut(df, CutRule.TOP_65_TIES)
    assert made_cut == {"A", "B"}


def test_apply_cut_top_50_plus_10_shots(monkeypatch):
    monkeypatch.setattr(points_mod, "CUT_TOP_50", 2)
    monkeypatch.setattr(points_mod, "CUT_PLUS_SHOTS", 10.0)
    df = pd.DataFrame({
        "player_id": ["A", "B", "C", "D"],
        "total_2r": [140, 141, 145, 200],
    })
    # leader=140, base_cut(rank2)=141, leader+10=150 -> cut_score = max(141,150)=150
    # so C (145) also survives even though it's outside the top-2, but D (200) does not.
    made_cut = points_mod.apply_cut(df, CutRule.TOP_50_PLUS_10_SHOTS)
    assert made_cut == {"A", "B", "C"}


def test_apply_cut_unknown_rule_raises():
    df = pd.DataFrame({"player_id": ["A"], "total_2r": [140]})
    with pytest.raises(ValueError):
        points_mod.apply_cut(df, "not-a-rule")
