"""Tests for golf_simulator.player_field."""

import pandas as pd
import pytest

from golf_simulator.distributions import build_player_generators
from golf_simulator.player_field import FieldError, load_custom_field


def _write(tmp_path, content):
    path = tmp_path / "field.csv"
    path.write_text(content)
    return path


def test_valid_field_matches_build_player_generators_directly(tmp_path):
    path = _write(
        tmp_path,
        "player_id,mean,variance,skew,weight\n"
        "1,70.0,4.0,0.1,1.0\n"
        "2,71.5,5.0,-0.2,2.0\n",
    )

    loaded = load_custom_field(path)

    equivalent_df = pd.DataFrame({
        "Player": [1, 2],
        "Mean": [70.0, 71.5],
        "Variance": [4.0, 5.0],
        "Skew": [0.1, -0.2],
        "Weight": [1.0, 2.0],
    })
    expected = build_player_generators(equivalent_df)

    assert loaded.keys() == expected.keys()
    for pid in expected:
        assert loaded[pid] == pytest.approx(expected[pid])


def test_missing_required_column_raises(tmp_path):
    # variance is required; its absence must error.
    path = _write(tmp_path, "player_id,mean,skew,weight\n1,70.0,0.1,1.0\n")
    with pytest.raises(FieldError, match="variance"):
        load_custom_field(path)


def test_weight_column_optional_defaults_to_one(tmp_path):
    # No weight column -> every player gets weight 1.0 (equal selection odds).
    path = _write(
        tmp_path,
        "player_id,mean,variance,skew\n1,70.0,4.0,0.1\n2,71.0,4.0,0.1\n",
    )
    loaded = load_custom_field(path)
    assert loaded["1"][3] == pytest.approx(1.0)
    assert loaded["2"][3] == pytest.approx(1.0)


def test_skew_column_optional_defaults_to_zero(tmp_path):
    # No skew column -> skew 0 (symmetric): a=0 in the fitted params.
    path = _write(tmp_path, "player_id,mean,variance\n1,70.0,4.0\n")
    loaded = load_custom_field(path)
    assert loaded["1"][0] == pytest.approx(0.0)  # a
    assert loaded["1"][3] == pytest.approx(1.0)  # weight default too


def test_duplicate_player_id_raises(tmp_path):
    path = _write(
        tmp_path,
        "player_id,mean,variance,skew,weight\n"
        "1,70.0,4.0,0.1,1.0\n"
        "1,71.0,4.0,0.1,1.0\n",
    )
    with pytest.raises(FieldError, match="duplicate"):
        load_custom_field(path)


def test_nonpositive_variance_raises(tmp_path):
    path = _write(tmp_path, "player_id,mean,variance,skew,weight\n1,70.0,0.0,0.1,1.0\n")
    with pytest.raises(FieldError, match="variance"):
        load_custom_field(path)


def test_nonpositive_weight_raises(tmp_path):
    path = _write(tmp_path, "player_id,mean,variance,skew,weight\n1,70.0,4.0,0.1,0.0\n")
    with pytest.raises(FieldError, match="weight"):
        load_custom_field(path)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FieldError):
        load_custom_field(tmp_path / "does_not_exist.csv")


def test_empty_file_raises(tmp_path):
    path = _write(tmp_path, "player_id,mean,variance,skew,weight\n")
    with pytest.raises(FieldError, match="no rows"):
        load_custom_field(path)


def test_example_field_file_loads():
    loaded = load_custom_field("data/custom_fields/example_field.csv")
    assert len(loaded) == 15


def _write_season(path, player, n_rounds=25):
    path.write_text(
        "player,score\n" + "\n".join(f"{player},{70 + i % 5}" for i in range(n_rounds)) + "\n"
    )


def test_load_player_pool_season_files_selects_subset(tmp_path):
    from golf_simulator.player_field import load_player_pool
    from golf_simulator.settings import DataConfig, ParticipationWeightConfig

    _write_season(tmp_path / "yr_a.csv", "AlphaOnly")
    _write_season(tmp_path / "yr_b.csv", "BravoOnly")
    participation = ParticipationWeightConfig(min_avg_rounds=5)

    # Only yr_a.csv -> only AlphaOnly is in the pool.
    cfg = DataConfig(season_dir=str(tmp_path), season_files=["yr_a.csv"])
    pool = load_player_pool(cfg, participation)
    assert "AlphaOnly" in pool
    assert "BravoOnly" not in pool

    # No season_files -> globs the folder, so both appear.
    cfg_all = DataConfig(season_dir=str(tmp_path))
    pool_all = load_player_pool(cfg_all, participation)
    assert {"AlphaOnly", "BravoOnly"} <= set(pool_all)


def test_load_player_pool_missing_season_file_raises(tmp_path):
    from golf_simulator.player_field import load_player_pool
    from golf_simulator.settings import DataConfig, ParticipationWeightConfig

    _write_season(tmp_path / "yr_a.csv", "AlphaOnly")
    cfg = DataConfig(season_dir=str(tmp_path), season_files=["yr_a.csv", "yr_missing.csv"])
    with pytest.raises(FieldError, match="yr_missing.csv"):
        load_player_pool(cfg, ParticipationWeightConfig(min_avg_rounds=5))
