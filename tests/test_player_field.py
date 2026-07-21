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


def test_missing_column_raises(tmp_path):
    path = _write(tmp_path, "player_id,mean,variance,weight\n1,70.0,4.0,1.0\n")
    with pytest.raises(FieldError, match="skew"):
        load_custom_field(path)


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
