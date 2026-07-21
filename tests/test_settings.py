"""Tests for golf_simulator.settings."""

import pytest

from golf_simulator.settings import SettingsError, load_settings


def _write(tmp_path, content):
    path = tmp_path / "settings.yaml"
    path.write_text(content)
    return path


def test_missing_keys_fall_back_to_defaults(tmp_path):
    path = _write(tmp_path, "dynamic_weights:\n  nudge_amount: 0.2\n")
    settings = load_settings(path)

    assert settings.dynamic_weights.nudge_amount == 0.2
    assert settings.dynamic_weights.enabled is True  # default preserved
    assert settings.monte_carlo.n_simulations == 10  # default preserved


def test_fully_populated_yaml_round_trips(tmp_path):
    path = _write(
        tmp_path,
        """
        dynamic_weights:
          enabled: false
          nudge_amount: 0.1
          top_pct: 0.3
          bot_pct: 0.2
          min_weight: 0.01
          max_weight_multiplier: 3.0
        participation_weights:
          min_avg_rounds: 15
          weight_power: 0.8
          weight_floor: 0.02
        monte_carlo:
          n_simulations: 25
          base_seed: 7
          season_seed: 42
        data:
          season_dir: my_data
          player_column: golfer
          score_column: strokes
        output:
          output_dir: results
          season_static_filename: a.csv
          season_dynamic_filename: b.csv
          mc_results_filename: c.csv
        """,
    )
    settings = load_settings(path)

    assert settings.dynamic_weights.enabled is False
    assert settings.monte_carlo.n_simulations == 25
    assert settings.data.season_dir == "my_data"
    assert settings.output.mc_results_filename == "c.csv"


def test_out_of_range_value_raises_settings_error(tmp_path):
    path = _write(tmp_path, "dynamic_weights:\n  nudge_amount: 1.5\n")
    with pytest.raises(SettingsError, match="dynamic_weights.nudge_amount"):
        load_settings(path)


def test_unknown_key_raises_settings_error(tmp_path):
    path = _write(tmp_path, "dynamic_weights:\n  typo_field: 1\n")
    with pytest.raises(SettingsError, match="typo_field"):
        load_settings(path)


def test_missing_file_raises_settings_error(tmp_path):
    with pytest.raises(SettingsError):
        load_settings(tmp_path / "does_not_exist.yaml")


def test_invalid_yaml_raises_settings_error(tmp_path):
    path = _write(tmp_path, "dynamic_weights: [this is not a mapping\n")
    with pytest.raises(SettingsError):
        load_settings(path)
