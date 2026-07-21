"""Tests for golf_simulator.cli."""

import numpy as np
import pandas as pd

from golf_simulator.cli import main

N_PLAYERS = 170  # must exceed the largest TournamentType field_size (156, REGULAR)


def _write_field_csv(path, n_players):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n_players):
        rows.append({
            "player_id": i,
            "mean": float(68 + rng.uniform(0, 6)),
            "variance": float(rng.uniform(2, 8)),
            "skew": float(rng.uniform(-0.3, 0.3)),
            "weight": 1.0,
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_schedule_csv(path):
    path.write_text(
        "event_number,tournament_type\n"
        "1,REGULAR\n"
        "2,SIGNATURE_CUT\n"
    )


def _write_settings_yaml(path, field_file, output_dir):
    path.write_text(
        f"""
        dynamic_weights:
          enabled: true
        data:
          field_file: {field_file}
        output:
          output_dir: {output_dir}
        monte_carlo:
          n_simulations: 2
        """
    )


def test_main_runs_end_to_end_with_custom_field_and_dynamic_weights(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    field_path = tmp_path / "field.csv"
    _write_field_csv(field_path, N_PLAYERS)

    schedule_path = tmp_path / "season_schedule.csv"
    _write_schedule_csv(schedule_path)

    settings_path = tmp_path / "settings.yaml"
    out_dir = tmp_path / "outputs"
    _write_settings_yaml(settings_path, field_path, out_dir)

    exit_code = main(["--settings", str(settings_path), "--schedule", str(schedule_path)])

    assert exit_code == 0
    assert (out_dir / "season_static.csv").exists()
    assert (out_dir / "season_dynamic.csv").exists()
    assert (out_dir / "mc_results.csv").exists()

    mc = pd.read_csv(out_dir / "mc_results.csv")
    assert len(mc) == N_PLAYERS


def test_main_reports_clear_error_when_field_too_small(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    field_path = tmp_path / "field.csv"
    _write_field_csv(field_path, 10)  # too small for a REGULAR event (156)

    schedule_path = tmp_path / "season_schedule.csv"
    _write_schedule_csv(schedule_path)

    settings_path = tmp_path / "settings.yaml"
    out_dir = tmp_path / "outputs"
    _write_settings_yaml(settings_path, field_path, out_dir)

    exit_code = main(["--settings", str(settings_path), "--schedule", str(schedule_path)])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Not enough players" in captured.err
