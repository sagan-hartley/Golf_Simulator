"""Tests for golf_simulator.cli."""

import numpy as np
import pandas as pd

from golf_simulator.cli import main, parse_args

N_PLAYERS = 170  # must exceed the largest TournamentType field_size (156, REGULAR)


def _write_monday_chase_field_csv(path, n_players, mean_base=71.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_players):
        rows.append({
            "player_id": f"{path.stem}_{i}",
            "mean": float(mean_base + rng.uniform(0, 3)),
            "variance": float(rng.uniform(2, 6)),
            "skew": float(rng.uniform(-0.2, 0.2)),
            "weight": 1.0,
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_monday_chase_settings_yaml(path, aspirant_file, main_file, output_dir):
    path.write_text(
        f"""
        aspirant_pool:
          field_file: {aspirant_file}
        main_event_pool:
          field_file: {main_file}
        chase:
          advance_pct: 0.2
          parlay_top_n: 25
          n_weeks: 2
          n_simulations: 3
        output:
          output_dir: {output_dir}
          filename: chase.csv
        """
    )


def _write_card_retention_field_csv(path, n_players, mean_base=70.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_players):
        rows.append({
            "player_id": f"{path.stem}_{i}",
            "mean": float(mean_base + rng.uniform(0, 3)),
            "variance": float(rng.uniform(2, 6)),
            "skew": float(rng.uniform(-0.2, 0.2)),
            "weight": 1.0,
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_alignment_schedule_csv(path):
    path.write_text(
        "event_number,tournament_type\n"
        "1,ALIGNMENT_REGULAR\n"
        "2,MAJOR_MASTERS\n"
    )


def _write_card_retention_settings_yaml(path, card_file, outside_file, schedule_path, output_dir):
    path.write_text(
        f"""
        card_pool:
          field_file: {card_file}
        outside_pool:
          field_file: {outside_file}
        schedule:
          path: {schedule_path}
        retention:
          cutoff: 90
          n_simulations: 3
        output:
          output_dir: {output_dir}
          filename: retention.csv
        """
    )


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


def test_no_subcommand_defaults_to_season():
    assert parse_args([]).command == "season"
    assert parse_args(["--settings", "x.yaml"]).command == "season"
    assert parse_args(["season"]).command == "season"
    assert parse_args(["monday-chase"]).command == "monday-chase"
    assert parse_args(["card-retention"]).command == "card-retention"


def test_bare_help_shows_top_level_subcommand_list(capsys):
    try:
        parse_args(["--help"])
    except SystemExit:
        pass
    captured = capsys.readouterr()
    assert "monday-chase" in captured.out
    assert "card-retention" in captured.out
    assert "season" in captured.out


def test_monday_chase_runs_end_to_end(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    aspirant_path = tmp_path / "aspirants.csv"
    _write_monday_chase_field_csv(aspirant_path, 130, mean_base=71.0)

    main_path = tmp_path / "main_field.csv"
    _write_monday_chase_field_csv(main_path, 160, mean_base=69.0, seed=1)

    settings_path = tmp_path / "monday_chase.yaml"
    out_dir = tmp_path / "outputs"
    _write_monday_chase_settings_yaml(settings_path, aspirant_path, main_path, out_dir)

    exit_code = main(["monday-chase", "--settings", str(settings_path)])

    assert exit_code == 0
    out_file = out_dir / "chase.csv"
    assert out_file.exists()

    results = pd.read_csv(out_file)
    assert len(results) == 130
    assert "Any_Parlay_pct" in results.columns


def test_card_retention_runs_end_to_end(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    card_path = tmp_path / "card_pool.csv"
    _write_card_retention_field_csv(card_path, 130, mean_base=70.0)

    outside_path = tmp_path / "outside_pool.csv"
    _write_card_retention_field_csv(outside_path, 40, mean_base=69.0, seed=1)

    schedule_path = tmp_path / "alignment_schedule.csv"
    _write_alignment_schedule_csv(schedule_path)

    settings_path = tmp_path / "card_retention.yaml"
    out_dir = tmp_path / "outputs"
    _write_card_retention_settings_yaml(
        settings_path, card_path, outside_path, schedule_path, out_dir
    )

    exit_code = main(["card-retention", "--settings", str(settings_path)])

    assert exit_code == 0
    out_file = out_dir / "retention.csv"
    assert out_file.exists()

    results = pd.read_csv(out_file)
    assert len(results) == 130
    assert "Retained_Card_pct" in results.columns
    assert "Avg_SeasonRank" in results.columns
