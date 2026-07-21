"""cli.py.

Command-line entry point (installed as the ``golf-sim`` console
script). Loads config/settings.yaml and config/season_schedule.csv,
fits player distributions from the historical data, simulates one
static season, one dynamic-weight season, and a Monte Carlo batch of
seasons, and writes all three results to the configured output folder.
"""

import argparse
import sys
from glob import glob
from pathlib import Path

from golf_simulator.data_loading import compute_player_stats
from golf_simulator.distributions import build_player_generators
from golf_simulator.monte_carlo import run_n_simulations
from golf_simulator.schedule import DEFAULT_SCHEDULE_PATH, ScheduleError, load_season_schedule
from golf_simulator.season import simulate_season
from golf_simulator.settings import DEFAULT_SETTINGS_PATH, SettingsError, load_settings


def parse_args(argv=None):
    """Parse command-line arguments for the `golf-sim` entry point."""
    parser = argparse.ArgumentParser(
        description="Simulate a PGA Tour season and estimate finish-position probabilities."
    )
    parser.add_argument(
        "--settings",
        default=DEFAULT_SETTINGS_PATH,
        help=f"Path to settings.yaml (default: {DEFAULT_SETTINGS_PATH})",
    )
    parser.add_argument(
        "--schedule",
        default=DEFAULT_SCHEDULE_PATH,
        help=f"Path to season_schedule.csv (default: {DEFAULT_SCHEDULE_PATH})",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    """Run the full simulation pipeline and write results to disk."""
    args = parse_args(argv)

    try:
        settings = load_settings(args.settings)
        schedule = load_season_schedule(args.schedule)
    except (SettingsError, ScheduleError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    csv_paths = sorted(glob(str(Path(settings.data.season_dir) / "*.csv")))
    if not csv_paths:
        print(f"Error: no CSV files found in {settings.data.season_dir}", file=sys.stderr)
        return 1

    moments = compute_player_stats(
        csv_paths,
        settings.data.player_column,
        settings.data.score_column,
        min_avg_rounds=settings.participation_weights.min_avg_rounds,
        weight_power=settings.participation_weights.weight_power,
        weight_floor=settings.participation_weights.weight_floor,
    )
    player_params = build_player_generators(moments)

    out_dir = Path(settings.output.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    simulate_season(
        player_params,
        schedule,
        seed=settings.monte_carlo.season_seed,
        dynamic_weight_config=None,
        output_csv_path=out_dir / settings.output.season_static_filename,
    )

    simulate_season(
        player_params,
        schedule,
        seed=settings.monte_carlo.season_seed,
        dynamic_weight_config=settings.dynamic_weights,
        output_csv_path=out_dir / settings.output.season_dynamic_filename,
    )

    mc_results = run_n_simulations(
        player_params,
        schedule,
        n=settings.monte_carlo.n_simulations,
        base_seed=settings.monte_carlo.base_seed,
        dynamic_weight_config=settings.dynamic_weights,
        output_csv_path=out_dir / settings.output.mc_results_filename,
    )

    print(mc_results.head(20).to_string(index=False))
    print(f"\nDone. Results written to {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
