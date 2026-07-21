"""cli.py.

Command-line entry point (installed as the ``golf-sim`` console
script), with two subcommands:

``golf-sim season`` (the default when no subcommand is given, for
backward compatibility) -- loads config/settings.yaml and
config/season_schedule.csv, builds player distributions, simulates one
static season, one dynamic-weight season, and a Monte Carlo batch of
seasons, and writes all three results to the configured output folder.

``golf-sim monday-chase`` -- loads config/monday_chase.yaml, builds an
aspirant pool and a main-event pool, and simulates the Monday-qualifier
chase analysis (see golf_simulator.monday_chase).
"""

import argparse
import sys
from pathlib import Path

from golf_simulator.monday_chase import run_n_monday_chases
from golf_simulator.monday_chase_settings import (
    DEFAULT_MONDAY_CHASE_SETTINGS_PATH,
    MondayChaseSettingsError,
    load_monday_chase_settings,
)
from golf_simulator.monte_carlo import run_n_simulations
from golf_simulator.player_field import FieldError, load_player_pool
from golf_simulator.schedule import DEFAULT_SCHEDULE_PATH, ScheduleError, load_season_schedule
from golf_simulator.season import simulate_season
from golf_simulator.settings import (
    DEFAULT_SETTINGS_PATH,
    ParticipationWeightConfig,
    SettingsError,
    load_settings,
)

_SUBCOMMANDS = ("season", "monday-chase")


def parse_args(argv=None):
    """Parse command-line arguments for the `golf-sim` entry point."""
    if argv is None:
        argv = sys.argv[1:]
    argv = list(argv)

    # No subcommand given -> default to "season" so `golf-sim` (with or
    # without --settings/--schedule) keeps working exactly as before this
    # subcommand split was introduced. Exception: `golf-sim -h`/`--help`
    # with no subcommand should show the top-level help (listing both
    # subcommands), not get silently rewritten into `season --help`.
    is_bare_help = argv[:1] in (["-h"], ["--help"])
    if not is_bare_help and (not argv or argv[0] not in _SUBCOMMANDS):
        argv = ["season", *argv]

    parser = argparse.ArgumentParser(
        description="Simulate a PGA Tour season, or run a specific-question analysis."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    season_parser = subparsers.add_parser("season", help="Simulate a full season (default)")
    season_parser.add_argument(
        "--settings",
        default=DEFAULT_SETTINGS_PATH,
        help=f"Path to settings.yaml (default: {DEFAULT_SETTINGS_PATH})",
    )
    season_parser.add_argument(
        "--schedule",
        default=DEFAULT_SCHEDULE_PATH,
        help=f"Path to season_schedule.csv (default: {DEFAULT_SCHEDULE_PATH})",
    )

    chase_parser = subparsers.add_parser(
        "monday-chase", help="Simulate the Monday-qualifier chase"
    )
    chase_parser.add_argument(
        "--settings",
        default=DEFAULT_MONDAY_CHASE_SETTINGS_PATH,
        help=f"Path to monday_chase.yaml (default: {DEFAULT_MONDAY_CHASE_SETTINGS_PATH})",
    )

    return parser.parse_args(argv)


def _run_season(args) -> int:
    """Run the full season simulation pipeline and write results to disk."""
    try:
        settings = load_settings(args.settings)
        schedule = load_season_schedule(args.schedule)
        player_params = load_player_pool(settings.data, settings.participation_weights)
    except (SettingsError, ScheduleError, FieldError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    out_dir = Path(settings.output.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
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
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(mc_results.head(20).to_string(index=False))
    print(f"\nDone. Results written to {out_dir}/")
    return 0


def _run_monday_chase(args) -> int:
    """Run the Monday-qualifier chase analysis and write results to disk."""
    try:
        settings = load_monday_chase_settings(args.settings)
        default_participation = ParticipationWeightConfig()
        aspirant_params = load_player_pool(settings.aspirant_pool, default_participation)
        main_event_params = load_player_pool(settings.main_event_pool, default_participation)
    except (MondayChaseSettingsError, FieldError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    out_dir = Path(settings.output.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        results = run_n_monday_chases(
            aspirant_params,
            main_event_params,
            settings.chase,
            output_csv_path=out_dir / settings.output.filename,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(results.head(20).to_string(index=False))
    print(f"\nDone. Results written to {out_dir}/")
    return 0


def main(argv=None) -> int:
    """Dispatch to the requested subcommand (default: `season`)."""
    args = parse_args(argv)
    if args.command == "monday-chase":
        return _run_monday_chase(args)
    return _run_season(args)


if __name__ == "__main__":
    sys.exit(main())
