# Golf Simulator

A Monte Carlo simulator for PGA Tour season standings. It fits a per-player
skew-normal scoring distribution from historical round-by-round data, then
simulates full seasons — tournament by tournament, with cuts, points, and
(optionally) rank-based dynamic participation weights — to estimate each
player's probability of winning, finishing top-10, top-20, or top-50, and
their average season rank.

Not a coder? See **[docs/USER_GUIDE.md](docs/USER_GUIDE.md)** — it walks
through changing inputs and running the simulator without touching any code.

## Install

Requires Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Quick start

```bash
golf-sim
```

This reads [`config/settings.yaml`](config/settings.yaml) and
[`config/season_schedule.csv`](config/season_schedule.csv), fits player
distributions from the CSVs in `data/seasons/` (or loads a custom field
directly if `data.field_file` is set — see USER_GUIDE), and writes three
result files to `outputs/`:

- `season_static.csv` — one simulated season, fixed participation odds
- `season_dynamic.csv` — one simulated season, with dynamic weight nudging
- `mc_results.csv` — win/top-10/top-20/top-50 probabilities across many
  simulated seasons

Run `golf-sim --help` for CLI options (custom settings/schedule paths).

## Architecture

```
config/settings.yaml         user-tunable scalar settings (see USER_GUIDE)
config/season_schedule.csv   the season calendar (see USER_GUIDE)
data/seasons/*.csv           historical round-score data, one file per season
data/custom_fields/          example synthetic/hypothetical field files (see USER_GUIDE)
src/golf_simulator/
    domain.py                 fixed tournament structure: enums, points tables
    settings.py                settings.yaml loader/validator
    schedule.py                 season_schedule.csv loader/validator
    data_loading.py            historical data -> per-player mean/variance/skew
    player_field.py            custom field file loader (alternative to historical data)
    distributions.py           skew-normal fitting + score sampling
    points.py                  points-with-ties assignment, cut logic
    weights.py                  dynamic (rank-based) weight nudging
    season.py                   simulate_season: one full season
    monte_carlo.py              run_n_simulations: many seasons, aggregated
    diagnostics.py              plotting/comparison helpers (interactive use)
    cli.py                      `golf-sim` entry point
tests/                        pytest suite, one file per module above
```

## Development

```bash
pytest -v            # run tests
ruff check src tests # lint (PEP 8 / PEP 257 / numpy-style docstrings)
```

CI (`.github/workflows/ci.yml`) runs both on every push/PR to `main`.

## License

MIT — see [LICENSE](LICENSE).
