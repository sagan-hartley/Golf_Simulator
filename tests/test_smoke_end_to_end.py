"""End-to-end smoke test: synthetic data through the full pipeline.

Asserts the pipeline produces a sane season summary and that the
simulation is fully deterministic given its seed: the same seed
reproduces identical output, and different seeds diverge.
"""

import numpy as np
import pandas as pd
import pytest

from golf_simulator.data_loading import compute_player_stats
from golf_simulator.distributions import build_player_generators
from golf_simulator.domain import TournamentType
from golf_simulator.season import simulate_season
from golf_simulator.settings import DynamicWeightConfig

N_PLAYERS = 170  # must exceed the largest TournamentType field_size (156, REGULAR)
N_ROUNDS_PER_PLAYER = 40


@pytest.fixture
def synthetic_csv(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(N_PLAYERS):
        player = f"Player{i}"
        base = 68 + (i % 10)
        scores = np.rint(rng.normal(loc=base, scale=2.5, size=N_ROUNDS_PER_PLAYER))
        for s in scores:
            rows.append({"player": player, "score": s})

    path = tmp_path / "synthetic_season.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


@pytest.fixture
def player_params(synthetic_csv):
    moments = compute_player_stats(
        [synthetic_csv], player_col="player", value_col="score", min_avg_rounds=5
    )
    return build_player_generators(moments)


def test_season_summary_has_valid_ranks_and_no_nans(player_params):
    schedule = [TournamentType.REGULAR, TournamentType.SIGNATURE_NO_CUT]

    summary, _, _ = simulate_season(player_params, schedule, seed=123)

    assert not summary["SeasonPoints"].isna().any()
    assert sorted(summary["SeasonRank"].tolist()) == list(range(1, len(summary) + 1))


def test_same_seed_produces_identical_output(player_params):
    # `seed=` fully determines both field selection and round-score sampling,
    # so two runs with the same seed must produce byte-identical output.
    schedule = [TournamentType.REGULAR, TournamentType.SIGNATURE_CUT]
    dw_cfg = DynamicWeightConfig(enabled=True)

    summary_1, _, _ = simulate_season(
        player_params, schedule, seed=42, dynamic_weight_config=dw_cfg
    )
    summary_2, _, _ = simulate_season(
        player_params, schedule, seed=42, dynamic_weight_config=dw_cfg
    )

    pd.testing.assert_frame_equal(summary_1, summary_2)


def test_different_seeds_produce_different_output(player_params):
    schedule = [TournamentType.REGULAR, TournamentType.SIGNATURE_CUT]

    summary_1, _, _ = simulate_season(player_params, schedule, seed=1)
    summary_2, _, _ = simulate_season(player_params, schedule, seed=2)

    assert not summary_1["SeasonPoints"].equals(summary_2["SeasonPoints"])
