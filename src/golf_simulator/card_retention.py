"""card_retention.py.

Simulates "card retention" under a proposed new PGA Tour alignment: a
fixed pool of ~130 "carded" players compete in mostly-120-player
events plus 4 majors (topped up from a second "outside qualifiers"
pool, since a 130-player card pool alone can't fill a 156-player
major field). Answers: what's the probability each card-pool player
finishes ranked well enough (a configurable cutoff) to keep their
card for next season.
"""

from typing import Optional

import numpy as np
import pandas as pd

from golf_simulator.distributions import add_skill_columns
from golf_simulator.domain import TournamentType
from golf_simulator.season import play_event
from golf_simulator.settings import DynamicWeightConfig
from golf_simulator.weights import nudge_weights

# The 4 true majors -- deliberately an explicit set, not derived from
# EventType.MAJOR_PLAYERS (that points-table grouping also includes
# PLAYERS/PLAYOFF, which are Tour-member-only invitationals in reality,
# not open to outside qualifiers).
TRUE_MAJORS = frozenset({
    TournamentType.MAJOR_MASTERS,
    TournamentType.MAJOR_US_OPEN,
    TournamentType.MAJOR_PGA,
    TournamentType.MAJOR_OPEN,
})


def validate_alignment_pools(card_params: dict, outside_params: dict, schedule: list) -> None:
    """
    Check the card/outside pools are disjoint and large enough for `schedule`.

    Parameters
    ----------
    card_params : dict
        pid -> (a, loc, scale, weight). The fixed eligible pool.
    outside_params : dict
        pid -> (a, loc, scale, weight). Fills major fields beyond the card pool.
    schedule : list[TournamentType]

    Raises
    ------
    ValueError
        If the two pools share player id(s), if `card_params` is smaller
        than the largest non-major event's field size, or if
        `card_params` + `outside_params` combined can't fill the largest
        major's field size.
    """
    card_ids = set(card_params.keys())
    outside_ids = set(outside_params.keys())

    overlap = card_ids & outside_ids
    if overlap:
        raise ValueError(
            "card_pool and outside_pool share player id(s): "
            f"{', '.join(str(pid) for pid in sorted(overlap, key=str))}. "
            "They must be disjoint populations."
        )

    non_major_field_sizes = [t.value.field_size for t in schedule if t not in TRUE_MAJORS]
    max_non_major_field = max(non_major_field_sizes, default=0)
    if len(card_params) < max_non_major_field:
        raise ValueError(
            f"Not enough card_pool players to simulate this schedule: it has "
            f"{len(card_params)} player(s), but the largest non-major event needs a "
            f"field of {max_non_major_field}. Add more players to card_pool."
        )

    major_field_sizes = [t.value.field_size for t in schedule if t in TRUE_MAJORS]
    max_major_field = max(major_field_sizes, default=0)
    combined = len(card_params) + len(outside_params)
    if combined < max_major_field:
        raise ValueError(
            f"Not enough players to fill the largest major's field: card_pool + "
            f"outside_pool together have {combined} player(s), but the largest major "
            f"needs a field of {max_major_field}. Add more players to card_pool or "
            "outside_pool."
        )


def simulate_alignment_season(
    card_params: dict,
    outside_params: dict,
    schedule: list,
    seed: int = 123,
    dynamic_weight_config: Optional[DynamicWeightConfig] = None,
) -> pd.DataFrame:
    """
    Simulate one alignment season and return card-pool season standings.

    Non-major events draw their field entirely from `card_params`. Major
    events draw `min(len(card_params), field_size)` players from
    `card_params` (in practice, every card holder is eligible) plus
    however many more are needed from `outside_params` to fill the field.
    Only `card_params` players' points/weight-nudging are tracked --
    `outside_params` players are pure filler for major fields.

    Parameters
    ----------
    card_params : dict
        pid -> (a, loc, scale, weight). The fixed eligible pool.
    outside_params : dict
        pid -> (a, loc, scale, weight). Fills major fields beyond the card pool.
    schedule : list[TournamentType]
    seed : int
    dynamic_weight_config : DynamicWeightConfig or None
        None disables dynamic weights. Applies only within the card pool.

    Returns
    -------
    pd.DataFrame
        Columns: Player, SeasonPoints, SeasonRank. One row per card-pool
        player; outside-pool players never appear.
    """
    validate_alignment_pools(card_params, outside_params, schedule)

    card_ids = [str(pid) for pid in card_params.keys()]
    outside_ids = [str(pid) for pid in outside_params.keys()]
    merged_params = {**card_params, **outside_params}

    baseline_weights = {pid: float(card_params[pid][3]) for pid in card_ids}
    current_weights = dict(baseline_weights)

    if dynamic_weight_config is None:
        dynamic_weight_config = DynamicWeightConfig(enabled=False)

    outside_weights = np.array(
        [float(outside_params[pid][3]) for pid in outside_ids], dtype=float
    )
    if outside_weights.size and outside_weights.sum() > 0:
        outside_weights = outside_weights / outside_weights.sum()

    season_points = {pid: 0.0 for pid in card_ids}
    rng = np.random.default_rng(seed)

    for tournament_type in schedule:
        field_size = tournament_type.value.field_size

        card_p = np.array([current_weights[pid] for pid in card_ids], dtype=float)
        card_p /= card_p.sum()

        if tournament_type in TRUE_MAJORS:
            card_slots = min(len(card_ids), field_size)
            outside_slots = field_size - card_slots

            card_field = rng.choice(
                card_ids, size=card_slots, replace=False, p=card_p
            ).tolist()
            if outside_slots > 0:
                outside_field = rng.choice(
                    outside_ids, size=outside_slots, replace=False, p=outside_weights
                ).tolist()
            else:
                outside_field = []
            field = card_field + outside_field
        else:
            field = rng.choice(card_ids, size=field_size, replace=False, p=card_p).tolist()

        results, _, _, _ = play_event(merged_params, field, tournament_type, rng)
        card_results = results[results["Player"].isin(card_ids)]

        for row in card_results.itertuples(index=False):
            season_points[str(row.Player)] += float(row.Points)

        if dynamic_weight_config.enabled:
            current_weights = nudge_weights(
                current_weights, baseline_weights, card_results, dynamic_weight_config
            )

    season_summary = (
        pd.DataFrame(
            [(pid, season_points[pid]) for pid in card_ids],
            columns=["Player", "SeasonPoints"],
        )
        .sort_values("SeasonPoints", ascending=False)
        .reset_index(drop=True)
    )
    season_summary["SeasonRank"] = np.arange(1, len(season_summary) + 1)

    return season_summary


def run_n_alignment_seasons(
    card_params: dict,
    outside_params: dict,
    schedule: list,
    n: int,
    base_seed: int = 0,
    dynamic_weight_config: Optional[DynamicWeightConfig] = None,
    retention_cutoff: int = 90,
    output_csv_path=None,
) -> pd.DataFrame:
    """
    Run n independent alignment seasons and estimate card-retention probability.

    Parameters
    ----------
    card_params : dict
    outside_params : dict
    schedule : list[TournamentType]
    n : int
        Number of seasons to simulate.
    base_seed : int
        Seeds run from base_seed to base_seed + n - 1.
    dynamic_weight_config : DynamicWeightConfig or None
    retention_cutoff : int
        A player ranked this or better (within the card pool) keeps their
        card for that simulated season.
    output_csv_path : str or None
        If provided, the summary DataFrame is written to this path.

    Returns
    -------
    pd.DataFrame
        One row per card-pool player, sorted by Retained_Card_pct
        descending. Columns: Player, Retained_Card_pct, Avg_SeasonRank.
    """
    card_ids = [str(pid) for pid in card_params.keys()]
    rank_sums = {pid: 0 for pid in card_ids}
    retained_counts = {pid: 0 for pid in card_ids}

    for i in range(n):
        summary = simulate_alignment_season(
            card_params,
            outside_params,
            schedule,
            seed=base_seed + i,
            dynamic_weight_config=dynamic_weight_config,
        )
        rank_lookup = summary.set_index("Player")["SeasonRank"].to_dict()

        for pid in card_ids:
            rank = rank_lookup[pid]
            rank_sums[pid] += rank
            if rank <= retention_cutoff:
                retained_counts[pid] += 1

    rows = [
        {
            "Player": pid,
            "Retained_Card_pct": round(retained_counts[pid] / n * 100, 2),
            "Avg_SeasonRank": round(rank_sums[pid] / n, 2),
        }
        for pid in card_ids
    ]

    results = (
        pd.DataFrame(rows)
        .sort_values("Retained_Card_pct", ascending=False)
        .reset_index(drop=True)
    )
    results = add_skill_columns(results, card_params)

    if output_csv_path is not None:
        results.to_csv(output_csv_path, index=False)

    return results
