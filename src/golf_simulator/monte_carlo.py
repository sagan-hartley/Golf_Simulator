"""monte_carlo.py.

Runs many independent season simulations and aggregates finish-position
probabilities per player.
"""

import pandas as pd

from golf_simulator.distributions import add_skill_columns
from golf_simulator.season import simulate_season

# ── Monte Carlo wrapper ────────────────────────────────────────────────────────


def run_n_simulations(
    player_params: dict,
    schedule: list,
    n: int = 100,
    base_seed: int = 0,
    dynamic_weight_config=None,
    output_csv_path=None,
) -> "pd.DataFrame":
    """
    Run n independent season simulations and summarise finish-position probabilities.

    Each simulation uses a distinct seed (base_seed + i) so results are
    independent but reproducible.

    Parameters
    ----------
    player_params : dict
        pid -> (a, loc, scale, weight)
    schedule : list[TournamentType]
    n : int
        Number of simulations to run.
    base_seed : int
        Seeds run from base_seed to base_seed + n - 1.
    dynamic_weight_config : DynamicWeightConfig or None
    output_csv_path : str or None
        If provided, the summary DataFrame is written to this path.

    Returns
    -------
    pd.DataFrame
        One row per player, sorted by Win_pct descending. Columns:
        Player, Win_pct, Top10_pct, Top20_pct, Top50_pct, Avg_rank
    """
    pids = list(player_params.keys())
    counts = {pid: {"win": 0, "top10": 0, "top20": 0, "top50": 0, "rank_sum": 0}
              for pid in pids}

    for i in range(n):
        season_summary, _, _ = simulate_season(
            player_params,
            schedule,
            seed=base_seed + i,
            dynamic_weight_config=dynamic_weight_config,
        )

        rank_lookup = season_summary.set_index("Player")["SeasonRank"].to_dict()

        for pid in pids:
            rank = rank_lookup.get(pid)
            if rank is None:
                continue
            counts[pid]["rank_sum"] += rank
            if rank == 1:
                counts[pid]["win"]   += 1
            if rank <= 10:
                counts[pid]["top10"] += 1
            if rank <= 20:
                counts[pid]["top20"] += 1
            if rank <= 50:
                counts[pid]["top50"] += 1

    rows = []
    for pid in pids:
        c = counts[pid]
        rows.append({
            "Player":    pid,
            "Win_pct":   round(c["win"]   / n * 100, 2),
            "Top10_pct": round(c["top10"] / n * 100, 2),
            "Top20_pct": round(c["top20"] / n * 100, 2),
            "Top50_pct": round(c["top50"] / n * 100, 2),
            "Avg_rank":  round(c["rank_sum"] / n, 2),
        })

    results = (
        pd.DataFrame(rows)
        .sort_values("Win_pct", ascending=False)
        .reset_index(drop=True)
    )
    results = add_skill_columns(results, player_params)

    if output_csv_path is not None:
        results.to_csv(output_csv_path, index=False)

    return results
