"""qschool.py.

Simulates the Korn Ferry Tour Qualifying ("Q-School") gauntlet: a player
enters at stage 1, 2, or 3, must survive a 20%-advance cut at each of
stages 1-3 (each a 4-round tournament against a progressively stronger
field), and, if they reach the final, earns playing status by finish --
top 5 a PGA Tour card, the rest of the top 25% a full Korn Ferry card,
the next 25% conditional status, everyone else nothing.

Answers: for a player of a given caliber, how likely is each status
outcome, and how much does the entry stage matter?
"""

import numpy as np
import pandas as pd

from golf_simulator.distributions import sample_round_scores_for_players
from golf_simulator.points import top_n_with_ties
from golf_simulator.qschool_settings import QSchoolConfig

# Status outcomes, best to worst.
PGA_CARD = "PGA_CARD"
FULL_KF = "FULL_KF"
CONDITIONAL = "CONDITIONAL"
NONE = "NONE"

N_STAGES = 4  # stages 1-3 are advance-cuts; stage 4 is the status final
START_STAGES = (1, 2, 3)

_ASPIRANT_KEY = "__aspirant__"


def _shift_params(params: tuple, strokes: float) -> tuple:
    """
    Return player params made ``strokes`` better (lower scores) by shifting loc.

    Parameters
    ----------
    params : tuple
        (a, loc, scale, weight).
    strokes : float
        How many strokes better to make the player; subtracted from loc.

    Returns
    -------
    tuple
        (a, loc - strokes, scale, weight).
    """
    a, loc, scale, weight = params
    return (a, loc - strokes, scale, weight)


def validate_qschool_pools(
    aspirant_params: dict, competition_params: dict, config: QSchoolConfig
) -> None:
    """
    Check the aspirant/competition pools are disjoint and the field is fillable.

    Raises
    ------
    ValueError
        If the two pools share player id(s), or the competition pool is too
        small to fill a stage field alongside the aspirant.
    """
    overlap = set(aspirant_params) & set(competition_params)
    if overlap:
        raise ValueError(
            "aspirant_pool and competition_pool share player id(s): "
            f"{', '.join(str(pid) for pid in sorted(overlap, key=str))}. "
            "They must be disjoint populations."
        )

    needed = config.stage_field_size - 1
    if len(competition_params) < needed:
        raise ValueError(
            f"Not enough competition players to fill a stage field: the pool has "
            f"{len(competition_params)}, but each {config.stage_field_size}-player stage "
            f"needs {needed} opponents. Add more players to competition_pool or lower "
            "qschool.stage_field_size."
        )


def _final_tier(rank: int, config: QSchoolConfig) -> str:
    """Map a 1-based finishing rank in the final field to a status tier."""
    full_kf_cut = int(np.ceil(config.full_kf_pct * config.stage_field_size))
    conditional_cut = full_kf_cut + int(np.ceil(config.conditional_pct * config.stage_field_size))

    if rank <= config.pga_card_spots:
        return PGA_CARD
    if rank <= full_kf_cut:
        return FULL_KF
    if rank <= conditional_cut:
        return CONDITIONAL
    return NONE


def simulate_qschool_attempt(
    aspirant_params: tuple,
    competition_ids: list,
    competition_params: dict,
    config: QSchoolConfig,
    start_stage: int,
    rng: np.random.Generator,
) -> str:
    """
    Simulate one aspirant's gauntlet from ``start_stage`` through the final.

    Parameters
    ----------
    aspirant_params : tuple
        The aspirant's (a, loc, scale, weight).
    competition_ids : list
        Player ids available in the competition pool.
    competition_params : dict
        pid -> (a, loc, scale, weight) for the competition pool.
    config : QSchoolConfig
    start_stage : int
        1, 2, or 3 -- the stage the aspirant enters at.
    rng : numpy.random.Generator

    Returns
    -------
    str
        One of PGA_CARD, FULL_KF, CONDITIONAL, NONE. NONE covers both missing
        a 20% cut along the way and reaching the final outside the status
        tiers.
    """
    comp_ids = np.asarray(competition_ids)
    comp_weights = np.array(
        [float(competition_params[pid][3]) for pid in competition_ids], dtype=float
    )
    comp_weights = comp_weights / comp_weights.sum()

    n_opponents = config.stage_field_size - 1
    advance_n = max(1, int(np.ceil(config.stage_field_size * config.advance_pct)))

    for stage in range(start_stage, N_STAGES + 1):
        # Stronger field at later stages: shift opponents better by
        # (stage - 1) * strength_step strokes. Stage 1 is the base strength.
        toughening = (stage - 1) * config.strength_step

        drawn = rng.choice(comp_ids, size=n_opponents, replace=False, p=comp_weights)
        field_params = {
            str(pid): _shift_params(competition_params[pid], toughening) for pid in drawn
        }
        field_params[_ASPIRANT_KEY] = aspirant_params
        field_ids = list(field_params.keys())

        scores = sample_round_scores_for_players(
            field_params, config.rounds_per_stage, field_ids, random_state=rng
        )
        totals = scores.sum(axis=1)
        results = pd.DataFrame({"pid": field_ids, "total": totals})

        if stage < N_STAGES:
            advancers = top_n_with_ties(results, "pid", "total", advance_n)
            if _ASPIRANT_KEY not in advancers:
                return NONE
        else:
            aspirant_total = float(
                results.loc[results["pid"] == _ASPIRANT_KEY, "total"].iloc[0]
            )
            # Competition ranking: 1 + number of players who scored strictly lower.
            rank = 1 + int((results["total"] < aspirant_total).sum())
            return _final_tier(rank, config)

    return NONE  # unreachable, but keeps the return type total


def run_qschool(
    aspirant_params: dict,
    competition_params: dict,
    config: QSchoolConfig,
    output_csv_path=None,
) -> pd.DataFrame:
    """
    Estimate Q-School status probabilities per aspirant and entry stage.

    Each aspirant is run solo against fresh draws from the competition pool,
    starting at stages 1, 2, and 3, ``config.n_simulations`` times each.

    Parameters
    ----------
    aspirant_params : dict
        pid -> (a, loc, scale, weight). Keep small (a few skill levels).
    competition_params : dict
        pid -> (a, loc, scale, weight). The opponents (stage-1 strength).
    config : QSchoolConfig
    output_csv_path : str or None
        If provided, the summary DataFrame is written to this path.

    Returns
    -------
    pd.DataFrame
        One row per (aspirant, start stage), sorted by Player then Start_Stage.
        Columns: Player, Start_Stage, PGA_Card_pct, Full_KF_pct,
        Conditional_pct, Any_Status_pct.
    """
    validate_qschool_pools(aspirant_params, competition_params, config)

    competition_ids = [str(pid) for pid in competition_params.keys()]
    competition_params = {str(pid): v for pid, v in competition_params.items()}

    rows = []
    for aspirant_id, params in aspirant_params.items():
        for start_stage in START_STAGES:
            counts = {PGA_CARD: 0, FULL_KF: 0, CONDITIONAL: 0}
            for i in range(config.n_simulations):
                # Distinct, reproducible stream per simulation.
                rng = np.random.default_rng(config.base_seed + i)
                outcome = simulate_qschool_attempt(
                    params, competition_ids, competition_params, config, start_stage, rng
                )
                if outcome in counts:
                    counts[outcome] += 1

            n = config.n_simulations
            any_status = counts[PGA_CARD] + counts[FULL_KF] + counts[CONDITIONAL]
            rows.append({
                "Player": aspirant_id,
                "Start_Stage": start_stage,
                "PGA_Card_pct": round(counts[PGA_CARD] / n * 100, 2),
                "Full_KF_pct": round(counts[FULL_KF] / n * 100, 2),
                "Conditional_pct": round(counts[CONDITIONAL] / n * 100, 2),
                "Any_Status_pct": round(any_status / n * 100, 2),
            })

    results = (
        pd.DataFrame(rows)
        .sort_values(["Player", "Start_Stage"])
        .reset_index(drop=True)
    )

    if output_csv_path is not None:
        results.to_csv(output_csv_path, index=False)

    return results
