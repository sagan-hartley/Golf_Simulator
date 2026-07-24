"""monday_chase.py.

Simulates "chasing Mondays": a non-exempt player competes in a 1-round
Monday qualifier for a handful of spots into that week's regular Tour
event: if they then finish well enough, they earn direct entry into
next week's field and skip Monday qualifying (a "parlay"); otherwise
they're back to grinding Mondays. Answers: what's the probability of
ever advancing through Monday, and of ever parlaying, over N weeks --
and how does that change with player skill and pool size.
"""

import numpy as np
import pandas as pd

from golf_simulator.distributions import add_skill_columns, sample_round_scores_for_players
from golf_simulator.domain import TournamentType
from golf_simulator.monday_chase_settings import ChaseConfig
from golf_simulator.points import top_n_with_ties
from golf_simulator.season import play_event, validate_field_size

MAIN_EVENT_TYPE = TournamentType.REGULAR


def simulate_monday_chase(
    aspirant_params: dict,
    main_event_params: dict,
    config: ChaseConfig,
    seed: int,
) -> pd.DataFrame:
    """
    Simulate one full `config.n_weeks` Monday-qualifier chase for every aspirant.

    All aspirants start week 1 non-exempt (a player's skill is treated as
    fixed across the chase, matching how `simulate_season` treats skill as
    fixed for a season). Each week, aspirants who aren't currently exempt
    make up the *entire* Monday field (there is no separate Monday-field-size
    setting -- every non-exempt aspirant in the pool attempts it, matching
    how a real Monday qualifier doesn't pre-select who shows up).

    Parameters
    ----------
    aspirant_params : dict
        pid -> (a, loc, scale, weight). The players chasing Mondays.
    main_event_params : dict
        pid -> (a, loc, scale, weight). The field they're trying to crash;
        must have at least `TournamentType.REGULAR`'s field size (156) and
        share no player ids with `aspirant_params`.
    config : ChaseConfig
    seed : int

    Returns
    -------
    pd.DataFrame
        One row per aspirant per week (long format). Columns: Player, Week,
        AttemptedMonday, AdvancedMonday, PlayedMainEvent, FinalRank,
        Parlayed.

    Raises
    ------
    ValueError
        If the two pools share player id(s), if `main_event_params` is
        smaller than the main event's field size, or if a week's combined
        exempt + Monday-qualified aspirant count exceeds the main event's
        field size (an oversubscribed-field settings combination).
    """
    aspirant_ids = [str(pid) for pid in aspirant_params.keys()]
    main_ids = [str(pid) for pid in main_event_params.keys()]

    overlap = set(aspirant_ids) & set(main_ids)
    if overlap:
        raise ValueError(
            "aspirant_pool and main_event_pool share player id(s): "
            f"{', '.join(sorted(overlap))}. They must be disjoint populations."
        )

    validate_field_size(main_event_params, [MAIN_EVENT_TYPE])

    field_size = MAIN_EVENT_TYPE.value.field_size
    merged_params = {**main_event_params, **aspirant_params}

    main_weights = np.array(
        [float(main_event_params[pid][3]) for pid in main_ids], dtype=float
    )
    main_weights /= main_weights.sum()

    rng = np.random.default_rng(seed)
    has_exemption = {pid: False for pid in aspirant_ids}
    rows = []

    for week in range(1, config.n_weeks + 1):
        exempt_entrants = [pid for pid in aspirant_ids if has_exemption[pid]]
        must_attempt = [pid for pid in aspirant_ids if not has_exemption[pid]]

        advanced = []
        if must_attempt:
            monday_scores = sample_round_scores_for_players(
                aspirant_params, 1, must_attempt, random_state=rng
            )[:, 0]
            monday_df = pd.DataFrame({"Player": must_attempt, "Score": monday_scores})
            n_advance = max(1, int(np.ceil(len(must_attempt) * config.advance_pct)))
            monday_advance_set = top_n_with_ties(monday_df, "Player", "Score", n_advance)
            advanced = [pid for pid in must_attempt if pid in monday_advance_set]

        aspirant_entrants = exempt_entrants + advanced

        if len(aspirant_entrants) > field_size:
            raise ValueError(
                f"Week {week}: {len(aspirant_entrants)} aspirants (exempt + Monday "
                f"qualifiers) exceed the main event's field size of {field_size}. "
                "Lower chase.advance_pct, raise chase.parlay_top_n's selectivity, "
                "or use a smaller aspirant pool."
            )

        remaining_slots = field_size - len(aspirant_entrants)
        filler = rng.choice(
            main_ids, size=remaining_slots, replace=False, p=main_weights
        ).tolist()

        full_field = aspirant_entrants + filler
        results, _, _, _ = play_event(merged_params, full_field, MAIN_EVENT_TYPE, rng)
        rank_lookup = results.set_index("Player")["FinalRank"].to_dict()

        advanced_set = set(advanced)
        for pid in aspirant_ids:
            played_main = pid in aspirant_entrants
            final_rank = rank_lookup.get(pid, np.nan) if played_main else np.nan
            made_parlay_cutoff = not np.isnan(final_rank) and final_rank <= config.parlay_top_n
            parlayed = bool(played_main and made_parlay_cutoff)

            rows.append({
                "Player": pid,
                "Week": week,
                "AttemptedMonday": pid in must_attempt,
                "AdvancedMonday": pid in advanced_set,
                "PlayedMainEvent": played_main,
                "FinalRank": final_rank,
                "Parlayed": parlayed,
            })

            has_exemption[pid] = parlayed

    return pd.DataFrame(rows)


def run_n_monday_chases(
    aspirant_params: dict,
    main_event_params: dict,
    config: ChaseConfig,
    output_csv_path=None,
) -> pd.DataFrame:
    """
    Run `config.n_simulations` independent chases and aggregate per aspirant.

    Each aspirant's `n_weeks` of outcomes are first collapsed to
    per-simulation booleans/counts before averaging across simulations --
    averaging the raw per-week rows directly would bias results, since an
    exempt week and a Monday-attempt week aren't comparable events.

    Parameters
    ----------
    aspirant_params : dict
    main_event_params : dict
    config : ChaseConfig
    output_csv_path : str or None
        If provided, the summary DataFrame is written to this path.

    Returns
    -------
    pd.DataFrame
        One row per aspirant, sorted by Any_Parlay_pct descending. Columns:
        Player, Any_Monday_Advance_pct, Any_Parlay_pct, Avg_Weeks_Played,
        Avg_Monday_Attempts, Advance_Rate_Per_Monday_Attempt.

        Advance_Rate_Per_Monday_Attempt (advances / attempts, not / n_weeks)
        is the fair way to compare skill levels: a skilled aspirant earns
        exemptions and so attempts fewer Mondays, which would otherwise
        understate their true Monday ability if only judged per n_weeks.
    """
    aspirant_ids = [str(pid) for pid in aspirant_params.keys()]

    per_sim_records = []
    for i in range(config.n_simulations):
        weekly = simulate_monday_chase(
            aspirant_params, main_event_params, config, seed=config.base_seed + i
        )
        grouped = weekly.groupby("Player")

        for pid in aspirant_ids:
            g = grouped.get_group(pid)
            per_sim_records.append({
                "Player": pid,
                "any_monday_advance": bool(g["AdvancedMonday"].any()),
                "any_parlay": bool(g["Parlayed"].any()),
                "weeks_played": int(g["PlayedMainEvent"].sum()),
                "monday_attempts": int(g["AttemptedMonday"].sum()),
                "monday_advances": int(g["AdvancedMonday"].sum()),
            })

    per_sim_df = pd.DataFrame(per_sim_records)

    rows = []
    for pid, g in per_sim_df.groupby("Player"):
        total_attempts = g["monday_attempts"].sum()
        total_advances = g["monday_advances"].sum()
        advance_rate = (
            round(total_advances / total_attempts * 100, 2) if total_attempts > 0 else np.nan
        )

        rows.append({
            "Player": pid,
            "Any_Monday_Advance_pct": round(g["any_monday_advance"].mean() * 100, 2),
            "Any_Parlay_pct": round(g["any_parlay"].mean() * 100, 2),
            "Avg_Weeks_Played": round(g["weeks_played"].mean(), 2),
            "Avg_Monday_Attempts": round(g["monday_attempts"].mean(), 2),
            "Advance_Rate_Per_Monday_Attempt": advance_rate,
        })

    results = (
        pd.DataFrame(rows)
        .sort_values("Any_Parlay_pct", ascending=False)
        .reset_index(drop=True)
    )
    results = add_skill_columns(results, aspirant_params)

    if output_csv_path is not None:
        results.to_csv(output_csv_path, index=False)

    return results
