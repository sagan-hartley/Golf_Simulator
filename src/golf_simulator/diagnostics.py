"""diagnostics.py.

Developer-facing plotting and comparison helpers. Not part of the
default `golf-sim` run — import and call these interactively when
investigating dynamic-weight behavior.
"""

import matplotlib.pyplot as plt
import pandas as pd


def plot_weight_trajectory(weight_history: pd.DataFrame, player_id: str,
                            baseline_weights: dict):
    """
    Plot how a single player's participation weight evolved across the season.

    Parameters
    ----------
    weight_history : pd.DataFrame
        Returned by simulate_season when dynamic weights are enabled.
    player_id : str
    baseline_weights : dict
    """
    if player_id not in weight_history.columns:
        print(f"Player '{player_id}' not found in weight history.")
        return

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(weight_history.index, weight_history[player_id],
            marker="o", label="dynamic weight")
    ax.axhline(baseline_weights[player_id], linestyle="--",
               color="grey", label="baseline weight")
    ax.set_title(f"Participation weight over season — {player_id}")
    ax.set_xlabel("Event index")
    ax.set_ylabel("Weight (normalised)")
    ax.legend()
    plt.tight_layout()
    plt.show()


def compare_appearance_counts(events_a: list, events_b: list,
                               label_a="static", label_b="dynamic",
                               top_n=20) -> pd.DataFrame:
    """
    Compare how many events each player appeared in across two simulations.

    Parameters
    ----------
    events_a : list[pd.DataFrame]
    events_b : list[pd.DataFrame]
    label_a : str
    label_b : str
    top_n : int

    Returns
    -------
    pd.DataFrame
    """
    def count(event_list):
        return (
            pd.concat([df[["Player"]] for df in event_list], ignore_index=True)
            ["Player"].value_counts()
        )

    df = pd.DataFrame({
        label_a: count(events_a),
        label_b: count(events_b),
    }).fillna(0).astype(int)

    df["diff"] = df[label_b] - df[label_a]
    return df.sort_values("diff", ascending=False).head(top_n)
