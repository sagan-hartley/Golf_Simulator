# User Guide (no coding experience required)

This guide is for anyone comfortable in Excel who wants to change what this
simulator does — without reading or writing any code.

## What this tool actually does

Think of it like a giant, very fast spreadsheet Monte Carlo model. It:

1. Looks at how every PGA Tour player has actually scored in past seasons
   (their average, how spread out their scores are, and whether they tend
   to have more really-good rounds or really-bad rounds).
2. Uses that to simulate an entire *fake* season — every tournament, every
   cut, every points table — the same way you might simulate thousands of
   coin flips in Excel to estimate probabilities.
3. Repeats that simulated season many times (this is the "Monte Carlo"
   part) and counts how often each player wins, finishes in the top 10, top
   20, or top 50.

The two things you can change are **how the model behaves** (settings) and
**what tournaments happen, in what order** (the schedule). You never need to
open or edit any `.py` file.

## One-time setup (macOS)

You only need to do this once.

1. Open the **Terminal** app (press `Cmd+Space`, type `Terminal`, press
   Enter).
2. Go to the project folder:
   ```
   cd ~/VSCodeProjects/Golf_Simulator
   ```
3. Create a private Python environment for this project (this keeps it from
   interfering with anything else on your computer):
   ```
   python3 -m venv .venv
   source .venv/bin/activate
   ```
4. Install the simulator:
   ```
   pip install -e .
   ```

You'll know it worked if the last line printed something like
`Successfully installed golf-simulator-0.1.0`.

## The two files you're meant to edit

### 1. `config/settings.yaml` — how the model behaves

Open this file in any text editor (double-click it, or open it in VS Code /
TextEdit / Notepad). Every line is `setting_name: value`, with a `#` comment
above it explaining what it does.

**Example — making the model react more strongly to hot/cold streaks:**

Before:
```yaml
dynamic_weights:
  enabled: true
  nudge_amount: 0.05
```

After (bigger number = stronger reaction to recent form):
```yaml
dynamic_weights:
  enabled: true
  nudge_amount: 0.08
```

That's it — save the file. Two rules to avoid breaking it:

- Use **spaces**, never the Tab key, for indentation. Keep each line lined
  up exactly like the lines around it.
- Always put a space after the colon (`nudge_amount: 0.08`, not
  `nudge_amount:0.08`).

**Example — running more simulated seasons for more stable probabilities:**

Before:
```yaml
monte_carlo:
  n_simulations: 10
```

After:
```yaml
monte_carlo:
  n_simulations: 200
```

(More simulations = more accurate probabilities, but a slower run.)

### 2. `config/season_schedule.csv` — the season calendar

This is a plain CSV file, so you can open it directly in **Excel** (or
Numbers, or Google Sheets). It has two columns:

| event_number | tournament_type |
|---|---|
| 1 | REGULAR |
| 2 | REGULAR |
| ... | ... |
| 13 | MAJOR_MASTERS |
| ... | ... |

**To change the order of two tournaments:** swap the values in the
`tournament_type` column for the two rows (leave `event_number` as-is, or
renumber — either works, since the program sorts by `event_number` before
running).

**To add or remove a tournament:** insert or delete a row, then make sure
the `event_number` column still counts up with no repeats and no gaps
(1, 2, 3, ... with no skipped numbers), and save.

**Valid values for `tournament_type`** (spelling must match one of these
exactly — capitalization doesn't matter):

```
REGULAR, SIGNATURE_NO_CUT, SIGNATURE_CUT, MAJOR_MASTERS, MAJOR_US_OPEN,
MAJOR_PGA, MAJOR_OPEN, PLAYERS, PLAYOFF
```

Save the file as CSV (if Excel asks "Keep current format" or similar when
saving a `.csv`, choose to keep the CSV format).

## Running the simulation

Each time you want to run it:

1. Open Terminal.
2. Go to the folder and turn on the project's Python environment:
   ```
   cd ~/VSCodeProjects/Golf_Simulator
   source .venv/bin/activate
   ```
3. Run it:
   ```
   golf-sim
   ```

It'll print a preview table and finish with `Done. Results written to
outputs/`.

## Where to find your results

Open the `outputs/` folder — you can open any of these directly in Excel:

- **`season_static.csv`** — one simulated season where every player's odds
  of being picked for each tournament field stay fixed all year.
- **`season_dynamic.csv`** — the same single season, but players who are
  finishing well get slightly better odds as the season goes on (and
  players finishing poorly get slightly worse odds) — controlled by the
  `dynamic_weights` settings.
- **`mc_results.csv`** — the main output: for every player, their
  percentage chance of winning (`Win_pct`), finishing top 10, top 20, top
  50, and their average finishing rank, based on many simulated seasons.

## Troubleshooting

| What you see | What it means | What to do |
|---|---|---|
| `command not found: golf-sim` | Your project environment isn't turned on | Run `source .venv/bin/activate`, then try again |
| `Error: config/settings.yaml: ... must be between 0 and 1 (got 1.5)` | A setting is outside its allowed range | Open `settings.yaml`, find the named setting, fix the value (the message tells you exactly which one and its valid range) |
| `Error: config/settings.yaml: could not parse YAML` | Indentation or a missing colon broke the file | Check that you used spaces (not Tab) and that every setting has a space after its colon |
| `Error: ... row with event_number=7 has unknown tournament_type 'REGULER'` | A typo in the schedule CSV | Fix the spelling to match one of the valid values listed above |
| `Error: no CSV files found in data/seasons` | You're running the command from the wrong folder | Run `cd ~/VSCodeProjects/Golf_Simulator` first |

If an error message doesn't match anything above, copy the exact text of
the error — it's written in plain language and names the specific file and
setting that caused it.

## Glossary

- **Seed** — a starting number for the random number generator. Changing it
  gives you a different (but equally valid) simulated outcome; keeping it
  the same lets you reproduce a specific result.
- **Monte Carlo simulation** — running the same random process many times
  and looking at the overall pattern of outcomes, instead of trying to
  predict one exact result.
- **Cut** — after the first two rounds of a tournament, only some players
  continue to the final rounds; everyone else is "cut" and scores no
  points.
- **Field size** — how many players start a given tournament.
- **Points table** — how many points each finishing position earns; bigger
  tournaments (majors, signature events) award more points than regular
  events.
