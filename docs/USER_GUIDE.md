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

The things you can change are **how the model behaves** (settings), **what
tournaments happen, in what order** (the schedule), and — optionally —
**who's in the field** (either real historical players, or a field you make
up yourself). You never need to open or edit any `.py` file.

There are four separate analyses you can run: the full-season simulation
(`golf-sim season`, the default — described first below), "Chasing Mondays"
(`golf-sim monday-chase`), "Card Retention" (`golf-sim card-retention`), and
"Q-School" (`golf-sim qschool`) — jump to whichever section answers your
question.

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

## The files you're meant to edit

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

### 3. (Optional) Using your own field instead of history

By default the simulator looks at real historical PGA Tour scores to figure
out how good each player is. But sometimes you want to make up a field
yourself — for example, to ask "what if a field of 150 players all had
roughly the same skill level?" instead of using real players.

You do this with a second kind of CSV file, separate from the historical
data, with **one row per player** and exactly these five columns:

| player_id | mean | variance | skew | weight |
|---|---|---|---|---|
| 1 | 69.2 | 3.2 | 0.15 | 1.0 |
| 2 | 70.4 | 4.5 | 0.20 | 1.0 |
| ... | ... | ... | ... | ... |

- **`player_id`** — anything that identifies the player: a name, or just a
  number (as in the example above).
- **`mean`** — the score you expect this player to average per round (for
  golf, lower is better — e.g. `69.2` means they average just under par).
- **`variance`** — how spread out (inconsistent) their scores are. A bigger
  number means a less predictable player; `0` or a negative number isn't
  allowed.
- **`skew`** — whether they lean toward more really-bad rounds (positive
  skew) or more really-good rounds (negative skew). `0` means symmetric.
  If you're not sure, `0` is a reasonable default.
- **`weight`** — how often this player shows up in a simulated field,
  relative to the others. Equal numbers (e.g. `1.0` for everyone) means
  everyone is equally likely to be selected. This is the "participation"
  knob — bump one player's `weight` up if you want them to show up more
  often than the rest.

An example file with 15 players is included at
`data/custom_fields/example_field.csv` — open it in Excel to see the format
firsthand (it's too small to run a full season on its own, though — see the
error message note below).

To actually use your file, point `config/settings.yaml` at it:

```yaml
data:
  field_file: data/custom_fields/example_field.csv
```

When `field_file` is set, the simulator uses **only** that file and ignores
the historical `data/seasons/` CSVs entirely. To go back to using real
historical data, either delete that line or set it back to `null`.

**Important:** your custom field needs at least as many players as the
biggest tournament in your schedule (156, for a `REGULAR` event). If it
doesn't, you'll get a clear error telling you how many more players you
need, rather than the simulation just failing partway through.

## Running the simulation

Each time you want to run the full-season simulation:

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
   (this is short for `golf-sim season`; there's also `golf-sim monday-chase`
   for the separate analysis described below in "Chasing Mondays")

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

## Chasing Mondays

This is a separate, second analysis, answering a different question: for a
player without full Tour status, is it worth trying to "chase Mondays"?

**What "chasing Mondays" means:** a player without status can enter a
1-round Monday qualifier (usually 100-150 players) where roughly the top
3-5% earn their way into that week's tournament. If they then finish well
enough in that tournament (inside a rank you choose, e.g. top 25), they earn
a direct spot in *next* week's field too — skipping Monday qualifying that
week. If not, they're back to grinding Mondays. This tool simulates that
whole cycle over several weeks and tells you how often a player of a given
skill level ever makes it through, and ever "parlays" a good finish into an
exemption.

### Setting it up

This analysis needs **two separate populations**, each set up the same way
as the custom field file described above (see "Using your own field instead
of history"):

1. **`aspirant_pool`** — the players attempting to chase Mondays. There's no
   separate "how many players show up" setting: every non-exempt player in
   this pool plays Monday each week, so to realistically model a 100-150
   player Monday qualifier, this pool needs that many rows. This is where
   you'd put one or more hypothetical players at different skill levels (say,
   a `mean` of 70 vs. 74) to compare how much skill matters.
2. **`main_event_pool`** — the (presumably stronger) field they're trying to
   crash. This can be left pointing at the real historical data, or be its
   own custom field, but it needs at least 156 players and **cannot share
   any player ids with `aspirant_pool`**.

Both are configured in `config/monday_chase.yaml`, which has the same
heavily-commented format as `settings.yaml`. Example:

```yaml
aspirant_pool:
  field_file: data/custom_fields/my_aspirants.csv

main_event_pool:
  field_file: null   # null = use real historical Tour data

chase:
  advance_pct: 0.04    # ~4% of the Monday field qualifies
  parlay_top_n: 25     # top-25 finish earns next week's exemption
  n_weeks: 8            # simulate an 8-week chase
  n_simulations: 200
```

### Running it

```
cd ~/VSCodeProjects/Golf_Simulator
source .venv/bin/activate
golf-sim monday-chase
```

Results land in `outputs/monday_chase_results.csv`, one row per aspirant:

- **`Any_Monday_Advance_pct`** — chance of qualifying through at least one
  Monday over the whole chase.
- **`Any_Parlay_pct`** — chance of ever earning a next-week exemption. This
  is the headline "is this worth it" number.
- **`Avg_Weeks_Played`** — average number of weeks (out of `n_weeks`) they
  actually got to play the real tournament.
- **`Avg_Monday_Attempts`** — average number of Mondays they had to attempt
  (players who parlay attempt fewer Mondays, since exempt weeks skip it).
- **`Advance_Rate_Per_Monday_Attempt`** — chance of advancing on any single
  Monday attempt. Use this one (not `Any_Monday_Advance_pct`) to fairly
  compare skill levels, since a stronger player attempts fewer Mondays
  overall.

## Card Retention

This is a third, separate analysis: under a possible future PGA Tour setup
with a fixed pool of ~130 card-holding players competing in a season of
mostly-120-player events plus the 4 majors, what's the probability a given
player finishes ranked well enough to keep their card for next season?

**Why majors need a second pool:** majors have 156-player fields in real
life, and a 130-player card pool can't fill that alone, so majors also draw
some players from a second "outside qualifiers" pool (representing
world-ranking or open-qualifier entrants who don't hold a card). Every
card-pool player is treated as eligible for every major; the outside pool
only ever fills whatever spots are left over. Non-major events, by
contrast, are drawn entirely from the card pool.

### Setting it up

Like "Chasing Mondays," this needs two populations set up the same way as
the custom field file described earlier:

1. **`card_pool`** — the ~130 players whose card status you're evaluating.
   This is where you'd put players at different skill levels to compare.
2. **`outside_pool`** — the players who fill out major fields beyond the
   card pool. Needs no minimum size on its own, but `card_pool` +
   `outside_pool` together must reach 156 (a major's field size), and the
   two pools **cannot share any player ids**.

Both are configured in `config/card_retention.yaml`. There's also a
`schedule.path` setting pointing at a season-schedule CSV — use
`config/alignment_schedule.csv` (or your own copy), which uses
`ALIGNMENT_REGULAR` for non-major events (120-player field) instead of the
regular season's `REGULAR` (156-player field, generally too big for a
130-player card pool to fill alone).

```yaml
card_pool:
  field_file: data/custom_fields/my_card_pool.csv

outside_pool:
  field_file: data/custom_fields/my_outside_pool.csv

schedule:
  path: config/alignment_schedule.csv

retention:
  cutoff: 90    # finish this rank or better (within the card pool) to keep your card
  n_simulations: 200
```

### Running it

```
cd ~/VSCodeProjects/Golf_Simulator
source .venv/bin/activate
golf-sim card-retention
```

Results land in `outputs/card_retention_results.csv`, one row per card-pool
player:

- **`Retained_Card_pct`** — chance of finishing the season ranked at or
  better than `retention.cutoff`, i.e. chance of keeping the card. This is
  the headline number.
- **`Avg_SeasonRank`** — average season-ending rank (within the card pool)
  across all simulations.

Try `retention.cutoff: 90` (as originally framed) versus something like
`104` (closer to a ~20% turnover target for a 130-player pool) to see how
sensitive the answer is to exactly where the cutoff line falls.

## Q-School

A fourth analysis: how likely is a player to earn playing status through
Q-School — the multi-stage qualifying gauntlet — and how much does their
entry point matter?

**How the gauntlet works.** There are four stages. Stages 1, 2, and 3 are
each a four-round tournament where only the **top 20%** advance to the next
stage. The final (stage 4) hands out status by finish: the top 5 earn a PGA
Tour card, the rest of the top 25% a full Korn Ferry card, the next 25%
conditional status, and everyone else nothing. A strong résumé (say, a
standout college career) can let a player **skip straight into stage 2 or
3** — so the tool always reports what happens starting from stage 1, 2, and
3, side by side.

### Setting it up

This needs two populations, set up the same way as the custom field file
described earlier:

1. **`aspirant_pool`** — the players whose odds you want. **Keep this
   small** — a handful of skill levels (e.g. a top prospect at 68.5, a
   journeyman at 71) rather than a big pool. Each is run on its own so you
   can read the odds straight off.
2. **`competition_pool`** — the field they play against. This sets the
   **stage-1** difficulty; later stages are made tougher automatically (see
   `strength_step`). Needs no minimum skill, but must have at least
   `stage_field_size − 1` players and share no ids with the aspirants.

The key knob is **`strength_step`** — how many strokes better each stage's
field is than the one before. `0.5` means stage 2 averages half a stroke
better than stage 1, stage 3 a full stroke, the final 1.5. Turn it up to
make climbing from an early stage harder.

```yaml
aspirant_pool:
  field_file: data/custom_fields/my_prospects.csv

competition_pool:
  field_file: null   # null = historical Tour data as the stage-1 field

qschool:
  advance_pct: 0.20      # top 20% advance each stage
  strength_step: 0.5     # each stage's field is 0.5 strokes stronger
  n_simulations: 500
```

### Running it

```
cd ~/VSCodeProjects/Golf_Simulator
source .venv/bin/activate
golf-sim qschool
```

Results land in `outputs/qschool_results.csv`, one row per aspirant **and
start stage**:

- **`PGA_Card_pct`** — chance of finishing top-5 in the final (a PGA card).
- **`Full_KF_pct`** — chance of landing in the full-Korn-Ferry-card tier.
- **`Conditional_pct`** — chance of landing in the conditional-status tier.
- **`Any_Status_pct`** — chance of earning *any* status (the sum of the
  three above). This is the headline number.

Compare the three `Start_Stage` rows for a player to see how much a better
entry point is worth. The effect is usually largest for *marginal* players:
each 20% gate is a brutal filter, so skipping one or two of them matters far
more to a bubble player than to an elite who'd clear them anyway.

## Troubleshooting

| What you see | What it means | What to do |
|---|---|---|
| `command not found: golf-sim` | Your project environment isn't turned on | Run `source .venv/bin/activate`, then try again |
| `Error: config/settings.yaml: ... must be between 0 and 1 (got 1.5)` | A setting is outside its allowed range | Open `settings.yaml`, find the named setting, fix the value (the message tells you exactly which one and its valid range) |
| `Error: config/settings.yaml: could not parse YAML` | Indentation or a missing colon broke the file | Check that you used spaces (not Tab) and that every setting has a space after its colon |
| `Error: ... row with event_number=7 has unknown tournament_type 'REGULER'` | A typo in the schedule CSV | Fix the spelling to match one of the valid values listed above |
| `Error: no CSV files found in data/seasons` | You're running the command from the wrong folder | Run `cd ~/VSCodeProjects/Golf_Simulator` first |
| `Error: Not enough players to simulate this schedule: it has 15 player(s), but the largest scheduled event needs a field of 156.` | Your custom `field_file` doesn't have enough players for the schedule | Add more rows to your field file, or use a schedule with smaller events |
| `Error: ... player id '7' has invalid variance=0.0 (must be a positive, finite number)` | A row in your `field_file` has a bad `variance` or `weight` value | Open the file, find that `player_id`, fix the value (must be greater than 0) |
| `Error: aspirant_pool and main_event_pool share player id(s): ...` | Both Monday-chase pools point at the same data (e.g. both left as historical data) | Point at least one of `aspirant_pool`/`main_event_pool` at a distinct custom `field_file` |
| `Error: Week 3: ... aspirants ... exceed the main event's field size of 156` | Your `chase` settings let too many aspirants into one week's field | Lower `advance_pct`, tighten `parlay_top_n`, or shrink `aspirant_pool` |
| `Error: card_pool and outside_pool share player id(s): ...` | Both card-retention pools point at the same data, or use overlapping ids | Point `card_pool`/`outside_pool` at distinct custom `field_file`s |
| `Error: Not enough players to fill the largest major's field: card_pool + outside_pool together have 135 player(s), but the largest major needs a field of 156.` | Your two card-retention pools combined are too small for a major | Add more players to `card_pool` or `outside_pool` |
| `Error: aspirant_pool and competition_pool share player id(s): ...` | Both Q-School pools point at the same data | Point at least one at a distinct custom `field_file` |
| `Error: Not enough competition players to fill a stage field: ...` | Your Q-School `competition_pool` is smaller than a stage field | Add more players, or lower `qschool.stage_field_size` |

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
