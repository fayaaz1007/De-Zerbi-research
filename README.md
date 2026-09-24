# De Zerbi tactical-performance model

> Also in this repo: [**coach-style/**](coach-style/README.md), which asks whether a new
> manager changes how a team plays, using 33 mid-season manager changes from 2015/16 and
> StatsBomb event data.

Tests three hypotheses about Roberto De Zerbi's teams (Sassuolo, Brighton, Marseille):

| | Hypothesis | Variable in the model | Supported if |
|---|---|---|---|
| H1 | They perform better with **less than 55% possession** | `low_poss` = 1 if possession < 55% | coefficient > 0 |
| H2 | They perform better when the **opponent presses harder** | `opp_press_z` = opponent PPDA, sign-flipped and standardised | coefficient > 0 |
| H3 | They perform better when the **opponent plays a higher defensive line** | `opp_line_z` = opponent line height, standardised | coefficient > 0 |

"Perform better" mainly means **xG difference (xGD)** per match. Goal difference and
win/draw/loss results are used as robustness checks.

## Read this first: four traps in the hypothesis

1. **PPDA runs backwards.** PPDA = opponent passes allowed per defensive action.
   A **low** PPDA (e.g. 7) means an **intense** press; a high PPDA (e.g. 18) means a passive
   block. So "the opponent presses harder" means a **lower** PPDA, not a higher one.
   The code flips the sign for you (`opp_press_z` goes up as pressing gets harder).
2. **Possession is partly an outcome, not a cause.** When a team goes 2–0 up, it
   usually gives the ball away. So low possession can look "good" only because the team
   was already winning (game state). If you can get **possession at 0–0** or
   **first-half possession**, put that in the `possession` column instead. It is a much
   cleaner test of H1.
3. **Opponent quality confounds everything.** Strong opponents take more of the ball
   *and* are harder to beat. That makes low possession look *worse* than it really is.
   The model controls for `opp_strength` (the opponent's xGD per match over the rest of
   the season) and for team-season fixed effects.
4. **Use the opponent's usual style, not the in-match number.** An opponent's PPDA in
   the match against De Zerbi is shaped by De Zerbi's team: if Brighton keep the ball
   well, the opponent's PPDA goes up. `opp_ppda_pre` (the opponent's PPDA over their
   *other* matches that season) describes how they *intend* to play. The pipeline uses it
   when it is available and warns you when it is not.

**Sample size:** one season is 38 matches. With about 150 league matches, a 0.25 xGD
effect has a standard error of about 0.16, so even a real effect can come out
non-significant or with the wrong sign. The synthetic demo shows this: its default seed
puts H1 on the wrong side of zero even though the true effect is +0.25. Use every
De Zerbi season you can get (the fetcher includes Sassuolo 2018–21 for this reason) and
read the confidence intervals, not just the p-values.

## What the model does

`run_analysis.py` writes `report.md` plus charts:

1. **Verdicts:** OLS `xgd ~ low_poss + opp_press_z + opp_line_z + home + opp_strength_z + C(team_season)`
   with HC3 robust standard errors. It reports one-sided p-values in the direction each
   hypothesis predicts.
2. **Robustness:** the same model on goal difference; an ordered logit on L/D/W;
   possession as a continuous variable; and a `low_poss × press` interaction, which asks
   whether giving up the ball pays off *specifically* against pressing teams (the
   "bait the press" idea).
3. **Simple splits:** mean xGD above and below each cut-off, with bootstrap CIs.
4. **Is 55% the right cut-off?** It scans cut-offs from 45% to 65%. The p-value comes from
   a permutation test on the *best* cut-off, so searching many cut-offs does not
   produce a false positive.
5. **Out-of-sample check:** leave one season out. It tests whether the tactical variables
   predict matches the model has never seen better than home advantage and opponent
   strength alone.

## Workflow

```bash
pip install -r requirements.txt

# 1. xG, goals, PPDA and opponent strength from Understat (run on your machine)
python scripts/fetch_understat.py --out data/raw/dezerbi_matches.csv

# 2. add possession (e.g. FBref "Scores & Fixtures" table saved as CSV)
python scripts/merge_columns.py data/raw/dezerbi_matches.csv fbref_logs.csv \
    --on date --rename Date=date Poss=possession

# 3. (optional, needed for H3) add the opponent defensive-line measure, per opponent-season
python scripts/merge_columns.py data/raw/dezerbi_matches.csv opp_lines.csv \
    --on season opponent --rename Squad=opponent LineHeight=opp_line_height_pre

# 4. run the model
python run_analysis.py data/raw/dezerbi_matches.csv --out outputs/real
```

To try the pipeline without real data:

```bash
python scripts/make_synthetic_data.py
python run_analysis.py data/synthetic_matches.csv --out outputs/synthetic
pytest
```

### Sources for the defensive line (H3)

Understat and FBref don't publish line height directly. Options, best first:

- **Wyscout / Opta / SkillCorner:** average defensive line height or "PPDA height", if
  you have access.
- **WhoScored / The Analyst:** team average defensive-action height.
- **Free proxy (FBref squad "Defensive Actions"):** the share of an opponent's tackles
  made in the middle and attacking thirds, `(Tkl Mid 3rd + Tkl Att 3rd) / Tkl`. Teams
  with a high line win the ball higher up the pitch. Compute it per opponent per season.

Understat, FBref and others spell team names differently. `merge_columns.py` prints how
many rows did not match, so fix any names until that count is 0.

## Data schema (`one row = one De Zerbi match`)

| column | required | meaning |
|---|---|---|
| `date`, `season`, `opponent`, `venue` (H/A) | yes | match id |
| `goals_for`, `goals_against`, `xg_for`, `xg_against` | yes | outcomes |
| `possession` | yes | De Zerbi team's possession %, from 0–100 or as a 0–1 fraction |
| `opp_ppda_pre` | one of these two | opponent's PPDA in their *other* matches (preferred) |
| `opp_ppda` | one of these two | opponent's PPDA in this match |
| `opp_line_height_pre` / `opp_line_height` | for H3 | opponent defensive-line height (any consistent unit) |
| `opp_strength` | recommended | opponent quality, e.g. xGD per match or Elo |
| `team`, `competition` | optional | used for team-season fixed effects |

## Layout

```
dezerbi/data.py              validation + feature engineering
dezerbi/models.py            all statistical tests
run_analysis.py              runs everything -> outputs/<name>/report.md + PNGs
scripts/fetch_understat.py   builds the dataset from Understat
scripts/merge_columns.py     adds possession / line height from other CSVs
scripts/make_synthetic_data.py  fake data with known effects, for testing
tests/                       pytest suite
```
