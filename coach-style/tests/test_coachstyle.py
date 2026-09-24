import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coachstyle import analysis, synthetic  # noqa: E402
from coachstyle.features import match_rows, team_metrics  # noqa: E402
from coachstyle.spells import placebo_changes, real_changes, spells  # noqa: E402


# --------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------
def ev(team, kind, x, y=40, possession=1, poss_team=None, **extra):
    e = {"team": {"name": team}, "type": {"name": kind}, "location": [x, y], "period": 1,
         "possession": possession, "possession_team": {"name": poss_team or team},
         "play_pattern": {"name": "Regular Play"}}
    e.update(extra)
    return e


def pas(team, x, end_x, length=None, kind=None, cross=False, possession=1):
    p = {"length": length if length is not None else abs(end_x - x), "end_location": [end_x, 40]}
    if kind:
        p["type"] = {"name": kind}
    if cross:
        p["cross"] = True
    return ev(team, "Pass", x, possession=possession, **{"pass": p})


@pytest.fixture
def events():
    return [
        # A: 4 passes in 2 possessions (one a goal kick), B: 2 passes
        pas("A", 10, 50, kind="Goal Kick"),           # long goal kick, not open play
        pas("A", 30, 40, possession=1),
        pas("A", 85, 90, cross=True, possession=2),   # final third, into final third, cross
        pas("A", 60, 50, possession=2),               # backwards pass
        pas("B", 20, 30, possession=3),               # in B's own 60% (x < 72)
        pas("B", 80, 85, possession=3),               # in B's attacking zone
        # A defends: one tackle high up, one interception deep, one pressure high
        ev("A", "Duel", 70, duel={"type": {"name": "Tackle"}}, possession=3, poss_team="B"),
        ev("A", "Interception", 20, possession=3, poss_team="B"),
        ev("A", "Pressure", 90, possession=3, poss_team="B"),
        ev("A", "Ball Recovery", 100, possession=3, poss_team="B"),
        ev("A", "Shot", 110, shot={"statsbomb_xg": 0.3, "type": {"name": "Open Play"}},
           play_pattern={"name": "From Counter"}),
        ev("B", "Shot", 100, shot={"statsbomb_xg": 0.76, "type": {"name": "Penalty"}}),
    ]


def test_team_metrics(events):
    m = team_metrics(events, "A")
    assert m["possession"] == pytest.approx(100 * 4 / 6)
    assert m["field_tilt"] == pytest.approx(50)            # A: 1 pass from x>=80, B: 1
    assert m["passes_per_poss"] == pytest.approx(3 / 2)    # 3 open-play passes, 2 possessions
    assert m["directness"] == pytest.approx((10 + 5 - 10) / 25)
    assert m["gk_short_share"] == 0
    assert m["cross_share"] == 100
    # B passes in own 60%: 1; A defensive actions at x >= 48: the tackle only.
    assert m["ppda"] == pytest.approx(1.0)
    assert m["press_intensity"] == pytest.approx(1 / 2)
    assert m["press_high_share"] == 100
    assert m["def_height"] == pytest.approx((70 + 20 + 100) / 3)
    assert m["high_recoveries"] == 1
    assert m["counter_shots"] == 1
    assert m["xg_for"] == pytest.approx(0.3) and m["xg_against"] == pytest.approx(0.76)
    assert m["np_xg_against"] == 0


def test_match_rows_mirror(events):
    match = {"match_id": 1, "match_date": "2015-08-08", "season": {"season_name": "2015/2016"},
             "match_week": 1, "home_score": 2, "away_score": 2,
             "home_team": {"home_team_name": "A", "managers": [{"name": "Alpha Coach"}]},
             "away_team": {"away_team_name": "B", "managers": [{"name": "Beta", "nickname": "B"}]}}
    home, away = match_rows(match, events, "Test")
    assert home["possession"] + away["possession"] == pytest.approx(100)
    assert home["xg_for"] == away["xg_against"]
    assert (home["manager"], away["manager"]) == ("Alpha Coach", "B")
    assert home["points"] == away["points"] == 1


# --------------------------------------------------------------------------
# spells
# --------------------------------------------------------------------------
def test_caretaker_is_dropped_and_change_is_kept():
    mgrs = ["Mourinho"] * 16 + ["Holland"] + ["Hiddink"] * 21
    df = pd.DataFrame({"team": "Chelsea", "league": "PL", "manager": mgrs,
                       "date": pd.date_range("2015-08-08", periods=38, freq="7D")})
    sp = spells(df)
    assert sp["manager"].tolist() == ["Mourinho", "Hiddink"]
    (c,) = real_changes(sp)
    assert (c.old, c.new, c.w, c.caretaker_matches) == ("Mourinho", "Hiddink", 10, 1)
    assert df.loc[c.pre, "manager"].eq("Mourinho").all()
    assert df.loc[c.post, "manager"].eq("Hiddink").all()
    # 16 matches -> fake breaks at 10..6: 16 - 2*6 + 1 = 5 with w=6
    assert len([p for p in placebo_changes(sp, 6) if p.old == "Mourinho"]) == 5


def test_same_manager_either_side_of_caretaker_is_merged():
    mgrs = ["Simeone"] * 10 + ["Tiago"] * 2 + ["Simeone"] * 10
    df = pd.DataFrame({"team": "Atleti", "league": "Liga", "manager": mgrs,
                       "date": pd.date_range("2015-08-08", periods=22, freq="7D")})
    sp = spells(df)
    assert len(sp) == 1 and sp["n"].iloc[0] == 20
    assert real_changes(sp) == []


# --------------------------------------------------------------------------
# statistics on a season with known answers
# --------------------------------------------------------------------------
def run(season):
    df = analysis.standardise(analysis.prepare(season))
    sp = spells(df)
    return df, sp, real_changes(sp)


def test_adjust_removes_opponent_effects():
    raw = synthetic.season(jump=0, noise=0.5, opp_effect=1.0, seed=4)
    df = analysis.prepare(raw)
    # Opponent explains a lot of the raw metric, almost none of the adjusted one.
    raw_r2 = df.groupby("opponent")["possession"].mean().var() / df["possession"].var()
    adj_r2 = df.groupby("opponent")["possession_adj"].mean().var() / df["possession_adj"].var()
    assert raw_r2 > 0.3 and adj_r2 < 0.05


def test_style_shift_detects_real_change():
    df, sp, changes = run(synthetic.season(jump=1.0, seed=1))
    assert len(changes) == 6
    res = analysis.style_shift(df, sp, changes, n_perm=2000)
    assert res["summary"]["norm"]["ratio"] > 1.2
    assert res["summary"]["norm"]["p_value"] < 0.01
    # The planted jump is +1 on every metric.
    assert (res["by_metric"]["mean_signed_shift"] > 0).mean() > 0.8


def test_style_shift_null():
    pvals = []
    for seed in range(5):
        df, sp, changes = run(synthetic.season(jump=0.0, seed=seed))
        pvals.append(analysis.style_shift(df, sp, changes, n_perm=1000)["summary"]["norm"]["p_value"])
    assert min(pvals) > 0.01


def test_holm():
    adj = analysis.holm([0.01, 0.04, 0.03, 0.5])
    assert adj == pytest.approx([0.04, 0.09, 0.09, 0.5])


def test_bounce_is_zero_when_changes_do_nothing():
    df, sp, changes = run(synthetic.season(n_changes=10, jump=0.0, seed=2))
    b = analysis.bounce(df, sp, changes, n_boot=1000)["xgd_adj"]
    lo, hi = b["beyond_rtm_ci"]
    assert lo < 0 < hi


def test_fingerprints_and_pca():
    df, sp, _ = run(synthetic.season(seed=3))
    fp = analysis.fingerprints(df, sp)
    assert len(fp) == 26  # 20 teams + 6 new managers
    coords, loadings, evr = analysis.pca_2d(fp)
    assert coords.shape == (26, 2) and loadings.shape[0] == 2
    assert 0 < evr.sum() <= 1
    assert np.isfinite(coords).all()
