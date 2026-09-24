import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import fetch_understat  # noqa: E402
from make_synthetic_data import TRUE_EFFECTS, make  # noqa: E402
from merge_columns import merge  # noqa: E402

from dezerbi import models  # noqa: E402
from dezerbi.data import prepare  # noqa: E402


@pytest.fixture(scope="module")
def big():
    # 10x the matches of a real sample so the planted effects are estimated tightly.
    return prepare(make(n_per_season=380, seed=1))


def test_prepare_features():
    df = prepare(make(seed=0))
    assert set(df["result"]) <= {-1, 0, 1}
    assert (df["low_poss"] == (df["possession"] < 55)).all()
    # Lower PPDA = harder press = higher press score.
    assert df["opp_press_z"].corr(df["opp_ppda_pre"]) == pytest.approx(-1)
    assert df.attrs["press_col"] == "opp_ppda_pre"


def test_prepare_accepts_fraction_possession_and_warns_on_in_match_ppda():
    raw = make(seed=0).drop(columns=["opp_ppda_pre"])
    raw["possession"] /= 100
    with pytest.warns(UserWarning, match="in-match"):
        df = prepare(raw)
    assert df["possession"].max() > 1


def test_prepare_rejects_missing_columns():
    with pytest.raises(ValueError, match="possession"):
        prepare(make(seed=0).drop(columns=["possession"]))


def test_main_model_recovers_planted_effects(big):
    table = models.hypothesis_table(models.main_model(big)).set_index("term")
    for term in ["low_poss", "opp_press_z"]:
        assert table.loc[term, "ci_low"] <= TRUE_EFFECTS[term] <= table.loc[term, "ci_high"]
        assert table.loc[term, "verdict"] == "supported"


def test_null_effects_are_not_supported():
    df = prepare(make(seed=3))
    rng = np.random.default_rng(0)
    df["xgd"] = rng.normal(0, 1, len(df))  # outcome unrelated to everything
    table = models.hypothesis_table(models.main_model(df))
    assert (table["p_one_sided"] > 0.01).all()


def test_robustness_models_run(big):
    assert "low_poss:opp_press_z" in models.interaction_model(big).params
    assert models.continuous_possession_model(big).params["poss10"] < 0
    ol = models.hypothesis_table(models.ordered_logit(big)).set_index("term")
    assert ol.loc["low_poss", "coef"] > 0
    assert len(models.group_comparison(big, n_boot=200)) == 3


def test_threshold_scan_and_cv():
    df = prepare(make(seed=0))
    scan, p = models.threshold_scan(df, thresholds=[50, 55, 60], n_perm=9)
    assert list(scan["threshold"]) == [50, 55, 60]
    assert 0 < p <= 1
    cv = models.leave_one_season_out(df)
    assert len(cv.folds) == 4


def _hist(date, h_a, xg, xga, s, m, att, dfn):
    return {"date": date, "h_a": h_a, "xG": xg, "xGA": xga, "scored": s, "missed": m,
            "ppda": {"att": att, "def": dfn}}


def test_fetch_understat_pairs_opponents(monkeypatch):
    teams = {
        "1": {"title": "Brighton", "history": [
            _hist("2023-08-12 15:00:00", "h", "2.0", "0.5", "2", "0", 200, 20),
            _hist("2023-08-19 15:00:00", "a", "1.0", "1.0", "1", "1", 150, 15)]},
        "2": {"title": "Luton", "history": [
            _hist("2023-08-12 15:00:00", "a", "0.5", "2.0", "0", "2", 300, 10),
            _hist("2023-08-26 15:00:00", "h", "1.5", "0.5", "1", "0", 100, 20)]},
        "3": {"title": "Wolves", "history": [
            _hist("2023-08-19 15:00:00", "h", "1.0", "1.0", "1", "1", 120, 12),
            _hist("2023-08-12 15:00:00", "a", "1.0", "1.0", "0", "0", 80, 10)]},
    }
    monkeypatch.setattr(fetch_understat, "fetch_teams_data", lambda *a: teams)
    df = fetch_understat.build_spell("EPL", 2023, "Brighton", "2023-07-01", "2024-06-30")
    assert list(df["opponent"]) == ["Luton", "Wolves"]
    luton = df.iloc[0]
    assert luton["opp_ppda"] == pytest.approx(30)           # 300 / 10 in this match
    assert luton["opp_ppda_pre"] == pytest.approx(5)        # 100 / 20 excluding it
    assert luton["opp_strength"] == pytest.approx(1.0)      # xGD of their other match
    assert luton["venue"] == "H" and luton["season"] == "2023-24"


def test_merge_columns():
    matches = pd.DataFrame({"date": ["2023-08-12"], "season": ["2023-24"], "opponent": ["Luton"]})
    extra = pd.DataFrame({"Date": ["2023-08-12"], "Poss": [48.0]})
    out = merge(matches, extra, ["date"], {"Date": "date", "Poss": "possession"})
    assert out.loc[0, "possession"] == 48.0


def test_end_to_end_report(tmp_path):
    csv = tmp_path / "m.csv"
    make(seed=0).to_csv(csv, index=False)
    import run_analysis
    sys.argv = ["run_analysis", str(csv), "--out", str(tmp_path / "out"), "--n-perm", "5"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_analysis.main()
    report = (tmp_path / "out" / "report.md").read_text()
    assert "H1: <55% possession" in report
    assert (tmp_path / "out" / "coefficients.png").exists()
