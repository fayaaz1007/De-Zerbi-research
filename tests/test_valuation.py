import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from manu_valuation import edgar, elo, engine, finance, results, sporting, valuation  # noqa: E402
from manu_valuation.config import Config, apply, scenarios  # noqa: E402

LINK = elo.OutcomeModel(b=0.55, c1=-0.9, c2=0.26)  # close to the fitted Premier League link


def round_robin(strength: dict, rng, season=2020, legs=1) -> pd.DataFrame:
    teams = list(strength)
    rows = []
    for leg in range(legs):
        for h in teams:
            for a in teams:
                if h == a:
                    continue
                ph, pd_, _ = LINK.probs(strength[h] - strength[a])
                u = rng.random()
                hg, ag = (1, 0) if u < ph else (0, 0) if u < ph + pd_ else (0, 1)
                rows.append(dict(season=season, date=pd.Timestamp(f"{season}-08-01") + pd.Timedelta(days=leg),
                                 home=h, away=a, hg=float(hg), ag=float(ag)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- results


def test_parse_season_handles_both_score_formats_and_unplayed():
    payload = {"matches": [
        {"date": "2024-08-16", "team1": "Manchester United FC", "team2": "Fulham FC", "score": {"ft": [1, 0]}},
        {"date": "2024-08-17", "team1": "AFC Bournemouth", "team2": "Arsenal FC", "score": [0, 0]},
        {"date": "2024-08-18", "team1": "Chelsea FC", "team2": "Everton FC"},
    ]}
    df = results.parse_season(payload, 2024)
    assert list(df["home"]) == ["Manchester United", "Bournemouth", "Chelsea"]
    assert df.loc[1, ["hg", "ag"]].tolist() == [0, 0]
    assert df.loc[2, ["hg", "ag"]].isna().all()


def test_table_and_as_of():
    df = pd.DataFrame(dict(season=2020, date=pd.to_datetime(["2020-08-01", "2020-08-02", "2020-08-03"]),
                           home=["A", "B", "C"], away=["B", "C", "A"], hg=[2.0, 1.0, 0.0], ag=[0.0, 1.0, 3.0]))
    t = results.table(df)
    assert t.iloc[0]["team"] == "A" and t.iloc[0]["pts"] == 6
    hidden = results.as_of(df, "2020-08-02")
    assert hidden["hg"].isna().sum() == 2


# ---------------------------------------------------------------- Elo and strengths


def test_outcome_probabilities_are_valid_and_monotone():
    gaps = np.linspace(-400, 400, 9)
    ph, pd_, pa = LINK.probs(gaps)
    assert np.allclose(ph + pd_ + pa, 1)
    assert (np.diff(ph) > 0).all() and (np.diff(pa) < 0).all() and (pd_ > 0).all()


def test_elo_ranks_the_strongest_team_first():
    rng = np.random.default_rng(0)
    strength = {f"T{i}": 40.0 * i for i in range(10)}
    df = pd.concat([round_robin(strength, rng, s, legs=2) for s in (2018, 2019, 2020)])
    r = elo.run(df, elo.EloParams())
    assert max(r.current, key=r.current.get) == "T9"
    link = elo.fit_outcomes(r.matches)
    assert link.b > 0


def test_season_strengths_recover_true_strengths():
    rng = np.random.default_rng(1)
    true = {f"T{i}": v for i, v in enumerate(np.linspace(-200, 200, 20))}
    df = round_robin(true, rng, legs=4)  # 4x a season, so the estimate is tight
    est = sporting.season_strengths(df, LINK).set_index("team")["theta"]
    assert abs(est.mean()) < 1e-6
    assert np.corrcoef(est[list(true)], list(true.values()))[0, 1] > 0.95
    assert est.loc["T19"] > est.loc["T0"]


def test_state_space_recovers_a_drifting_level():
    rng = np.random.default_rng(2)
    rows = []
    for c in range(40):
        level, form = rng.normal(0, 80), 0.0
        for s in range(25):
            level += rng.normal(0, 35)
            form = 0.3 * form + rng.normal(0, 20)
            rows.append(dict(season=2000 + s, team=f"C{c}", theta=level + form + rng.normal(0, 50), se=50.0))
    d = sporting.fit_dynamics(pd.DataFrame(rows))
    assert 25 < d.sigma_level < 45
    m, S = d.predict("C0", 2025)
    assert S[0, 0] > d.states["C0"][2][0, 0]  # uncertainty grows one season ahead


def test_europe_stronger_team_goes_further():
    rng = np.random.default_rng(3)
    cfg = Config().sporting
    comp = np.ones(20_000, dtype=int)
    weak = sporting.europe_stage(np.full(20_000, 0.0), comp, cfg, 200, rng).mean()
    strong = sporting.europe_stage(np.full(20_000, 300.0), comp, cfg, 200, rng).mean()
    assert strong > weak
    none = sporting.europe_stage(np.zeros(5), np.zeros(5, dtype=int), cfg, 200, rng)
    assert (none == -1).all()


# ---------------------------------------------------------------- finance


def _fin():
    return edgar.load_financials(ROOT / "data" / "manu_financials_snapshot.csv")


def _positions(season, pos):
    return pd.DataFrame(dict(season=[season], team=["Manchester United"], pos=[pos], pts=[0], gd=[0]))


def test_calibration_reproduces_the_base_year():
    cfg, fin = Config(), _fin()
    cal = finance.calibrate(fin, _positions(2022, 3), 2023, cfg)
    # FY2023 = 2022-23: 3rd in the league, Europa League quarter-final, UCL the season before
    paths = engine.Paths(np.array([2022]), np.array([[3]]), np.array([[2]]), np.array([[3]]), np.zeros((1, 1)),
                         np.zeros(1), True, 1.0, pd.DataFrame())
    p = finance.project(paths, cal, cfg)
    row = fin.set_index("fy").loc[2023]
    for line in ("broadcasting", "matchday", "commercial", "wages", "revenue"):
        assert p[line][0, 0] == pytest.approx(row[line], abs=1e-6)
    assert p["ebitda"][0, 0] == pytest.approx(row["adj_ebitda"], abs=1e-6)


def test_kit_penalty_needs_two_seasons_without_ucl():
    cfg = Config()
    now = np.array([False, False, True])
    prev = np.array([True, False, False])
    pen = finance.kit_penalty(now, prev, cfg)
    assert pen.tolist() == [0, cfg.revenue.kit_deal * cfg.revenue.kit_penalty, 0]


def test_champions_league_is_worth_more_than_no_europe():
    cfg = Config()
    ucl = finance.uefa_prize(np.array([1, 0]), np.array([2, -1]), 0, cfg)
    assert ucl[0] > 0 and ucl[1] == 0
    assert finance.pl_central(np.array([1]), 2023, cfg)[0] > finance.pl_central(np.array([20]), 2023, cfg)[0]


def test_dcf_matches_gordon_growth_for_flat_cash_flows():
    cfg = Config()
    g, w = 0.0, 0.08
    fcff = np.full((1, 10), 100.0)
    d = finance.dcf(fcff, 1.0, cfg, wacc=w, growth=g)
    t = np.arange(10) + 0.5
    expected = (100 / (1 + w) ** t).sum() + 100 / w / (1 + w) ** 10
    assert d.ev[0] == pytest.approx(expected)


def test_required_multiple_and_irr_agree():
    rng = np.random.default_rng(4)
    fcff = rng.normal(20, 10, (50, 10))
    revenue = rng.normal(800, 50, (50, 10))
    req = finance.required_exit_multiple(5000, fcff, revenue, 0.4, 0.086)
    irr = np.array([finance.irr_at_exit(5000, fcff[[i]], revenue[[i]], req[i], 0.4)[0] for i in range(5)])
    assert irr == pytest.approx(0.086, abs=1e-6)


def test_apply_changes_settings_without_touching_the_original():
    cfg = Config()
    changed = apply(cfg, {"costs.opex_saving": 0.2})
    assert changed.costs.opex_saving == 0.2 and cfg.costs.opex_saving == 0.0
    with pytest.raises(AttributeError):
        apply(cfg, {"costs.nonsense": 1})
    assert [s.name for s in scenarios()][0] == "Status quo"


# ---------------------------------------------------------------- EDGAR parsing (offline)

INSTANCE = b"""<?xml version="1.0"?>
<xbrli:xbrl xmlns:xbrli="http://www.xbrl.org/2003/instance" xmlns:ifrs-full="http://xbrl.ifrs.org/taxonomy/ifrs-full"
 xmlns:xbrldi="http://xbrl.org/2006/xbrldi" xmlns:manu="http://www.manutd.com" xmlns:dei="http://xbrl.sec.gov/dei">
 <xbrli:context id="FY"><xbrli:entity><xbrli:identifier scheme="x">1549107</xbrli:identifier></xbrli:entity>
  <xbrli:period><xbrli:startDate>2023-07-01</xbrli:startDate><xbrli:endDate>2024-06-30</xbrli:endDate></xbrli:period></xbrli:context>
 <xbrli:context id="FY_MD"><xbrli:entity><xbrli:identifier scheme="x">1549107</xbrli:identifier><xbrli:segment>
  <xbrldi:explicitMember dimension="ifrs-full:ProductsAndServicesAxis">manu:MatchdayMember</xbrldi:explicitMember></xbrli:segment></xbrli:entity>
  <xbrli:period><xbrli:startDate>2023-07-01</xbrli:startDate><xbrli:endDate>2024-06-30</xbrli:endDate></xbrli:period></xbrli:context>
 <xbrli:context id="FY_BC"><xbrli:entity><xbrli:identifier scheme="x">1549107</xbrli:identifier><xbrli:segment>
  <xbrldi:explicitMember dimension="ifrs-full:ProductsAndServicesAxis">manu:BroadcastingMember</xbrldi:explicitMember></xbrli:segment></xbrli:entity>
  <xbrli:period><xbrli:startDate>2023-07-01</xbrli:startDate><xbrli:endDate>2024-06-30</xbrli:endDate></xbrli:period></xbrli:context>
 <xbrli:context id="FY_SP"><xbrli:entity><xbrli:identifier scheme="x">1549107</xbrli:identifier><xbrli:segment>
  <xbrldi:explicitMember dimension="ifrs-full:ProductsAndServicesAxis">manu:SponsorshipMember</xbrldi:explicitMember></xbrli:segment></xbrli:entity>
  <xbrli:period><xbrli:startDate>2023-07-01</xbrli:startDate><xbrli:endDate>2024-06-30</xbrli:endDate></xbrli:period></xbrli:context>
 <xbrli:context id="FY_RT"><xbrli:entity><xbrli:identifier scheme="x">1549107</xbrli:identifier><xbrli:segment>
  <xbrldi:explicitMember dimension="ifrs-full:ProductsAndServicesAxis">manu:RetailMerchandisingApparelAndProductLicensingMember</xbrldi:explicitMember></xbrli:segment></xbrli:entity>
  <xbrli:period><xbrli:startDate>2023-07-01</xbrli:startDate><xbrli:endDate>2024-06-30</xbrli:endDate></xbrli:period></xbrli:context>
 <ifrs-full:Revenue contextRef="FY" unitRef="GBP" decimals="-5">661800000</ifrs-full:Revenue>
 <ifrs-full:Revenue contextRef="FY_MD" unitRef="GBP" decimals="-5">137100000</ifrs-full:Revenue>
 <ifrs-full:Revenue contextRef="FY_BC" unitRef="GBP" decimals="-5">221800000</ifrs-full:Revenue>
 <ifrs-full:Revenue contextRef="FY_SP" unitRef="GBP" decimals="-5">200000000</ifrs-full:Revenue>
 <ifrs-full:Revenue contextRef="FY_RT" unitRef="GBP" decimals="-5">102900000</ifrs-full:Revenue>
</xbrli:xbrl>"""


def test_parse_instance_and_segments():
    facts = edgar.parse_instance(INSTANCE)
    assert len(facts) == 5
    seg = edgar.revenue_segments(facts)
    assert seg.loc[2024, "matchday"] == pytest.approx(137.1)
    assert seg.loc[2024, "broadcasting"] == pytest.approx(221.8)
    assert seg.loc[2024, "commercial"] == pytest.approx(302.9)  # sponsorship + retail


def test_companyfacts_annual_values_prefer_latest_filing():
    def fact(val, start, end, filed, form="20-F"):
        return {"val": val, "start": start, "end": end, "form": form, "filed": filed, "accn": "x"}
    payload = {"facts": {"ifrs-full": {
        "Revenue": {"units": {"GBP": [fact(640e6, "2022-07-01", "2023-06-30", "2023-09-21"),
                                      fact(648.4e6, "2022-07-01", "2023-06-30", "2024-09-19"),
                                      fact(150e6, "2023-04-01", "2023-06-30", "2023-09-21", "6-K")]}},
        "CashAndCashEquivalents": {"units": {"GBP": [{"val": 75e6, "end": "2023-06-30", "form": "20-F",
                                                       "filed": "2023-09-21", "accn": "x"}]}},
    }}}
    vals = edgar.annual_values(edgar.facts_frame(json.loads(json.dumps(payload))))
    assert vals.loc[2023, "revenue"] == pytest.approx(648.4)  # the restated, later filing
    assert vals.loc[2023, "cash"] == pytest.approx(75)


def test_merge_with_snapshot_fills_gaps_and_labels_sources():
    snap = _fin()
    e = pd.DataFrame({"revenue": [661.8], "wages": [364.8]}, index=pd.Index([2024], name="fy"))
    m = edgar.merge_with_snapshot(e, snap).set_index("fy")
    assert m.loc[2024, "matchday"] == snap.set_index("fy").loc[2024, "matchday"]
    assert m.loc[2024, "source"] == "EDGAR XBRL + snapshot"
    assert m.loc[2019, "source"].startswith("approx.")


def test_edgar_requires_contact_user_agent():
    with pytest.raises(ValueError):
        edgar.Edgar("python-requests")


# ---------------------------------------------------------------- end to end on real results


@pytest.fixture(scope="module")
def deal_run():
    matches = results.load(ROOT / "data" / "epl_matches.csv")
    cfg = Config(n_paths=400, horizon=6)
    model = engine.fit(matches, cfg.as_of)
    return matches, cfg, model, valuation.run(cfg, matches, _fin(), model)


def test_no_look_ahead(deal_run):
    _, cfg, model, _ = deal_run
    played = model.matches.dropna(subset=["hg"])
    assert played["date"].max() < pd.Timestamp(cfg.as_of)


def test_known_facts_are_respected(deal_run):
    _, _, _, res = deal_run
    p = res.paths
    assert p.seasons[0] == 2023
    assert (p.comp[:, 0] == 1).all() and (p.stage[:, 0] == 0).all()  # UCL group exit, Dec 2023
    fin = p.finish[:, 1:]
    assert ((fin >= 1) & (fin <= 20) | (fin == engine.CHAMPIONSHIP)).all()
    assert 0 < p.first_fraction < 1


def test_valuation_outputs_are_sane(deal_run):
    matches, _, _, res = deal_run
    assert np.isfinite(res.dcf.ev).all()
    assert 0 <= res.percentile <= 1
    assert res.entry_multiple == pytest.approx(res.price / 648.4)
    rc = valuation.reality_check(res, matches)
    assert list(rc["actual"][:3]) == [8, 15, 3]


def test_a_stronger_united_is_worth_more(deal_run):
    matches, cfg, model, _ = deal_run
    sw = valuation.sweep_level(cfg, matches, _fin(), model, [0, 250], n_paths=400)
    assert sw["p_ucl"].iloc[1] > sw["p_ucl"].iloc[0]
    assert sw["ev_p50"].iloc[1] > sw["ev_p50"].iloc[0]
