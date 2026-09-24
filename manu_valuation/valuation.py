"""Glue: sporting paths -> cash flows -> value, plus the analyses built on top."""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from . import engine, finance, results, sporting
from .config import Config, apply


@dataclass
class Result:
    cfg: Config
    model: engine.SportingModel
    paths: engine.Paths
    cal: finance.Calibration
    proj: dict
    dcf: finance.DCF
    base_fy: int
    entry_fy: int  # last fiscal year reported before the deal
    entry_revenue: float  # its revenue
    irr: np.ndarray  # IRR per path if the club is sold at the entry multiple
    required_multiple: np.ndarray  # exit multiple per path that earns exactly the WACC

    @property
    def price(self) -> float:
        return self.cfg.deal.enterprise_value

    @property
    def entry_multiple(self) -> float:
        return self.price / self.entry_revenue

    @property
    def exit_multiple(self) -> float:
        m = self.cfg.valuation.exit_multiple
        return self.entry_multiple if m is None else m

    @property
    def percentile(self) -> float:
        """Share of simulated intrinsic EVs below the deal price."""
        return float((self.dcf.ev < self.price).mean())

    def ucl_seasons(self) -> np.ndarray:
        """Champions League seasons per path, counting only seasons not yet decided."""
        return self.paths.ucl[:, 1:].sum(axis=1)


def run(cfg: Config, matches: pd.DataFrame, fin: pd.DataFrame,
        model: engine.SportingModel | None = None, n_paths: int | None = None, seed: int | None = None) -> Result:
    model = model or engine.fit(matches, cfg.as_of)
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    paths = engine.simulate(model, cfg, rng, n_paths)
    fy = finance.base_fiscal_year(fin, cfg.as_of)
    cal = finance.calibrate(fin, model.positions, fy, cfg)
    proj = finance.project(paths, cal, cfg)
    dcf = finance.dcf(proj["fcff"], paths.first_fraction, cfg)
    entry_fy = finance.base_fiscal_year(fin, cfg.deal.date)
    entry_rev = float(fin.set_index("fy").loc[entry_fy, "revenue"])
    res = Result(cfg, model, paths, cal, proj, dcf, fy, entry_fy, entry_rev, np.array([]), np.array([]))
    res.irr = finance.irr_at_exit(res.price, proj["fcff"], proj["revenue"], res.exit_multiple, paths.first_fraction)
    res.required_multiple = finance.required_exit_multiple(res.price, proj["fcff"], proj["revenue"],
                                                           paths.first_fraction, dcf.wacc)
    return res


def summary(res: Result) -> dict:
    ev = res.dcf.ev
    f = res.paths.finish[:, 1:]
    return dict(
        ev_p5=np.percentile(ev, 5), ev_p50=np.median(ev), ev_p95=np.percentile(ev, 95), ev_mean=ev.mean(),
        deal_percentile=res.percentile,
        p_top4=float((f <= 4).mean()), p_ucl=float(res.paths.ucl[:, 1:].mean()),
        irr_p50=float(np.median(res.irr)), p_irr_above_wacc=float((res.irr >= res.dcf.wacc).mean()),
        req_multiple_p50=float(np.median(res.required_multiple)),
    )


def scenario_table(cfg: Config, matches, fin, model, scenarios, n_paths=None) -> pd.DataFrame:
    rows = []
    for sc in scenarios:
        r = run(apply(cfg, sc.changes), matches, fin, model, n_paths)
        rows.append(dict(scenario=sc.name, note=sc.note, **summary(r)))
    return pd.DataFrame(rows)


def sweep_level(cfg: Config, matches, fin, model, levels, n_paths=3000, changes: dict | None = None) -> pd.DataFrame:
    """Re-run with United's underlying level at the valuation date forced to each value
    (Elo points vs the league average). Maps a sporting ambition to value and to the IRR."""
    rows = []
    for mu in levels:
        c = apply(cfg, {**(changes or {}), "sporting.level_override": float(mu)})
        r = run(c, matches, fin, model, n_paths)
        rows.append(dict(level=mu, **summary(r)))
    return pd.DataFrame(rows)


def reality_check(res: Result, all_matches: pd.DataFrame) -> pd.DataFrame:
    """Where United's actual finishes since the valuation date fell in the simulated distribution."""
    actual = results.final_positions(all_matches).query("team == @results.MUFC").set_index("season")["pos"]
    rows = []
    for y, season in enumerate(res.paths.seasons):
        if season not in actual.index:
            continue
        f = res.paths.finish[:, y]
        a = int(actual.loc[season])
        rows.append(dict(season=results.season_label(season), actual=a,
                         p_exact=float((f == a).mean()), p_this_or_worse=float((f >= a).mean()),
                         p_top4=float((f <= 4).mean()), median=float(np.median(f))))
    return pd.DataFrame(rows)


def backtest(matches: pd.DataFrame, link, first: int = 2012, n_paths: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Out-of-sample test of the pre-season forecasts. For each season, the dynamics are fitted
    only on earlier seasons, then the whole season is simulated from the pre-season prior."""
    rng = np.random.default_rng(seed)
    strengths = sporting.all_season_strengths(matches, link)
    positions = results.final_positions(matches)
    rows = []
    for season in sorted(positions["season"].unique()):
        if season < first:
            continue
        past = strengths[strengths["season"] < season]
        dyn = sporting.fit_dynamics(past, positions[positions["season"] < season])
        g = matches[matches["season"] == season]
        teams = sorted(set(g["home"]) | set(g["away"]))
        model = engine.SportingModel(pd.Timestamp(f"{season}-07-01"), matches, link, None, None,
                                     past, positions, dyn)
        prior = engine._prior(model, teams, season)
        theta = prior["mean"].to_numpy() + prior["sd"].to_numpy() * rng.standard_normal((n_paths, len(teams)))
        pos = sporting.simulate_league(theta, link, rng)
        act = positions[positions["season"] == season].set_index("team").loc[teams, "pos"].to_numpy()
        for i, t in enumerate(teams):
            rows.append(dict(season=season, team=t, actual=act[i], exp_pos=pos[:, i].mean(),
                             p_top4=(pos[:, i] <= 4).mean(), p_releg=(pos[:, i] >= 18).mean()))
    return pd.DataFrame(rows)


def brier(bt: pd.DataFrame) -> pd.DataFrame:
    out = []
    for event, col, cond in (("top 4", "p_top4", bt["actual"] <= 4), ("relegated", "p_releg", bt["actual"] >= 18)):
        y = cond.astype(float)
        model = float(((bt[col] - y) ** 2).mean())
        clim = float(((y.mean() - y) ** 2).mean())
        out.append(dict(event=event, brier=model, climatology=clim, skill=1 - model / clim))
    return pd.DataFrame(out)


def with_seed(cfg: Config, seed: int) -> Config:
    return replace(cfg, seed=seed)
