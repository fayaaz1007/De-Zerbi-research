"""From sporting paths to revenue, cash flow and enterprise value.

Revenue lines follow the club's own split:

  broadcasting = Premier League central payments (equal share + merit by finish)
               + UEFA prize money (competition, stage, value pillar)
               + other (domestic cups, MUTV...): calibrated residual
  matchday     = core (league + domestic cups) + European home games x gate
  commercial   = core x (1 + UCL bonus clauses) - kit-deal penalty after two seasons out of the UCL

The "core" and "other" pieces are calibrated so that the model reproduces the base year's
reported numbers exactly, given what actually happened on the pitch that year.

Free cash flow to the firm:
  FCFF = EBITDA - tax - net player-registration capex - PP&E capex
with player capex standing in for amortisation in the tax computation (steady state).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import KNOWN_EUROPE, Config
from .engine import CHAMPIONSHIP, Paths


def base_fiscal_year(fin: pd.DataFrame, as_of) -> int:
    """Latest fiscal year whose annual results were public at `as_of` (released mid-September)."""
    as_of = pd.Timestamp(as_of)
    public = [fy for fy in fin["fy"] if pd.Timestamp(f"{fy}-09-20") <= as_of]
    if not public:
        raise ValueError(f"no annual results published before {as_of.date()}")
    return int(max(public))


# ---------------------------------------------------------------- tariffs


def pl_central(finish: np.ndarray, season, cfg: Config) -> np.ndarray:
    r = cfg.revenue
    growth = (1 + r.pl_growth) ** (np.asarray(season) - r.pl_season)
    in_pl = finish <= 20
    merit = r.pl_merit_per_place * (21 - np.where(in_pl, finish, 20))
    return np.where(in_pl, r.pl_equal + merit, r.parachute_share * r.pl_equal) * growth


def uefa_prize(comp: np.ndarray, stage: np.ndarray, years_from_base, cfg: Config) -> np.ndarray:
    r = cfg.revenue
    tables = np.array([np.zeros(7), r.ucl_prize_eur, r.uel_prize_eur, r.uecl_prize_eur])
    pillar = np.array([0.0, *r.value_pillar_eur])
    eur = tables[comp, np.clip(stage, 0, 6)] + pillar[comp]
    return np.where(comp > 0, eur, 0.0) * r.gbp_per_eur * (1 + r.uefa_growth) ** years_from_base


def euro_home_games(comp: np.ndarray, stage: np.ndarray) -> np.ndarray:
    """4 league-phase home games + one per knockout tie played (the final is neutral)."""
    return np.where(comp > 0, 4 + np.clip(stage, 0, 4), 0)


def euro_gate(comp, stage, years_from_base, cfg: Config) -> np.ndarray:
    r = cfg.revenue
    gate = np.array([0.0, *r.euro_gate])[comp]
    return euro_home_games(comp, stage) * gate * (1 + r.matchday_growth) ** years_from_base


def kit_penalty(ucl_now: np.ndarray, ucl_prev: np.ndarray, cfg: Config) -> np.ndarray:
    return np.where(~ucl_now & ~ucl_prev, cfg.revenue.kit_deal * cfg.revenue.kit_penalty, 0.0)


# ---------------------------------------------------------------- calibration


@dataclass
class Calibration:
    fy: int
    broadcast_other: float
    matchday_core: float
    commercial_core: float
    wage_core: float
    other_opex: float
    player_capex_ratio: float
    table: pd.DataFrame  # reported vs modelled pieces, for the report


def calibrate(fin: pd.DataFrame, positions: pd.DataFrame, fy: int, cfg: Config) -> Calibration:
    row = fin.set_index("fy").loc[fy]
    season = fy - 1  # FY2024 = season 2023-24 (start year 2023)
    pos = positions.query("season == @season and team == 'Manchester United'")["pos"]
    if pos.empty:
        raise ValueError(f"no final league position for season {season}")
    finish = np.array([int(pos.iloc[0])])
    eu = KNOWN_EUROPE.get(season, dict(comp=0, stage=None))
    comp = np.array([eu["comp"]])
    stage = np.array([eu["stage"] if eu["stage"] is not None else -1])
    prev_ucl = np.array([KNOWN_EUROPE.get(season - 1, {}).get("comp") == 1])
    ucl = comp == 1

    pl = float(pl_central(finish, season, cfg)[0])
    uefa = float(uefa_prize(comp, stage, 0, cfg)[0])
    gate = float(euro_gate(comp, stage, 0, cfg)[0])
    pen = float(kit_penalty(ucl, prev_ucl, cfg)[0])
    c = cfg.costs
    bonus = 1 + cfg.revenue.ucl_commercial_bonus * float(ucl[0])
    commercial_core = (row["commercial"] + pen) / bonus
    wage_core = (row["wages"] - c.europe_bonus_share * uefa) / (1 - c.ucl_pay_cut * c.clause_share * (not ucl[0]))
    other_opex = row["revenue"] - row["wages"] - row["adj_ebitda"]
    ratio = c.player_capex_ratio
    if ratio is None:
        if pd.isna(row.get("amortisation", np.nan)):
            raise ValueError(f"FY{fy} has no amortisation; set costs.player_capex_ratio")
        ratio = row["amortisation"] / row["revenue"] - c.player_sale_profit_ratio
    table = pd.DataFrame([
        ("Broadcasting", row["broadcasting"], f"PL central {pl:.1f} (finish {finish[0]}) + UEFA {uefa:.1f}",
         row["broadcasting"] - pl - uefa),
        ("Matchday", row["matchday"], f"European gates {gate:.1f}", row["matchday"] - gate),
        ("Commercial", row["commercial"], f"kit penalty {pen:.1f}, UCL bonus x{bonus:.2f}", commercial_core),
        ("Wages", row["wages"], f"UEFA bonuses {c.europe_bonus_share * uefa:.1f}, UCL clause {'off' if ucl[0] else 'on'}", wage_core),
        ("Other opex", other_opex, "revenue - wages - adjusted EBITDA", other_opex),
    ], columns=["line", "reported", "sporting-driven part", "calibrated core"])
    return Calibration(fy, row["broadcasting"] - pl - uefa, row["matchday"] - gate, commercial_core,
                       wage_core, other_opex, float(ratio), table)


# ---------------------------------------------------------------- projection


def project(paths: Paths, cal: Calibration, cfg: Config) -> dict:
    """Every line of the P&L and cash flow, each (paths, years), GBP m nominal."""
    r, c = cfg.revenue, cfg.costs
    fy = paths.seasons + 1
    k = (fy - cal.fy)[None, :]  # years of growth since the base year
    finish, comp, stage = paths.finish, paths.comp, paths.stage
    down = finish == CHAMPIONSHIP
    ucl = comp == 1
    ucl_prev = np.concatenate([np.full((len(ucl), 1), paths.prev_ucl), ucl[:, :-1]], axis=1)

    pl = pl_central(finish, paths.seasons[None, :], cfg)
    uefa = uefa_prize(comp, stage, k, cfg)
    broadcasting = pl + uefa + cal.broadcast_other * (1 + r.other_broadcast_growth) ** k
    matchday = (cal.matchday_core * (1 + r.matchday_growth) ** k * (1 - r.relegation_matchday * down)
                + euro_gate(comp, stage, k, cfg))
    commercial = (cal.commercial_core * (1 + r.commercial_growth) ** k * (1 - r.relegation_commercial * down)
                  * (1 + r.ucl_commercial_bonus * ucl) - kit_penalty(ucl, ucl_prev, cfg))
    revenue = broadcasting + matchday + commercial

    wages = (cal.wage_core * (1 + c.wage_growth) ** k * (1 - c.ucl_pay_cut * c.clause_share * ~ucl)
             * (1 - c.relegation_wage_cut * down) + c.europe_bonus_share * uefa)
    other_opex = cal.other_opex * (1 + c.other_opex_growth) ** k * (1 - c.opex_saving) * np.ones_like(revenue)
    ebitda = revenue - wages - other_opex
    player_capex = cal.player_capex_ratio * revenue
    ppe_capex = c.ppe_capex_ratio * revenue

    taxable = ebitda - player_capex - ppe_capex
    tax = np.zeros_like(taxable)
    losses = np.zeros(len(taxable))
    for y in range(taxable.shape[1]):  # losses carried forward
        profit = taxable[:, y] - losses
        tax[:, y] = c.tax_rate * np.clip(profit, 0, None)
        losses = np.clip(-profit, 0, None)
    fcff = ebitda - tax - player_capex - ppe_capex
    return dict(pl=pl, uefa=uefa, broadcasting=broadcasting, matchday=matchday, commercial=commercial,
                revenue=revenue, wages=wages, other_opex=other_opex, ebitda=ebitda,
                player_capex=player_capex, ppe_capex=ppe_capex, tax=tax, fcff=fcff)


# ---------------------------------------------------------------- DCF


@dataclass
class DCF:
    ev: np.ndarray
    pv_explicit: np.ndarray
    pv_terminal: np.ndarray
    wacc: float
    growth: float


def dcf(fcff: np.ndarray, first_fraction: float, cfg: Config, wacc: float | None = None,
        growth: float | None = None) -> DCF:
    v = cfg.valuation
    w = v.wacc if wacc is None else wacc
    g = v.terminal_growth if growth is None else growth
    Y = fcff.shape[1]
    cf = fcff.copy()
    cf[:, 0] *= first_fraction  # only the rest of the current year is still to come
    t = _times(first_fraction, Y)
    pv_explicit = (cf / (1 + w) ** t).sum(axis=1)
    n = min(v.terminal_window, Y - 1)
    # average the last n years in last-year money, so one lucky or unlucky season does not
    # set the value of the perpetuity
    lift = (1 + g) ** np.arange(n - 1, -1, -1)
    normal = (fcff[:, -n:] * lift).mean(axis=1)
    tv = normal * (1 + g) / (w - g)
    pv_terminal = tv / (1 + w) ** (first_fraction + Y - 1)
    return DCF(pv_explicit + pv_terminal, pv_explicit, pv_terminal, w, g)


# ---------------------------------------------------------------- the buyer's lens


def _times(first_fraction: float, years: int) -> np.ndarray:
    return np.concatenate([[first_fraction / 2], first_fraction + np.arange(1, years) - 0.5])


def required_exit_multiple(price: float, fcff: np.ndarray, revenue: np.ndarray, first_fraction: float,
                           wacc: float) -> np.ndarray:
    """Exit EV / revenue (at the end of the horizon) that makes paying `price` earn exactly `wacc`."""
    cf = fcff.copy()
    cf[:, 0] *= first_fraction
    t = _times(first_fraction, fcff.shape[1])
    pv = (cf / (1 + wacc) ** t).sum(axis=1)
    T = first_fraction + fcff.shape[1] - 1
    return (price - pv) * (1 + wacc) ** T / revenue[:, -1]


def irr_at_exit(price: float, fcff: np.ndarray, revenue: np.ndarray, multiple: float,
                first_fraction: float) -> np.ndarray:
    """IRR of paying `price` today, collecting FCFF and selling at `multiple` x final revenue.
    Vectorised bisection, one IRR per path."""
    cf = fcff.copy()
    cf[:, 0] *= first_fraction
    t = _times(first_fraction, fcff.shape[1])
    T = first_fraction + fcff.shape[1] - 1
    exit_ = multiple * revenue[:, -1]
    lo, hi = np.full(len(cf), -0.5), np.full(len(cf), 1.0)
    for _ in range(60):
        mid = (lo + hi) / 2
        npv = (cf / (1 + mid[:, None]) ** t).sum(axis=1) + exit_ / (1 + mid) ** T - price
        lo = np.where(npv > 0, mid, lo)
        hi = np.where(npv > 0, hi, mid)
    return (lo + hi) / 2
