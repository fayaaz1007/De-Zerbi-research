"""Every assumption the engine uses, in one place, with where it comes from.

Money is in GBP millions unless a name says otherwise. "FY2024" means the year to
30 June 2024, which is the 2023-24 season (the club's fiscal year matches the season).
Seasons are identified by their start year: season 2023 = 2023-24.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# What actually happened in Europe: known facts, used to calibrate the base year and to fix
# anything already settled at the valuation date. comp: 1 UCL, 2 UEL, 3 UECL, 0 none.
# `qualified` = date the place was settled; stage = exit stage index in sporting.STAGES
# (0 league phase/group ... 5 lost final, 6 won); `decided` = date the stage was settled.
KNOWN_EUROPE = {
    2019: dict(comp=2, qualified="2019-05-12", stage=4, decided="2020-08-16"),  # UEL SF v Sevilla
    2020: dict(comp=1, qualified="2020-07-26", stage=0, decided="2020-12-08"),  # UCL group exit (then UEL final)
    2021: dict(comp=1, qualified="2021-05-23", stage=2, decided="2022-03-15"),  # UCL R16 v Atletico
    2022: dict(comp=2, qualified="2022-05-22", stage=3, decided="2023-04-20"),  # UEL QF v Sevilla
    2023: dict(comp=1, qualified="2023-05-28", stage=0, decided="2023-12-12"),  # UCL group, 4th
    2024: dict(comp=2, qualified="2024-05-25", stage=5, decided="2025-05-21"),  # FA Cup winners; UEL final lost
    2025: dict(comp=0, qualified="2025-05-25", stage=None, decided=None),  # 15th in 2024-25, no Europe
    2026: dict(comp=1, qualified="2026-05-24", stage=None, decided=None),  # 3rd in 2025-26 -> UCL
}
# FA Cup (winner -> Europa League): `decided` = date United won it or went out
KNOWN_FA_CUP = {2023: dict(won=True, decided="2024-05-25"),  # beat Manchester City in the final
                2024: dict(won=False, decided="2025-03-02")}  # out to Fulham, fifth round


@dataclass
class Deal:
    """INEOS / Sir Jim Ratcliffe minority investment (announced 24 Dec 2023, completed
    20 Feb 2024): 25% of Class B and up to 25% of Class A shares at $33.00 per share."""
    date: str = "2024-02-20"
    price_usd: float = 33.0
    shares_m: float = 163.5  # ordinary shares before the deal (Class A + B), approx.; EDGAR dei data overrides
    usd_per_gbp: float = 1.27  # GBP/USD around Dec 2023 - Feb 2024
    net_debt: float = 575.0  # borrowings less cash, approx. at 31 Dec 2023 (estimate, check 6-K)
    transfer_payables: float = 300.0  # net transfer-fee payables, treated as debt-like (estimate)

    @property
    def equity(self) -> float:
        return self.price_usd * self.shares_m / self.usd_per_gbp

    @property
    def enterprise_value(self) -> float:
        return self.equity + self.net_debt + self.transfer_payables


@dataclass
class Sporting:
    # UEFA places. England earned a 5th Champions League place for 2024-25 and 2025-26
    # through the European Performance Spot; treat it as a coin flip each year.
    p_fifth_ucl: float = 0.5
    uel_places: tuple = (5, 6)  # finish -> Europa League (5th only if no 5th UCL place)
    uecl_places: tuple = (7,)
    # P(win the FA Cup) by league finish band (<=4, 5-8, 9+): the winner goes to the Europa League
    fa_cup_win: tuple = (0.10, 0.08, 0.02)
    p_return_from_championship: float = 0.6  # per season, for a club with United's budget
    # Probability of clearing each European hurdle for an average top-4 Premier League side:
    # (top 24 of league phase, top 8 | top 24, win play-off, win R16, win QF, win SF, win final).
    # Rough fit to English clubs' record 2015-2025. Log-odds move by europe_slope per +100 Elo.
    ucl_advance: tuple = (0.88, 0.45, 0.60, 0.50, 0.45, 0.45, 0.45)
    uel_advance: tuple = (0.95, 0.60, 0.70, 0.62, 0.58, 0.55, 0.55)
    uecl_advance: tuple = (0.97, 0.65, 0.75, 0.70, 0.65, 0.60, 0.60)
    europe_slope: float = 0.8
    # Force United's underlying level at the valuation date (Elo points vs league average)
    # instead of the Kalman-filtered estimate. Used for the "what did the buyer bet on" sweep.
    level_override: float | None = None


@dataclass
class Revenue:
    # Premier League central payments to United (2023-24 distribution): equal shares of UK
    # and international TV plus central commercial (~£86m), facility fees for live UK games
    # (United are always near the cap, ~£28m), plus a merit payment of ~£2.8m per place
    # (20th gets 1 share, 1st gets 20).
    pl_season: int = 2023
    pl_equal: float = 114.0
    pl_merit_per_place: float = 2.8
    pl_growth: float = 0.03  # 2025-29 UK deal +4% a year; international roughly flat to up
    parachute_share: float = 0.45  # Championship season: parachute ~45% of the equal share
    # UEFA prize money, 2024-27 cycle, EUR m, cumulative by exit stage
    # (league phase, play-off, R16, QF, SF, lost final, won). Starting fee, league-phase results
    # and ranking bonus included; the "value pillar" (coefficient + market pool) is separate.
    ucl_prize_eur: tuple = (28.0, 35.0, 50.0, 62.5, 77.5, 96.0, 102.5)
    uel_prize_eur: tuple = (7.0, 9.0, 11.0, 13.5, 17.5, 24.5, 31.0)
    uecl_prize_eur: tuple = (4.0, 5.0, 6.5, 8.0, 10.0, 13.0, 15.5)
    value_pillar_eur: tuple = (20.0, 5.0, 2.0)  # UCL, UEL, UECL
    gbp_per_eur: float = 0.85
    uefa_growth: float = 0.02
    # Old Trafford gate receipts per European home game
    euro_gate: tuple = (5.0, 3.5, 2.5)  # UCL, UEL, UECL
    # Commercial clauses. Adidas: ~£90m a year. The 2015 deal cut payments 30% after two
    # consecutive seasons without Champions League football; the 2025 renewal is reported to
    # keep a similar clause (terms not public, so treat as an assumption).
    kit_deal: float = 90.0
    kit_penalty: float = 0.30
    ucl_commercial_bonus: float = 0.03  # share of commercial tied to UCL clauses/bonuses
    commercial_growth: float = 0.035
    matchday_growth: float = 0.03
    other_broadcast_growth: float = 0.02
    # Relegation hits (share lost)
    relegation_commercial: float = 0.30
    relegation_matchday: float = 0.20


@dataclass
class Costs:
    wage_growth: float = 0.035
    # Senior contracts carry a pay cut when the club is out of the Champions League
    # (widely reported as 25%). Share of the wage bill on such contracts: assumption.
    ucl_pay_cut: float = 0.25
    clause_share: float = 0.5
    europe_bonus_share: float = 0.10  # share of UEFA prize money paid out as bonuses
    relegation_wage_cut: float = 0.35
    other_opex_growth: float = 0.03
    opex_saving: float = 0.0  # INEOS restructuring lever (share of other opex removed)
    # Net player-registration capex. Steady state: gross spend ~ amortisation, and selling
    # players returns part of it (profit on disposal, which adjusted EBITDA excludes).
    # None = calibrate to base-year amortisation / revenue, less player_sale_profit_ratio.
    player_capex_ratio: float | None = None
    player_sale_profit_ratio: float = 0.05  # United's profit on player sales, ~5% of revenue (approx.)
    ppe_capex_ratio: float = 0.04
    tax_rate: float = 0.25


@dataclass
class Valuation:
    # WACC build-up at the deal date: 10y gilt ~4.1%, ERP 5.0%, levered beta 1.0 -> Ke 9.1%;
    # pre-tax cost of debt ~7% (floating term loan + fixed notes); debt ~12% of EV.
    risk_free: float = 0.041
    erp: float = 0.05
    beta: float = 1.0
    cost_of_debt: float = 0.07
    debt_weight: float = 0.12
    tax_rate: float = 0.25
    terminal_growth: float = 0.025
    terminal_window: int = 5  # years of FCFF averaged for the terminal value
    # Exit lens: sell at the end of the horizon at a multiple of that year's revenue.
    # None = the entry multiple (deal EV / trailing revenue at the deal).
    exit_multiple: float | None = None

    @property
    def wacc(self) -> float:
        ke = self.risk_free + self.beta * self.erp
        kd = self.cost_of_debt * (1 - self.tax_rate)
        return (1 - self.debt_weight) * ke + self.debt_weight * kd


@dataclass
class Scenario:
    """Named overrides, applied on top of the defaults (see scenarios())."""
    name: str
    note: str
    changes: dict


def scenarios() -> list[Scenario]:
    return [
        Scenario("Status quo", "Cost base and player spending as in the base year", {}),
        Scenario("INEOS cost plan", "2024-25 restructuring (c.250 roles cut, ~20% of other opex) "
                 "and player spend held to 15% of revenue",
                 {"costs.opex_saving": 0.20, "costs.player_capex_ratio": 0.15}),
        Scenario("Plan + growth", "Cost plan, plus commercial growth of 6% and PL rights growth of 5% a year",
                 {"costs.opex_saving": 0.20, "costs.player_capex_ratio": 0.15,
                  "revenue.commercial_growth": 0.06, "revenue.pl_growth": 0.05}),
    ]


def apply(cfg: "Config", changes: dict) -> "Config":
    import copy
    cfg = copy.deepcopy(cfg)
    for key, value in changes.items():
        obj = cfg
        *path, last = key.split(".")
        for part in path:
            obj = getattr(obj, part)
        if not hasattr(obj, last):
            raise AttributeError(f"unknown setting {key}")
        setattr(obj, last, value)
    return cfg


@dataclass
class Config:
    as_of: str = "2024-02-20"
    horizon: int = 10  # seasons simulated, including the one in progress
    n_paths: int = 10_000
    seed: int = 7
    deal: Deal = field(default_factory=Deal)
    sporting: Sporting = field(default_factory=Sporting)
    revenue: Revenue = field(default_factory=Revenue)
    costs: Costs = field(default_factory=Costs)
    valuation: Valuation = field(default_factory=Valuation)
