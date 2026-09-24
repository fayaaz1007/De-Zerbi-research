"""Elo ratings for Premier League teams and a calibrated win/draw/loss model.

Ratings update after every match (goal-difference weighted, as in World Football Elo).
Between seasons, ratings shrink towards the league mean, and promoted clubs start at
the average rating of the clubs they replaced.

The Elo "expected score" does not give a draw probability, so match outcomes come from
an ordered logit on the rating gap, fitted to the same history:

    P(away win)            = logistic(c1 - b * gap)
    P(away win or draw)    = logistic(c2 - b * gap),   gap = (R_home - R_away) / 100

The intercepts absorb home advantage.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

BASE = 1500.0


@dataclass
class EloParams:
    k: float = 20.0
    home_adv: float = 60.0
    carryover: float = 0.8  # share of (rating - mean) kept from one season to the next


@dataclass
class OutcomeModel:
    b: float
    c1: float
    c2: float

    def probs(self, gap):
        """(P home win, P draw, P away win) for rating gap(s) R_home - R_away."""
        x = np.asarray(gap, dtype=float) / 100
        p_away = expit(self.c1 - self.b * x)
        p_not_home = expit(self.c2 - self.b * x)
        return 1 - p_not_home, p_not_home - p_away, p_away


@dataclass
class EloRun:
    params: EloParams
    matches: pd.DataFrame  # played matches with pre-match ratings
    season_end: pd.DataFrame  # season, team, rating at the end of the season
    current: dict = field(default_factory=dict)  # latest rating of every team seen


def _gd_mult(gd: np.ndarray) -> np.ndarray:
    gd = np.abs(gd)
    return np.where(gd <= 1, 1.0, np.where(gd == 2, 1.5, (11 + gd) / 8))


def run(df: pd.DataFrame, p: EloParams = EloParams()) -> EloRun:
    """Walk through the fixtures in date order. Unplayed fixtures are skipped."""
    ratings: dict[str, float] = {}
    rows, ends = [], []
    for season, g in df.sort_values("date").groupby("season", sort=True):
        teams = set(g["home"]) | set(g["away"])
        if ratings:
            mean = np.mean(list(ratings.values()))
            stayed = {t: mean + p.carryover * (r - mean) for t, r in ratings.items() if t in teams}
            dropped = [r for t, r in ratings.items() if t not in teams]
            entry = np.mean(dropped) if dropped else mean
            entry = mean + p.carryover * (entry - mean)
            ratings = {t: stayed.get(t, entry) for t in teams}
        else:
            ratings = {t: BASE for t in teams}
        for h, a, hg, ag, date in g[["home", "away", "hg", "ag", "date"]].itertuples(index=False):
            if np.isnan(hg):
                continue
            rh, ra = ratings[h], ratings[a]
            exp_h = 1 / (1 + 10 ** (-(rh + p.home_adv - ra) / 400))
            s = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
            delta = p.k * _gd_mult(np.array(hg - ag)) * (s - exp_h)
            ratings[h], ratings[a] = rh + delta, ra - delta
            rows.append((season, date, h, a, hg, ag, rh, ra))
        ends += [(season, t, r) for t, r in ratings.items()]
    matches = pd.DataFrame(rows, columns=["season", "date", "home", "away", "hg", "ag", "r_home", "r_away"])
    matches["gap"] = matches["r_home"] - matches["r_away"]
    matches["outcome"] = np.sign(matches["hg"] - matches["ag"]).astype(int)  # 1 home, 0 draw, -1 away
    season_end = pd.DataFrame(ends, columns=["season", "team", "rating"])
    return EloRun(p, matches, season_end, dict(ratings))


def fit_outcomes(matches: pd.DataFrame) -> OutcomeModel:
    x = matches["gap"].to_numpy() / 100
    y = matches["outcome"].to_numpy()

    def nll(theta):
        b, c1, d = theta
        m = OutcomeModel(b, c1, c1 + np.exp(d))
        ph, pd_, pa = m.probs(x * 100)
        p = np.where(y == 1, ph, np.where(y == 0, pd_, pa))
        return -np.log(np.clip(p, 1e-12, None)).sum()

    res = minimize(nll, x0=[0.5, -1.0, 0.0], method="Nelder-Mead",
                   options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 4000})
    b, c1, d = res.x
    return OutcomeModel(b, c1, c1 + np.exp(d))


def log_loss(model: OutcomeModel, matches: pd.DataFrame) -> float:
    ph, pd_, pa = model.probs(matches["gap"].to_numpy())
    y = matches["outcome"].to_numpy()
    p = np.where(y == 1, ph, np.where(y == 0, pd_, pa))
    return float(-np.log(p).mean())


def calibrate(df: pd.DataFrame, burn_in: int = 1,
              ks=(12, 16, 20, 24, 28, 32), carryovers=(0.6, 0.7, 0.8, 0.9)):
    """Grid-search K and carry-over on the log loss of pre-match predictions.

    The first `burn_in` seasons are skipped because every club starts at the same rating.
    Returns (best EloRun, fitted OutcomeModel, grid of log losses).
    """
    first = df["season"].min() + burn_in
    grid = []
    for k in ks:
        for c in carryovers:
            r = run(df, EloParams(k=k, carryover=c))
            m = r.matches[r.matches["season"] >= first]
            grid.append((k, c, log_loss(fit_outcomes(m), m)))
    grid = pd.DataFrame(grid, columns=["k", "carryover", "log_loss"])
    best = grid.loc[grid["log_loss"].idxmin()]
    r = run(df, EloParams(k=float(best["k"]), carryover=float(best["carryover"])))
    m = r.matches[r.matches["season"] >= first]
    return r, fit_outcomes(m), grid
