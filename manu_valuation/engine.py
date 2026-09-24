"""Multi-season sporting paths for Manchester United.

`fit` uses only results known at the valuation date. `simulate` then produces, for every
path and every season from the one in progress onwards: United's league finish (99 = in the
Championship), the European competition played, the stage reached, and United's strength.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import elo, results, sporting
from .config import KNOWN_EUROPE, KNOWN_FA_CUP, Config

CHAMPIONSHIP = 99


@dataclass
class SportingModel:
    as_of: pd.Timestamp
    matches: pd.DataFrame  # all fixtures, results after as_of hidden
    link: elo.OutcomeModel
    elo_run: elo.EloRun
    elo_grid: pd.DataFrame
    strengths: pd.DataFrame  # theta-hat of completed seasons
    positions: pd.DataFrame
    dynamics: sporting.Dynamics


@dataclass
class Paths:
    seasons: np.ndarray  # (Y,) season start years
    finish: np.ndarray  # (P, Y) 1..20 or CHAMPIONSHIP
    comp: np.ndarray  # (P, Y) 0 none, 1 UCL, 2 UEL, 3 UECL
    stage: np.ndarray  # (P, Y) exit stage (see sporting.STAGES), -1 if not in Europe
    theta: np.ndarray  # (P, Y) United's strength
    level: np.ndarray  # (P,) United's underlying level at the start of each path
    prev_ucl: bool  # United were in the UCL the season before the first one
    first_fraction: float  # share of the first fiscal year still ahead of the valuation date
    start_prior: pd.DataFrame  # current-season strength posterior of every club

    @property
    def ucl(self):
        return self.comp == 1


def season_of(date) -> int:
    d = pd.Timestamp(date)
    return d.year if d.month >= 7 else d.year - 1


def fit(all_matches: pd.DataFrame, as_of) -> SportingModel:
    as_of = pd.Timestamp(as_of)
    known = results.as_of(all_matches, as_of)
    known = known[known["season"] <= season_of(as_of)]
    run, link, grid = elo.calibrate(known)
    strengths = sporting.all_season_strengths(known, link)
    positions = results.final_positions(known)
    dyn = sporting.fit_dynamics(strengths, positions)
    return SportingModel(as_of, known, link, run, grid, strengths, positions, dyn)


def _known(table: dict, season: int, key: str, date_key: str, as_of) -> object | None:
    entry = table.get(season)
    if entry is None or entry.get(date_key) is None or pd.Timestamp(entry[date_key]) > as_of:
        return None
    return entry[key]


def _prior(model: SportingModel, teams: list, season: int) -> pd.DataFrame:
    rows = [(t, *model.dynamics.theta_prior(t, season)) for t in teams]
    return pd.DataFrame(rows, columns=["team", "mean", "sd"])


def _next_comp(finish, comp, stage, cup, rng, cfg):
    s = cfg.sporting
    ucl = (finish <= 4) | ((finish == 5) & (rng.random(finish.shape) < s.p_fifth_ucl)) \
        | ((comp == 2) & (stage == 6))
    uel = ~ucl & (np.isin(finish, s.uel_places) | cup | ((comp == 3) & (stage == 6)))
    uecl = ~ucl & ~uel & np.isin(finish, s.uecl_places)
    return np.select([ucl, uel, uecl], [1, 2, 3], 0)


def _cup(finish, rng, cfg):
    p = np.select([finish <= 4, finish <= 8, finish <= 20], list(cfg.sporting.fa_cup_win), 0.01)
    return rng.random(finish.shape) < p


def simulate(model: SportingModel, cfg: Config, rng=None, n_paths: int | None = None) -> Paths:
    rng = rng or np.random.default_rng(cfg.seed)
    P = n_paths or cfg.n_paths
    Y = cfg.horizon
    as_of, d, link, s = model.as_of, model.dynamics, model.link, cfg.sporting
    t0 = season_of(as_of)
    season0 = model.matches[model.matches["season"] == t0]
    if season0.empty:
        raise ValueError(f"no fixtures for season {t0} in the results data")
    teams = sorted(set(season0["home"]) | set(season0["away"]), key=lambda t: (t != results.MUFC, t))
    if teams[0] != results.MUFC:
        raise ValueError("the engine assumes United start the valuation season in the Premier League")
    ix = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    # --- the season in progress: posterior strength given the results so far; then each
    # club's underlying level and form, drawn conditionally on that season's strength
    prior = _prior(model, teams, t0)
    post = sporting.season_strengths(season0, link, prior).set_index("team").loc[teams]
    theta = post["theta"].to_numpy()[None, :] + post["se"].to_numpy()[None, :] * rng.standard_normal((P, n))
    level = np.empty((P, n))
    for i, t in enumerate(teams):
        m, S = d.predict(t, t0)
        cov = S[0, 0] + S[0, 1]
        k = cov / S.sum()
        sd = np.sqrt(max(S[0, 0] - k * cov, 0.0))
        level[:, i] = m[0] + k * (theta[:, i] - m.sum()) + sd * rng.standard_normal(P)
    form = theta - level
    if s.level_override is not None:
        level[:, 0], form[:, 0] = s.level_override, 0.0

    table = results.table(season0).set_index("team").reindex(teams)
    base_points = table["pts"].fillna(0).to_numpy()[None, :]
    todo = season0[season0["hg"].isna()]
    fixtures0 = (todo["home"].map(ix).to_numpy(), todo["away"].map(ix).to_numpy())

    finish = np.zeros((P, Y), dtype=int)
    comp = np.zeros((P, Y), dtype=int)
    stage = np.full((P, Y), -1)
    th_utd = np.zeros((P, Y))
    in_pl = np.ones(P, dtype=bool)
    utd, utd_level, utd_form = theta[:, 0].copy(), level[:, 0].copy(), form[:, 0].copy()
    start_level = utd_level.copy()

    known_comp = _known(KNOWN_EUROPE, t0, "comp", "qualified", as_of)
    if known_comp is None:  # fall back to last season's finish (ignores cups)
        last = model.positions.query("season == @t0 - 1 and team == @results.MUFC")["pos"]
        known_comp = int(_next_comp(last.to_numpy(), np.zeros(1), np.zeros(1), np.zeros(1, bool),
                                    np.random.default_rng(0), cfg)[0]) if len(last) else 0
    comp[:, 0] = known_comp
    prev_ucl = KNOWN_EUROPE.get(t0 - 1, {}).get("comp") == 1

    for y in range(Y):
        season = t0 + y
        if y > 0:
            level = level + d.sigma_level * rng.standard_normal((P, n))
            form = d.rho * form + d.sigma_form * rng.standard_normal((P, n))
            utd_level = utd_level + d.sigma_level * rng.standard_normal(P)
            utd_form = d.rho * utd_form + d.sigma_form * rng.standard_normal(P)
            utd = utd_level + utd_form
            level[:, 0] = np.where(in_pl, utd_level, level[:, 0])
            form[:, 0] = np.where(in_pl, utd_form, form[:, 0])
            theta = level + form
            pos = sporting.simulate_league(theta, link, rng)
        else:
            pos = sporting.simulate_league(theta, link, rng, base_points, fixtures0)
        th_utd[:, y] = utd
        finish[:, y] = np.where(in_pl, pos[:, 0], CHAMPIONSHIP)

        st = _known(KNOWN_EUROPE, season, "stage", "decided", as_of) if y == 0 else None
        stage[:, y] = st if st is not None else sporting.europe_stage(utd, comp[:, y], s, d.top4_theta, rng)
        stage[comp[:, y] == 0, y] = -1

        won = _known(KNOWN_FA_CUP, season, "won", "decided", as_of) if y == 0 else None
        cup = np.full(P, bool(won)) if won is not None else _cup(finish[:, y], rng, cfg)

        if y + 1 < Y:
            nxt = _known(KNOWN_EUROPE, season + 1, "comp", "qualified", as_of)
            comp[:, y + 1] = nxt if nxt is not None else _next_comp(finish[:, y], comp[:, y], stage[:, y], cup, rng, cfg)

        # promotion and relegation: relegated clubs are replaced by newly promoted ones
        relegated = pos >= n - 2
        level = np.where(relegated, d.promoted_mean + d.promoted_level_sd * rng.standard_normal((P, n)), level)
        form = np.where(relegated, d.form_sd * rng.standard_normal((P, n)), form)
        down = in_pl & relegated[:, 0]
        back = ~in_pl & (rng.random(P) < s.p_return_from_championship)
        in_pl = (in_pl & ~down) | back
        # while United are down, slot 0 holds another club
        level[:, 0] = np.where(in_pl, utd_level, level[:, 0])
        form[:, 0] = np.where(in_pl, utd_form, form[:, 0])

    return Paths(np.arange(t0, t0 + Y), finish, comp, stage, th_utd, start_level, prev_ucl,
                 _remaining_fraction(as_of, t0), prior.merge(post.reset_index(), on="team"))


def _remaining_fraction(as_of: pd.Timestamp, season: int) -> float:
    start, end = pd.Timestamp(f"{season}-07-01"), pd.Timestamp(f"{season + 1}-07-01")
    return float(np.clip((end - as_of) / (end - start), 0.0, 1.0))
