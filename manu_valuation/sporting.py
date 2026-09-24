"""Season-level team strength, its dynamics, and the league/Europe simulator.

Strength theta is measured in Elo points relative to the league average of that season.

1. `season_strengths` fits one theta per team-season to that season's results through the
   Elo-calibrated outcome link (a penalised maximum-likelihood fit, like a Bradley-Terry
   rating for one season).
2. `fit_dynamics` estimates how theta moves from season to season with a two-part
   state-space model (a drifting club level plus fading form), fitted with a Kalman filter.
3. `simulate_league` plays out many seasons at once in numpy, for every path.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from .elo import OutcomeModel

# ---------------------------------------------------------------- season strengths


def _nll_grad(theta, h, a, y, link: OutcomeModel, prior_mean, prior_prec):
    z = link.b * (theta[h] - theta[a]) / 100
    s1, s2 = expit(link.c1 - z), expit(link.c2 - z)
    p = np.where(y == 1, 1 - s2, np.where(y == 0, s2 - s1, s1))
    p = np.clip(p, 1e-12, None)
    # d log p / d z for each outcome
    dz = np.where(y == 1, s2,
                  np.where(y == 0, -(s2 * (1 - s2) - s1 * (1 - s1)) / p, -(1 - s1)))
    g = np.zeros_like(theta)
    np.add.at(g, h, dz * link.b / 100)
    np.add.at(g, a, -dz * link.b / 100)
    dev = theta - prior_mean
    nll = -np.log(p).sum() + 0.5 * (prior_prec * dev**2).sum()
    return nll, -g + prior_prec * dev


def season_strengths(season: pd.DataFrame, link: OutcomeModel, prior: pd.DataFrame | None = None,
                     ridge_sd: float = 400.0) -> pd.DataFrame:
    """Strength of every team from the season's played matches.

    `prior` (columns team, mean, sd) turns the fit into a posterior mode; teams missing
    from it get a flat-ish N(0, ridge_sd) prior. Without a prior the result is centred on 0.
    """
    teams = sorted(set(season["home"]) | set(season["away"]))
    ix = {t: i for i, t in enumerate(teams)}
    played = season.dropna(subset=["hg", "ag"])
    h = played["home"].map(ix).to_numpy()
    a = played["away"].map(ix).to_numpy()
    y = np.sign(played["hg"] - played["ag"]).to_numpy().astype(int)
    mean = np.zeros(len(teams))
    sd = np.full(len(teams), ridge_sd)
    if prior is not None:
        pr = prior.set_index("team")
        for t, i in ix.items():
            if t in pr.index:
                mean[i], sd[i] = pr.loc[t, "mean"], pr.loc[t, "sd"]
    prec = 1 / sd**2
    res = minimize(_nll_grad, mean.copy(), jac=True, method="L-BFGS-B",
                   args=(h, a, y, link, mean, prec))
    theta = res.x
    # observed information by finite differences of the analytic gradient
    eps = 1e-3
    hess = np.empty((len(teams), len(teams)))
    for i in range(len(teams)):
        d = np.zeros(len(teams))
        d[i] = eps
        hess[i] = (_nll_grad(theta + d, h, a, y, link, mean, prec)[1]
                   - _nll_grad(theta - d, h, a, y, link, mean, prec)[1]) / (2 * eps)
    cov = np.linalg.inv(hess)
    if prior is None:
        # only differences are identified: report the centred ratings and their covariance
        centre = np.eye(len(teams)) - 1 / len(teams)
        theta, cov = centre @ theta, centre @ cov @ centre.T
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    n = pd.concat([played["home"], played["away"]]).value_counts()
    return pd.DataFrame({"team": teams, "theta": theta, "se": se,
                         "played": [int(n.get(t, 0)) for t in teams]})


def all_season_strengths(df: pd.DataFrame, link: OutcomeModel) -> pd.DataFrame:
    """theta-hat for every *completed* season."""
    out = []
    for season, g in df.groupby("season"):
        if g["hg"].isna().any():
            continue
        s = season_strengths(g, link)
        s.insert(0, "season", season)
        out.append(s)
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------- dynamics
#
# State-space model per club, seasons t = 1, 2, ...
#   theta_t   = level_t + form_t                 (true strength that season)
#   level_t   = level_{t-1} + sigma_level * e1   (permanent shifts: owners, managers, recruitment)
#   form_t    = rho * form_{t-1} + sigma_form * e2   (transient, fades)
#   theta-hat = theta_t + measurement error with the fitted standard error
# It nests "fixed club mean + AR(1)" (sigma_level = 0) and a pure local level (rho = 0).
# A Kalman filter gives the likelihood; seasons a club spent outside the league are
# bridged by propagating the state. Parameters are pooled across clubs by maximum likelihood.

INIT_LEVEL_SD = 150.0


def _step(m, S, rho, q_level, q_form, k: int = 1):
    F = np.array([[1.0, 0.0], [0.0, rho]])
    Q = np.diag([q_level, q_form])
    for _ in range(k):
        m = F @ m
        S = F @ S @ F.T + Q
    return m, S


def _init(rho, sigma_form):
    return np.zeros(2), np.diag([INIT_LEVEL_SD**2, sigma_form**2 / max(1 - rho**2, 1e-6)])


def _panel(strengths: pd.DataFrame):
    """theta-hat and its standard error on a (season x club) grid, NaN where not in the league."""
    y = strengths.pivot(index="season", columns="team", values="theta")
    se = strengths.pivot(index="season", columns="team", values="se").reindex_like(y)
    return y.index.to_numpy(), list(y.columns), y.to_numpy(), se.to_numpy()


def _filter(y, se, rho, sigma_level, sigma_form):
    """Kalman filter run for every club at once. y, se: (seasons, clubs).
    Returns (total loglik, index of each club's last observed season, means (C,2), covs (C,2,2))."""
    T, C = y.shape
    F = np.array([[1.0, 0.0], [0.0, rho]])
    Q = np.diag([sigma_level**2, sigma_form**2])
    m0, S0 = _init(rho, sigma_form)
    m = np.tile(m0, (C, 1))
    S = np.tile(S0, (C, 1, 1))
    started = np.zeros(C, dtype=bool)
    last = np.full(C, -1)
    ll = 0.0
    for t in range(T):
        if t > 0:  # predict (clubs not seen yet stay at the initial state)
            mp = m @ F.T
            Sp = F @ S @ F.T + Q
            m = np.where(started[:, None], mp, m)
            S = np.where(started[:, None, None], Sp, S)
        obs = ~np.isnan(y[t])
        f = S[:, 0, 0] + S[:, 1, 1] + 2 * S[:, 0, 1] + np.nan_to_num(se[t]) ** 2
        v = np.nan_to_num(y[t]) - m.sum(axis=1)
        ll += float((-0.5 * (np.log(2 * np.pi * f) + v * v / f))[obs].sum())
        SH = S.sum(axis=2)  # S @ [1, 1]
        K = SH / f[:, None]
        m_new = m + K * v[:, None]
        S_new = S - K[:, :, None] * SH[:, None, :]
        m = np.where(obs[:, None], m_new, m)
        S = np.where(obs[:, None, None], S_new, S)
        started |= obs
        last = np.where(obs, t, last)
    return ll, last, m, S


def _roll_back(y, se, last, rho, sigma_level, sigma_form):
    """Filtered state at each club's *last observed* season (the grid filter keeps
    propagating after a club leaves the league)."""
    out = {}
    for c in range(y.shape[1]):
        _, _, m, S = _filter(y[: last[c] + 1, [c]], se[: last[c] + 1, [c]], rho, sigma_level, sigma_form)
        out[c] = (m[0], S[0])
    return out


@dataclass
class Dynamics:
    rho: float
    sigma_level: float
    sigma_form: float
    states: dict  # club -> (last season observed, state mean [level, form], 2x2 cov)
    promoted_mean: float
    promoted_level_sd: float
    top4_theta: float  # average theta of top-4 finishers (reference level for Europe)
    loglik: float
    diagnostics: dict = field(default_factory=dict)

    @property
    def form_sd(self) -> float:
        """Stationary spread of the transient part."""
        return self.sigma_form / np.sqrt(max(1 - self.rho**2, 1e-6))

    def predict(self, team: str, season: int):
        """Predicted [level, form] mean and covariance for `season` (before any of its games)."""
        if team in self.states:
            last, m, S = self.states[team]
            if season > last:
                return _step(m, S, self.rho, self.sigma_level**2, self.sigma_form**2, int(season - last))
            return m, S
        return self.promoted_state()

    def promoted_state(self):
        return (np.array([self.promoted_mean, 0.0]),
                np.diag([self.promoted_level_sd**2, self.form_sd**2]))

    def theta_prior(self, team: str, season: int) -> tuple[float, float]:
        m, S = self.predict(team, season)
        return float(m.sum()), float(np.sqrt(S.sum()))

    def level(self, team: str) -> float:
        """Filtered level at the club's last observed season."""
        return float(self.states[team][1][0]) if team in self.states else self.promoted_mean


def fit_dynamics(strengths: pd.DataFrame, positions: pd.DataFrame | None = None) -> Dynamics:
    seasons, clubs, y, se = _panel(strengths)

    def nll(z):
        return -_filter(y, se, np.tanh(z[0]), np.exp(z[1]), np.exp(z[2]))[0]

    best = None
    for start in ([0.5, np.log(25), np.log(50)], [1.5, np.log(5), np.log(40)], [0.1, np.log(40), np.log(20)]):
        res = minimize(nll, start, method="Nelder-Mead", options={"xatol": 1e-5, "fatol": 1e-5, "maxiter": 3000})
        if best is None or res.fun < best.fun:
            best = res
    rho, sl, sf = np.tanh(best.x[0]), np.exp(best.x[1]), np.exp(best.x[2])
    _, last, _, _ = _filter(y, se, rho, sl, sf)
    rolled = _roll_back(y, se, last, rho, sl, sf)
    states = {club: (int(seasons[last[c]]), *rolled[c]) for c, club in enumerate(clubs)}

    s = strengths
    first = s[s["season"] > s["season"].min()]
    prev = set(zip(s["season"] + 1, s["team"]))
    promoted = first[[(se_, t) not in prev for se_, t in zip(first["season"], first["team"])]]
    form_var = sf**2 / max(1 - rho**2, 1e-6)
    level_var = promoted["theta"].var() - (promoted["se"] ** 2).mean() - form_var
    top4 = 0.0
    if positions is not None:
        j = strengths.merge(positions, on=["season", "team"])
        top4 = float(j.loc[j["pos"] <= 4, "theta"].mean())
    diag = dict(n_obs=len(s), n_clubs=len(clubs), me_sd=float(np.sqrt((s["se"] ** 2).mean())),
                n_promoted=len(promoted))
    return Dynamics(float(rho), float(sl), float(sf), states, float(promoted["theta"].mean()),
                    float(np.sqrt(max(level_var, 30.0**2))), top4, float(-best.fun), diag)


# ---------------------------------------------------------------- simulation


def _fixture_matrices(n: int):
    h, a = np.array([(i, j) for i in range(n) for j in range(n) if i != j]).T
    H = np.zeros((len(h), n))
    A = np.zeros((len(h), n))
    H[np.arange(len(h)), h] = 1
    A[np.arange(len(a)), a] = 1
    return h, a, H, A


def play(theta: np.ndarray, h: np.ndarray, a: np.ndarray, link: OutcomeModel, rng) -> tuple:
    """Simulate fixtures (h[k] v a[k]) on every path. theta is (paths, teams).
    Returns home and away points, each (paths, fixtures)."""
    ph, pd_, _ = link.probs(theta[:, h] - theta[:, a])
    u = rng.random(ph.shape)
    home_win, draw = u < ph, (u >= ph) & (u < ph + pd_)
    hp = np.where(home_win, 3, np.where(draw, 1, 0))
    ap = np.where(home_win, 0, np.where(draw, 1, 3))
    return hp, ap


def rank(points: np.ndarray, rng) -> np.ndarray:
    """Final positions (1 = champion) from points, ties broken at random
    (goals are not simulated; goal difference would break most real ties)."""
    key = points + rng.random(points.shape) * 0.5
    order = np.argsort(-key, axis=1)
    pos = np.empty_like(order)
    np.put_along_axis(pos, order, np.arange(1, points.shape[1] + 1)[None, :].repeat(len(points), 0), 1)
    return pos


def simulate_league(theta: np.ndarray, link: OutcomeModel, rng, base_points=None,
                    fixtures: tuple | None = None) -> np.ndarray:
    """Positions for one season. `fixtures` = (h, a) index arrays of the matches still to play
    (defaults to a full double round robin); `base_points` = points already banked."""
    n = theta.shape[1]
    if fixtures is None:
        h, a, H, A = _fixture_matrices(n)
    else:
        h, a = fixtures
        H = np.eye(n)[h]
        A = np.eye(n)[a]
    hp, ap = play(theta, h, a, link, rng)
    pts = hp @ H + ap @ A
    if base_points is not None:
        pts = pts + base_points
    return rank(pts, rng)


# ---------------------------------------------------------------- Europe

STAGES = ["league", "playoff", "r16", "qf", "sf", "final", "winner"]  # exit stage (winner = won it)


def europe_stage(theta: np.ndarray, comp: np.ndarray, cfg, ref: float, rng) -> np.ndarray:
    """Stage index reached (0..6, see STAGES) in the European competition `comp`
    (0 none, 1 UCL, 2 UEL, 3 UECL). -1 where not in Europe.

    Each hurdle is passed with a probability whose log-odds move by `cfg.europe_slope`
    per 100 Elo above the reference level (an average top-4 Premier League side)."""
    out = np.full(theta.shape, -1)
    x = (theta - ref) / 100 * cfg.europe_slope
    for c, probs in ((1, cfg.ucl_advance), (2, cfg.uel_advance), (3, cfg.uecl_advance)):
        m = comp == c
        if not m.any():
            continue
        top24, top8, playoff, r16, qf, sf, final = probs
        logit = lambda p: np.log(p / (1 - p))  # noqa: E731
        adv = lambda p: rng.random(m.sum()) < expit(logit(p) + x[m])  # noqa: E731
        stage = np.zeros(m.sum(), dtype=int)  # out in league phase
        alive = adv(top24)
        direct = alive & adv(top8)
        through_playoff = alive & ~direct & adv(playoff)
        stage[alive & ~direct & ~through_playoff] = 1
        alive = direct | through_playoff
        for s, p in zip(range(2, 6), (r16, qf, sf, final)):
            nxt = alive & adv(p)
            stage[alive & ~nxt] = s
            alive = nxt
        stage[alive] = 6
        out[m] = stage
    return out
