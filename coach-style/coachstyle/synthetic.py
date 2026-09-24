"""A fake season with known answers, for testing the statistics.

Every team has a fixed style plus match-to-match noise. `n_changes` teams sack
their manager at mid-season, and the new manager moves their style by `jump`
between-team SDs on every metric. With jump=0 a change does nothing, so the
tests can check that the method finds nothing when there is nothing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import STYLE_METRICS


def season(n_teams: int = 20, n_changes: int = 6, jump: float = 1.0, noise: float = 2.0,
           opp_effect: float = 0.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    teams = [f"Team {i:02d}" for i in range(n_teams)]
    metrics = list(STYLE_METRICS)
    base = rng.normal(0, 1, (n_teams, len(metrics)))
    # How much each team pushes its opponents' metrics around (e.g. everyone has
    # less of the ball against a possession side).
    push = rng.normal(0, opp_effect, (n_teams, len(metrics)))
    quality = rng.normal(0, 0.5, n_teams)
    changers = set(rng.choice(n_teams, n_changes, replace=False))

    # Double round robin via the circle method.
    order = list(range(n_teams))
    rounds = []
    for _ in range(n_teams - 1):
        rounds.append([(order[i], order[-1 - i]) for i in range(n_teams // 2)])
        order = [order[0], order[-1]] + order[1:-1]
    rounds += [[(b, a) for a, b in r] for r in rounds]

    rows, match_id = [], 0
    start = pd.Timestamp("2015-08-15")
    half = len(rounds) // 2
    for rnd, fixtures in enumerate(rounds):
        date = (start + pd.Timedelta(days=7 * rnd)).date().isoformat()
        for h, a in fixtures:
            match_id += 1
            xgd = quality[h] - quality[a] + 0.3 + rng.normal(0, 1)
            gd = int(np.round(xgd + rng.normal(0, 1)))
            for t, o, venue, sign in [(h, a, "H", 1), (a, h, "A", -1)]:
                changed = t in changers and rnd >= half
                style = base[t] + (jump if changed else 0) + push[o] + rng.normal(0, noise, len(metrics))
                rows.append({
                    "match_id": match_id, "date": date, "league": "Test League",
                    "season": "2015/2016", "matchweek": rnd + 1,
                    "team": teams[t], "opponent": teams[o], "venue": venue,
                    "manager": f"{teams[t]} new boss" if changed else f"{teams[t]} boss",
                    "opp_manager": None,
                    "goals_for": max(sign * gd, 0), "goals_against": max(-sign * gd, 0),
                    "points": 3 if sign * gd > 0 else 1 if gd == 0 else 0,
                    **dict(zip(metrics, style)),
                    "np_xg_for": 1.3 + sign * xgd / 2, "np_xg_against": 1.3 - sign * xgd / 2,
                })
    return pd.DataFrame(rows)
