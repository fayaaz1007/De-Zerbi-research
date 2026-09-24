"""Premier League results from openfootball (public domain, hosted on GitHub).

    https://github.com/openfootball/football.json  ->  <season>/en.1.json

One row per fixture. Unplayed fixtures keep NaN goals, so the same table holds the
finished part of a season and the fixtures still to simulate.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import requests

URL = "https://raw.githubusercontent.com/openfootball/football.json/master/{season}/en.1.json"
FIRST_SEASON = 2010  # earliest season openfootball publishes as JSON
MUFC = "Manchester United"


def season_label(start_year: int) -> str:
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def clean_name(name: str) -> str:
    for token in (" FC", " AFC"):
        if name.endswith(token):
            name = name[: -len(token)]
    return name.removeprefix("AFC ").strip()


def _score(match: dict):
    """openfootball stores a score as {"ft": [h, a], ...} or as a bare [h, a] list."""
    s = match.get("score")
    if isinstance(s, dict):
        s = s.get("ft")
    if isinstance(s, (list, tuple)) and len(s) == 2:
        return float(s[0]), float(s[1])
    return np.nan, np.nan


def parse_season(payload: dict, start_year: int) -> pd.DataFrame:
    rows = []
    for m in payload["matches"]:
        hg, ag = _score(m)
        rows.append({"season": start_year, "date": m["date"],
                     "home": clean_name(m["team1"]), "away": clean_name(m["team2"]),
                     "hg": hg, "ag": ag})
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def fetch(first: int = FIRST_SEASON, last: int | None = None, session=None) -> pd.DataFrame:
    """Download every season from `first` until the first season that is not published."""
    session = session or requests.Session()
    frames, year = [], first
    while last is None or year <= last:
        r = session.get(URL.format(season=season_label(year)), timeout=30)
        if r.status_code == 404:
            break
        r.raise_for_status()
        frames.append(parse_season(r.json(), year))
        year += 1
    if not frames:
        raise RuntimeError("no seasons downloaded")
    return pd.concat(frames, ignore_index=True).sort_values(["date", "home"], ignore_index=True)


def load(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values(["date", "home"], ignore_index=True)


def as_of(df: pd.DataFrame, date) -> pd.DataFrame:
    """Hide every result on or after `date`, so a valuation only uses what was known then."""
    df = df.copy()
    future = df["date"] >= pd.Timestamp(date)
    df.loc[future, ["hg", "ag"]] = np.nan
    return df


def table(season: pd.DataFrame) -> pd.DataFrame:
    """League table from the played matches (points, then goal difference, then goals)."""
    played = season.dropna(subset=["hg", "ag"])
    teams = sorted(set(season["home"]) | set(season["away"]))
    rows = {t: dict(team=t, p=0, w=0, d=0, l=0, gf=0, ga=0, pts=0) for t in teams}
    for h, a, hg, ag in played[["home", "away", "hg", "ag"]].itertuples(index=False):
        for team, f, g in ((h, hg, ag), (a, ag, hg)):
            r = rows[team]
            r["p"] += 1
            r["gf"] += f
            r["ga"] += g
            if f > g:
                r["w"] += 1
                r["pts"] += 3
            elif f == g:
                r["d"] += 1
                r["pts"] += 1
            else:
                r["l"] += 1
    t = pd.DataFrame(rows.values())
    t["gd"] = t["gf"] - t["ga"]
    t = t.sort_values(["pts", "gd", "gf"], ascending=False, ignore_index=True)
    t["pos"] = np.arange(1, len(t) + 1)
    return t


def final_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Finishing position of every team in every completed season."""
    out = []
    for season, g in df.groupby("season"):
        if g["hg"].isna().any():
            continue
        t = table(g)
        t["season"] = season
        out.append(t[["season", "team", "pos", "pts", "gd"]])
    return pd.concat(out, ignore_index=True)
