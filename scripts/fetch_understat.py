"""Build the De Zerbi match dataset from Understat (run this on your own machine).

Understat gives, per league match: xG, xGA, goals, and PPDA for both sides.
It does NOT give possession or defensive-line height; add those afterwards
with scripts/merge_columns.py (see README).

    python scripts/fetch_understat.py --out data/raw/dezerbi_matches.csv

For every De Zerbi match it writes:
  opp_ppda       opponent's PPDA *in this match* (partly caused by De Zerbi's team)
  opp_ppda_pre   opponent's PPDA over the rest of the season, excluding this match
                 (how hard they *usually* press, the cleaner measure)
  opp_strength   opponent's xG difference per match over the rest of the season
"""
from __future__ import annotations

import argparse
import codecs
import json
import re
import time

import pandas as pd
import requests

# (understat league, season start year, understat team name, first date, last date)
# Check the dates. They trim matches before/after De Zerbi was in charge.
SPELLS = [
    ("Serie_A", 2018, "Sassuolo", "2018-07-01", "2019-06-30"),
    ("Serie_A", 2019, "Sassuolo", "2019-07-01", "2020-08-31"),
    ("Serie_A", 2020, "Sassuolo", "2020-09-01", "2021-06-30"),
    ("EPL", 2022, "Brighton", "2022-09-18", "2023-06-30"),
    ("EPL", 2023, "Brighton", "2023-07-01", "2024-06-30"),
    ("Ligue_1", 2024, "Marseille", "2024-07-01", "2025-06-30"),
    ("Ligue_1", 2025, "Marseille", "2025-07-01", "2026-06-30"),
]

HEADERS = {"User-Agent": "Mozilla/5.0 (research script)", "X-Requested-With": "XMLHttpRequest"}


def fetch_teams_data(league: str, season: int) -> dict:
    """Return Understat's per-team match history for a league season."""
    # Newer Understat serves the data from a JSON endpoint...
    r = requests.get(f"https://understat.com/getLeagueData/{league}/{season}",
                     headers=HEADERS, timeout=30)
    if r.ok and r.headers.get("content-type", "").startswith("application/json"):
        return r.json()["teams"]
    # ...older Understat embeds it in the page as JSON.parse('<hex-escaped>').
    r = requests.get(f"https://understat.com/league/{league}/{season}", headers=HEADERS, timeout=30)
    r.raise_for_status()
    m = re.search(r"var\s+teamsData\s*=\s*JSON\.parse\('(.*?)'\)", r.text)
    if not m:
        raise RuntimeError(f"could not find team data for {league} {season}")
    return json.loads(codecs.decode(m.group(1), "unicode_escape"))


def history_frame(teams: dict) -> pd.DataFrame:
    rows = []
    for t in teams.values():
        for h in t["history"]:
            rows.append({
                "team": t["title"], "datetime": h["date"], "h_a": h["h_a"],
                "xg": float(h["xG"]), "xga": float(h["xGA"]),
                "scored": int(h["scored"]), "missed": int(h["missed"]),
                "ppda_att": float(h["ppda"]["att"]), "ppda_def": float(h["ppda"]["def"]),
            })
    return pd.DataFrame(rows)


def build_spell(league, season, team, start, end) -> pd.DataFrame:
    hist = history_frame(fetch_teams_data(league, season))
    hist["xgd"] = hist["xg"] - hist["xga"]

    # Understat's history has no opponent name. Pair each row with the row of
    # the other team at the same kick-off whose xG/goals mirror it.
    key = ["datetime", "xg", "xga", "scored", "missed"]
    mirror = hist.rename(columns={"xg": "xga", "xga": "xg", "scored": "missed", "missed": "scored"})
    pairs = hist.merge(mirror[key + ["team"]], on=key, suffixes=("", "_opp"))
    pairs = pairs[pairs["team"] != pairs["team_opp"]]

    # Leave-one-out season profile of every team: totals minus the match itself.
    tot = hist.groupby("team")[["ppda_att", "ppda_def", "xgd"]].sum()
    n = hist.groupby("team").size()

    ours = pairs[pairs["team"] == team].copy()
    if ours.empty:
        raise RuntimeError(f"{team} not found in {league} {season}")
    opp_rows = hist.set_index(["team", "datetime"])
    rows = []
    for _, r in ours.iterrows():
        o = opp_rows.loc[(r["team_opp"], r["datetime"])]
        opp = r["team_opp"]
        rows.append({
            "date": r["datetime"][:10],
            "season": f"{season}-{str(season + 1)[2:]}",
            "team": team, "competition": league, "opponent": opp,
            "venue": "H" if r["h_a"] == "h" else "A",
            "goals_for": r["scored"], "goals_against": r["missed"],
            "xg_for": r["xg"], "xg_against": r["xga"],
            "possession": None,
            "opp_ppda": o["ppda_att"] / o["ppda_def"] if o["ppda_def"] else None,
            "opp_ppda_pre": (tot.loc[opp, "ppda_att"] - o["ppda_att"])
                            / (tot.loc[opp, "ppda_def"] - o["ppda_def"]),
            "opp_strength": (tot.loc[opp, "xgd"] - o["xgd"]) / (n[opp] - 1),
            "opp_line_height_pre": None,
        })
    out = pd.DataFrame(rows)
    return out[(out["date"] >= start) & (out["date"] <= end)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/dezerbi_matches.csv")
    args = ap.parse_args()
    frames = []
    for spell in SPELLS:
        print("fetching", *spell[:3])
        try:
            frames.append(build_spell(*spell))
        except Exception as e:  # keep going if one season fails
            print(f"  skipped: {e}")
        time.sleep(2)
    df = pd.concat(frames).sort_values("date")
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} matches to {args.out}")
    print("Next: add possession (and ideally opp_line_height_pre) with scripts/merge_columns.py")


if __name__ == "__main__":
    main()
