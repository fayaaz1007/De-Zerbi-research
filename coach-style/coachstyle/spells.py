"""Manager spells, real manager changes, and placebo ("fake") changes.

A spell is a run of consecutive matches with the same manager. Caretakers who
take charge for only a few matches are dropped, so Mourinho -> Holland (1 match)
-> Hiddink counts as one change, Mourinho -> Hiddink.

To judge whether a manager change moved a team's style, we compare it with what
happens at *fake* change points: the same before/after comparison made in the
middle of a spell where the manager did not change. Teams drift during a season
anyway (injuries, form, fixture congestion), so "style moved after the change"
only means something if it moved more than it does at a random point.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

CARETAKER_MAX = 3  # spells this short are treated as caretaker spells and ignored
MIN_MATCHES = 6    # a manager needs this many matches before/after to be compared
WINDOW = 10        # compare at most the last/first 10 matches either side


@dataclass
class Window:
    team: str
    league: str
    pre: list[int]    # row indices of the matches before the (real or fake) change
    post: list[int]   # row indices after
    old: str          # manager before
    new: str          # manager after (== old for a placebo)
    date: str         # first match after the change
    caretaker_matches: int = 0

    @property
    def w(self) -> int:
        return len(self.pre)

    @property
    def label(self) -> str:
        return f"{self.team}: {short(self.old)} → {short(self.new)}"


def short(name: str) -> str:
    """'José Mourinho' -> 'Mourinho'; keeps multi-word surnames reasonable."""
    parts = name.split(" / ")[0].split()
    return parts[-1] if parts else name


def spells(df: pd.DataFrame, caretaker_max: int = CARETAKER_MAX) -> pd.DataFrame:
    """One row per (team, manager) spell after dropping caretakers.

    Columns: team, league, manager, rows (list of df index labels, in date order),
    n, start, end, caretaker_before (caretaker matches dropped just before it).
    """
    out = []
    for team, g in df.sort_values("date").groupby("team", sort=False):
        runs = []  # [manager, [row labels]]
        for idx, mgr in zip(g.index, g["manager"]):
            if runs and runs[-1][0] == mgr:
                runs[-1][1].append(idx)
            else:
                runs.append([mgr, [idx]])
        kept, dropped = [], 0
        for mgr, rows in runs:
            if len(rows) <= caretaker_max and len(runs) > 1:
                dropped += len(rows)
                continue
            if kept and kept[-1]["manager"] == mgr:  # same manager either side of a caretaker
                kept[-1]["rows"] += rows
                continue
            kept.append({"manager": mgr, "rows": list(rows), "caretaker_before": dropped})
            dropped = 0
        for s in kept:
            s.update(team=team, league=g["league"].iloc[0], n=len(s["rows"]),
                     start=df.loc[s["rows"][0], "date"], end=df.loc[s["rows"][-1], "date"])
            out.append(s)
    return pd.DataFrame(out, columns=["team", "league", "manager", "rows", "n", "start", "end",
                                      "caretaker_before"])


def real_changes(sp: pd.DataFrame, min_matches: int = MIN_MATCHES,
                 window: int = WINDOW) -> list[Window]:
    changes = []
    for team, g in sp.groupby("team", sort=False):
        g = g.sort_values("start")
        for (_, a), (_, b) in zip(g.iloc[:-1].iterrows(), g.iloc[1:].iterrows()):
            if a["n"] < min_matches or b["n"] < min_matches:
                continue
            w = min(window, a["n"], b["n"])
            changes.append(Window(team, a["league"], a["rows"][-w:], b["rows"][:w],
                                  a["manager"], b["manager"], str(b["start"])[:10],
                                  b["caretaker_before"]))
    return changes


def placebo_changes(sp: pd.DataFrame, w: int) -> list[Window]:
    """Every fake change point with `w` same-manager matches on both sides."""
    out = []
    for _, s in sp.iterrows():
        rows = s["rows"]
        for k in range(w, len(rows) - w + 1):
            out.append(Window(s["team"], s["league"], rows[k - w:k], rows[k:k + w],
                              s["manager"], s["manager"], ""))
    return out
