"""Turn one match's StatsBomb events into a style profile for each team.

StatsBomb records every location from the acting team's point of view: the
pitch is 120 x 80 yards and that team always attacks towards x = 120. So for
team T, x >= 80 is T's attacking third, and for T's opponent the same x >= 80 is
*their* attacking third (T's defensive third).

Style metrics describe *how* a team plays, not how well:

  with the ball      possession, field_tilt, passes_per_poss, pass_length,
                     long_ball_share, directness, gk_short_share, cross_share
  without the ball   ppda, press_intensity, press_high_share, def_height,
                     high_recoveries
  transition         counter_shots

Outcome columns (xg_for, xg_against, goals, points) are kept separately.
"""
from __future__ import annotations

import numpy as np

from .statsbomb import manager_name

STYLE_METRICS = {
    # column: chart label
    "possession": "Possession % (share of passes)",
    "field_tilt": "Field tilt % (share of final-third passes)",
    "passes_per_poss": "Passes per possession",
    "pass_length": "Average open-play pass length (yd)",
    "long_ball_share": "Long balls % (35+ yd)",
    "directness": "Directness (forward yards / pass yards)",
    "gk_short_share": "Goal kicks played short %",
    "cross_share": "Crosses % of passes into the final third",
    "ppda": "PPDA (opp passes per defensive action; LOW = presses harder)",
    "press_intensity": "Pressures per opponent pass",
    "press_high_share": "Pressures in opponent half %",
    "def_height": "Average height of defensive actions (yd from own goal)",
    "high_recoveries": "Ball wins in the final third per match",
    "counter_shots": "Shots from counter-attacks per match",
}

# Passes that restart play are not "open play" and say little about style
# (apart from goal kicks, which get their own metric).
SET_PIECE_PASSES = {"Throw-in", "Goal Kick", "Free Kick", "Corner", "Kick Off"}
LONG_BALL = 35.0  # yards
# Actions used for PPDA (the usual definition: tackles, interceptions, fouls, challenges).
PPDA_ACTIONS = {"Interception", "Foul Committed", "Dribbled Past"}
# Actions used for the height of the defence (ball-winning actions, not box clearances).
HEIGHT_ACTIONS = PPDA_ACTIONS | {"Ball Recovery"}


def _kind(e: dict) -> str:
    return e["type"]["name"]


def _is_tackle(e: dict) -> bool:
    return _kind(e) == "Duel" and e.get("duel", {}).get("type", {}).get("name") == "Tackle"


def _ratio(num: float, den: float, scale: float = 1.0) -> float:
    return scale * num / den if den else np.nan


def team_metrics(ev: list[dict], team: str) -> dict:
    """Style and xG numbers for `team` from the full event list of one match."""
    mine = [e for e in ev if e["team"]["name"] == team and e.get("period", 1) <= 4]
    theirs = [e for e in ev if e["team"]["name"] != team and e.get("period", 1) <= 4]

    passes = [e for e in mine if _kind(e) == "Pass"]
    opp_passes = [e for e in theirs if _kind(e) == "Pass"]
    open_play = [p for p in passes if p["pass"].get("type", {}).get("name") not in SET_PIECE_PASSES]
    lengths = np.array([p["pass"]["length"] for p in open_play])
    forward = np.array([p["pass"]["end_location"][0] - p["location"][0] for p in open_play])
    into_final_third = [p for p in open_play if p["pass"]["end_location"][0] >= 80]
    goal_kicks = [p for p in passes if p["pass"].get("type", {}).get("name") == "Goal Kick"]
    possessions = {e["possession"] for e in ev if e["possession_team"]["name"] == team}

    def_actions = [e for e in mine if _kind(e) in PPDA_ACTIONS or _is_tackle(e)]
    height_actions = [e for e in mine if (_kind(e) in HEIGHT_ACTIONS or _is_tackle(e))
                      and not e.get("ball_recovery", {}).get("recovery_failure")]
    pressures = [e for e in mine if _kind(e) == "Pressure"]
    shots = [e for e in mine if _kind(e) == "Shot"]
    opp_shots = [e for e in theirs if _kind(e) == "Shot"]
    ft_passes = sum(p["location"][0] >= 80 for p in passes)
    opp_ft_passes = sum(p["location"][0] >= 80 for p in opp_passes)

    return {
        "possession": _ratio(len(passes), len(passes) + len(opp_passes), 100),
        "field_tilt": _ratio(ft_passes, ft_passes + opp_ft_passes, 100),
        "passes_per_poss": _ratio(len(open_play), len(possessions)),
        "pass_length": lengths.mean() if len(lengths) else np.nan,
        "long_ball_share": _ratio((lengths >= LONG_BALL).sum(), len(lengths), 100),
        "directness": _ratio(forward.sum(), lengths.sum()),
        "gk_short_share": _ratio(sum(p["pass"]["length"] < LONG_BALL for p in goal_kicks),
                                 len(goal_kicks), 100),
        "cross_share": _ratio(sum(bool(p["pass"].get("cross")) for p in into_final_third),
                              len(into_final_third), 100),
        # Opponent passes in *their* own 60% of the pitch, per defensive action
        # of ours in *our* attacking 60% (x >= 48 in our frame).
        "ppda": _ratio(sum(p["location"][0] < 72 for p in opp_passes),
                       sum(e["location"][0] >= 48 for e in def_actions)),
        "press_intensity": _ratio(len(pressures), len(opp_passes)),
        "press_high_share": _ratio(sum(e["location"][0] >= 60 for e in pressures), len(pressures), 100),
        "def_height": np.mean([e["location"][0] for e in height_actions]) if height_actions else np.nan,
        "high_recoveries": sum(e["location"][0] >= 80 for e in height_actions
                               if _kind(e) in {"Ball Recovery", "Interception"}),
        "counter_shots": sum(s["play_pattern"]["name"] == "From Counter" for s in shots),
        "shots": len(shots),
        "xg_for": sum(s["shot"].get("statsbomb_xg", 0.0) for s in shots),
        "xg_against": sum(s["shot"].get("statsbomb_xg", 0.0) for s in opp_shots),
        "np_xg_for": sum(s["shot"].get("statsbomb_xg", 0.0) for s in shots
                         if s["shot"]["type"]["name"] != "Penalty"),
        "np_xg_against": sum(s["shot"].get("statsbomb_xg", 0.0) for s in opp_shots
                             if s["shot"]["type"]["name"] != "Penalty"),
    }


def match_rows(match: dict, ev: list[dict], league: str) -> list[dict]:
    """Two rows (home view and away view) for one match."""
    home, away = match["home_team"], match["away_team"]
    sides = [
        (home["home_team_name"], away["away_team_name"], "H", home, away,
         match["home_score"], match["away_score"]),
        (away["away_team_name"], home["home_team_name"], "A", away, home,
         match["away_score"], match["home_score"]),
    ]
    rows = []
    for team, opp, venue, side, opp_side, gf, ga in sides:
        rows.append({
            "match_id": match["match_id"], "date": match["match_date"],
            "league": league, "season": match["season"]["season_name"],
            "matchweek": match.get("match_week"),
            "team": team, "opponent": opp, "venue": venue,
            "manager": manager_name(side), "opp_manager": manager_name(opp_side),
            "goals_for": gf, "goals_against": ga,
            "points": 3 if gf > ga else 1 if gf == ga else 0,
            **team_metrics(ev, team),
        })
    return rows
