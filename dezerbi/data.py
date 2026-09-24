"""Loading, validation and feature engineering for De Zerbi match data.

One row = one match played by a De Zerbi team, seen from De Zerbi's side.
See README.md ("Data schema") for the column definitions.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

REQUIRED = [
    "date", "season", "opponent", "venue",
    "goals_for", "goals_against", "xg_for", "xg_against",
    "possession",
]

# Opponent-style columns. "_pre" columns describe how the opponent *usually*
# plays (measured before / outside this match) and are preferred, because the
# in-match numbers are partly caused by what De Zerbi's team did in that game.
PRESS_COLS = ["opp_ppda_pre", "opp_ppda"]
LINE_COLS = ["opp_line_height_pre", "opp_line_height"]

POSSESSION_THRESHOLD = 55.0


def load_matches(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return prepare(df)


def prepare(df: pd.DataFrame, possession_threshold: float = POSSESSION_THRESHOLD) -> pd.DataFrame:
    df = df.copy()
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    if not any(c in df.columns for c in PRESS_COLS):
        raise ValueError(f"need at least one pressing column: {PRESS_COLS}")

    df["date"] = pd.to_datetime(df["date"])
    df["season"] = df["season"].astype(str)
    df["venue"] = df["venue"].str.strip().str.upper().str[0]
    if not df["venue"].isin(["H", "A", "N"]).all():
        raise ValueError("venue must be H, A or N")
    if "team" not in df.columns:
        df["team"] = "De Zerbi"
    df["team_season"] = df["team"].astype(str) + " " + df["season"]

    # Possession given as a fraction (0.55) instead of a percentage (55).
    if df["possession"].max() <= 1.0:
        df["possession"] = df["possession"] * 100
    if not df["possession"].between(0, 100).all():
        raise ValueError("possession must be in [0, 100]")

    # --- outcomes -------------------------------------------------------
    df["gd"] = df["goals_for"] - df["goals_against"]
    df["xgd"] = df["xg_for"] - df["xg_against"]
    df["result"] = np.sign(df["gd"]).astype(int)  # -1 loss, 0 draw, 1 win
    df["points"] = df["result"].map({1: 3, 0: 1, -1: 0})
    df["win"] = (df["result"] == 1).astype(int)

    # --- hypothesis features ----------------------------------------------
    df["home"] = (df["venue"] == "H").astype(int)
    df["low_poss"] = (df["possession"] < possession_threshold).astype(int)

    press_col = next(c for c in PRESS_COLS if c in df.columns and df[c].notna().any())
    if press_col == "opp_ppda":
        warnings.warn(
            "Using in-match opponent PPDA. It is partly an outcome of the match "
            "(a dominant De Zerbi side inflates it). Prefer opp_ppda_pre.",
            stacklevel=2,
        )
    df["opp_ppda_used"] = df[press_col]
    # PPDA is inverted: LOW PPDA = the opponent presses HARD. Flip the sign so
    # that a higher value always means "more pressing", then standardise.
    df["opp_press_z"] = _z(-df["opp_ppda_used"])
    df["high_press"] = (df["opp_ppda_used"] < df["opp_ppda_used"].median()).astype(int)

    line_col = next((c for c in LINE_COLS if c in df.columns and df[c].notna().any()), None)
    if line_col is not None:
        df["opp_line_used"] = df[line_col]
        df["opp_line_z"] = _z(df["opp_line_used"])
        df["high_line"] = (df["opp_line_used"] > df["opp_line_used"].median()).astype(int)

    if "opp_strength" in df.columns and df["opp_strength"].notna().any():
        df["opp_strength_z"] = _z(df["opp_strength"])

    df.attrs["press_col"] = press_col
    df.attrs["line_col"] = line_col
    df.attrs["possession_threshold"] = possession_threshold
    return df.sort_values("date").reset_index(drop=True)


def _z(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd > 0 else s * 0.0
