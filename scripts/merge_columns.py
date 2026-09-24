"""Fill columns in the match file from another CSV (possession, line height...).

Possession, one row per match (e.g. an FBref "Scores & Fixtures" export):
    python scripts/merge_columns.py data/raw/dezerbi_matches.csv fbref_logs.csv \
        --on date --rename Date=date Poss=possession

Opponent defensive-line proxy, one row per opponent per season:
    python scripts/merge_columns.py data/raw/dezerbi_matches.csv opp_lines.csv \
        --on season opponent --rename Squad=opponent LineHeight=opp_line_height_pre

Team names differ between sites ("Brighton" vs "Brighton & Hove Albion"): check
the "unmatched" count and fix names in the second file if it is not zero.
"""
from __future__ import annotations

import argparse

import pandas as pd


def merge(matches: pd.DataFrame, extra: pd.DataFrame, on: list[str], rename: dict) -> pd.DataFrame:
    extra = extra.rename(columns=rename)
    if "date" in on:
        extra["date"] = pd.to_datetime(extra["date"]).dt.strftime("%Y-%m-%d")
        matches = matches.assign(date=pd.to_datetime(matches["date"]).dt.strftime("%Y-%m-%d"))
    new = [c for c in rename.values() if c not in on]
    extra = extra[on + new].drop_duplicates(on)
    out = matches.drop(columns=[c for c in new if c in matches]).merge(extra, on=on, how="left")
    for c in new:
        print(f"{c}: {out[c].isna().sum()} of {len(out)} rows unmatched")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("matches")
    ap.add_argument("extra")
    ap.add_argument("--on", nargs="+", required=True)
    ap.add_argument("--rename", nargs="+", default=[], help="old=new pairs")
    args = ap.parse_args()
    rename = dict(pair.split("=", 1) for pair in args.rename)
    out = merge(pd.read_csv(args.matches), pd.read_csv(args.extra), args.on, rename)
    out.to_csv(args.matches, index=False)
    print(f"updated {args.matches}")


if __name__ == "__main__":
    main()
