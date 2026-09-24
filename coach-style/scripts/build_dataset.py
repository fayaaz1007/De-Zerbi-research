"""Build data/team_matches.csv: one row per team per match, 2015/16 big four leagues.

    python scripts/build_dataset.py            # ~1,500 matches, ~4.5 GB streamed, ~10 min

Each event file is reduced to two rows of features as soon as it arrives, so
nothing large is written to disk. Rerunning skips matches already in the CSV.
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from coachstyle import statsbomb  # noqa: E402
from coachstyle.features import match_rows  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "team_matches.csv"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, help="only this many matches per league (for a quick test)")
    args = ap.parse_args()
    out = Path(args.out)

    done = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=["match_id"])
    have = set(done["match_id"])
    todo = []
    for league, (comp, season) in statsbomb.LEAGUES_2015_16.items():
        fixtures = statsbomb.matches(comp, season, cache_dir=ROOT / "data" / "cache")
        todo += [(m, league) for m in fixtures[: args.limit] if m["match_id"] not in have]
    print(f"{len(have)} matches already built, {len(todo)} to fetch")

    rows, failed = [], []

    def work(m, league):
        return match_rows(m, statsbomb.events(m["match_id"]), league)

    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(work, m, lg): m["match_id"] for m, lg in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                rows += fut.result()
            except Exception as e:  # keep going; rerun to retry
                failed.append((futures[fut], repr(e)))
            if i % 100 == 0:
                print(f"  {i}/{len(todo)}", flush=True)

    df = pd.concat([done, pd.DataFrame(rows)], ignore_index=True) if rows else done
    df = df.sort_values(["league", "date", "match_id", "venue"], ascending=[True, True, True, False])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, float_format="%.4f")
    print(f"wrote {len(df)} rows ({df['match_id'].nunique()} matches) to {out}")
    for mid, err in failed:
        print(f"  failed {mid}: {err}")


if __name__ == "__main__":
    main()
