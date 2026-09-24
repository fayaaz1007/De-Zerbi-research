"""Rebuild Manchester United's financials from SEC EDGAR XBRL (run on your own machine).

    python scripts/fetch_edgar.py --user-agent "Your Name you@example.com"

Writes data/manu_financials.csv, which run_valuation.py then prefers over the bundled
snapshot. EDGAR values win; anything XBRL does not provide (e.g. the club's own adjusted
EBITDA, a non-IFRS measure) is filled from data/manu_financials_snapshot.csv, and the
`source` column says which is which. Also prints the share count from the latest 20-F cover.

Use --dump to list every revenue-like concept and dimension member EDGAR returned, if the
segment split comes back empty (member names can change between filings).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from manu_valuation import edgar  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user-agent", default=os.environ.get("SEC_USER_AGENT", ""),
                    help='"Name email" (or set SEC_USER_AGENT)')
    ap.add_argument("--out", default=str(ROOT / "data" / "manu_financials.csv"))
    ap.add_argument("--snapshot", default=str(ROOT / "data" / "manu_financials_snapshot.csv"))
    ap.add_argument("--filings", type=int, default=8, help="how many 20-Fs to parse for the revenue split")
    ap.add_argument("--dump", action="store_true")
    args = ap.parse_args()

    client = edgar.Edgar(args.user_agent)
    print("companyfacts ...")
    facts = edgar.facts_frame(client.company_facts())
    if args.dump:
        rev = facts[facts["concept"].str.contains("Revenue|Employee|Amorti|Borrow", regex=True)]
        print(rev.groupby(["taxonomy", "concept"]).size().to_string())
    filings = client.annual_filings()
    print(f"{len(filings)} annual filings; parsing the latest {args.filings} for the revenue split ...")
    data = edgar.build(client, max_filings=args.filings)
    latest = client.instance(filings["accessionNumber"].iloc[0])
    inst = edgar.parse_instance(latest)
    if args.dump:
        members = inst[inst["dims"].map(len) > 0]
        print(members.assign(dims=members["dims"].astype(str)).groupby(["concept", "dims"]).size().to_string())
    shares = edgar.shares_outstanding(inst)
    if shares:
        print(f"shares outstanding on the latest 20-F cover: {shares:.2f}m (config.Deal.shares_m)")

    merged = edgar.merge_with_snapshot(data, pd.read_csv(args.snapshot))
    merged.to_csv(args.out, index=False, float_format="%.1f")
    print(merged.to_string(index=False))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
