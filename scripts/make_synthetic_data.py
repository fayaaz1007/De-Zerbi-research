"""Generate a SYNTHETIC match dataset with known effects, to test the pipeline.

The numbers are invented. They exist only so the analysis can be run
end-to-end, and so the tests can check the models recover effects that were
planted on purpose. Never draw football conclusions from this file.

    python scripts/make_synthetic_data.py --out data/synthetic_matches.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

SEASONS = [("Brighton", "2022-23"), ("Brighton", "2023-24"),
           ("Marseille", "2024-25"), ("Marseille", "2025-26")]

# Planted "true" effects on xG difference.
TRUE_EFFECTS = {"low_poss": 0.25, "opp_press_z": 0.15, "opp_line_z": 0.10}


def make(n_per_season: int = 38, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for team, season in SEASONS:
        base = rng.normal(0.3, 0.2)  # squad quality differs by season
        start = pd.Timestamp(f"{season[:4]}-08-10")
        for i in range(n_per_season):
            strength = rng.normal(0, 1)                     # opponent quality
            ppda_pre = np.clip(rng.normal(11, 2.5), 6, 20)  # opponent's usual PPDA
            line_pre = np.clip(42 + 0.8 * (11 - ppda_pre) + rng.normal(0, 3), 32, 55)
            home = i % 2 == 0
            poss = np.clip(rng.normal(60 - 3 * strength, 7), 30, 80)
            low = poss < 55
            press_z = -(ppda_pre - 11) / 2.5
            line_z = (line_pre - 42) / 3.5
            xgd = (base + 0.2 * home - 0.35 * strength
                   + TRUE_EFFECTS["low_poss"] * low
                   + TRUE_EFFECTS["opp_press_z"] * press_z
                   + TRUE_EFFECTS["opp_line_z"] * line_z
                   + rng.normal(0, 0.7))
            xg_for = max(0.1, 1.5 + xgd / 2 + rng.normal(0, 0.2))
            xg_against = max(0.1, xg_for - xgd)
            rows.append({
                "date": (start + pd.Timedelta(days=7 * i)).date().isoformat(),
                "season": season, "team": team, "competition": "League",
                "opponent": f"Opponent {i:02d}", "venue": "H" if home else "A",
                "goals_for": rng.poisson(xg_for), "goals_against": rng.poisson(xg_against),
                "xg_for": round(xg_for, 2), "xg_against": round(xg_against, 2),
                "possession": round(poss, 1),
                "opp_ppda": round(ppda_pre * np.exp(rng.normal(0, 0.25)), 2),
                "opp_ppda_pre": round(ppda_pre, 2),
                "opp_line_height_pre": round(line_pre, 1),
                "opp_strength": round(strength, 3),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic_matches.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    make(seed=args.seed).to_csv(args.out, index=False)
    print(f"wrote {args.out}  (SYNTHETIC - for testing only)")
