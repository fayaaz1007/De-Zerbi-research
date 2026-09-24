"""Write web/style_map_data.js, the data behind the interactive style map.

    python scripts/export_web_data.py

Uses the same pipeline as run_analysis.py, so the web map matches results/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from coachstyle import analysis  # noqa: E402
from coachstyle.features import STYLE_METRICS  # noqa: E402
from coachstyle.spells import real_changes, spells  # noqa: E402

M = analysis.METRICS


def clean(x, nd=3):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


def main():
    df = analysis.standardise(analysis.prepare(pd.read_csv(ROOT / "data" / "team_matches.csv")))
    sp = spells(df)
    changes = real_changes(sp)
    shift = analysis.style_shift(df, sp, changes)
    bnc = analysis.bounce(df, sp, changes)
    fp = analysis.fingerprints(df, sp)
    coords, loadings, evr = analysis.pca_2d(fp)

    long_spells = sp[sp["n"] >= 6].reset_index(drop=True)  # same order as fp
    spell_rows = []
    for i, (s, f) in enumerate(zip(long_spells.itertuples(), fp.itertuples())):
        g = df.loc[s.rows]
        spell_rows.append({
            "id": i, "team": s.team, "league": s.league, "manager": s.manager, "n": int(s.n),
            "start": str(s.start)[:10], "end": str(s.end)[:10],
            "pc1": clean(coords[i, 0]), "pc2": clean(coords[i, 1]),
            "ppg": clean(g["points"].mean(), 2), "xgd": clean(g["xgd"].mean(), 2),
            "z": {m: clean(getattr(f, m)) for m in M},
            "adj": {m: clean(g[f"{m}_adj"].mean()) for m in M},
        })
    key = {(r["team"], r["manager"]): r["id"] for r in spell_rows}

    pc = shift["per_change"].set_index("change")
    pts = bnc["points_adj"]["per_change"].set_index("change")
    xg = bnc["xgd_adj"]["per_change"].set_index("change")
    change_rows = []
    for c in changes:
        r = pc.loc[c.label]
        change_rows.append({
            "label": c.label, "team": c.team, "league": c.league, "old": c.old, "new": c.new,
            "from": key.get((c.team, c.old)), "to": key.get((c.team, c.new)),
            "date": c.date, "window": c.w, "caretaker_matches": c.caretaker_matches,
            "shift_norm": clean(r["shift_norm"]), "percentile": clean(r["percentile"], 1),
            "placebo_p95": clean(r["placebo_p95"]),
            "ppg_pre": clean(pts.loc[c.label, "pre"], 2), "ppg_post": clean(pts.loc[c.label, "post"], 2),
            "ppg_beyond_rtm": clean(pts.loc[c.label, "beyond_rtm"], 2),
            "xgd_pre": clean(xg.loc[c.label, "pre"], 2), "xgd_post": clean(xg.loc[c.label, "post"], 2),
            "xgd_beyond_rtm": clean(xg.loc[c.label, "beyond_rtm"], 2),
        })

    s = shift["summary"]
    data = {
        "metrics": [{"key": m, "label": STYLE_METRICS[m]} for m in M],
        "pca": {"evr": [clean(v) for v in evr],
                "loadings": [{m: clean(v) for m, v in zip(M, row)} for row in loadings]},
        "spells": spell_rows,
        "changes": change_rows,
        "summary": {
            "matches": int(df["match_id"].nunique()), "teams": int(df["team"].nunique()),
            "n_changes": s["n_changes"], "style_ratio": clean(s["norm"]["ratio"], 2),
            "style_p": clean(s["norm"]["p_value"], 3),
            "ppg_naive": clean(bnc["points_adj"]["naive"], 2),
            "ppg_beyond_rtm": clean(bnc["points_adj"]["beyond_rtm"], 2),
            "xgd_beyond_rtm": clean(bnc["xgd_adj"]["beyond_rtm"], 2),
        },
    }
    out = ROOT / "web" / "style_map_data.js"
    out.parent.mkdir(exist_ok=True)
    out.write_text("window.STYLE_DATA = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n")
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB): {len(spell_rows)} spells, {len(change_rows)} changes")


if __name__ == "__main__":
    main()
