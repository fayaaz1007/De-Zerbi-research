"""Run the whole study: data/team_matches.csv -> results/report.md + charts + CSVs.

    python run_analysis.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from coachstyle import analysis, plots
from coachstyle.spells import real_changes, spells

ROOT = Path(__file__).resolve().parent
# team -> metrics to plot: the clearest style change, and the most famous non-change.
CASE_STUDIES = {
    "AS Roma": ["pass_length", "long_ball_share", "passes_per_poss"],
    "Liverpool": ["ppda", "press_high_share", "press_intensity"],
}


def slug(team):
    return team.lower().replace(" ", "_")


def fmt(x, nd=2):
    return f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="?", default=str(ROOT / "data" / "team_matches.csv"))
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = analysis.standardise(analysis.prepare(pd.read_csv(args.data)))
    sp = spells(df)
    changes = real_changes(sp)
    print(f"{df['match_id'].nunique()} matches, {len(sp)} spells, {len(changes)} comparable changes")

    rel = analysis.reliability(df)
    shift = analysis.style_shift(df, sp, changes)
    bnc = analysis.bounce(df, sp, changes)
    fp = analysis.fingerprints(df, sp)
    coords, loadings, evr = analysis.pca_2d(fp)
    cvc = analysis.coach_vs_club(fp)

    plots.style_map(fp, coords, loadings, evr, changes, out / "style_map.png",
                    highlight=set(CASE_STUDIES) | {"Sampdoria", "Real Madrid"})
    plots.shift_vs_placebo(shift["per_change"], out / "shift_vs_placebo.png")
    plots.metric_ratios(shift["by_metric"], out / "metric_shifts.png")
    plots.bounce(bnc, out / "bounce.png")
    for team, metrics in CASE_STUDIES.items():
        plots.team_timeline(df, team, metrics, changes, out / f"timeline_{slug(team)}.png")

    shift["per_change"].to_csv(out / "changes.csv", index=False, float_format="%.3f")
    shift["by_metric"].to_csv(out / "metric_shifts.csv", index=False, float_format="%.3f")
    fp.to_csv(out / "fingerprints.csv", index=False, float_format="%.3f")

    s = shift["summary"]
    b_pts, b_xg = bnc["points_adj"], bnc["xgd_adj"]
    pc = shift["per_change"]
    lines = [
        "# Does a new manager change how a team plays?",
        "",
        f"*{df['match_id'].nunique()} matches · {df['team'].nunique()} teams · "
        f"{len(sp)} manager spells · {s['n_changes']} mid-season changes · StatsBomb open data 2015/16*",
        "",
        "## Headline numbers",
        "",
        f"- **Style (test statistic, noise-normalised):** real changes moved style "
        f"**{fmt(s['norm']['ratio'])}×** as much as placebo change points "
        f"({fmt(s['norm']['mean'])} vs {fmt(s['norm']['placebo_mean'])}; permutation p = {s['norm']['p_value']:.3f}).",
        f"- **Style (raw, in between-team SDs):** {fmt(s['dist']['mean'])} vs {fmt(s['dist']['placebo_mean'])} "
        f"(ratio {fmt(s['dist']['ratio'])}, p = {s['dist']['p_value']:.3f}). Noisy metrics dominate this version; "
        "see the README.",
        f"- {s['share_above_p95']:.0%} of changes moved style more than 95% of placebo change points "
        "(5% would by chance).",
        f"- **Points bounce:** naive {b_pts['naive']:+.2f} ppg "
        f"(95% CI {b_pts['naive_ci'][0]:+.2f} to {b_pts['naive_ci'][1]:+.2f}); beyond regression to the mean "
        f"{b_pts['beyond_rtm']:+.2f} (95% CI {b_pts['beyond_rtm_ci'][0]:+.2f} to {b_pts['beyond_rtm_ci'][1]:+.2f}).",
        f"- **xG bounce:** naive {b_xg['naive']:+.2f} xGD/game "
        f"(95% CI {b_xg['naive_ci'][0]:+.2f} to {b_xg['naive_ci'][1]:+.2f}); beyond regression to the mean "
        f"{b_xg['beyond_rtm']:+.2f} (95% CI {b_xg['beyond_rtm_ci'][0]:+.2f} to {b_xg['beyond_rtm_ci'][1]:+.2f}).",
        "",
        "![style map](style_map.png)",
        "",
        "## 1. Style shift at each change",
        "",
        "![shift vs placebo](shift_vs_placebo.png)",
        "",
        "`shift_norm` = RMS over the 14 metrics of (shift ÷ that metric's placebo SD); "
        "`shift` = RMS shift in between-team SDs; placebo columns and `percentile` refer to `shift_norm`.",
        "",
        pc[["change", "date", "window", "shift_norm", "shift", "placebo_median", "placebo_p95", "percentile"]]
        .to_markdown(index=False, floatfmt=".2f"),
        "",
        "## 2. Which dimensions move",
        "",
        "![metric shifts](metric_shifts.png)",
        "",
        "`ratio` = average |shift| at real changes ÷ at placebo points. `mean_signed_shift` is the "
        "average direction (in between-team SDs); the placebo 95% band for it is "
        "`signed_lo`–`signed_hi`.",
        "",
        shift["by_metric"][["label", "real_abs_shift", "placebo_abs_shift", "ratio", "p_abs", "p_holm",
                            "mean_signed_shift", "signed_lo", "signed_hi"]]
        .to_markdown(index=False, floatfmt=".3f"),
        "",
        "## 3. The new-manager bounce",
        "",
        "![bounce](bounce.png)",
        "",
        b_pts["per_change"].merge(b_xg["per_change"], on=["change", "window"],
                                  suffixes=("_pts", "_xgd"))
        [["change", "pre_pts", "post_pts", "beyond_rtm_pts", "pre_xgd", "post_xgd", "beyond_rtm_xgd"]]
        .to_markdown(index=False, floatfmt="+.2f"),
        "",
        "## 4. Case studies",
        "",
        *[f"![{t}](timeline_{slug(t)}.png)\n" for t in CASE_STUDIES],
        "## 5. Coach or club? Managers seen at two clubs",
        "",
        cvc.to_markdown(index=False, floatfmt=".2f") if len(cvc) else "_none in this sample_",
        "",
        "## Appendix: how reliable is each metric?",
        "",
        "`share_between` = share of match-level variance that is a stable team difference; "
        "`reliability_10` = reliability of a 10-match average.",
        "",
        rel[["label", "share_between", "reliability_10"]].to_markdown(index=False, floatfmt=".2f"),
        "",
    ]
    (out / "report.md").write_text("\n".join(lines))
    print(f"wrote {out / 'report.md'}")


if __name__ == "__main__":
    main()
