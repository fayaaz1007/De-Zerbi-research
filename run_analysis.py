"""Run every test of the De Zerbi hypotheses and write a Markdown report.

    python run_analysis.py data/raw/dezerbi_matches.csv --out outputs
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dezerbi import models
from dezerbi.data import load_matches

INK, MUTED, GRID, SURFACE, SERIES = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#2a78d6"


def _style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.tick_params(colors=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRID)


def plot_possession_bins(df, out: Path, threshold: float):
    bins = np.arange(30, 85, 5)
    df = df.assign(bin=pd.cut(df["possession"], bins))
    g = df.groupby("bin", observed=True)["xgd"].agg(["mean", "sem", "count"]).dropna()
    g = g[g["count"] >= 3]
    mids = [b.mid for b in g.index]
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=SURFACE)
    ax.errorbar(mids, g["mean"], yerr=1.96 * g["sem"], fmt="o", color=SERIES,
                ms=8, lw=2, capsize=0)
    for x, y, n in zip(mids, g["mean"], g["count"]):
        ax.annotate(f"n={n}", (x, y), textcoords="offset points", xytext=(8, -3),
                    fontsize=8, color=MUTED)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axvline(threshold, color=MUTED, lw=1, ls="--")
    ax.text(threshold, ax.get_ylim()[1], f" {threshold:g}%", color=MUTED, va="top", fontsize=9)
    _style(ax, "xG difference by possession band (mean ± 95% CI)", "Possession (%)", "xG difference")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_coefficients(table: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(7, 2.8), facecolor=SURFACE)
    y = np.arange(len(table))[::-1]
    ax.errorbar(table["coef"], y, xerr=[table["coef"] - table["ci_low"],
                                        table["ci_high"] - table["coef"]],
                fmt="o", color=SERIES, ms=8, lw=2, capsize=0)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(y, table["hypothesis"])
    _style(ax, "Effect on xG difference per match (95% CI)",
           "Right of 0 supports the hypothesis", "")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_threshold_scan(scan: pd.DataFrame, out: Path, threshold: float):
    fig, ax = plt.subplots(figsize=(7, 3.5), facecolor=SURFACE)
    ax.plot(scan["threshold"], scan["coef"], color=SERIES, lw=2, marker="o", ms=5)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axvline(threshold, color=MUTED, lw=1, ls="--")
    _style(ax, "Effect of being below each possession cut-off", "Cut-off (%)",
           "Effect on xG difference")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def md(df: pd.DataFrame, floatfmt=".3f") -> str:
    return df.to_markdown(index=False, floatfmt=floatfmt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--threshold", type=float, default=55.0)
    ap.add_argument("--n-perm", type=int, default=500)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        df = load_matches(args.csv)
        if args.threshold != 55.0:
            from dezerbi.data import prepare
            df = prepare(pd.read_csv(args.csv), args.threshold)
    notes = [str(w.message) for w in caught]

    main_fit = models.main_model(df, "xgd")
    table = models.hypothesis_table(main_fit)
    gd_table = models.hypothesis_table(models.main_model(df, "gd"))
    inter_fit = models.interaction_model(df, "xgd")
    cont_fit = models.continuous_possession_model(df, "xgd")
    ologit = models.ordered_logit(df)
    ol_table = models.hypothesis_table(ologit)
    splits = models.group_comparison(df, "xgd")
    scan, scan_p = models.threshold_scan(df, "xgd", n_perm=args.n_perm)
    cv = models.leave_one_season_out(df, "xgd")

    plot_possession_bins(df, out / "possession_bins.png", args.threshold)
    plot_coefficients(table, out / "coefficients.png")
    if not scan.empty:
        plot_threshold_scan(scan, out / "threshold_scan.png", args.threshold)

    lines = [
        "# De Zerbi hypothesis report", "",
        f"Data: `{args.csv}`: {len(df)} matches, {df['team_season'].nunique()} team-seasons "
        f"({', '.join(df['team_season'].unique())}).", "",
        f"Pressing measured with `{df.attrs['press_col']}`; defensive line with "
        f"`{df.attrs['line_col'] or 'not available (H3 skipped)'}`.",
        "Pressing is sign-flipped so that a positive effect means \"the harder the opponent "
        "presses (lower PPDA), the better De Zerbi's team does\".", "",
    ]
    if "synthetic" in Path(args.csv).name:
        lines += ["> **SYNTHETIC DATA.** These numbers are invented to test the pipeline. "
                  "They say nothing about De Zerbi.", ""]
    if notes:
        lines += ["**Warnings:**", *[f"- {n}" for n in notes], ""]
    lines += [
        "## 1. Verdicts (main model: xG difference, with controls)", "",
        f"`{models.formula('xgd', models.hypothesis_terms(df) + models.control_terms(df))}`, "
        "HC3 robust SEs. Press and line effects are per 1 standard deviation.", "",
        md(table), "", "![coefficients](coefficients.png)", "",
        "## 2. Robustness", "",
        "### Same model, goal difference", "", md(gd_table), "",
        "### Ordered logit on result (loss < draw < win), log-odds", "", md(ol_table), "",
        "### Possession as a continuous variable (per +10 pp)", "",
        f"coef = {cont_fit.params['poss10']:.3f}, p = {cont_fit.pvalues['poss10']:.3f} "
        "(negative = more possession, worse xGD).", "",
        "### Interaction: does conceding possession pay off more vs. pressing teams?", "",
        f"`low_poss:opp_press_z` coef = {inter_fit.params['low_poss:opp_press_z']:.3f}, "
        f"p = {inter_fit.pvalues['low_poss:opp_press_z']:.3f}.", "",
        "## 3. Simple splits (xG difference)", "", md(splits), "",
        "![possession bins](possession_bins.png)", "",
        "## 4. Is 55% the right cut-off?", "",
    ]
    if scan.empty:
        lines += ["Not enough matches either side of the cut-offs to scan.", ""]
    else:
        best = scan.loc[scan["t"].idxmax()]
        lines += [
            f"Strongest cut-off: **{best['threshold']:g}%** (effect {best['coef']:.3f}, "
            f"t = {best['t']:.2f}). Permutation p-value for the best of "
            f"{len(scan)} cut-offs: **{scan_p:.3f}**. This corrects for having searched "
            "for the best cut-off.", "", "![threshold scan](threshold_scan.png)", "",
        ]
    lines += ["## 5. Out-of-sample (leave one season out)", ""]
    if cv is None:
        lines += ["Needs at least 3 team-seasons.", ""]
    else:
        lines += [
            f"Mean absolute error on xGD: baseline (home + opponent strength) "
            f"**{cv.baseline_mae:.3f}** vs. with tactical variables **{cv.full_mae:.3f}**. "
            "A lower value means the tactical variables predict matches they were not "
            "trained on.", "", md(cv.folds), "",
        ]
    lines += ["## Full regression output", "", "```", str(main_fit.summary()), "```"]
    (out / "report.md").write_text("\n".join(lines))
    print(table[["hypothesis", "coef", "p_one_sided", "verdict"]].to_string(index=False))
    print(f"\nreport: {out / 'report.md'}")


if __name__ == "__main__":
    main()
