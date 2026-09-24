"""Charts for the report. One idea per chart; blue = real manager changes, gray = placebo."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .analysis import METRICS  # noqa: E402
from .features import STYLE_METRICS  # noqa: E402
from .spells import short  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e6e5e1"
BLUE = "#2a78d6"
BLUE_LIGHT = "#9ec5f4"
ORANGE = "#eb6834"
PLACEBO = "#c9c8c2"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "figure.dpi": 110,
})


def _finish(fig, path, note=None):
    if note:
        fig.text(0.01, 0.005, note, fontsize=8, color=MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.03 if note else 0, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


SOURCE = "Data: StatsBomb open data, 2015/16 Premier League, La Liga, Serie A, Ligue 1."


def style_map(fp: pd.DataFrame, coords, loadings, evr, changes, path, highlight=()):
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(coords[:, 0], coords[:, 1], s=22, color=PLACEBO, zorder=2, linewidths=0)
    key = {(r.team, r.manager): i for i, r in enumerate(fp.itertuples())}
    for c in changes:
        a, b = key.get((c.team, c.old)), key.get((c.team, c.new))
        if a is None or b is None:
            continue
        hot = c.team in highlight
        ax.annotate("", xy=coords[b], xytext=coords[a], zorder=4 if hot else 3,
                    arrowprops=dict(arrowstyle="-|>", color=BLUE if hot else BLUE_LIGHT,
                                    lw=1.8 if hot else 1.1, shrinkA=3, shrinkB=3))
        if c.team in highlight:
            ax.annotate(f"{c.team}\n{short(c.old)} → {short(c.new)}", coords[b],
                        xytext=(6, 4), textcoords="offset points", fontsize=8, color=INK)
    for team in ("Barcelona", "Leicester City", "Atlético Madrid", "Paris Saint-Germain",
                 "West Bromwich Albion"):
        for i in np.flatnonzero(fp["team"].to_numpy() == team):
            ax.annotate(team, coords[i], xytext=(5, -9), textcoords="offset points",
                        fontsize=8, color=INK_2)

    def axis_name(j):
        top = np.argsort(-np.abs(loadings[j]))[:3]
        return " · ".join(("+" if loadings[j, t] > 0 else "−") + METRICS[t] for t in top)

    ax.set_xlabel(f"PC1 ({evr[0]:.0%} of variation): {axis_name(0)}")
    ax.set_ylabel(f"PC2 ({evr[1]:.0%}): {axis_name(1)}")
    ax.set_title("Style map of 2015/16: every team–manager spell (gray);\n"
                 "arrows show where a new manager moved the team (labelled: case studies)")
    _finish(fig, path, SOURCE + " Spells of 6+ matches, opponent-adjusted.")


def shift_vs_placebo(per_change: pd.DataFrame, path):
    t = per_change.sort_values("shift_norm")
    fig, ax = plt.subplots(figsize=(8.5, 0.28 * len(t) + 1.6))
    y = np.arange(len(t))
    ax.hlines(y, t["placebo_p05"], t["placebo_p95"], color=PLACEBO, lw=5, zorder=2,
              label="Placebo 5–95% range")
    ax.scatter(t["placebo_median"], y, color=MUTED, s=10, zorder=3, label="Placebo median")
    ax.axvline(1, color=GRID, lw=1.2, zorder=1)
    ax.scatter(t["shift_norm"], y, color=BLUE, s=36, zorder=4, edgecolor=SURFACE, linewidth=1.5,
               label="Real change")
    ax.set_yticks(y, [f"{c}  ({w})" for c, w in zip(t["change"], t["window"])], fontsize=8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Style shift relative to normal in-season drift  (≈1 = what happens with no change)")
    ax.set_title("How much did style move after each manager change?")
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.045), ncol=3, frameon=False, fontsize=8)
    _finish(fig, path, SOURCE + " (n) = matches compared on each side.")


def metric_ratios(by_metric: pd.DataFrame, path):
    t = by_metric.sort_values("ratio")
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    y = np.arange(len(t))
    ax.axvline(1, color=INK_2, lw=1)
    ax.hlines(y, 1, t["ratio"], color=GRID, lw=2, zorder=2)
    ax.scatter(t["ratio"], y, color=BLUE, s=40, zorder=3)
    labels = [f"{STYLE_METRICS[m].split(' (')[0]}{'  *' if p < 0.05 else ''}"
              for m, p in zip(t["metric"], t["p_holm"])]
    ax.set_yticks(y, labels, fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Average shift at real changes ÷ at placebo points  (1 = no more than chance)")
    ax.set_title("Which parts of a team's style does a new manager change?")
    _finish(fig, path, SOURCE + " * = larger than placebo after Holm correction for 14 tests, p < 0.05.")


def bounce(b: dict, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    titles = {"points_adj": "Points per game", "xgd_adj": "Non-penalty xG difference per game"}
    for ax, (key, r) in zip(axes, b.items()):
        pl = r["placebo_points"]
        ax.scatter(pl[:, 0], pl[:, 1], s=6, color=PLACEBO, linewidths=0, zorder=2,
                   label=f"Placebo: {r['window']} matches before/after a fake change")
        t = r["per_change"]
        ax.scatter(t["pre"], t["post"], s=36, color=BLUE, edgecolor=SURFACE, linewidth=1.5,
                   zorder=4, label="Real change")
        lo, hi = np.nanmin(pl), np.nanmax(pl)
        xs = np.array([lo, hi])
        ax.plot(xs, xs, color=GRID, lw=1.2, zorder=1)
        ax.plot(xs, np.polyval(r["rtm_line"], xs), color=INK_2, lw=2, zorder=3,
                label="Expected next (regression to the mean)")
        ax.set_xlabel(f"Before the change (last {r['window']} matches)")
        ax.set_ylabel("After the change")
        ax.set_title(titles[key])
        ax.annotate(f"naive bounce {r['naive']:+.2f}\nbeyond regression to mean {r['beyond_rtm']:+.2f}",
                    (0.03, 0.97), xycoords="axes fraction", va="top", fontsize=9, color=INK)
        ax.legend(loc="lower right", frameon=False, fontsize=8)
    fig.suptitle("The new-manager bounce: real, or just a bad run ending?", x=0.01, ha="left",
                 fontweight="bold", fontsize=12)
    _finish(fig, path, SOURCE + " Opponent- and venue-adjusted. Gray line: after = before.")


def team_timeline(df: pd.DataFrame, team: str, metrics: list[str], changes, path, roll=5):
    g = df[df["team"] == team].sort_values("date")
    fig, axes = plt.subplots(len(metrics), 1, figsize=(9, 2.4 * len(metrics) + 0.6), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, metrics):
        ax.scatter(g["date"], g[f"{m}_adj"], s=10, color=PLACEBO, zorder=2)
        # Rolling average within each manager's spell, so it never mixes two managers.
        for _, s in g.groupby((g["manager"] != g["manager"].shift()).cumsum()):
            ax.plot(s["date"], s[f"{m}_adj"].rolling(roll, min_periods=3).mean(),
                    color=BLUE, lw=2, zorder=3)
        ax.set_ylabel(STYLE_METRICS[m].split(" (")[0], fontsize=9)
        for c in changes:
            if c.team == team:
                ax.axvline(pd.Timestamp(c.date), color=INK_2, lw=1)
    for c in changes:
        if c.team == team:
            axes[0].annotate(f" {short(c.new)} arrives", (pd.Timestamp(c.date), 1), fontsize=9,
                             xycoords=("data", "axes fraction"), va="top", color=INK)
    axes[0].set_title(f"{team} 2015/16: match values (dots) and {roll}-match average per manager")
    _finish(fig, path, SOURCE + " Opponent- and venue-adjusted.")
