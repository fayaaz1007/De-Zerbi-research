"""Monte Carlo valuation of Manchester United, driven by on-pitch results.

    python run_valuation.py                       # as of the INEOS deal (20 Feb 2024)
    python run_valuation.py --paths 2000 --quick  # fast smoke run

Writes outputs/valuation/report.md plus charts. See manu_valuation/README.md.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from manu_valuation import edgar, engine, results, valuation
from manu_valuation.config import Config, apply, scenarios

# Palette: validated categorical slots 1-3 (all-pairs) and a 4-step blue ordinal ramp.
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
RAMP = ["#0d366b", "#1c5cab", "#3987e5", "#86b6ef"]  # dark = best outcome

ROOT = Path(__file__).resolve().parent
RESULTS_CSV = ROOT / "data" / "epl_matches.csv"
EDGAR_CSV = ROOT / "data" / "manu_financials.csv"
SNAPSHOT_CSV = ROOT / "data" / "manu_financials_snapshot.csv"
LEVELS = [0, 50, 100, 150, 200, 250, 300]


def _style(ax, title, xlabel="", ylabel="", subtitle=None):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=11.5, pad=22 if subtitle else 8)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=MUTED, fontsize=9, va="bottom")
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRID)


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- charts


def plot_finishes(res: valuation.Result, out: Path):
    f = res.paths.finish
    bands = [("Top 4", f <= 4), ("5th-7th", (f >= 5) & (f <= 7)), ("8th-17th", (f >= 8) & (f <= 17)),
             ("18th-20th / Championship", f >= 18)]
    labels = [results.season_label(s) for s in res.paths.seasons]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4), facecolor=SURFACE)
    bottom = np.zeros(len(x))
    for (name, m), color in zip(bands, RAMP):
        share = m.mean(axis=0) * 100
        ax.bar(x, share, bottom=bottom, color=color, width=0.72, label=name, edgecolor=SURFACE, linewidth=2)
        bottom += share
    top4 = bands[0][1].mean(axis=0) * 100
    for xi, v in zip(x, top4):
        ax.text(xi, v / 2, f"{v:.0f}%", ha="center", va="center", color="white", fontsize=8)
    ax.set_xticks(x, labels, rotation=0, fontsize=8)
    ax.set_ylim(0, 100)
    _style(ax, "United's simulated league finish, by season", "", "% of paths",
           subtitle=f"{len(f):,} paths as of {res.cfg.as_of}; labels = chance of a top-4 finish")
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.08), labelcolor=MUTED)
    _save(fig, out)


def plot_ev(runs: dict, price: float, out: Path):
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
    lo = min(r.dcf.ev.min() for r in runs.values()) / 1000
    bins = np.linspace(lo, max(r.dcf.ev.max() for r in runs.values()) / 1000, 70)
    peak = 0
    # direct labels: centred, then right- and left-aligned so neighbours do not collide
    placement = [(0, "center"), (-4, "right"), (4, "left")]
    for (name, r), color, (dx, ha) in zip(runs.items(), SERIES, placement):
        ev = r.dcf.ev / 1000
        h, e, _ = ax.hist(ev, bins=bins, histtype="step", color=color, linewidth=2, density=True)
        i = int(np.argmax(h))
        peak = max(peak, h.max())
        ax.annotate(f"{name}\nmedian {gbp(np.median(r.dcf.ev))}", ((e[i] + e[i + 1]) / 2, h[i]),
                    xytext=(dx, 6), textcoords="offset points", ha=ha, va="bottom", fontsize=8, color=INK)
    p = price / 1000
    ax.axvline(p, color=INK, linewidth=1.5)
    ax.text(p, peak * 1.25, f" INEOS entry\n £{p:.1f}bn EV\n ($33/share)", color=INK, fontsize=9, va="top")
    ax.set_ylim(0, peak * 1.35)
    ax.set_xlim(lo - 0.2, p + 1.0)
    ax.set_yticks([])
    best = max(np.median(r.dcf.ev) for r in runs.values()) / 1000
    ax.annotate("", (p, peak * 0.55), (best, peak * 0.55),
                arrowprops=dict(arrowstyle="<->", color=MUTED, linewidth=1))
    ax.text((p + best) / 2, peak * 0.58, f"£{p - best:.1f}bn not explained by\ncash flows in any scenario",
            ha="center", va="bottom", fontsize=8.5, color=MUTED)
    _style(ax, "Intrinsic enterprise value: 10,000 simulated futures vs the INEOS price",
           "Enterprise value (£bn)", "", subtitle="DCF of free cash flow to the firm on each sporting path "
           f"(WACC {next(iter(runs.values())).dcf.wacc:.1%})")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    _save(fig, out)


def plot_bet(sweeps: dict, base: valuation.Result, out: Path):
    fig, ax = plt.subplots(figsize=(8, 4.4), facecolor=SURFACE)
    for (name, sw), color in zip(sweeps.items(), SERIES):
        ax.plot(sw["p_ucl"] * 100, sw["irr_p50"] * 100, color=color, linewidth=2, marker="o", markersize=5)
        ax.annotate(name, (sw["p_ucl"].iloc[-1] * 100, sw["irr_p50"].iloc[-1] * 100), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8.5, color=INK)
    w = base.dcf.wacc * 100
    ax.axhline(w, color=INK, linewidth=1, linestyle="--")
    ax.text(2, w + 0.15, f"Cost of capital {w:.1f}%", fontsize=8.5, color=INK, va="bottom")
    s = valuation.summary(base)
    ax.plot(s["p_ucl"] * 100, s["irr_p50"] * 100, marker="o", markersize=10, markerfacecolor="none",
            markeredgecolor=INK, markeredgewidth=1.5)
    ax.annotate("model's view of United\nat the deal (status quo)", (s["p_ucl"] * 100, s["irr_p50"] * 100),
                xytext=(14, -30), textcoords="offset points", fontsize=8, color=MUTED, ha="left",
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8))
    ax.set_xlim(0, 115)
    _style(ax, "What did Ratcliffe implicitly bet on?", "Share of future seasons in the Champions League (%)",
           "Median IRR (%)",
           subtitle=f"IRR from paying £{base.price / 1000:.1f}bn and selling in FY{base.paths.seasons[-1] + 1} "
           f"at the same {base.exit_multiple:.1f}x revenue multiple")
    ax.set_xticks(range(0, 101, 20))
    _save(fig, out)


def plot_required_multiple(res: valuation.Result, out: Path):
    n = res.ucl_seasons()
    req = res.required_multiple
    ks = [k for k in range(n.max() + 1) if (n == k).sum() >= 30]
    med = [np.median(req[n == k]) for k in ks]
    lo = [np.percentile(req[n == k], 10) for k in ks]
    hi = [np.percentile(req[n == k], 90) for k in ks]
    fig, ax = plt.subplots(figsize=(8, 3.8), facecolor=SURFACE)
    ax.vlines(ks, lo, hi, color=SERIES[0], linewidth=2, alpha=0.45)
    ax.plot(ks, med, "o", color=SERIES[0], markersize=8)
    ax.axhline(res.entry_multiple, color=INK, linewidth=1, linestyle="--")
    ax.text(ks[0], res.entry_multiple + 0.2, f"entry multiple paid: {res.entry_multiple:.1f}x", ha="left",
            va="bottom", fontsize=8.5, color=INK)
    ax.set_ylim(0, max(hi) * 1.05)
    for k, m in zip(ks[::3], med[::3]):
        ax.annotate(f"{m:.1f}x", (k, m), xytext=(8, 0), textcoords="offset points", fontsize=8, color=INK,
                    va="center")
    _style(ax, "Exit multiple needed to earn the cost of capital", "Champions League seasons on the path "
           f"(out of {res.paths.seasons.size - 1})", "EV / revenue at exit",
           subtitle="Median and 10th-90th percentile across paths (status quo costs)")
    ax.set_xticks(ks)
    _save(fig, out)


def plot_ucl_economics(econ: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(8, 3.6), facecolor=SURFACE)
    y = np.arange(len(econ))[::-1]
    h = 0.26
    for j, (col, color) in enumerate(zip(["revenue", "ebitda", "fcff"], SERIES)):
        vals = econ[col].to_numpy()
        ax.barh(y + (1 - j) * h, vals, height=h - 0.03, color=color, label=col.upper() if col != "revenue" else "Revenue")
        for yi, v in zip(y + (1 - j) * h, vals):
            ax.text(v + (5 if v >= 0 else -5), yi, f"{v:,.0f}", va="center", ha="left" if v >= 0 else "right",
                    fontsize=7.5, color=MUTED)
    ax.set_yticks(y, econ.index)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    _style(ax, "One season's money, by European competition", "£m (nominal, one fiscal year)", "",
           subtitle=f"Mean across paths, FY{econ.attrs['fy']}")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=MUTED)
    _save(fig, out)


def plot_calibration(bt: pd.DataFrame, out: Path):
    bins = [0, .05, .2, .4, .6, .8, .95, 1]
    g = bt.assign(bin=pd.cut(bt["p_top4"], bins, include_lowest=True)).groupby("bin", observed=True)
    t = g.agg(pred=("p_top4", "mean"), obs=("actual", lambda a: (a <= 4).mean()), n=("p_top4", "size"))
    fig, ax = plt.subplots(figsize=(4.8, 4.2), facecolor=SURFACE)
    ax.plot([0, 1], [0, 1], color=GRID, linewidth=1.5)
    ax.plot(t["pred"], t["obs"], "o-", color=SERIES[0], linewidth=2, markersize=8)
    for p, o, n in zip(t["pred"], t["obs"], t["n"]):
        ax.annotate(f"n={n}", (p, o), xytext=(7, -10), textcoords="offset points", fontsize=7.5, color=MUTED)
    _style(ax, "Pre-season top-4 forecasts, 2012-2025", "Forecast probability", "Observed frequency",
           subtitle="Out of sample, every club; diagonal = perfect calibration")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    _save(fig, out)


# ---------------------------------------------------------------- tables


def ucl_economics(res: valuation.Result, y: int = 3) -> pd.DataFrame:
    comp = res.paths.comp[:, y]
    names = {1: "Champions League", 2: "Europa League", 3: "Conference League", 0: "No Europe"}
    rows = {}
    for c, name in names.items():
        m = comp == c
        if m.sum() >= 50:
            rows[name] = {k: res.proj[k][m, y].mean() for k in ("revenue", "ebitda", "fcff", "wages", "uefa")}
    df = pd.DataFrame(rows).T
    df.attrs["fy"] = int(res.paths.seasons[y] + 1)
    return df


def revenue_check(res: valuation.Result, fin: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fy, actual in fin.set_index("fy")["revenue"].items():
        y = fy - 1 - res.paths.seasons[0]
        if fy <= res.base_fy or y < 0 or y >= res.paths.seasons.size:
            continue
        sim = res.proj["revenue"][:, y]
        rows.append(dict(fy=f"FY{fy}", actual=actual, model_p5=np.percentile(sim, 5), model_p50=np.median(sim),
                         model_p95=np.percentile(sim, 95), actual_percentile=(sim < actual).mean()))
    return pd.DataFrame(rows)


def md(df: pd.DataFrame, floatfmt=".2f") -> str:
    return df.to_markdown(index=False, floatfmt=floatfmt)


def pct(x):
    return f"{x:.0%}"


def gbp(x):
    sign = "−" if x < 0 else ""
    x = abs(x)
    return f"{sign}£{x / 1000:.1f}bn" if x >= 1000 else f"{sign}£{x:,.0f}m"


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(RESULTS_CSV))
    ap.add_argument("--financials", default=None, help="default: data/manu_financials.csv if present, else the snapshot")
    ap.add_argument("--out", default="outputs/valuation")
    ap.add_argument("--paths", type=int, default=10_000)
    ap.add_argument("--sweep-paths", type=int, default=2_000)
    ap.add_argument("--as-of", default=None, help="valuation date (default: INEOS completion, 2024-02-20)")
    ap.add_argument("--refresh-results", action="store_true", help="re-download results from openfootball")
    ap.add_argument("--quick", action="store_true", help="skip the backtest and the since-the-deal re-run")
    args = ap.parse_args()
    t_start = time.time()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.refresh_results or not Path(args.results).exists():
        results.fetch().to_csv(args.results, index=False, date_format="%Y-%m-%d")
    matches = results.load(args.results)
    fin_path = Path(args.financials) if args.financials else (EDGAR_CSV if EDGAR_CSV.exists() else SNAPSHOT_CSV)
    fin = edgar.load_financials(fin_path)
    snapshot = fin_path.resolve() == SNAPSHOT_CSV.resolve()

    cfg = Config(n_paths=args.paths)
    if args.as_of:
        cfg.as_of = args.as_of
    print(f"fitting sporting model as of {cfg.as_of} ...")
    model = engine.fit(matches, cfg.as_of)

    runs = {}
    for sc in scenarios():
        print(f"simulating: {sc.name}")
        runs[sc.name] = valuation.run(apply(cfg, sc.changes), matches, fin, model)
    base = runs["Status quo"]
    print("sweeping United's level ...")
    sweeps = {sc.name: valuation.sweep_level(cfg, matches, fin, model, LEVELS, args.sweep_paths, sc.changes)
              for sc in scenarios()}
    econ = ucl_economics(base)
    reality = valuation.reality_check(base, matches)
    rev_check = revenue_check(base, fin)

    bt = brier = None
    now = None
    if not args.quick:
        print("backtesting pre-season forecasts ...")
        completed = matches[matches["season"] < engine.season_of(matches.dropna(subset=["hg"])["date"].max())]
        bt = valuation.backtest(completed, model.link)
        brier = valuation.brier(bt)
        plot_calibration(bt, out / "backtest_calibration.png")
        latest = matches.dropna(subset=["hg"])["date"].max() + pd.Timedelta(days=1)
        if latest > pd.Timestamp(cfg.as_of) + pd.Timedelta(days=365):
            print(f"re-running as of {latest.date()} ...")
            cfg_now = Config(n_paths=args.paths, as_of=str(latest.date()))
            model_now = engine.fit(matches, cfg_now.as_of)
            now = {sc.name: valuation.run(apply(cfg_now, sc.changes), matches, fin, model_now)
                   for sc in scenarios()[:2]}

    plot_finishes(base, out / "finish_distribution.png")
    plot_ev(runs, base.price, out / "ev_distribution.png")
    plot_bet(sweeps, base, out / "the_bet.png")
    plot_required_multiple(base, out / "required_multiple.png")
    plot_ucl_economics(econ, out / "ucl_economics.png")

    report = build_report(args, cfg, model, runs, sweeps, econ, reality, rev_check, bt, brier, now,
                          fin_path, snapshot)
    (out / "report.md").write_text(report)
    s = valuation.summary(base)
    print(f"\nmedian intrinsic EV (status quo) {gbp(s['ev_p50'])}; INEOS EV {gbp(base.price)} "
          f"-> percentile {s['deal_percentile']:.1%}; median IRR at entry multiple {s['irr_p50']:.1%}")
    print(f"report: {out / 'report.md'}  ({time.time() - t_start:.0f}s)")


def build_report(args, cfg, model, runs, sweeps, econ, reality, rev_check, bt, brier, now, fin_path, snapshot):
    base = runs["Status quo"]
    d = model.dynamics
    s = {k: valuation.summary(r) for k, r in runs.items()}
    sb = s["Status quo"]
    deal = cfg.deal
    L = ["# Manchester United: Monte Carlo valuation driven by on-pitch results", ""]
    L += [f"Valuation date **{cfg.as_of}** (INEOS completion). {base.paths.finish.shape[0]:,} paths × "
          f"{cfg.horizon} seasons ({results.season_label(base.paths.seasons[0])} to "
          f"{results.season_label(base.paths.seasons[-1])}). Base financial year: **FY{base.base_fy}**, "
          f"the last annual report published before the valuation date.", ""]
    if snapshot:
        L += ["> **Financials come from the bundled snapshot** (`data/manu_financials_snapshot.csv`, rounded "
              "figures from the annual reports). Run `python scripts/fetch_edgar.py` to rebuild them from SEC "
              "EDGAR XBRL and re-run.", ""]
    L += ["## The answer", "",
          f"* **Price.** ${deal.price_usd:.0f}/share × {deal.shares_m:.1f}m shares at "
          f"${deal.usd_per_gbp:.2f}/£ = {gbp(deal.equity)} equity; plus net debt {gbp(deal.net_debt)} and "
          f"net transfer payables {gbp(deal.transfer_payables)} = **{gbp(base.price)} enterprise value**, "
          f"or **{base.entry_multiple:.1f}x** trailing revenue (FY{base.entry_fy}).",
          f"* **Cash flows alone do not get there.** Status quo median intrinsic EV {gbp(sb['ev_p50'])} "
          f"(5th-95th percentile {gbp(sb['ev_p5'])} to {gbp(sb['ev_p95'])}). The INEOS price sits "
          f"**{percentile_text(base)}**, and above every path in all three scenarios "
          f"(best case: {gbp(s['Plan + growth']['ev_p95'])} at the 95th percentile of *Plan + growth*). The gap to the "
          "price is what the buyer pays for something other than the club's own cash flows.",
          "* **Why:** United's cash conversion. Adjusted EBITDA is about a fifth of revenue and the squad costs "
          "almost as much again in net transfer spend, so free cash flow is close to zero before any "
          "on-pitch upside.",
          f"* **The implied bet** is therefore on the *exit*, not on the dividends. Buying at "
          f"{base.entry_multiple:.1f}x and selling in FY{base.paths.seasons[-1] + 1} at the same multiple returns a "
          f"median **{sb['irr_p50']:.1%} IRR** (status quo), against an {base.dcf.wacc:.1%} cost of capital. "
          f"Earning the cost of capital needs an exit at about **{sb['req_multiple_p50']:.1f}x** revenue: the "
          "bet is that football's scarcity premium keeps expanding.",
          f"* **On the pitch, the price needs the Champions League habit back.** With costs as they were, even a "
          f"United that plays Champions League football {sweeps['Status quo']['p_ucl'].iloc[-1]:.0%} of the time "
          f"only lifts the IRR to {sweeps['Status quo']['irr_p50'].iloc[-1]:.1%}. Only the cost plan *plus* "
          f"faster growth *plus* regular Champions League football gets close "
          f"({sweeps['Plan + growth']['irr_p50'].iloc[-1]:.1%}). No combination tested clears "
          f"{base.dcf.wacc:.1%} without a rising multiple.", "",
          "![the bet](the_bet.png)", "", "![EV distribution](ev_distribution.png)", ""]

    tbl = pd.DataFrame([dict(scenario=k, **{c: v for c, v in x.items()}) for k, x in s.items()])
    tbl = tbl.assign(**{c: tbl[c].map(gbp) for c in ["ev_p5", "ev_p50", "ev_p95", "ev_mean"]})
    for c in ["deal_percentile", "p_top4", "p_ucl", "p_irr_above_wacc"]:
        tbl[c] = tbl[c].map(pct)
    tbl["irr_p50"] = tbl["irr_p50"].map(lambda v: f"{v:.1%}")
    tbl["req_multiple_p50"] = tbl["req_multiple_p50"].map(lambda v: f"{v:.1f}x")
    notes = {sc.name: sc.note for sc in scenarios()}
    L += ["## Scenarios", "", "Same 10,000 sporting paths each time; only the business assumptions change.", "",
          *[f"* **{k}:** {v}" for k, v in notes.items()], "",
          md(tbl.rename(columns={"ev_p5": "EV p5", "ev_p50": "EV median", "ev_p95": "EV p95", "ev_mean": "EV mean",
                                 "deal_percentile": "INEOS price percentile", "p_top4": "P(top 4)",
                                 "p_ucl": "P(UCL season)", "irr_p50": "IRR at entry multiple (median)",
                                 "p_irr_above_wacc": "P(IRR ≥ WACC)", "req_multiple_p50": "exit multiple for WACC"})),
          ""]

    L += ["## 1. Sporting model", "",
          f"**Elo** on every Premier League match since 2010-11 (openfootball). K = {model.elo_run.params.k:g} and "
          f"season carry-over = {model.elo_run.params.carryover:g} were picked by log loss "
          f"({model.elo_grid['log_loss'].min():.4f} vs {base_rate_ll(model):.4f} for base rates). An ordered logit "
          f"on the rating gap turns ratings into win/draw/loss probabilities "
          f"(P(home win) at equal ratings {model.link.probs(0)[0]:.1%}, draw {model.link.probs(0)[1]:.1%}).", "",
          "**Season strength and its dynamics.** Each team-season gets a strength (Elo points vs the league "
          "average) fitted to that season's results. A Kalman-filtered state-space model (club *level* that "
          "drifts + *form* that fades) is fitted to those by maximum likelihood across all clubs:", "",
          f"| parameter | estimate |\n|---|---|\n| level drift per season (sd) | {d.sigma_level:.1f} |\n"
          f"| form shock (sd) | {d.sigma_form:.1f} |\n| form persistence ρ | {d.rho:.2f} (≈ 0: last season's luck does not carry over) |\n"
          f"| newly promoted club, mean level | {d.promoted_mean:.0f} |\n"
          f"| United's filtered level after the last completed season | {d.level(results.MUFC):.0f} |", "",
          "Most of the season-to-season movement is permanent (level), not noise that reverts. This matters: a "
          "bad United season is partly news about the future, not just bad luck.", "",
          "**Each simulated season** plays all 380 fixtures; relegated clubs are replaced by promoted ones "
          "(United can go down and come back). Qualification follows the English rules (top 4 to the UCL, "
          "a 5th place half the time through UEFA's performance spot, FA Cup winner to the UEL). European runs "
          "are simulated round by round, tilted by United's strength that season.", "",
          "![finishes](finish_distribution.png)", ""]
    if brier is not None:
        L += ["**Does it forecast?** Every season from 2012-13 to 2025-26 was forecast before a ball was kicked, "
              "with the dynamics fitted only on earlier seasons:", "", md(brier, ".3f"), "",
              "Skill = 1 − Brier / Brier of always predicting the base rate.", "",
              "![calibration](backtest_calibration.png)", ""]
    L += ["## 2. From results to revenue", "",
          "The base year is reproduced exactly: each revenue line is split into a part driven by that year's "
          "results (league merit payments, UEFA money, European gates, sponsor clauses) and a calibrated core.", "",
          md(base.cal.table, ".1f"), "",
          "Sporting links in the projection:", "",
          "* **League finish → PL merit payment** (~£2.8m a place) and relegation (parachute payment, "
          "commercial and matchday hits, relegation wage cuts).",
          "* **Previous finish → European competition → UEFA prize money** (starting fee, results, value pillar, "
          "knockout bonuses) **and extra home gates**.",
          f"* **Kit deal clause:** -{cfg.revenue.kit_penalty:.0%} of the £{cfg.revenue.kit_deal:.0f}m adidas "
          "fee after two consecutive seasons without the Champions League.",
          f"* **Wages hedge part of it:** senior contracts lose ~{cfg.costs.ucl_pay_cut:.0%} without the "
          f"Champions League (assumed to cover {cfg.costs.clause_share:.0%} of the bill), and "
          f"{cfg.costs.europe_bonus_share:.0%} of UEFA money goes out as bonuses.", "",
          md(econ.reset_index().rename(columns={"index": "competition"}), ",.0f"), "",
          "![UCL economics](ucl_economics.png)", "",
          f"Champions League vs no Europe: +{gbp(econ.loc['Champions League', 'revenue'] - econ.loc['No Europe', 'revenue'])} "
          f"revenue but only +{gbp(econ.loc['Champions League', 'ebitda'] - econ.loc['No Europe', 'ebitda'])} "
          "EBITDA, because wage clauses give a large part of the upside back to the players."
          if {"Champions League", "No Europe"} <= set(econ.index) else "", ""]
    L += ["## 3. Cash flow and DCF", "",
          "FCFF = adjusted EBITDA − tax (25%, losses carried forward) − net player-registration spend − PP&E capex. "
          f"Net player spend = {base.cal.player_capex_ratio:.1%} of revenue in the status quo (base-year amortisation "
          f"less typical player-sale profits). WACC {base.dcf.wacc:.2%} "
          f"(Rf {cfg.valuation.risk_free:.1%}, ERP {cfg.valuation.erp:.1%}, β {cfg.valuation.beta:.1f}, "
          f"Kd {cfg.valuation.cost_of_debt:.0%} pre-tax, D/V {cfg.valuation.debt_weight:.0%}); terminal growth "
          f"{cfg.valuation.terminal_growth:.1%} on the average FCFF of the last {cfg.valuation.terminal_window} "
          "years, so one lucky final season does not set the perpetuity.", "",
          "Mean projection, status quo (£m):", "",
          md(projection_table(base), ",.0f"), "",
          "![required multiple](required_multiple.png)", "",
          "## 4. Sensitivity: how good do United need to be?", "",
          "United's starting level is forced to each value and the whole simulation re-run "
          f"({args.sweep_paths:,} paths each). *Level* is Elo points above the league average; the "
          "model's own estimate at the deal date, after 25 games of 2023-24, was "
          f"{float(np.mean(base.paths.level)):.0f}.", ""]
    for name, sw in sweeps.items():
        t = sw[["level", "p_top4", "p_ucl", "ev_p50", "irr_p50", "req_multiple_p50"]].copy()
        t["ev_p50"] = t["ev_p50"].map(gbp)
        for c in ["p_top4", "p_ucl"]:
            t[c] = t[c].map(pct)
        t["irr_p50"] = t["irr_p50"].map(lambda v: f"{v:.1%}")
        t["req_multiple_p50"] = t["req_multiple_p50"].map(lambda v: f"{v:.1f}x")
        L += [f"**{name}**", "", md(t.rename(columns={"p_top4": "P(top 4)", "p_ucl": "P(UCL season)",
                                                      "ev_p50": "median EV", "irr_p50": "median IRR",
                                                      "req_multiple_p50": "exit multiple for WACC"})), ""]
    L += ["## 5. Reality check: what happened after the deal", ""]
    if not reality.empty:
        r = reality.copy()
        for c in ["p_exact", "p_this_or_worse", "p_top4"]:
            r[c] = r[c].map(lambda v: f"{v:.1%}")
        L += ["United's actual finishes against the distribution simulated on 20 February 2024:", "", md(r), ""]
    if not rev_check.empty:
        L += ["Reported revenue against the simulated distribution (status quo):", "", md(rev_check, ",.2f"), "",
              "The model ran hot on revenue in the first years after the deal: commercial income was flat in "
              "FY2024 (the model grows it 3.5% a year plus a Champions League uplift), and the old-format group "
              "stage paid less than the 2024-27 tariffs assumed here. Both errors flatter the valuation, so they "
              "strengthen the conclusion rather than weaken it.", ""]
    if now is not None:
        first = next(iter(now.values()))
        L += [f"**Re-run as of {first.cfg.as_of}** (same code, all results to date, base year FY{first.base_fy}):", ""]
        rows = []
        for k in now:
            a, b = valuation.summary(runs[k]), valuation.summary(now[k])
            rows.append(dict(scenario=k, ev_at_deal=gbp(a["ev_p50"]), ev_now=gbp(b["ev_p50"]),
                             p_ucl_at_deal=pct(a["p_ucl"]), p_ucl_now=pct(b["p_ucl"])))
        L += [md(pd.DataFrame(rows)), "",
              "Values are at different dates (and the later run is calibrated to FY2025 costs, after the first "
              "INEOS cuts), so compare the sporting columns more than the money.", "",
              f"United's filtered level: {model.dynamics.level(results.MUFC):.0f} at the deal, "
              f"{now[next(iter(now))].model.dynamics.level(results.MUFC):.0f} now.", ""]
    L += ["## Limitations", "",
          "* **Strength is free.** The sweep treats a better United as costless. In reality strength is bought "
          "with wages and transfers, so the IRR gains in section 4 are an upper bound.",
          "* **Tariffs are approximations** (UEFA 2024-27 cycle, PL 2023-24 distribution). Calibration to the base "
          "year absorbs level errors but not the *size* of the sporting swings.",
          "* **Sponsor clauses beyond the kit deal are not public.** The kit penalty, the UCL pay-cut share "
          "and the player-sale profit ratio are assumptions (see `manu_valuation/config.py`).",
          "* **Not modelled:** the proposed new stadium (a large capex *and* matchday bet), Old Trafford "
          "regeneration funding, working capital, FX on USD debt, and changes to European competition formats.",
          "* **One league's history.** The dynamics are fitted on Premier League seasons since 2010-11 "
          f"({d.diagnostics['n_obs']} team-seasons).", "",
          "## Data", "",
          f"* Results: `{_rel(args.results)}` (openfootball, public domain), {len(model.matches):,} fixtures "
          "known at the valuation date.",
          f"* Financials: `{_rel(fin_path)}`.",
          "* Every assumption, with its source: `manu_valuation/config.py`."]
    return "\n".join(L) + "\n"


def _rel(path) -> str:
    p = Path(path).resolve()
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(path)


def percentile_text(res) -> str:
    n = len(res.dcf.ev)
    above = int((res.dcf.ev < res.price).sum())
    if above == n:
        return f"above all {n:,} simulated paths"
    return f"at the {above / n:.1%} percentile of simulated paths"


def base_rate_ll(model) -> float:
    f = model.elo_run.matches["outcome"].value_counts(normalize=True)
    return float(-(f * np.log(f)).sum())


def projection_table(res) -> pd.DataFrame:
    cols = ["revenue", "broadcasting", "matchday", "commercial", "wages", "other_opex", "ebitda",
            "player_capex", "ppe_capex", "tax", "fcff"]
    t = pd.DataFrame({c: res.proj[c].mean(axis=0) for c in cols}, index=[f"FY{s + 1}" for s in res.paths.seasons])
    return t.T.reset_index().rename(columns={"index": "£m"})


if __name__ == "__main__":
    main()
