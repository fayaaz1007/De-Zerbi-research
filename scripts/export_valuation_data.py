"""Export simulated paths for the interactive valuation page.

    python scripts/export_valuation_data.py --out docs/valuation/valuation_data.js

The page recomputes the DCF, IRR and required exit multiple in the browser, so it gets
per-path free cash flow and revenue (GBP m, rounded) rather than finished statistics.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_valuation as rv  # noqa: E402
from manu_valuation import edgar, engine, results, valuation  # noqa: E402
from manu_valuation.config import Config, apply, scenarios  # noqa: E402


def ints(a) -> list:
    return np.rint(np.asarray(a)).astype(int).ravel().tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "valuation" / "valuation_data.js"))
    ap.add_argument("--paths", type=int, default=5000)
    ap.add_argument("--sweep-paths", type=int, default=500)
    args = ap.parse_args()

    matches = results.load(rv.RESULTS_CSV)
    fin_path = rv.EDGAR_CSV if rv.EDGAR_CSV.exists() else rv.SNAPSHOT_CSV
    fin = edgar.load_financials(fin_path)
    cfg = Config(n_paths=args.paths)
    model = engine.fit(matches, cfg.as_of)

    runs = {sc.name: valuation.run(apply(cfg, sc.changes), matches, fin, model) for sc in scenarios()}
    base = runs["Status quo"]
    for r in runs.values():  # same sporting futures in every scenario
        assert (r.paths.finish == base.paths.finish).all()
    P, Y = base.paths.finish.shape

    sweep = {}
    for sc in scenarios():
        rows = []
        for level in rv.LEVELS:
            r = valuation.run(apply(cfg, {**sc.changes, "sporting.level_override": float(level)}),
                              matches, fin, model, args.sweep_paths)
            f = r.paths.finish[:, 1:]
            rows.append(dict(level=level, p_top4=round(float((f <= 4).mean()), 4),
                             p_ucl=round(float(r.paths.ucl[:, 1:].mean()), 4),
                             fcff=ints(r.proj["fcff"]), rev_last=ints(r.proj["revenue"][:, -1])))
        sweep[sc.name] = rows

    bt = valuation.backtest(matches[matches["season"] <= 2025], model.link)
    bins = [0, .05, .2, .4, .6, .8, .95, 1]
    cal = (bt.assign(bin=pd.cut(bt["p_top4"], bins, include_lowest=True)).groupby("bin", observed=True)
           .agg(pred=("p_top4", "mean"), obs=("actual", lambda a: (a <= 4).mean()), n=("p_top4", "size")))
    econ = rv.ucl_economics(base)
    d, deal = model.dynamics, cfg.deal

    data = dict(
        meta=dict(as_of=cfg.as_of, horizon=Y, paths=P, sweep_paths=args.sweep_paths,
                  seasons=[results.season_label(s) for s in base.paths.seasons],
                  fys=[f"FY{s + 1}" for s in base.paths.seasons],
                  first_fraction=base.paths.first_fraction, wacc=base.dcf.wacc,
                  growth=cfg.valuation.terminal_growth, window=cfg.valuation.terminal_window,
                  base_fy=base.base_fy, entry_fy=base.entry_fy, entry_revenue=base.entry_revenue,
                  financials="EDGAR XBRL" if fin_path == rv.EDGAR_CSV else "annual-report snapshot"),
        deal=dict(price_usd=deal.price_usd, shares_m=deal.shares_m, usd_per_gbp=deal.usd_per_gbp,
                  net_debt=deal.net_debt, transfer_payables=deal.transfer_payables),
        scenarios=[dict(name=sc.name, note=sc.note) for sc in scenarios()],
        paths=dict(finish=ints(base.paths.finish), comp=ints(base.paths.comp), stage=ints(base.paths.stage)),
        cash={k: dict(fcff=ints(r.proj["fcff"]), revenue=ints(r.proj["revenue"]), ebitda=ints(r.proj["ebitda"]))
              for k, r in runs.items()},
        sweep=sweep,
        model=dict(level_at_deal=float(np.mean(base.paths.level)), level_filtered=d.level(results.MUFC),
                   sigma_level=d.sigma_level, sigma_form=d.sigma_form, rho=d.rho,
                   elo_k=model.elo_run.params.k, elo_carry=model.elo_run.params.carryover,
                   log_loss=float(model.elo_grid["log_loss"].min()), base_rate_ll=rv.base_rate_ll(model),
                   brier=valuation.brier(bt).round(4).to_dict("records"),
                   calibration=cal.round(4).reset_index(drop=True).to_dict("records")),
        reality=valuation.reality_check(base, matches).round(4).to_dict("records"),
        revenue_check=rv.revenue_check(base, fin).round(2).to_dict("records"),
        econ=[dict(comp=k, **{c: round(float(v), 1) for c, v in row.items()}) for k, row in econ.iterrows()],
        econ_fy=econ.attrs["fy"],
        calibration_table=base.cal.table.round(1).to_dict("records"),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("window.VAL_DATA = " + json.dumps(data, separators=(",", ":")) + ";\n")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in valuation.summary(base).items()})


if __name__ == "__main__":
    main()
