"""The statistics: opponent adjustment, style shifts vs placebo, and the bounce.

Pipeline
  1. adjust()        remove opponent and home effects from every metric
  2. standardise()   express metrics in units of "how much teams differ"
  3. style_shift()   how far did style move at each real change, compared with
                     fake change points inside unchanged spells?
  4. bounce()        did results improve after the change by more than
                     regression to the mean predicts?
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import STYLE_METRICS
from .spells import Window, placebo_changes

METRICS = list(STYLE_METRICS)
OUTCOMES = ["xgd", "points"]


# --------------------------------------------------------------------------
# 1. Opponent adjustment
# --------------------------------------------------------------------------
def adjust(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Remove opponent and home-advantage effects, league by league.

    Style depends on the opponent: everyone has less of the ball against
    Barcelona. For each metric we fit  y = team + opponent + home  and subtract
    the opponent and home parts. The team part and the match-to-match variation
    (which is where a manager change would show up) are left alone.
    """
    df = df.copy()
    for league, g in df.groupby("league"):
        teams = sorted(set(g["team"]) | set(g["opponent"]))
        pos = {t: i for i, t in enumerate(teams)}
        n, k = len(g), len(teams)
        X = np.zeros((n, 2 * k + 1))
        X[np.arange(n), g["team"].map(pos).to_numpy()] = 1
        X[np.arange(n), k + g["opponent"].map(pos).to_numpy()] = 1
        X[:, -1] = (g["venue"] == "H").to_numpy() - 0.5
        for c in cols:
            y = g[c].to_numpy(float)
            ok = ~np.isnan(y)
            beta = np.linalg.lstsq(X[ok], y[ok], rcond=None)[0]
            opp = beta[k:2 * k] - beta[k:2 * k].mean()  # centred: the average opponent adds 0
            df.loc[g.index, f"{c}_adj"] = y - opp[g["opponent"].map(pos).to_numpy()] - beta[-1] * X[:, -1]
    return df


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["xgd"] = df["np_xg_for"] - df["np_xg_against"]
    return adjust(df, METRICS + OUTCOMES)


# --------------------------------------------------------------------------
# 2. Scale
# --------------------------------------------------------------------------
def between_team_sd(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """SD of team-season averages: the typical style difference between two teams."""
    return df.groupby("team")[cols].mean().std()


def standardise(df: pd.DataFrame) -> pd.DataFrame:
    """z columns: adjusted metric in units of the between-team SD.

    A shift of 1.0 on `possession_z` means the team changed its possession by as
    much as the typical gap between two teams in these leagues.
    """
    adj = [f"{m}_adj" for m in METRICS]
    sd = between_team_sd(df, adj)
    for m in METRICS:
        df[f"{m}_z"] = (df[f"{m}_adj"] - df[f"{m}_adj"].mean()) / sd[f"{m}_adj"]
    return df


def reliability(df: pd.DataFrame) -> pd.DataFrame:
    """How much of each metric is team identity vs. match-to-match noise.

    share_between: between-team variance / total variance, at the single-match level.
    reliability_10: reliability of a 10-match average (Spearman-Brown), i.e. how
                    well a 10-match window pins down a team's level. Shifts in
                    low-reliability metrics are mostly noise.
    """
    rows = []
    for m in METRICS:
        c = f"{m}_adj"
        d = df[["team", c]].dropna()
        n_bar = d.groupby("team").size().mean()
        within = d.groupby("team")[c].var().mean()
        between = max(d.groupby("team")[c].mean().var() - within / n_bar, 0)
        icc = between / (between + within)
        rows.append({"metric": m, "label": STYLE_METRICS[m], "share_between": icc,
                     "reliability_10": 10 * icc / (1 + 9 * icc)})
    return pd.DataFrame(rows).sort_values("share_between", ascending=False)


# --------------------------------------------------------------------------
# 3. Style shift
# --------------------------------------------------------------------------
def _means(Z: np.ndarray, rows: list[int]) -> np.ndarray:
    return np.nanmean(Z[rows], axis=0)


def shift_vector(Z: np.ndarray, w: Window) -> np.ndarray:
    return _means(Z, w.post) - _means(Z, w.pre)


def distance(v: np.ndarray) -> float:
    """Root-mean-square shift across metrics, in between-team SDs."""
    return float(np.sqrt(np.nanmean(v ** 2)))


def _rms(vecs: np.ndarray) -> np.ndarray:
    return np.sqrt(np.nanmean(vecs ** 2, axis=-1))


def holm(p: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    out = np.empty_like(p)
    out[order] = np.minimum(adj, 1)
    return out


def style_shift(df: pd.DataFrame, sp: pd.DataFrame, changes: list[Window],
                n_perm: int = 10_000, seed: int = 0) -> dict:
    """Compare style shifts at real manager changes with placebo change points.

    Each real change is compared with placebos using the same window length
    (a 6-match average is noisier than a 10-match one, so it moves more by chance).

    Two summaries of the 14-metric shift vector:
      shift       RMS shift in between-team SDs: "how far did the team move,
                  compared with how different teams are from each other".
      shift_norm  RMS of each metric's shift divided by that metric's placebo
                  SD: "how unusual is this move, compared with normal drift".
                  This is the test statistic. The raw RMS is dominated by noisy
                  metrics (counter-attack shots swing by >1.5 SDs over 10
                  matches with no change at all), which drowns out stable ones.
                  Under no effect, shift_norm is about 1.
    """
    rng = np.random.default_rng(seed)
    zcols = [f"{m}_z" for m in METRICS]
    Z = df[zcols].to_numpy(float)
    pos = {label: i for i, label in enumerate(df.index)}

    def rows(w: Window) -> Window:  # index labels -> positions in Z
        return Window(w.team, w.league, [pos[r] for r in w.pre], [pos[r] for r in w.post],
                      w.old, w.new, w.date, w.caretaker_matches)

    placebo = {}
    for size in sorted({c.w for c in changes}):
        vecs = np.array([shift_vector(Z, rows(p)) for p in placebo_changes(sp, size)])
        sd = np.nanstd(vecs, axis=0)
        placebo[size] = {"vec": vecs, "sd": sd, "dist": _rms(vecs), "norm": _rms(vecs / sd)}

    per_change, real_vecs = [], []
    for c in changes:
        v = shift_vector(Z, rows(c))
        real_vecs.append(v)
        pl = placebo[c.w]
        d, dn = distance(v), distance(v / pl["sd"])
        per_change.append({
            "change": c.label, "team": c.team, "league": c.league, "old": c.old, "new": c.new,
            "date": c.date, "window": c.w, "caretaker_matches": c.caretaker_matches,
            "shift": d, "shift_norm": dn,
            "placebo_median": np.median(pl["norm"]),
            "placebo_p05": np.quantile(pl["norm"], 0.05),
            "placebo_p95": np.quantile(pl["norm"], 0.95),
            "percentile": (pl["norm"] < dn).mean() * 100,
            **{f"d_{m}": x for m, x in zip(METRICS, v)},
        })
    per_change = pd.DataFrame(per_change).sort_values("shift_norm", ascending=False)
    real_vecs = np.array(real_vecs)

    # Null distribution of the *average* over all changes: draw one placebo of
    # matching window length for every real change, many times.
    sizes = [c.w for c in changes]
    draws = {s: rng.integers(0, len(placebo[s]["dist"]), size=(n_perm, sizes.count(s)))
             for s in set(sizes)}

    def null_mean(key):
        return sum(placebo[s][key][draws[s]].sum(axis=1) for s in draws) / len(changes)

    def null_vec(f):
        return sum(f(placebo[s]["vec"][draws[s]]).sum(axis=1) for s in draws) / len(changes)

    summary = {"n_changes": len(changes)}
    for key, col in [("norm", "shift_norm"), ("dist", "shift")]:
        null, obs = null_mean(key), per_change[col].mean()
        summary[key] = {
            "mean": obs, "placebo_mean": null.mean(), "ratio": obs / null.mean(),
            "p_value": (np.sum(null >= obs) + 1) / (n_perm + 1),
        }
    summary["share_above_p95"] = (per_change["shift_norm"] > per_change["placebo_p95"]).mean()

    null_abs = null_vec(np.abs)
    null_signed = null_vec(lambda x: x)
    obs_abs = np.nanmean(np.abs(real_vecs), axis=0)
    obs_signed = np.nanmean(real_vecs, axis=0)
    by_metric = pd.DataFrame({
        "metric": METRICS,
        "label": [STYLE_METRICS[m] for m in METRICS],
        "real_abs_shift": obs_abs,
        "placebo_abs_shift": np.nanmean(null_abs, axis=0),
        "ratio": obs_abs / np.nanmean(null_abs, axis=0),
        "p_abs": (np.sum(null_abs >= obs_abs, axis=0) + 1) / (n_perm + 1),
        "mean_signed_shift": obs_signed,
        "signed_lo": np.nanquantile(null_signed, 0.025, axis=0),
        "signed_hi": np.nanquantile(null_signed, 0.975, axis=0),
    })
    by_metric["p_holm"] = holm(by_metric["p_abs"])
    by_metric = by_metric.sort_values("ratio", ascending=False)
    return {"per_change": per_change, "summary": summary, "by_metric": by_metric,
            "placebo": placebo}


# --------------------------------------------------------------------------
# 4. New-manager bounce vs regression to the mean
# --------------------------------------------------------------------------
def bounce(df: pd.DataFrame, sp: pd.DataFrame, changes: list[Window],
           n_boot: int = 5000, seed: int = 0) -> dict:
    """Did results improve after a change by more than regression to the mean?

    Managers get sacked after bad runs, and bad runs end on their own. So we ask
    what happened next to teams *without* a change that had equally bad runs:
    across all placebo change points, regress the post-window average on the
    pre-window average (separately per window length). The fitted line is the
    expected recovery; the bounce "beyond regression to the mean" is how far
    real changes land above it.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for outcome in ["points_adj", "xgd_adj"]:
        y = df[outcome]
        lines, pl_points = {}, []
        for size in sorted({c.w for c in changes}):
            pl = placebo_changes(sp, size)
            pre = np.array([y.loc[p.pre].mean() for p in pl])
            post = np.array([y.loc[p.post].mean() for p in pl])
            lines[size] = np.polyfit(pre, post, 1)
            if size == max(c.w for c in changes):
                pl_points = np.column_stack([pre, post])
        rows = []
        for c in changes:
            pre, post = y.loc[c.pre].mean(), y.loc[c.post].mean()
            expected = np.polyval(lines[c.w], pre)
            rows.append({"change": c.label, "window": c.w, "pre": pre, "post": post,
                         "naive": post - pre, "expected_post": expected,
                         "beyond_rtm": post - expected})
        t = pd.DataFrame(rows)
        boot = rng.choice(len(t), size=(n_boot, len(t)))
        out[outcome] = {
            "per_change": t,
            "naive": t["naive"].mean(),
            "naive_ci": np.quantile(t["naive"].to_numpy()[boot].mean(axis=1), [0.025, 0.975]),
            "beyond_rtm": t["beyond_rtm"].mean(),
            "beyond_rtm_ci": np.quantile(t["beyond_rtm"].to_numpy()[boot].mean(axis=1), [0.025, 0.975]),
            "rtm_line": lines[max(lines)],
            "placebo_points": pl_points,
            "window": max(lines),
        }
    return out


# --------------------------------------------------------------------------
# 5. Fingerprints
# --------------------------------------------------------------------------
def fingerprints(df: pd.DataFrame, sp: pd.DataFrame, min_matches: int = 6) -> pd.DataFrame:
    """Average z-scored style of every (team, manager) spell with enough matches."""
    zcols = [f"{m}_z" for m in METRICS]
    rows = []
    for _, s in sp[sp["n"] >= min_matches].iterrows():
        rows.append({"team": s["team"], "league": s["league"], "manager": s["manager"],
                     "n": s["n"], "start": s["start"],
                     **df.loc[s["rows"], zcols].mean().rename(lambda c: c[:-2]).to_dict()})
    return pd.DataFrame(rows)


def pca_2d(fp: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project fingerprints onto their first two principal components.

    Returns (coords, loadings[2 x metrics], explained variance ratio[2]).
    """
    X = fp[METRICS].to_numpy(float)
    X = np.where(np.isnan(X), np.nanmean(X, axis=0), X)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    coords = U[:, :2] * S[:2]
    # Orient axes so that PC1 grows with possession and PC2 with pressing height.
    for j, m in enumerate(["possession", "press_high_share"]):
        if Vt[j, METRICS.index(m)] < 0:
            coords[:, j] *= -1
            Vt[j] *= -1
    return coords, Vt[:2], (S ** 2 / (S ** 2).sum())[:2]


def coach_vs_club(fp: pd.DataFrame) -> pd.DataFrame:
    """For managers seen at two clubs: is their new team closer to their old
    team's style (the coach carries the style) or to its own previous style
    (the squad carries it)?"""
    rows = []
    for mgr, g in fp.groupby("manager"):
        if g["team"].nunique() < 2:
            continue
        g = g.sort_values("start")
        old, new = g.iloc[0], g.iloc[-1]
        predecessors = fp[(fp["team"] == new["team"]) & (fp["start"] < new["start"])]
        if predecessors.empty:
            continue
        prev = predecessors.sort_values("start").iloc[-1]
        v = lambda r: r[METRICS].to_numpy(float)  # noqa: E731
        rows.append({
            "manager": mgr, "old_club": old["team"], "new_club": new["team"],
            "predecessor": prev["manager"],
            "dist_to_own_old_team": distance(v(new) - v(old)),
            "dist_to_predecessor_at_new_club": distance(v(new) - v(prev)),
        })
    return pd.DataFrame(rows)
