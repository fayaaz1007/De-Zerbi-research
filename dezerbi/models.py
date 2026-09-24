"""Statistical tests of the De Zerbi hypotheses.

H1  De Zerbi's team performs better with < 55% possession.
H2  ... performs better when the opponent presses harder (LOWER PPDA).
H3  ... performs better when the opponent plays a higher defensive line.

Performance is measured primarily by xG difference (xGD): it is much less
noisy than goals/points, so it needs far fewer matches to detect an effect.
Points (ordered logit on W/D/L) and goal difference are secondary checks.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from patsy import dmatrices
from scipy import stats
from statsmodels.miscmodels.ordinal_model import OrderedModel

# Hypothesis term -> expected sign of its coefficient if the hypothesis holds.
HYPOTHESES = {
    "low_poss": ("H1: <55% possession", +1),
    "opp_press_z": ("H2: opponent presses harder (lower PPDA)", +1),
    "opp_line_z": ("H3: opponent's defensive line higher", +1),
}


def control_terms(df: pd.DataFrame) -> list[str]:
    terms = ["home"]
    if "opp_strength_z" in df:
        terms.append("opp_strength_z")
    # Team-season fixed effects absorb squad quality (Brighton 22/23 is not
    # Marseille 24/25), so the tactical effects are estimated *within* a season.
    if df["team_season"].nunique() > 1:
        terms.append("C(team_season)")
    return terms


def hypothesis_terms(df: pd.DataFrame) -> list[str]:
    return [t for t in HYPOTHESES if t in df]


def formula(outcome: str, terms: list[str]) -> str:
    return f"{outcome} ~ " + " + ".join(terms)


# --------------------------------------------------------------------------
# 1. Simple split comparisons
# --------------------------------------------------------------------------
def group_comparison(df: pd.DataFrame, outcome: str = "xgd", n_boot: int = 2000,
                     seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    splits = {
        "low_poss": (f"poss < {df.attrs.get('possession_threshold', 55):g}%", "poss >= threshold"),
        "high_press": ("opp PPDA below median (presses harder)", "opp PPDA above median"),
        "high_line": ("opp line above median (higher)", "opp line below median"),
    }
    rows = []
    for col, (lab1, lab0) in splits.items():
        if col not in df:
            continue
        a = df.loc[df[col] == 1, outcome].to_numpy()
        b = df.loc[df[col] == 0, outcome].to_numpy()
        if len(a) < 2 or len(b) < 2:
            continue
        boots = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n_boot)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append({
            "split": col, "group": lab1, "n": len(a), "mean": a.mean(),
            "other_n": len(b), "other_mean": b.mean(),
            "diff": a.mean() - b.mean(), "ci_low": lo, "ci_high": hi,
            "p_welch": stats.ttest_ind(a, b, equal_var=False).pvalue,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 2. Regression models
# --------------------------------------------------------------------------
def fit_ols(df: pd.DataFrame, outcome: str, terms: list[str]):
    return smf.ols(formula(outcome, terms), data=df).fit(cov_type="HC3")


def main_model(df: pd.DataFrame, outcome: str = "xgd"):
    """All three hypotheses jointly, with controls."""
    return fit_ols(df, outcome, hypothesis_terms(df) + control_terms(df))


def interaction_model(df: pd.DataFrame, outcome: str = "xgd"):
    """Does conceding possession pay off *specifically* against pressing teams?"""
    terms = hypothesis_terms(df) + ["low_poss:opp_press_z"] + control_terms(df)
    return fit_ols(df, outcome, terms)


def continuous_possession_model(df: pd.DataFrame, outcome: str = "xgd"):
    """Possession as a continuous variable (per 10 percentage points)."""
    df = df.assign(poss10=df["possession"] / 10)
    terms = ["poss10"] + [t for t in hypothesis_terms(df) if t != "low_poss"] + control_terms(df)
    return fit_ols(df, outcome, terms)


def ordered_logit(df: pd.DataFrame):
    """Loss < Draw < Win as a function of the same predictors."""
    terms = hypothesis_terms(df) + control_terms(df)
    _, X = dmatrices(formula("result", terms), df, return_type="dataframe")
    X = X.drop(columns="Intercept")
    y = pd.Categorical(df["result"], categories=[-1, 0, 1], ordered=True)
    return OrderedModel(y, X, distr="logit").fit(method="bfgs", disp=False)


def hypothesis_table(fit, terms=None) -> pd.DataFrame:
    """One row per hypothesis: effect, CI, one-sided p in the predicted direction."""
    terms = terms or [t for t in HYPOTHESES if t in fit.params.index]
    ci = fit.conf_int()
    rows = []
    for t in terms:
        label, sign = HYPOTHESES[t]
        coef, p = fit.params[t], fit.pvalues[t]
        p_one = p / 2 if np.sign(coef) == sign else 1 - p / 2
        rows.append({
            "hypothesis": label, "term": t, "coef": coef,
            "ci_low": ci.loc[t, 0], "ci_high": ci.loc[t, 1],
            "p_two_sided": p, "p_one_sided": p_one,
            "direction_ok": bool(np.sign(coef) == sign),
            "verdict": _verdict(np.sign(coef) == sign, p_one),
        })
    return pd.DataFrame(rows)


def _verdict(direction_ok: bool, p_one: float) -> str:
    if not direction_ok:
        return "not supported (effect goes the other way)"
    if p_one < 0.05:
        return "supported"
    if p_one < 0.10:
        return "weak evidence"
    return "not supported (direction right, not significant)"


# --------------------------------------------------------------------------
# 3. Is 55% the right cut-off?
# --------------------------------------------------------------------------
def threshold_scan(df: pd.DataFrame, outcome: str = "xgd", thresholds=None,
                   n_perm: int = 500, seed: int = 0) -> tuple[pd.DataFrame, float]:
    """Fit the main model for each possession threshold.

    Picking the best of many thresholds inflates significance, so the returned
    p-value is a permutation test of the *maximum* t-statistic across the scan.
    """
    thresholds = thresholds if thresholds is not None else np.arange(45, 66, 1)
    thresholds = [t for t in thresholds
                  if 10 <= (df["possession"] < t).sum() <= len(df) - 10]
    others = [t for t in hypothesis_terms(df) if t != "low_poss"] + control_terms(df)

    def scan(y: np.ndarray) -> list[tuple[float, float, float]]:
        out = []
        for t in thresholds:
            d = df.assign(_y=y, below=(df["possession"] < t).astype(int))
            f = smf.ols(formula("_y", ["below"] + others), data=d).fit(cov_type="HC3")
            out.append((t, f.params["below"], f.tvalues["below"]))
        return out

    res = pd.DataFrame(scan(df[outcome].to_numpy()), columns=["threshold", "coef", "t"])
    if res.empty:
        return res, float("nan")
    observed = res["t"].max()

    # Permute the outcome within team-season so fixed effects stay meaningful.
    rng = np.random.default_rng(seed)
    groups = df.groupby("team_season").indices
    exceed = 0
    for _ in range(n_perm):
        y = df[outcome].to_numpy().copy()
        for idx in groups.values():
            y[idx] = rng.permutation(y[idx])
        exceed += max(t for _, _, t in scan(y)) >= observed
    return res, (exceed + 1) / (n_perm + 1)


# --------------------------------------------------------------------------
# 4. Out-of-sample check: do the tactical variables help *predict*?
# --------------------------------------------------------------------------
@dataclass
class CVResult:
    baseline_mae: float
    full_mae: float
    folds: pd.DataFrame


def leave_one_season_out(df: pd.DataFrame, outcome: str = "xgd") -> CVResult | None:
    """Train on all other seasons, predict the held-out one.

    Team-season dummies can't be estimated for an unseen season, so the CV
    models use only the transferable predictors.
    """
    if df["team_season"].nunique() < 3:
        return None
    base = ["home"] + (["opp_strength_z"] if "opp_strength_z" in df else [])
    full = base + hypothesis_terms(df)
    rows = []
    for ts in df["team_season"].unique():
        train, test = df[df["team_season"] != ts], df[df["team_season"] == ts]
        row = {"held_out": ts, "n": len(test)}
        for name, terms in [("baseline", base), ("full", full)]:
            fit = smf.ols(formula(outcome, terms), data=train).fit()
            row[f"{name}_mae"] = np.abs(test[outcome] - fit.predict(test)).mean()
        rows.append(row)
    folds = pd.DataFrame(rows)
    w = folds["n"] / folds["n"].sum()
    return CVResult((folds["baseline_mae"] * w).sum(), (folds["full_mae"] * w).sum(), folds)
