"""
utils to run statistical analyses in scripts
"""
import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.formula.api import mixedlm
from scipy import stats
from typing import Tuple
LESION_ROI_KEY = "Lesion mask"
FDR_Q = 0.05


def bh_fdr(p_values: pd.Series, q: float = FDR_Q) -> Tuple[np.ndarray, np.ndarray]:
    """Benjamini–Hochberg FDR correction."""
    p = np.asarray(p_values, dtype=float)
    out = np.full(p.shape, np.nan)
    reject = np.zeros(p.shape, dtype=bool)

    ok = np.isfinite(p)
    if not ok.any():
        return out, reject

    order = np.argsort(p[ok])
    ranked = p[ok][order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    temp = np.empty_like(adjusted)
    temp[order] = adjusted

    out[ok] = temp
    reject[ok] = out[ok] < q
    return out, reject


def cohens_d(hc: np.ndarray, rrms: np.ndarray) -> float:
    """Cohen's d for RRMS vs HC."""
    hc = np.asarray(hc, float)
    rrms = np.asarray(rrms, float)
    hc = hc[np.isfinite(hc)]
    rrms = rrms[np.isfinite(rrms)]

    if len(hc) < 2 or len(rrms) < 2:
        return np.nan

    pooled = np.sqrt(
        (
            (len(hc) - 1) * np.var(hc, ddof=1)
            + (len(rrms) - 1) * np.var(rrms, ddof=1)
        )
        / (len(hc) + len(rrms) - 2)
    )
    if pooled <= 0:
        return np.nan

    return float((np.mean(rrms) - np.mean(hc)) / pooled)


def contrast_vector(result, roi: str) -> np.ndarray:
    """Build contrast vector for RRMS–HC difference at a given ROI."""
    names = list(result.model.exog_names)
    vector = np.zeros(len(names))

    group_term = "C(group, Treatment(reference='HC'))[T.RRMS]"
    if group_term in names:
        vector[names.index(group_term)] = 1.0

    candidates = [
        f"C(group, Treatment(reference='HC'))[T.RRMS]:C(roi)[T.{roi}]",
        f"C(roi)[T.{roi}]:C(group, Treatment(reference='HC'))[T.RRMS]",
    ]
    for term in candidates:
        if term in names:
            vector[names.index(term)] += 1.0
            break

    return vector


def joint_wald_p_value(result, term_names: list[str]) -> float:
    """Joint Wald test for a set of fixed-effect coefficients."""
    fe = result.fe_params
    fe_terms = list(fe.index)
    selected = [t for t in term_names if t in fe_terms]

    if len(selected) == 0:
        return np.nan

    b = fe.loc[selected].to_numpy(dtype=float)
    cov_fe = result.cov_params().loc[fe_terms, fe_terms]
    cov_sub = cov_fe.loc[selected, selected].to_numpy(dtype=float)

    try:
        stat = float(b.T @ np.linalg.pinv(cov_sub) @ b)
    except Exception:
        return np.nan

    df_num = len(selected)
    if df_num == 0 or not np.isfinite(stat):
        return np.nan

    return float(stats.chi2.sf(stat, df_num))


def format_p_value(p: float) -> str:
    """Format p-value for manuscript display."""
    if not np.isfinite(p):
        return "NA"
    if p < 1e-3:
        return "<0.001"
    return f"{p:.3f}"


def format_mean_sd(mean: float, sd: float, decimals: int = 2) -> str:
    """Format mean ± SD for manuscript display."""
    if not (np.isfinite(mean) and np.isfinite(sd)):
        return "NA"
    return f"{mean:.{decimals}f} ± {sd:.{decimals}f}"


def prepare_metric_lmm_data(long_df: pd.DataFrame, metric_label: str) -> pd.DataFrame:
    """Filter long_df for one metric label, drop lesion ROI, and enforce ROI presence in both groups."""
    dat = long_df[
        (long_df["metric_label"] == metric_label) & (
            long_df["roi"].ne(LESION_ROI_KEY))
    ].copy()

    dat = dat[["subject_id", "group", "roi",
               "value", "age_year", "gender"]].dropna()

    counts = dat.groupby(["roi", "group"]).size().unstack(fill_value=0)
    valid = counts.index[(counts.get("HC", 0) > 0) &
                         (counts.get("RRMS", 0) > 0)]
    dat = dat[dat["roi"].isin(valid)].copy()

    roi_levels = sorted(dat["roi"].astype(str).unique())

    dat["group"] = pd.Categorical(dat["group"], categories=["HC", "RRMS"])
    dat["roi"] = pd.Categorical(dat["roi"].astype(str), categories=roi_levels)
    dat["gender"] = pd.Categorical(dat["gender"].astype(str))

    return dat


def primary_formula() -> str:
    """Fixed-effects formula for the primary LMM."""
    return "value ~ C(group, Treatment(reference='HC')) * C(roi) + age_year + C(gender)"


def fit_primary_lmm(long_df: pd.DataFrame, metric_label: str):
    """Fit primary LMM for one metric."""
    dat = prepare_metric_lmm_data(long_df, metric_label)
    model = mixedlm(
        primary_formula(),
        data=dat,
        groups=dat["subject_id"],
        re_formula="1",  # subject-specific random intercept
    )
    result = model.fit()
    return dat, result


def model_status(result) -> str:
    """Convergence / covariance status indicator."""
    if not bool(getattr(result, "converged", False)):
        return "not_converged"

    cov = np.asarray(result.cov_re, dtype=float)
    if cov.size and np.linalg.eigvalsh(cov).min() < -1e-8:
        return "indefinite_random_covariance"

    return "converged"


def fixed_effect_rows(metric_label: str, dat: pd.DataFrame, result) -> pd.DataFrame:
    """Summarise fixed-effect coefficients, SEs, p-values, and CIs for one metric."""
    ci = result.conf_int()
    rows = []
    for term in result.fe_params.index:
        rows.append(
            {
                "Metric": metric_label,
                "Term": term,
                "Estimate": float(result.fe_params[term]),
                "SE": float(result.bse_fe[term]),
                "p_value": float(result.pvalues[term]),
                "CI_low": float(ci.loc[term, 0]),
                "CI_high": float(ci.loc[term, 1]),
                "N_subjects": dat["subject_id"].nunique(),
                "N_ROIs": dat["roi"].nunique(),
                "N_observations": len(dat),
                "Converged": model_status(result),
                "LogLikelihood": float(result.llf),
            }
        )
    return pd.DataFrame(rows)


# used in figure3 script


def mean_diff_ci(group1: np.ndarray, group2: np.ndarray, ci: float = 0.95) -> Tuple[float, float, float]:
    """Calculate mean difference (group1 - group2) with confidence interval."""
    g1 = np.asarray(group1, dtype=float)[np.isfinite(group1)]
    g2 = np.asarray(group2, dtype=float)[np.isfinite(group2)]

    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return np.nan, np.nan, np.nan

    mean1, mean2 = np.nanmean(g1), np.nanmean(g2)
    var1, var2 = np.nanvar(g1, ddof=1), np.nanvar(g2, ddof=1)

    se = np.sqrt(var1 / n1 + var2 / n2)
    df_num = (var1 / n1 + var2 / n2) ** 2
    df_den = (var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1)
    df = df_num / df_den if df_den > 0 else (n1 + n2 - 2)

    mean_diff = mean1 - mean2
    t_crit = stats.t.ppf(1 - (1 - ci) / 2, df)
    ci_lower = mean_diff - t_crit * se
    ci_upper = mean_diff + t_crit * se

    return float(mean_diff), float(ci_lower), float(ci_upper)


def test_groups(group1: np.ndarray, group2: np.ndarray) -> Tuple[float, str]:
    """Adaptive two-sample test: Welch t-test if normal, else Mann-Whitney U."""
    g1 = np.asarray(group1, dtype=float)[np.isfinite(group1)]
    g2 = np.asarray(group2, dtype=float)[np.isfinite(group2)]

    if len(g1) < 2 or len(g2) < 2:
        return np.nan, "insufficient_n"

    _, p1 = stats.shapiro(g1) if len(g1) >= 3 else (1.0, True)
    _, p2 = stats.shapiro(g2) if len(g2) >= 3 else (1.0, True)

    if p1 > 0.05 and p2 > 0.05:
        _, p_val = stats.ttest_ind(g1, g2, equal_var=False)
        return float(p_val), "welch_t"

    _, p_val = stats.mannwhitneyu(g1, g2, alternative="two-sided")
    return float(p_val), "mannwhitney_u"
