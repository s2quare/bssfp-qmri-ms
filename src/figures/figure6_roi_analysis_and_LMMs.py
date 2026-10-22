'''
Figure 6 and Tables 3-4: ROI-based comparison of R1, R2, and AI between HC and
RRMS groups.

The script calculates ROI-level summaries, fits primary linear mixed-effects
models, performs ROI-specific group comparisons with FDR correction, and
generates the corresponding boxplots and result tables.
'''
# %% Imports and paths
from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
import yaml
sys.path.append(str(Path(__file__).parent.parent))

try:
    from utils import paths
    from utils.paper_figure_helpers import *
    from utils.paper_meta import *
    from utils.paper_stats import *
except Exception as exc:
    raise ImportError(
        "Could not import utils. Ensure that the parent directory is in sys.path."
    ) from exc

filename = Path(__file__).stem

sns.set_theme(style="whitegrid", context="notebook")

METRICS: Dict[str, Dict[str, str]] = {
    "t1_pinn": {"label": "R1"},
    "t2_pinn": {"label": "R2"},
    "ai_map.unringed": {"label": "AI"},
}
ROI_ORDER, ROI_LABELS = get_roi_label_config()
BUNDLE_PATTERNS = ("AF_", "ILF", "IFOF", "CST", "SLF")
LESION_ROI_KEY = "Lesion mask"
FDR_Q = 0.05

# %% Functions


def build_long_form_roi_table(metadata_fp: Path) -> Tuple[pd.DataFrame, list[str], list[str], pd.DataFrame]:
    """Build long-form table: one row per subject–ROI–metric with age and gender."""
    subject_id_map = get_subject_id_mapping()
    hc_subjects = get_hc_baseline_subject_ids(subject_id_map)
    rrms_subjects = get_age_matched_rrms_subject_ids(
        subject_id_map, hc_subjects, metadata_fp=metadata_fp
    )
    cfg_fp = paths.default_config
    with open(cfg_fp, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    demo = prepare_demographics(metadata_fp)

    records = []
    for group, subjects in [("HC", hc_subjects), ("RRMS", rrms_subjects)]:
        for subject_id in subjects:
            roi_means = compute_roi_means(
                subject_id,
                group,
                roi_labels=ROI_LABELS,
                bundle_patterns=BUNDLE_PATTERNS,
                lesion_roi_key=LESION_ROI_KEY,
                metrics=METRICS,
                wm_r1_limits=(cfg["r1_mask_wm_lower"],
                              cfg["r1_mask_wm_upper"]),
                wm_r2_limits=(cfg["r2_mask_wm_lower"],
                              cfg["r2_mask_wm_upper"]),
                gm_r1_limits=(cfg["r1_mask_gm_lower"],
                              cfg["r1_mask_gm_upper"]),
                gm_r2_limits=(cfg["r2_mask_gm_lower"],
                              cfg["r2_mask_gm_upper"]),
            )
            row_demo = demo[demo["subject_id"].eq(str(subject_id))]
            age = float(row_demo["age_year"].iloc[0]
                        ) if len(row_demo) else np.nan
            gender = row_demo["gender"].iloc[0] if len(row_demo) else np.nan

            for roi, metric_values in roi_means.items():
                for metric_key, value in metric_values.items():
                    metric_label = METRICS.get(
                        metric_key, {}).get("label", metric_key)
                    records.append(
                        {
                            "subject_id": str(subject_id),
                            "group": group,
                            "roi": str(roi),
                            "metric_key": str(metric_key),
                            "metric_label": str(metric_label),
                            "value": float(value),
                            "age_year": age,
                            "gender": gender,
                        }
                    )

    long_df = pd.DataFrame(records)
    if not long_df.empty:
        long_df = long_df.sort_values(
            ["group", "subject_id", "roi", "metric_label"]
        ).reset_index(drop=True)

    return long_df, list(hc_subjects), list(rrms_subjects), demo


def roi_specific_table(main_models: Dict[str, Tuple[pd.DataFrame, object]]
                       ) -> pd.DataFrame:
    """Build ROI-level table with descriptive stats and LMM contrasts per metric."""
    rows = []
    for metric_label, (dat_m, result_m) in main_models.items():
        beta = np.asarray(result_m.fe_params, float)
        cov_full = np.asarray(result_m.cov_params(), float)
        k = len(beta)
        covariance = cov_full[:k, :k]

        # Match the notebook's asymptotic Wald inference for ROI contrasts.
        zcrit = stats.norm.ppf(0.975)

        ref_roi = str(dat_m["roi"].cat.categories[0])

        for roi in sorted(dat_m["roi"].astype(str).unique()):
            roi_dat = dat_m[dat_m["roi"].astype(str).eq(roi)]

            hc = roi_dat.loc[roi_dat["group"].eq(
                "HC"), "value"].to_numpy(float)
            rrms = roi_dat.loc[roi_dat["group"].eq(
                "RRMS"), "value"].to_numpy(float)

            vector = contrast_vector(result_m, roi)
            estimate = float(vector @ beta)
            quad = float(vector @ covariance @ vector)
            se = float(np.sqrt(max(quad, 0.0)))
            t_value = estimate / se if se > 0 else np.nan
            p_value = (
                float(2 * stats.norm.sf(abs(t_value)))
                if np.isfinite(t_value)
                else np.nan
            )

            rows.append(
                {
                    "Metric": metric_label,
                    "ROI": roi,
                    "Reference_ROI": ref_roi,
                    "HC_n": int(len(hc)),
                    "HC_mean": float(np.mean(hc)) if len(hc) else np.nan,
                    "HC_sd": float(np.std(hc, ddof=1)) if len(hc) > 1 else np.nan,
                    "RRMS_n": int(len(rrms)),
                    "RRMS_mean": float(np.mean(rrms)) if len(rrms) else np.nan,
                    "RRMS_sd": float(np.std(rrms, ddof=1)) if len(rrms) > 1 else np.nan,
                    "MeanDifference_RRMS_minus_HC": float(np.mean(rrms) - np.mean(hc))
                    if len(hc) and len(rrms)
                    else np.nan,
                    "Cohens_d": cohens_d(hc, rrms),
                    "ContrastEstimate_RRMS_minus_HC": estimate,
                    "ContrastSE": se,
                    "ContrastCI_low": float(estimate - zcrit * se)
                    if np.isfinite(se)
                    else np.nan,
                    "ContrastCI_high": float(estimate + zcrit * se)
                    if np.isfinite(se)
                    else np.nan,
                    "p_value_raw": p_value,
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_value_fdr_bh"] = np.nan
        out["FDR_reject"] = False
        for metric_label in out["Metric"].unique():
            mask = out["Metric"].eq(metric_label)
            q_vals, reject = bh_fdr(out.loc[mask, "p_value_raw"])
            out.loc[mask, "p_value_fdr_bh"] = q_vals
            out.loc[mask, "FDR_reject"] = reject

    return out

# main figure



def plot_roi_separate_boxplots(long_df: pd.DataFrame, roi_sep_results_df: pd.DataFrame, fig_fp: Path, lesion_mask_vals_fp: Path, metric_order: list[str] | None = None, roi_order_all: list[str] | None = None):

    # Determine significant ROIs per metric
    roi_sep_sig_df = roi_sep_results_df[[
        "Metric", "ROI", "p_value_fdr_bh"]].copy()
    roi_sep_sig_df["is_sig_fdr"] = roi_sep_sig_df["p_value_fdr_bh"].map(
        lambda p: bool(np.isfinite(p) and p < 0.05))

    if metric_order is None:
        plot_metrics = ["R1", "R2", "AI"]
    else:
        plot_metrics = metric_order
    plot_metrics_xlabels = [r"$\mathbf{R_1}$", r"$\mathbf{R_2}$", "AI"]

    plot_rois_source = roi_order_all if roi_order_all is not None else sorted(
        long_df["roi"].dropna().unique().tolist())

    fs_order = [
        "Caudate",
        "Thalamus",
        "Cerebral GM",
        "Cerebral WM",
        "Occipital-Parietal WM",
        "Temporo-Parietal WM",
        "Frontal WM",
        "Cerebellar WM",
        "CC",
    ]
    bundle_order = ["AF", "CST", "IFOF", "ILF", "SLF1", "SLF2", "SLF3"]

    desired_top_to_bottom = []
    if LESION_ROI_KEY in plot_rois_source:
        desired_top_to_bottom.append(LESION_ROI_KEY)
    desired_top_to_bottom.extend(
        [r for r in fs_order if r in plot_rois_source])
    desired_top_to_bottom.extend(
        [r for r in bundle_order if r in plot_rois_source])
    remaining_rois = [
        r for r in plot_rois_source if r not in desired_top_to_bottom]
    desired_top_to_bottom.extend(remaining_rois)

    # Matplotlib shows larger y at top, so reverse
    plot_rois = list(reversed(desired_top_to_bottom))
    n_rois = len(plot_rois)
    n_bundles = len([r for r in bundle_order if r in plot_rois_source])

    fig_height = max(9, n_rois * 0.65 + 2)
    fig, axes = plt.subplots(1, 3, figsize=(16, fig_height), sharey=True)

    for ax, panel_label in zip(axes, ("(a)", "(b)", "(c)")):
        ax.text(
            0.02,
            1.005,
            panel_label,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=16,
            fontweight="bold",
        )

    offset = 0.20
    box_width = 0.32

    palette = {"HC": "#1f77b4", "RRMS": "#d95f02"}

    lesion_masK_vals = pd.DataFrame(columns=["Metric", "LesionMaskMeanValue"])

    for ax, metric_name, plot_metric_xlabel in zip(axes, plot_metrics, plot_metrics_xlabels):
        metric_df = long_df[(long_df["metric_label"] == metric_name) & (
            long_df["roi"].isin(plot_rois))].copy()
        metric_sig = roi_sep_sig_df[(roi_sep_sig_df["Metric"].astype(
            str) == str(metric_name)) & (roi_sep_sig_df["is_sig_fdr"])]

        for i, roi_name in enumerate(plot_rois):
            hc_vals = metric_df[(metric_df["roi"] == roi_name) & (
                metric_df["group"] == "HC")]["value"].dropna().to_numpy(dtype=float)
            rrms_vals = metric_df[(metric_df["roi"] == roi_name) & (
                metric_df["group"] == "RRMS")]["value"].dropna().to_numpy(dtype=float)

            boxes_to_plot = [
                (hc_vals, i + offset, palette["HC"]),
                (rrms_vals, i - offset, palette["RRMS"]),
            ]
            if roi_name == LESION_ROI_KEY:
                boxes_to_plot = [(rrms_vals, i, palette["RRMS"])]
                print(f"Lesion Mask value for {metric_name}: {rrms_vals.mean()}")
                lesion_masK_vals = pd.concat([
                    lesion_masK_vals,
                    pd.DataFrame({
                        "Metric": [metric_name],
                        "LesionMaskMeanValue": [rrms_vals.mean()]
                    })
                ], ignore_index=True)

            for vals, y_center, color in boxes_to_plot:
                if len(vals) == 0:
                    continue
                bp = ax.boxplot(
                    vals,
                    positions=[y_center],
                    vert=False,
                    widths=box_width,
                    patch_artist=True,
                    manage_ticks=False,
                    flierprops=dict(
                        marker="o", markersize=4, markerfacecolor=color, alpha=0.6, linestyle="none"),
                    medianprops=dict(color="black", linewidth=2),
                    whiskerprops=dict(color=color, linewidth=1.2),
                    capprops=dict(color=color, linewidth=1.2),
                    boxprops=dict(linewidth=0),
                )
                for patch in bp["boxes"]:
                    patch.set_facecolor(color)
                    patch.set_alpha(0.6)

            # significance marker
            if roi_name in metric_sig["ROI"].astype(str).values:
                all_vals = np.concatenate([hc_vals, rrms_vals]) if (
                    len(hc_vals) + len(rrms_vals)) > 0 else np.array([])
                if all_vals.size > 0:
                    x_span = float(np.nanmax(all_vals) - np.nanmin(all_vals)
                                   ) if np.nanmax(all_vals) > np.nanmin(all_vals) else 1.0
                    x_star = float(np.nanmax(all_vals) + 0.04 * x_span)
                    ax.text(x_star, i, "*", ha="left", va="center",
                            fontsize=16, fontweight="bold")

        # dashed separator between FS ROIs and bundles
        if 0 < n_bundles < n_rois:
            ax.axhline(n_bundles - 0.5, color="gray", linewidth=1.0,
                       linestyle="--", alpha=0.6, zorder=2)

        ax.set_xlabel(plot_metric_xlabel, fontsize=14, fontweight="bold")
        ax.set_yticks(range(n_rois))
        ax.set_yticklabels(plot_rois, fontsize=12, fontweight="bold")
        ax.grid(False, axis="y")
        ax.grid(True, axis="x", linestyle="--", alpha=0.3)
        ax.tick_params(axis="both", which="major", labelsize=12)
        for lbl in ax.get_xticklabels():
            lbl.set_fontweight("bold")
        ax.set_ylim(-0.7, n_rois - 0.3)

    legend_elements = [
        Patch(facecolor=palette["HC"], alpha=0.6, label="HC"),
        Patch(facecolor=palette["RRMS"], alpha=0.6, label="RRMS"),
    ]
    axes[0].legend(handles=legend_elements, loc="lower right", fontsize=11)
    fig.tight_layout()

    fig.savefig(fig_fp, dpi=300,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved Figure 6 (ROI-separate) SVG: {fig_fp}")

    # save lesion mask values as one csv
    lesion_masK_vals.to_csv(lesion_mask_vals_fp, index=False)


def build_table3_primary_lmm(
    main_models: Dict[str, Tuple[pd.DataFrame, object]], output_fp: Path
):
    """
    Build Table 3: Primary LMM results with one row per metric.

    Columns (manuscript):
    Metric, N subjects, N observations, Fixed effects, Random effects,
    RRMS–HC β (95% CI), Group effect p, ROI effect p, Group×ROI p, Age effect p, Sex effect p
    """
    rows = []

    for metric_name, (dat, result) in main_models.items():
        fe = result.fe_params
        pv = result.pvalues
        ci = result.conf_int()
        fe_terms = list(fe.index)

        group_term = "C(group, Treatment(reference='HC'))[T.RRMS]"
        group_beta = float(fe.get(group_term, np.nan))
        group_p = float(pv.get(group_term, np.nan))
        group_ci_low = float(ci.loc[group_term, 0]
                             ) if group_term in ci.index else np.nan
        group_ci_high = float(
            ci.loc[group_term, 1]) if group_term in ci.index else np.nan

        roi_terms = [t for t in fe_terms if t.startswith("C(roi)[T.")]
        interaction_terms = [
            t for t in fe_terms if "C(group" in t and "C(roi)" in t]
        gender_terms = [t for t in fe_terms if t.startswith("C(gender)")]

        roi_effect_p = joint_wald_p_value(result, roi_terms)
        group_roi_p = joint_wald_p_value(result, interaction_terms)
        gender_p = joint_wald_p_value(result, gender_terms)

        age_term = "age_year"
        age_p = float(pv.get(age_term, np.nan))

        if np.isfinite(group_beta) and np.isfinite(group_ci_low) and np.isfinite(
            group_ci_high
        ):
            beta_ci_str = (
                f"{group_beta:.2f} ({group_ci_low:.2f} to {group_ci_high:.2f})"
            )
        else:
            beta_ci_str = "NA"

        rows.append(
            {
                "Metric": metric_name,
                "N subjects": int(dat["subject_id"].nunique()),
                "N observations": len(dat),
                "Fixed effects": "group × ROI + age + sex",
                "Random effects": "subject intercept",
                "RRMS–HC β (95% CI)": beta_ci_str,
                "Group effect p": format_p_value(group_p),
                "ROI effect p": format_p_value(roi_effect_p),
                "Group × ROI p": format_p_value(group_roi_p),
                "Age effect p": format_p_value(age_p),
                "Sex effect p": format_p_value(gender_p),
            }
        )

    table3_df = pd.DataFrame(rows)
    table3_fp = output_fp
    table3_df.to_csv(table3_fp, index=False)
    print(f"Saved Table 3 (primary LMM): {table3_fp}")


def build_table4_roi_specific(
    roi_specific_df: pd.DataFrame,
    output_fp: Path,
    metric_order: list[str] = None,
    roi_order: list[str] = None,
):
    """
    Build Table 4: ROI-specific HC vs RRMS results (all ROIs, all metrics).

    This merges the information that was previously in a separate S1 table
    into one ROI-based difference table, while keeping descriptive stats
    and effect sizes from the original Table 4.

    Columns:
    Metric, ROI,
    HC mean ± SD,
    RRMS mean ± SD,
    Cohen's d,
    LMM β (95% CI),
    Observed_mean_diff_RRMS_minus_HC,
    LMM_contrast_estimate_RRMS_minus_HC,
    Contrast_SE,
    p_uncorr,
    p_corr,
    FDR_significant
    """
    if metric_order is None:
        metric_order = ["R1", "R2", "AI"]

    df = roi_specific_df.copy()

    # Descriptive: HC mean ± SD and RRMS mean ± SD
    df["HC mean ± SD"] = df.apply(
        lambda r: format_mean_sd(r["HC_mean"], r["HC_sd"], decimals=2), axis=1
    )
    df["RRMS mean ± SD"] = df.apply(
        lambda r: format_mean_sd(r["RRMS_mean"], r["RRMS_sd"], decimals=2), axis=1
    )

    # Effect size: Cohen's d
    df["Cohen's d"] = df["Cohens_d"].map(
        lambda v: f"{v:.2f}" if np.isfinite(v) else "NA"
    )

    # Model-based contrast: LMM β (95% CI)
    df["LMM β (95% CI)"] = df.apply(
        lambda r: (
            f"{r['ContrastEstimate_RRMS_minus_HC']:.2f} "
            f"({r['ContrastCI_low']:.2f} to {r['ContrastCI_high']:.2f})"
            if np.isfinite(r["ContrastEstimate_RRMS_minus_HC"])
            else "NA"
        ),
        axis=1,
    )

    df["MeanDifference_RRMS_minus_HC"] = df.apply(
        lambda r: f"{r['MeanDifference_RRMS_minus_HC']:.3f}"
        if np.isfinite(r["MeanDifference_RRMS_minus_HC"])
        else "NA",
        axis=1,
    )

    df["ContrastEstimate_RRMS_minus_HC"] = df.apply(
        lambda r: f"{r['ContrastEstimate_RRMS_minus_HC']:.3f}"
        if np.isfinite(r["ContrastEstimate_RRMS_minus_HC"])
        else "NA",
        axis=1,
    )

    df["ContrastSE"] = df.apply(
        lambda r: f"{r['ContrastSE']:.3f}"
        if np.isfinite(r["ContrastSE"])
        else "NA",
        axis=1,
    )

    df["p_value_raw"] = df["p_value_raw"].apply(format_p_value)
    df["p_value_fdr_bh"] = df["p_value_fdr_bh"].apply(format_p_value)
    df["FDR_reject"] = df["FDR_reject"].map(lambda x: "yes" if x else "no")

    # Difference and inference columns (from 2_roi_specific.csv)
    df = df.rename(
        columns={
            "MeanDifference_RRMS_minus_HC": "Observed_mean_diff_RRMS_minus_HC",
            "ContrastEstimate_RRMS_minus_HC": "LMM_contrast_estimate_RRMS_minus_HC",
            "ContrastSE": "Contrast_SE",
            "p_value_raw": "p_uncorr",
            "p_value_fdr_bh": "p_corr",
            "FDR_reject": "FDR_significant",
        }
    )

    # Select final columns in a logical order
    table4_df = df[
        [
            "Metric",
            "ROI",
            "HC mean ± SD",
            "RRMS mean ± SD",
            "Cohen's d",
            "LMM β (95% CI)",
            "Observed_mean_diff_RRMS_minus_HC",
            "LMM_contrast_estimate_RRMS_minus_HC",
            "Contrast_SE",
            "p_uncorr",
            "p_corr",
            "FDR_significant",
        ]
    ].copy()

    # Sort by metric and ROI
    table4_df["Metric"] = pd.Categorical(
        table4_df["Metric"], categories=metric_order, ordered=True
    )
    if roi_order is not None:
        table4_df["ROI"] = pd.Categorical(
            table4_df["ROI"], categories=roi_order, ordered=True
        )
    table4_df = table4_df.sort_values(["Metric", "ROI"]).reset_index(drop=True)

    table4_fp = output_fp
    table4_df.to_csv(table4_fp, index=False)
    print(f"Saved Table 4 (ROI-specific): {table4_fp}")


# %% run main


def main():
    metadata_fp = (paths.data / "251217-b_get_gender_age_participants" /
                   "251217-b_participant_metadata.csv")
    data_cache = paths.data / "paper_figures" / filename
    data_cache.mkdir(parents=True, exist_ok=True)
    fig_dir = paths.root / "fig" / filename
    fig_dir.mkdir(parents=True, exist_ok=True)

    # %% 0. create long-form ROI table and save
    # One row per subject–ROI–metric, with age and sex.
    # Saved as: 0_roi_mean_values.csv
    long_df_fp = data_cache / "0_roi_mean_values.csv"
    if long_df_fp.exists():
        long_df = pd.read_csv(long_df_fp)
        hc_subjects = long_df.loc[long_df["group"].eq(
            "HC"), "subject_id"].dropna().unique().tolist()
        rrms_subjects = long_df.loc[long_df["group"].eq(
            "RRMS"), "subject_id"].dropna().unique().tolist()
        demographics_df = prepare_demographics(metadata_fp)
        print(
            f"Loaded cached long-form ROI table: {long_df_fp} (shape: {long_df.shape})")
    else:
        long_df, hc_subjects, rrms_subjects, demographics_df = build_long_form_roi_table(
            metadata_fp)
        long_df.to_csv(data_cache / "0_roi_mean_values.csv", index=False)
        print(
            f"HC subjects: {len(hc_subjects)}; RRMS subjects: {len(rrms_subjects)}")
        print(f"Long-form shape: {long_df.shape}")
        print(f"Saved: {data_cache / '0_roi_mean_values.csv'}")

    # %% 1. Primary LMM (one per metric)
    # value ~ group × ROI + age + sex + (1 | subject)
    # Saved full output as: 1_lmm_main_results.csv
    lmm_main_fp = data_cache / "1_lmm_main_results.csv"
    print("Fitting primary LMMs for each metric...")
    lmm_main_df = pd.DataFrame()
    main_rows = []
    main_models: Dict[str, Tuple[pd.DataFrame, object]] = {}

    for metric_label in ["R1", "R2", "AI"]:
        try:
            dat_m, result_m = fit_primary_lmm(long_df, metric_label)
            main_models[metric_label] = (dat_m, result_m)
            main_rows.append(fixed_effect_rows(
                metric_label, dat_m, result_m))
            print(f"Finished fitting LMM for {metric_label}")
        except Exception as exc:
            main_rows.append(
                pd.DataFrame(
                    [{"Metric": metric_label,
                        "Term": "MODEL_FAILED", "Error": repr(exc)}]
                )
            )
    # save main models
    for metric, (data, model) in main_models.items():
        model_fp = data_cache / f"1_lmm_main_model_{metric}.pkl"
        data_fp = data_cache / f"1_lmm_main_data_{metric}.csv"
        model.save(model_fp)
        data.to_csv(data_fp, index=False)
        print(f"Saved LMM model for {metric}: {model_fp}")
        print(f"Saved LMM data for {metric}: {data_fp}")
    lmm_main_df = pd.concat(main_rows, ignore_index=True)
    lmm_main_df.to_csv(lmm_main_fp, index=False)
    print(f"Saved: {lmm_main_fp}")

    # %% 2. ROI-specific LMM results
    # Full ROI-level table (all ROIs, all metrics) saved as: 2_roi_specific.csv
    roi_specific_df = roi_specific_table(main_models)
    roi_specific_df.to_csv(data_cache / "2_roi_specific.csv", index=False)
    print(f"Saved: {data_cache / '2_roi_specific.csv'}")

    # %% 3. Figure 6 for ROI-Separate Analysis (boxplots with FDR asterisks)
    roi_display_order = [
        "Lesion mask",
        "CC",
        "Cerebellar WM",
        "Frontal WM",
        "Temporo-Parietal WM",
        "Occipital-Parietal WM",
        "Cerebral WM",
        "Cerebral GM",
        "Thalamus",
        "Caudate",
        "AF",
        "CST",
        "IFOF",
        "ILF",
        "SLF1",
        "SLF2",
        "SLF3",
    ]
    # Call the ROI-separate boxplot function
    fig_fp = fig_dir / "figure6_roi_separate_boxplots_asterisk.svg"
    plot_roi_separate_boxplots(long_df, roi_specific_df, fig_fp, lesion_mask_vals_fp=data_cache / "lesion_mask_values.csv", metric_order=[
        "R1", "R2", "AI"], roi_order_all=roi_display_order)

    # %% 4. Table 3: Primary LMM results (global, per metric)
    table3_fp = data_cache / "table3_primary_lmm.csv"
    build_table3_primary_lmm(main_models, table3_fp)

    # %% 5. Table 4: ROI-specific HC vs RRMS results (all ROIs, all metrics)
    table4_fp = data_cache / "table4_roi_specific.csv"
    # ROI order for display in Table 4

    build_table4_roi_specific(
        roi_specific_df,
        table4_fp,
        metric_order=["R1", "R2", "AI"],
        roi_order=roi_display_order,
    )


if __name__ == "__main__":
    print("Starting ROI analysis with LMMs...")
    main()
