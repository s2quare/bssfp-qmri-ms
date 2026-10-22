"""
Figure 3: Orientation dependence of R1, R2, AI for HC vs RRMS.

Outputs:
1. longform_subject_values.csv - Per-subject per-bin values (Group, SID, Metric, NuFO, bin, value)
2. mean_bin_diff.csv - RRMS-HC differences per bin with t-tests and effect sizes
3. anisotropy_comparison_with_ci.csv - Per-subject anisotropy with group comparisons
4. tract_composition_per_bin.csv - WM tract contributions per bin (not NuFO-separated)
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import yaml
from matplotlib.gridspec import GridSpec


sys.path.append(str(Path(__file__).parent.parent))

try:
    from utils.plotting import apply_figure_style, add_subplot_label, create_figure as make_figure
    from utils import paths
    from utils.paper_figure_helpers import *
    from utils.paper_stats import *
    from utils.paper_meta import *
    from colorresourcespy.py.relaxationColorMap import generate_relaxation_colormap
except ImportError as exc:
    raise RuntimeError(f"Could not import project utilities: {exc}") from exc


# =============================================================================
# Configuration
# =============================================================================

cfg = yaml.safe_load(open(paths.default_config, "r", encoding="utf-8"))

METRICS = {
    "t1_pinn": {"label": "R1", "ylabel": r"$\mathbf{R_1}$ [1/s]", "order": 0},
    "t2_pinn": {"label": "R2", "ylabel": r"$\mathbf{R_2}$ [1/s]", "order": 1},
    "ai_map.unringed": {"label": "AI", "ylabel": "AI", "order": 2},
}

ANGLE_BINS = np.arange(cfg["angle_bins_start"],
                       cfg["angle_bins_end"], cfg["angle_bins_step"])
ANGLE_BIN_CENTERS = (ANGLE_BINS[:-1] + ANGLE_BINS[1:]) / 2

GROUPS = {"HC": {"color": "tab:blue"}, "RRMS": {"color": "tab:orange"}}

PROFILE_KEYS = ("nufo1_theta1", "nufo2_theta1", "nufo3_theta1")

BUNDLE_PATTERNS = ("AF_", "ILF", "IFOF", "CST", "SLF")
LINE_STYLES = {"theta1": "-"}

PAPER_FIG_DIR = paths.root / "fig" / Path(__file__).stem
PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
R1_MIN, R1_MAX = 0.3, 2.5
R2_MIN, R2_MAX = 5.0, 23.0
AI_MIN, AI_MAX = -0.02, 0.08
R1_CMAP = generate_relaxation_colormap(
    paths.colorresource / "lipari.csv", R1_MIN, R1_MAX, True)
R1_CMAP.set_under(R1_CMAP(1))
R1_CMAP.set_over(R1_CMAP(R1_CMAP.N - 1))
R2_CMAP = generate_relaxation_colormap(
    paths.colorresource / "navia.csv", R2_MIN, R2_MAX, True)
R2_CMAP.set_under(R2_CMAP(1))
R2_CMAP.set_over(R2_CMAP(R2_CMAP.N - 1))
NUFO_CMAP = mcolors.ListedColormap(["#1f77b4", "#ff7f0e", "#2ca02c"])
NUFO_NORM = mcolors.BoundaryNorm([0.5, 1.5, 2.5, 3.5], NUFO_CMAP.N)


# =============================================================================
# Helper Functions
# =============================================================================

def get_bundle_masks(subject_id: str, bundle_patterns: tuple[str, ...]) -> Dict[str, np.ndarray]:
    """Load BundleSeg NIfTI masks filtered by bundle_patterns."""
    bundle_dir = paths.out_subject_space / subject_id / "Bundleseg_HCP1065"
    if not bundle_dir.exists():
        return {}

    raw: Dict[str, np.ndarray] = {}
    for mask_fp in sorted(bundle_dir.glob("*_mask.nii.gz")):
        bundle_name = mask_fp.name.replace("_mask.nii.gz", "")
        if any(bundle_name.startswith(pat) for pat in bundle_patterns):
            mask_img = load_nifti(mask_fp)
            if mask_img is not None:
                raw[bundle_name] = mask_img.get_fdata() > 0

    masks: Dict[str, np.ndarray] = {}
    for name, mask in raw.items():
        base = name[:-2] if name.endswith(("_L", "_R")) else name
        if base in masks:
            masks[base] = masks[base] | mask
        else:
            masks[base] = mask.copy()
    return masks


def build_mask(sid: str, exclude_lesions: bool = False) -> np.ndarray | None:
    """Build refined WM mask for a subject."""
    # Load R1/R2 WM limits from defaults.yaml (consistent with figure2)
    r1_limits = (cfg.get("r1_mask_wm_lower", np.nan),
                 cfg.get("r1_mask_wm_upper", np.nan))
    r2_limits = (cfg.get("r2_mask_wm_lower", np.nan),
                 cfg.get("r2_mask_wm_upper", np.nan))

    # Use the cerebral WM mask builder from utils.paper_figure_helpers
    return build_subject_cerebral_wm_mask(
        sid,
        r1_limits=r1_limits,
        r2_limits=r2_limits,
        exclude_lesions=exclude_lesions,
    )


def calculate_anisotropy(profile: np.ndarray, metric_key: str) -> float:
    """Calculate anisotropy from orientation-dependent profile."""
    if np.all(np.isnan(profile)) or len(profile) == 0:
        return np.nan

    M_max = np.nanmax(profile)
    M_min = np.nanmin(profile)

    if np.isnan(M_max) or np.isnan(M_min):
        return np.nan

    if metric_key == "ai_map.unringed":
        anisotropy = (M_max - M_min) * 100
    else:
        if (M_max + M_min) == 0:
            return np.nan
        anisotropy = (M_max - M_min) / (M_max + M_min) * 100

    return anisotropy


# =============================================================================
# Step 1: Compute per-subject profiles and create long-form table
# =============================================================================


def compute_subject_profiles(sid: str, exclude_lesions: bool = False) -> Tuple[Dict, Dict] | None:
    """
    Compute binned profiles for all metrics and track voxel counts per bin.

    Returns:
        profiles: {metric_key: {profile_key: binned_values}}
        bin_voxel_counts: {metric_key: {profile_key: {bin_idx: n_voxels}}}
    """
    mask = build_mask(sid, exclude_lesions)
    if mask is None or not np.any(mask):
        return None

    nufo = load_subject_metric("nufo", sid)
    theta1_data = load_or_compute_theta1_from_peaks(sid, save_if_missing=True)
    if nufo is None or theta1_data is None:
        return None
    theta1, _ = theta1_data

    nufo_masks = {
        "nufo1_theta1": mask & (nufo == 1),
        "nufo2_theta1": mask & (nufo == 2),
        "nufo3_theta1": mask & (nufo == 3),
    }

    profiles = {}
    bin_voxel_counts = {}

    for metric_key in METRICS:
        data = load_subject_metric(metric_key, sid)
        if data is None:
            continue

        profiles[metric_key] = {}
        bin_voxel_counts[metric_key] = {}

        for profile_key, region_mask in nufo_masks.items():
            binned = bin_by_angle(data, theta1, region_mask, ANGLE_BINS)
            profiles[metric_key][profile_key] = binned

            # Count voxels per bin
            valid_mask = region_mask & np.isfinite(theta1) & np.isfinite(data)
            theta1_valid = theta1[valid_mask]

            counts = {}
            for i in range(len(ANGLE_BINS) - 1):
                in_bin = (theta1_valid >= ANGLE_BINS[i]) & (
                    theta1_valid < ANGLE_BINS[i + 1])
                counts[i] = int(np.sum(in_bin))

            bin_voxel_counts[metric_key][profile_key] = counts

    return profiles, bin_voxel_counts


def compute_tract_composition_per_bin(
    sid: str,
    exclude_lesions: bool = False
) -> Dict[int, Dict[str, int]] | None:
    """
    For each angle bin, count how many voxels belong to each WM tract.
    Not NuFO-separated - uses the whole WM mask.

    Returns:
        {bin_idx: {tract_name: n_voxels, "total": total_voxels}}
    """
    wm_mask = build_mask(sid, exclude_lesions)
    if wm_mask is None:
        return None

    bundle_masks = get_bundle_masks(sid, BUNDLE_PATTERNS)
    if not bundle_masks:
        return None

    nufo = load_subject_metric("nufo", sid)
    theta1_data = load_or_compute_theta1_from_peaks(sid, save_if_missing=True)
    if nufo is None or theta1_data is None:
        return None
    theta1, _ = theta1_data

    # Flatten all masks
    flat_wm_mask = wm_mask.ravel() > 0
    flat_theta1 = theta1.ravel()
    flat_bundles = {name: mask.ravel() for name, mask in bundle_masks.items()}

    # Get valid voxels (WM mask + finite theta1)
    valid_mask = flat_wm_mask & np.isfinite(flat_theta1)
    theta1_valid = flat_theta1[valid_mask]

    result = {}
    for i in range(len(ANGLE_BINS) - 1):
        bin_lower, bin_upper = ANGLE_BINS[i], ANGLE_BINS[i + 1]
        in_bin = (theta1_valid >= bin_lower) & (theta1_valid < bin_upper)

        # Map back to original indices
        valid_indices = np.where(valid_mask)[0]
        bin_indices = valid_indices[in_bin]

        tract_counts = {"total": int(np.sum(in_bin))}
        for tract_name, flat_bundle in flat_bundles.items():
            tract_voxels = flat_bundle[bin_indices]
            tract_counts[tract_name] = int(np.sum(tract_voxels))

        result[i] = tract_counts

    return result


def create_longform_table(
    subjects: List[str],
    group: str,
    exclude_lesions: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Step 1: Create long-form table with one row per subject × metric × NuFO × bin.

    Returns:
        longform_df: Group, SID, Metric, NuFO, bin_index, bin_label, value
        anisotropy_df: Per-subject anisotropy values
        tract_df: Tract composition per bin
    """
    longform_records = []
    anisotropy_records = []
    tract_records = []

    for sid in subjects:
        print(f"  Processing {sid}...")
        result = compute_subject_profiles(sid, exclude_lesions)
        if result is None:
            continue

        profiles, bin_voxel_counts = result
        subject_record = {"subject_id": sid}

        # Compute tract composition (not NuFO-separated)
        tract_comp = compute_tract_composition_per_bin(sid, exclude_lesions)

        for metric_key, profile_dict in profiles.items():
            metric_label = METRICS[metric_key]["label"]

            for profile_key, values in profile_dict.items():
                nufo_num = int(profile_key.replace("nufo", "").split("_")[0])

                # Calculate anisotropy
                if not np.all(np.isnan(values)):
                    aniso = calculate_anisotropy(values, metric_key)
                    subject_record[f"{metric_key}_{profile_key}_anisotropy"] = aniso

                # Long-form records
                for bin_idx, value in enumerate(values):
                    if np.isfinite(value):
                        bin_label = f"{bin_idx + 1}: {ANGLE_BINS[bin_idx]:.1f}-{ANGLE_BINS[bin_idx + 1]:.1f} deg"
                        longform_records.append({
                            "Group": group,
                            "SID": sid,
                            "Metric": metric_label,
                            "NuFO": nufo_num,
                            "bin_index": bin_idx,
                            "bin_label": bin_label,
                            "value": float(value),
                            "n_voxels": bin_voxel_counts[metric_key][profile_key].get(bin_idx, 0),
                        })

        anisotropy_records.append(subject_record)

        # Tract composition records
        if tract_comp:
            for bin_idx, tract_data in tract_comp.items():
                bin_label = f"{bin_idx + 1}: {ANGLE_BINS[bin_idx]:.1f}-{ANGLE_BINS[bin_idx + 1]:.1f} deg"
                tract_records.append({
                    "Group": group,
                    "SID": sid,
                    "bin_index": bin_idx,
                    "bin_label": bin_label,
                    "total_voxels": tract_data["total"],
                    **{f"tract_{name}": count for name, count in tract_data.items() if name != "total"}
                })

    longform_df = pd.DataFrame(longform_records)
    anisotropy_df = pd.DataFrame(anisotropy_records)
    tract_df = pd.DataFrame(tract_records)

    return longform_df, anisotropy_df, tract_df


# =============================================================================
# Step 2a: Mean bin differences (RRMS - HC)
# =============================================================================

def compute_mean_bin_diff(
    hc_long: pd.DataFrame,
    rrms_long: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Step 2a: Calculate RRMS-HC differences per bin with t-tests.

    Returns:
        df: Full per-bin statistics
        summary: Per-metric summary (mean diff ± SD, mean effect size, significance)
    """
    results = []

    for metric_key, metric_info in METRICS.items():
        metric_label = metric_info["label"]

        for profile_key in PROFILE_KEYS:
            nufo_num = int(profile_key.replace("nufo", "").split("_")[0])

            hc_subset = hc_long[
                (hc_long["Metric"] == metric_label) &
                (hc_long["NuFO"] == nufo_num)
            ]
            rrms_subset = rrms_long[
                (rrms_long["Metric"] == metric_label) &
                (rrms_long["NuFO"] == nufo_num)
            ]

            for bin_idx in range(len(ANGLE_BINS) - 1):
                hc_vals = hc_subset[hc_subset["bin_index"]
                                    == bin_idx]["value"].dropna().values
                rrms_vals = rrms_subset[rrms_subset["bin_index"]
                                        == bin_idx]["value"].dropna().values

                if len(hc_vals) == 0 and len(rrms_vals) == 0:
                    continue

                hc_mean = float(np.nanmean(hc_vals)) if len(
                    hc_vals) > 0 else np.nan
                hc_sd = float(np.nanstd(hc_vals, ddof=1)) if len(
                    hc_vals) > 1 else np.nan
                rrms_mean = float(np.nanmean(rrms_vals)) if len(
                    rrms_vals) > 0 else np.nan
                rrms_sd = float(np.nanstd(rrms_vals, ddof=1)) if len(
                    rrms_vals) > 1 else np.nan

                mean_diff, ci_lower, ci_upper = mean_diff_ci(
                    rrms_vals, hc_vals)
                cohens = cohens_d(hc_vals, rrms_vals)
                p_val, test_name = test_groups(rrms_vals, hc_vals)

                bin_label = f"{bin_idx + 1}: {ANGLE_BINS[bin_idx]:.1f}-{ANGLE_BINS[bin_idx + 1]:.1f} deg"

                results.append({
                    "Metric": metric_label,
                    "NuFO": nufo_num,
                    "bin_index": bin_idx,
                    "bin_label": bin_label,
                    "hc_mean": hc_mean,
                    "hc_sd": hc_sd,
                    "hc_n": len(hc_vals),
                    "rrms_mean": rrms_mean,
                    "rrms_sd": rrms_sd,
                    "rrms_n": len(rrms_vals),
                    "mean_diff": mean_diff,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "cohens_d": cohens,
                    "p_value": p_val,
                    "test_name": test_name,
                })

    df = pd.DataFrame(results)

    # FDR correction per metric
    if not df.empty:
        df["p_value_fdr_bh"] = np.nan
        df["reject_fdr_bh"] = False

        for metric in df["Metric"].unique():
            idx = df[df["Metric"] == metric].index
            pvals = df.loc[idx, "p_value"].dropna().tolist()
            if len(pvals) > 0:
                corrected, reject = bh_fdr(pvals, q=0.05)
                df.loc[idx, "p_value_fdr_bh"] = corrected
                df.loc[idx, "reject_fdr_bh"] = reject

    # =========================================================================
    # Summary per metric: mean diff ± SD, mean effect size, overall significance
    # =========================================================================
    summary_records = []

    for metric in df["Metric"].unique():
        metric_df = df[df["Metric"] == metric]

        # Mean and SD of mean differences across all bins
        mean_diffs = metric_df["mean_diff"].dropna()
        mean_diff_avg = float(np.nanmean(mean_diffs)) if len(
            mean_diffs) > 0 else np.nan
        mean_diff_sd = float(np.nanstd(mean_diffs, ddof=1)
                             ) if len(mean_diffs) > 1 else np.nan

        # Mean effect size across all bins
        effect_sizes = metric_df["cohens_d"].dropna()
        mean_d = float(np.nanmean(effect_sizes)) if len(
            effect_sizes) > 0 else np.nan
        mean_d_sd = float(np.nanstd(effect_sizes, ddof=1)) if len(
            effect_sizes) > 1 else np.nan

        # Check if all bins are significant (FDR-corrected)
        all_significant = bool(metric_df["reject_fdr_bh"].all(
        )) if metric_df["reject_fdr_bh"].notna().all() else False

        # Get minimum p-value (most significant bin)
        min_p = float(metric_df["p_value"].min()) if len(
            metric_df["p_value"].dropna()) > 0 else np.nan

        # Average CI width
        ci_widths = metric_df["ci_upper"] - metric_df["ci_lower"]
        mean_ci_width = float(np.nanmean(ci_widths)) if len(
            ci_widths.dropna()) > 0 else np.nan

        summary_records.append({
            "Metric": metric,
            "mean_diff": f"{mean_diff_avg:.2f}" if np.isfinite(mean_diff_avg) else "NA",
            "mean_diff_sd": f"{mean_diff_sd:.2f}" if np.isfinite(mean_diff_sd) else "NA",
            "mean_cohens_d": f"{mean_d:.2f}" if np.isfinite(mean_d) else "NA",
            "mean_cohens_d_sd": f"{mean_d_sd:.2f}" if np.isfinite(mean_d_sd) else "NA",
            "mean_ci_width": f"{mean_ci_width:.2f}" if np.isfinite(mean_ci_width) else "NA",
            "all_bins_significant": "Yes" if all_significant else "No",
            "min_p_value": f"{min_p:.2f}" if np.isfinite(min_p) else "NA",
            "n_bins": len(metric_df),
        })

    summary_df = pd.DataFrame(summary_records)

    return df, summary_df


# =============================================================================
# Step 2b: Anisotropy comparison
# =============================================================================

def compute_anisotropy_comparison(
    hc_aniso: pd.DataFrame,
    rrms_aniso: pd.DataFrame
) -> pd.DataFrame:
    """
    Step 2b: Compare anisotropy between RRMS and HC.

    Returns DataFrame with group comparisons, effect sizes, and CIs.
    """
    results = []

    for metric_key, metric_info in METRICS.items():
        metric_label = metric_info["label"]

        for profile_key in PROFILE_KEYS:
            col_name = f"{metric_key}_{profile_key}_anisotropy"

            if col_name not in hc_aniso.columns or col_name not in rrms_aniso.columns:
                continue

            hc_vals = hc_aniso[col_name].dropna().values
            rrms_vals = rrms_aniso[col_name].dropna().values

            if len(hc_vals) == 0 and len(rrms_vals) == 0:
                continue

            nufo_num = int(profile_key.replace("nufo", "").split("_")[0])
            theta_type = "_".join(profile_key.split("_")[1:])

            hc_mean = float(np.nanmean(hc_vals)) if len(
                hc_vals) > 0 else np.nan
            hc_sd = float(np.nanstd(hc_vals, ddof=1)) if len(
                hc_vals) > 1 else np.nan
            rrms_mean = float(np.nanmean(rrms_vals)) if len(
                rrms_vals) > 0 else np.nan
            rrms_sd = float(np.nanstd(rrms_vals, ddof=1)) if len(
                rrms_vals) > 1 else np.nan

            mean_diff, ci_lower, ci_upper = mean_diff_ci(rrms_vals, hc_vals)
            cohens = cohens_d(hc_vals, rrms_vals)
            p_val, test_name = test_groups(rrms_vals, hc_vals)

            results.append({
                "Metric": metric_label,
                "NuFO": nufo_num,
                "theta_type": theta_type,
                "hc_n": len(hc_vals),
                "hc_mean": hc_mean,
                "hc_sd": hc_sd,
                "rrms_n": len(rrms_vals),
                "rrms_mean": rrms_mean,
                "rrms_sd": rrms_sd,
                "mean_diff": mean_diff,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "cohens_d": cohens,
                "p_value": p_val,
                "test_name": test_name,
            })

    df = pd.DataFrame(results)

    # FDR correction per metric
    if not df.empty:
        df["p_value_fdr_bh"] = np.nan
        df["reject_fdr_bh"] = False

        for metric in df["Metric"].unique():
            idx = df[df["Metric"] == metric].index
            pvals = df.loc[idx, "p_value"].dropna().tolist()
            if len(pvals) > 0:
                corrected, reject = bh_fdr(pvals, q=0.05)
                df.loc[idx, "p_value_fdr_bh"] = corrected
                df.loc[idx, "reject_fdr_bh"] = reject

    return df


# =============================================================================
# Step 3: Tract composition statistics
# =============================================================================

def compute_tract_composition_stats(
    hc_tract: pd.DataFrame,
    rrms_tract: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Step 3: Test whether tract composition differs between groups per bin.

    Returns:
        df: Full per-tract per-bin statistics
        summary_table: Wide format (bins as rows, tracts as columns, cells: "HC mean ± SD | RRMS mean ± SD")
        significance_summary: Min/max p-values and significance flag
    """
    results = []

    # Get all tract columns
    tract_cols = [col for col in hc_tract.columns if col.startswith("tract_")]

    for bin_idx in range(len(ANGLE_BINS) - 1):
        hc_bin = hc_tract[hc_tract["bin_index"] == bin_idx]
        rrms_bin = rrms_tract[rrms_tract["bin_index"] == bin_idx]

        for tract_col in tract_cols:
            hc_vals = hc_bin[tract_col].dropna().values
            rrms_vals = rrms_bin[tract_col].dropna().values

            if len(hc_vals) == 0 and len(rrms_vals) == 0:
                continue

            tract_name = tract_col.replace("tract_", "")

            hc_mean = float(np.nanmean(hc_vals)) if len(
                hc_vals) > 0 else np.nan
            hc_sd = float(np.nanstd(hc_vals, ddof=1)) if len(
                hc_vals) > 1 else np.nan
            rrms_mean = float(np.nanmean(rrms_vals)) if len(
                rrms_vals) > 0 else np.nan
            rrms_sd = float(np.nanstd(rrms_vals, ddof=1)) if len(
                rrms_vals) > 1 else np.nan

            mean_diff, ci_lower, ci_upper = mean_diff_ci(rrms_vals, hc_vals)
            cohens = cohens_d(hc_vals, rrms_vals)
            p_val, test_name = test_groups(rrms_vals, hc_vals)

            bin_label = f"{bin_idx + 1}: {ANGLE_BINS[bin_idx]:.1f}-{ANGLE_BINS[bin_idx + 1]:.1f} deg"

            results.append({
                "bin_index": bin_idx,
                "bin_label": bin_label,
                "tract": tract_name,
                "hc_mean": hc_mean,
                "hc_sd": hc_sd,
                "hc_n": len(hc_vals),
                "rrms_mean": rrms_mean,
                "rrms_sd": rrms_sd,
                "rrms_n": len(rrms_vals),
                "mean_diff": mean_diff,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "cohens_d": cohens,
                "p_value": p_val,
                "test_name": test_name,
            })

    df = pd.DataFrame(results)

    # FDR correction across all tests
    if not df.empty:
        pvals = df["p_value"].dropna().tolist()
        if len(pvals) > 0:
            corrected, reject = bh_fdr(pvals, q=0.05)
            df["p_value_fdr_bh"] = corrected
            df["reject_fdr_bh"] = reject

    # =========================================================================
    # Summary table: bins as rows, tracts as columns, cells: "HC mean ± SD | RRMS mean ± SD"
    # =========================================================================
    summary_records = []

    for bin_idx in sorted(df["bin_index"].unique()):
        bin_df = df[df["bin_index"] == bin_idx]
        bin_label = bin_df["bin_label"].iloc[0]

        row = {"bin_index": bin_idx, "bin_label": bin_label}

        for tract_col in tract_cols:
            tract_name = tract_col.replace("tract_", "")
            tract_data = bin_df[bin_df["tract"] == tract_name]

            if len(tract_data) > 0:
                hc_mean = tract_data["hc_mean"].iloc[0]
                hc_sd = tract_data["hc_sd"].iloc[0]
                rrms_mean = tract_data["rrms_mean"].iloc[0]
                rrms_sd = tract_data["rrms_sd"].iloc[0]

                if np.isfinite(hc_mean) and np.isfinite(hc_sd) and np.isfinite(rrms_mean) and np.isfinite(rrms_sd):
                    # cell_text = f"{hc_mean:.1f} ± {hc_sd:.1f} | {rrms_mean:.1f} ± {rrms_sd:.1f}"
                    cell_text = f"{int(hc_mean)}|{int(rrms_mean)}"
                else:
                    cell_text = "NA"
            else:
                cell_text = "NA"

            row[tract_name] = cell_text

        summary_records.append(row)

    summary_table = pd.DataFrame(summary_records)

    # =========================================================================
    # Significance summary: min/max p-values and whether any significant
    # =========================================================================
    all_pvals = df["p_value"].dropna()
    all_pvals_fdr = df["p_value_fdr_bh"].dropna()

    any_significant = bool(df["reject_fdr_bh"].any()
                           ) if "reject_fdr_bh" in df.columns else False
    min_p = float(all_pvals.min()) if len(all_pvals) > 0 else np.nan
    max_p = float(all_pvals.max()) if len(all_pvals) > 0 else np.nan
    min_p_fdr = float(all_pvals_fdr.min()) if len(
        all_pvals_fdr) > 0 else np.nan

    significance_summary = pd.DataFrame({
        "any_FDR_significant": ["Yes" if any_significant else "No"],
        "min_p_value": [f"{min_p:.4f}" if np.isfinite(min_p) else "NA"],
        "max_p_value": [f"{max_p:.4f}" if np.isfinite(max_p) else "NA"],
        "min_p_value_FDR": [f"{min_p_fdr:.4f}" if np.isfinite(min_p_fdr) else "NA"],
        "n_tests": [len(df)],
    })

    return df, summary_table, significance_summary


def aggregate_profiles(subjects: list[str], exclude_lesions: bool = False) -> tuple[dict, pd.DataFrame]:
    storage = {m: {k: [] for k in PROFILE_KEYS} for m in METRICS}
    anisotropy_records = []

    for sid in subjects:
        print(f"  Processing {sid}...")
        result = compute_subject_profiles(sid, exclude_lesions)
        if result is None:
            continue

        profiles, _ = result
        subject_record = {"subject_id": sid}

        for metric, profile_dict in profiles.items():
            for key, values in profile_dict.items():
                if not np.all(np.isnan(values)):
                    storage[metric][key].append(values)
                    anisotropy = calculate_anisotropy(values, metric)
                    subject_record[f"{metric}_{key}_anisotropy"] = anisotropy

        anisotropy_records.append(subject_record)

    anisotropy_df = pd.DataFrame(anisotropy_records)
    return storage, anisotropy_df


def summarize_profiles(storage: dict) -> dict:
    summary = {}
    for metric, profiles in storage.items():
        summary[metric] = {}
        for key, arrays in profiles.items():
            if arrays:
                stack = np.stack(arrays)
                summary[metric][key] = {"mean": np.nanmean(
                    stack, axis=0), "std": np.nanstd(stack, axis=0)}
            else:
                summary[metric][key] = {"mean": np.full(
                    len(ANGLE_BIN_CENTERS), np.nan), "std": np.full(len(ANGLE_BIN_CENTERS), np.nan)}
    return summary


def get_cache_path(exp_name: str, group: str) -> Path:
    cache_dir = get_paper_figure_cache_dir(exp_name)
    return cache_dir / f"{group}_summary.npz"


def save_summary(fp: Path, summary: dict) -> None:
    payload = {}
    for metric, profiles in summary.items():
        for key, stats in profiles.items():
            payload[f"{metric}__{key}__mean"] = stats["mean"]
            payload[f"{metric}__{key}__std"] = stats["std"]
    np.savez_compressed(fp, **payload)


def load_summary(fp: Path) -> dict:
    data = np.load(fp)
    summary = {m: {} for m in METRICS}
    for metric in METRICS:
        for key in PROFILE_KEYS:
            mean_key = f"{metric}__{key}__mean"
            std_key = f"{metric}__{key}__std"
            if mean_key in data and std_key in data:
                summary[metric][key] = {
                    "mean": data[mean_key], "std": data[std_key]}
            else:
                summary[metric][key] = {"mean": np.full(
                    len(ANGLE_BIN_CENTERS), np.nan), "std": np.full(len(ANGLE_BIN_CENTERS), np.nan)}
    return summary


def _extract_axial_slice(data: np.ndarray, slice_index: int) -> np.ndarray:
    sl_idx = min(slice_index, data.shape[2] - 1)
    return np.rot90(data[:, :, sl_idx])


def _mask_zeros(data: np.ndarray | None) -> np.ndarray | None:
    if data is None:
        return None
    out = data.copy()
    out[out == 0] = np.nan
    return out


def create_orientation_figure(summaries: dict, exp_name: str, source: str, suffix: str, x_label: str) -> plt.Figure:
    metric_order = sorted(METRICS.items(), key=lambda x: x[1]["order"])
    nufo_levels = [1, 2, 3]

    fig = make_figure(height=9, width=16)
    gs = GridSpec(3, 3, figure=fig, wspace=0.18, hspace=0.25)
    axes = np.empty((3, 3), dtype=object)
    for r in range(3):
        for c in range(3):
            sharex_ax = axes[2, c] if r < 2 else None
            sharey_ax = axes[r, 0] if c > 0 else None
            axes[r, c] = fig.add_subplot(
                gs[r, c], sharex=sharex_ax, sharey=sharey_ax)

    subplot_labels = get_subplot_labels(9)
    label_idx = 0

    for row_idx, (metric_key, info) in enumerate(metric_order):
        for col_idx, nufo_level in enumerate(nufo_levels):
            ax = axes[row_idx, col_idx]
            profile_key = f"nufo{nufo_level}_{source}"

            for group in GROUPS.keys():
                stats = summaries[group].get(
                    metric_key, {}).get(profile_key, {})
                mean = stats.get("mean")
                std = stats.get("std")
                if mean is None or np.all(np.isnan(mean)):
                    continue

                color = GROUPS[group]["color"]
                ax.plot(ANGLE_BIN_CENTERS, mean, color=color,
                        linestyle=LINE_STYLES[source], linewidth=1.8)
                if std is not None:
                    ax.fill_between(ANGLE_BIN_CENTERS, mean -
                                    std, mean + std, color=color, alpha=0.08)

            if col_idx == 0:
                ax.set_ylabel(info["ylabel"], fontsize=14, fontweight="bold")
            if row_idx == 2:
                ax.set_xlabel(x_label, fontsize=14, fontweight="bold")

            if row_idx == 0:
                ax.set_title(f"NuFO={nufo_level}",
                             fontsize=13, fontweight="bold")

            ax.set_xlim(0, 90)
            ax.set_xticks(np.arange(0, 91, 15))
            ax.tick_params(axis="both", which="major", labelsize=12)
            ax.grid(True, linestyle="--", alpha=0.4)

            add_subplot_label(ax, subplot_labels[label_idx])
            label_idx += 1

    legend_handles = [plt.Line2D([], [], color=GROUPS["HC"]["color"], linestyle=LINE_STYLES[source], linewidth=2.0, label="HC"), plt.Line2D(
        [], [], color=GROUPS["RRMS"]["color"], linestyle=LINE_STYLES[source], linewidth=2.0, label="RRMS")]
    fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(
        0.5, -0.01), frameon=False, ncol=2, fontsize=13, handlelength=3.0)

    out_fp = PAPER_FIG_DIR / f"{exp_name}_{suffix}.svg"
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fp, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out_fp}")
    return fig

# =============================================================================
# Main
# =============================================================================


def main():
    apply_figure_style()
    exp_name = Path(__file__).stem
    out_dir = get_paper_figure_cache_dir(exp_name)

    print("\n" + "=" * 80)
    print("FIGURE 3: Orientation dependence HC vs RRMS (Revision 2-5)")
    print("=" * 80)

    # Get subjects
    subject_map = get_subject_id_mapping()
    hc_subjects = get_hc_baseline_subject_ids(subject_map)
    rrms_subjects = get_age_matched_rrms_subject_ids(subject_map, hc_subjects)

    print(f"\nHC subjects: {len(hc_subjects)}")
    print(f"RRMS subjects: {len(rrms_subjects)}")

    # =========================================================================
    # Step 1: Create long-form tables (load from cache if exists)
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 1: Computing per-subject profiles...")
    print("=" * 80)

    hc_long_fp = out_dir / "HC_longform_subject_values.csv"
    rrms_long_fp = out_dir / "RRMS_longform_subject_values.csv"
    hc_aniso_fp = out_dir / "HC_anisotropy.csv"
    rrms_aniso_fp = out_dir / "RRMS_anisotropy.csv"
    hc_tract_fp = out_dir / "HC_tract_composition.csv"
    rrms_tract_fp = out_dir / "RRMS_tract_composition.csv"

    # Load or compute HC
    if hc_long_fp.exists() and hc_aniso_fp.exists() and hc_tract_fp.exists():
        print(f"Loading HC from cache...")
        hc_long = pd.read_csv(hc_long_fp)
        hc_aniso = pd.read_csv(hc_aniso_fp)
        hc_tract = pd.read_csv(hc_tract_fp)
    else:
        print(f"Computing HC profiles...")
        hc_long, hc_aniso, hc_tract = create_longform_table(
            hc_subjects, "HC", exclude_lesions=False)
        hc_long.to_csv(hc_long_fp, index=False)
        hc_aniso.to_csv(hc_aniso_fp, index=False)
        hc_tract.to_csv(hc_tract_fp, index=False)
        print(f"  Saved: {hc_long_fp}, {hc_aniso_fp}, {hc_tract_fp}")

    # Load or compute RRMS
    if rrms_long_fp.exists() and rrms_aniso_fp.exists() and rrms_tract_fp.exists():
        print(f"Loading RRMS from cache...")
        rrms_long = pd.read_csv(rrms_long_fp)
        rrms_aniso = pd.read_csv(rrms_aniso_fp)
        rrms_tract = pd.read_csv(rrms_tract_fp)
    else:
        print(f"Computing RRMS profiles...")
        rrms_long, rrms_aniso, rrms_tract = create_longform_table(
            rrms_subjects, "RRMS", exclude_lesions=True)
        rrms_long.to_csv(rrms_long_fp, index=False)
        rrms_aniso.to_csv(rrms_aniso_fp, index=False)
        rrms_tract.to_csv(rrms_tract_fp, index=False)
        print(f"  Saved: {rrms_long_fp}, {rrms_aniso_fp}, {rrms_tract_fp}")

    # Combine long-form tables
    longform_combined = pd.concat([hc_long, rrms_long], ignore_index=True)
    longform_combined.to_csv(
        out_dir / "longform_subject_values.csv", index=False)
    print(
        f"\nSaved combined: {out_dir / 'longform_subject_values.csv'} ({len(longform_combined)} rows)")

    # ------------------------------------------------------------------
    # Compute aggregated summaries for plotting (cache-aware)
    # ------------------------------------------------------------------
    def get_summary_local(group: str, subjects: list, exclude_lesions: bool) -> tuple[dict, pd.DataFrame]:
        cache_fp = get_cache_path(exp_name, group)
        if cache_fp.exists():
            print(f"Loading {group} from cache...")
            summary = load_summary(cache_fp)
            # ensure anisotropy df exists
            anisotropy_fp = cache_fp.parent / f"{group}_anisotropy.csv"
            if anisotropy_fp.exists():
                anisotropy_df = pd.read_csv(anisotropy_fp)
            else:
                storage, anisotropy_df = aggregate_profiles(
                    subjects, exclude_lesions)
                summary = summarize_profiles(storage)
                save_summary(cache_fp, summary)
                anisotropy_df.to_csv(anisotropy_fp, index=False)
            return summary, anisotropy_df

        print(f"Computing {group} summary...")
        storage, anisotropy_df = aggregate_profiles(subjects, exclude_lesions)
        summary = summarize_profiles(storage)
        save_summary(cache_fp, summary)
        anisotropy_fp = cache_fp.parent / f"{group}_anisotropy.csv"
        anisotropy_df.to_csv(anisotropy_fp, index=False)
        return summary, anisotropy_df

    summaries = {}
    anisotropy_dfs = {}

    for group, subjects, exclude_lesions in [("HC", hc_subjects, False), ("RRMS", rrms_subjects, True)]:
        summary, aniso_df = get_summary_local(group, subjects, exclude_lesions)
        summaries[group] = summary
        anisotropy_dfs[group] = aniso_df
    # Create orientation figure (theta1)
    fig_theta_1 = create_orientation_figure(
        summaries, exp_name, source="theta1", suffix="theta_1", x_label=r"$\mathbf{\theta}$ [deg]")
    plt.close(fig_theta_1)

    # =========================================================================
    # Step 2a: Mean bin differences
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 2a: Computing mean bin differences (RRMS - HC)...")
    print("=" * 80)

    mean_bin_diff, mean_bin_summary = compute_mean_bin_diff(hc_long, rrms_long)
    mean_bin_diff.to_csv(out_dir / "mean_bin_diff.csv", index=False)
    mean_bin_summary.to_csv(out_dir / "mean_bin_diff_summary.csv", index=False)
    print(
        f"\nSaved: {out_dir / 'mean_bin_diff.csv'} ({len(mean_bin_diff)} rows)")
    print(
        f"Saved summary: {out_dir / 'mean_bin_diff_summary.csv'} ({len(mean_bin_summary)} rows)")

    # Print summary
    print("\nPer-metric summary (RRMS - HC across all bins):")
    for _, row in mean_bin_summary.iterrows():
        sig_status = "all significant" if row[
            "all_bins_significant"] == "Yes" else f"not all significant (min p={row['min_p_value']})"
        print(f"  {row['Metric']}: Δ = {row['mean_diff']} ± {row['mean_diff_sd']}, "
              f"d = {row['mean_cohens_d']} ± {row['mean_cohens_d_sd']}, {sig_status}")

    # Print key bin results
    print("\nKey bin differences (30-60 deg range):")
    key_results = mean_bin_diff[mean_bin_diff["bin_index"].between(4, 7)]
    for _, row in key_results.head(10).iterrows():
        sig = "*" if row["reject_fdr_bh"] else ""
        print(f"  {row['Metric']}, NuFO={row['NuFO']}, bin {row['bin_index']}: "
              f"Δ = {row['mean_diff']:.3f} [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}], "
              f"d = {row['cohens_d']:.2f}, p = {row['p_value']:.3f}{sig}")

    # =========================================================================
    # Step 2b: Anisotropy comparison
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 2b: Comparing anisotropy (RRMS - HC)...")
    print("=" * 80)

    aniso_comparison = compute_anisotropy_comparison(hc_aniso, rrms_aniso)
    aniso_comparison.to_csv(
        out_dir / "anisotropy_comparison_with_ci.csv", index=False)
    print(
        f"\nSaved: {out_dir / 'anisotropy_comparison_with_ci.csv'} ({len(aniso_comparison)} rows)")

    # Print summary
    print("\nAnisotropy group comparisons:")
    for _, row in aniso_comparison.iterrows():
        sig = "*" if row["reject_fdr_bh"] else ""
        print(f"  {row['Metric']}, NuFO={row['NuFO']}: "
              f"RRMS {row['rrms_mean']:.2f} ± {row['rrms_sd']:.2f} vs "
              f"HC {row['hc_mean']:.2f} ± {row['hc_sd']:.2f}, "
              f"Δ = {row['mean_diff']:.2f} [{row['ci_lower']:.2f}, {row['ci_upper']:.2f}], "
              f"d = {row['cohens_d']:.2f}, p = {row['p_value']:.3f}{sig}")

    # =========================================================================
    # Step 3: Tract composition
    # =========================================================================
    print("\n" + "=" * 80)
    print("Step 3: Analyzing tract composition per bin...")
    print("=" * 80)

    tract_stats, tract_summary, tract_sig_summary = compute_tract_composition_stats(
        hc_tract, rrms_tract)
    tract_stats.to_csv(out_dir / "tract_composition_per_bin.csv", index=False)
    tract_summary.to_csv(
        out_dir / "tract_composition_summary.csv", index=False)
    tract_sig_summary.to_csv(
        out_dir / "tract_composition_significance.csv", index=False)
    print(
        f"\nSaved: {out_dir / 'tract_composition_per_bin.csv'} ({len(tract_stats)} rows)")
    print(f"Saved summary: {out_dir / 'tract_composition_summary.csv'}")
    print(
        f"Saved significance: {out_dir / 'tract_composition_significance.csv'}")

    # Print summary
    print("\nTract composition (no significant differences expected):")
    for _, row in tract_stats.head(10).iterrows():
        sig = "*" if row["reject_fdr_bh"] else ""
        print(f"  Bin {row['bin_index']}, {row['tract']}: "
              f"RRMS {row['rrms_mean']:.1f} ± {row['rrms_sd']:.1f} vs "
              f"HC {row['hc_mean']:.1f} ± {row['hc_sd']:.1f}, "
              f"Δ = {row['mean_diff']:.1f}, d = {row['cohens_d']:.2f}, p = {row['p_value']:.3f}{sig}")

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
