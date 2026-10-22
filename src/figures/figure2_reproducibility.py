"""
Figure 2: HC reproducibility - Orientation dependence and Bland-Altman.

Layout: 2×3 grid
    Row 1 (a-c): Metric vs fiber-to-field angle profiles for R1, R2, AI
                             with mean ± std across HC baseline subjects
    Row 2 (d-f): Bland-Altman plots (baseline vs follow-up ROI means)
                             for R1, R2, AI — combining repeat and reposition subjects
                             color = ROI, marker = session type (●=repeat, ▲=reposition)

Supplementary: ROI overview on S1 warped T1 (axial/coronal/sagittal)
"""

import sys
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import pingouin as pg
from matplotlib.gridspec import GridSpec

sys.path.append(str(Path(__file__).parent.parent))

try:
    from utils import paths
    from utils.paper_figure_helpers import *
    from utils.paper_meta import *
    from utils.plotting import apply_figure_style, add_subplot_label, create_figure as make_figure
except ImportError as e:
    raise ImportError("Failed to import project modules. "
                      "Ensure you are running the script from the project root.") from e

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
config = yaml.safe_load(open(paths.default_config, "r", encoding="utf-8"))

SUBJECTS = {
    "repeat": {"baseline": "HC_7GCNPA_baseline", "repeat": "HC_7GCNPA_repeat"},
    "reposition": {"baseline": "HC_RKWY4Y_baseline", "reposition": "HC_RKWY4Y_reposition"},
}
SUBJECT_COLORS = {"repeat": "tab:blue", "reposition": "tab:green"}
SESSION_LINESTYLES = {"baseline": "-", "repeat": "--", "reposition": ":"}

METRICS = {
    "t1_pinn": {"label": "R1", "ylabel": r"$\mathbf{R_1}$", "vmin": 1.75, "vmax": 2.1},
    "t2_pinn": {"label": "R2", "ylabel": r"$\mathbf{R_2}$", "vmin": 18.5, "vmax": 24.5},
    "ai_map.unringed": {"label": "AI", "ylabel": "AI", "vmin": 0, "vmax": 0.093},
}

FA_MASK = (config["fa_mask_lower"], config["fa_mask_upper"])
ANGLE_BINS = np.arange(config["angle_bins_start"],
                       config["angle_bins_end"], config["angle_bins_step"])
ANGLE_BIN_CENTERS = (ANGLE_BINS[:-1] + ANGLE_BINS[1:]) / 2
ANGLE_SOURCE_LABEL = r"$\mathbf{\theta}$ [deg]"

PAPER_FIG_DIR = paths.root / "fig" / Path(__file__).stem
PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)

# Fixed slices for supplementary ROI overview
AXIAL_ROI = 82
CORONAL_ROI = 53
SAGITTAL_ROI = 69

# ROI configuration loaded from shared defaults via helper.
ROI_ORDER, ROI_LABELS = get_roi_label_config()

# Map display ROI names to wmparc label groups from ROI_LABELS config.
ROI_WMPARC_KEY_GROUPS = {
    "Cerebellar WM": ("cerebellar_wm_l", "cerebellar_wm_r"),
    "Frontal WM": ("frontal_wm_l", "frontal_wm_r"),
    "Temporo-Parietal WM": ("mid_wm_l", "mid_wm_r"),
    "Occipital-Parietal WM": ("posterior_wm_l", "posterior_wm_r"),
    "Thalamus": ("thalamus_l", "thalamus_r"),
    "Caudate": ("caudate_l", "caudate_r"),
    "CC": ("cc",),
}


# ROI colors and session-type markers
ROI_COLORS = {
    "CC": "tab:cyan",
    "Cerebellar WM": "tab:green",
    "Frontal WM": "tab:purple",
    "Temporo-Parietal WM": "tab:brown",
    "Occipital-Parietal WM": "tab:pink",
    "Cerebral WM": "tab:blue",
    "Cerebral GM": "tab:orange",
    "Thalamus": "tab:gray",
    "Caudate": "tab:olive",
}
SESSION_MARKERS = {"repeat": "o", "reposition": "^"}

TISSUE_CLASS_ORDER = [
    "WM",
    "GM",
]

TISSUE_CLASS_MAP = {
    "CC": "WM",
    "Cerebellar WM": "WM",
    "Frontal WM": "WM",
    "Temporo-Parietal WM": "WM",
    "Occipital-Parietal WM": "WM",
    "Cerebral WM": "WM",
    "Cerebral GM": "GM",
    "Thalamus": "GM",
    "Caudate": "GM",
}


# Global WM masks loaded from shared bounds CSV
# Load R1/R2 mask limits from defaults.yaml
WM_LIMITS: dict[str, tuple[float, float]] = {
    "R1": (config.get("r1_mask_wm_lower", np.nan), config.get("r1_mask_wm_upper", np.nan)),
    "R2": (config.get("r2_mask_wm_lower", np.nan), config.get("r2_mask_wm_upper", np.nan)),
}

# GM-specific limits (used when computing ROI means for GM ROIs)
GM_LIMITS = {
    "R1": (config.get("r1_mask_gm_lower", np.nan), config.get("r1_mask_gm_upper", np.nan)),
    "R2": (config.get("r2_mask_gm_lower", np.nan), config.get("r2_mask_gm_upper", np.nan)),
}


def get_tissue_class(roi: str) -> str:
    return TISSUE_CLASS_MAP.get(roi, roi)


def _get_cache_path(out_fop: Path, sid: str) -> Path:
    out_fop.mkdir(parents=True, exist_ok=True)
    return out_fop / f"{sid}__angle_metrics_theta1.npz"


def get_baseline_subjects() -> dict[str, str]:
    return {
        key: cfg["baseline"].replace("_baseline", "")
        for key, cfg in SUBJECTS.items()
    }


def compute_angle_binned_metrics(out_fop: Path, sid: str, wm_mask: np.ndarray) -> dict | None:
    cache_fp = _get_cache_path(out_fop, sid)
    if cache_fp.exists():
        with np.load(cache_fp) as f:
            return {k.replace("metric_", ""): f[k] for k in f.files if k.startswith("metric_")}

    theta1_data = load_or_compute_theta1_from_peaks(sid, save_if_missing=True)
    if theta1_data is None:
        return None

    inclination, _ = theta1_data

    results = {}
    for metric_key, info in METRICS.items():
        data = load_subject_metric(metric_key, sid)
        if data is None:
            continue
        results[info["label"]] = bin_by_angle(
            data, inclination, wm_mask, ANGLE_BINS)

    np.savez(cache_fp, **{f"metric_{k}": v for k, v in results.items()})
    return results


# -----------------------------------------------------------------------------
# Data preparation
# -----------------------------------------------------------------------------

def prepare_angle_data(out_fop: Path, subject_id_map: dict[str, str]) -> dict:
    cache_fp = out_fop / "subject_angle_data.npz"
    if cache_fp.exists():
        with np.load(cache_fp, allow_pickle=True) as f:
            return f["subject_angle_data"].item()

    angle_data: dict[str, dict[str, dict]] = {}
    for subj_name in subject_id_map:
        # Process subjects with follow-up scans (baseline/repeat/reposition)
        if "_baseline" in subj_name:
            base, session = subj_name.replace("_baseline", ""), "baseline"
        elif "_repeat" in subj_name:
            base, session = subj_name.replace("_repeat", ""), "repeat"
        elif "_reposition" in subj_name:
            base, session = subj_name.replace("_reposition", ""), "reposition"
        # Also process HC subjects without follow-up scans (for mean±SD calculation)
        elif "HC" in subj_name:
            base, session = subj_name, "baseline"
        else:
            continue

        sid = subject_id_map.get(subj_name)
        print(f"Processing {subj_name} (ID: {sid})...")
        if sid is None:
            continue
        wm_mask = build_subject_cerebral_wm_mask(
            sid,
            fa_limits=FA_MASK,
            r1_limits=WM_LIMITS["R1"],
            r2_limits=WM_LIMITS["R2"],
        )
        if wm_mask is None:
            continue
        binned = compute_angle_binned_metrics(out_fop, sid, wm_mask)
        if binned is None:
            continue
        angle_data.setdefault(base, {})[session] = binned

    np.savez(cache_fp, subject_angle_data=angle_data)
    return angle_data


def save_angle_relative_differences(
    angle_data: dict,
    selected_subjects: dict[str, str],
    out_fop: Path,
) -> None:
    """Save follow-up-vs-baseline relative differences for the two plotted HCs."""
    records: list[dict] = []
    summary_records: list[dict] = []

    for session_type, base_name in selected_subjects.items():
        sessions = angle_data.get(base_name, {})
        baseline_metrics = sessions.get("baseline", {})
        followup_metrics = sessions.get(session_type, {})
        for metric in ["R1", "R2", "AI"]:
            baseline = np.asarray(baseline_metrics.get(metric), dtype=float)
            followup = np.asarray(followup_metrics.get(metric), dtype=float)
            if baseline.shape != followup.shape:
                continue

            with np.errstate(divide="ignore", invalid="ignore"):
                if metric in {"R1", "R2"}:
                    relative_diff = np.where(
                        baseline != 0,
                        100.0 * (followup - baseline) / np.abs(baseline),
                        np.nan,
                    )
                else:
                    relative_diff = np.where(
                        baseline != 0,
                        100.0 * (followup - baseline),
                        np.nan,
                    )

            valid = np.isfinite(relative_diff)
            for index, value in enumerate(relative_diff):
                records.append({
                    "subject": base_name,
                    "session_type": session_type,
                    "metric": metric,
                    "theta_bin_start_deg": ANGLE_BINS[index],
                    "theta_bin_end_deg": ANGLE_BINS[index + 1],
                    "theta_bin_center_deg": ANGLE_BIN_CENTERS[index],
                    "baseline": baseline[index],
                    "followup": followup[index],
                    "relative_difference_percent": value,
                })

            summary_records.append({
                "subject": base_name,
                "session_type": session_type,
                "metric": metric,
                "n_theta_bins": int(valid.sum()),
                "mean_relative_difference_percent": (
                    float(np.mean(relative_diff[valid]))
                    if np.any(valid) else np.nan
                ),
            })

    if records:
        out_fp = out_fop / "angle_relative_differences.csv"
        pd.DataFrame(records).to_csv(out_fp, index=False)
        print(f"Saved angle relative differences to: {out_fp}")
    if summary_records:
        summary_fp = out_fop / "angle_relative_difference_summary.csv"
        pd.DataFrame(summary_records).to_csv(summary_fp, index=False)
        print(f"Saved angle relative-difference summary to: {summary_fp}")


def compute_roi_absolute_values(subject_id_map: dict[str, str]) -> list[dict]:
    """Return paired ROI mean values (baseline + follow-up) for all subjects.

    Returns
    -------
    list of dicts with keys:
        subject, session_type, roi, tissue_class, metric, baseline, followup
    """
    sessions_by_subject: dict[str, dict[str, str]] = {}
    for name in subject_id_map:
        if "_baseline" in name:
            base, sess = name.replace("_baseline", ""), "baseline"
        elif "_repeat" in name:
            base, sess = name.replace("_repeat", ""), "repeat"
        elif "_reposition" in name:
            base, sess = name.replace("_reposition", ""), "reposition"
        else:
            continue
        sid = subject_id_map.get(name)
        if sid:
            sessions_by_subject.setdefault(base, {})[sess] = sid

    records: list[dict] = []
    for base_name, sessions in sessions_by_subject.items():
        baseline_id = sessions.get("baseline")
        if not baseline_id:
            continue
        baseline_means = compute_roi_means(
            baseline_id,
            "HC",
            roi_labels=ROI_LABELS,
            bundle_patterns=tuple(ROI_WMPARC_KEY_GROUPS.keys()),
            lesion_roi_key="Lesion mask",
            metrics=METRICS,
            selection="wm_and_fs",
            include_bundles=False,
            wm_r1_limits=WM_LIMITS["R1"],
            wm_r2_limits=WM_LIMITS["R2"],
            gm_r1_limits=GM_LIMITS["R1"],
            gm_r2_limits=GM_LIMITS["R2"],
        )

        for followup_key in ["repeat", "reposition"]:
            followup_id = sessions.get(followup_key)
            if not followup_id:
                continue
            followup_means = compute_roi_means(
                followup_id,
                "HC",
                roi_labels=ROI_LABELS,
                bundle_patterns=tuple(ROI_WMPARC_KEY_GROUPS.keys()),
                lesion_roi_key="Lesion mask",
                metrics=METRICS,
                selection="wm_and_fs",
                include_bundles=False,
                wm_r1_limits=WM_LIMITS["R1"],
                wm_r2_limits=WM_LIMITS["R2"],
                gm_r1_limits=GM_LIMITS["R1"],
                gm_r2_limits=GM_LIMITS["R2"],
            )

            for roi in ROI_ORDER:
                for metric in ["R1", "R2", "AI"]:
                    v_base = baseline_means.get(roi, {}).get(metric)
                    v_follow = followup_means.get(roi, {}).get(metric)
                    if v_base is None or v_follow is None:
                        continue
                    if not np.isfinite(v_base) or not np.isfinite(v_follow):
                        continue
                    records.append({
                        "subject": base_name,
                        "session_type": followup_key,
                        "roi": roi,
                        "tissue_class": get_tissue_class(roi),
                        "metric": metric,
                        "baseline": float(v_base),
                        "followup": float(v_follow),
                    })
    return records


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def _build_legend_handles() -> list:
    handles = []
    labels = {"repeat": "$HC_1$", "reposition": "$HC_2$"}
    for key in ["repeat", "reposition"]:
        color = SUBJECT_COLORS[key]
        name = labels[key]
        followup = key
        handles.append(mlines.Line2D([], [], color=color, linestyle=SESSION_LINESTYLES["baseline"],
                                     linewidth=1.8, label=f"{name} baseline"))
        handles.append(mlines.Line2D([], [], color=color, linestyle=SESSION_LINESTYLES[followup],
                                     linewidth=1.8, label=f"{name} {followup}"))
    handles.append(mlines.Line2D([], [], color="black",
                   linestyle="-.", linewidth=2.0, label="$HC$ mean"))
    handles.append(mpatches.Patch(color="gray", alpha=0.25,
                   label="$HC$ ±1 SD"))
    return handles


def plot_angle_profiles(axes: list, angle_data: dict, selected_subjects: dict) -> None:
    metric_order = ["R1", "R2", "AI"]
    ylabels = [r"$\mathbf{R_1}$ [1/s]", r"$\mathbf{R_2}$ [1/s]", "AI"]
    subplot_labels = get_subplot_labels(3)

    for ax, metric_label, ylabel, sublabel in zip(axes, metric_order, ylabels, subplot_labels):
        # Plot selected HC subjects
        for subj_key, color in SUBJECT_COLORS.items():
            base_name = selected_subjects.get(subj_key)
            if not base_name:
                continue
            sessions = angle_data.get(base_name, {})
            for sess in ["baseline", "repeat", "reposition"]:
                if sess not in sessions:
                    continue
                arr = sessions[sess].get(metric_label)
                if arr is None or np.all(np.isnan(arr)):
                    continue
                ax.plot(ANGLE_BIN_CENTERS, arr, color=color,
                        linestyle=SESSION_LINESTYLES[sess], linewidth=1.5, alpha=0.9)

        # Mean ± std across all baseline subjects
        baseline_arrs = [s.get("baseline", {}).get(metric_label) for s in angle_data.values()
                         if s.get("baseline", {}).get(metric_label) is not None]
        if baseline_arrs:
            stack = np.stack(baseline_arrs, axis=0)
            mean = np.nanmean(stack, axis=0)
            std = np.nanstd(stack, axis=0)
            print(
                f"  {metric_label}: Computing mean ± std from {len(baseline_arrs)} HC baseline subjects")
            ax.plot(ANGLE_BIN_CENTERS, mean, color="black", linewidth=1.5)
            ax.fill_between(ANGLE_BIN_CENTERS, mean - std,
                            mean + std, color="gray", alpha=0.25)

        ax.set_xlim(0, 90)
        ax.set_xticks(np.arange(0, 91, 15))
        ax.set_xlabel(ANGLE_SOURCE_LABEL, fontweight="bold", fontsize=14)
        ax.set_ylabel(ylabel, fontweight="bold", fontsize=14)
        ax.grid(True, linestyle="--", alpha=0.5)

        # Apply fixed y-axis limits if specified
        metric_key = [k for k, v in METRICS.items() if v["label"]
                      == metric_label][0]
        if METRICS[metric_key]["vmin"] is not None and METRICS[metric_key]["vmax"] is not None:
            ax.set_ylim(METRICS[metric_key]["vmin"],
                        METRICS[metric_key]["vmax"])
        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.grid(True, linestyle="--", alpha=0.4)

        add_subplot_label(ax, sublabel)

    # Legend on last axis
    axes[-1].legend(handles=_build_legend_handles(), loc="center left",
                    bbox_to_anchor=(1.05, 0.5), frameon=False, fontsize=12)


def save_abs_data_to_csv(abs_data: list[dict], out_fop: Path) -> None:
    """Save absolute ROI mean values (baseline + follow-up pairs) to CSV."""
    if not abs_data:
        return
    df = pd.DataFrame(abs_data)
    out_fp = out_fop / "roi_absolute_values.csv"
    df.to_csv(out_fp, index=False)
    print(f"Saved ROI absolute values to: {out_fp}")


def _compute_repeatability_arrays(
    recs: list[dict],
    metric: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return paired values, native differences, and plotted differences.

    R1/R2:
        plotted difference = relative difference in percent:
        100 * (followup - baseline) / abs(pair mean)

    AI:
        plotted difference = absolute difference multiplied by 100.
        Because AI is already normalized, this is expressed in percentage
        points, not relative percent.
    """
    baseline = np.array(
        [r["baseline"] for r in recs],
        dtype=float,
    )
    followup = np.array(
        [r["followup"] for r in recs],
        dtype=float,
    )

    mean_vals = (baseline + followup) / 2.0
    native_diff = followup - baseline

    with np.errstate(divide="ignore", invalid="ignore"):
        if metric in {"R1", "R2"}:
            plot_diff = np.where(
                mean_vals != 0,
                100.0 * native_diff / np.abs(mean_vals),
                np.nan,
            )
        elif metric == "AI":
            plot_diff = 100.0 * native_diff
        else:
            raise ValueError(f"Unknown metric: {metric}")

    return baseline, followup, mean_vals, native_diff, plot_diff


def _empty_repeatability_summary() -> dict[str, float]:
    """Return an empty repeatability summary."""
    return {
        "n_pairs": 0,
        "bias_native": np.nan,
        "sd_native": np.nan,
        "loa_lower_native": np.nan,
        "loa_upper_native": np.nan,
        "bias_plot_units": np.nan,
        "sd_plot_units": np.nan,
        "loa_lower_plot_units": np.nan,
        "loa_upper_plot_units": np.nan,
        "icc_a_1": np.nan,
    }


def compute_icc_2way_absolute_single(
    baseline: np.ndarray,
    followup: np.ndarray,
) -> dict[str, float]:
    """Compute all ICC variants from Pingouin for comparison."""
    valid_mask = np.isfinite(baseline) & np.isfinite(followup)
    if valid_mask.sum() < 2:
        return _empty_icc_summary()

    x = np.column_stack([baseline[valid_mask], followup[valid_mask]])
    n, k = x.shape
    if n < 2 or k < 2:
        return _empty_icc_summary()
    if n * k < 5:
        return _empty_icc_summary()

    # The manual ICC(2,1) formula was validated against Pingouin's ICC(A,1)
    # and produced the same results. It is intentionally retained here as a
    # commented reference so that Pingouin is now the sole implementation.
    # grand_mean = float(np.mean(x))
    # mean_subject = np.mean(x, axis=1)
    # mean_rater = np.mean(x, axis=0)
    # ss_subject = k * np.sum((mean_subject - grand_mean) ** 2)
    # ss_rater = n * np.sum((mean_rater - grand_mean) ** 2)
    # ss_error = np.sum(
    #     (x - mean_subject[:, None] - mean_rater[None, :] + grand_mean) ** 2)
    # ms_subject = ss_subject / (n - 1)
    # ms_rater = ss_rater / (k - 1)
    # ms_error = ss_error / ((n - 1) * (k - 1))
    # icc_manual = (ms_subject - ms_error) / (
    #     ms_subject + (k - 1) * ms_error
    #     + k * (ms_rater - ms_error) / n)

    icc_data = pd.DataFrame({
        "subject": np.repeat(np.arange(n), k),
        "measurement": np.tile(np.arange(k), n),
        "score": x.ravel(),
    })
    icc_table = pg.intraclass_corr(
        data=icc_data,
        targets="subject",
        raters="measurement",
        ratings="score",
    ).set_index("Type")

    icc_names = [
        "ICC(1,1)",
        "ICC(A,1)",
        "ICC(C,1)",
        "ICC(1,k)",
        "ICC(A,k)",
        "ICC(C,k)",
    ]
    return {
        name: float(icc_table.loc[name, "ICC"])
        for name in icc_names
    }


def _empty_icc_summary() -> dict[str, float]:
    """Return an empty ICC summary."""
    return {
        "ICC(1,1)": np.nan,
        "ICC(A,1)": np.nan,
        "ICC(C,1)": np.nan,
        "ICC(1,k)": np.nan,
        "ICC(A,k)": np.nan,
        "ICC(C,k)": np.nan,
    }


def _summarize_repeatability_recs(
    recs: list[dict],
    metric: str,
) -> dict[str, float]:
    """Summarize repeatability in native and plotted units."""
    if not recs:
        return _empty_repeatability_summary()

    (
        baseline,
        followup,
        mean_vals,
        native_diff,
        plot_diff,
    ) = _compute_repeatability_arrays(recs, metric)

    valid_mask = (
        np.isfinite(baseline)
        & np.isfinite(followup)
        & np.isfinite(mean_vals)
        & np.isfinite(native_diff)
    )

    baseline = baseline[valid_mask]
    followup = followup[valid_mask]
    mean_vals = mean_vals[valid_mask]
    native_diff = native_diff[valid_mask]
    plot_diff = plot_diff[valid_mask]

    n_pairs = int(native_diff.size)
    if n_pairs == 0:
        return _empty_repeatability_summary()

    bias_native = float(np.mean(native_diff))
    sd_native = float(np.std(native_diff, ddof=1)) if n_pairs > 1 else 0.0
    loa_lower_native = bias_native - 1.96 * sd_native
    loa_upper_native = bias_native + 1.96 * sd_native

    finite_plot_diff = plot_diff[np.isfinite(plot_diff)]
    if finite_plot_diff.size > 0:
        bias_plot = float(np.mean(finite_plot_diff))
        sd_plot = float(np.std(finite_plot_diff, ddof=1)
                        ) if finite_plot_diff.size > 1 else 0.0
        loa_lower_plot = bias_plot - 1.96 * sd_plot
        loa_upper_plot = bias_plot + 1.96 * sd_plot
    else:
        bias_plot = np.nan
        sd_plot = np.nan
        loa_lower_plot = np.nan
        loa_upper_plot = np.nan

    icc_summary = compute_icc_2way_absolute_single(baseline, followup)

    return {
        "n_pairs": n_pairs,
        "bias_native": round(bias_native, 2),
        "sd_native": round(sd_native, 2),
        "loa_lower_native": round(loa_lower_native, 2),
        "loa_upper_native": round(loa_upper_native, 2),
        "bias_plot_units": round(bias_plot, 2),
        "sd_plot_units": round(sd_plot, 2),
        "loa_lower_plot_units": round(loa_lower_plot, 2),
        "loa_upper_plot_units": round(loa_upper_plot, 2),
        "ICC(A,1)": round(icc_summary["ICC(A,1)"], 2),
    }


def _compute_bland_altman_stats(
    recs: list[dict],
    metric: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    float,
    float,
    float,
    float,
]:
    """Compute Bland–Altman values and summary statistics.

    For AI, plotted differences are percentage-point differences.
    """
    (
        _,
        _,
        mean_vals,
        _,
        plot_diff,
    ) = _compute_repeatability_arrays(recs, metric)

    finite_diffs = plot_diff[np.isfinite(plot_diff)]

    if finite_diffs.size == 0:
        return (
            mean_vals,
            finite_diffs,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )

    bias = float(np.mean(finite_diffs))
    sd = (
        float(np.std(finite_diffs, ddof=1))
        if finite_diffs.size > 1
        else 0.0
    )

    loa_lower = bias - 1.96 * sd
    loa_upper = bias + 1.96 * sd

    return (
        mean_vals,
        finite_diffs,
        bias,
        sd,
        loa_upper,
        loa_lower,
    )


def plot_bland_altman(
    ax,
    abs_data: list[dict],
    metric: str,
    subplot_label: str,
) -> None:
    """Create a Bland–Altman panel for one metric.

    R1/R2 are shown using relative differences in percent.
    AI is shown using absolute differences in percentage points because
    AI is already normalized.
    """
    recs = [
        r for r in abs_data
        if r["metric"] == metric
    ]

    if not recs:
        ax.text(
            0.5,
            0.5,
            "No data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return

    (
        mean_vals,
        diff_vals,
        bias,
        _,
        loa_upper,
        loa_lower,
    ) = _compute_bland_altman_stats(
        recs,
        metric,
    )

    for rec, mean_v, diff_v in zip(
        recs,
        mean_vals,
        diff_vals,
    ):
        if not np.isfinite(diff_v):
            continue

        color = ROI_COLORS.get(
            rec["roi"],
            "gray",
        )
        marker = SESSION_MARKERS.get(
            rec["session_type"],
            "o",
        )

        ax.scatter(
            mean_v,
            diff_v,
            color=color,
            marker=marker,
            s=50,
            zorder=3,
            edgecolors="black",
            linewidths=0.6,
            alpha=0.9,
        )

    if diff_vals.size > 0:
        ax.axhline(
            bias,
            color="black",
            linestyle="-",
            linewidth=1.2,
        )
        ax.axhline(
            loa_upper,
            color="black",
            linestyle="--",
            linewidth=1.0,
        )
        ax.axhline(
            loa_lower,
            color="black",
            linestyle="--",
            linewidth=1.0,
        )
        ax.axhline(
            0,
            color="gray",
            linestyle=":",
            linewidth=0.8,
            zorder=1,
        )

    if metric == "R1":
        xlabel = r"Mean $\mathbf{R_1}$ [1/s]"
        ylabel = r"$\mathbf{\Delta R_1}$ rel. [%]"
    elif metric == "R2":
        xlabel = r"Mean $\mathbf{R_2}$ [1/s]"
        ylabel = r"$\mathbf{\Delta R_2}$ rel. [%]"
    else:
        xlabel = "Mean AI"
        ylabel = r"$\mathbf{\Delta}$AI [percentage points]"

    ax.set_xlabel(
        xlabel,
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylabel(
        ylabel,
        fontsize=14,
        fontweight="bold",
    )
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=12,
    )
    ax.grid(
        True,
        linestyle="--",
        alpha=0.4,
    )
    add_subplot_label(
        ax,
        subplot_label,
    )


def build_repeatability_summary_table(
    abs_data: list[dict],
    group_cols: list[str],
    scope_name: str,
) -> pd.DataFrame:
    """Build a repeatability summary table for a chosen aggregation level."""
    if not abs_data:
        return pd.DataFrame()

    df = pd.DataFrame(abs_data)
    key_cols = group_cols + ["metric"]
    rows: list[dict] = []

    for keys, group_df in df.groupby(key_cols, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = {"summary_scope": scope_name}
        for col, value in zip(key_cols, keys):
            row[col] = value

        stats = _summarize_repeatability_recs(
            group_df.to_dict("records"), row["metric"])
        row.update(stats)
        rows.append(row)

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        return out_df

    if "metric" in out_df.columns:
        out_df["metric"] = pd.Categorical(out_df["metric"], categories=[
                                          "R1", "R2", "AI"], ordered=True)
    if "session_type" in out_df.columns:
        out_df["session_type"] = pd.Categorical(out_df["session_type"], categories=[
                                                "repeat", "reposition", "pooled"], ordered=True)
    if "roi" in out_df.columns:
        roi_order = [roi for roi in ROI_ORDER if roi in set(
            out_df["roi"].astype(str))]
        remaining_roi = sorted(r for r in out_df["roi"].astype(
            str).unique() if r not in roi_order)
        out_df["roi"] = pd.Categorical(out_df["roi"].astype(
            str), categories=roi_order + remaining_roi, ordered=True)
    if "tissue_class" in out_df.columns:
        tissue_order = [tc for tc in TISSUE_CLASS_ORDER if tc in set(
            out_df["tissue_class"].astype(str))]
        remaining_tissue = sorted(tc for tc in out_df["tissue_class"].astype(
            str).unique() if tc not in tissue_order)
        out_df["tissue_class"] = pd.Categorical(out_df["tissue_class"].astype(
            str), categories=tissue_order + remaining_tissue, ordered=True)

    sort_cols = [
        col for col in group_cols if col in out_df.columns] + ["metric"]
    out_df = out_df.sort_values(sort_cols).reset_index(drop=True)
    for col in ["metric", "session_type", "roi", "tissue_class"]:
        if col in out_df.columns:
            out_df[col] = out_df[col].astype(str)
    return out_df


def save_repeatability_summaries(abs_data: list[dict], out_fop: Path) -> None:
    """Save pooled, ROI-level, and tissue-class repeatability summaries."""
    if not abs_data:
        return

    pooled_df = build_repeatability_summary_table(
        abs_data, [], "pooled_metric")
    pooled_fp = out_fop / "bland_altman_summary.csv"
    pooled_df.to_csv(pooled_fp, index=False)
    print(f"Saved Bland-Altman summary to: {pooled_fp}")

    session_df = build_repeatability_summary_table(
        abs_data, ["session_type"], "by_session")
    session_fp = out_fop / "repeatability_summary_by_session.csv"
    session_df.to_csv(session_fp, index=False)
    print(f"Saved repeatability-by-session summary to: {session_fp}")

    roi_df = build_repeatability_summary_table(
        abs_data, ["session_type", "roi"], "by_roi")
    roi_fp = out_fop / "repeatability_summary_by_roi.csv"
    roi_df.to_csv(roi_fp, index=False)
    print(f"Saved repeatability-by-ROI summary to: {roi_fp}")

    tissue_df = build_repeatability_summary_table(
        abs_data, ["session_type", "tissue_class"], "by_tissue_class")
    tissue_fp = out_fop / "repeatability_summary_by_tissue_class.csv"
    tissue_df.to_csv(tissue_fp, index=False)
    print(f"Saved repeatability-by-tissue-class summary to: {tissue_fp}")


def _build_bland_altman_legend_handles() -> list:
    """Legend handles for Bland-Altman plots."""
    handles = []
    for roi in reversed(ROI_ORDER):
        color = ROI_COLORS.get(roi)
        if color is None:
            continue
        handles.append(mpatches.Patch(facecolor=color, edgecolor="black",
                                      linewidth=0.5, label=roi))
    handles.append(mlines.Line2D([], [], color="gray", marker="o", linestyle="None",
                                 markeredgecolor="black", markeredgewidth=0.6,
                                 markersize=6, label="Repeat"))
    handles.append(mlines.Line2D([], [], color="gray", marker="^", linestyle="None",
                                 markeredgecolor="black", markeredgewidth=0.6,
                                 markersize=6, label="Reposition"))
    handles.append(mlines.Line2D([], [], color="black", linestyle="-", linewidth=1.2,
                                 alpha=0.9, label="Bias"))
    handles.append(mlines.Line2D([], [], color="black", linestyle="--", linewidth=1.0,
                                 alpha=0.8, label="±1.96 SD (LoA)"))
    return handles


def _slice_2d(arr3d: np.ndarray, plane: str, idx: int) -> np.ndarray:
    """Extract and orient a 2D slice from a 3D volume."""
    if plane == "axial":
        sl = arr3d[:, :, idx]
    elif plane == "coronal":
        sl = arr3d[:, idx, :]
    elif plane == "sagittal":
        sl = arr3d[idx, :, :]
    else:
        raise ValueError(f"Unknown plane: {plane}")
    return np.rot90(sl)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def create_figure() -> plt.Figure:
    apply_figure_style()

    exp_name = Path(__file__).stem
    out_fop = get_paper_figure_cache_dir(exp_name)

    print("Loading subject mappings...")
    subject_map = get_subject_id_mapping()

    print(f"Using R1 Bounds for WM: {WM_LIMITS['R1']}")
    print(f"Using R2 Bounds for WM: {WM_LIMITS['R2']}")
    print(f"Using R1 Bounds for GM: {GM_LIMITS['R1']}")
    print(f"Using R2 Bounds for GM: {GM_LIMITS['R2']}")

    print("Preparing angle-binned data...")
    angle_data = prepare_angle_data(out_fop, subject_map)
    selected = get_baseline_subjects()
    save_angle_relative_differences(angle_data, selected, out_fop)

    print("Preparing ROI absolute values...")
    abs_data = compute_roi_absolute_values(subject_map)

    # Save absolute ROI values to CSV
    save_abs_data_to_csv(abs_data, out_fop)
    save_repeatability_summaries(abs_data, out_fop)

    print("Creating figure...")
    fig = make_figure(height=10, width=14)
    gs = GridSpec(2, 3, figure=fig, wspace=0.3, hspace=0.25)

    # Row 1: angle profiles (A, B, C)
    ax_r1 = fig.add_subplot(gs[0, 0])
    ax_r2 = fig.add_subplot(gs[0, 1])
    ax_ai = fig.add_subplot(gs[0, 2])
    plot_angle_profiles([ax_r1, ax_r2, ax_ai], angle_data, selected)

    # Row 2: Bland-Altman (D, E, F) — one panel per metric, all subjects combined
    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1])
    ax_f = fig.add_subplot(gs[1, 2])
    for ax, metric, label in zip(
        [ax_d, ax_e, ax_f], ["R1", "R2",
                             "AI"], get_subplot_labels(3, start="d")
    ):
        plot_bland_altman(ax, abs_data, metric, label)

    # Shared Bland-Altman legend on rightmost panel
    ax_f.legend(handles=_build_bland_altman_legend_handles(), loc="center left",
                bbox_to_anchor=(1.05, 0.5), frameon=False, fontsize=12)

    out_fp = PAPER_FIG_DIR / f"{exp_name}.svg"
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fp, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out_fp}")
    plt.close(fig)
    return fig


if __name__ == "__main__":
    print("=" * 80)
    print("FIGURE 2: HC reproducibility - orientation dependence + Bland-Altman")
    print("=" * 80)
    create_figure()
    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80 + "\n")
