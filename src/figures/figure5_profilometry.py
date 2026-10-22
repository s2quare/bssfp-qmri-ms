"""
Figure 5: Bundle profilometry: R1, R2, AI profiles along WM bundles for HC vs RRMS.

For each bundle and metric, generates a figure showing the mean ± std
profile along the bundle for HC (blue) vs RRMS (orange) groups.

Uses tractometry_flow output from 4_tractometry_results.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
import yaml

sys.path.append(str(Path(__file__).parent.parent))

try:
    from utils import paths
    from utils.paper_figure_helpers import (
        build_subject_cerebral_wm_mask,
        load_subject_metric,
    )
    from utils.paper_meta import *
    from utils.plotting import apply_figure_style
except ImportError as exc:
    raise RuntimeError(f"Could not import project utilities: {exc}") from exc

# Tractometry paths
TRACTOMETRY_RESULTS = paths.derivatives / "4_tractometry_results"

# Metrics to plot (matching tractometry JSON naming)
METRICS = {
    "R1_metric": {"label": r"$\mathbf{R_1}$ [1/s]", "short": "R1"},
    "R2_metric": {"label": r"$\mathbf{R_2}$ [1/s]", "short": "R2"},
    "AI_metric": {"label": "AI", "short": "AI"},
}

# FODF metrics (plotted separately without error bands)
FODF_METRIC = "NUFO_metric"
THETA_METRIC = "THETA1_metric"  # Fiber-to-field angle
FA_METRIC = "FA_metric"  # Fractional Anisotropy

# Group styling
GROUPS = {
    "HC": {"color": "tab:blue", "alpha": 0.9},
    "RRMS": {"color": "tab:orange", "alpha": 0.85},
}
FILL_ALPHA = 0.2

# Metadata path
METADATA_FP = paths.data / "251217-b_get_gender_age_participants" / \
    "251217-b_participant_metadata.csv"

PROFILE_METRICS = {
    "R1_metric": "t1_pinn",
    "R2_metric": "t2_pinn",
    "AI_metric": "ai_map.unringed",
    "NUFO_metric": "nufo",
    "THETA1_metric": "theta_1",
    "FA_metric": "fa",
}
N_BUNDLE_SEGMENTS = 20
_BUNDLE_PROFILE_CACHE: dict[str, dict] = {}
PROFILE_CACHE_DIR = paths.data / "paper_figures" / Path(__file__).stem / "subject_profiles"

with open(paths.default_config, "r", encoding="utf-8") as config_file:
    _CONFIG = yaml.safe_load(config_file)
R1_WM_LIMITS = (_CONFIG["r1_mask_wm_lower"], _CONFIG["r1_mask_wm_upper"])
R2_WM_LIMITS = (_CONFIG["r2_mask_wm_lower"], _CONFIG["r2_mask_wm_upper"])


# =============================================================================
# Subject loading
# =============================================================================


# =============================================================================
# Data loading
# =============================================================================


def _profile_cache_path(sid: str) -> Path:
    return PROFILE_CACHE_DIR / f"{sid}_bundle_profiles.csv"


def _load_cached_bundle_profiles(sid: str) -> dict | None:
    cache_fp = _profile_cache_path(sid)
    if not cache_fp.exists():
        return None

    try:
        cached = pd.read_csv(cache_fp)
        required_columns = {"bundle", "metric", "segment", "mean"}
        if not required_columns.issubset(cached.columns):
            return None

        profiles: dict[str, dict] = {}
        for row in cached.itertuples(index=False):
            bundle_profiles = profiles.setdefault(row.bundle, {})
            metric_values = bundle_profiles.setdefault(row.metric, {})
            metric_values[f"{int(row.segment):03d}"] = {
                "mean": float(row.mean)
            }
        return profiles
    except (OSError, ValueError, TypeError):
        return None


def _save_bundle_profiles(sid: str, profiles: dict[str, dict]) -> None:
    rows = []
    for bundle, bundle_profiles in profiles.items():
        for metric, metric_values in bundle_profiles.items():
            for segment, values in metric_values.items():
                rows.append(
                    {
                        "subject_id": sid,
                        "bundle": bundle,
                        "metric": metric,
                        "segment": int(segment),
                        "mean": values["mean"],
                    }
                )

    PROFILE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        rows,
        columns=["subject_id", "bundle", "metric", "segment", "mean"],
    ).to_csv(_profile_cache_path(sid), index=False)


def load_bundle_profiles(sid: str) -> dict | None:
    """Calculate bundle-segment means from label maps and subject-space maps.

    Each bundle label map contains labels 1--20. Values are calculated after
    intersecting each segment with the subject's refined cerebral-WM mask.
    """
    if sid in _BUNDLE_PROFILE_CACHE:
        return _BUNDLE_PROFILE_CACHE[sid]

    cached_profiles = _load_cached_bundle_profiles(sid)
    if cached_profiles is not None:
        _BUNDLE_PROFILE_CACHE[sid] = cached_profiles
        return cached_profiles

    refined_wm = build_subject_cerebral_wm_mask(
        sid, r1_limits=R1_WM_LIMITS, r2_limits=R2_WM_LIMITS, exclude_lesions=True
    )
    if refined_wm is None:
        return None

    metric_maps = {
        profile_key: load_subject_metric(metric_name, sid)
        for profile_key, metric_name in PROFILE_METRICS.items()
    }
    metric_maps = {
        key: data for key, data in metric_maps.items() if data is not None
    }
    if not metric_maps:
        return None

    label_dir = TRACTOMETRY_RESULTS / sid / "Bundle_Label_And_Distance_Maps"
    profiles: dict[str, dict] = {}
    for label_fp in sorted(label_dir.glob(f"{sid}__*_labels.nii.gz")):
        bundle = label_fp.name.removeprefix(f"{sid}__").removesuffix(
            "_labels.nii.gz"
        )
        label_data = nib.load(str(label_fp)).get_fdata()
        if label_data.shape != refined_wm.shape:
            continue

        bundle_profiles = {metric: {} for metric in metric_maps}
        for segment in range(1, N_BUNDLE_SEGMENTS + 1):
            segment_mask = refined_wm & (label_data == segment)
            point_key = f"{segment:03d}"
            for metric, metric_data in metric_maps.items():
                if metric_data.shape != segment_mask.shape:
                    continue
                values = metric_data[segment_mask]
                bundle_profiles[metric][point_key] = {
                    "mean": float(np.nanmean(values)) if values.size else np.nan
                }
        profiles[bundle] = bundle_profiles

    _save_bundle_profiles(sid, profiles)
    _BUNDLE_PROFILE_CACHE[sid] = profiles
    return profiles


def extract_profile_means(bundle_data: dict, metric: str) -> np.ndarray | None:
    """Extract mean values for a metric across all points in a bundle.

    Returns array of shape (n_points,) or None if metric not found.
    """
    if metric not in bundle_data:
        return None

    metric_data = bundle_data[metric]
    # Points are numbered as strings: "001", "002", ...
    points = sorted(metric_data.keys())
    means = [metric_data[pt]["mean"] for pt in points]
    return np.array(means)


def aggregate_profiles(sids: list[str], bundle: str, metric: str
                       ) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Aggregate profiles across subjects for a bundle/metric.

    Returns (mean_across_subjects, std_across_subjects) arrays.
    """
    all_profiles = []

    for sid in sids:
        subject_data = load_bundle_profiles(sid)
        if subject_data is None:
            continue
        if bundle not in subject_data:
            continue

        profile = extract_profile_means(subject_data[bundle], metric)
        if profile is not None:
            all_profiles.append(profile)

    if not all_profiles:
        return None, None

    # Stack and compute mean/std across subjects
    # Handle potentially different lengths by padding with nan
    max_len = max(len(p) for p in all_profiles)
    padded = np.full((len(all_profiles), max_len), np.nan)
    for i, p in enumerate(all_profiles):
        padded[i, :len(p)] = p

    mean_profile = np.nanmean(padded, axis=0)
    std_profile = np.nanstd(padded, axis=0)

    return mean_profile, std_profile


def get_subject_bundle_means(sids: list[str], bundle: str, metric: str
                             ) -> np.ndarray:
    """Return one across-segment mean per subject for a bundle and metric."""
    subject_means = []

    for sid in sids:
        subject_data = load_bundle_profiles(sid)
        if subject_data is None or bundle not in subject_data:
            continue

        profile = extract_profile_means(subject_data[bundle], metric)
        if profile is None:
            continue

        subject_mean = np.nanmean(profile)
        if np.isfinite(subject_mean):
            subject_means.append(subject_mean)

    return np.asarray(subject_means, dtype=float)


def bh_fdr(p_values: pd.Series, q: float = 0.05
           ) -> tuple[np.ndarray, np.ndarray]:
    """Apply Benjamini-Hochberg FDR correction to p-values."""
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full(p.shape, np.nan)
    reject = np.zeros(p.shape, dtype=bool)

    finite = np.isfinite(p)
    if not finite.any():
        return adjusted, reject

    order = np.argsort(p[finite])
    ranked = p[finite][order]
    corrected = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    corrected = np.minimum.accumulate(corrected[::-1])[::-1]
    corrected = np.clip(corrected, 0.0, 1.0)

    reordered = np.empty_like(corrected)
    reordered[order] = corrected
    adjusted[finite] = reordered
    reject[finite] = adjusted[finite] < q
    return adjusted, reject


def get_available_bundles() -> list[str]:
    """Get list of bundles from the first available subject."""
    for d in sorted(TRACTOMETRY_RESULTS.iterdir()):
        if d.is_dir() and d.name.startswith("S"):
            data = load_bundle_profiles(d.name)
            if data:
                return sorted(data.keys())
    return []


def load_lesion_load_per_point(sid: str) -> dict | None:
    """Load lesion load per point JSON for a subject.

    Returns dict: bundle -> lesion_total_volume -> point_idx -> volume
    """
    json_fp = (TRACTOMETRY_RESULTS / sid / "Lesion_Load" /
               f"{sid}__lesion_load_per_point.json")
    if not json_fp.exists():
        return None

    with open(json_fp, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data.get(sid, {})


def load_volume_per_label(sid: str) -> dict | None:
    """Load bundle volume per label JSON for a subject.

    Returns dict: bundle_labels -> volume -> point_idx -> volume
    """
    json_fp = (TRACTOMETRY_RESULTS / sid / "Bundle_Volume_Per_Label" /
               f"{sid}__volume_per_label.json")
    if not json_fp.exists():
        return None

    with open(json_fp, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data.get(sid, {})


def extract_lesion_load_profile(lesion_data: dict, bundle: str) -> np.ndarray | None:
    """Extract lesion_total_volume for a bundle across all points."""
    if bundle not in lesion_data:
        return None
    if "lesion_total_volume" not in lesion_data[bundle]:
        return None

    vol_data = lesion_data[bundle]["lesion_total_volume"]
    points = sorted(vol_data.keys())
    return np.array([vol_data[pt] for pt in points])


def extract_bundle_volume_profile(volume_data: dict, bundle: str) -> np.ndarray | None:
    """Extract volume per label for a bundle across all points.

    Note: volume_per_label uses bundle names with '_labels' suffix.
    """
    labels_key = f"{bundle}_labels"
    if labels_key not in volume_data:
        return None
    if "volume" not in volume_data[labels_key]:
        return None

    vol_data = volume_data[labels_key]["volume"]
    points = sorted(vol_data.keys())
    return np.array([vol_data[pt] for pt in points])


def aggregate_relative_lesion_load(sids: list[str], bundle: str
                                   ) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Aggregate relative lesion load (%) across RRMS subjects for a bundle.

    Relative lesion load = (lesion_vol / tract_point_vol) * 100

    Returns (mean_across_subjects, std_across_subjects) arrays.
    """
    all_profiles = []

    for sid in sids:
        lesion_data = load_lesion_load_per_point(sid)
        volume_data = load_volume_per_label(sid)

        if lesion_data is None or volume_data is None:
            continue

        lesion_vol = extract_lesion_load_profile(lesion_data, bundle)
        tract_vol = extract_bundle_volume_profile(volume_data, bundle)

        if lesion_vol is None or tract_vol is None:
            continue

        # Ensure same length
        min_len = min(len(lesion_vol), len(tract_vol))
        lesion_vol = lesion_vol[:min_len]
        tract_vol = tract_vol[:min_len]

        # Calculate relative lesion load (%)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_load = (lesion_vol / tract_vol) * 100.0
            rel_load[~np.isfinite(rel_load)] = 0.0

        all_profiles.append(rel_load)

    if not all_profiles:
        return None, None

    # Stack and compute mean/std across subjects
    max_len = max(len(p) for p in all_profiles)
    padded = np.full((len(all_profiles), max_len), np.nan)
    for i, p in enumerate(all_profiles):
        padded[i, :len(p)] = p

    mean_profile = np.nanmean(padded, axis=0)
    std_profile = np.nanstd(padded, axis=0)

    return mean_profile, std_profile


# =============================================================================
# Plotting
# =============================================================================

def plot_bundle_all_metrics(bundle: str,
                            hc_means: dict[str, np.ndarray],
                            hc_stds: dict[str, np.ndarray],
                            rrms_means: dict[str, np.ndarray],
                            rrms_stds: dict[str, np.ndarray],
                            nufo_hc_mean: np.ndarray | None,
                            nufo_rrms_mean: np.ndarray | None,
                            theta_hc_mean: np.ndarray | None,
                            theta_rrms_mean: np.ndarray | None,
                            lesion_load_mean: np.ndarray | None,
                            fa_hc_mean: np.ndarray | None,
                            fa_rrms_mean: np.ndarray | None,
                            out_dir: Path) -> Path:
    """Plot all three metrics + supplementary data for a bundle in a 4x1 figure.

    Subplots:
    - R1, R2, AI (top 3): HC/RRMS profiles with HC/RRMS legends
    - Additional metrics (bottom): theta_1, NuFO, FA (HC solid, RRMS dashed)
      and lesion load
    """
    apply_figure_style()

    fig, axes = plt.subplots(4, 1, figsize=(7, 10), sharex=True)

    metric_keys = ["R1_metric", "R2_metric", "AI_metric"]

    # Plot the three main metrics
    for idx, (metric_key, ax) in enumerate(zip(metric_keys, axes[:3])):
        metric_info = METRICS[metric_key]
        hc_mean = hc_means[metric_key]
        hc_std = hc_stds[metric_key]
        rrms_mean = rrms_means[metric_key]
        rrms_std = rrms_stds[metric_key]

        n_points = len(hc_mean)
        x = np.arange(1, n_points + 1)

        # HC profile
        ax.plot(x, hc_mean, color=GROUPS["HC"]["color"], linewidth=2,
                alpha=GROUPS["HC"]["alpha"], label="HC")
        ax.fill_between(x, hc_mean - hc_std, hc_mean + hc_std,
                        color=GROUPS["HC"]["color"], alpha=FILL_ALPHA)

        # RRMS profile
        ax.plot(x, rrms_mean, color=GROUPS["RRMS"]["color"], linewidth=2,
                alpha=GROUPS["RRMS"]["alpha"], label="RRMS")
        ax.fill_between(x, rrms_mean - rrms_std, rrms_mean + rrms_std,
                        color=GROUPS["RRMS"]["color"], alpha=FILL_ALPHA)

        ax.set_ylabel(metric_info["label"], fontsize=13)
        ax.tick_params(axis="both", which="major", labelsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_xlim(1, n_points)

        # Add legend on the top subplot
        if idx == 0:
            ax.legend(loc="best", frameon=False, fontsize=9)

    # Plot supplementary metrics on the 4th subplot
    ax_supp = axes[3]
    n_points = len(hc_means["R1_metric"])
    x = np.arange(1, n_points + 1)

    # Plot FA (purple, left outer y-axis, offset further left)
    ax_fa = None
    if ((fa_hc_mean is not None and len(fa_hc_mean) == n_points) or
            (fa_rrms_mean is not None and len(fa_rrms_mean) == n_points)):
        ax_fa = ax_supp.twinx()
        ax_fa.spines["left"].set_position(("axes", -0.18))
        ax_fa.spines["left"].set_visible(True)
        ax_fa.spines["right"].set_visible(False)
        ax_fa.yaxis.set_label_position("left")
        ax_fa.yaxis.set_ticks_position("left")

        if fa_hc_mean is not None and len(fa_hc_mean) == n_points:
            ax_fa.plot(x, fa_hc_mean, color="purple", linewidth=1.5,
                       label="FA HC", linestyle="-")
        if fa_rrms_mean is not None and len(fa_rrms_mean) == n_points:
            ax_fa.plot(x, fa_rrms_mean, color="purple", linewidth=1.5,
                       label="FA RRMS", linestyle="--")

        ax_fa.set_ylabel("FA", fontsize=12, color="purple")
        ax_fa.tick_params(axis="both", which="major",
                          labelsize=11, labelcolor="purple")
        ax_fa.set_ylim(0, 1)
        ax_fa.set_yticks([0, 0.5, 1])

    # Plot theta_1 (green, left y-axis)
    if ((theta_hc_mean is not None and len(theta_hc_mean) == n_points) or
            (theta_rrms_mean is not None and len(theta_rrms_mean) == n_points)):
        ax_theta = ax_supp

        if theta_hc_mean is not None and len(theta_hc_mean) == n_points:
            ax_theta.plot(x, theta_hc_mean, color="green", linewidth=1.5,
                          label=r"$\theta$ HC", linestyle="-")
        if theta_rrms_mean is not None and len(theta_rrms_mean) == n_points:
            ax_theta.plot(x, theta_rrms_mean, color="green", linewidth=1.5,
                          label=r"$\theta$ RRMS", linestyle="--")

        ax_theta.set_ylabel(r"$\theta$ [°]", fontsize=12, color="green")
        ax_theta.tick_params(axis="both", which="major",
                             labelsize=11, labelcolor="green")
        ax_theta.set_ylim(0, 90)
        ax_theta.set_yticks([0, 30, 60, 90])

    # Plot NuFO (black, right y-axis)
    ax_nufo = None
    if ((nufo_hc_mean is not None and len(nufo_hc_mean) == n_points) or
            (nufo_rrms_mean is not None and len(nufo_rrms_mean) == n_points)):
        ax_nufo = ax_supp.twinx()

        if nufo_hc_mean is not None and len(nufo_hc_mean) == n_points:
            ax_nufo.plot(x, nufo_hc_mean, color="black", linewidth=1.5,
                         label="NuFO HC", linestyle="-")
        if nufo_rrms_mean is not None and len(nufo_rrms_mean) == n_points:
            ax_nufo.plot(x, nufo_rrms_mean, color="black", linewidth=1.5,
                         label="NuFO RRMS", linestyle="--")

        ax_nufo.set_ylabel("NuFO", fontsize=12, color="black")
        ax_nufo.tick_params(axis="both", which="major",
                            labelsize=11, labelcolor="black")
        ax_nufo.set_ylim(0.5, 2.5)
        ax_nufo.set_yticks([1, 2])

    # Plot lesion load (red, right outer y-axis)
    ax_lesion = None
    if lesion_load_mean is not None and len(lesion_load_mean) == n_points:
        ax_lesion = ax_supp.twinx()
        # Offset lesion axis to the right
        ax_lesion.spines["right"].set_position(("axes", 1.15))
        ax_lesion.spines["right"].set_visible(True)
        ax_lesion.yaxis.set_label_position("right")
        ax_lesion.yaxis.set_ticks_position("right")

        ax_lesion.plot(x, lesion_load_mean, color="red", linewidth=1.5,
                       label="Lesion load", linestyle=":")
        ax_lesion.set_ylabel("Lesion load [%]", fontsize=12, color="red")
        ax_lesion.set_ylim(0, 4)
        ax_lesion.tick_params(axis="both", which="major",
                              labelsize=11, labelcolor="red")
        ax_lesion.set_yticks([0, 2, 4])

    # Grid and labels for supplementary subplot
    ax_supp.grid(True, linestyle="--", alpha=0.5)
    ax_supp.set_xlim(1, n_points)
    ax_supp.tick_params(axis="both", which="major", labelsize=11)

    # Create unified legend for the supplementary subplot
    handles = []
    labels = []

    # Get FA from offset left axis
    if ax_fa is not None:
        h, l = ax_fa.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

    # Get theta from primary axis
    h, l = ax_supp.get_legend_handles_labels()
    handles.extend(h)
    labels.extend(l)

    # Get NuFO if it exists
    if ax_nufo is not None:
        h, l = ax_nufo.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

    # Get lesion load if it exists
    if ax_lesion is not None:
        h, l = ax_lesion.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

    # Create unified legend
    ax_supp.legend(handles, labels, loc="best",
                   frameon=False, fontsize=9, ncol=2)

    # Set xlabel only on the bottom subplot
    if bundle == "CST_L" or bundle == "CST_R":
        ax_supp.set_xlabel(
            "Streamline direction: Inferior - Superior", fontsize=13)
        ax_supp.set_xticks(np.arange(0, 21, 4))
        ax_supp.set_xticklabels([str(int(x)) for x in np.arange(0, 21, 4)])
    elif bundle == "SLF1_L" or bundle == "SLF1_R":
        ax_supp.set_xlabel(
            "Streamline direction: Posterior - Anterior", fontsize=13)
        ax_supp.set_xticks(np.arange(0, 21, 4))
        ax_supp.set_xticklabels([str(int(x)) for x in np.arange(0, 21, 4)])
    else:
        ax_supp.set_xlabel("Location along the streamline", fontsize=13)
        ax_supp.set_xticks(np.arange(0, 21, 4))
        ax_supp.set_xticklabels([str(int(x)) for x in np.arange(0, 21, 4)])

    # Add main title
    fig.suptitle(bundle, fontsize=13, fontweight="bold", y=0.995)

    # Tight spacing between subplots
    fig.subplots_adjust(hspace=0.35, left=0.22, right=0.85,
                        top=0.96, bottom=0.08)

    out_fp = out_dir / f"{bundle}_combined.svg"
    fig.savefig(out_fp, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_fp


# =============================================================================
# Main
# =============================================================================

def create_figure() -> None:
    print("=" * 60)
    print("BUNDLE PROFILOMETRY: R1, R2, AI along WM bundles (HC vs RRMS)")
    print("=" * 60)

    # Setup output directory
    exp_name = Path(__file__).stem
    out_dir = paths.root / "fig" / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Get subject mappings and metadata
    print("\nLoading subject mappings...")
    subject_map = get_subject_id_mapping()
    metadata = pd.read_csv(METADATA_FP)

    # Get HC subjects (baseline only)
    hc_sids = get_hc_baseline_subject_ids(subject_map)
    hc_metadata = metadata[metadata["subject_id"].isin(hc_sids)]
    hc_ages = hc_metadata["age"].dropna()

    # Print HC statistics
    print(
        f"\nHC: n={len(hc_sids)}, age range: {hc_ages.min():.1f}-{hc_ages.max():.1f} years")
    print(f"HC: age mean={hc_ages.mean():.1f}, age std={hc_ages.std():.1f}")
    print(f"HC subject IDs: {hc_sids}")

    # Get RRMS subjects (age-capped by max HC age using shared helper)
    rrms_sids = get_age_matched_rrms_subject_ids(subject_map, hc_sids)
    rrms_metadata = metadata[metadata["subject_id"].isin(rrms_sids)]
    rrms_ages = rrms_metadata["age"].dropna()

    # Print RRMS statistics
    print(
        f"RRMS: n={len(rrms_sids)}, age range: {rrms_ages.min():.1f}-{rrms_ages.max():.1f} years")
    print(
        f"RRMS: age mean={rrms_ages.mean():.1f}, age std={rrms_ages.std():.1f}")
    print(f"RRMS subject IDs: {rrms_sids}")

    # Get available bundles
    bundles = get_available_bundles()
    if not bundles:
        print("ERROR: No bundles found in tractometry results.")
        return

    print(f"\nFound {len(bundles)} bundles: {bundles}")

    # Generate figures for each bundle (all metrics combined)
    print("\nGenerating figures...")
    generated = 0
    segment_summary_rows: list[dict] = []

    for bundle in bundles:
        # Collect all metric data for this bundle
        hc_means = {}
        hc_stds = {}
        rrms_means = {}
        rrms_stds = {}
        has_data = True
        bundle_segment_rows: list[dict] = []

        for metric_key in METRICS.keys():
            # Aggregate profiles
            hc_mean, hc_std = aggregate_profiles(hc_sids, bundle, metric_key)
            rrms_mean, rrms_std = aggregate_profiles(
                rrms_sids, bundle, metric_key)

            if hc_mean is None or rrms_mean is None:
                print(
                    f"  Skipping {bundle}: insufficient data for {METRICS[metric_key]['short']}")
                has_data = False
                break

            # Ensure same length (use minimum)
            min_len = min(len(hc_mean), len(rrms_mean))
            hc_means[metric_key] = hc_mean[:min_len]
            hc_stds[metric_key] = hc_std[:min_len]
            rrms_means[metric_key] = rrms_mean[:min_len]
            rrms_stds[metric_key] = rrms_std[:min_len]

            # Store cohort-level segment means/stds and differences for CSV export.
            metric_short = METRICS[metric_key]["short"]
            for seg_idx in range(min_len):
                hc_val = float(hc_means[metric_key][seg_idx])
                rrms_val = float(rrms_means[metric_key][seg_idx])
                diff = rrms_val - hc_val
                bundle_segment_rows.append(
                    {
                        "bundle": bundle,
                        "metric": metric_short,
                        "segment_index": seg_idx + 1,
                        "hc_mean": hc_val,
                        "hc_std": float(hc_stds[metric_key][seg_idx]),
                        "rrms_mean": rrms_val,
                        "rrms_std": float(rrms_stds[metric_key][seg_idx]),
                        "rrms_minus_hc": diff,
                        "abs_rrms_minus_hc": abs(diff),
                    }
                )

        if not has_data:
            continue

        # Keep only complete bundles (all requested metrics available).
        segment_summary_rows.extend(bundle_segment_rows)

        metric_len = len(hc_means["R1_metric"])

        # Load NuFO data (mean only for HC and age-matched RRMS)
        hc_nufo_mean, _ = aggregate_profiles(hc_sids, bundle, FODF_METRIC)
        rrms_nufo_mean, _ = aggregate_profiles(rrms_sids, bundle, FODF_METRIC)
        nufo_hc_mean = None
        nufo_rrms_mean = None

        if hc_nufo_mean is not None and len(hc_nufo_mean) >= metric_len:
            nufo_hc_mean = hc_nufo_mean[:metric_len]
        if rrms_nufo_mean is not None and len(rrms_nufo_mean) >= metric_len:
            nufo_rrms_mean = rrms_nufo_mean[:metric_len]

        # Load theta_1 data (mean only for HC and age-matched RRMS)
        hc_theta_mean, _ = aggregate_profiles(hc_sids, bundle, THETA_METRIC)
        rrms_theta_mean, _ = aggregate_profiles(
            rrms_sids, bundle, THETA_METRIC)
        theta_hc_mean = None
        theta_rrms_mean = None

        if hc_theta_mean is not None and len(hc_theta_mean) >= metric_len:
            theta_hc_mean = hc_theta_mean[:metric_len]
        if rrms_theta_mean is not None and len(rrms_theta_mean) >= metric_len:
            theta_rrms_mean = rrms_theta_mean[:metric_len]

        # Load FA data (mean only for HC and age-matched RRMS)
        hc_fa_mean, _ = aggregate_profiles(hc_sids, bundle, FA_METRIC)
        rrms_fa_mean, _ = aggregate_profiles(rrms_sids, bundle, FA_METRIC)
        fa_hc_mean = None
        fa_rrms_mean = None

        if hc_fa_mean is not None and len(hc_fa_mean) >= metric_len:
            fa_hc_mean = hc_fa_mean[:metric_len]
        if rrms_fa_mean is not None and len(rrms_fa_mean) >= metric_len:
            fa_rrms_mean = rrms_fa_mean[:metric_len]

        # Load relative lesion load data (mean only, using RRMS subjects)
        rrms_lesion_mean, _ = aggregate_relative_lesion_load(rrms_sids, bundle)
        lesion_load_mean = None

        if rrms_lesion_mean is not None and len(rrms_lesion_mean) >= metric_len:
            lesion_load_mean = rrms_lesion_mean[:metric_len]

        # Plot all metrics for this bundle
        print(f"  Plotting bundle: {bundle}")
        out_fp = plot_bundle_all_metrics(
            bundle,
            hc_means,
            hc_stds,
            rrms_means,
            rrms_stds,
            nufo_hc_mean,
            nufo_rrms_mean,
            theta_hc_mean,
            theta_rrms_mean,
            lesion_load_mean,
            fa_hc_mean,
            fa_rrms_mean,
            out_dir,
        )
        print(f"  Saved: {out_fp.name}")
        generated += 1

    print(f"\n{'=' * 60}")
    if segment_summary_rows:
        segment_summary_fp = out_dir / "bundle_segment_group_means.csv"
        segment_summary_df = pd.DataFrame(segment_summary_rows)
        segment_summary_df.to_csv(segment_summary_fp, index=False)
        print(f"Saved cohort segment means: {segment_summary_fp}")

        # Summarize the cohort means across all available segments per bundle.
        bundle_summary_rows = []
        for (bundle, metric), group in segment_summary_df.groupby(
                ["bundle", "metric"]):
            metric_key = next(
                key for key, info in METRICS.items() if info["short"] == metric
            )
            hc_subject_means = get_subject_bundle_means(
                hc_sids, bundle, metric_key
            )
            rrms_subject_means = get_subject_bundle_means(
                rrms_sids, bundle, metric_key
            )
            if len(hc_subject_means) >= 2 and len(rrms_subject_means) >= 2:
                t_stat, p_value = ttest_ind(
                    hc_subject_means,
                    rrms_subject_means,
                    equal_var=False,
                )
            else:
                t_stat, p_value = np.nan, np.nan

            bundle_summary_rows.append(
                {
                    "bundle": bundle,
                    "metric": metric,
                    "n_segments": len(group),
                    "hc_mean_across_segments": group["hc_mean"].mean(),
                    "hc_std_across_segments": np.nanstd(group["hc_mean"]),
                    "rrms_mean_across_segments": group["rrms_mean"].mean(),
                    "rrms_std_across_segments": np.nanstd(group["rrms_mean"]),
                    "rrms_minus_hc_mean_across_segments": group[
                        "rrms_minus_hc"].mean(),
                    "rrms_minus_hc_std_across_segments": np.nanstd(
                        group["rrms_minus_hc"]),
                    "hc_n_subjects": len(hc_subject_means),
                    "rrms_n_subjects": len(rrms_subject_means),
                    "welch_t_statistic": t_stat,
                    "p_value": p_value,
                }
            )
        bundle_summary_fp = out_dir / "bundle_group_means.csv"
        bundle_summary_df = pd.DataFrame(bundle_summary_rows)
        bundle_summary_df["p_value_fdr_bh"], bundle_summary_df["fdr_reject"] = bh_fdr(
            bundle_summary_df["p_value"]
        )
        bundle_summary_df["p_value_significant"] = (
            bundle_summary_df["p_value"] < 0.05
        )
        bundle_summary_df.to_csv(bundle_summary_fp, index=False)
        print(f"Saved bundle-level segment summaries: {bundle_summary_fp}")

        # Per bundle/metric maximum absolute segment difference (RRMS - HC).
        max_diff_df = (
            segment_summary_df
            .sort_values(["bundle", "metric", "abs_rrms_minus_hc"], ascending=[True, True, False])
            .groupby(["bundle", "metric"], as_index=False)
            .first()[
                [
                    "bundle",
                    "metric",
                    "segment_index",
                    "rrms_minus_hc",
                    "abs_rrms_minus_hc",
                    "hc_mean",
                    "rrms_mean",
                ]
            ]
            .rename(
                columns={
                    "segment_index": "max_abs_diff_segment_index",
                    "rrms_minus_hc": "max_diff_rrms_minus_hc",
                    "abs_rrms_minus_hc": "max_abs_diff_rrms_minus_hc",
                    "hc_mean": "hc_mean_at_max_abs_diff",
                    "rrms_mean": "rrms_mean_at_max_abs_diff",
                }
            )
        )
        max_diff_fp = out_dir / "bundle_segment_max_differences.csv"
        max_diff_df.to_csv(max_diff_fp, index=False)
        print(f"Saved max segment differences: {max_diff_fp}")
    print(f"Generated {generated} figures in {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    print("=" * 80)
    print("FIGURE 5: Bundle profilometry - R1, R2, AI profiles along WM bundles for HC vs RRMS")
    print("=" * 80)
    create_figure()
    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80 + "\n")
