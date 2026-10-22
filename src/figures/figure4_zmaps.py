"""
Figure 4: Multi-slice axial comparison of R1 and R2 in healthy controls and
subject S27 with corresponding unsigned z-maps and T2-FLAIR lesion overlays.

Layout (4 x 6):
- Columns are grouped by slice (3 slices x 2 columns): [R1, R2] per slice.
- Row 1: HC atlas mean
- Row 2: RRMS subject
- Row 3: Z-map
- Row 4: T2-FLAIR + lesion overlay (duplicated in both columns per slice)

Colorbars are horizontal and placed below row 3, with width matching one column.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
import nibabel as nib
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

try:
    from colorresourcespy.py.relaxationColorMap import generate_relaxation_colormap
    from utils import paths
    from utils.plotting import apply_figure_style, create_figure as make_figure
    from utils.paper_figure_helpers import *
    from utils.paper_meta import *
except ImportError as exc:
    raise RuntimeError(f"Could not import project utilities: {exc}") from exc


# =============================================================================
# CONFIGURATION
# =============================================================================

exp_name = Path(__file__).stem

OUTPUT_DIR = paths.root / "fig" / exp_name
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = get_paper_figure_cache_dir(exp_name)

ZSCORE_MIN_THRESHOLD = 1.0
ZSCORE_MAX_THRESHOLD_VIZ = 6.0
ZSCORE_CMAP = "plasma"

WM_MASK_THRESHOLD = 0.8

# Three middle axial slices from the previous 5-slice set.
AXIAL_SLICE_INDICES = [107, 97, 93]
TARGET_SUBJECT = "S27"

r1_min, r1_max = 0.3, 3.0
r2_min, r2_max = 10.0, 30.0

r1_cmap = generate_relaxation_colormap(
    paths.colorresource / "lipari.csv", r1_min, r1_max, True
)
r1_cmap.set_under(r1_cmap(1))

r2_cmap = generate_relaxation_colormap(
    paths.colorresource / "navia.csv", r2_min, r2_max, True
)
r2_cmap.set_under(r2_cmap(1))

METRICS_CONFIG: Dict[str, Dict[str, object]] = {
    "R1": {
        "nifti_name": "t1_pinn",
        "label": r"$\mathbf{R_1}$ [1/s]",
        "cmap_mean": r1_cmap,
        "vmin_mean": r1_min,
        "vmax_mean": r1_max,
    },
    "R2": {
        "nifti_name": "t2_pinn",
        "label": r"$\mathbf{R_2}$ [1/s]",
        "cmap_mean": r2_cmap,
        "vmin_mean": r2_min,
        "vmax_mean": r2_max,
    },
}


# =============================================================================
# COMPUTE GROUP STATISTICS
# =============================================================================


def compute_hc_stats() -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    hc_subjects = get_hc_baseline_subject_ids(get_subject_id_mapping())
    print(f"HC baseline subjects: {len(hc_subjects)}")

    group_stats: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for metric_key, metric_cfg in METRICS_CONFIG.items():
        nifti_name = str(metric_cfg["nifti_name"])
        mean_fp = CACHE_DIR / f"{metric_key}_hc_mean.nii.gz"
        std_fp = CACHE_DIR / f"{metric_key}_hc_std.nii.gz"

        if mean_fp.exists() and std_fp.exists():
            mean_img = load_nifti(mean_fp)
            std_img = load_nifti(std_fp)
            if mean_img is None or std_img is None:
                continue
            mean_data = mean_img.get_fdata()
            std_data = std_img.get_fdata()
            group_stats[metric_key] = (
                mask_zero(mean_data), mask_zero(std_data))
            continue
        else:
            print(f"  Computing HC mean/std for {metric_key}...")
            data_stack = []
            reference_shape = None
            reference_affine = None

            for subject_id in hc_subjects:
                metric_fp = paths.out_common_space / subject_id / \
                    "bSSFP_to_ICBM152_qMRI" / \
                    f"{nifti_name}.space-icbm152.nii.gz"
                metric_img = load_nifti(metric_fp)
                if metric_img is None:
                    continue
                data = mask_zero(
                    convert_to_relaxation_rate(
                        metric_img.get_fdata(), nifti_name)
                )
                affine = metric_img.affine

                if reference_shape is None:
                    reference_shape = data.shape
                    reference_affine = affine

                if data.shape != reference_shape:
                    continue
                data_stack.append(data)

            if not data_stack or reference_affine is None:
                print(f"    Skipping {metric_key}: no valid data found.")
                continue

            stack = np.stack(data_stack, axis=0)
            with np.errstate(invalid="ignore"):
                mean_data = np.nanmean(stack, axis=0)
                std_data = np.nanstd(stack, axis=0)

            nib.save(nib.Nifti1Image(mean_data, reference_affine), mean_fp)
            nib.save(nib.Nifti1Image(std_data, reference_affine), std_fp)

        group_stats[metric_key] = (mask_zero(mean_data), mask_zero(std_data))

    return group_stats


# =============================================================================
# PLOTTING
# =============================================================================


def _prepare_metric_maps(
    subject_id: str,
    metric_key: str,
    metric_cfg: Dict[str, object],
    hc_stats: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> Optional[Dict[str, np.ndarray]]:
    if metric_key not in hc_stats:
        return None

    hc_mean, hc_std = hc_stats[metric_key]
    nifti_name = str(metric_cfg["nifti_name"])
    metric_fp = paths.out_common_space / subject_id / \
        "bSSFP_to_ICBM152_qMRI" / f"{nifti_name}.space-icbm152.nii.gz"
    metric_img = load_nifti(metric_fp)
    subject_data = (
        mask_zero(convert_to_relaxation_rate(
            metric_img.get_fdata(), nifti_name))
        if metric_img is not None
        else None
    )
    if subject_data is None or subject_data.shape != hc_mean.shape:
        return None

    wm_mask_template = load_wm_mask_common_space(WM_MASK_THRESHOLD)
    wm_mask_subject = load_subject_wm_mask_common_space(subject_id)
    lesion_mask = load_lesion_mask_common_space(subject_id)

    combined_mask = build_combined_common_space_mask(
        wm_mask_template,
        wm_mask_subject,
        lesion_mask,
        subject_data.shape,
    )

    # save the combined mask for debugging/inspection
    combined_mask_fp = CACHE_DIR / \
        f"{subject_id}_{metric_key}_combined_mask.nii.gz"
    if combined_mask is not None:
        nib.save(nib.Nifti1Image(combined_mask.astype(
            np.uint8), metric_img.affine), combined_mask_fp)

    hc_mean_masked = np.where(combined_mask, hc_mean,
                              np.nan) if combined_mask is not None else hc_mean
    subject_masked = np.where(combined_mask, subject_data,
                              np.nan) if combined_mask is not None else subject_data
    abs_z = calculate_unsigned_zscore_map(
        subject_masked, hc_mean_masked, hc_std, combined_mask
    )

    return {
        "atlas": hc_mean_masked,
        "subject": subject_masked,
        "z": abs_z,
    }


def save_rrms_lesion_nawm_zstats(
    hc_stats: Dict[str, Tuple[np.ndarray, np.ndarray]]
) -> Optional[Path]:
    subject_map = get_subject_id_mapping()
    hc_sids = get_hc_baseline_subject_ids(subject_map)
    rrms_sids = get_age_matched_rrms_subject_ids(subject_map, hc_sids)
    if not rrms_sids:
        print("No age-matched RRMS subjects found for lesion/NAWM z-stat export.")
        return None

    rows = []
    pooled_by_metric = {
        metric_key: {"lesion": [], "nawm": []} for metric_key in METRICS_CONFIG
    }

    for subject_id in rrms_sids:
        wm_template = load_wm_mask_common_space(WM_MASK_THRESHOLD)
        wm_subject = load_subject_wm_mask_common_space(subject_id)
        lesion_mask = load_lesion_mask_common_space(subject_id)

        for metric_key, metric_cfg in METRICS_CONFIG.items():
            if metric_key not in hc_stats:
                continue

            nifti_name = str(metric_cfg["nifti_name"])
            metric_fp = (
                paths.out_common_space
                / subject_id
                / "bSSFP_to_ICBM152_qMRI"
                / f"{nifti_name}.space-icbm152.nii.gz"
            )
            metric_img = load_nifti(metric_fp)
            if metric_img is None:
                continue

            subject_data = mask_zero(
                convert_to_relaxation_rate(metric_img.get_fdata(), nifti_name)
            )
            hc_mean, hc_std = hc_stats[metric_key]
            if subject_data.shape != hc_mean.shape:
                continue

            lesion_bool = (
                lesion_mask.astype(bool)
                if lesion_mask is not None and lesion_mask.shape == subject_data.shape
                else np.zeros(subject_data.shape, dtype=bool)
            )

            wm_nawm = None
            if wm_template is not None and wm_template.shape == subject_data.shape:
                wm_nawm = wm_template.copy()
                if wm_subject is not None and wm_subject.shape == subject_data.shape:
                    wm_nawm &= wm_subject
            elif wm_subject is not None and wm_subject.shape == subject_data.shape:
                wm_nawm = wm_subject.copy()

            if wm_nawm is None:
                combined = build_combined_common_space_mask(
                    wm_template, wm_subject, lesion_mask, subject_data.shape
                )
                wm_nawm = combined.copy() if combined is not None else None

            nawm_bool = (
                wm_nawm & ~lesion_bool if wm_nawm is not None else np.zeros(
                    subject_data.shape, dtype=bool)
            )

            abs_z = calculate_unsigned_zscore_map(
                subject_data, hc_mean, hc_std)

            lesion_vals = abs_z[lesion_bool]
            lesion_vals = lesion_vals[np.isfinite(lesion_vals)]
            nawm_vals = abs_z[nawm_bool]
            nawm_vals = nawm_vals[np.isfinite(nawm_vals)]

            pooled_by_metric[metric_key]["lesion"].append(lesion_vals)
            pooled_by_metric[metric_key]["nawm"].append(nawm_vals)

            rows.append(
                {
                    "subject_id": subject_id,
                    "metric": metric_key,
                    "lesion_n_voxels": int(lesion_vals.size),
                    "lesion_abs_z_mean": float(np.nanmean(lesion_vals)) if lesion_vals.size else np.nan,
                    "lesion_abs_z_std": float(np.nanstd(lesion_vals)) if lesion_vals.size else np.nan,
                    "nawm_n_voxels": int(nawm_vals.size),
                    "nawm_abs_z_mean": float(np.nanmean(nawm_vals)) if nawm_vals.size else np.nan,
                    "nawm_abs_z_std": float(np.nanstd(nawm_vals)) if nawm_vals.size else np.nan,
                }
            )

    for metric_key in METRICS_CONFIG:
        lesion_concat = [
            arr for arr in pooled_by_metric[metric_key]["lesion"] if arr.size > 0]
        nawm_concat = [
            arr for arr in pooled_by_metric[metric_key]["nawm"] if arr.size > 0]
        lesion_all = np.concatenate(
            lesion_concat) if lesion_concat else np.array([], dtype=float)
        nawm_all = np.concatenate(
            nawm_concat) if nawm_concat else np.array([], dtype=float)

        rows.append(
            {
                "subject_id": "ALL_AGE_MATCHED_RRMS",
                "metric": metric_key,
                "lesion_n_voxels": int(lesion_all.size),
                "lesion_abs_z_mean": float(np.nanmean(lesion_all)) if lesion_all.size else np.nan,
                "lesion_abs_z_std": float(np.nanstd(lesion_all)) if lesion_all.size else np.nan,
                "nawm_n_voxels": int(nawm_all.size),
                "nawm_abs_z_mean": float(np.nanmean(nawm_all)) if nawm_all.size else np.nan,
                "nawm_abs_z_std": float(np.nanstd(nawm_all)) if nawm_all.size else np.nan,
            }
        )

    if not rows:
        print("No lesion/NAWM z-stat rows generated.")
        return None

    out_fp = CACHE_DIR / "rrms_lesion_nawm_abs_zstats.csv"
    pd.DataFrame(rows).to_csv(out_fp, index=False)
    print(f"Saved lesion/NAWM z-stats: {out_fp}")
    return out_fp


def _compute_crop_bounds(arr_2d: np.ndarray, pad: int = 4) -> Tuple[int, int, int, int]:
    """Return (rmin, rmax, cmin, cmax) bounding box of finite, non-zero data with padding."""
    valid = np.isfinite(arr_2d) & (arr_2d != 0)
    rows = np.any(valid, axis=1)
    cols = np.any(valid, axis=0)
    if not rows.any() or not cols.any():
        h, w = arr_2d.shape
        return 0, h - 1, 0, w - 1
    rmin = int(np.where(rows)[0][0])
    rmax = int(np.where(rows)[0][-1])
    cmin = int(np.where(cols)[0][0])
    cmax = int(np.where(cols)[0][-1])
    h, w = arr_2d.shape
    return (
        max(0, rmin - pad),
        min(h - 1, rmax + pad),
        max(0, cmin - pad),
        min(w - 1, cmax + pad),
    )


def plot_combined_r1_r2_figure(
    subject_id: str,
    hc_stats: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> Optional[plt.Figure]:
    apply_figure_style()

    bg_color = "white"
    fg_color = "black"

    r1_maps = _prepare_metric_maps(
        subject_id, "R1", METRICS_CONFIG["R1"], hc_stats)
    r2_maps = _prepare_metric_maps(
        subject_id, "R2", METRICS_CONFIG["R2"], hc_stats)
    flair_data = load_t2flair_common_space(subject_id)
    lesion_mask = load_lesion_mask_common_space(subject_id)

    if r1_maps is None or r2_maps is None:
        print("Missing R1 or R2 data/stats for combined figure.")
        return None

    flair_vmin, flair_vmax = 0.0, 1.0
    if flair_data is not None and np.isfinite(flair_data).any():
        flair_vmin = 47  # float(np.nanpercentile(flair_data, 5))
        flair_vmax = 153  # float(np.nanpercentile(flair_data, 95))
        if not np.isfinite(flair_vmin) or not np.isfinite(flair_vmax) or flair_vmax <= flair_vmin:
            flair_vmin, flair_vmax = 0.0, 1.0

    n_rows = 4
    n_cols = 6
    fig = make_figure(height=8.2, width=11.0)
    fig.patch.set_facecolor(bg_color)

    # 4x6 content grid with slim spacer columns between slice pairs.
    gs = GridSpec(
        n_rows,
        8,
        figure=fig,
        width_ratios=[1.0, 1.0, 0.11, 1.0, 1.0, 0.11, 1.0, 1.0],
        wspace=0.02,
        hspace=0.12,
    )
    axes = np.empty((n_rows, n_cols), dtype=object)
    visible_to_grid = [0, 1, 3, 4, 6, 7]
    for r in range(n_rows):
        for c in range(n_cols):
            axes[r, c] = fig.add_subplot(gs[r, visible_to_grid[c]])
            axes[r, c].set_facecolor(bg_color)

    row_labels = ["HC atlas", "RRMS", "Z-map", "Lesions"]

    mappable_r1 = None
    mappable_r2 = None
    mappable_z = None

    for slice_idx_pos, slice_idx in enumerate(AXIAL_SLICE_INDICES):
        col_r1 = 2 * slice_idx_pos
        col_r2 = col_r1 + 1

        # Compute a single crop bounding box for this slice from the R1 atlas.
        _ref = extract_axial_slice(r1_maps["atlas"], slice_idx)
        _rmin, _rmax, _cmin, _cmax = _compute_crop_bounds(_ref, pad=16)

        for metric_key, col_idx, maps in [("R1", col_r1, r1_maps), ("R2", col_r2, r2_maps)]:
            cfg = METRICS_CONFIG[metric_key]

            atlas_slice = extract_axial_slice(maps["atlas"], slice_idx)
            subj_slice = extract_axial_slice(maps["subject"], slice_idx)
            z_slice = extract_axial_slice(maps["z"], slice_idx)

            if flair_data is not None:
                flair_slice = mask_zero(
                    extract_axial_slice(flair_data, slice_idx))
                axes[0, col_idx].imshow(
                    flair_slice, cmap="gray", vmin=flair_vmin, vmax=flair_vmax)
                axes[1, col_idx].imshow(
                    flair_slice, cmap="gray", vmin=flair_vmin, vmax=flair_vmax)
                axes[2, col_idx].imshow(
                    flair_slice, cmap="gray", vmin=flair_vmin, vmax=flair_vmax)

            axes[0, col_idx].imshow(
                atlas_slice,
                cmap=cfg["cmap_mean"],
                vmin=cfg["vmin_mean"],
                vmax=cfg["vmax_mean"],
            )
            im1 = axes[1, col_idx].imshow(
                subj_slice,
                cmap=cfg["cmap_mean"],
                vmin=cfg["vmin_mean"],
                vmax=cfg["vmax_mean"],
            )
            im2 = axes[2, col_idx].imshow(
                z_slice,
                cmap=ZSCORE_CMAP,
                vmin=ZSCORE_MIN_THRESHOLD,
                vmax=ZSCORE_MAX_THRESHOLD_VIZ,
            )

            if metric_key == "R1":
                mappable_r1 = im1
            else:
                mappable_r2 = im1
            mappable_z = im2

            for rr in [0, 1, 2]:
                axes[rr, col_idx].axis("off")

        for col_idx in [col_r1, col_r2]:
            if flair_data is not None:
                flair_slice = mask_zero(
                    extract_axial_slice(flair_data, slice_idx))
                axes[3, col_idx].imshow(
                    flair_slice, cmap="gray", vmin=flair_vmin, vmax=flair_vmax)

            if lesion_mask is not None:
                lesion_slice = extract_axial_slice(
                    lesion_mask.astype(np.float32), slice_idx)
                if np.any(lesion_slice > 0):
                    lesion_display = np.where(lesion_slice > 0.5, 1.0, np.nan)
                    axes[3, col_idx].imshow(
                        lesion_display,
                        cmap=ListedColormap(["red"]),
                        alpha=1.0,
                        vmin=0,
                        vmax=1,
                    )

            axes[3, col_idx].axis("off")

        # Per-slice labels over each R1/R2 column pair.
        axes[0, col_r1].text(
            1.0,
            1.03,
            f"Slice {slice_idx_pos + 1}",
            transform=axes[0, col_r1].transAxes,
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color=fg_color,
        )
        axes[0, col_r1].text(
            0.5,
            1.01,
            r"$\mathbf{R_1}$",
            transform=axes[0, col_r1].transAxes,
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color=fg_color,
        )
        axes[0, col_r2].text(
            0.5,
            1.01,
            r"$\mathbf{R_2}$",
            transform=axes[0, col_r2].transAxes,
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color=fg_color,
        )

        # Crop all axes in this slice pair to the brain bounding box.
        for _rr in range(n_rows):
            for _cc in [col_r1, col_r2]:
                axes[_rr, _cc].set_xlim(_cmin - 0.5, _cmax + 0.5)
                axes[_rr, _cc].set_ylim(_rmax + 0.5, _rmin - 0.5)

    fig.subplots_adjust(left=0.06, right=0.995, top=0.95,
                        bottom=0.065, wspace=0.02, hspace=0.12)

    # Row labels on the left.
    for row_idx, row_label in enumerate(row_labels):
        pos = axes[row_idx, 0].get_position()
        y_center = pos.y0 + pos.height / 2
        fig.text(
            0.012,
            y_center,
            row_label,
            rotation=90,
            va="center",
            ha="left",
            fontsize=9.5,
            fontweight="bold",
            color=fg_color,
        )

    # R1/R2 colorbars below row 2 (one per column, width = column width).
    if mappable_r1 is not None and mappable_r2 is not None and mappable_z is not None:
        row2_pos = axes[1, 0].get_position()
        row3_pos = axes[2, 0].get_position()
        gap_row2_bottom = row3_pos.y1
        gap_row2_top = row2_pos.y0
        gap_row2_h = max(0.01, gap_row2_top - gap_row2_bottom)

        cbar_h_row2 = min(0.010, 0.62 * gap_row2_h)
        cbar_y_row2 = gap_row2_bottom + 0.5 * (gap_row2_h - cbar_h_row2)

        for col_idx in range(6):
            col_pos = axes[1, col_idx].get_position()
            cax = fig.add_axes(
                [col_pos.x0, cbar_y_row2, col_pos.width, cbar_h_row2])
            cax.set_facecolor(bg_color)

            if col_idx % 2 == 0:
                cbar = fig.colorbar(mappable_r1, cax=cax,
                                    orientation="horizontal")
                cbar.set_label(
                    str(METRICS_CONFIG["R1"]["label"]), fontsize=9, color=fg_color, fontweight="bold")
            else:
                cbar = fig.colorbar(mappable_r2, cax=cax,
                                    orientation="horizontal")
                cbar.set_label(
                    str(METRICS_CONFIG["R2"]["label"]), fontsize=9, color=fg_color, fontweight="bold")

            cbar.ax.tick_params(labelsize=8, length=2, colors=fg_color)
            for tick_label in cbar.ax.get_xticklabels():
                tick_label.set_fontweight("bold")
            cbar.outline.set_edgecolor(fg_color)

        # Z colorbars below row 3, one per slice pair spanning two columns.
        row4_pos = axes[3, 0].get_position()
        gap_row3_bottom = row4_pos.y1
        gap_row3_top = row3_pos.y0
        gap_row3_h = max(0.01, gap_row3_top - gap_row3_bottom)

        cbar_h_row3 = min(0.011, 0.68 * gap_row3_h)
        cbar_y_row3 = gap_row3_bottom + 0.5 * (gap_row3_h - cbar_h_row3)

        for pair_idx in range(3):
            c0 = 2 * pair_idx
            c1 = c0 + 1
            left = axes[2, c0].get_position().x0
            right = axes[2, c1].get_position().x1
            cax = fig.add_axes([left, cbar_y_row3, right - left, cbar_h_row3])
            cax.set_facecolor(bg_color)
            cbar = fig.colorbar(mappable_z, cax=cax, orientation="horizontal")
            cbar.set_label("|Z|", fontsize=9,
                           color=fg_color, fontweight="bold")
            cbar.ax.tick_params(labelsize=8, length=2, colors=fg_color)
            for tick_label in cbar.ax.get_xticklabels():
                tick_label.set_fontweight("bold")
            cbar.outline.set_edgecolor(fg_color)

    return fig


# =============================================================================
# MAIN
# =============================================================================


def create_figure() -> None:
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Cache directory: {CACHE_DIR}")
    print(f"Target subject: {TARGET_SUBJECT}")
    print(f"Axial slice indices: {AXIAL_SLICE_INDICES}")

    print("\nComputing HC statistics...")
    hc_stats = compute_hc_stats()
    save_rrms_lesion_nawm_zstats(hc_stats)

    print(
        f"\nGenerating bright 4x6 R1+R2 figure for subject {TARGET_SUBJECT}...")

    fig = plot_combined_r1_r2_figure(TARGET_SUBJECT, hc_stats)
    if fig is not None:
        output_fp = OUTPUT_DIR / f"{exp_name}_{TARGET_SUBJECT}.svg"
        fig.savefig(output_fp, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Saved: {output_fp}")
    else:
        print("Skipping bright figure: could not generate figure")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("FIGURE 4: Z-maps comparing RRMS subject to HC atlas")
    print("=" * 80)
    create_figure()
    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80 + "\n")
