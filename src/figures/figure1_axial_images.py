"""
Figure 1: Axial images and common-space comparisons for subject S23.

This script generates axial slices of R1, R2, AI, FA, NuFO, and fiber-to-field
angle maps in subject space. It also generates common-space HC mean maps, the
corresponding RRMS subject maps, and unsigned z-maps for R1 and R2 using
white-matter and lesion masks.
"""

from __future__ import annotations
import numpy as np

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))

try:
    from colorresourcespy.py.relaxationColorMap import generate_relaxation_colormap
    from utils import paths
    from utils.paper_figure_helpers import (
        build_combined_common_space_mask,
        calculate_unsigned_zscore_map,
        convert_to_relaxation_rate,
        extract_axial_slice,
        get_paper_figure_cache_dir,
        load_lesion_mask_common_space,
        load_nifti,
        load_subject_wm_mask_common_space,
        load_t2flair_common_space,
        load_wm_mask_common_space,
    )
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(f"Could not import project utilities: {exc}") from exc


# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

SUBJECT_ID = "S23"
SLICE_INDEX = 82
SLICE_INDEX_MNI = 92
WM_MASK_THRESHOLD = 0.8

OUTPUT_DIR = paths.root / "fig" / Path(__file__).stem

r1_min, r1_max = 0.3, 3.0
r2_min, r2_max = 10.0, 30.0
ai_min, ai_max = -0.02, 0.08
r1_cmap = generate_relaxation_colormap(
    paths.colorresource / "lipari.csv", r1_min, r1_max, True
)
r1_cmap.set_under(r1_cmap(1))  # Set under color to lowest color in range

r2_cmap = generate_relaxation_colormap(
    paths.colorresource / "navia.csv", r2_min, r2_max, True
)
r2_cmap.set_under(r2_cmap(1))
ai_cmap = "inferno"


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------


def mask_data(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if data.ndim == 4:
        return np.where(mask[..., None], data, np.nan)
    return np.where(mask, data, np.nan)


def save_slice_image(
    slice_data: np.ndarray,
    out_name: str,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "gray",
) -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(slice_data, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis("off")
    plt.tight_layout()
    out_fp = OUTPUT_DIR / out_name
    plt.savefig(out_fp, dpi=300, transparent=True)
    plt.close(fig)
    print(f"Saved slice: {out_fp}")


def save_slice_with_background(
    main_slice: np.ndarray,
    out_name: str,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "gray",
    bg_slice: np.ndarray | None = None,
    bg_vmin: float | None = None,
    bg_vmax: float | None = None,
    overlay_mask: np.ndarray | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    if main_slice is not None:
        main_slice = main_slice.copy()
        main_slice[main_slice == 0] = np.nan
    if bg_slice is not None:
        bg_slice = bg_slice.copy()
        bg_slice[bg_slice == 0] = np.nan
        ax.imshow(bg_slice, cmap="gray", vmin=bg_vmin, vmax=bg_vmax)
    ax.imshow(main_slice, cmap=cmap, vmin=vmin, vmax=vmax)
    if overlay_mask is not None and np.any(overlay_mask):
        ax.contour(overlay_mask.astype(np.float32), levels=[0.5],
                   colors="#00ffff", linewidths=0.4)
    ax.axis("off")
    plt.tight_layout()
    out_fp = OUTPUT_DIR / out_name
    plt.savefig(out_fp, dpi=300, transparent=True)
    plt.close(fig)
    print(f"Saved slice: {out_fp}")


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------


def create_figure() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Processing {SUBJECT_ID}...")

    # ---------------- Subject-space single-slice assets ----------------
    out_subject_fop = paths.out_subject_space / SUBJECT_ID
    bssfp_fop = out_subject_fop / "bSSFP_to_T1_qMRI"
    tf_subject_fop = paths.tf_results_pft / SUBJECT_ID
    t1_mask_fp = tf_subject_fop / "Register_T1" / \
        f"{SUBJECT_ID}__t1_mask_warped.nii.gz"
    t1_mask_img = load_nifti(t1_mask_fp, log_missing=True)
    if t1_mask_img is None:
        print(f"  missing T1 mask for {SUBJECT_ID}; aborting")
        return
    t1_mask = t1_mask_img.get_fdata()
    mask_bool = t1_mask > 0

    assets = {
        "R1": {
            "fp": bssfp_fop / "t1_pinn.space-tf.nii.gz",
            "vmin": r1_min,
            "vmax": r1_max,
            "cmap": r1_cmap,
            "transform": lambda x: 1000.0 / x,
        },
        "R2": {
            "fp": bssfp_fop / "t2_pinn.space-tf.nii.gz",
            "vmin": r2_min,
            "vmax": r2_max,
            "cmap": r2_cmap,
            "transform": lambda x: 1000.0 / x,
        },
        "AI": {
            "fp": bssfp_fop / "ai_map.unringed.space-tf.nii.gz",
            "vmin": ai_min,
            "vmax": ai_max,
            "cmap": ai_cmap,
            "transform": None,
        },
        "FA": {
            "fp": tf_subject_fop / "DTI_Metrics" / f"{SUBJECT_ID}__fa.nii.gz",
            "vmin": 0.0,
            "vmax": 0.9,
            "cmap": "gray",
            "transform": None,
        },
        "NuFO": {
            "fp": tf_subject_fop / "FODF_Metrics" / f"{SUBJECT_ID}__nufo.nii.gz",
            "vmin": None,
            "vmax": None,
            "cmap": "gray",
            "transform": None,
        },
        "Theta1": {
            "fp": tf_subject_fop / "FODF_Metrics" / f"{SUBJECT_ID}__theta_1.nii.gz",
            "vmin": 0,
            "vmax": 90,
            "cmap": "gray",
            "transform": None,
        },
    }

    for label, meta in assets.items():
        data_img = load_nifti(meta["fp"], log_missing=True)
        if data_img is None:
            continue
        data = data_img.get_fdata()
        if meta.get("transform"):
            data = meta["transform"](data)
        data = mask_data(data, mask_bool)
        if data.ndim == 4:
            data = data[..., 0]
        sl = np.rot90(data[:, :, SLICE_INDEX])
        save_slice_image(
            sl,
            out_name=f"{SUBJECT_ID}_{label.lower()}_slice{SLICE_INDEX}.svg",
            vmin=meta["vmin"],
            vmax=meta["vmax"],
            cmap=meta["cmap"],
        )

    # ---------------- Common-space HC/RRMS/Z-map assets ----------------
    cache_dir = get_paper_figure_cache_dir("figure4_zmaps")
    mni_dir = paths.out_common_space / SUBJECT_ID / "bSSFP_to_ICBM152_qMRI"

    wm_mask_common = load_wm_mask_common_space(WM_MASK_THRESHOLD)
    wm_mask_subject = load_subject_wm_mask_common_space(SUBJECT_ID)
    if wm_mask_subject is None:
        print(
            f"  warning: subject WM mask in common space missing for {SUBJECT_ID}")
    lesion_mask = load_lesion_mask_common_space(SUBJECT_ID)
    flair_data = load_t2flair_common_space(SUBJECT_ID)

    flair_vmin, flair_vmax = 0.0, 1.0
    if flair_data is not None and np.isfinite(flair_data).any():
        flair_vmin = float(np.nanpercentile(flair_data, 5))
        flair_vmax = float(np.nanpercentile(flair_data, 99))
        if not np.isfinite(flair_vmin) or not np.isfinite(flair_vmax):
            flair_vmin, flair_vmax = 0.0, 1.0
        elif flair_vmax <= flair_vmin:
            flair_vmin = float(np.nanpercentile(flair_data, 1))
            flair_vmax = float(np.nanpercentile(flair_data, 99))
            if flair_vmax <= flair_vmin:
                flair_vmax = flair_vmin + 1.0

    for metric, meta in [
        ("R1", {"nifti": "t1_pinn", "cmap": r1_cmap, "vmin": 0.3, "vmax": 3.0}),
        ("R2", {"nifti": "t2_pinn", "cmap": r2_cmap, "vmin": 10.0, "vmax": 30.0}),
    ]:
        hc_mean_fp = cache_dir / f"{metric}_hc_mean.nii.gz"
        hc_std_fp = cache_dir / f"{metric}_hc_std.nii.gz"
        rrms_fp = mni_dir / f"{meta['nifti']}.space-icbm152.nii.gz"

        hc_mean_img = load_nifti(hc_mean_fp, log_missing=True)
        hc_std_img = load_nifti(hc_std_fp, log_missing=True)
        rrms_img = load_nifti(rrms_fp, log_missing=True)
        hc_mean = None if hc_mean_img is None else hc_mean_img.get_fdata()
        hc_std = None if hc_std_img is None else hc_std_img.get_fdata()
        rrms = None if rrms_img is None else rrms_img.get_fdata()
        if rrms is not None:
            rrms = convert_to_relaxation_rate(rrms, meta["nifti"])

        if hc_mean is None or hc_std is None or rrms is None:
            continue

        # combined_mask = (template WM & subject WM) | lesion
        combined_mask = build_combined_common_space_mask(
            wm_mask_common, wm_mask_subject, lesion_mask, rrms.shape
        )
        hc_mean_masked = np.where(combined_mask, hc_mean, np.nan)
        rrms_masked = np.where(combined_mask, rrms, np.nan)
        abs_z = calculate_unsigned_zscore_map(
            rrms_masked, hc_mean_masked, hc_std, combined_mask
        )

        bg_slice = extract_axial_slice(
            flair_data, SLICE_INDEX_MNI) if flair_data is not None else None
        if bg_slice is not None:
            bg_slice = np.where(np.isfinite(bg_slice), bg_slice, np.nan)

        save_slice_with_background(
            extract_axial_slice(hc_mean_masked, SLICE_INDEX_MNI),
            f"hc_mean_{metric.lower()}_slice{SLICE_INDEX_MNI}.svg",
            vmin=meta["vmin"],
            vmax=meta["vmax"],
            cmap=meta["cmap"],
            bg_slice=bg_slice,
            bg_vmin=flair_vmin,
            bg_vmax=flair_vmax,
        )

        save_slice_with_background(
            extract_axial_slice(rrms_masked, SLICE_INDEX_MNI),
            f"{SUBJECT_ID}_mni_{metric.lower()}_slice{SLICE_INDEX_MNI}.svg",
            vmin=meta["vmin"],
            vmax=meta["vmax"],
            cmap=meta["cmap"],
            bg_slice=bg_slice,
            bg_vmin=flair_vmin,
            bg_vmax=flair_vmax,
        )

        save_slice_with_background(
            extract_axial_slice(abs_z, SLICE_INDEX_MNI),
            f"zmap_{metric.lower()}_slice{SLICE_INDEX_MNI}.svg",
            vmin=1.0,
            vmax=6.0,
            cmap="plasma",
            bg_slice=bg_slice,
            bg_vmin=flair_vmin,
            bg_vmax=flair_vmax,
        )


if __name__ == "__main__":
    create_figure()
