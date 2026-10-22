"""Shared helper functions for paper figure scripts.

These helpers centralize repeated logic across final paper figure scripts
(subject mapping, common-space mask loading, and theta_1-from-peaks handling).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

import nibabel as nib
import numpy as np
import pandas as pd
import yaml

from utils import paths
from utils.dti import calc_angles_from_v1


METADATA_FP = (
    paths.data
    / "251217-b_get_gender_age_participants"
    / "251217-b_participant_metadata.csv"
)


def get_paper_figure_cache_dir(exp_name: str) -> Path:
    """Return the cache directory for a paper figure script."""
    cache_dir = paths.data / "paper_figures" / exp_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_subplot_labels(count: int, start: str = "a") -> list[str]:
    """Return subplot labels formatted as lowercase letters in parentheses."""
    start_idx = ord(start.lower())
    return [f"({chr(start_idx + index)})" for index in range(count)]


def load_nifti(filepath: str | Path, log_missing: bool = False) -> nib.Nifti1Image | None:
    """Load a NIfTI image and optionally report when the file is missing."""
    try:
        return nib.load(str(filepath))
    except FileNotFoundError:
        if log_missing:
            print(f"File not found: {filepath}")
        return None


@lru_cache(maxsize=1)
def get_roi_label_config() -> tuple[list[str], dict[str, list[int]]]:
    """Load shared ROI order and FreeSurfer wmparc label sets from config."""
    config = yaml.safe_load(open(paths.default_config, "r", encoding="utf-8"))
    roi_cfg = config.get("roi_label_config", {}) if isinstance(
        config, dict) else {}
    order = roi_cfg.get("order", []) if isinstance(roi_cfg, dict) else []
    wmparc_labels = roi_cfg.get(
        "wmparc_labels", {}) if isinstance(roi_cfg, dict) else {}

    if not isinstance(order, list) or not isinstance(wmparc_labels, dict):
        raise ValueError("Invalid roi_label_config format in defaults.yaml")

    parsed_order = [str(name) for name in order]
    parsed_labels: dict[str, list[int]] = {}
    for key, values in wmparc_labels.items():
        if not isinstance(values, list):
            raise ValueError(
                f"Invalid wmparc_labels entry for '{key}' in defaults.yaml")
        parsed_labels[str(key)] = [int(v) for v in values]

    return parsed_order, parsed_labels


def convert_to_relaxation_rate(data: np.ndarray, metric: str) -> np.ndarray:
    """Convert T1/T2 maps (ms) to relaxation rates (1/s), keeping finite values."""
    if metric in ("t1_pinn", "t2_pinn"):
        with np.errstate(divide="ignore", invalid="ignore"):
            out = 1000.0 / data
            out[~np.isfinite(out)] = 0
        return out
    return data


def get_subject_space_assets(subject_id: str) -> dict[str, dict[str, Path | str | None]]:
    """Return standardized subject-space asset definitions.

    Structure intentionally mirrors the per-script `assets = {...}` style used in
    paper figures, with `fp` and optional `transform` entries.
    """
    out_subject = paths.out_subject_space / subject_id
    tf_subject = paths.tf_results_pft / subject_id
    return {
        "t1_pinn": {
            "fp": out_subject / "bSSFP_to_T1_qMRI" / "t1_pinn.space-tf.nii.gz",
            "transform": "relax_rate",
        },
        "t2_pinn": {
            "fp": out_subject / "bSSFP_to_T1_qMRI" / "t2_pinn.space-tf.nii.gz",
            "transform": "relax_rate",
        },
        "t1_miracle": {
            "fp": out_subject / "bSSFP_to_T1_qMRI" / "t1_miracle.space-tf.nii.gz",
            "transform": "relax_rate",
        },
        "t2_miracle": {
            "fp": out_subject / "bSSFP_to_T1_qMRI" / "t2_miracle.space-tf.nii.gz",
            "transform": "relax_rate",
        },
        "ai_map.unringed": {
            "fp": out_subject / "bSSFP_to_T1_qMRI" / "ai_map.unringed.space-tf.nii.gz",
            "transform": None,
        },
        "fa": {
            "fp": tf_subject / "DTI_Metrics" / f"{subject_id}__fa.nii.gz",
            "transform": None,
        },
        "nufo": {
            "fp": tf_subject / "FODF_Metrics" / f"{subject_id}__nufo.nii.gz",
            "transform": None,
        },
        "theta_1": {
            "fp": tf_subject / "FODF_Metrics" / f"{subject_id}__theta_1.nii.gz",
            "transform": None,
        },
        "wm_map": {
            "fp": tf_subject / "Segment_Tissues" / f"{subject_id}__map_wm.nii.gz",
            "transform": None,
        },
        "wm_mask_gt90": {
            "fp": tf_subject / "Segment_Tissues" / f"{subject_id}__mask_wm_gt90.nii.gz",
            "transform": None,
        },
        "lesion_mask": {
            "fp": out_subject / "Lesion_Masks_to_T1" / "airascore_ms_lesion_mask.space-tf.nii.gz",
            "transform": None,
        },
        "lesion_instances": {
            "fp": out_subject / "Lesion_Masks_to_T1" / f"airascore_ms_lesion_instances_to_{subject_id}__t1_warped.nii.gz",
            "transform": None,
        },
    }


def load_subject_metric(
    metric: str,
    subject_id: str,
    *,
    log_missing: bool = False,
    zero_to_nan: bool = True,
) -> np.ndarray | None:
    """Load one standardized metric map in subject space.

    Supported metrics include: `t1_pinn`, `t2_pinn`, `t1_miracle`, `t2_miracle`, `ai_map.unringed`,
    `fa`, `nufo`, and `theta_1`.
    """
    assets = get_subject_space_assets(subject_id)
    info = assets.get(metric)
    if info is None:
        return None
    fp = info.get("fp")
    if not isinstance(fp, Path):
        return None
    img = load_nifti(fp, log_missing=log_missing)
    if img is None:
        return None

    data = img.get_fdata()
    transform = get_subject_space_assets(
        subject_id).get(metric, {}).get("transform")
    if transform == "relax_rate":
        with np.errstate(divide="ignore", invalid="ignore"):
            data = np.where(data == 0, np.nan, 1000.0 / data)
    elif zero_to_nan:
        data = np.where(data == 0, np.nan, data)
    return data


def load_subject_wm_mask(
    subject_id: str,
    *,
    threshold: float = 0.9,
    prefer_mask_gt90: bool = False,
) -> np.ndarray | None:
    """Load WM mask in subject space, with configurable file preference."""
    candidates: list[str] = ["wm_mask_gt90", "wm_map"] if prefer_mask_gt90 else [
        "wm_map", "wm_mask_gt90"]
    assets = get_subject_space_assets(subject_id)
    for key in candidates:
        info = assets.get(key)
        if info is None:
            continue
        fp = info.get("fp")
        if not isinstance(fp, Path):
            continue
        if fp is None:
            continue
        img = load_nifti(fp)
        if img is None:
            continue
        return img.get_fdata() > threshold
    return None


def load_subject_lesion_mask(subject_id: str) -> np.ndarray | None:
    """Load lesion mask in subject space from supported naming variants."""
    assets = get_subject_space_assets(subject_id)
    for key in ("lesion_mask", "lesion_instances"):
        info = assets.get(key)
        if info is None:
            continue
        fp = info.get("fp")
        if not isinstance(fp, Path):
            continue
        if fp is None:
            continue
        img = load_nifti(fp)
        if img is not None:
            return img.get_fdata() > 0
    return None


def load_subject_parcellation(subject_id: str, parc_name: str) -> np.ndarray | None:
    """Load a FreeSurfer parcellation volume in subject space."""
    fp = paths.tf_results_pft / subject_id / "Register_Freesurfer" / \
        f"{subject_id}__{parc_name}_warped.nii.gz"
    img = load_nifti(fp)
    return None if img is None else img.get_fdata()


def load_subject_gm_mask(subject_id: str) -> np.ndarray | None:
    """Load cerebral GM mask in subject space."""
    gm_fp = paths.tf_results_pft / subject_id / \
        "Segment_Tissues" / f"{subject_id}__map_gm.nii.gz"
    gm_img = load_nifti(gm_fp)
    if gm_img is None:
        return None
    return gm_img.get_fdata() > 0.9


def build_subject_cerebral_wm_mask(
    subject_id: str,
    fa_limits: tuple[float, float] | None = None,
    r1_limits: tuple[float, float] | None = None,
    r2_limits: tuple[float, float] | None = None,
    exclude_lesions: bool = False,
    prefer_mask_gt90: bool = False,
    wm_threshold: float = 0.9,
) -> np.ndarray | None:
    """Build a reusable subject-space WM refinement mask.

    Base mask is WM (`wm_map`/`wm_mask_gt90`), optionally refined by:
    - FA bounds,
    - shared R1/R2 bounds,
    - lesion exclusion.
    """
    wm = load_subject_wm_mask(
        subject_id,
        threshold=wm_threshold,
        prefer_mask_gt90=prefer_mask_gt90,
    )
    if wm is None:
        return None

    mask = wm.copy()

    if fa_limits is not None:
        fa = load_subject_metric("fa", subject_id)
        if fa is None:
            return None
        fa_low, fa_high = fa_limits
        mask &= (fa > fa_low) & (fa <= fa_high)

    if r1_limits is not None:
        r1 = load_subject_metric("t1_pinn", subject_id)
        if r1 is None:
            return None
        r1_low, r1_high = r1_limits
        mask &= (r1 >= r1_low) & (r1 <= r1_high)

    if r2_limits is not None:
        r2 = load_subject_metric("t2_pinn", subject_id)
        if r2 is None:
            return None
        r2_low, r2_high = r2_limits
        mask &= (r2 >= r2_low) & (r2 <= r2_high)

    if exclude_lesions:
        lesion = load_subject_lesion_mask(subject_id)
        if lesion is not None and lesion.shape == mask.shape:
            print(f"Excluding lesion voxels from WM mask for subject {subject_id}")
            mask &= ~lesion

    return mask


# def build_subject_wm_mask(
#     subject_id: str,
#     *,
#     exclude_lesions: bool = False,
#     prefer_mask_gt90: bool = False,
#     wm_threshold: float = 0.9,
# ) -> np.ndarray | None:
#     """Build a reusable subject-space WM refinement mask.

#     Base mask is WM (`wm_map`/`wm_mask_gt90`), optionally refined by:
#     - FA bounds,
#     - shared R1/R2 bounds,
#     - lesion exclusion.
#     """
#     wm = load_subject_wm_mask(
#         subject_id,
#         threshold=wm_threshold,
#         prefer_mask_gt90=prefer_mask_gt90,
#     )
#     if wm is None:
#         return None

#     mask = wm.copy()

#     if exclude_lesions:
#         lesion = load_subject_lesion_mask(subject_id)
#         if lesion is not None and lesion.shape == mask.shape:
#             mask &= ~lesion

#     return mask


def mask_zero(data: np.ndarray) -> np.ndarray:
    """Replace exact zeros by NaN for cleaner plotting/statistics."""
    out = data.copy()
    out[out == 0] = np.nan
    return out


def extract_axial_slice(data: np.ndarray, slice_idx: int) -> np.ndarray:
    """Extract and orient one axial slice."""
    idx = max(0, min(slice_idx, data.shape[2] - 1))
    return np.rot90(data[:, :, idx])


def bin_by_angle(
    metric: np.ndarray,
    angle: np.ndarray,
    mask: np.ndarray,
    angle_bins: np.ndarray,
) -> np.ndarray:
    """Bin a metric array by angle using a boolean mask.

    For each consecutive bin defined by `angle_bins`, collects voxels where
    `mask` is True and `angle` falls within (start, end], then returns the
    nanmean across those voxels.
    """
    result = []
    for start, end in zip(angle_bins[:-1], angle_bins[1:]):
        vals = metric[mask & (angle > start) & (angle <= end)]
        result.append(np.nanmean(vals) if vals.size else np.nan)
    return np.array(result)


def load_wm_mask_common_space(threshold: float = 0.8) -> Optional[np.ndarray]:
    wm_fp = (
        paths.atlas_hcp1065
        / "mni_icbm152_nlin_asym_09a_nifti"
        / "mni_icbm152_nlin_asym_09a"
        / "mni_icbm152_wm_tal_nlin_asym_09a.nii"
    )
    if not wm_fp.exists():
        return None
    return nib.load(str(wm_fp)).get_fdata() > threshold


def load_subject_wm_mask_common_space(subject_id: str) -> Optional[np.ndarray]:
    wm_fp = (
        paths.out_common_space
        / subject_id
        / "WM_mask_to_ICBM152"
        / f"{subject_id}_mask_wm_gt90_space-ICBM152.nii.gz"
    )
    if not wm_fp.exists():
        return None
    return nib.load(str(wm_fp)).get_fdata().astype(bool)


def load_lesion_mask_common_space(subject_id: str) -> Optional[np.ndarray]:
    lesion_fp = (
        paths.out_common_space
        / subject_id
        / "Lesion_Masks_to_ICBM152"
        / "airascore_ms_lesion_instances_to_icbm152.nii.gz"
    )
    if not lesion_fp.exists():
        return None
    return nib.load(str(lesion_fp)).get_fdata() > 0


def load_t2flair_common_space(subject_id: str) -> Optional[np.ndarray]:
    flair_fp = (
        paths.out_common_space
        / subject_id
        / "T2FLAIR_to_ICBM152"
        / "airascore_ms_t2flair_to_icbm152.nii.gz"
    )
    if not flair_fp.exists():
        return None
    return nib.load(str(flair_fp)).get_fdata()


def build_combined_common_space_mask(
    wm_mask_template: Optional[np.ndarray],
    wm_mask_subject: Optional[np.ndarray],
    lesion_mask: Optional[np.ndarray],
    target_shape: tuple[int, int, int],
) -> Optional[np.ndarray]:
    """Build (template WM & subject WM) | lesion in common space."""
    combined_mask = None
    if wm_mask_template is not None and wm_mask_template.shape == target_shape:
        combined_mask = wm_mask_template.copy()
        if wm_mask_subject is not None and wm_mask_subject.shape == target_shape:
            combined_mask &= wm_mask_subject
        if lesion_mask is not None and lesion_mask.shape == target_shape:
            combined_mask |= lesion_mask
    elif lesion_mask is not None and lesion_mask.shape == target_shape:
        combined_mask = lesion_mask.copy()
    return combined_mask


def calculate_unsigned_zscore_map(
    subject_data: np.ndarray,
    hc_mean: np.ndarray,
    hc_std: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute absolute z-score map with NaN-safe handling and optional masking."""
    with np.errstate(divide="ignore", invalid="ignore"):
        zscore = (subject_data - hc_mean) / hc_std
    zscore[~np.isfinite(zscore)] = np.nan
    if mask is not None:
        zscore = np.where(mask, zscore, np.nan)
    return np.abs(zscore)


def extract_peak_vectors(peaks_data: np.ndarray) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Extract peak vectors from FODF peaks map in 4D/5D layouts."""
    if peaks_data.ndim == 5:
        peak1 = peaks_data[..., 0]
        peak2 = peaks_data[..., 1] if peaks_data.shape[4] > 1 else None
        peak3 = peaks_data[..., 2] if peaks_data.shape[4] > 2 else None
        return peak1, peak2, peak3

    peak1 = peaks_data[..., 0:3]
    peak2 = peaks_data[..., 3:6] if peaks_data.shape[-1] >= 6 else None
    peak3 = peaks_data[..., 6:9] if peaks_data.shape[-1] >= 9 else None
    return peak1, peak2, peak3


def load_or_compute_theta1_from_peaks(subject_id: str, save_if_missing: bool = True) -> tuple[np.ndarray, np.ndarray] | None:
    """Load theta_1 if available, otherwise compute it from FODF peaks.

    Returns (theta_1_map, affine), or None if neither theta_1 nor peaks are available.
    """
    fodf_dir = paths.tf_results_pft / subject_id / "FODF_Metrics"
    theta1_fp = fodf_dir / f"{subject_id}__theta_1.nii.gz"
    if theta1_fp.exists():
        nii = nib.load(str(theta1_fp))
        return nii.get_fdata(), nii.affine

    peaks_fp = fodf_dir / f"{subject_id}__peaks.nii.gz"
    if not peaks_fp.exists():
        return None

    peaks_nii = nib.load(str(peaks_fp))
    peaks_data = peaks_nii.get_fdata()
    peaks_affine = peaks_nii.affine

    peak1_vectors, _, _ = extract_peak_vectors(peaks_data)
    _, _, theta1 = calc_angles_from_v1(peak1_vectors, affine=peaks_affine)

    if save_if_missing:
        theta1_fp.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(theta1, peaks_affine), str(theta1_fp))

    return theta1, peaks_affine


# %% Figure 6


def get_bundle_masks(
    subject_id: str,
    bundle_patterns: tuple[str, ...],
) -> Dict[str, np.ndarray]:
    """Load BundleSeg NIfTI masks filtered by BUNDLE_PATTERNS.

    Masks are in subject (TractoFlow T1w) space and treated identically to
    FreeSurfer ROI masks — lesion voxels are excluded in compute_roi_means.
    Merges left/right into bilateral masks.
    """
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

    # Merge left/right into bilateral mask
    masks: Dict[str, np.ndarray] = {}
    for name, mask in raw.items():
        base = name[:-2] if name.endswith(("_L", "_R")) else name
        if base in masks:
            masks[base] = masks[base] | mask
        else:
            masks[base] = mask.copy()
    return masks


def build_roi_masks(
    subject_id: str,
    group: str,
    *,
    roi_labels: dict[str, list[int]],
    bundle_patterns: tuple[str, ...],
) -> Dict[str, np.ndarray]:
    """Build FreeSurfer ROI masks and BundleSeg NIfTI bundle masks (bilateral/combined).

    All masks share the TractoFlow T1w space so they can be combined with
    metric maps loaded by load_subject_metric.
    """
    frontal_wm_labels_l = roi_labels["frontal_wm_l"]
    frontal_wm_labels_r = roi_labels["frontal_wm_r"]
    mid_wm_labels_l = roi_labels["mid_wm_l"]
    mid_wm_labels_r = roi_labels["mid_wm_r"]
    posterior_wm_labels_l = roi_labels["posterior_wm_l"]
    posterior_wm_labels_r = roi_labels["posterior_wm_r"]
    cerebellar_wm_labels_l = roi_labels["cerebellar_wm_l"]
    cerebellar_wm_labels_r = roi_labels["cerebellar_wm_r"]
    thalamus_labels_l = roi_labels["thalamus_l"]
    thalamus_labels_r = roi_labels["thalamus_r"]
    caudate_labels_l = roi_labels["caudate_l"]
    caudate_labels_r = roi_labels["caudate_r"]
    cc_labels = roi_labels["cc"]

    masks: Dict[str, np.ndarray] = {}
    wmparc = load_subject_parcellation(subject_id, "wmparc")
    if wmparc is not None:
        masks["Cerebellar WM"] = np.isin(
            wmparc, cerebellar_wm_labels_l + cerebellar_wm_labels_r)
        masks["Frontal WM"] = np.isin(
            wmparc, frontal_wm_labels_l + frontal_wm_labels_r)
        masks["Temporo-Parietal WM"] = np.isin(
            wmparc, mid_wm_labels_l + mid_wm_labels_r)
        masks["Occipital-Parietal WM"] = np.isin(
            wmparc, posterior_wm_labels_l + posterior_wm_labels_r)
        masks["Thalamus"] = np.isin(
            wmparc, thalamus_labels_l + thalamus_labels_r)
        masks["Caudate"] = np.isin(
            wmparc, caudate_labels_l + caudate_labels_r)
        masks["CC"] = np.isin(wmparc, cc_labels)

    cerebral_wm_mask = build_subject_cerebral_wm_mask(
        subject_id,
        exclude_lesions=(group == "RRMS"),
    )
    if cerebral_wm_mask is not None:
        masks["Cerebral WM"] = cerebral_wm_mask

    cerebral_gm_mask = load_subject_gm_mask(subject_id)
    if cerebral_gm_mask is not None:
        masks["Cerebral GM"] = cerebral_gm_mask

    # Bundle NIfTI masks (lesion voxels excluded downstream in compute_roi_means)
    masks.update(get_bundle_masks(subject_id, bundle_patterns=bundle_patterns))

    return masks


# =============================================================================
# METRIC COMPUTATION
# =============================================================================


def compute_roi_means(
    subject_id: str,
    group: str,
    *,
    roi_labels: dict[str, list[int]],
    bundle_patterns: tuple[str, ...],
    lesion_roi_key: str,
    metrics: dict[str, dict[str, str]],
    selection: str = "all",
    include_bundles: bool = True,
    wm_r1_limits: tuple[float, float] | None = None,
    wm_r2_limits: tuple[float, float] | None = None,
    gm_r1_limits: tuple[float, float] | None = None,
    gm_r2_limits: tuple[float, float] | None = None,
) -> Dict[str, Dict[str, float]]:
    """Compute mean metrics for ROIs with selectable ROI subsets and thresholds.

    Parameters
    - selection: one of {"all", "wm_and_fs", "cerebral_wm"} selecting which
      ROI sets to include in the output.
    - include_bundles: when False, drop bundle masks (BundleSeg) from results.
    - wm_r1_limits / wm_r2_limits: optional PINN caps to apply only to WM ROIs.
    - gm_r1_limits / gm_r2_limits: optional PINN caps to apply only to Cerebral GM.

    Behaviour
    - WM ROIs (Cerebral WM and FreeSurfer WM groups) are thresholded using the
      shared R1/R2 bounds returned by `get_r1_r2_mask_limits()` (unchanged).
    - Cerebral GM, when present, will be thresholded using the provided
      `gm_r1_limits` / `gm_r2_limits` if supplied; otherwise it falls back to
      the shared R1/R2 bounds.
    """

    # Build raw ROI and bundle masks (these are in subject space)
    roi_masks = build_roi_masks(
        subject_id,
        group,
        roi_labels=roi_labels,
        bundle_patterns=bundle_patterns,
    )

    lesion_mask = load_subject_lesion_mask(
        subject_id) if group == "RRMS" else None

    if lesion_mask is not None:
        # Exclude lesion voxels from all ROI masks for RRMS subjects
        for roi, mask in list(roi_masks.items()):
            if mask.shape == lesion_mask.shape:
                roi_masks[roi] = mask & ~lesion_mask
            else:
                print(
                    f"  Lesion mask shape mismatch for {subject_id} in ROI {roi}; "
                    "skipping lesion exclusion")
        # Add lesion mask as a separate ROI (only for RRMS)
        roi_masks[lesion_roi_key] = lesion_mask

    # Refine each ROI/bundle mask with appropriate R1/R2 bounds. WM or GM refinement is applied based on the ROI type.
    r1 = load_subject_metric("t1_pinn", subject_id, log_missing=True)
    r2 = load_subject_metric("t2_pinn", subject_id, log_missing=True)
    if r1 is None or r2 is None:
        return {}

    # Global WM bounds
    r1_wm_min, r1_wm_max = wm_r1_limits
    r2_wm_min, r2_wm_max = wm_r2_limits
    wm_refinement = (
        np.isfinite(r1) & np.isfinite(r2)
        & (r1 >= r1_wm_min) & (r1 <= r1_wm_max)
        & (r2 >= r2_wm_min) & (r2 <= r2_wm_max)
    )

    # Global GM bounds
    r1_gm_min, r1_gm_max = gm_r1_limits
    r2_gm_min, r2_gm_max = gm_r2_limits
    gm_refinement = (
        np.isfinite(r1) & np.isfinite(r2)
        & (r1 >= r1_gm_min) & (r1 <= r1_gm_max)
        & (r2 >= r2_gm_min) & (r2 <= r2_gm_max)
    )

    # Define which wmparc keys are WM and which are GM-like.
    wmparc_wm_keys = {
        "Cerebellar WM",
        "Frontal WM",
        "Temporo-Parietal WM",
        "Occipital-Parietal WM",
        "CC",
    }
    gm_like_rois = {"Cerebral GM", "Thalamus", "Caudate"}

    for roi, mask in list(roi_masks.items()):
        if roi == lesion_roi_key:
            # Keep lesion ROI as the native lesion mask (no R1/R2 refinement).
            continue
        if mask.shape != wm_refinement.shape:
            print(
                f"  R1/R2 mask shape mismatch for {subject_id} in ROI {roi}; "
                "skipping this ROI")
            roi_masks.pop(roi)
            continue
        # Apply GM refinement for GM-like ROIs, global refinement for WM-like ROIs
        if roi in gm_like_rois:
            roi_masks[roi] = mask & gm_refinement
        else:
            roi_masks[roi] = mask & wm_refinement

    # Optionally drop bundle masks when requested. Bundle masks are those not
    # present in the known FreeSurfer / cerebral mask set and not the lesion key.
    if not include_bundles:
        known_keys = set(wmparc_wm_keys) | gm_like_rois | {
            "Cerebral WM", lesion_roi_key}
        for k in list(roi_masks.keys()):
            if k not in known_keys:
                roi_masks.pop(k, None)

    # Optionally select a subset of ROI masks for analysis
    if selection == "cerebral_wm":
        roi_masks = {"Cerebral WM": roi_masks.get(
            "Cerebral WM")} if "Cerebral WM" in roi_masks else {}
    elif selection == "wm_and_fs":
        desired = {"Cerebral WM"} | wmparc_wm_keys | gm_like_rois
        roi_masks = {k: v for k, v in roi_masks.items() if k in desired}

    results: Dict[str, Dict[str, float]] = {}

    for metric_key, meta in metrics.items():
        metric_label = meta["label"]
        data = load_subject_metric(metric_key, subject_id, log_missing=True)
        if data is None:
            continue
        for roi, mask in roi_masks.items():
            if mask.shape != data.shape:
                continue
            vals = data[mask]
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            results.setdefault(roi, {})[metric_label] = float(np.nanmean(vals))
    return results
