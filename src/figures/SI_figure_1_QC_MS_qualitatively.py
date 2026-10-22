"""
Supplementary Figure 1: Qualitative assessment of R1 and R2 maps in RRMS
subjects with white-matter outlier values.

The script generates representative axial map images and visualizes the
associated image-quality findings.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Add src to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

try:
    from utils import paths
    from utils.paper_figure_helpers import (
        get_paper_figure_cache_dir,
        load_nifti,
    )
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

exp_name = Path(__file__).stem
OUTPUT_DIR = paths.root / "fig" / exp_name
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = get_paper_figure_cache_dir(exp_name)

METRICS = {
    "r1": {"label": r"$\mathbf{R_1}$ [1/s]"},
    "r2": {"label": r"$\mathbf{R_2}$ [1/s]"},
}

BAR_COLORS = {"HC": "tab:blue", "RRMS": "tab:orange"}


def get_relaxometry_maps_per_subject(subject_id: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the T1 and T2 maps in subject space from one subject. 
    """
    sub_fop = paths.out_subject_space / subject_id
    qmri_fop = sub_fop / "bSSFP_to_T1_qMRI"
    try:
        print(f"Processing subject {subject_id}...")
        print(f"Looking for T1 and T2 maps in {qmri_fop}...")
        t1_fp = list(qmri_fop.glob("t1_pinn*.nii.gz"))[0]
        t2_fp = list(qmri_fop.glob("t2_pinn*.nii.gz"))[0]
        t1_map = load_nifti(t1_fp).get_fdata()
        t2_map = load_nifti(t2_fp).get_fdata()

        # convert T1 and T2 maps to R1 and R2 maps
        r1_map = 1000.0 / t1_map
        r2_map = 1000.0 / t2_map

    except Exception as e:
        print(f"Error processing subject {subject_id}: {e}")
        r1_map = np.array([])
        r2_map = np.array([])

    return r1_map, r2_map


def plot_axial_slice_all_four_subjects(wm_data: Dict[str, Dict[str, np.ndarray]], metric: str, number_of_slices: int = 5):
    """
    Plot exemplary axial slices of the specified metric (R1 or R2) for each subject. default should be 5 slices per subject
    """

    fig, axs = plt.subplots(len(wm_data), number_of_slices, figsize=(12, 12))

    for subject_id, data in wm_data.items():
        if metric == "r1":
            map_data = data["r1"]
            vmin, vmax = 0, 4
            cbar_label = r"$\mathbf{R_1}$ [1/s]"
        elif metric == "r2":
            map_data = data["r2"]
            vmin, vmax = 5, 30
            cbar_label = r"$\mathbf{R_2}$ [1/s]"
        else:
            raise ValueError(f"Unknown metric: {metric}")
        center_slice = wm_data[subject_id]['slice']
        slice_step_size = 5
        all_slices = list(range(center_slice - (number_of_slices // 2) * slice_step_size,
                                center_slice + (number_of_slices // 2) * slice_step_size + 1, slice_step_size))

        # rot90 on data
        map_data = np.rot90(map_data)

        for i, slice in enumerate(all_slices):
            axial_slice = map_data[:, :, slice]

            ax = axs.flatten()[list(wm_data.keys()).index(
                subject_id) * number_of_slices + i]
            im = ax.imshow(axial_slice, cmap='gray', vmin=vmin, vmax=vmax)
            # if i == 2:  # only set title for the center slice
            #     ax.set_title(f"{data['suptitle']}", fontsize=14)

            # set the color of the frame around the image based on the subject
            for spine in ax.spines.values():
                spine.set_edgecolor(data["frame_color"])
                spine.set_linewidth(3)

            # remove the axis ticks
            ax.set_xticks([])
            ax.set_yticks([])

    plt.subplots_adjust(right=0.9)
    colorbar_ax = fig.add_axes([0.91, 0.15, 0.03, 0.6])
    cbar = fig.colorbar(im, cax=colorbar_ax)
    cbar.set_label(cbar_label, fontsize=16, fontweight='bold')
    # colorbar tick label fontsize
    cbar.ax.tick_params(labelsize=14)
    output_fp = OUTPUT_DIR / f"{metric}_qualitative_maps.svg"
    plt.savefig(output_fp)
    plt.close()


def main():
    subject_list = ["S27", "S29", "S19", "S31"]
    center_slice_list = [92, 94, 84, 90]
    frame_color = ["tab:green", "tab:orange", "tab:red",  "tab:red"]
    suptitles = [
        "Valid", "Exclusion (Noise)", "Exclusion (Movement)", "Exclusion (Movement)"]

    wm_data = {}
    for i, subject_id in enumerate(subject_list):
        r1_wm, r2_wm = get_relaxometry_maps_per_subject(subject_id)
        wm_data[subject_id] = {
            "slice": center_slice_list[i], "r1": r1_wm, "r2": r2_wm, "frame_color": frame_color[i], "suptitle": suptitles[i]}

    # plot the R1 and R2 maps for each subject
    for metric in ["r1", "r2"]:
        plot_axial_slice_all_four_subjects(wm_data, metric)


if __name__ == "__main__":
    main()
