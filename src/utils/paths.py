"""
Path management utilities for the project.
Stores all important paths for saving figures, data, and other outputs.

Author: florian.birk@tuebingen.mpg.de
"""
from pathlib import Path

root = Path(__file__).resolve().parent.parent.parent.absolute()
data = root / "dat"
figures = root / "fig"
source = root / "src"
doc = root / "doc"
data_figures = data / "figures"
derivatives = root.parent / "derivatives"
neurodesk = root.parent.parent / "neurodesk"
data_nyx = Path("/home/fbirk/ptmp/data/2025-09-qmri-ms-3t")
colorresource = source / "colorresourcespy" / "py"

# default config
default_config = root / "src" / "utils" / "defaults.yaml"

# bSSFP dir
bssfp_data = derivatives / "2_bssfp"

# lesion masks
lesion_masks = derivatives / "1_flair_t1w_lesion_seg"
t2flair_airascore_source = root.parent / "source" / "MS_seg"
source_dcm = root.parent / "source" / "dcm"

# SPACES
common_space = derivatives / "common_space"
subject_space = derivatives / "subject_space"
common_space_nyx = data_nyx / "common_space"
subject_space_nyx = data_nyx / "subject_space"

# TRACTOFLOW DEFAULT PATHS
tf_root = derivatives / "0_tf_root"
tf_results_pft = derivatives / "0_tf_results_noABS_PFT_tracking"

# ATLASES
atlas_hcp1065 = common_space / "HCP1065_atlas"
atlas_bundleseg = common_space / "Bundleseg_atlases_10103446"
atlas_hcp1065_nyx = common_space_nyx / "HCP1065_atlas"

# OUTDIRS
out_common_space = common_space / "0.out"
out_subject_space = subject_space / "0.out"
out_common_space_nyx = common_space_nyx / "0.out"
out_subject_space_nyx = subject_space_nyx / "0.out"

# FASTRELAXDL
fastrelax = root / "dat" / "fastrelax"

### UNTIL 251127 ###
# for NYX cluster
# Tractometry_flow
nyx_tflow_container = root / "scilus_2.1.0.sif"
nyx_tflow_inp = Path("/home/fbirk/ptmp/data/2025-09-qmri-ms-3t/4_tractometry")
nyx_tflow_out = nyx_tflow_inp.parent / "4_tractometry_results"
stonecrop_tractometry_flow_out = derivatives / "4_tractometry_results"
# # Freesurfer
fs_container_stonecrop = neurodesk / "freesurfer_8.1.0_latest.sif"
# fs_out_stonecrop = derivatives / "1_fs_out"
