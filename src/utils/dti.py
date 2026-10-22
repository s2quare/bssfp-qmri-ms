"""
Functions used for processing DTI data.
"""

import nibabel as nib
import numpy as np
from pathlib import Path


def calc_angles_from_v1(v1_fp_or_data, affine=None):
    """Calculate the inclination angle theta between direction vectors and the main magnetic field B0.

    To account for oblique slice orientations (not strict axial) during acquisition,
    the vectors are rotated back into the scanner reference space.

    This function can be called in two ways:
    1. With a file path: calc_angles_from_v1(v1_fp) - loads NIfTI and extracts data + affine
    2. With array + affine: calc_angles_from_v1(vectors, affine=affine) - uses provided data

    Args:
        v1_fp_or_data: Either a Path to the V1 NIfTI file, or a 4D array of shape (X, Y, Z, 3)
                       containing direction vectors.
        affine: The affine matrix (required when v1_fp_or_data is an array, ignored for file path).

    Returns:
        azimuth (np.ndarray): Azimuth angle in degrees.
        elevation (np.ndarray): Elevation angle in degrees.
        theta (np.ndarray): Inclination angle (theta) in degrees - angle to B0 field.

    """
    # Determine input type: file path or array
    if isinstance(v1_fp_or_data, (str, Path)):
        # File path mode
        v1_nii = nib.load(v1_fp_or_data)
        v1_data = v1_nii.get_fdata()
        v1_affine = v1_nii.affine[:3, :3]
    else:
        # Array mode
        v1_data = v1_fp_or_data
        if affine is None:
            raise ValueError("affine matrix is required when providing an array")
        v1_affine = affine[:3, :3]

    # QR decomposition of the affine matrix into a rotation matrix q and a upper triangular matrix r
    q, r = np.linalg.qr(v1_affine)

    if v1_data.ndim != 4:
        raise ValueError("Input v1 must be a 4D array")

    # Apply the rotation matrix with normalization
    v1_rot = np.zeros_like(v1_data)
    for i in range(v1_data.shape[0]):
        for j in range(v1_data.shape[1]):
            for k in range(v1_data.shape[2]):
                v1_temp = v1_data[i, j, k, :]
                if not np.all(v1_temp == 0):
                    v1_rot[i, j, k, :] = q @ v1_temp

    # separate into x, y, z components
    v1_rot = np.abs(v1_rot)
    v1_x = v1_rot[..., 0]
    v1_y = v1_rot[..., 1]
    v1_z = v1_rot[..., 2]

    # get spherical coord
    azimuth = np.arctan2(v1_y, v1_x)
    elevation = np.arctan2(v1_z, np.sqrt(v1_x**2 + v1_y**2))
    inclination = np.arccos(v1_z)

    # convert to deg
    azimuth = np.degrees(azimuth)
    elevation = np.degrees(elevation)
    inclination = np.degrees(inclination)

    return azimuth, elevation, inclination