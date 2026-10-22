import numpy as np
import os
from .colorLogRemap import color_log_remap  # Use relative import if the module is in the same package
from matplotlib.colors import ListedColormap

def relaxation_color_map(maptype, x, loLev, upLev):
    """
    RelaxationColorMap: acts in two ways:
      1. Generate a colormap to be used on display, given image type 
         (which must be one of 
         "T1", "R1", "T2", "T2*", "R2", "R2*", "T1rho", "T1ρ", "R1rho", "R1ρ", 
         "t1", "r1", "t2", "t2*", "r2", "r2*", "t1rho", "t1ρ", "r1rho", "r1ρ")
         and given the range of the image to be displayed.
      2. Generate a 'clipped' image, which is a copy of the input image except that values are clipped to the lower level,
         while respecting the special value of 0 (which has to map to the "invalid" color).
    INPUTS:
        maptype: a string from the aforementioned series, e.g., "T1" or "R2"
        x      : ND array containing the image to be displayed
        loLev  : lower level of the range to be displayed
        upLev  : upper level of the range to be displayed
    OUTPUTS:  
        xClip  : value-clipped image with the same size as x
        lutCmap: 256 by 3 colormap to be used in image-display functions (in Colors.RGB format)
    """
    # Determine the file path for the colormap
    script_dir = os.path.dirname(os.path.abspath(__file__))
    maptype = maptype.capitalize()
    
    if maptype in {'T1', 'R1'}:
        colormap_file = os.path.join(script_dir, 'lipari.csv')
    elif maptype in {'T2', 'T2*', 'R2', 'R2*', 'T1rho', 'T1ρ', 'R1rho', 'R1ρ'}:
        colormap_file = os.path.join(script_dir, 'navia.csv')
    else:
        raise ValueError("Expect 'T1', 'T2', 'R1', or 'R2' as maptype")
    
    # Read the colormap file
    colortable = np.loadtxt(colormap_file, delimiter=' ')
    
    # Reverse the colortable if maptype starts with 'R'
    if maptype.startswith('R'):
        colortable = np.flipud(colortable)
    
    # Set the 'invalid value' color
    colortable[0, :] = 0.0

    # Modification of the image to be displayed
    eps = (upLev - loLev) / colortable.shape[0]

    if loLev < 0:
        xClip = np.where(x < eps, loLev - eps, x)
    else:
        xClip = np.where(
            x < eps, loLev - eps,
            np.where(
                x < loLev + eps, loLev + 1.5 * eps,
                x
            )
        )
    
    # Generate the colormap
    lutCmap = color_log_remap(colortable, loLev, upLev)
    
    return xClip, lutCmap

def generate_relaxation_colormap(colormap_file, loLev, upLev, flip_colortable=False):
    """
    GenerateRelaxationColormap: Generate a colormap to be used on display, given the file path 
    to the colormap CSV and the range of the image to be displayed.
    INPUTS:
        colormap_file  : file path to the colormap CSV
        loLev          : lower level of the range to be displayed
        upLev          : upper level of the range to be displayed
        flip_colortable: boolean flag to indicate whether to flip the colortable or not
    OUTPUTS:  
        lutCmap: 256 by 3 colormap to be used in image-display functions (in Colors.RGB format)
    """
    # Read the colormap file
    colortable = np.loadtxt(colormap_file, delimiter=' ')
    
    # Flip the colortable if the flag is set
    if flip_colortable:
        colortable = np.flipud(colortable)
    
    # Set the 'invalid value' color
    colortable[0, :] = 0.0

    # Generate the colormap
    lutCmap = color_log_remap(colortable, loLev, upLev)
    
    return ListedColormap(lutCmap)
