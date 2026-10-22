#%%
from pathlib import Path
import h5py
import matplotlib.pyplot as plt
from relaxationColorMap import relaxation_color_map
from matplotlib.colors import ListedColormap

# Load the MATLAB file containing an example T1 map
fn = Path(__file__).parent / 'sampleT1map.mat'
with h5py.File(fn, 'r') as f:
	data = {key: f[key][()] for key in f.keys()}
im = data['sampleT1']

loLev = 400.0
upLev = 2000.0

# Call the relaxation_color_map function
im_clip, rgb_vec = relaxation_color_map('T1', im, loLev, upLev)

# Create a custom colormap from the rgb_vec array
custom_cmap = ListedColormap(rgb_vec)

#%% Display the image using matplotlib with the custom colormap
plt.figure()
plt.imshow(im_clip, vmin=loLev, vmax=upLev, cmap=custom_cmap)
plt.colorbar()
plt.title('Relaxation Color Map')

#%% save the im as numpy array
import numpy as np
np.save(Path(__file__).parent / 'sampleT1map.npy', np.rot90(im, k=-1))