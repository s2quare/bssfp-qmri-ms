import numpy as np

def color_log_remap(ori_cmap, lo_lev, up_lev):
    """
    colorLogRemap: Lookup of the original color map table according to a "log-like" curve.
    The log-like curve contains a linear part and a logarithmic part; the size of the parts
    depends on the range (lo_lev, up_lev).

    Arguments:
        ori_cmap: Original colormap, provided as an N x 3 matrix (numpy array).
        lo_lev: Lower level of the range to be displayed.
        up_lev: Upper level of the range to be displayed.

    Returns:
        log_cmap: Modified colormap.
    """
    assert up_lev > 0, 'Upper level must be positive'
    assert up_lev > lo_lev, 'Upper level must be larger than lower level'

    map_length = ori_cmap.shape[0]
    e_inv = np.exp(-1.0)
    a_val = e_inv * up_lev
    m_val = max(a_val, lo_lev)
    b_val = (1.0 / map_length) + (a_val >= lo_lev) * ((a_val - lo_lev) / (2 * a_val - lo_lev))
    b_val += 1e-7  # Small adjustment to ensure rounding to the darkest valid color

    log_cmap = np.zeros_like(ori_cmap)
    log_cmap[0, :] = ori_cmap[0, :]

    log_portion = 1.0 / (np.log(m_val) - np.log(up_lev))

    for g in range(1, map_length):
        f = 0.0
        x = g * (up_lev - lo_lev) / map_length + lo_lev

        if x > m_val:
            # Logarithmic segment of the curve
            f = map_length * ((np.log(m_val) - np.log(x)) * log_portion * (1 - b_val) + b_val)
        else:
            if lo_lev < a_val and x > lo_lev:
                # Linear segment of the curve
                f = map_length * ((x - lo_lev) / (a_val - lo_lev) * (b_val - (1.0 / map_length))) + 1.0

            if x <= lo_lev:
                # Lowest valid color
                f = 1.0

        # Lookup from original color map
        log_cmap[g, :] = ori_cmap[min(map_length - 1, int(np.floor(f))), :]

    return log_cmap
