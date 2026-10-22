"""
Shared matplotlib configuration for consistent figure styling across the project.

Usage:
    from utils.plotting import apply_figure_style, add_subplot_label, create_figure
    apply_figure_style()  # Call once at module level or before creating figures
    fig = create_figure(height=6)  # Creates figure with A4 width
"""

import matplotlib.pyplot as plt

# DIN A4 dimensions in inches (210 mm × 297 mm)
A4_WIDTH_MM = 210
A4_HEIGHT_MM = 297
MM_TO_INCH = 1 / 25.4
A4_WIDTH_INCH = A4_WIDTH_MM * MM_TO_INCH  # ~8.27 inches
A4_HEIGHT_INCH = A4_HEIGHT_MM * MM_TO_INCH  # ~11.69 inches

# Default figure style parameters
FIGURE_STYLE = {
    "font.size": 10,
    "font.family": "sans-serif",
    "axes.labelsize": 10,
    "axes.labelweight": "bold",
    "axes.titlesize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 9,
    "figure.dpi": 300,
}


def apply_figure_style(custom_params: dict | None = None) -> None:
    """Apply project-wide matplotlib style.

    Args:
        custom_params: Optional dict to override default style parameters.
    """
    params = FIGURE_STYLE.copy()
    if custom_params:
        params.update(custom_params)
    plt.rcParams.update(params)


def create_figure(
    height: float | None = None,
    aspect_ratio: float | None = None,
    width: float = A4_WIDTH_INCH,
    **kwargs,
) -> plt.Figure:
    """Create a figure with consistent width (default: A4 width = 210 mm).

    Specify either `height` (in inches) or `aspect_ratio` (width/height).
    If neither is given, defaults to a 4:3 aspect ratio.

    Args:
        height: Figure height in inches.
        aspect_ratio: Width-to-height ratio (e.g., 16/9, 4/3).
        width: Figure width in inches (default: A4 width ~8.27 inches).
        **kwargs: Additional arguments passed to plt.figure().

    Returns:
        Matplotlib Figure object.
    """
    if height is None:
        if aspect_ratio is None:
            aspect_ratio = 16 / 9  # default aspect ratio
        height = width / aspect_ratio

    return plt.figure(figsize=(width, height), **kwargs)


# Default subplot label position (axes coordinates)
# x=0 aligns with y-axis, y=1 is top of subplot, va="bottom" puts text above subplot
SUBPLOT_LABEL_X = 0.0
SUBPLOT_LABEL_Y = 1.0


def add_subplot_label(
    ax,
    label: str,
    x: float | None = None,
    y: float | None = None,
    colspan: int = 1,
    fontsize: int = 12,
    fontweight: str = "bold",
) -> None:
    """Add a subplot label (e.g., '(A)', '(B)') to the top-left of an axis.

    Labels are positioned at the top-left corner of the subplot, aligned with
    the y-axis. For multi-column axes, use `colspan` to keep alignment with
    single-column axes.

    Args:
        ax: Matplotlib Axes object.
        label: The label text, e.g. "(A)".
        x: Horizontal position in axes coordinates. If None, uses default (0.0)
           adjusted for colspan.
        y: Vertical position in axes coordinates. If None, uses default (1.0).
        colspan: Number of columns this axis spans. Used to scale x-offset
                 so labels align vertically across rows.
        fontsize: Font size for the label.
        fontweight: Font weight for the label.
    """
    if x is None:
        x = SUBPLOT_LABEL_X / colspan if colspan > 1 else SUBPLOT_LABEL_X
    if y is None:
        y = SUBPLOT_LABEL_Y

    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight=fontweight,
        va="bottom",  # bottom of text aligns with top of subplot
        ha="left",
    )
