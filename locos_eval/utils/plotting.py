"""Plotting utilities with consistent styling for the project.

Conventions enforced by this module:
- No in-figure titles (titles belong in LaTeX captions)
- Legend always saved as a separate file alongside the main figure
- Default linewidth of 2 for lines, borders, and patches
"""

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

FIGURE_SIZE = (5.5, 3.5)
FONT_SIZE_AXIS_LABEL = 12
FONT_SIZE_AXIS_TICKS = 10
FONT_SIZE_LEGEND = 10
LINE_WIDTH = 2.0


MODEL_PRETTY_NAMES = {
    "Qwen3-4B": "Qwen3-4B",
    "Qwen3-8B": "Qwen3-8B",
    "Qwen3-14B": "Qwen3-14B",
    "Qwen3-32B": "Qwen3-32B",
    "Olmo-3-7B-Instruct": "Olmo3-7B",
    "Olmo-3.1-32B-Instruct": "Olmo3.1-32B",
    "Meta-Llama-3-8B-Instruct": "Llama3-8B",
    "gemma-3-4b-it": "Gemma3-4B",
    "gemma-3-12b-it": "Gemma3-12B",
    "gemma-3-27b-it": "Gemma3-27B",
}


def setup_plot_style():
    """Set up consistent plotting style for all visualizations in the project."""
    rc_params = plt.rcParams

    sns.set_theme(style="whitegrid")
    rc_params["text.usetex"] = False
    rc_params["font.size"] = "12.5"
    rc_params["figure.dpi"] = 190
    rc_params["axes.unicode_minus"] = False
    rc_params["font.family"] = "cmr10"
    rc_params["mathtext.fontset"] = "cm"
    rc_params["axes.formatter.use_mathtext"] = True

    # No in-figure titles — titles belong in LaTeX captions
    rc_params["axes.titley"] = None
    rc_params["axes.titlesize"] = 0
    rc_params["figure.titlesize"] = 0

    # Border color and ticks
    rc_params["axes.edgecolor"] = "black"
    rc_params["axes.linewidth"] = LINE_WIDTH

    # Black border on all bars (patches)
    rc_params["patch.edgecolor"] = "black"
    rc_params["patch.linewidth"] = LINE_WIDTH
    rc_params["patch.force_edgecolor"] = True
    rc_params["xtick.color"] = "black"
    rc_params["ytick.color"] = "black"

    # General linewidth
    rc_params["lines.linewidth"] = LINE_WIDTH
    rc_params["lines.markeredgewidth"] = LINE_WIDTH

    # Background color
    # rc_params["axes.facecolor"] = "#E5E4DF"
    rc_params["axes.facecolor"] = "None"

    # Grid
    rc_params["grid.color"] = "#FAFAF7"
    rc_params["grid.alpha"] = 0.5
    rc_params["grid.linewidth"] = 0.5
    rc_params["grid.linestyle"] = "-"

    # Ticks
    rc_params["xtick.bottom"] = False
    rc_params["ytick.left"] = False

    sns.set_context(context="talk", font_scale=0.9)

    # Standardized font sizes
    rc_params["axes.labelsize"] = FONT_SIZE_AXIS_LABEL
    rc_params["axes.labelpad"] = 0
    rc_params["xtick.labelsize"] = FONT_SIZE_AXIS_TICKS
    rc_params["ytick.labelsize"] = FONT_SIZE_AXIS_TICKS
    rc_params["xtick.major.pad"] = 1
    rc_params["ytick.major.pad"] = 1
    rc_params["legend.fontsize"] = FONT_SIZE_LEGEND
    rc_params["figure.figsize"] = FIGURE_SIZE


def facecolor_alpha(color, alpha: float):
    """Return an RGBA tuple with alpha applied only to the facecolor.

    Use instead of the `alpha` kwarg when you want a transparent fill but a
    fully opaque (black) border on the same patch.
    """
    import matplotlib.colors as mcolors

    r, g, b, _ = mcolors.to_rgba(color)
    return (r, g, b, alpha)


def save_figure(fig, path, keep_title=False, **kwargs):
    """Save figure as SVG and auto-save its legend as a sibling file.

    The legend is removed from the main figure and saved separately as
    ``<stem>_legend.svg`` next to the main file.  Titles are also stripped
    by default (titles belong in LaTeX captions, not inside the figure).

    Args:
        keep_title: If True, preserve any title set on axes/figure.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Strip titles (unless caller explicitly wants to keep them)
    if not keep_title:
        for ax in fig.get_axes():
            ax.set_title("")
        fig.suptitle("")

    # Collect legend handles/labels and remove legends from the figure
    handles, labels = [], []
    legend_kwargs = {}
    for ax in fig.get_axes():
        leg = ax.get_legend()
        if leg is not None:
            h, l = ax.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)
            ncol = getattr(leg, "_ncols", 1)
            legend_kwargs["ncol"] = ncol
            leg.remove()
    # fig.get_legend() requires matplotlib >= 3.8; fall back to .legends list
    fig_leg = None
    if hasattr(fig, "get_legend"):
        fig_leg = fig.get_legend()
    elif hasattr(fig, "legends") and fig.legends:
        fig_leg = fig.legends[0]
    if fig_leg is not None:
        if not handles:
            handles, labels = fig_leg.get_legend_handles_labels()
            ncol = getattr(fig_leg, "_ncols", 1)
            legend_kwargs["ncol"] = ncol
        fig_leg.remove()

    # Save main figure
    fig.patch.set_alpha(0)
    fig.savefig(path, format="svg", bbox_inches="tight", pad_inches=0.02, **kwargs)
    plt.close(fig)

    # Save legend as a separate file
    if handles:
        legend_path = path.with_name(f"{path.stem}_legend{path.suffix}")
        save_legend(handles, labels, legend_path, **legend_kwargs)


def save_legend(handles, labels, path, ncol=1, **kwargs):
    """Save legend as a separate SVG file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(0.1, 0.1))
    ax.axis("off")
    # legend = ax.legend(handles, labels, loc="center", ncol=ncol, fontsize=FONT_SIZE_LEGEND, frameon=True, **kwargs)
    legend = ax.legend(handles, labels, loc="center", ncol=ncol, fontsize=FONT_SIZE_LEGEND, frameon=False, **kwargs)
    fig.savefig(
        path,
        format="svg",
        bbox_inches=legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted()),
        pad_inches=0.1,
        transparent=True,
    )
    plt.close(fig)
