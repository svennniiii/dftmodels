"""Shared visualization utilities for dftmodels example notebooks."""

import numpy as np
import matplotlib.pyplot as plt


def violin_panel(
    ax,
    data,
    labels=None,
    noise_floor=None,
    crb=None,
    ylabel=None,
    title=None,
    seed=None,
    legend=True,
):
    """
    Styled violin plot with jitter, zero line, and optional reference lines.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    data : list of array-like
        One array per violin.
    labels : list of str, optional
        x-tick labels. If None, ticks are suppressed.
    noise_floor : float, optional
        Draws ±noise_floor reference lines (black dotted).
    crb : float, optional
        Draws ±crb reference lines (C1 dash-dot).
    ylabel : str, optional
    title : str, optional
    seed : int, optional
        RNG seed for jitter scatter.
    legend : bool, optional
        Whether to add a legend (default True). Pass False to suppress.
    """
    positions = list(range(len(data)))

    parts = ax.violinplot(data, positions=positions, widths=0.7,
                          showmedians=True, showextrema=True)
    for pc in parts["bodies"]:
        pc.set_facecolor("C0")
        pc.set_alpha(0.5)
        pc.set_edgecolor("black")
        pc.set_linewidth(0.8)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(2)
    for key in ("cmaxes", "cmins", "cbars"):
        parts[key].set_color("gray")
        parts[key].set_linewidth(0.8)

    rng = np.random.default_rng(seed=seed)
    for k, arr in enumerate(data):
        jitter = rng.uniform(-0.1, 0.1, size=len(arr))
        ax.scatter(positions[k] + jitter, arr, s=3, color="C0", alpha=0.25, zorder=5)

    ax.axhline(0, color="black", ls="--", lw=0.8, alpha=0.4)

    if noise_floor is not None:
        ax.axhline( noise_floor, color="black", ls=":", lw=1.4, alpha=0.7, label="Noise floor")
        ax.axhline(-noise_floor, color="black", ls=":", lw=1.4, alpha=0.7)

    if crb is not None:
        ax.axhline( crb, color="C1", ls="-.", lw=1.4, alpha=0.8, label="CRB")
        ax.axhline(-crb, color="C1", ls="-.", lw=1.4, alpha=0.8)

    if labels is not None:
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=30, ha="right")
    else:
        ax.set_xticks([])

    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if legend and (noise_floor is not None or crb is not None):
        ax.legend(fontsize=8)
