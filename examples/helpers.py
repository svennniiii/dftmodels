"""Shared visualization utilities for dftmodels example notebooks."""

import numpy as np
import matplotlib.pyplot as plt


def _whisker_ends(vals, q1, q3):
    """Furthest data point within 1.5 × IQR of each quartile (Tukey fences)."""
    iqr = q3 - q1
    lo = np.min(vals[vals >= q1 - 1.5 * iqr])
    hi = np.max(vals[vals <= q3 + 1.5 * iqr])
    return lo, hi


def violin_panel(
    ax,
    data,
    labels=None,
    crb=None,
    ylabel=None,
    title=None,
    legend=True,
):
    """
    Styled violin plot with quartile/whisker markers, optional reference lines.

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
        RNG seed for jitter scatter (unused, kept for API compatibility).
    legend : bool, optional
        Whether to add a legend (default True). Pass False to suppress.
    """
    positions = list(range(len(data)))
    arrays = [np.asarray(d, dtype=float) for d in data]

    parts = ax.violinplot(arrays, positions=positions, widths=0.7,
                          showmedians=False, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor("C0")
        pc.set_alpha(0.8)
        pc.set_edgecolor("C0")
        pc.set_linewidth(0.5)

    # Quartile boxes and whiskers
    q1s, medians, q3s = np.percentile(arrays, [14.9, 50, 84.1], axis=1)
    whiskers = np.array([_whisker_ends(arr, q1, q3)
                         for arr, q1, q3 in zip(arrays, q1s, q3s)])
    w_lo, w_hi = whiskers[:, 0], whiskers[:, 1]

    ax.vlines(positions, w_lo, w_hi, color="black", linewidth=1.2, alpha=0.8, zorder=3)
    ax.vlines(positions, q1s, q3s, color="black", linewidth=5, alpha=0.9, zorder=4)
    ax.scatter(positions, medians, marker="o", color="white",
               edgecolors="C0", linewidths=1.2, s=28, zorder=5)

    ax.axhline(0, color="black", ls="--", lw=0.8, alpha=0.4, label="Ground truth")

    if crb is not None:
        ax.axhline( crb, color="C1", ls="-.", lw=1.4, alpha=0.8, label="Cramér–Rao")
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
    if legend and (crb is not None):
        ax.legend()
