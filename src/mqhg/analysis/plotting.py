"""Plotting utilities for magic-gravity research.

Generates the key diagnostic plots specified in the research spec:
- Magic vs entanglement
- Magic vs backreaction proxy
- Entanglement vs backreaction proxy
- Non-local magic vs backreaction proxy
- Geometry smoothness vs magic density
- Phase diagrams
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def plot_magic_vs_backreaction(
    magic_values: list[float],
    backreaction_values: list[float],
    labels: list[str] | None = None,
    title: str = "Non-local Magic vs Backreaction",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Scatter plot of magic vs backreaction observable."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.scatter(magic_values, backreaction_values, alpha=0.7, edgecolors="k", linewidths=0.5)

    if labels:
        for i, label in enumerate(labels):
            ax.annotate(label, (magic_values[i], backreaction_values[i]),
                        fontsize=8, alpha=0.7)

    ax.set_xlabel("Non-local Magic (SRE₂)")
    ax.set_ylabel("Backreaction ||ΔD||_F")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Fit line
    if len(magic_values) > 2:
        coeffs = np.polyfit(magic_values, backreaction_values, 1)
        x_fit = np.linspace(min(magic_values), max(magic_values), 50)
        ax.plot(x_fit, np.polyval(coeffs, x_fit), "r--", alpha=0.5,
                label=f"Linear fit: slope={coeffs[0]:.3f}")
        ax.legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_geometry_deformation(
    distance_before: NDArray[np.float64],
    distance_after: NDArray[np.float64],
    title: str = "Geometry Deformation",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Visualize emergent metric before and after excitation."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    vmax = max(distance_before.max(), distance_after.max())

    im0 = axes[0].imshow(distance_before, cmap="viridis", vmin=0, vmax=vmax)
    axes[0].set_title("Before Excitation")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(distance_after, cmap="viridis", vmin=0, vmax=vmax)
    axes[1].set_title("After Excitation")
    plt.colorbar(im1, ax=axes[1])

    diff = distance_after - distance_before
    vabs = np.abs(diff).max()
    im2 = axes[2].imshow(diff, cmap="RdBu_r", vmin=-vabs, vmax=vabs)
    axes[2].set_title("Difference (ΔD)")
    plt.colorbar(im2, ax=axes[2])

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_phase_diagram(
    p_magic_values: list[float],
    smoothness_values: list[float],
    sre_values: list[float],
    entropy_values: list[float],
    title: str = "Phase Diagram: Magic Rate vs Observables",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot the phase diagram showing regimes of geometry behavior."""
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    axes[0].plot(p_magic_values, smoothness_values, "o-", color="tab:blue")
    axes[0].set_ylabel("Geometry Smoothness")
    axes[0].axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    axes[0].set_title("Smooth geometry regime")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(p_magic_values, sre_values, "s-", color="tab:red")
    axes[1].set_ylabel("Stabilizer Rényi Entropy")
    axes[1].set_title("Magic content")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(p_magic_values, entropy_values, "^-", color="tab:green")
    axes[2].set_ylabel("Mean Entanglement")
    axes[2].set_xlabel("p_magic (non-Clifford gate rate)")
    axes[2].set_title("Entanglement growth")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_sweep_results(
    sweep_data: list[dict],
    x_key: str = "phase",
    y_keys: list[str] | None = None,
    title: str = "Parameter Sweep",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Generic sweep plot from list of dicts."""
    if y_keys is None:
        y_keys = [k for k in sweep_data[0].keys() if k != x_key]

    n_plots = len(y_keys)
    fig, axes = plt.subplots(n_plots, 1, figsize=(8, 3 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]

    x_vals = [d[x_key] for d in sweep_data]

    colors = list(mcolors.TABLEAU_COLORS.values())
    for idx, key in enumerate(y_keys):
        y_vals = [d[key] for d in sweep_data]
        axes[idx].plot(x_vals, y_vals, "o-", color=colors[idx % len(colors)])
        axes[idx].set_ylabel(key)
        axes[idx].grid(True, alpha=0.3)

    axes[-1].set_xlabel(x_key)
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
