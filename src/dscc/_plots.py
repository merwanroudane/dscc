"""Journal-style figures. Parula is the default colour scale for every surface and heatmap.

Every function returns the matplotlib ``Figure`` so it can be restyled or saved.

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from ._colors import get_cmap, parula_colors
from .elliptope import elliptope_boundary

__all__ = ["journal_style", "save_figure", "plot_elliptope", "plot_slice_coverage",
           "plot_coverage_grid", "plot_correlations", "plot_eigenvalues", "plot_wealth",
           "plot_correlation_heatmap"]

_RED, _GREEN, _BLUE, _GREY = "#C0392B", "#2E8B57", "#2563EB", "#6B7280"


def journal_style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": "#E5E7EB",
        "grid.linewidth": 0.6, "axes.edgecolor": "#374151", "figure.dpi": 110,
        "savefig.dpi": 300, "savefig.bbox": "tight", "lines.linewidth": 1.2,
    })


def save_figure(fig, path_stem: str):
    """Save ``path_stem`` as both vector PDF and 300-dpi PNG."""
    fig.savefig(f"{path_stem}.pdf")
    fig.savefig(f"{path_stem}.png", dpi=300)


def _ellipse_xy(c: float, n: int = 400):
    """Boundary of the section of E_3 at a fixed correlation c (other two coordinates)."""
    th = np.linspace(0, 2 * np.pi, n)
    a, b = np.sqrt(1 + c), np.sqrt(1 - c)
    u, v = a * np.cos(th), b * np.sin(th)
    return (u + v) / np.sqrt(2), (u - v) / np.sqrt(2)


def plot_elliptope(n: int = 81, levels=(0.0, 0.25, 0.75), colorscale: str = "Parula", figsize=(5.2, 4.6)):
    """Figure 1: the elliptope E_3 with the sections at the given omega_3 levels."""
    journal_style()
    g = np.linspace(-1, 1, n)
    W1, W2 = np.meshgrid(g, g)
    lo, hi = elliptope_boundary(W1, W2)
    cmap = get_cmap(colorscale)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection="3d")
    for Z in (lo, hi):
        ax.plot_surface(W1, W2, Z, facecolors=cmap((Z + 1) / 2), rstride=2, cstride=2,
                        linewidth=0, antialiased=True, alpha=0.55, shade=False)
    for c, col in zip(levels, ["black", _BLUE, _RED]):
        x, y = _ellipse_xy(c)
        ax.plot(x, y, np.full_like(x, c), color=col, lw=1.6, label=rf"$\omega_3={c}$")
    ax.set_xlabel(r"$\omega_1$"); ax.set_ylabel(r"$\omega_2$"); ax.set_zlabel(r"$\omega_3$")
    ax.view_init(22, -55)
    ax.legend(loc="upper left", frameon=False)
    return fig


def plot_slice_coverage(points: np.ndarray, lb: float, ub: float, slice_on: int = 2, ax=None,
                        max_points: int = 20000, show_projection: bool = True):
    """Figure 3: model points in one slice (red) against the maximal region (green)."""
    journal_style()
    other = [i for i in range(3) if i != slice_on]
    m = (points[:, slice_on] > lb) & (points[:, slice_on] < ub)
    P = points[m][:, other]
    if P.shape[0] > max_points:
        P = P[np.random.default_rng(0).choice(P.shape[0], max_points, replace=False)]
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(3.6, 3.6))
    if show_projection:
        from scipy.spatial import ConvexHull
        pts = np.vstack([np.c_[_ellipse_xy(c, 200)] for c in np.linspace(lb, ub, 60)])
        hull = ConvexHull(pts)
        ax.add_patch(Polygon(pts[hull.vertices], closed=True, color=_GREEN, alpha=0.35, lw=0))
    x, y = _ellipse_xy(lb)
    ax.fill(x, y, color=_GREEN, alpha=0.35, lw=0)
    ax.scatter(P[:, 0], P[:, 1], s=0.4, color=_RED, alpha=0.6, rasterized=True)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_aspect("equal")
    ax.set_title(rf"$\omega_{slice_on + 1}\in({lb:.3f},\,{ub:.3f})$")
    return fig if fig is not None else ax.figure


def plot_coverage_grid(points: np.ndarray, n_slices: int = 8, slice_on: int = 2, figsize=(9.5, 5.0)):
    """Figure 2: model coverage (red) versus the elliptope (green) for each slice."""
    journal_style()
    edges = np.linspace(0, 1, n_slices + 1)
    ncol = int(np.ceil(n_slices / 2))
    fig, axes = plt.subplots(2, ncol, figsize=figsize)
    for a, lb, ub in zip(axes.ravel(), edges[:-1], edges[1:]):
        plot_slice_coverage(points, lb, ub, slice_on, ax=a, max_points=6000)
    fig.tight_layout()
    return fig


def plot_correlations(res, index=None, pairs=None, figsize=(7.0, 3.2)):
    """Time paths of the conditional correlations Omega_t (2.14)."""
    journal_style()
    R = res.correlation()
    k = R.shape[1]
    names = res.model.names
    pairs = pairs or [(i, j) for i in range(k) for j in range(i + 1, k)]
    cols = parula_colors(max(len(pairs), 2))
    x = res.model.index if index is None else index
    fig, ax = plt.subplots(figsize=figsize)
    for c, (i, j) in zip(cols, pairs):
        ax.plot(x[1:], R[1:, i, j], color=c, lw=0.8, label=f"{names[i]}-{names[j]}")
    ax.set_ylabel("conditional correlation"); ax.set_ylim(-1, 1)
    ax.legend(ncol=min(5, len(pairs)), frameon=False, loc="lower left")
    return fig


def plot_eigenvalues(res, index=None, figsize=(7.0, 3.0)):
    """Eigenvalue paths Lambda_t (log scale)."""
    journal_style()
    x = res.model.index if index is None else index
    cols = parula_colors(max(res.k, 2))
    fig, ax = plt.subplots(figsize=figsize)
    for i in range(res.k):
        ax.plot(x[1:], res.lam[1:, i], color=cols[i], lw=0.8, label=rf"$\lambda_{{{i + 1}t}}$")
    ax.set_yscale("log"); ax.set_ylabel(r"$\Lambda_t$")
    ax.legend(ncol=res.k, frameon=False)
    return fig


def plot_wealth(bt, initial: float = 100.0, log: bool = True, exclude=("1/N",), figsize=(7.0, 3.4)):
    """Figures 4-5: value of ``initial`` invested at the start of the evaluation period."""
    journal_style()
    Wl = bt.wealth(initial)
    cols = [c for c in Wl if c not in exclude]
    pal = parula_colors(max(len(cols), 2))
    fig, ax = plt.subplots(figsize=figsize)
    for c, col in zip(cols, pal):
        ax.plot(Wl.index, Wl[c], color=col, lw=1.1, label=c)
    if log:
        ax.set_yscale("log")
    ax.set_ylabel(f"value of {initial:g} invested")
    ax.legend(frameon=False)
    return fig


def plot_correlation_heatmap(R: np.ndarray, names=None, colorscale: str = "Parula", figsize=(4.0, 3.4)):
    """Heatmap of one correlation matrix (default Parula colour scale)."""
    journal_style()
    k = R.shape[0]
    names = names or [f"x{i + 1}" for i in range(k)]
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(R, cmap=get_cmap(colorscale), vmin=-1, vmax=1)
    ax.set_xticks(range(k), names, rotation=45, ha="right"); ax.set_yticks(range(k), names)
    ax.grid(False)
    for i in range(k):
        for j in range(k):
            ax.text(j, i, f"{R[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if R[i, j] < 0.3 else "black")
    fig.colorbar(im, ax=ax, shrink=0.85)
    return fig
