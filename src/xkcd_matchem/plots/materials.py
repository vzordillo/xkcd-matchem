# Copyright 2026 xkcd-matchem contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Materials-informatics plots inspired by pymatviz in xkcd style."""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
import math
import pathlib
from typing import Any
from typing import cast

from matplotlib import axes as maxes
from matplotlib import collections
from matplotlib import colors as mcolors
from matplotlib import patches
from matplotlib import pyplot as plt
import numpy as np
from scipy import spatial
from scipy import stats

from xkcd_matchem.core import brillouin
from xkcd_matchem.core import exceptions
from xkcd_matchem.core import geometry
from xkcd_matchem.core import structure
from xkcd_matchem.core import wulff
from xkcd_matchem.io import loader
from xkcd_matchem.render import primitives
from xkcd_matchem.style import palettes
from xkcd_matchem.style import patterns
from xkcd_matchem.style import theme


_CRYSTAL_SYSTEM_RANGES: tuple[tuple[str, int, int], ...] = (
    ("Triclinic", 1, 2),
    ("Monoclinic", 3, 15),
    ("Orthorhombic", 16, 74),
    ("Tetragonal", 75, 142),
    ("Trigonal", 143, 167),
    ("Hexagonal", 168, 194),
    ("Cubic", 195, 230),
)

_SQRT3_HALF = 0.8660254037844386


def spacegroup_to_crystal_system(sg_number: int) -> str:
    """Maps an International Tables space-group number (1-230) to its system."""
    for name, lo, hi in _CRYSTAL_SYSTEM_RANGES:
        if lo <= sg_number <= hi:
            return name
    raise exceptions.StructureError(
        f"Space group number {sg_number} out of valid range 1..230."
    )


def _barycentric_to_xy(
    compositions: np.ndarray,
) -> np.ndarray:
    """Maps normalized ``(a, b, c)`` ternary fractions to equilateral 2D xy."""
    sums = np.sum(compositions, axis=1, keepdims=True)
    frac = compositions / np.where(sums > 1e-12, sums, 1.0)
    x = frac[:, 1] + 0.5 * frac[:, 2]
    y = _SQRT3_HALF * frac[:, 2]
    return np.column_stack([x, y])


def _collinear_lower_hull(
    uniq: list[int],
    xy_u: np.ndarray,
    e_u: np.ndarray,
) -> tuple[set[int], set[tuple[int, int]]]:
    """Computes 1D lower convex hull when ternary points are collinear."""
    deltas = xy_u - xy_u[0]
    span = np.ptp(deltas, axis=0)
    proj = deltas[:, 0] if span[0] >= span[1] else deltas[:, 1]
    order = sorted(
        range(len(uniq)), key=lambda i: (float(proj[i]), float(e_u[i]))
    )
    lower: list[int] = []
    for idx in order:
        x2, y2 = float(proj[idx]), float(e_u[idx])
        while len(lower) >= 2:
            i0, i1 = lower[-2], lower[-1]
            x0, y0 = float(proj[i0]), float(e_u[i0])
            x1, y1 = float(proj[i1]), float(e_u[i1])
            if (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0) <= 1e-12:
                lower.pop()
            else:
                break
        lower.append(idx)
    stable = {uniq[i] for i in lower}
    edges = {
        (
            min(uniq[lower[k]], uniq[lower[k + 1]]),
            max(uniq[lower[k]], uniq[lower[k + 1]]),
        )
        for k in range(len(lower) - 1)
    }
    return stable, edges


def _ternary_lower_hull(
    xy: np.ndarray,
    energies: np.ndarray,
) -> tuple[set[int], set[tuple[int, int]]]:
    """Returns lower convex-hull vertex indices and tie-line edges."""
    uniq: list[int] = []
    for idx, (xy_pt, en) in enumerate(zip(xy, energies, strict=True)):
        hit = next(
            (
                pos
                for pos, rep in enumerate(uniq)
                if float(np.linalg.norm(xy_pt - xy[rep])) <= 1e-6
            ),
            None,
        )
        if hit is None:
            uniq.append(idx)
        elif float(en) < float(energies[uniq[hit]]):
            uniq[hit] = idx

    if len(uniq) <= 1:
        return set(uniq), set()
    if len(uniq) == 2:
        return set(uniq), {(min(uniq[0], uniq[1]), max(uniq[0], uniq[1]))}

    xy_u = xy[uniq]
    e_u = energies[uniq]
    dists = np.linalg.norm(xy_u - xy_u[0], axis=1)
    ref = xy_u[int(np.argmax(dists))] - xy_u[0]
    ref /= max(float(np.linalg.norm(ref)), 1e-12)
    perp_max = float(
        np.max(
            np.abs(
                (xy_u[:, 0] - xy_u[0, 0]) * ref[1]
                - (xy_u[:, 1] - xy_u[0, 1]) * ref[0]
            )
        )
    )
    if perp_max <= 1e-9:
        return _collinear_lower_hull(uniq, xy_u, e_u)

    e_top = float(np.max(e_u)) + max(10.0, 10.0 * float(np.ptp(e_u)))
    cap = np.array(
        [[0.0, 0.0, e_top], [1.0, 0.0, e_top], [0.5, _SQRT3_HALF, e_top]],
        dtype=np.float64,
    )
    pts3 = np.vstack([np.column_stack([xy_u, e_u]), cap])
    hull = spatial.ConvexHull(pts3)
    n_u = len(uniq)
    stable: set[int] = set()
    edges: set[tuple[int, int]] = set()
    for simplex, eq in zip(hull.simplices, hull.equations, strict=True):
        if float(eq[2]) >= -1e-7:
            continue
        loc = (int(simplex[0]), int(simplex[1]), int(simplex[2]))
        if any(v >= n_u for v in loc):
            continue
        i0, i1, i2 = uniq[loc[0]], uniq[loc[1]], uniq[loc[2]]
        stable.update((i0, i1, i2))
        for u, v in ((i0, i1), (i1, i2), (i2, i0)):
            edges.add((min(u, v), max(u, v)))
    return stable, edges


def _draw_ternary_frame(
    ax: maxes.Axes,
    corners: tuple[str, str, str],
    ink_col: str,
    show_grid: bool,
    shade: str | None,
    bg_color: str = palettes.PAPER,
) -> None:
    """Draws the Gibbs triangle border, barycentric grid, and corner labels."""
    tri = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, _SQRT3_HALF]])
    hatch = patterns.resolve_hatch(shade)
    bg_poly = patches.Polygon(
        tri,
        closed=True,
        facecolor=primitives.muted_surface_color(bg_color) if hatch else "none",
        edgecolor=ink_col,
        linewidth=2.0,
        hatch=hatch,
        alpha=0.22 if hatch else 1.0,
        zorder=1.0,
    )
    if hatch:
        bg_poly.set_sketch_params(0.0, 0.0, 0.0)
    ax.add_patch(bg_poly)
    if hatch:
        ax.add_patch(
            patches.Polygon(
                tri,
                closed=True,
                facecolor="none",
                edgecolor=ink_col,
                linewidth=2.0,
                zorder=2.2,
            )
        )
    if show_grid:
        for frac in (0.2, 0.4, 0.6, 0.8):
            p_ab = np.array([frac, 0.0])
            p_ac = np.array([0.5 * frac, _SQRT3_HALF * frac])
            p_bc = np.array([1.0 - 0.5 * frac, _SQRT3_HALF * frac])
            p_ba = np.array([1.0 - frac, 0.0])
            for s_pt, e_pt in (
                (p_ac, p_bc),
                (p_ab, np.array([0.5 + 0.5 * frac, _SQRT3_HALF * (1 - frac)])),
                (p_ba, np.array([0.5 * (1 - frac), _SQRT3_HALF * (1 - frac)])),
            ):
                (g_line,) = ax.plot(
                    [s_pt[0], e_pt[0]],
                    [s_pt[1], e_pt[1]],
                    color=palettes.MUTED_GRAY,
                    linewidth=0.75,
                    alpha=0.35,
                    linestyle="--",
                    zorder=1.2,
                )
                g_line.set_sketch_params(0.0, 0.0, 0.0)
    ax.text(
        -0.05,
        -0.05,
        corners[0],
        ha="right",
        va="top",
        fontsize=13.0,
        color=ink_col,
    )
    ax.text(
        1.05,
        -0.05,
        corners[1],
        ha="left",
        va="top",
        fontsize=13.0,
        color=ink_col,
    )
    ax.text(
        0.50,
        _SQRT3_HALF + 0.05,
        corners[2],
        ha="center",
        va="bottom",
        fontsize=13.0,
        color=ink_col,
    )


def plot_ternary_phase_diagram(
    compositions: Sequence[tuple[float, float, float]] | np.ndarray,
    energies: Sequence[float] | np.ndarray | None = None,
    *,
    corners: tuple[str, str, str] = ("A", "B", "C"),
    labels: Sequence[str] | None = None,
    ax: maxes.Axes | None = None,
    show_grid: bool = True,
    shade: str | None = None,
    palette: str | Sequence[str] = "okabe_ito",
    preset: str | theme.ThemePreset = "classic",
    title: str | None = None,
) -> maxes.Axes:
    """Plots an xkcd-style ternary Gibbs composition and convex-hull triangle.

    Args:
      compositions: Sequence of ``(x_A, x_B, x_C)`` composition triples.
      energies: Optional formation energies (eV/atom). When supplied, computes
        lower convex-hull tie-lines and separates stable vs. above-hull phases.
      corners: Corner labels ``(A, B, C)`` for bottom-left, bottom-right, top.
      labels: Optional formula labels for each point.
      ax: Optional target Matplotlib ``Axes``.
      show_grid: Whether to draw dashed 20% barycentric gridlines.
      shade: Optional subtle interior triangle hatch (e.g., ``"dots"``).
      palette: Series palette name or sequence of hex colors.
      preset: Visual theme preset name or ``ThemePreset``.
      title: Optional heading above the triangle.

    Returns:
      Matplotlib ``Axes`` containing the ternary diagram.
    """
    comp_arr = np.asarray(compositions, dtype=np.float64)
    if comp_arr.ndim != 2 or comp_arr.shape[1] != 3:
        raise exceptions.StructureError(
            "compositions must have shape (N, 3) for ternary diagrams."
        )
    xy = _barycentric_to_xy(comp_arr)
    series = palettes.get_palette(palette)
    stable_col = series[3 % len(series)]
    tie_col = series[5 % len(series)]
    above_col = series[6 % len(series)]
    preset_obj = theme.get_preset(preset)
    ink_col = preset_obj.ink_color

    stable_idx: set[int] = set(range(len(xy)))
    tie_edges: set[tuple[int, int]] = set()
    if energies is not None:
        e_arr = np.asarray(energies, dtype=np.float64)
        stable_idx, tie_edges = _ternary_lower_hull(xy, e_arr)

    with theme.xkcd_style(preset=preset):
        if ax is None:
            _, ax = plt.subplots(figsize=(6.2, 5.6))
        else:
            theme.style_existing_axes(ax, preset=preset)
        _draw_ternary_frame(
            ax,
            corners,
            ink_col,
            show_grid,
            shade,
            preset_obj.axes_facecolor,
        )

        for u, v in sorted(tie_edges):
            ax.plot(
                [xy[u, 0], xy[v, 0]],
                [xy[u, 1], xy[v, 1]],
                color=tie_col,
                linewidth=1.8,
                zorder=3.0,
            )

        s_list = sorted(stable_idx)
        u_list = [i for i in range(len(xy)) if i not in stable_idx]
        if s_list:
            ax.scatter(
                xy[s_list, 0],
                xy[s_list, 1],
                s=90,
                facecolor=stable_col,
                edgecolor=ink_col,
                linewidth=1.5,
                zorder=5.0,
                label="Stable",
            )
        if u_list:
            ax.scatter(
                xy[u_list, 0],
                xy[u_list, 1],
                s=75,
                marker="D",
                facecolor=preset_obj.axes_facecolor,
                edgecolor=above_col,
                linewidth=1.5,
                zorder=4.5,
                label="Above Hull",
            )
        if labels:
            for idx, lbl in enumerate(labels):
                if not lbl:
                    continue
                dy = 8 if (idx in stable_idx and xy[idx, 1] > 0.08) else -13
                ax.annotate(
                    lbl,
                    xy=(float(xy[idx, 0]), float(xy[idx, 1])),
                    xytext=(0, dy),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8.8,
                    color=ink_col,
                    zorder=6.0,
                )
        if energies is not None:
            ax.legend(loc="upper right", frameon=True, fontsize=8.5)
        ax.set_xlim(-0.18, 1.18)
        ax.set_ylim(-0.16, _SQRT3_HALF + 0.17)
        ax.set_aspect("equal")
        ax.set_axis_off()
        if title:
            ax.set_title(title, color=ink_col, pad=6.0)
    return ax


def _aggregate_spacegroups(
    spacegroups: Sequence[int] | Mapping[int, int],
) -> tuple[dict[str, int], dict[str, list[tuple[int, int]]]]:
    """Aggregates space-group counts by crystal system."""
    sg_counts: dict[int, int] = {}
    if isinstance(spacegroups, Mapping):
        for sg, cnt in spacegroups.items():
            sg_counts[int(sg)] = sg_counts.get(int(sg), 0) + int(cnt)
    else:
        for sg in spacegroups:
            sg_int = int(sg)
            sg_counts[sg_int] = sg_counts.get(sg_int, 0) + 1

    sys_totals: dict[str, int] = {}
    sys_members: dict[str, list[tuple[int, int]]] = {}
    for sg_num, cnt in sg_counts.items():
        if cnt <= 0:
            continue
        sys_name = spacegroup_to_crystal_system(sg_num)
        sys_totals[sys_name] = sys_totals.get(sys_name, 0) + cnt
        sys_members.setdefault(sys_name, []).append((sg_num, cnt))
    for members in sys_members.values():
        members.sort(key=lambda pair: (-pair[1], pair[0]))
    return sys_totals, sys_members


def _draw_sunburst_rings(
    ax: maxes.Axes,
    sys_totals: dict[str, int],
    sys_members: dict[str, list[tuple[int, int]]],
    series: Sequence[str],
    ink_col: str,
    shade: str | Sequence[str | None] | None,
    bg_color: str = palettes.PAPER,
) -> None:
    """Draws inner crystal-system ring and outer space-group ring wedges."""
    total = float(sum(sys_totals.values()))
    if total <= 0.0:
        return
    ordered = [
        (name, sys_totals[name])
        for name, _, _ in _CRYSTAL_SYSTEM_RANGES
        if name in sys_totals
    ]
    angle = 90.0
    for idx, (sys_name, sys_cnt) in enumerate(ordered):
        span = 360.0 * (sys_cnt / total)
        col = series[(idx + 1) % len(series)]
        hatch = patterns.resolve_hatch(shade, index=idx)
        inner = patches.Wedge(
            (0.0, 0.0),
            0.58,
            angle,
            angle + span,
            width=0.33,
            facecolor=col,
            edgecolor=ink_col,
            linewidth=1.5,
            alpha=0.86,
            hatch=hatch,
            zorder=3.0,
        )
        if hatch:
            inner.set_sketch_params(0.0, 0.0, 0.0)
        ax.add_patch(inner)
        mid_rad = math.radians(angle + 0.5 * span)
        if span >= 22.0:
            ax.text(
                0.415 * math.cos(mid_rad),
                0.415 * math.sin(mid_rad),
                sys_name[:5] if span < 42.0 else sys_name,
                ha="center",
                va="center",
                fontsize=8.0,
                color=ink_col,
                bbox={
                    "boxstyle": "round,pad=0.14",
                    "facecolor": bg_color,
                    "edgecolor": "none",
                    "alpha": 0.85,
                },
                zorder=5.0,
            )

        sub_ang = angle
        for sg_num, cnt in sys_members.get(sys_name, []):
            sub_span = 360.0 * (cnt / total)
            outer = patches.Wedge(
                (0.0, 0.0),
                0.96,
                sub_ang,
                sub_ang + sub_span,
                width=0.34,
                facecolor=col,
                edgecolor=ink_col,
                linewidth=1.2,
                alpha=0.52,
                zorder=2.5,
            )
            ax.add_patch(outer)
            if sub_span >= 15.0:
                s_rad = math.radians(sub_ang + 0.5 * sub_span)
                ax.text(
                    0.79 * math.cos(s_rad),
                    0.79 * math.sin(s_rad),
                    f"#{sg_num}",
                    ha="center",
                    va="center",
                    fontsize=7.8,
                    color=ink_col,
                    zorder=5.0,
                )
            sub_ang += sub_span
        angle += span

    ax.text(
        0.0,
        0.0,
        f"N={int(total)}",
        ha="center",
        va="center",
        fontsize=11.0,
        color=ink_col,
        zorder=6.0,
    )


def plot_spacegroup_distribution(
    spacegroups: Sequence[int] | Mapping[int, int],
    *,
    mode: str = "sunburst",
    ax: maxes.Axes | None = None,
    shade: str | Sequence[str | None] | None = None,
    background_grid: str | None = None,
    palette: str | Sequence[str] = "okabe_ito",
    preset: str | theme.ThemePreset = "classic",
    title: str | None = None,
) -> maxes.Axes:
    """Plots crystal-system & space-group counts as a sunburst or bar chart.

    Inspired by ``pymatviz.spacegroup_sunburst`` and ``spacegroup_bar``,
    automatically grouping International Tables space groups ``1..230`` into
    the 7 crystal systems.

    Args:
      spacegroups: Sequence of space-group numbers or ``{sg_num: count}`` dict.
      mode: ``"sunburst"`` (two-ring radial chart) or ``"bar"`` (crystal-system
        count bars).
      ax: Optional target Matplotlib ``Axes``.
      shade: Optional hatched/stippled shading (e.g., ``"cycle"``, ``"hatch"``).
      background_grid: Optional background grid style (applies in ``"bar"``).
      palette: Series palette name or hex color list.
      preset: Visual theme preset name or ``ThemePreset``.
      title: Optional plot heading.

    Returns:
      Matplotlib ``Axes`` containing the space-group breakdown.
    """
    view_mode = mode.strip().lower()
    if view_mode not in ("sunburst", "bar"):
        raise exceptions.StructureError(
            f"Unsupported mode {mode!r}. Use 'sunburst' or 'bar'."
        )
    sys_totals, sys_members = _aggregate_spacegroups(spacegroups)
    series = palettes.get_palette(palette)
    preset_obj = theme.get_preset(preset)
    ink_col = preset_obj.ink_color

    with theme.xkcd_style(preset=preset):
        if ax is None:
            figsize = (5.8, 5.4) if view_mode == "sunburst" else (7.0, 4.3)
            _, ax = plt.subplots(figsize=figsize)
        else:
            theme.style_existing_axes(ax, preset=preset)

        if view_mode == "sunburst":
            _draw_sunburst_rings(
                ax,
                sys_totals,
                sys_members,
                series,
                ink_col,
                shade,
                preset_obj.axes_facecolor,
            )
            ax.set_xlim(-1.12, 1.12)
            ax.set_ylim(-1.12, 1.12)
            ax.set_aspect("equal")
            ax.set_axis_off()
        else:
            if background_grid:
                patterns.apply_background_grid(ax, style=background_grid)
            ordered = [
                (name, sys_totals[name])
                for name, _, _ in _CRYSTAL_SYSTEM_RANGES
                if name in sys_totals
            ]
            names = [item[0] for item in ordered]
            counts = [item[1] for item in ordered]
            xs = np.arange(len(names))
            for idx, (cnt, x_pos) in enumerate(zip(counts, xs, strict=True)):
                col = series[(idx + 1) % len(series)]
                hatch = patterns.resolve_hatch(shade, index=idx)
                bar_list = ax.bar(
                    [x_pos],
                    [cnt],
                    color=col,
                    edgecolor=ink_col,
                    linewidth=1.5,
                    width=0.64,
                    alpha=0.88,
                    hatch=hatch,
                    zorder=3.0,
                )
                if hatch:
                    bar_list[0].set_sketch_params(0.0, 0.0, 0.0)
                ax.text(
                    float(x_pos),
                    float(cnt),
                    str(cnt),
                    ha="center",
                    va="bottom",
                    fontsize=9.5,
                    color=ink_col,
                )
            ax.set_xticks(xs)
            ax.set_xticklabels(names, rotation=22, ha="right")
            ax.set_ylabel("Structure Count")
            theme.apply_xkcd_spines(ax, preset=preset)
        if title:
            ax.set_title(title, color=ink_col)
    return ax


def _draw_hist_series(
    ax: maxes.Axes,
    lbl: str,
    vals: np.ndarray,
    bins: int | np.ndarray,
    style_cfg: tuple[str, str, str | None, bool, bool, bool, int],
) -> None:
    """Draws one histogram series with optional hatch, KDE, and mean line."""
    col, ink_col, hatch, multi, show_kde, show_mean, series_idx = style_cfg
    clean = vals[np.isfinite(vals)]
    if clean.size == 0:
        return
    bins_arg: Any = bins.tolist() if isinstance(bins, np.ndarray) else bins
    _, edges, bar_patches = ax.hist(
        clean,
        bins=bins_arg,
        color=col,
        edgecolor=ink_col,
        linewidth=1.3,
        alpha=0.58 if multi else 0.72,
        hatch=hatch,
        label=lbl if multi else None,
        zorder=3.0,
    )
    if hatch:
        for patch in cast(Any, bar_patches):
            patch.set_sketch_params(0.0, 0.0, 0.0)
    if show_kde and clean.size >= 4 and float(np.std(clean)) > 1e-9:
        kde = stats.gaussian_kde(clean)
        x_grid = np.linspace(float(edges[0]), float(edges[-1]), 140)
        bin_w = float(edges[1] - edges[0])
        ax.plot(
            x_grid,
            kde(x_grid) * clean.size * bin_w,
            color=col,
            linewidth=2.2,
            zorder=4.5,
        )
    if show_mean:
        mu = float(np.mean(clean))
        mean_col = col if multi else ink_col
        ax.axvline(
            mu,
            color=mean_col,
            linestyle="--",
            linewidth=1.45,
            alpha=0.9,
            zorder=4.2,
        )
        txt = f" {lbl}: {mu:.2f}" if multi else f" mean={mu:.2f}"
        y_frac = max(0.18, 0.96 - 0.10 * series_idx)
        ax.text(
            mu,
            y_frac,
            txt,
            transform=ax.get_xaxis_transform(),
            ha="left",
            va="top",
            fontsize=8.6,
            color=mean_col,
            zorder=5.0,
        )


def plot_histogram(
    data: Sequence[float]
    | np.ndarray
    | Mapping[str, Sequence[float] | np.ndarray],
    *,
    bins: int = 18,
    ax: maxes.Axes | None = None,
    xlabel: str = "Property Value",
    ylabel: str = "Count",
    show_kde: bool = True,
    show_mean: bool = True,
    shade: str | Sequence[str | None] | None = "hatch",
    background_grid: str | None = None,
    palette: str | Sequence[str] = "okabe_ito",
    preset: str | theme.ThemePreset = "classic",
    title: str | None = None,
) -> maxes.Axes:
    """Plots an xkcd-style property histogram with hatched fills and KDE.

    Args:
      data: 1D array/sequence of values or ``{series_label: values}`` mapping.
      bins: Number of histogram bins.
      ax: Optional target Matplotlib ``Axes``.
      xlabel: Horizontal axis label.
      ylabel: Vertical axis label.
      show_kde: Whether to overlay a smooth SciPy Gaussian KDE curve.
      show_mean: Whether to mark each series mean with a vertical dashed line.
      shade: Bar fill pattern (``"hatch"``, ``"crosshatch"``, ``"dots"``,
        ``"cycle"``, ``"solid"``, or raw Matplotlib hatch string).
      background_grid: Optional background grid (``"grid"``, ``"dots"``, etc.).
      palette: Series palette name or hex color sequence.
      preset: Visual theme preset name or ``ThemePreset``.
      title: Optional heading above the histogram.

    Returns:
      Matplotlib ``Axes`` containing the histogram.
    """
    series_map: dict[str, np.ndarray] = (
        {str(k): np.asarray(v, dtype=np.float64) for k, v in data.items()}
        if isinstance(data, Mapping)
        else {"Distribution": np.asarray(data, dtype=np.float64)}
    )
    series_cols = palettes.get_palette(palette)
    ink_col = theme.get_preset(preset).ink_color
    multi = len(series_map) > 1
    non_empty = [v[np.isfinite(v)] for v in series_map.values()]
    non_empty = [c for c in non_empty if c.size > 0]
    if non_empty:
        pooled = np.concatenate(non_empty)
        lo, hi = float(np.min(pooled)), float(np.max(pooled))
        if hi - lo < 1e-12:
            lo -= 0.5
            hi += 0.5
        bin_spec: int | np.ndarray = np.linspace(lo, hi, max(int(bins), 1) + 1)
    else:
        bin_spec = bins

    with theme.xkcd_style(preset=preset):
        if ax is None:
            _, ax = plt.subplots(figsize=(6.4, 4.2))
        else:
            theme.style_existing_axes(ax, preset=preset)
        if background_grid:
            patterns.apply_background_grid(ax, style=background_grid)

        for idx, (lbl, vals) in enumerate(series_map.items()):
            col = series_cols[(idx + 1) % len(series_cols)]
            hatch = patterns.resolve_hatch(shade, index=idx)
            _draw_hist_series(
                ax,
                lbl,
                vals,
                bin_spec,
                (col, ink_col, hatch, multi, show_kde, show_mean, idx),
            )

        if multi:
            ax.legend(loc="best", frameon=True, fontsize=8.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        theme.apply_xkcd_spines(ax, preset=preset)
        if title:
            ax.set_title(title, color=ink_col)
    return ax


def _coerce_wulff_cell(
    source: (
        structure.AtomsData
        | np.ndarray
        | Sequence[Sequence[float]]
        | str
        | pathlib.Path
    ),
) -> np.ndarray:
    """Resolves a ``(3, 3)`` unit-cell matrix for Wulff construction."""
    if isinstance(source, (str, pathlib.Path)):
        loaded = loader.load(source)
        if not isinstance(loaded, structure.AtomsData) or loaded.cell is None:
            raise exceptions.StructureError(
                f"Structure {source!r} has no periodic unit cell."
            )
        return np.asarray(loaded.cell, dtype=np.float64)
    if isinstance(source, structure.AtomsData):
        if source.cell is None:
            raise exceptions.StructureError(
                "AtomsData has no periodic unit cell for Wulff construction."
            )
        return np.asarray(source.cell, dtype=np.float64)
    return np.asarray(source, dtype=np.float64)


def _format_hkl_label(hkl: tuple[int, int, int]) -> str:
    """Formats crystallographic Miller indices cleanly onto facets."""
    parts = [str(int(v)) for v in hkl]
    if all(len(p) == 1 for p in parts):
        return f"({''.join(parts)})"
    return f"({','.join(parts)})"


def _lambert_shade_rgb(
    hex_col: str,
    dot_light: float,
) -> tuple[float, float, float]:
    """Computes clamped RGB color modulated by directional Lambert shading."""
    r, g, b = mcolors.to_rgb(hex_col)
    factor = 0.62 + 0.38 * max(float(dot_light), 0.0)
    return (
        float(np.clip(r * factor, 0.0, 1.0)),
        float(np.clip(g * factor, 0.0, 1.0)),
        float(np.clip(b * factor, 0.0, 1.0)),
    )


def _project_wulff_facets(
    shape: wulff.WulffShape,
    cam: geometry.Camera,
    series: Sequence[str],
    show_labels: bool,
) -> tuple[
    list[np.ndarray],
    list[tuple[float, float, float]],
    list[tuple[np.ndarray, str, str]],
    tuple[float, float, float, float],
]:
    """Projects, depth-sorts, and shades Wulff facets back-to-front."""
    basis = geometry.rotation_matrix(cam.azimuth, cam.elevation, cam.roll)
    light_dir = np.array([-0.45, 0.55, 0.70], dtype=np.float64)
    light_dir /= float(np.linalg.norm(light_dir))
    span = geometry.scene_projection_span(
        shape.vertices, cam, center=np.zeros(3, dtype=np.float64)
    )

    family_colors: dict[tuple[int, int, int], str] = {}
    records: list[tuple[float, wulff.WulffFacet, np.ndarray, np.ndarray]] = []
    for facet in shape.facets:
        s_vals = sorted(abs(x) for x in facet.hkl)
        fam_key = (s_vals[0], s_vals[1], s_vals[2])
        if fam_key not in family_colors:
            family_colors[fam_key] = series[
                (len(family_colors) + 1) % len(series)
            ]
        xy_f, dep_f = geometry.project_points(
            facet.vertices, cam, center=np.zeros(3), span=span
        )
        records.append(
            (float(np.mean(dep_f)), facet, xy_f, facet.normal @ basis.T)
        )
    records.sort(key=lambda item: item[0])

    polys: list[np.ndarray] = []
    face_cols: list[tuple[float, float, float]] = []
    labels_meta: list[tuple[np.ndarray, str, str]] = []
    for _, facet, xy_f, n_rot in records:
        s_vals = sorted(abs(x) for x in facet.hkl)
        fam_key = (s_vals[0], s_vals[1], s_vals[2])
        base_hex = family_colors[fam_key]
        shaded = _lambert_shade_rgb(base_hex, float(np.dot(n_rot, light_dir)))
        polys.append(xy_f)
        face_cols.append(shaded)
        if show_labels and float(n_rot[2]) > 0.15:
            labels_meta.append(
                (
                    np.mean(xy_f, axis=0),
                    _format_hkl_label(facet.hkl),
                    primitives.contrast_text_color(mcolors.to_hex(shaded)),
                )
            )

    all_xy, _ = geometry.project_points(
        shape.vertices, cam, center=np.zeros(3), span=span
    )
    x_min, y_min = np.min(all_xy, axis=0)
    x_max, y_max = np.max(all_xy, axis=0)
    pad = 0.22 * max(float(x_max - x_min), float(y_max - y_min), 1e-3)
    bounds = (
        float(x_min - pad),
        float(x_max + pad),
        float(y_min - pad),
        float(y_max + pad),
    )
    return polys, face_cols, labels_meta, bounds


def plot_wulff_shape(
    source: (
        structure.AtomsData
        | np.ndarray
        | Sequence[Sequence[float]]
        | str
        | pathlib.Path
    ),
    surface_energies: Mapping[tuple[int, int, int], float],
    *,
    symops: (
        Sequence[tuple[np.ndarray, np.ndarray]] | str | int | None
    ) = "cubic",
    camera: geometry.Camera | str | None = "isometric",
    ax: maxes.Axes | None = None,
    palette: str | Sequence[str] = "okabe_ito",
    show_labels: bool = True,
    alpha: float = 0.84,
    preset: str | theme.ThemePreset = "classic",
    title: str | None = None,
) -> maxes.Axes:
    """Renders an xkcd-style 3D shaded equilibrium Wulff crystal shape.

    Args:
      source: ``AtomsData``, ``(3, 3)`` cell matrix, file path, or sample.
      surface_energies: Mapping ``{(h, k, l): gamma}`` with ``gamma > 0``.
      symops: Crystal symmetry expansion mode passed to ``compute_wulff_shape``.
      camera: ``Camera`` object or preset string.
      ax: Optional target Matplotlib ``Axes``.
      palette: Series palette name or hex color list for Miller families.
      show_labels: Whether to label front-facing ``(hkl)`` facets.
      alpha: Facet polygon fill opacity.
      preset: Visual theme preset name or ``ThemePreset``.
      title: Optional heading above the Wulff crystal.

    Returns:
      Matplotlib ``Axes`` containing the projected Wulff polyhedron.
    """
    cell = _coerce_wulff_cell(source)
    shape = wulff.compute_wulff_shape(cell, surface_energies, symops=symops)
    cam = geometry.resolve_camera(camera, cell=cell)
    series = palettes.get_palette(palette)
    ink_col = theme.get_preset(preset).ink_color
    polys, face_cols, labels_meta, bounds = _project_wulff_facets(
        shape, cam, series, show_labels
    )

    with theme.xkcd_style(preset=preset):
        if ax is None:
            _, ax = plt.subplots(figsize=(5.6, 5.2))
        else:
            theme.style_existing_axes(ax, preset=preset)
        poly_coll = collections.PolyCollection(
            polys,
            facecolors=face_cols,
            edgecolors=ink_col,
            linewidths=1.7,
            alpha=alpha,
            joinstyle="round",
            zorder=3.0,
        )
        ax.add_collection(poly_coll)
        for centroid, txt_lbl, txt_col in labels_meta:
            ax.text(
                float(centroid[0]),
                float(centroid[1]),
                txt_lbl,
                ha="center",
                va="center",
                fontsize=8.2,
                color=txt_col,
                zorder=5.0,
            )
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[2], bounds[3])
        ax.set_aspect("equal")
        ax.set_axis_off()
        if title:
            ax.set_title(title, color=ink_col)
    return ax


def _draw_bz_kpath_and_axes(
    ax: maxes.Axes,
    bz: brillouin.BrillouinZone,
    cam: geometry.Camera,
    span: float,
    ink_col: str,
    path_color: str,
    *,
    kpath: Sequence[str] | None,
    show_kpoints: bool,
    show_axes: bool,
) -> None:
    """Draws reciprocal basis vectors and high-symmetry k-path inside a BZ."""
    origin = np.zeros(3, dtype=np.float64)
    if show_axes:
        ax_cols = ("#D55E00", "#009E73", "#0072B2")
        for idx, b_vec in enumerate(bz.reciprocal_cell):
            tip = b_vec * (0.42 / max(float(np.linalg.norm(b_vec)), 1e-9))
            pts = np.vstack([origin, tip])
            xy, _ = geometry.project_points(pts, cam, center=origin, span=span)
            ax.annotate(
                "",
                xy=(float(xy[1, 0]), float(xy[1, 1])),
                xytext=(float(xy[0, 0]), float(xy[0, 1])),
                arrowprops={
                    "arrowstyle": "->,head_width=0.22,head_length=0.28",
                    "color": ax_cols[idx],
                    "lw": 1.6,
                },
                zorder=5.5,
            )
            ax.text(
                float(xy[1, 0]) * 1.12,
                float(xy[1, 1]) * 1.12,
                f"b{idx + 1}",
                fontsize=8.5,
                color=ax_cols[idx],
                ha="center",
                va="center",
                zorder=6.0,
            )

    tour = list(bz.default_path if kpath is None else kpath)
    valid = [name for name in tour if name in bz.high_symmetry_points]
    if not show_kpoints or not valid:
        return
    q_arr = np.vstack([bz.high_symmetry_points[name] for name in valid])
    q_xy, _ = geometry.project_points(q_arr, cam, center=origin, span=span)
    if len(q_xy) >= 2:
        ax.plot(
            q_xy[:, 0],
            q_xy[:, 1],
            color=path_color,
            linewidth=2.3,
            zorder=7.0,
        )
    ax.scatter(
        q_xy[:, 0],
        q_xy[:, 1],
        s=42.0,
        facecolor=path_color,
        edgecolor=ink_col,
        linewidth=1.1,
        zorder=8.0,
    )
    seen_pts: set[str] = set()
    for idx, name in enumerate(valid):
        if name in seen_pts:
            continue
        seen_pts.add(name)
        disp_lbl = "Γ" if name == "G" else name
        ax.text(
            float(q_xy[idx, 0]),
            float(q_xy[idx, 1]) + 0.045 * span,
            disp_lbl,
            fontsize=9.5,
            fontweight="bold",
            color=ink_col,
            ha="center",
            va="bottom",
            zorder=9.0,
        )


def _project_bz_facets(
    bz: brillouin.BrillouinZone,
    cam: geometry.Camera,
    color: str,
) -> tuple[
    list[np.ndarray],
    list[tuple[float, float, float]],
    list[np.ndarray],
    tuple[float, float, float, float],
    float,
]:
    """Projects, splits front/back, and Lambert-shades Brillouin zone faces."""
    basis = geometry.rotation_matrix(cam.azimuth, cam.elevation, cam.roll)
    light_dir = np.array([-0.45, 0.55, 0.70], dtype=np.float64)
    light_dir /= float(np.linalg.norm(light_dir))
    origin = np.zeros(3, dtype=np.float64)
    span = geometry.scene_projection_span(bz.vertices, cam, center=origin)

    records: list[
        tuple[float, np.ndarray, tuple[float, float, float], float]
    ] = []
    for facet in bz.facets:
        xy_f, dep_f = geometry.project_points(
            facet.vertices, cam, center=origin, span=span
        )
        n_rot = facet.normal @ basis.T
        shaded = _lambert_shade_rgb(color, float(np.dot(n_rot, light_dir)))
        records.append((float(np.mean(dep_f)), xy_f, shaded, float(n_rot[2])))
    records.sort(key=lambda item: item[0])

    front_polys = [it[1] for it in records if it[3] >= -1e-3]
    front_cols = [it[2] for it in records if it[3] >= -1e-3]
    back_polys = [it[1] for it in records if it[3] < -1e-3]

    all_xy, _ = geometry.project_points(
        bz.vertices, cam, center=origin, span=span
    )
    x_min, y_min = np.min(all_xy, axis=0)
    x_max, y_max = np.max(all_xy, axis=0)
    pad = 0.24 * max(float(x_max - x_min), float(y_max - y_min), 1e-3)
    bounds = (
        float(x_min - pad),
        float(x_max + pad),
        float(y_min - pad),
        float(y_max + pad),
    )
    return front_polys, front_cols, back_polys, bounds, span


def plot_brillouin_zone(
    source: (
        brillouin.BrillouinZone
        | structure.AtomsData
        | np.ndarray
        | Sequence[Sequence[float]]
        | str
        | pathlib.Path
    ),
    *,
    use_primitive: bool = True,
    kpath: Sequence[str] | None = None,
    show_kpoints: bool = True,
    show_axes: bool = True,
    camera: geometry.Camera | str | None = "isometric",
    ax: maxes.Axes | None = None,
    color: str = "#56B4E9",
    path_color: str = "#D55E00",
    alpha: float = 0.36,
    preset: str | theme.ThemePreset = "classic",
    title: str | None = None,
) -> maxes.Axes:
    """Plots 3D first Brillouin zone with high-symmetry k-path in xkcd style.

    Args:
      source: ``BrillouinZone``, ``AtomsData``, ``(3, 3)`` lattice matrix,
        file path, or bundled sample filename.
      use_primitive: Reduce centered F/I/C conventional cells to primitive form.
      kpath: Optional ordered sequence of high-symmetry point names to trace.
      show_kpoints: Whether to plot the high-symmetry k-path and markers.
      show_axes: Whether to draw reciprocal basis vectors ``b1, b2, b3``.
      camera: ``Camera`` object or preset string.
      ax: Optional target Matplotlib ``Axes``.
      color: Facet fill hex color string.
      path_color: High-symmetry k-path and marker accent hex color.
      alpha: Translucent polyhedron facet opacity.
      preset: Visual theme preset name or ``ThemePreset``.
      title: Optional plot heading.

    Returns:
      Matplotlib ``Axes`` containing the projected first Brillouin zone.
    """
    if isinstance(source, brillouin.BrillouinZone):
        bz = source
    elif isinstance(source, (str, pathlib.Path)):
        loaded = loader.load(source)
        if not isinstance(loaded, structure.AtomsData):
            raise exceptions.StructureError(
                f"Expected AtomsData from {source!r}."
            )
        bz = brillouin.compute_brillouin_zone(
            loaded, use_primitive=use_primitive
        )
    else:
        bz = brillouin.compute_brillouin_zone(
            source, use_primitive=use_primitive
        )

    cam = geometry.resolve_camera(camera)
    ink_col = theme.get_preset(preset).ink_color
    front_polys, front_cols, back_polys, bounds, span = _project_bz_facets(
        bz, cam, color
    )

    with theme.xkcd_style(preset=preset):
        if ax is None:
            _, ax = plt.subplots(figsize=(5.8, 5.2))
        else:
            theme.style_existing_axes(ax, preset=preset)
        if back_polys:
            back_coll = collections.PolyCollection(
                cast(Any, back_polys),
                facecolors=[
                    _lambert_shade_rgb(color, 0.15) for _ in back_polys
                ],
                edgecolors=palettes.MUTED_GRAY,
                linewidths=1.0,
                linestyles="--",
                alpha=alpha * 0.45,
                zorder=1.5,
            )
            ax.add_collection(back_coll)
        if front_polys:
            front_coll = collections.PolyCollection(
                cast(Any, front_polys),
                facecolors=front_cols,
                edgecolors=ink_col,
                linewidths=1.7,
                alpha=alpha,
                joinstyle="round",
                zorder=3.0,
            )
            ax.add_collection(front_coll)
        _draw_bz_kpath_and_axes(
            ax,
            bz,
            cam,
            span,
            ink_col,
            path_color,
            kpath=kpath,
            show_kpoints=show_kpoints,
            show_axes=show_axes,
        )
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[2], bounds[3])
        ax.set_aspect("equal")
        ax.set_axis_off()
        if title:
            ax.set_title(title, color=ink_col)
    return ax
