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
"""Ternary Gibbs composition and convex-hull phase diagram renderer."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from matplotlib import axes as maxes
from matplotlib import patches
from matplotlib import pyplot as plt
import numpy as np
from scipy import spatial

from xkcd_matchem.core import exceptions
from xkcd_matchem.render import primitives
from xkcd_matchem.style import palettes
from xkcd_matchem.style import patterns
from xkcd_matchem.style import theme


SQRT3_HALF = 0.8660254037844386


def theme_muted_guide(bg_color: str) -> str:
    """Returns a muted reference guide color readable on ``bg_color``."""
    if primitives.is_dark_color(bg_color):
        return primitives.lighten_hex(palettes.MUTED_GRAY, 0.45)
    return palettes.MUTED_GRAY


def theme_series_color(color: str, bg_color: str, ink_col: str) -> str:
    """Lifts dark-on-dark series accents while keeping light themes exact."""
    if not primitives.is_dark_color(bg_color):
        return color
    low = color.strip().lower()
    if low in ("#000000", "black", "k"):
        return ink_col
    if low == "#0072b2":
        return "#56B4E9"
    return primitives.ensure_readable_text_color(color, bg_color)


def style_legend(
    ax: maxes.Axes,
    ink_col: str,
    bg_color: str,
    **kwargs: Any,
) -> Any:
    """Creates a legend with theme-consistent frame and text colors."""
    leg = ax.legend(**kwargs)
    if leg is not None:
        frame = leg.get_frame()
        frame.set_facecolor(bg_color)
        frame.set_edgecolor(ink_col)
        frame.set_alpha(0.88)
        for text in leg.get_texts():
            text.set_color(ink_col)
    return leg


def barycentric_to_xy(
    compositions: np.ndarray,
) -> np.ndarray:
    """Maps normalized ``(a, b, c)`` ternary fractions to equilateral 2D xy."""
    sums = np.sum(compositions, axis=1, keepdims=True)
    frac = compositions / np.where(sums > 1e-12, sums, 1.0)
    x = frac[:, 1] + 0.5 * frac[:, 2]
    y = SQRT3_HALF * frac[:, 2]
    return np.column_stack([x, y])


def collinear_lower_hull(
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


def ternary_lower_hull(
    xy: np.ndarray,
    energies: np.ndarray,
) -> tuple[set[int], set[tuple[int, int]]]:
    """Returns lower convex-hull vertex indices and tie-line edges."""
    if len(xy) == 0:
        return set(), set()
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
        return collinear_lower_hull(uniq, xy_u, e_u)

    e_top = float(np.max(e_u)) + max(10.0, 10.0 * float(np.ptp(e_u)))
    cap = np.array(
        [[0.0, 0.0, e_top], [1.0, 0.0, e_top], [0.5, SQRT3_HALF, e_top]],
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


def draw_ternary_frame(
    ax: maxes.Axes,
    corners: tuple[str, str, str],
    ink_col: str,
    show_grid: bool,
    shade: str | None,
    bg_color: str = palettes.PAPER,
    sw: theme.StrokeHierarchy | None = None,
) -> None:
    """Draws the Gibbs triangle border, barycentric grid, and corner labels."""
    weights = sw if sw is not None else theme.stroke_weights("classic")
    tri = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, SQRT3_HALF]])
    hatch = patterns.resolve_hatch(shade)
    bg_poly = patches.Polygon(
        tri,
        closed=True,
        facecolor=primitives.muted_surface_color(bg_color) if hatch else "none",
        edgecolor=ink_col,
        linewidth=weights.frame,
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
                linewidth=weights.frame,
                zorder=2.2,
            )
        )
    if show_grid:
        grid_lw = round(0.85 * weights.guide, 2)
        grid_col = theme_muted_guide(bg_color)
        for frac in (0.2, 0.4, 0.6, 0.8):
            p_ab = np.array([frac, 0.0])
            p_ac = np.array([0.5 * frac, SQRT3_HALF * frac])
            p_bc = np.array([1.0 - 0.5 * frac, SQRT3_HALF * frac])
            p_ba = np.array([1.0 - frac, 0.0])
            for s_pt, e_pt in (
                (p_ac, p_bc),
                (p_ab, np.array([0.5 + 0.5 * frac, SQRT3_HALF * (1 - frac)])),
                (p_ba, np.array([0.5 * (1 - frac), SQRT3_HALF * (1 - frac)])),
            ):
                (g_line,) = ax.plot(
                    [s_pt[0], e_pt[0]],
                    [s_pt[1], e_pt[1]],
                    color=grid_col,
                    linewidth=grid_lw,
                    alpha=0.38,
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
        SQRT3_HALF + 0.05,
        corners[2],
        ha="center",
        va="bottom",
        fontsize=13.0,
        color=ink_col,
    )


def annotate_ternary_labels(
    ax: maxes.Axes,
    xy: np.ndarray,
    tie_edges: set[tuple[int, int]],
    labels: Sequence[str],
    stable_idx: set[int],
    ink_col: str,
    bg_color: str,
) -> None:
    """Places ink-first ternary labels clear of borders and tie-lines."""
    ctr = np.array([0.5, SQRT3_HALF / 3.0], dtype=np.float64)
    adj: dict[int, list[int]] = {i: [] for i in range(len(xy))}
    for u, v in tie_edges:
        adj[u].append(v)
        adj[v].append(u)

    for idx, lbl in enumerate(labels):
        if not lbl:
            continue
        p = xy[idx]
        if idx not in stable_idx:
            out_dir = p - ctr
            on_edge = (
                p[1] < 0.04
                or abs((p[0] - 0.5) * SQRT3_HALF - 0.5 * (p[1] - SQRT3_HALF))
                < 0.04
                or abs((p[0] - 0.5) * SQRT3_HALF + 0.5 * (p[1] - SQRT3_HALF))
                < 0.04
            )
            if on_edge and float(np.linalg.norm(out_dir)) > 1e-6:
                u_out = out_dir / float(np.linalg.norm(out_dir))
                dx = int(round(16.0 * float(u_out[0])))
                dy = int(round(12.0 * float(u_out[1])))
            else:
                dx, dy = 0, -13
        elif adj[idx]:
            rays = [
                (xy[nb] - p) / max(float(np.linalg.norm(xy[nb] - p)), 1e-9)
                for nb in adj[idx]
            ]
            avg = np.sum(rays, axis=0)
            avg_n = float(np.linalg.norm(avg))
            if avg_n > 0.15:
                push = -avg / avg_n
            else:
                out_dir = p - ctr
                push = out_dir / max(float(np.linalg.norm(out_dir)), 1e-6)
            dx = int(round(14.0 * float(push[0])))
            dy = int(round(12.0 * float(push[1])))
        else:
            dy = 9 if p[1] > 0.08 else -13
            dx = 0
        ann = ax.annotate(
            lbl,
            xy=(float(p[0]), float(p[1])),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=8.6,
            color=ink_col,
            bbox=theme.label_halo_bbox(bg_color, pad=0.08, alpha=0.86),
            zorder=6.0,
        )
        patch = ann.get_bbox_patch()
        if patch is not None:
            patch.set_sketch_params(0.0, 0.0, 0.0)


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
    xy = barycentric_to_xy(comp_arr)
    series = palettes.get_palette(palette)
    preset_obj = theme.get_preset(preset)
    ink_col = preset_obj.ink_color
    bg_col = preset_obj.axes_facecolor
    stable_col = theme_series_color(series[3 % len(series)], bg_col, ink_col)
    tie_col = theme_series_color(series[5 % len(series)], bg_col, ink_col)
    above_col = theme_series_color(series[6 % len(series)], bg_col, ink_col)
    sw = theme.stroke_weights(preset)

    stable_idx: set[int] = set(range(len(xy)))
    tie_edges: set[tuple[int, int]] = set()
    if energies is not None:
        e_arr = np.asarray(energies, dtype=np.float64)
        if e_arr.ndim != 1 or len(e_arr) != len(comp_arr):
            raise exceptions.StructureError(
                "energies must be 1D with the same length as compositions."
            )
        stable_idx, tie_edges = ternary_lower_hull(xy, e_arr)

    with theme.xkcd_style(preset=preset):
        if ax is None:
            _, ax = plt.subplots(figsize=(6.2, 5.6))
        else:
            theme.style_existing_axes(ax, preset=preset)
        draw_ternary_frame(
            ax,
            corners,
            ink_col,
            show_grid,
            shade,
            bg_col,
            sw=sw,
        )

        for u, v in sorted(tie_edges):
            ax.plot(
                [xy[u, 0], xy[v, 0]],
                [xy[u, 1], xy[v, 1]],
                color=tie_col,
                linewidth=sw.overlay,
                alpha=0.84,
                zorder=2.8,
            )

        s_list = sorted(stable_idx)
        u_list = [i for i in range(len(xy)) if i not in stable_idx]
        if s_list:
            ax.scatter(
                xy[s_list, 0],
                xy[s_list, 1],
                s=86,
                facecolor=stable_col,
                edgecolor=ink_col,
                linewidth=1.35,
                zorder=5.0,
                label="Stable",
            )
        if u_list:
            ax.scatter(
                xy[u_list, 0],
                xy[u_list, 1],
                s=72,
                marker="D",
                facecolor=bg_col,
                edgecolor=above_col,
                linewidth=1.35,
                zorder=4.5,
                label="Above Hull",
            )
        if labels:
            annotate_ternary_labels(
                ax,
                xy,
                tie_edges,
                labels,
                stable_idx,
                ink_col,
                bg_col,
            )
        if energies is not None:
            style_legend(
                ax,
                ink_col,
                bg_col,
                loc="upper right",
                frameon=True,
                fontsize=8.5,
            )
        ax.set_xlim(-0.20, 1.20)
        ax.set_ylim(-0.18, SQRT3_HALF + 0.18)
        ax.set_aspect("equal")
        ax.set_axis_off()
        if title:
            ax.set_title(title, color=ink_col, pad=6.0)
    return ax
