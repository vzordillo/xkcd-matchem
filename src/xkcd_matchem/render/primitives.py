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
"""Hand-drawn 2D/3D atomic, bond, polyhedron, and unit-cell primitives."""

from __future__ import annotations

import math

from matplotlib import axes as maxes
from matplotlib import colors as mcolors
from matplotlib import patches
import numpy as np

from xkcd_matchem.core import geometry
from xkcd_matchem.style import palettes


def lighten_hex(color: str, amount: float = 0.55) -> str:
    """Blends a hex or named color toward white by ``amount`` in ``[0, 1]``."""
    r, g, b = mcolors.to_rgb(color)
    mix = max(0.0, min(1.0, amount))
    return mcolors.to_hex(
        (r + (1.0 - r) * mix, g + (1.0 - g) * mix, b + (1.0 - b) * mix)
    )


def darken_hex(color: str, amount: float = 0.35) -> str:
    """Blends a hex or named color toward black by ``amount`` in ``[0, 1]``."""
    r, g, b = mcolors.to_rgb(color)
    keep = 1.0 - max(0.0, min(1.0, amount))
    return mcolors.to_hex((r * keep, g * keep, b * keep))


def contrast_text_color(bg_color: str) -> str:
    """Returns dark ink or warm white depending on background luminance."""
    r, g, b = mcolors.to_rgb(bg_color)
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return palettes.INK if lum > 0.45 else "#FFFDF7"


def draw_atom_sphere(
    ax: maxes.Axes,
    x: float,
    y: float,
    radius: float,
    facecolor: str,
    edgecolor: str = palettes.INK,
    linewidth: float = 1.5,
    label: str | None = None,
    highlight: bool = True,
    alpha: float = 1.0,
    zorder: float = 3.0,
) -> None:
    """Draws a hand-drawn shaded atomic sphere with optional element symbol.

    Args:
      ax: Target Matplotlib ``Axes``.
      x: Projected horizontal center coordinate.
      y: Projected vertical center coordinate.
      radius: Visual circle radius in data units.
      facecolor: Fill color string.
      edgecolor: Outline stroke color.
      linewidth: Outline width in points.
      label: Optional symbol label centered inside the sphere.
      highlight: Whether to draw an upper-left specular highlight ellipse.
      alpha: Artist opacity in ``[0, 1]``.
      zorder: Matplotlib layer depth ordering.
    """
    circle = patches.Circle(
        (x, y),
        radius=radius,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        alpha=alpha,
        zorder=zorder,
    )
    ax.add_patch(circle)
    if highlight and radius > 0.02:
        spot = patches.Ellipse(
            (x - 0.28 * radius, y + 0.28 * radius),
            width=0.46 * radius,
            height=0.28 * radius,
            angle=38.0,
            facecolor=lighten_hex(facecolor, 0.72),
            edgecolor="none",
            alpha=0.72 * alpha,
            zorder=zorder + 0.05,
        )
        ax.add_patch(spot)
    if label:
        fontsize = max(6.5, min(11.5, radius * 24.0))
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            color=contrast_text_color(facecolor),
            zorder=zorder + 0.1,
        )


def _draw_half_stroke(
    ax: maxes.Axes,
    p_start: np.ndarray,
    p_end: np.ndarray,
    color: str,
    edgecolor: str,
    width: float,
    zorder: float,
    linestyle: str = "-",
) -> None:
    """Draws an inked outer casing behind a colored inner bond segment."""
    xs = [float(p_start[0]), float(p_end[0])]
    ys = [float(p_start[1]), float(p_end[1])]
    if linestyle == "-":
        ax.plot(
            xs,
            ys,
            color=edgecolor,
            linewidth=width + 2.0,
            solid_capstyle="round",
            zorder=zorder,
        )
        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=max(0.8, width),
            solid_capstyle="round",
            zorder=zorder + 0.05,
        )
    else:
        ax.plot(
            xs,
            ys,
            color=edgecolor,
            linewidth=max(1.0, width * 0.55),
            linestyle=linestyle,
            zorder=zorder + 0.05,
        )


def draw_two_tone_bond(
    ax: maxes.Axes,
    p1: np.ndarray,
    p2: np.ndarray,
    c1: str,
    c2: str,
    width: float = 4.2,
    order: float = 1.0,
    edgecolor: str = palettes.INK,
    offset_scale: float = 0.06,
    zorder: float = 2.0,
) -> None:
    """Draws a two-tone ink-outlined bond split at the geometric midpoint."""
    a = np.asarray(p1, dtype=np.float64)
    b = np.asarray(p2, dtype=np.float64)
    vec = b - a
    length = float(np.linalg.norm(vec))
    if length < 1e-6:
        return
    perp = np.array([-vec[1], vec[0]], dtype=np.float64) / length

    if abs(order - 2.0) < 0.25:
        shift = perp * offset_scale
        for sgn in (-1.0, 1.0):
            p_a = a + sgn * shift
            p_b = b + sgn * shift
            mid = 0.5 * (p_a + p_b)
            _draw_half_stroke(ax, p_a, mid, c1, edgecolor, width * 0.62, zorder)
            _draw_half_stroke(ax, mid, p_b, c2, edgecolor, width * 0.62, zorder)
        return

    if abs(order - 3.0) < 0.25:
        shift = perp * (offset_scale * 1.25)
        for sgn in (-1.0, 0.0, 1.0):
            p_a = a + sgn * shift
            p_b = b + sgn * shift
            mid = 0.5 * (p_a + p_b)
            _draw_half_stroke(ax, p_a, mid, c1, edgecolor, width * 0.48, zorder)
            _draw_half_stroke(ax, mid, p_b, c2, edgecolor, width * 0.48, zorder)
        return

    mid = 0.5 * (a + b)
    _draw_half_stroke(ax, a, mid, c1, edgecolor, width, zorder)
    _draw_half_stroke(ax, mid, b, c2, edgecolor, width, zorder)
    if abs(order - 1.5) < 0.2:
        shift = perp * (offset_scale * 1.1)
        _draw_half_stroke(
            ax,
            a + 0.15 * vec + shift,
            b - 0.15 * vec + shift,
            edgecolor,
            edgecolor,
            width * 0.65,
            zorder,
            linestyle="--",
        )


def draw_polyhedron_faces(
    ax: maxes.Axes,
    verts_2d: np.ndarray,
    faces: tuple[tuple[int, int, int], ...],
    depths: np.ndarray,
    facecolor: str,
    edgecolor: str = palettes.INK,
    alpha: float = 0.28,
    base_zorder: float = 1.5,
) -> None:
    """Draws depth-sorted translucent triangular polyhedron facets."""
    ordered = sorted(
        faces,
        key=lambda tri: float(
            (depths[tri[0]] + depths[tri[1]] + depths[tri[2]]) / 3.0
        ),
    )
    for step, (i0, i1, i2) in enumerate(ordered):
        pts = np.vstack([verts_2d[i0], verts_2d[i1], verts_2d[i2]])
        poly = patches.Polygon(
            pts,
            closed=True,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=1.15,
            alpha=alpha,
            joinstyle="round",
            zorder=base_zorder + step * 1e-4,
        )
        ax.add_patch(poly)


def draw_unit_cell_box(
    ax: maxes.Axes,
    cell: np.ndarray,
    camera: geometry.Camera,
    center: np.ndarray | None = None,
    origin: np.ndarray | None = None,
    edgecolor: str = palettes.INK,
    linewidth: float = 1.25,
    show_axes: bool = True,
) -> list[np.ndarray]:
    """Draws a periodic parallelepiped unit cell with dashed back edges.

    Returns:
      List of all projected corner 2D coordinates for bounding-box sizing.
    """
    edges = geometry.unit_cell_edges(cell, camera, center=center, origin=origin)
    pts_2d: list[np.ndarray] = []
    for p0, p1, is_back in edges:
        pts_2d.extend([p0, p1])
        if is_back:
            ax.plot(
                [p0[0], p1[0]],
                [p0[1], p1[1]],
                color=edgecolor,
                linestyle="--",
                linewidth=linewidth * 0.85,
                alpha=0.45,
                zorder=0.4,
            )
        else:
            ax.plot(
                [p0[0], p1[0]],
                [p0[1], p1[1]],
                color=edgecolor,
                linestyle="-",
                linewidth=linewidth,
                alpha=0.85,
                zorder=15.0,
            )

    if show_axes:
        orig_3d = (
            np.zeros(3, dtype=np.float64)
            if origin is None
            else np.asarray(origin, dtype=np.float64)
        )
        axis_tips = np.vstack(
            [
                orig_3d,
                orig_3d + cell[0] * 0.25,
                orig_3d + cell[1] * 0.25,
                orig_3d + cell[2] * 0.25,
            ]
        )
        proj_ctr = (
            orig_3d + 0.5 * (cell[0] + cell[1] + cell[2])
            if center is None
            else center
        )
        tips_2d, _ = geometry.project_points(axis_tips, camera, center=proj_ctr)
        axis_colors = ("#D55E00", "#009E73", "#0072B2")
        axis_names = ("a", "b", "c")
        for k in range(3):
            dx = tips_2d[k + 1][0] - tips_2d[0][0]
            dy = tips_2d[k + 1][1] - tips_2d[0][1]
            if math.hypot(dx, dy) > 0.15:
                ax.annotate(
                    "",
                    xy=(tips_2d[k + 1][0], tips_2d[k + 1][1]),
                    xytext=(tips_2d[0][0], tips_2d[0][1]),
                    arrowprops={
                        "arrowstyle": "->",
                        "color": axis_colors[k],
                        "lw": 1.6,
                    },
                    zorder=16.0,
                )
                ax.text(
                    tips_2d[k + 1][0] + 0.08 * dx,
                    tips_2d[k + 1][1] + 0.08 * dy,
                    axis_names[k],
                    fontsize=9.0,
                    color=axis_colors[k],
                    ha="center",
                    va="center",
                    zorder=16.5,
                )
    return pts_2d


def draw_miller_plane(
    ax: maxes.Axes,
    cell: np.ndarray,
    hkl: tuple[int, int, int],
    camera: geometry.Camera,
    center: np.ndarray | None = None,
    facecolor: str = "#56B4E9",
    edgecolor: str = "#0072B2",
    alpha: float = 0.26,
    show_label: bool = True,
    zorder: float = 2.5,
) -> None:
    """Draws a translucent crystallographic ``(hkl)`` slice inside the cell."""
    poly_3d = geometry.miller_plane_polygon(cell, hkl)
    if poly_3d.shape[0] < 3:
        return
    proj_ctr = 0.5 * (cell[0] + cell[1] + cell[2]) if center is None else center
    poly_2d, _ = geometry.project_points(poly_3d, camera, center=proj_ctr)
    patch = patches.Polygon(
        poly_2d,
        closed=True,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.6,
        alpha=alpha,
        zorder=zorder,
    )
    ax.add_patch(patch)
    if show_label:
        centroid = np.mean(poly_2d, axis=0)
        ax.text(
            float(centroid[0]),
            float(centroid[1]),
            f"({hkl[0]}{hkl[1]}{hkl[2]})",
            fontsize=9.5,
            color=edgecolor,
            ha="center",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": palettes.PAPER,
                "edgecolor": edgecolor,
                "alpha": 0.85,
            },
            zorder=zorder + 0.2,
        )
