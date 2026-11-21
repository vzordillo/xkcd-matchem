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
"""Playful xkcd-style icons for chemical elements and matchem objects."""

from __future__ import annotations

from collections.abc import Callable
import math

from matplotlib import axes as maxes
from matplotlib import colors as mcolors
from matplotlib import patches
from matplotlib import pyplot as plt
import numpy as np

from xkcd_matchem.core import exceptions
from xkcd_matchem.core import geometry
from xkcd_matchem.data import elements
from xkcd_matchem.render import primitives
from xkcd_matchem.style import palettes
from xkcd_matchem.style import theme


_IconFn = Callable[[maxes.Axes, float, float, float, str, str, float], None]


def _icon_flask(
    ax: maxes.Axes,
    cx: float,
    cy: float,
    s: float,
    col: str,
    ink: str,
    z: float,
) -> None:
    """Erlenmeyer flask with bubbling solution."""
    neck_w = 0.22 * s
    base_w = 0.86 * s
    body = np.array(
        [
            [cx - neck_w, cy + 0.50 * s],
            [cx - neck_w, cy + 0.22 * s],
            [cx - 0.5 * base_w, cy - 0.44 * s],
            [cx + 0.5 * base_w, cy - 0.44 * s],
            [cx + neck_w, cy + 0.22 * s],
            [cx + neck_w, cy + 0.50 * s],
        ]
    )
    liquid = np.array(
        [
            [cx - 0.34 * s, cy - 0.10 * s],
            [cx - 0.43 * s, cy - 0.44 * s],
            [cx + 0.43 * s, cy - 0.44 * s],
            [cx + 0.34 * s, cy - 0.10 * s],
        ]
    )
    ax.add_patch(
        patches.Polygon(
            liquid,
            closed=True,
            facecolor=col,
            edgecolor="none",
            alpha=0.72,
            zorder=z,
        )
    )
    ax.add_patch(
        patches.Polygon(
            body,
            closed=True,
            facecolor="none",
            edgecolor=ink,
            linewidth=1.8,
            joinstyle="round",
            zorder=z + 0.2,
        )
    )
    ax.plot(
        [cx - neck_w * 1.25, cx + neck_w * 1.25],
        [cy + 0.50 * s, cy + 0.50 * s],
        color=ink,
        linewidth=2.0,
        zorder=z + 0.3,
    )
    for bx, by, br in (
        (-0.08, -0.24, 0.055),
        (0.11, -0.29, 0.045),
        (0.02, 0.04, 0.048),
    ):
        ax.add_patch(
            patches.Circle(
                (cx + bx * s, cy + by * s),
                radius=br * s,
                facecolor="#FFFFFF",
                edgecolor=ink,
                linewidth=1.0,
                zorder=z + 0.4,
            )
        )


def _icon_beaker(
    ax: maxes.Axes,
    cx: float,
    cy: float,
    s: float,
    col: str,
    ink: str,
    z: float,
) -> None:
    """Graduated chemistry beaker with liquid level and tick marks."""
    w = 0.72 * s
    h = 0.86 * s
    x0, y0 = cx - 0.5 * w, cy - 0.5 * h
    ax.add_patch(
        patches.Rectangle(
            (x0 + 0.03 * s, y0 + 0.02 * s),
            w - 0.06 * s,
            0.52 * h,
            facecolor=col,
            edgecolor="none",
            alpha=0.70,
            zorder=z,
        )
    )
    ax.plot(
        [x0 - 0.06 * s, x0, x0, x0 + w, x0 + w],
        [y0 + h, y0 + h - 0.05 * s, y0, y0, y0 + h],
        color=ink,
        linewidth=1.9,
        zorder=z + 0.2,
    )
    for frac in (0.25, 0.45, 0.65):
        ax.plot(
            [x0 + 0.10 * w, x0 + 0.36 * w],
            [y0 + frac * h, y0 + frac * h],
            color=ink,
            linewidth=1.1,
            zorder=z + 0.3,
        )


def _icon_atom(
    ax: maxes.Axes,
    cx: float,
    cy: float,
    s: float,
    col: str,
    ink: str,
    z: float,
) -> None:
    """Three-orbit stylized Rutherford-Bohr atom icon."""
    for angle in (0.0, 60.0, 120.0):
        ax.add_patch(
            patches.Ellipse(
                (cx, cy),
                width=0.98 * s,
                height=0.38 * s,
                angle=angle,
                facecolor="none",
                edgecolor=ink,
                linewidth=1.4,
                zorder=z,
            )
        )
    primitives.draw_atom_sphere(
        ax,
        cx,
        cy,
        radius=0.14 * s,
        facecolor=col,
        edgecolor=ink,
        zorder=z + 0.3,
    )


def _icon_molecule(
    ax: maxes.Axes,
    cx: float,
    cy: float,
    s: float,
    col: str,
    ink: str,
    z: float,
) -> None:
    """Ball-and-stick triatomic molecule icon."""
    pts = [
        (cx, cy + 0.14 * s),
        (cx - 0.34 * s, cy - 0.20 * s),
        (cx + 0.34 * s, cy - 0.20 * s),
    ]
    for p1 in pts[1:]:
        ax.plot(
            [pts[0][0], p1[0]],
            [pts[0][1], p1[1]],
            color=ink,
            linewidth=2.4,
            zorder=z,
        )
    primitives.draw_atom_sphere(
        ax, pts[0][0], pts[0][1], 0.18 * s, col, ink, zorder=z + 0.3
    )
    for px, py in pts[1:]:
        primitives.draw_atom_sphere(
            ax, px, py, 0.12 * s, "#F5F5F5", ink, zorder=z + 0.3
        )


def _icon_benzene(
    ax: maxes.Axes,
    cx: float,
    cy: float,
    s: float,
    col: str,
    ink: str,
    z: float,
) -> None:
    """Aromatic benzene hexagon with inner ring."""
    angles = np.linspace(0.0, 2.0 * np.pi, 7)[:-1] + np.pi / 6.0
    r = 0.44 * s
    hex_pts = np.column_stack(
        [cx + r * np.cos(angles), cy + r * np.sin(angles)]
    )
    ax.add_patch(
        patches.Polygon(
            hex_pts,
            closed=True,
            facecolor=primitives.lighten_hex(col, 0.72),
            edgecolor=ink,
            linewidth=2.0,
            zorder=z,
        )
    )
    ax.add_patch(
        patches.Circle(
            (cx, cy),
            radius=0.25 * s,
            facecolor="none",
            edgecolor=ink,
            linewidth=1.5,
            linestyle="--",
            zorder=z + 0.2,
        )
    )


def _icon_crystal(
    ax: maxes.Axes,
    cx: float,
    cy: float,
    s: float,
    col: str,
    ink: str,
    z: float,
) -> None:
    """Simple cubic unit cell icon with all 8 corners and 12 edges."""
    cell = np.eye(3, dtype=np.float64)
    cam = geometry.Camera(azimuth=-118.0, elevation=24.0)
    corners_2d, depths, back_idx = geometry.cell_corners_projected(cell, cam)
    span = max(float(np.max(np.abs(corners_2d))), 1e-6)
    pts = np.column_stack(
        [
            cx + corners_2d[:, 0] * (0.44 * s / span),
            cy + corners_2d[:, 1] * (0.44 * s / span),
        ]
    )
    edge_pairs = (
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 4),
        (1, 5),
        (2, 4),
        (2, 6),
        (3, 5),
        (3, 6),
        (4, 7),
        (5, 7),
        (6, 7),
    )
    # Subtle tinted base plane so the 3D box reads immediately.
    ax.add_patch(
        patches.Polygon(
            pts[[0, 1, 4, 2]],
            closed=True,
            facecolor=primitives.lighten_hex(col, 0.55),
            edgecolor="none",
            alpha=0.55,
            zorder=z,
        )
    )
    # 3 dashed back edges meeting at the farthest corner.
    for u, v in edge_pairs:
        if back_idx in (u, v):
            ax.plot(
                [pts[u, 0], pts[v, 0]],
                [pts[u, 1], pts[v, 1]],
                color="#777777",
                linestyle=(0, (3, 2)),
                linewidth=1.25,
                zorder=z + 0.1,
            )
    # Back corner atom drawn behind the front edges.
    primitives.draw_atom_sphere(
        ax,
        float(pts[back_idx, 0]),
        float(pts[back_idx, 1]),
        0.068 * s,
        primitives.lighten_hex(col, 0.35),
        ink,
        linewidth=1.0,
        alpha=0.78,
        zorder=z + 0.15,
    )
    # 9 solid front edges.
    for u, v in edge_pairs:
        if back_idx not in (u, v):
            ax.plot(
                [pts[u, 0], pts[v, 0]],
                [pts[u, 1], pts[v, 1]],
                color=ink,
                linewidth=1.55,
                zorder=z + 0.25,
            )
    # Remaining 7 front/outer corner atoms depth-sorted.
    for idx in sorted(range(8), key=lambda k: float(depths[k])):
        if idx == back_idx:
            continue
        primitives.draw_atom_sphere(
            ax,
            float(pts[idx, 0]),
            float(pts[idx, 1]),
            0.075 * s,
            col,
            ink,
            linewidth=1.15,
            zorder=z + 0.4,
        )


def _icon_graphene(
    ax: maxes.Axes,
    cx: float,
    cy: float,
    s: float,
    col: str,
    ink: str,
    z: float,
) -> None:
    """2D honeycomb graphene lattice swatch."""
    r = 0.21 * s
    dx = 1.5 * r
    dy = math.sqrt(3.0) * r
    centers = [
        (cx - dx * 0.5, cy),
        (cx + dx * 0.5, cy + 0.5 * dy),
        (cx + dx * 0.5, cy - 0.5 * dy),
    ]
    angles = np.linspace(0.0, 2.0 * np.pi, 7)[:-1]
    for hx, hy in centers:
        pts = np.column_stack(
            [hx + r * np.cos(angles), hy + r * np.sin(angles)]
        )
        ax.add_patch(
            patches.Polygon(
                pts,
                closed=True,
                facecolor=primitives.lighten_hex(col, 0.65),
                edgecolor=ink,
                linewidth=1.5,
                zorder=z,
            )
        )


def _icon_battery(
    ax: maxes.Axes,
    cx: float,
    cy: float,
    s: float,
    col: str,
    ink: str,
    z: float,
) -> None:
    """Rechargeable Li-ion battery cell with charge bars."""
    w, h = 0.80 * s, 0.48 * s
    x0, y0 = cx - 0.5 * w, cy - 0.5 * h
    ax.add_patch(
        patches.FancyBboxPatch(
            (x0, y0),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            facecolor="#FFFFFF",
            edgecolor=ink,
            linewidth=1.8,
            zorder=z,
        )
    )
    ax.add_patch(
        patches.Rectangle(
            (x0 + w + 0.01 * s, cy - 0.12 * s),
            0.09 * s,
            0.24 * s,
            facecolor=ink,
            edgecolor=ink,
            zorder=z,
        )
    )
    for k in range(3):
        ax.add_patch(
            patches.Rectangle(
                (x0 + (0.08 + k * 0.23) * w, y0 + 0.14 * h),
                0.17 * w,
                0.72 * h,
                facecolor=col,
                edgecolor=ink,
                linewidth=1.0,
                zorder=z + 0.2,
            )
        )


def _icon_catalyst(
    ax: maxes.Axes,
    cx: float,
    cy: float,
    s: float,
    col: str,
    ink: str,
    z: float,
) -> None:
    """Catalytic surface slab with adsorbed species and activity spark."""
    ax.add_patch(
        patches.Rectangle(
            (cx - 0.44 * s, cy - 0.42 * s),
            0.88 * s,
            0.28 * s,
            facecolor=primitives.lighten_hex(col, 0.35),
            edgecolor=ink,
            linewidth=1.6,
            hatch="///",
            zorder=z,
        )
    )
    ax.plot(
        [cx, cx],
        [cy - 0.14 * s, cy + 0.10 * s],
        color=ink,
        linestyle="--",
        linewidth=1.5,
        zorder=z + 0.2,
    )
    primitives.draw_atom_sphere(
        ax, cx, cy + 0.16 * s, 0.14 * s, "#E76F51", ink, zorder=z + 0.3
    )
    primitives.draw_atom_sphere(
        ax,
        cx + 0.22 * s,
        cy + 0.30 * s,
        0.10 * s,
        "#F4A261",
        ink,
        zorder=z + 0.3,
    )


def _icon_magnet(
    ax: maxes.Axes,
    cx: float,
    cy: float,
    s: float,
    col: str,
    ink: str,
    z: float,
) -> None:
    """Classic U-shaped horseshoe magnet icon."""
    theta = np.linspace(0.0, np.pi, 24)
    r_out, r_in = 0.36 * s, 0.18 * s
    outer = np.column_stack(
        [cx + r_out * np.cos(theta), cy + 0.08 * s + r_out * np.sin(theta)]
    )
    inner = np.column_stack(
        [cx + r_in * np.cos(theta), cy + 0.08 * s + r_in * np.sin(theta)]
    )
    pts = np.vstack(
        [
            [[cx + r_out, cy - 0.36 * s]],
            outer,
            [[cx - r_out, cy - 0.36 * s]],
            [[cx - r_in, cy - 0.36 * s]],
            inner[::-1],
            [[cx + r_in, cy - 0.36 * s]],
        ]
    )
    ax.add_patch(
        patches.Polygon(
            pts,
            closed=True,
            facecolor=col,
            edgecolor=ink,
            linewidth=1.7,
            zorder=z,
        )
    )
    for sx in (-r_out, r_in):
        ax.add_patch(
            patches.Rectangle(
                (cx + sx, cy - 0.36 * s),
                r_out - r_in,
                0.15 * s,
                facecolor="#E5E7EB",
                edgecolor=ink,
                linewidth=1.2,
                zorder=z + 0.2,
            )
        )


def _icon_photon(
    ax: maxes.Axes,
    cx: float,
    cy: float,
    s: float,
    col: str,
    ink: str,
    z: float,
) -> None:
    """Wavy photon / spectroscopy light arrow icon."""
    xs = np.linspace(cx - 0.44 * s, cx + 0.30 * s, 40)
    ys = cy + 0.14 * s * np.sin(np.linspace(0.0, 5.0 * np.pi, 40))
    ax.plot(xs, ys, color=col, linewidth=2.3, zorder=z)
    ax.annotate(
        "",
        xy=(cx + 0.48 * s, cy),
        xytext=(cx + 0.24 * s, cy),
        arrowprops={"arrowstyle": "->", "color": col, "lw": 2.2},
        zorder=z + 0.2,
    )
    ax.text(
        cx,
        cy + 0.28 * s,
        "hv",
        ha="center",
        va="center",
        fontsize=9.0,
        color=ink,
        zorder=z + 0.3,
    )


def _icon_xrd(
    ax: maxes.Axes,
    cx: float,
    cy: float,
    s: float,
    col: str,
    ink: str,
    z: float,
) -> None:
    """Mini Bragg diffraction peaks icon."""
    xs = np.linspace(-0.44, 0.44, 80)
    ys = (
        0.58 * np.exp(-(((xs + 0.18) / 0.045) ** 2))
        + 0.38 * np.exp(-(((xs - 0.06) / 0.045) ** 2))
        + 0.24 * np.exp(-(((xs - 0.27) / 0.045) ** 2))
    )
    px = cx + xs * s
    py = cy - 0.30 * s + ys * s
    ax.fill_between(px, cy - 0.30 * s, py, color=col, alpha=0.35, zorder=z)
    ax.plot(px, py, color=col, linewidth=1.9, zorder=z + 0.2)
    ax.plot(
        [cx - 0.46 * s, cx + 0.46 * s],
        [cy - 0.30 * s, cy - 0.30 * s],
        color=ink,
        linewidth=1.6,
        zorder=z + 0.3,
    )


def _icon_thermometer(
    ax: maxes.Axes,
    cx: float,
    cy: float,
    s: float,
    col: str,
    ink: str,
    z: float,
) -> None:
    """Laboratory bulb thermometer icon."""
    ax.add_patch(
        patches.FancyBboxPatch(
            (cx - 0.09 * s, cy - 0.20 * s),
            0.18 * s,
            0.66 * s,
            boxstyle="round,pad=0.01,rounding_size=0.09",
            facecolor="#FFFFFF",
            edgecolor=ink,
            linewidth=1.6,
            zorder=z,
        )
    )
    ax.add_patch(
        patches.Rectangle(
            (cx - 0.045 * s, cy - 0.22 * s),
            0.09 * s,
            0.46 * s,
            facecolor=col,
            edgecolor="none",
            zorder=z + 0.1,
        )
    )
    ax.add_patch(
        patches.Circle(
            (cx, cy - 0.28 * s),
            radius=0.17 * s,
            facecolor=col,
            edgecolor=ink,
            linewidth=1.6,
            zorder=z + 0.2,
        )
    )


def _icon_flame(
    ax: maxes.Axes,
    cx: float,
    cy: float,
    s: float,
    col: str,
    ink: str,
    z: float,
) -> None:
    """Furnace / Bunsen heat flame icon."""
    outer = np.array(
        [
            [cx, cy + 0.48 * s],
            [cx + 0.28 * s, cy + 0.06 * s],
            [cx + 0.32 * s, cy - 0.22 * s],
            [cx + 0.14 * s, cy - 0.42 * s],
            [cx - 0.14 * s, cy - 0.42 * s],
            [cx - 0.32 * s, cy - 0.22 * s],
            [cx - 0.24 * s, cy + 0.08 * s],
        ]
    )
    inner = np.array(
        [
            [cx, cy + 0.16 * s],
            [cx + 0.14 * s, cy - 0.16 * s],
            [cx, cy - 0.36 * s],
            [cx - 0.14 * s, cy - 0.16 * s],
        ]
    )
    ax.add_patch(
        patches.Polygon(
            outer, closed=True, facecolor=col, edgecolor=ink, lw=1.7, zorder=z
        )
    )
    ax.add_patch(
        patches.Polygon(
            inner,
            closed=True,
            facecolor="#FDE047",
            edgecolor="none",
            zorder=z + 0.2,
        )
    )


def _icon_dft_cpu(
    ax: maxes.Axes,
    cx: float,
    cy: float,
    s: float,
    col: str,
    ink: str,
    z: float,
) -> None:
    """Compute chip icon for DFT / MD / HPC workflows."""
    w = 0.56 * s
    ax.add_patch(
        patches.FancyBboxPatch(
            (cx - 0.5 * w, cy - 0.5 * w),
            w,
            w,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            facecolor=col,
            edgecolor=ink,
            linewidth=1.7,
            zorder=z,
        )
    )
    for offset in (-0.16, 0.0, 0.16):
        ax.plot(
            [cx + offset * s, cx + offset * s],
            [cy + 0.28 * s, cy + 0.44 * s],
            color=ink,
            lw=1.5,
            zorder=z - 0.1,
        )
        ax.plot(
            [cx + offset * s, cx + offset * s],
            [cy - 0.28 * s, cy - 0.44 * s],
            color=ink,
            lw=1.5,
            zorder=z - 0.1,
        )
        ax.plot(
            [cx + 0.28 * s, cx + 0.44 * s],
            [cy + offset * s, cy + offset * s],
            color=ink,
            lw=1.5,
            zorder=z - 0.1,
        )
        ax.plot(
            [cx - 0.28 * s, cx - 0.44 * s],
            [cy + offset * s, cy + offset * s],
            color=ink,
            lw=1.5,
            zorder=z - 0.1,
        )
    ax.text(
        cx,
        cy,
        "DFT",
        ha="center",
        va="center",
        fontsize=8.5,
        color=primitives.contrast_text_color(mcolors.to_hex(col)),
        zorder=z + 0.2,
    )


def _icon_neural_net(
    ax: maxes.Axes,
    cx: float,
    cy: float,
    s: float,
    col: str,
    ink: str,
    z: float,
) -> None:
    """Mini 3-layer graph neural network / MLIP icon."""
    layers = [
        [(cx - 0.34 * s, cy - 0.22 * s), (cx - 0.34 * s, cy + 0.22 * s)],
        [
            (cx, cy - 0.30 * s),
            (cx, cy),
            (cx, cy + 0.30 * s),
        ],
        [(cx + 0.34 * s, cy - 0.16 * s), (cx + 0.34 * s, cy + 0.16 * s)],
    ]
    for l_idx in range(2):
        for p0 in layers[l_idx]:
            for p1 in layers[l_idx + 1]:
                ax.plot(
                    [p0[0], p1[0]],
                    [p0[1], p1[1]],
                    color=ink,
                    linewidth=1.1,
                    alpha=0.7,
                    zorder=z,
                )
    for layer in layers:
        for px, py in layer:
            primitives.draw_atom_sphere(
                ax, px, py, 0.08 * s, col, ink, linewidth=1.2, zorder=z + 0.3
            )


_ICON_REGISTRY: dict[str, tuple[_IconFn, str]] = {
    "flask": (_icon_flask, "#2A9D8F"),
    "beaker": (_icon_beaker, "#56B4E9"),
    "atom": (_icon_atom, "#E76F51"),
    "molecule": (_icon_molecule, "#E63946"),
    "benzene": (_icon_benzene, "#457B9D"),
    "crystal": (_icon_crystal, "#56B4E9"),
    "graphene": (_icon_graphene, "#2A9D8F"),
    "battery": (_icon_battery, "#2ECC71"),
    "catalyst": (_icon_catalyst, "#7F8C9A"),
    "magnet": (_icon_magnet, "#E74C3C"),
    "photon": (_icon_photon, "#E69F00"),
    "xrd": (_icon_xrd, "#0072B2"),
    "thermometer": (_icon_thermometer, "#D55E00"),
    "flame": (_icon_flame, "#F4A261"),
    "dft": (_icon_dft_cpu, "#3D405B"),
    "neural_net": (_icon_neural_net, "#9B5DE5"),
}


def list_icons() -> tuple[str, ...]:
    """Returns sorted names of all built-in xkcd chemistry/materials icons."""
    return tuple(sorted(_ICON_REGISTRY.keys()))


def draw_element_badge(
    ax: maxes.Axes,
    symbol: str,
    xy: tuple[float, float],
    *,
    size: float = 1.0,
    palette: str = "xkcd",
    color: str | None = None,
    edgecolor: str | None = None,
    show_mass: bool = True,
    zorder: float = 5.0,
) -> None:
    """Draws a standalone xkcd-style chemical element icon tile on an Axes.

    Args:
      ax: Target Matplotlib ``Axes``.
      symbol: Chemical symbol (e.g. ``"Ti"``, ``"Li"``, ``"C"``, ``"Au"``).
      xy: Center ``(x, y)`` position in data coordinates.
      size: Side length of the element tile in data units.
      palette: Element color scheme name used when ``color`` is ``None``.
      color: Optional custom hex tile fill override.
      edgecolor: Optional outline color; defaults to active theme ink.
      show_mass: Whether to print atomic mass along the bottom edge.
      zorder: Layer z-order.
    """
    rec = elements.get_element(symbol)
    fill = (
        palettes.get_element_color(rec.symbol, scheme=palette)
        if color is None
        else color
    )
    ink = (
        str(plt.rcParams.get("axes.edgecolor", palettes.INK))
        if edgecolor is None
        else edgecolor
    )
    cx, cy = float(xy[0]), float(xy[1])
    x0, y0 = cx - 0.46 * size, cy - 0.46 * size
    ax.add_patch(
        patches.FancyBboxPatch(
            (x0, y0),
            0.92 * size,
            0.92 * size,
            boxstyle=f"round,pad={0.02 * size},rounding_size={0.12 * size}",
            facecolor=fill,
            edgecolor=ink,
            linewidth=1.7,
            zorder=zorder,
        )
    )
    txt_col = primitives.contrast_text_color(mcolors.to_hex(fill))
    ax.text(
        x0 + 0.10 * size,
        y0 + 0.80 * size,
        str(rec.atomic_number),
        ha="left",
        va="center",
        fontsize=8.0,
        color=txt_col,
        zorder=zorder + 0.3,
    )
    ax.text(
        cx,
        cy + 0.02 * size,
        rec.symbol,
        ha="center",
        va="center",
        fontsize=14.5,
        color=txt_col,
        zorder=zorder + 0.3,
    )
    if show_mass:
        ax.text(
            cx,
            y0 + 0.14 * size,
            f"{rec.atomic_mass:.1f}",
            ha="center",
            va="center",
            fontsize=7.2,
            color=txt_col,
            zorder=zorder + 0.3,
        )


def draw_icon(
    ax: maxes.Axes,
    name: str,
    xy: tuple[float, float],
    *,
    size: float = 1.0,
    color: str | None = None,
    edgecolor: str | None = None,
    label: str | None = None,
    zorder: float = 5.0,
) -> None:
    """Draws a named xkcd matchem icon or element badge onto an ``Axes``.

    Accepts any built-in icon key from ``list_icons()`` (such as ``"flask"``,
    ``"crystal"``, ``"battery"``, ``"graphene"``, ``"catalyst"``, ``"dft"``,
    ``"neural_net"``) OR any element symbol prefixed with ``"element:"``
    (e.g. ``"element:Ti"``).

    Args:
      ax: Target Matplotlib ``Axes``.
      name: Icon name or ``"element:<Symbol>"``.
      xy: Center ``(x, y)`` position in data coordinates.
      size: Icon diameter in data units.
      color: Optional accent fill color override.
      edgecolor: Optional ink stroke override.
      label: Optional caption printed underneath the icon.
      zorder: Matplotlib z-order.

    Raises:
      StyleError: If ``name`` is not a known icon or element key.
    """
    ink = (
        str(plt.rcParams.get("axes.edgecolor", palettes.INK))
        if edgecolor is None
        else edgecolor
    )
    key = name.strip()
    if key.lower().startswith("element:"):
        sym = key.split(":", 1)[1].strip()
        draw_element_badge(
            ax, sym, xy, size=size, color=color, edgecolor=ink, zorder=zorder
        )
    else:
        low = key.lower()
        if low not in _ICON_REGISTRY:
            raise exceptions.StyleError(
                f"Unknown icon {name!r}. Available: {list_icons()} "
                "or 'element:<Symbol>'."
            )
        fn, default_col = _ICON_REGISTRY[low]
        fn(
            ax,
            float(xy[0]),
            float(xy[1]),
            float(size),
            default_col if color is None else color,
            ink,
            zorder,
        )
    if label:
        ax.text(
            float(xy[0]),
            float(xy[1]) - 0.62 * size,
            label,
            ha="center",
            va="top",
            fontsize=9.2,
            color=ink,
            zorder=zorder + 0.4,
        )


def plot_icon_sheet(
    icons: list[str] | tuple[str, ...] | None = None,
    *,
    elements_to_show: tuple[str, ...] = ("Li", "C", "Ti", "Fe", "Au", "Si"),
    ncols: int = 6,
    ax: maxes.Axes | None = None,
    preset: str | theme.ThemePreset = "classic",
    title: str | None = "xkcd-matchem Icon & Element Badge Pack",
) -> maxes.Axes:
    """Displays a visual catalog sheet of xkcd matchem icons & element tiles."""
    names = list(list_icons()) if icons is None else list(icons)
    all_items = names + [f"element:{sym}" for sym in elements_to_show]
    nrows = max(1, math.ceil(len(all_items) / max(ncols, 1)))
    ink = theme.get_preset(preset).ink_color

    with theme.xkcd_style(preset=preset):
        if ax is None:
            _, ax = plt.subplots(figsize=(ncols * 1.9, nrows * 1.9 + 0.6))
        else:
            theme.style_existing_axes(ax, preset=preset)
        for idx, item in enumerate(all_items):
            r_idx = idx // ncols
            c_idx = idx % ncols
            cx = c_idx * 1.65
            cy = (nrows - 1 - r_idx) * 1.75
            caption = (
                item.split(":", 1)[1] if item.startswith("element:") else item
            )
            draw_icon(ax, item, (cx, cy), size=0.95, label=caption)
        ax.set_xlim(-0.9, (ncols - 1) * 1.65 + 0.9)
        ax.set_ylim(-0.95, (nrows - 1) * 1.75 + 0.85)
        ax.set_aspect("equal")
        ax.set_axis_off()
        if title:
            ax.set_title(title, color=ink, pad=8.0)
    return ax
