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
"""xkcd-style arrows, callouts, bond distances, angles, and element legends."""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any
from typing import cast

from matplotlib import axes as maxes
from matplotlib import patches
from matplotlib import pyplot as plt
import matplotlib.transforms
import numpy as np

from xkcd_matchem.core import geometry
from xkcd_matchem.core import structure
from xkcd_matchem.style import palettes


def draw_curved_arrow(
    ax: maxes.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    rad: float = 0.25,
    color: str | None = None,
    linewidth: float = 1.8,
    label: str | None = None,
) -> None:
    """Draws an xkcd-style curved reaction or mechanism arrow."""
    ink = str(plt.rcParams.get("axes.edgecolor", palettes.INK))
    col = ink if color is None else color
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "->,head_width=0.32,head_length=0.45",
            "connectionstyle": f"arc3,rad={rad}",
            "color": col,
            "lw": linewidth,
        },
        zorder=18.0,
    )
    if label:
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        mid_x = 0.5 * (start[0] + end[0]) + 0.65 * rad * dy
        mid_y = 0.5 * (start[1] + end[1]) - 0.65 * rad * dx
        ax.text(
            mid_x,
            mid_y,
            label,
            ha="center",
            va="center",
            fontsize=10.0,
            color=col,
            zorder=19.0,
        )


def draw_callout(
    ax: maxes.Axes,
    text: str,
    xy: tuple[float, float],
    xytext: tuple[float, float],
    color: str = "#D55E00",
    fontsize: float = 10.0,
) -> None:
    """Draws an xkcd-style boxed callout pointing to an annotated coordinate."""
    bg = str(plt.rcParams.get("figure.facecolor", palettes.PAPER))
    ax.annotate(
        text,
        xy=xy,
        xytext=xytext,
        fontsize=fontsize,
        color=color,
        ha="center",
        va="center",
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": bg,
            "edgecolor": color,
            "lw": 1.3,
        },
        arrowprops={
            "arrowstyle": "->",
            "color": color,
            "lw": 1.4,
            "connectionstyle": "arc3,rad=-0.15",
        },
        zorder=20.0,
    )


def annotate_distance(
    ax: maxes.Axes,
    atoms: structure.AtomsData,
    i: int,
    j: int,
    camera: geometry.Camera | str | None = None,
    color: str = "#0072B2",
    offset: float = 0.25,
) -> float:
    """Annotates an interatomic distance (Angstrom) onto a 3D projected view."""
    cam = geometry.resolve_camera(camera, cell=atoms.cell)
    pivot = (
        0.5 * (atoms.cell[0] + atoms.cell[1] + atoms.cell[2])
        if atoms.cell is not None
        else np.mean(atoms.positions, axis=0)
    )
    proj_span = geometry.scene_projection_span(
        atoms.positions, cam, center=pivot, cell=atoms.cell
    )
    vec = atoms.positions[j] - atoms.positions[i]
    if atoms.cell is not None and any(atoms.pbc):
        vec = geometry.minimum_image_vector(vec, atoms.cell, atoms.pbc)
    dist = float(np.linalg.norm(vec))
    pts_2d, _ = geometry.project_points(
        np.vstack([atoms.positions[i], atoms.positions[i] + vec]),
        cam,
        center=pivot,
        span=proj_span,
    )
    p1 = pts_2d[0]
    p2 = pts_2d[1]
    span = p2 - p1
    norm = max(float(np.linalg.norm(span)), 1e-6)
    perp = np.array([-span[1], span[0]], dtype=np.float64) / norm * offset
    ax.plot(
        [p1[0] + perp[0], p2[0] + perp[0]],
        [p1[1] + perp[1], p2[1] + perp[1]],
        color=color,
        linestyle=":",
        lw=1.5,
        zorder=19.0,
    )
    mid = 0.5 * (p1 + p2) + perp * 1.4
    bg = str(plt.rcParams.get("figure.facecolor", palettes.PAPER))
    ax.text(
        float(mid[0]),
        float(mid[1]),
        f"{dist:.2f} A",
        fontsize=9.5,
        color=color,
        ha="center",
        va="center",
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": bg,
            "edgecolor": color,
        },
        zorder=19.5,
    )
    return dist


def annotate_angle(
    ax: maxes.Axes,
    atoms: structure.AtomsData,
    i: int,
    j: int,
    k: int,
    camera: geometry.Camera | str | None = None,
    color: str = "#009E73",
) -> float:
    """Annotates the ``i-j-k`` bond angle centered at atom ``j`` in degrees."""
    v1 = atoms.positions[i] - atoms.positions[j]
    v2 = atoms.positions[k] - atoms.positions[j]
    if atoms.cell is not None and any(atoms.pbc):
        v1 = geometry.minimum_image_vector(v1, atoms.cell, atoms.pbc)
        v2 = geometry.minimum_image_vector(v2, atoms.cell, atoms.pbc)
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cos_th = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    deg = math.degrees(math.acos(cos_th))
    cam = geometry.resolve_camera(camera, cell=atoms.cell)
    pivot = (
        0.5 * (atoms.cell[0] + atoms.cell[1] + atoms.cell[2])
        if atoms.cell is not None
        else np.mean(atoms.positions, axis=0)
    )
    proj_span = geometry.scene_projection_span(
        atoms.positions, cam, center=pivot, cell=atoms.cell
    )
    pts_2d, _ = geometry.project_points(
        np.vstack(
            [
                atoms.positions[j],
                atoms.positions[j] + v1,
                atoms.positions[j] + v2,
            ]
        ),
        cam,
        center=pivot,
        span=proj_span,
    )
    u1 = pts_2d[1] - pts_2d[0]
    u2 = pts_2d[2] - pts_2d[0]
    u1 /= max(float(np.linalg.norm(u1)), 1e-6)
    u2 /= max(float(np.linalg.norm(u2)), 1e-6)
    dir_avg = u1 + u2
    norm = float(np.linalg.norm(dir_avg))
    if norm < 1e-4:
        dir_avg = np.array([-u1[1], u1[0]], dtype=np.float64)
        norm = max(float(np.linalg.norm(dir_avg)), 1e-6)
    pos = pts_2d[0] + 0.55 * (dir_avg / norm)
    bg = str(plt.rcParams.get("figure.facecolor", palettes.PAPER))
    ax.text(
        float(pos[0]),
        float(pos[1]),
        f"{deg:.1f}\u00b0",
        fontsize=9.0,
        color=color,
        ha="center",
        va="center",
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": bg,
            "edgecolor": color,
        },
        zorder=19.5,
    )
    return deg


def draw_element_legend(
    ax: maxes.Axes,
    symbols: list[str],
    scheme: str | Mapping[str, str] = "xkcd",
    loc: str = "lower center",
    style: str = "inline",
) -> None:
    """Adds a clean inline circular legend or an optional boxed legend."""
    seen: list[str] = []
    for sym in symbols:
        if sym not in seen:
            seen.append(sym)
    if not seen:
        return
    if style == "box":
        handles = [
            patches.Patch(
                facecolor=palettes.get_element_color(sym, scheme=scheme),
                edgecolor=palettes.INK,
                linewidth=1.2,
                label=sym,
            )
            for sym in seen
        ]
        ax.legend(
            handles=handles,
            loc=cast(Any, loc),
            frameon=True,
            ncol=min(len(handles), 4),
            fontsize=9.0,
            framealpha=0.9,
        )
        return

    ink = str(plt.rcParams.get("axes.edgecolor", palettes.INK))
    y_ax = 0.945 if "upper" in loc.lower() else 0.055
    n_el = len(seen)
    step = min(0.18, 0.78 / max(n_el, 1))
    x0 = 0.5 - 0.5 * (n_el - 1) * step
    hi_trans = matplotlib.transforms.offset_copy(
        ax.transAxes,
        fig=cast(Any, ax.figure),
        x=-1.6,
        y=1.6,
        units="points",
    )
    txt_trans = matplotlib.transforms.offset_copy(
        ax.transAxes,
        fig=cast(Any, ax.figure),
        x=8.5,
        y=-0.3,
        units="points",
    )
    for k, sym in enumerate(seen):
        x_ax = x0 + k * step - 0.015
        col = palettes.get_element_color(sym, scheme=scheme)
        ax.scatter(
            [x_ax],
            [y_ax],
            s=115.0,
            facecolor=col,
            edgecolor=ink,
            linewidth=1.35,
            transform=ax.transAxes,
            clip_on=False,
            zorder=900.0,
        )
        ax.scatter(
            [x_ax],
            [y_ax],
            s=12.0,
            facecolor="#FFFFFF",
            edgecolor="none",
            alpha=0.85,
            transform=hi_trans,
            clip_on=False,
            zorder=901.0,
        )
        ax.text(
            x_ax,
            y_ax,
            sym,
            ha="left",
            va="center",
            fontsize=10.5,
            color=ink,
            transform=txt_trans,
            clip_on=False,
            zorder=901.0,
        )


def draw_defect_site(
    ax: maxes.Axes,
    atoms: structure.AtomsData,
    site: tuple[float, float, float] | np.ndarray,
    label: str = "V_O",
    *,
    fractional: bool = False,
    kind: str = "vacancy",
    camera: geometry.Camera | str | None = None,
    color: str = "#D55E00",
    label_offset: tuple[float, float] = (1.1, 1.1),
) -> None:
    """Marks a 3D vacancy, interstitial, or dopant site on a structure plot.

    Args:
      ax: Target Matplotlib ``Axes`` displaying ``atoms``.
      atoms: Reference ``AtomsData`` used to align the 3D camera projection.
      site: Cartesian (or fractional if ``fractional=True``) 3D coordinate.
      label: Callout label such as ``"V_O"`` or ``"Li_i"``.
      fractional: Treat ``site`` as fractional cell coordinates when True.
      kind: Visual marker style: ``"vacancy"`` (dashed square cage),
        ``"interstitial"`` (starred circle), or ``"dopant"`` (double ring).
      camera: Camera matching the underlying structure render.
      color: Accent color for the defect marker and callout badge.
      label_offset: 2D offset ``(dx, dy)`` in projected coordinates for label.
    """
    site_arr = np.asarray(site, dtype=np.float64).reshape(1, 3)
    if fractional:
        if atoms.cell is None:
            raise ValueError("fractional=True requires atoms.cell.")
        site_arr = site_arr @ atoms.cell
    cam = geometry.resolve_camera(camera, cell=atoms.cell)
    pivot = (
        0.5 * (atoms.cell[0] + atoms.cell[1] + atoms.cell[2])
        if atoms.cell is not None
        else np.mean(atoms.positions, axis=0)
    )
    proj_span = geometry.scene_projection_span(
        atoms.positions, cam, center=pivot, cell=atoms.cell
    )
    pts_2d, _ = geometry.project_points(
        site_arr, cam, center=pivot, span=proj_span
    )
    cx, cy = float(pts_2d[0, 0]), float(pts_2d[0, 1])
    mode = kind.strip().lower()
    if mode == "interstitial":
        ax.scatter(
            [cx],
            [cy],
            s=190.0,
            marker="*",
            facecolor=color,
            edgecolor=palettes.INK,
            linewidth=1.1,
            zorder=22.0,
        )
    elif mode == "dopant":
        ax.add_patch(
            patches.Circle(
                (cx, cy),
                radius=0.42,
                facecolor="none",
                edgecolor=color,
                linewidth=2.0,
                linestyle="-",
                zorder=22.0,
            )
        )
        ax.add_patch(
            patches.Circle(
                (cx, cy),
                radius=0.54,
                facecolor="none",
                edgecolor=color,
                linewidth=1.3,
                linestyle="--",
                zorder=22.0,
            )
        )
    else:
        ax.add_patch(
            patches.FancyBboxPatch(
                (cx - 0.34, cy - 0.34),
                0.68,
                0.68,
                boxstyle="round,pad=0.06,rounding_size=0.14",
                facecolor="none",
                edgecolor=color,
                linewidth=1.9,
                linestyle="--",
                zorder=22.0,
            )
        )
    if label:
        draw_callout(
            ax,
            label,
            xy=(cx, cy),
            xytext=(cx + label_offset[0], cy + label_offset[1]),
            color=color,
            fontsize=9.5,
        )
