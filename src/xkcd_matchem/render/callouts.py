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
"""Hand-drawn arrows, callouts, bond distances, angles, and element legends."""

from __future__ import annotations

import math

from matplotlib import axes as maxes
from matplotlib import patches
from matplotlib import pyplot as plt
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
    """Draws a hand-drawn curved reaction or mechanism arrow."""
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
        mid_x = 0.5 * (start[0] + end[0])
        mid_y = 0.5 * (start[1] + end[1]) + 0.15 * abs(end[0] - start[0]) * (
            1.0 if rad >= 0.0 else -1.0
        )
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
    """Draws a hand-drawn boxed callout pointing to an annotated coordinate."""
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
    camera: geometry.Camera | None = None,
    color: str = "#0072B2",
    offset: float = 0.25,
) -> float:
    """Annotates an interatomic distance (Angstrom) onto a 3D projected view."""
    cam = geometry.Camera() if camera is None else camera
    pivot = (
        0.5 * (atoms.cell[0] + atoms.cell[1] + atoms.cell[2])
        if atoms.cell is not None
        else np.mean(atoms.positions, axis=0)
    )
    xy, _ = geometry.project_points(atoms.positions, cam, center=pivot)
    vec = atoms.positions[j] - atoms.positions[i]
    if atoms.cell is not None and any(atoms.pbc):
        vec = geometry.minimum_image_vector(vec, atoms.cell, atoms.pbc)
    dist = float(np.linalg.norm(vec))

    p1 = xy[i]
    p2 = xy[j]
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
    camera: geometry.Camera | None = None,
    color: str = "#009E73",
) -> float:
    """Annotates the ``i-j-k`` bond angle centered at atom ``j`` in degrees."""
    v1 = atoms.positions[i] - atoms.positions[j]
    v2 = atoms.positions[k] - atoms.positions[j]
    if atoms.cell is not None and any(atoms.pbc):
        v1 = geometry.minimum_image_vector(v1, atoms.cell, atoms.pbc)
        v2 = geometry.minimum_image_vector(v2, atoms.cell, atoms.pbc)
    cos_th = float(
        np.clip(
            np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)),
            -1.0,
            1.0,
        )
    )
    deg = math.degrees(math.acos(cos_th))
    cam = geometry.Camera() if camera is None else camera
    pivot = (
        0.5 * (atoms.cell[0] + atoms.cell[1] + atoms.cell[2])
        if atoms.cell is not None
        else np.mean(atoms.positions, axis=0)
    )
    xy, _ = geometry.project_points(atoms.positions, cam, center=pivot)
    dir_avg = (xy[i] - xy[j]) + (xy[k] - xy[j])
    norm = max(float(np.linalg.norm(dir_avg)), 1e-6)
    pos = xy[j] + 0.55 * (dir_avg / norm)
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
    scheme: str = "xkcd",
    loc: str = "lower right",
) -> None:
    """Adds a compact hand-drawn legend showing element color swatches."""
    seen: list[str] = []
    for sym in symbols:
        if sym not in seen:
            seen.append(sym)
    handles = [
        patches.Patch(
            facecolor=palettes.get_element_color(sym, scheme=scheme),
            edgecolor=palettes.INK,
            linewidth=1.2,
            label=sym,
        )
        for sym in seen
    ]
    if handles:
        ax.legend(
            handles=handles,
            loc=loc,
            frameon=True,
            ncol=min(len(handles), 4),
            fontsize=9.0,
            framealpha=0.9,
        )
