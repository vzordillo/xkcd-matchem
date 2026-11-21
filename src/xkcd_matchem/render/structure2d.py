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
"""Clean 2D skeletal chemical diagram renderer."""

from __future__ import annotations

import collections
from collections.abc import Mapping

from matplotlib import axes as maxes
from matplotlib import patches
from matplotlib import pyplot as plt
import numpy as np

from xkcd_matchem.core import bonding
from xkcd_matchem.core import structure
from xkcd_matchem.style import palettes


def _pca_2d_projection(positions: np.ndarray) -> np.ndarray:
    """Projects 3D molecular coordinates onto their top-2 variance axes."""
    centered = positions - np.mean(positions, axis=0)
    if centered.shape[0] <= 1:
        return np.zeros((centered.shape[0], 2), dtype=np.float64)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    proj = centered @ vt[:2].T
    # Enforce deterministic orientation using widest positive spread.
    if proj.shape[0] >= 2 and float(np.max(proj[:, 0])) < abs(
        float(np.min(proj[:, 0]))
    ):
        proj[:, 0] *= -1.0
    return proj


def _find_ring_centroids(
    bonds: list[structure.Bond],
    neighbors: dict[int, list[tuple[int, float]]],
    xy: np.ndarray,
    max_ring_size: int = 8,
) -> dict[tuple[int, int], np.ndarray]:
    """Maps each small-ring bond edge to its shortest-cycle 2D centroid."""
    centroids: dict[tuple[int, int], np.ndarray] = {}
    for b in bonds:
        u, v = b.i, b.j
        queue: collections.deque[tuple[int, list[int]]] = collections.deque(
            [(u, [u])]
        )
        visited = {u}
        found_cycle: list[int] | None = None
        while queue:
            curr, path = queue.popleft()
            if len(path) >= max_ring_size:
                continue
            for nxt, _ in neighbors[curr]:
                if curr == u and nxt == v:
                    continue
                if nxt == v:
                    found_cycle = path + [v]
                    break
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, path + [nxt]))
            if found_cycle is not None:
                break
        if found_cycle is not None:
            key = (min(u, v), max(u, v))
            centroids[key] = np.mean(xy[found_cycle], axis=0)
    return centroids


def _draw_skeletal_bond(
    ax: maxes.Axes,
    p1: np.ndarray,
    p2: np.ndarray,
    order: float,
    ink_color: str,
    shorten_i: float = 0.0,
    shorten_j: float = 0.0,
    ring_center: np.ndarray | None = None,
) -> None:
    """Draws single, double, triple, or aromatic 2D skeletal lines."""
    vec = p2 - p1
    length = float(np.linalg.norm(vec))
    if length < 1e-6:
        return
    unit = vec / length
    perp = np.array([-unit[1], unit[0]], dtype=np.float64)
    start = p1 + unit * shorten_i
    end = p2 - unit * shorten_j

    if abs(order - 2.0) < 0.25:
        if ring_center is not None:
            mid = 0.5 * (p1 + p2)
            if float(np.dot(perp, ring_center - mid)) < 0.0:
                perp = -perp
            ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                color=ink_color,
                lw=2.1,
                solid_capstyle="round",
                zorder=2,
            )
            inner_off = perp * 0.165
            trim = 0.14 * (end - start)
            in_s = start + trim + inner_off
            in_e = end - trim + inner_off
            ax.plot(
                [in_s[0], in_e[0]],
                [in_s[1], in_e[1]],
                color=ink_color,
                lw=1.85,
                solid_capstyle="round",
                zorder=2,
            )
            return
        off = perp * 0.085
        ax.plot(
            [start[0] + off[0], end[0] + off[0]],
            [start[1] + off[1], end[1] + off[1]],
            color=ink_color,
            lw=1.9,
            solid_capstyle="round",
            zorder=2,
        )
        ax.plot(
            [start[0] - off[0], end[0] - off[0]],
            [start[1] - off[1], end[1] - off[1]],
            color=ink_color,
            lw=1.9,
            solid_capstyle="round",
            zorder=2,
        )
        return

    if abs(order - 3.0) < 0.25:
        off = perp * 0.105
        for mul in (-1.0, 0.0, 1.0):
            s_pt = start + mul * off
            e_pt = end + mul * off
            ax.plot(
                [s_pt[0], e_pt[0]],
                [s_pt[1], e_pt[1]],
                color=ink_color,
                lw=1.6,
                solid_capstyle="round",
                zorder=2,
            )
        return

    ax.plot(
        [start[0], end[0]],
        [start[1], end[1]],
        color=ink_color,
        lw=2.1,
        solid_capstyle="round",
        zorder=2,
    )
    if abs(order - 1.5) < 0.2:
        if ring_center is not None:
            mid = 0.5 * (p1 + p2)
            if float(np.dot(perp, ring_center - mid)) < 0.0:
                perp = -perp
        off = perp * 0.15
        ax.plot(
            [
                start[0] + off[0] + 0.12 * unit[0],
                end[0] + off[0] - 0.12 * unit[0],
            ],
            [
                start[1] + off[1] + 0.12 * unit[1],
                end[1] + off[1] - 0.12 * unit[1],
            ],
            color=ink_color,
            lw=1.5,
            linestyle="--",
            zorder=2,
        )


def _compute_visibility(
    atoms: structure.AtomsData,
    neighbors: dict[int, list[tuple[int, float]]],
    show_hydrogens: str,
) -> tuple[list[bool], dict[int, int]]:
    """Determines atom visibility flags and heteroatom attached-H counts."""
    visible = [True] * len(atoms)
    for idx, sym in enumerate(atoms.symbols):
        if sym != "H":
            continue
        if show_hydrogens == "none":
            visible[idx] = False
        elif show_hydrogens == "polar":
            nbrs = neighbors[idx]
            if nbrs and all(atoms.symbols[nb] == "C" for nb, _ in nbrs):
                visible[idx] = False
    hetero_h_count = {
        idx: sum(
            1
            for nb, _ in neighbors[idx]
            if atoms.symbols[nb] == "H" and not visible[nb]
        )
        for idx, sym in enumerate(atoms.symbols)
        if sym not in ("C", "H")
    }
    return visible, hetero_h_count


def _label_color(sym: str, palette: str | Mapping[str, str], ink: str) -> str:
    """Resolves high-contrast heteroatom text color against white paper."""
    if sym == "C":
        return ink
    col = palettes.get_element_color(sym, scheme=palette)
    hex_c = col.lstrip("#")
    if len(hex_c) == 6:
        r = int(hex_c[0:2], 16) / 255.0
        g = int(hex_c[2:4], 16) / 255.0
        b = int(hex_c[4:6], 16) / 255.0
        if (0.2126 * r + 0.7152 * g + 0.0722 * b) > 0.70:
            return ink
    return col


def _draw_skeletal_labels(
    ax: maxes.Axes,
    atoms: structure.AtomsData,
    xy: np.ndarray,
    neighbors: dict[int, list[tuple[int, float]]],
    visible: list[bool],
    hetero_h_count: dict[int, int],
    palette: str | Mapping[str, str],
    ink: str,
    bg: str,
) -> None:
    """Draws halo-backed heteroatom and terminal carbon labels."""
    for idx, sym in enumerate(atoms.symbols):
        if not visible[idx]:
            continue
        if sym == "C" and len(neighbors[idx]) >= 2:
            continue
        label = sym
        h_cnt = hetero_h_count.get(idx, 0)
        if h_cnt == 1:
            label = f"{sym}H"
        elif h_cnt > 1:
            label = f"{sym}H{h_cnt}"
        if sym == "C":
            label = "CH3" if len(neighbors[idx]) == 1 else "C"
        text_col = _label_color(sym, palette, ink)
        ax.add_patch(
            patches.Circle(
                (float(xy[idx, 0]), float(xy[idx, 1])),
                radius=0.17,
                facecolor=bg,
                edgecolor="none",
                zorder=3,
            )
        )
        ax.text(
            float(xy[idx, 0]),
            float(xy[idx, 1]),
            label,
            ha="center",
            va="center",
            fontsize=11.5,
            color=text_col,
            zorder=4,
        )


def render_structure_2d(
    atoms: structure.AtomsData,
    ax: maxes.Axes | None = None,
    show_hydrogens: str = "polar",
    palette: str | Mapping[str, str] = "xkcd",
    bond_scale: float = 1.20,
    pad: float = 0.75,
) -> maxes.Axes:
    """Draws a clean 2D skeletal chemical diagram via PCA plane projection.

    Args:
      atoms: Input molecular ``AtomsData``.
      ax: Optional Matplotlib ``Axes``. Created if ``None``.
      show_hydrogens: Hydrogen visibility mode: ``"polar"`` (hide C-H),
        ``"all"``, or ``"none"``.
      palette: Element coloring scheme for heteroatom labels.
      bond_scale: Covalent-radius scaling used if bonds are inferred.
      pad: Axis padding in Angstrom.

    Returns:
      Matplotlib ``Axes`` with the 2D skeletal structure.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5.0, 4.2))
    if len(atoms) == 0:
        ax.set_axis_off()
        return ax

    xy = _pca_2d_projection(atoms.positions)
    bonds = list(atoms.bonds) or bonding.infer_bonds(
        atoms, scale=bond_scale, use_pbc=False
    )
    neighbors: dict[int, list[tuple[int, float]]] = {
        i: [] for i in range(len(atoms))
    }
    for b in bonds:
        neighbors[b.i].append((b.j, b.order))
        neighbors[b.j].append((b.i, b.order))
    ring_centroids = _find_ring_centroids(bonds, neighbors, xy)
    visible, hetero_h_count = _compute_visibility(
        atoms, neighbors, show_hydrogens
    )

    ink = str(plt.rcParams.get("axes.edgecolor", palettes.INK))
    bg = str(plt.rcParams.get("axes.facecolor", palettes.PAPER))
    for b in bonds:
        if not (visible[b.i] and visible[b.j]):
            continue
        s_i = 0.17 if atoms.symbols[b.i] != "C" else 0.0
        s_j = 0.17 if atoms.symbols[b.j] != "C" else 0.0
        edge_key = (min(b.i, b.j), max(b.i, b.j))
        _draw_skeletal_bond(
            ax,
            xy[b.i],
            xy[b.j],
            b.order,
            ink_color=ink,
            shorten_i=s_i,
            shorten_j=s_j,
            ring_center=ring_centroids.get(edge_key),
        )

    _draw_skeletal_labels(
        ax, atoms, xy, neighbors, visible, hetero_h_count, palette, ink, bg
    )

    vis_xy = xy[np.array(visible, dtype=bool)]
    if vis_xy.size == 0:
        vis_xy = xy
    ax.set_xlim(
        float(np.min(vis_xy[:, 0])) - pad, float(np.max(vis_xy[:, 0])) + pad
    )
    ax.set_ylim(
        float(np.min(vis_xy[:, 1])) - pad, float(np.max(vis_xy[:, 1])) + pad
    )
    ax.set_aspect("equal")
    ax.set_axis_off()
    return ax
