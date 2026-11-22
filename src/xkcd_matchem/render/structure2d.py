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
import math

from matplotlib import axes as maxes
from matplotlib import patches
from matplotlib import pyplot as plt
import numpy as np

from xkcd_matchem.core import bonding
from xkcd_matchem.core import structure
from xkcd_matchem.render import primitives
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
    return np.asarray(proj, dtype=np.float64)


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


def _draw_stereo_bond(
    ax: maxes.Axes,
    start: np.ndarray,
    end: np.ndarray,
    perp: np.ndarray,
    stereo: str,
    ink_color: str,
    lw_scale: float = 1.0,
) -> None:
    """Draws wedge, dashed-wedge, or wavy stereo 2D bonds."""
    if stereo == "wedge":
        half_w = 0.135 * (0.82 + 0.18 * lw_scale)
        tri = np.vstack([start, end + half_w * perp, end - half_w * perp])
        ax.add_patch(
            patches.Polygon(
                tri,
                closed=True,
                facecolor=ink_color,
                edgecolor=ink_color,
                linewidth=0.95 * lw_scale,
                joinstyle="round",
                zorder=2.5,
            )
        )
    elif stereo == "dash":
        n_ticks = 6
        for step_k in range(1, n_ticks + 1):
            frac = step_k / float(n_ticks + 1)
            pt = start + frac * (end - start)
            hw = (0.04 + 0.105 * frac) * (0.85 + 0.15 * lw_scale)
            ax.plot(
                [pt[0] - hw * perp[0], pt[0] + hw * perp[0]],
                [pt[1] - hw * perp[1], pt[1] + hw * perp[1]],
                color=ink_color,
                lw=1.65 * lw_scale,
                solid_capstyle="round",
                zorder=2.5,
            )
    elif stereo == "wavy":
        ts = np.linspace(0.0, 1.0, 29)
        pts = (
            start[None, :]
            + ts[:, None] * (end - start)[None, :]
            + (0.09 * np.sin(ts * 4.0 * np.pi))[:, None] * perp[None, :]
        )
        ax.plot(
            pts[:, 0],
            pts[:, 1],
            color=ink_color,
            lw=1.85 * lw_scale,
            solid_capstyle="round",
            zorder=2.5,
        )


def _draw_skeletal_bond(
    ax: maxes.Axes,
    p1: np.ndarray,
    p2: np.ndarray,
    order: float,
    ink_color: str,
    shorten_i: float = 0.0,
    shorten_j: float = 0.0,
    ring_center: np.ndarray | None = None,
    stereo: str = "none",
    lw_scale: float = 1.0,
) -> None:
    """Draws single, double, triple, aromatic, or stereo 2D bonds."""
    vec = p2 - p1
    length = float(np.linalg.norm(vec))
    if length < 1e-6:
        return
    if shorten_i + shorten_j >= 0.88 * length and shorten_i + shorten_j > 0.0:
        fit = (0.88 * length) / (shorten_i + shorten_j)
        shorten_i *= fit
        shorten_j *= fit
    unit = vec / length
    perp = np.array([-unit[1], unit[0]], dtype=np.float64)
    start = p1 + unit * shorten_i
    end = p2 - unit * shorten_j
    gap_scale = 0.85 + 0.15 * lw_scale

    if stereo in ("wedge", "dash", "wavy"):
        _draw_stereo_bond(
            ax, start, end, perp, stereo, ink_color, lw_scale=lw_scale
        )
        return

    if abs(order - 2.0) < 0.25:
        if ring_center is not None:
            mid = 0.5 * (p1 + p2)
            if float(np.dot(perp, ring_center - mid)) < 0.0:
                perp = -perp
            ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                color=ink_color,
                lw=2.05 * lw_scale,
                solid_capstyle="round",
                zorder=2,
            )
            inner_off = perp * (0.165 * gap_scale)
            trim = 0.14 * (end - start)
            in_s = start + trim + inner_off
            in_e = end - trim + inner_off
            ax.plot(
                [in_s[0], in_e[0]],
                [in_s[1], in_e[1]],
                color=ink_color,
                lw=1.75 * lw_scale,
                solid_capstyle="round",
                zorder=2,
            )
            return
        off = perp * (0.088 * gap_scale)
        ax.plot(
            [start[0] + off[0], end[0] + off[0]],
            [start[1] + off[1], end[1] + off[1]],
            color=ink_color,
            lw=1.80 * lw_scale,
            solid_capstyle="round",
            zorder=2,
        )
        ax.plot(
            [start[0] - off[0], end[0] - off[0]],
            [start[1] - off[1], end[1] - off[1]],
            color=ink_color,
            lw=1.80 * lw_scale,
            solid_capstyle="round",
            zorder=2,
        )
        return

    if abs(order - 3.0) < 0.25:
        off = perp * (0.105 * gap_scale)
        for mul in (-1.0, 0.0, 1.0):
            s_pt = start + mul * off
            e_pt = end + mul * off
            ax.plot(
                [s_pt[0], e_pt[0]],
                [s_pt[1], e_pt[1]],
                color=ink_color,
                lw=1.55 * lw_scale,
                solid_capstyle="round",
                zorder=2,
            )
        return

    ax.plot(
        [start[0], end[0]],
        [start[1], end[1]],
        color=ink_color,
        lw=2.05 * lw_scale,
        solid_capstyle="round",
        zorder=2,
    )
    if abs(order - 1.5) < 0.2:
        if ring_center is not None:
            mid = 0.5 * (p1 + p2)
            if float(np.dot(perp, ring_center - mid)) < 0.0:
                perp = -perp
        off = perp * (0.15 * gap_scale)
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
            lw=1.45 * lw_scale,
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


def _label_color(
    sym: str,
    palette: str | Mapping[str, str],
    ink: str,
    bg: str = palettes.PAPER,
) -> str:
    """Resolves high-contrast heteroatom text color against ``bg``."""
    if sym == "C":
        return ink
    col = palettes.get_element_color(sym, scheme=palette)
    return primitives.ensure_readable_text_color(col, bg)


def _atom_has_label(
    idx: int,
    atoms: structure.AtomsData,
    neighbors: dict[int, list[tuple[int, float]]],
    visible: list[bool],
) -> bool:
    """Returns whether an atom renders a text label on the 2D diagram."""
    if not visible[idx]:
        return False
    if atoms.symbols[idx] == "C":
        vis_deg = sum(1 for nb, _ in neighbors[idx] if visible[nb])
        return vis_deg < 2
    return True


def _terminal_carbon_label(
    idx: int,
    atoms: structure.AtomsData,
    neighbors: dict[int, list[tuple[int, float]]],
    visible: list[bool],
) -> str:
    """Computes implicit-H label (CH3/CH2/CH/C) for terminal carbons."""
    hidden_h = sum(
        1
        for nb, _ in neighbors[idx]
        if atoms.symbols[nb] == "H" and not visible[nb]
    )
    if hidden_h > 0:
        h_cnt = hidden_h
    else:
        vis_order = sum(o for nb, o in neighbors[idx] if visible[nb])
        h_cnt = max(0, 4 - round(vis_order)) if vis_order > 0.0 else 0
    if h_cnt == 0:
        return "C"
    if h_cnt == 1:
        return "CH"
    return f"CH{h_cnt}"


def resolve_span_label_metrics(
    ax: maxes.Axes,
    span_x_raw: float,
    span_y_raw: float,
    pad: float,
) -> tuple[float, float, float, float, float]:
    """Resolves ``(fontsize, char_w, char_h, margin, pt_per_ang)``."""
    fig = ax.get_figure()
    dpi = float(fig.dpi) if fig is not None else 100.0
    bbox = ax.get_window_extent()
    ax_w_pt = max(float(bbox.width) * 72.0 / max(dpi, 1.0), 45.0)
    ax_h_pt = max(float(bbox.height) * 72.0 / max(dpi, 1.0), 45.0)
    span_x = max(float(span_x_raw) + 2.0 * pad, 2.2)
    span_y = max(float(span_y_raw) + 2.0 * pad, 2.2)
    pt_per_ang = min(ax_w_pt / span_x, ax_h_pt / span_y)
    fontsize = float(np.clip(0.54 * pt_per_ang, 7.6, 11.5))
    ang_per_pt = 1.0 / max(pt_per_ang, 1.0)
    char_w = 0.66 * fontsize * ang_per_pt
    char_h = 0.80 * fontsize * ang_per_pt
    margin = max(0.13, 0.24 * fontsize * ang_per_pt)
    return fontsize, char_w, char_h, margin, pt_per_ang


def _resolve_2d_label_metrics(
    ax: maxes.Axes,
    vis_xy: np.ndarray,
    pad: float,
) -> tuple[float, float, float, float]:
    """Resolves font size and Angstrom-space char box from axes extent."""
    fs, cw, ch, margin, _ = resolve_span_label_metrics(
        ax,
        float(np.ptp(vis_xy[:, 0])),
        float(np.ptp(vis_xy[:, 1])),
        pad,
    )
    return fs, cw, ch, margin


def project_molecule_2d(
    atoms: structure.AtomsData,
    show_hydrogens: str = "polar",
    bond_scale: float = 1.20,
) -> tuple[
    np.ndarray,
    list[structure.Bond],
    dict[int, list[tuple[int, float]]],
    list[bool],
    dict[int, int],
    tuple[float, float],
]:
    """Projects a molecule into normalized Angstrom 2D coordinates."""
    if len(atoms) == 0:
        return (
            np.zeros((0, 2), dtype=np.float64),
            [],
            {},
            [],
            {},
            (0.0, 0.0),
        )
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
    visible, hetero_h_count = _compute_visibility(
        atoms, neighbors, show_hydrogens
    )
    vis_xy = xy[np.array(visible, dtype=bool)]
    if vis_xy.size == 0:
        vis_xy = xy
    xy[:, 0] -= float(np.min(vis_xy[:, 0]))
    xy[:, 1] -= 0.5 * float(np.min(vis_xy[:, 1]) + np.max(vis_xy[:, 1]))
    vis_shifted = xy[np.array(visible, dtype=bool)]
    if vis_shifted.size == 0:
        vis_shifted = xy
    span_w = float(np.ptp(vis_shifted[:, 0]))
    span_h = float(np.ptp(vis_shifted[:, 1]))
    return xy, bonds, neighbors, visible, hetero_h_count, (span_w, span_h)


def _projected_molecule_bounds(
    xy: np.ndarray,
    visible: list[bool],
    label_specs: dict[int, tuple[str, np.ndarray, float, float]],
) -> tuple[float, float, float, float]:
    """Computes tight ``(xmin, xmax, ymin, ymax)`` covering bonds & labels."""
    vis_xy = xy[np.array(visible, dtype=bool)]
    if vis_xy.size == 0:
        vis_xy = xy
    xs_all = [vis_xy[:, 0]]
    ys_all = [vis_xy[:, 1]]
    for _, center, hw, hh in label_specs.values():
        xs_all.append(np.array([center[0] - hw, center[0] + hw]))
        ys_all.append(np.array([center[1] - hh, center[1] + hh]))
    x_comb = np.concatenate(xs_all)
    y_comb = np.concatenate(ys_all)
    return (
        float(np.min(x_comb)),
        float(np.max(x_comb)),
        float(np.min(y_comb)),
        float(np.max(y_comb)),
    )


def draw_projected_molecule_2d(
    ax: maxes.Axes,
    atoms: structure.AtomsData,
    proj_data: tuple[
        np.ndarray,
        list[structure.Bond],
        dict[int, list[tuple[int, float]]],
        list[bool],
        dict[int, int],
        tuple[float, float],
    ],
    *,
    offset: tuple[float, float] = (0.0, 0.0),
    metrics: tuple[float, float, float, float],
    lw_scale: float = 1.0,
    palette: str | Mapping[str, str] = "xkcd",
) -> tuple[float, float, float, float]:
    """Draws a pre-projected molecule at ``offset`` using shared metrics."""
    base_xy, bonds, neighbors, visible, hetero_h_count, _ = proj_data
    if len(atoms) == 0:
        return (offset[0], offset[0], offset[1], offset[1])
    xy = base_xy.copy()
    xy[:, 0] += float(offset[0])
    xy[:, 1] += float(offset[1])
    ring_centroids = _find_ring_centroids(bonds, neighbors, xy)
    label_specs = _build_label_specs(
        atoms, xy, neighbors, visible, hetero_h_count, metrics
    )
    ink = str(plt.rcParams.get("axes.edgecolor", palettes.INK))
    bg = str(plt.rcParams.get("axes.facecolor", palettes.PAPER))
    for b in bonds:
        if not (visible[b.i] and visible[b.j]):
            continue
        _draw_skeletal_bond(
            ax,
            xy[b.i],
            xy[b.j],
            b.order,
            ink_color=ink,
            shorten_i=_trim_bond_to_label(
                xy[b.i], xy[b.j], label_specs.get(b.i)
            ),
            shorten_j=_trim_bond_to_label(
                xy[b.j], xy[b.i], label_specs.get(b.j)
            ),
            ring_center=ring_centroids.get((min(b.i, b.j), max(b.i, b.j))),
            stereo=b.stereo,
            lw_scale=lw_scale,
        )
    _draw_skeletal_labels(
        ax, atoms, label_specs, metrics[0], palette, ink, bg
    )
    return _projected_molecule_bounds(xy, visible, label_specs)


def _build_label_specs(
    atoms: structure.AtomsData,
    xy: np.ndarray,
    neighbors: dict[int, list[tuple[int, float]]],
    visible: list[bool],
    hetero_h_count: dict[int, int],
    metrics: tuple[float, float, float, float],
) -> dict[int, tuple[str, np.ndarray, float, float]]:
    """Builds outward-shifted text center and halo half-axes per atom."""
    _, char_w, char_h, margin = metrics
    specs: dict[int, tuple[str, np.ndarray, float, float]] = {}
    for idx, sym in enumerate(atoms.symbols):
        if not _atom_has_label(idx, atoms, neighbors, visible):
            continue
        label = sym
        h_cnt = hetero_h_count.get(idx, 0)
        if h_cnt == 1:
            label = f"{sym}H"
        elif h_cnt > 1:
            label = f"{sym}H{h_cnt}"
        if sym == "C":
            label = _terminal_carbon_label(idx, atoms, neighbors, visible)

        center = xy[idx].copy()
        if len(label) > 1:
            vis_nbs = [nb for nb, _ in neighbors[idx] if visible[nb]]
            if len(vis_nbs) == 1:
                d_vec = xy[idx] - xy[vis_nbs[0]]
                d_norm = float(np.linalg.norm(d_vec))
                if d_norm > 1e-6 and abs(float(d_vec[0]) / d_norm) >= 0.22:
                    center[0] += math.copysign(
                        0.5 * float(len(label) - 1) * char_w, float(d_vec[0])
                    )
        hw = 0.5 * float(len(label)) * char_w + margin
        hh = 0.5 * char_h + margin
        specs[idx] = (label, center, hw, hh)
    return specs


def _trim_bond_to_label(
    p_atom: np.ndarray,
    p_other: np.ndarray,
    spec: tuple[str, np.ndarray, float, float] | None,
) -> float:
    """Computes ray-ellipse exit distance so skeletal bonds stop cleanly."""
    if spec is None:
        return 0.0
    bond_vec = p_other - p_atom
    bond_len = float(np.linalg.norm(bond_vec))
    if bond_len < 1e-6:
        return 0.0
    unit = bond_vec / bond_len
    _, center, hw, hh = spec
    dx = float(p_atom[0] - center[0])
    dy = float(p_atom[1] - center[1])
    a_coef = (float(unit[0]) / hw) ** 2 + (float(unit[1]) / hh) ** 2
    b_coef = (dx * float(unit[0])) / (hw**2) + (dy * float(unit[1])) / (hh**2)
    c_coef = (dx / hw) ** 2 + (dy / hh) ** 2 - 1.0
    if c_coef >= 0.0 or a_coef <= 1e-12:
        t_exit = 0.0
    else:
        disc = max(b_coef * b_coef - a_coef * c_coef, 0.0)
        t_exit = (-b_coef + math.sqrt(disc)) / a_coef
    return float(np.clip(t_exit, 0.16, 0.44 * bond_len))


def _draw_skeletal_labels(
    ax: maxes.Axes,
    atoms: structure.AtomsData,
    label_specs: dict[int, tuple[str, np.ndarray, float, float]],
    fontsize: float,
    palette: str | Mapping[str, str],
    ink: str,
    bg: str,
) -> None:
    """Draws un-sketched elliptical paper halos and readable atom labels."""
    for idx, (label, center, hw, hh) in label_specs.items():
        text_col = _label_color(atoms.symbols[idx], palette, ink, bg)
        halo = patches.Ellipse(
            (float(center[0]), float(center[1])),
            width=2.0 * hw,
            height=2.0 * hh,
            facecolor=bg,
            edgecolor="none",
            zorder=3,
        )
        halo.set_sketch_params(0.0, 0.0, 0.0)
        ax.add_patch(halo)
        ax.text(
            float(center[0]),
            float(center[1]),
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
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
    vis_xy = xy[np.array(visible, dtype=bool)]
    if vis_xy.size == 0:
        vis_xy = xy
    metrics = _resolve_2d_label_metrics(ax, vis_xy, pad)
    label_specs = _build_label_specs(
        atoms, xy, neighbors, visible, hetero_h_count, metrics
    )

    ink = str(plt.rcParams.get("axes.edgecolor", palettes.INK))
    bg = str(plt.rcParams.get("axes.facecolor", palettes.PAPER))
    for b in bonds:
        if not (visible[b.i] and visible[b.j]):
            continue
        s_i = _trim_bond_to_label(xy[b.i], xy[b.j], label_specs.get(b.i))
        s_j = _trim_bond_to_label(xy[b.j], xy[b.i], label_specs.get(b.j))
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
            stereo=b.stereo,
        )

    _draw_skeletal_labels(
        ax, atoms, label_specs, metrics[0], palette, ink, bg
    )

    xs_all = [vis_xy[:, 0]]
    ys_all = [vis_xy[:, 1]]
    for _, center, hw, hh in label_specs.values():
        xs_all.append(np.array([center[0] - hw, center[0] + hw]))
        ys_all.append(np.array([center[1] - hh, center[1] + hh]))
    x_comb = np.concatenate(xs_all)
    y_comb = np.concatenate(ys_all)
    ax.set_xlim(float(np.min(x_comb)) - pad, float(np.max(x_comb)) + pad)
    ax.set_ylim(float(np.min(y_comb)) - pad, float(np.max(y_comb)) + pad)
    ax.set_aspect("equal")
    ax.set_axis_off()
    return ax
