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
"""xkcd-style graph networks and crystal/molecular GNN graph visualizations."""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
import math
import pathlib

from matplotlib import axes as maxes
from matplotlib import colors as mcolors
from matplotlib import lines as mlines
from matplotlib import patches
from matplotlib import pyplot as plt
import numpy as np

from xkcd_matchem.core import bonding
from xkcd_matchem.core import exceptions
from xkcd_matchem.core import geometry
from xkcd_matchem.core import structure
from xkcd_matchem.data import elements
from xkcd_matchem.io import loader
from xkcd_matchem.render import primitives
from xkcd_matchem.style import palettes
from xkcd_matchem.style import patterns
from xkcd_matchem.style import theme


def _spring_layout(
    n_nodes: int,
    edges: Sequence[tuple[int, int]],
    seed: int = 7,
    iterations: int = 80,
) -> np.ndarray:
    """Computes a deterministic Fruchterman-Reingold force-directed layout."""
    if n_nodes == 0:
        return np.zeros((0, 2), dtype=np.float64)
    if n_nodes == 1:
        return np.zeros((1, 2), dtype=np.float64)
    angles = np.linspace(0.0, 2.0 * np.pi, n_nodes, endpoint=False)
    rng = np.random.default_rng(seed)
    pos = np.column_stack([np.cos(angles), np.sin(angles)]) + rng.normal(
        0.0, 0.04, size=(n_nodes, 2)
    )
    k_opt = math.sqrt(4.0 / max(n_nodes, 1))
    temp = 0.45
    for _ in range(iterations):
        disp = np.zeros_like(pos)
        delta = pos[:, None, :] - pos[None, :, :]
        dist = np.linalg.norm(delta, axis=2)
        np.fill_diagonal(dist, 1.0)
        rep = (k_opt * k_opt) / np.maximum(dist, 0.08)
        np.fill_diagonal(rep, 0.0)
        disp += np.sum(delta / dist[:, :, None] * rep[:, :, None], axis=1)
        for u, v in edges:
            if u == v or u >= n_nodes or v >= n_nodes:
                continue
            diff = pos[u] - pos[v]
            d = max(float(np.linalg.norm(diff)), 0.05)
            force = (d * d) / k_opt
            step = (diff / d) * force
            disp[u] -= step
            disp[v] += step
        norms = np.maximum(np.linalg.norm(disp, axis=1, keepdims=True), 1e-6)
        pos += (disp / norms) * np.minimum(norms, temp)
        pos -= np.mean(pos, axis=0)
        temp *= 0.96
    span = max(float(np.max(np.abs(pos))), 1e-6)
    return pos / span


def _bipartite_concentric_pos(
    n_nodes: int,
    adj: Sequence[set[int]],
) -> np.ndarray | None:
    """Places balanced 2-colorable 8-node graphs on two concentric squares."""
    if n_nodes != 8 or any(not nbrs for nbrs in adj):
        return None
    color_map: dict[int, int] = {0: 0}
    queue = [0]
    while queue:
        cur = queue.pop(0)
        for nb in sorted(adj[cur]):
            if nb not in color_map:
                color_map[nb] = 1 - color_map[cur]
                queue.append(nb)
            elif color_map[nb] == color_map[cur]:
                return None
    g0 = [i for i in range(n_nodes) if color_map.get(i, 0) == 0]
    g1 = [i for i in range(n_nodes) if color_map.get(i, 0) == 1]
    if len(g0) != 4 or len(g1) != 4:
        return None
    pos_out = np.zeros((n_nodes, 2), dtype=np.float64)
    a0 = np.linspace(np.pi / 4, 2.0 * np.pi + np.pi / 4, 4, endpoint=False)
    a1 = np.linspace(0.0, 2.0 * np.pi, 4, endpoint=False)
    for k, idx in enumerate(g0):
        pos_out[idx] = [1.05 * math.cos(a0[k]), 1.05 * math.sin(a0[k])]
    for k, idx in enumerate(g1):
        pos_out[idx] = [0.46 * math.cos(a1[k]), 0.46 * math.sin(a1[k])]
    return pos_out


def _planar_layout(
    n_nodes: int,
    edges: Sequence[tuple[int, int]],
    seed: int = 7,
) -> np.ndarray:
    """Computes a flat 2D planar-friendly layout with zero node overlap."""
    if n_nodes <= 1:
        return np.zeros((n_nodes, 2), dtype=np.float64)
    if n_nodes == 2:
        return np.array([[-1.0, 0.0], [1.0, 0.0]], dtype=np.float64)

    adj: list[set[int]] = [set() for _ in range(n_nodes)]
    for u, v in edges:
        if 0 <= u < n_nodes and 0 <= v < n_nodes and u != v:
            adj[u].add(v)
            adj[v].add(u)

    bip_pos = _bipartite_concentric_pos(n_nodes, adj)
    if bip_pos is not None:
        return bip_pos

    order: list[int] = []
    visited: set[int] = set()
    for start in range(n_nodes):
        if start in visited:
            continue
        queue = [start]
        visited.add(start)
        while queue:
            curr = queue.pop(0)
            order.append(curr)
            for nxt in sorted(adj[curr]):
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)

    if n_nodes <= 10:
        angles = np.linspace(0.0, 2.0 * np.pi, n_nodes, endpoint=False)
        pos = np.zeros((n_nodes, 2), dtype=np.float64)
        for rank, node_idx in enumerate(order):
            pos[node_idx] = [math.cos(angles[rank]), math.sin(angles[rank])]
        return pos

    return _spring_layout(n_nodes, edges, seed=seed, iterations=120)


def _compute_graph_positions(
    n_nodes: int,
    edges: Sequence[tuple[int, int]],
    layout: str,
    pos: np.ndarray | Sequence[tuple[float, float]] | None,
    seed: int,
) -> np.ndarray:
    """Computes 2D coordinates for a network graph."""
    if pos is not None:
        arr = np.asarray(pos, dtype=np.float64)
        if arr.shape != (n_nodes, 2):
            raise exceptions.StructureError(
                f"Expected pos shape ({n_nodes}, 2), got {arr.shape}."
            )
        return arr
    key = layout.strip().lower()
    if key == "planar":
        return _planar_layout(n_nodes, edges, seed=seed)
    if key == "circular":
        angles = np.linspace(0.0, 2.0 * np.pi, max(n_nodes, 1), endpoint=False)
        return np.column_stack([np.cos(angles), np.sin(angles)])[:n_nodes]
    if key == "grid":
        ncols = max(1, math.ceil(math.sqrt(max(n_nodes, 1))))
        pts = [
            (float(idx % ncols), -float(idx // ncols)) for idx in range(n_nodes)
        ]
        arr = np.asarray(pts, dtype=np.float64)
        if arr.size:
            arr -= np.mean(arr, axis=0)
            span = max(float(np.max(np.abs(arr))), 1.0)
            arr /= span
        return arr
    return _spring_layout(n_nodes, edges, seed=seed)


def _draw_graph_edge(
    ax: maxes.Axes,
    p_u: np.ndarray,
    p_v: np.ndarray,
    *,
    directed: bool,
    dashed: bool,
    rad: float,
    color: str,
    linewidth: float,
    label: str | None,
    bg_color: str,
    shrink_pt: float = 14.0,
) -> None:
    """Draws a straight or curved xkcd graph edge with optional label."""
    ls = "--" if dashed else "-"
    if directed or abs(rad) > 1e-4:
        arrowstyle = "-|>,head_width=0.25,head_length=0.40" if directed else "-"
        ax.annotate(
            "",
            xy=(float(p_v[0]), float(p_v[1])),
            xytext=(float(p_u[0]), float(p_u[1])),
            arrowprops={
                "arrowstyle": arrowstyle,
                "connectionstyle": f"arc3,rad={rad}",
                "color": color,
                "lw": linewidth,
                "linestyle": ls,
                "shrinkA": shrink_pt,
                "shrinkB": shrink_pt,
            },
            zorder=2.0,
        )
    else:
        ax.plot(
            [float(p_u[0]), float(p_v[0])],
            [float(p_u[1]), float(p_v[1])],
            color=color,
            linewidth=linewidth,
            linestyle=ls,
            zorder=2.0,
        )
    if label:
        mid = 0.5 * (p_u + p_v)
        vec = p_v - p_u
        if abs(rad) > 1e-4:
            # Matches Matplotlib ConnectionStyle.Arc3 quadratic midpoint.
            arc_offset = 0.5 * rad * np.array([vec[1], -vec[0]], dtype=float)
            mid = mid + arc_offset
        ax.text(
            float(mid[0]),
            float(mid[1]),
            label,
            ha="center",
            va="center",
            fontsize=8.2,
            color=color,
            bbox={
                "boxstyle": "round,pad=0.16",
                "facecolor": bg_color,
                "edgecolor": "none",
                "alpha": 0.92,
            },
            zorder=4.5,
        )


def _draw_all_graph_edges(
    ax: maxes.Axes,
    n_nodes: int,
    norm_edges: Sequence[tuple[int, int, str | None]],
    xy: np.ndarray,
    dashed_flags: Sequence[bool],
    style_cfg: tuple[bool, bool, str, str, float],
) -> None:
    """Draws all network edges with symmetric multi-edge arc fanning."""
    directed, curved, ink_col, bg_color, shrink_pt = style_cfg
    pair_totals: dict[tuple[int, int], int] = {}
    for u, v, _ in norm_edges:
        if u < n_nodes and v < n_nodes:
            key = (min(u, v), max(u, v))
            pair_totals[key] = pair_totals.get(key, 0) + 1

    pair_seen: dict[tuple[int, int], int] = {}
    for idx_e, (u, v, lbl) in enumerate(norm_edges):
        if u >= n_nodes or v >= n_nodes:
            continue
        key = (min(u, v), max(u, v))
        mult = pair_seen.get(key, 0)
        pair_seen[key] = mult + 1
        total = pair_totals[key]
        is_dash = bool(dashed_flags[idx_e])
        if total > 1:
            max_rad = min(0.54, 0.17 * total)
            rads = np.linspace(-max_rad, max_rad, total)
            rad = float(rads[mult])
            if u > v:
                rad = -rad
        elif is_dash:
            rad = 0.22
        else:
            rad = 0.14 if curved else 0.0
        _draw_graph_edge(
            ax,
            xy[u],
            xy[v],
            directed=directed,
            dashed=is_dash,
            rad=rad,
            color="#0072B2" if is_dash else ink_col,
            linewidth=1.55 if is_dash else 2.0,
            label=lbl,
            bg_color=bg_color,
            shrink_pt=shrink_pt,
        )


def _resolve_node_color(
    idx: int,
    raw_lbl: str,
    node_colors: Sequence[str] | Mapping[int, str] | None,
    series: Sequence[str],
) -> str:
    """Selects explicit, element-matched, or palette-cycled fill color."""
    if isinstance(node_colors, Mapping) and idx in node_colors:
        return str(node_colors[idx])
    if isinstance(node_colors, Sequence) and not isinstance(node_colors, str):
        return str(node_colors[idx % len(node_colors)])
    if raw_lbl in elements.ELEMENTS_BY_SYMBOL:
        return palettes.get_element_color(raw_lbl, scheme="xkcd")
    return series[(idx + 1) % len(series)]


def _draw_graph_nodes(
    ax: maxes.Axes,
    nodes: Sequence[str],
    xy: np.ndarray,
    node_colors: Sequence[str] | Mapping[int, str] | None,
    series: Sequence[str],
    ink_col: str,
    node_shape: str = "circle",
) -> float:
    """Draws shaded xkcd circular or rounded-pill network nodes."""
    n_nodes = len(nodes)
    span_x = float(np.ptp(xy[:, 0])) if n_nodes > 1 else 2.0
    span_y = float(np.ptp(xy[:, 1])) if n_nodes > 1 else 2.0
    scale = max(0.5 * max(span_x, span_y), 0.8)
    base_r = max(0.085, min(0.14, 0.48 / max(math.sqrt(n_nodes), 2.8))) * scale
    use_pill = node_shape.strip().lower() == "pill"

    for idx, raw_lbl in enumerate(nodes):
        col = _resolve_node_color(idx, str(raw_lbl), node_colors, series)
        cx, cy = float(xy[idx, 0]), float(xy[idx, 1])
        if use_pill:
            fill = primitives.lighten_hex(col, 0.62)
            ax.text(
                cx,
                cy,
                str(raw_lbl),
                ha="center",
                va="center",
                fontsize=9.5,
                color=ink_col,
                bbox={
                    "boxstyle": "round,pad=0.45",
                    "facecolor": fill,
                    "edgecolor": ink_col,
                    "linewidth": 1.6,
                },
                zorder=6.0,
            )
            continue
        ax.add_patch(
            patches.Circle(
                (cx, cy),
                radius=base_r,
                facecolor=col,
                edgecolor=ink_col,
                linewidth=1.7,
                zorder=5.0,
            )
        )
        ax.add_patch(
            patches.Circle(
                (cx - 0.28 * base_r, cy + 0.28 * base_r),
                radius=0.18 * base_r,
                facecolor="#FFFFFF",
                edgecolor="none",
                alpha=0.72,
                zorder=5.2,
            )
        )
        txt_col = primitives.contrast_text_color(mcolors.to_hex(col))
        ax.text(
            cx,
            cy,
            str(raw_lbl),
            ha="center",
            va="center",
            fontsize=max(7.8, min(9.8, 9.2)),
            color=txt_col,
            zorder=6.0,
        )
    return base_r


def _set_graph_limits(
    ax: maxes.Axes,
    xy: np.ndarray,
    base_r: float,
    node_shape: str,
) -> None:
    """Applies balanced x/y viewport limits around graph nodes and arcs."""
    if xy.shape[0] == 0:
        return
    pad_x = (
        base_r * 3.4 + 0.35
        if node_shape.strip().lower() == "pill"
        else base_r * 2.2 + 0.25
    )
    pad_y = base_r * 2.8 + 0.30
    x_lo = float(np.min(xy[:, 0])) - pad_x
    x_hi = float(np.max(xy[:, 0])) + pad_x
    y_lo = float(np.min(xy[:, 1])) - pad_y
    y_hi = float(np.max(xy[:, 1])) + pad_y
    y_mid = 0.5 * (y_lo + y_hi)
    min_y_half = 0.44 * (x_hi - x_lo)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(min(y_lo, y_mid - min_y_half), max(y_hi, y_mid + min_y_half))


def plot_graph_network(
    nodes: Sequence[str],
    edges: Sequence[tuple[int, int] | tuple[int, int, str]],
    *,
    pos: np.ndarray | Sequence[tuple[float, float]] | None = None,
    layout: str = "spring",
    node_shape: str = "circle",
    node_colors: Sequence[str] | Mapping[int, str] | None = None,
    dashed_edges: Sequence[tuple[int, int]] | Sequence[bool] | None = None,
    directed: bool = False,
    background_grid: str | None = None,
    palette: str | Sequence[str] = "okabe_ito",
    preset: str | theme.ThemePreset = "classic",
    ax: maxes.Axes | None = None,
    title: str | None = None,
) -> maxes.Axes:
    """Plots a clean xkcd-style graph network for materials/chem informatics.

    Use ``node_shape="circle"`` for atom/element symbols or ``"pill"`` for
    synthesis pathways, polymorph maps, and chemical-space networks.

    Args:
      nodes: Sequence of node labels (e.g. ``["Anatase TiO2", "Rutile TiO2"]``
        or ``["Ti0", "O1", ...]``).
      edges: Sequence of ``(u, v)`` or ``(u, v, label)`` index tuples.
      pos: Optional explicit ``(N, 2)`` coordinates.
      layout: Layout mode when ``pos`` is ``None``: ``"spring"``,
        ``"planar"``, ``"circular"``, or ``"grid"``.
      node_shape: ``"circle"`` (shaded spheres) or ``"pill"`` (rounded boxes).
      node_colors: Optional per-node hex color list or ``{idx: color}``.
      dashed_edges: Optional edge pairs or per-edge boolean flags rendered as
        dashed arcs.
      directed: Whether to draw directional arrowheads along edges.
      background_grid: Optional background grid (default ``None`` for clean bg).
      palette: Series palette name or sequence of colors.
      preset: Visual theme preset name or ``ThemePreset``.
      ax: Optional target Matplotlib ``Axes``.
      title: Optional heading above the graph.

    Returns:
      Matplotlib ``Axes`` containing the graph network.
    """
    n_nodes = len(nodes)
    norm_edges = [
        (int(e[0]), int(e[1]), str(e[2]) if len(e) >= 3 else None)
        for e in edges
    ]
    xy = _compute_graph_positions(
        n_nodes, [(u, v) for u, v, _ in norm_edges], layout, pos, seed=7
    )
    series = palettes.get_palette(palette)
    preset_obj = theme.get_preset(preset)
    ink_col = preset_obj.ink_color
    if (
        dashed_edges is not None
        and len(dashed_edges) == len(norm_edges)
        and all(isinstance(v, bool) for v in dashed_edges)
    ):
        dashed_flags = [bool(v) for v in dashed_edges]
    else:
        dashed_set = {
            (min(int(a), int(b)), max(int(a), int(b)))
            for a, b in (dashed_edges or ())  # type: ignore[misc]
        }
        dashed_flags = [
            (min(u, v), max(u, v)) in dashed_set for u, v, _ in norm_edges
        ]
    shrink_pt = 26.0 if node_shape.strip().lower() == "pill" else 14.0

    with theme.xkcd_style(preset=preset):
        if ax is None:
            _, ax = plt.subplots(figsize=(6.4, 5.0))
        else:
            theme.style_existing_axes(ax, preset=preset)
        if background_grid:
            patterns.apply_background_grid(ax, style=background_grid)
        _draw_all_graph_edges(
            ax,
            n_nodes,
            norm_edges,
            xy,
            dashed_flags,
            (directed, False, ink_col, preset_obj.axes_facecolor, shrink_pt),
        )
        base_r = _draw_graph_nodes(
            ax, nodes, xy, node_colors, series, ink_col, node_shape=node_shape
        )
        _set_graph_limits(ax, xy, base_r, node_shape)
        ax.set_aspect("equal")
        ax.set_axis_off()
        if title:
            ax.set_title(title, color=ink_col)
    return ax


def _format_shift_tag(shift: tuple[int, int, int]) -> str:
    """Formats a periodic lattice shift ``(sx, sy, sz)`` compactly."""
    if all(0 <= v <= 9 for v in shift):
        return f"{shift[0]}{shift[1]}{shift[2]}"
    return f"({shift[0]},{shift[1]},{shift[2]})"


def _extract_structure_edges(
    atoms: structure.AtomsData,
    show_distances: bool,
    show_periodic_edges: bool,
    multi_edges: bool = False,
    show_shift_labels: bool = False,
) -> tuple[list[tuple[int, int, str]], list[bool]]:
    """Builds solid bond and dashed periodic-image edges from ``AtomsData``."""
    bonds = list(atoms.bonds) or bonding.infer_bonds(atoms)
    edge_items: list[tuple[int, int, str]] = []
    dashed_flags: list[bool] = []
    seen_pbc: set[tuple[int, int]] = set()
    for b in bonds:
        is_pbc = b.shift != (0, 0, 0)
        if is_pbc and not show_periodic_edges:
            continue
        if is_pbc and atoms.cell is not None:
            pair = (min(b.i, b.j), max(b.i, b.j))
            if not multi_edges:
                if pair in seen_pbc:
                    continue
                seen_pbc.add(pair)
            lbl = _format_shift_tag(b.shift) if show_shift_labels else ""
            dashed_flags.append(True)
            edge_items.append((b.i, b.j, lbl))
            continue
        lbl = ""
        if show_shift_labels:
            lbl = _format_shift_tag((0, 0, 0))
        elif show_distances:
            dist = float(
                np.linalg.norm(atoms.positions[b.j] - atoms.positions[b.i])
            )
            order_tag = "=" if abs(b.order - 2.0) < 0.25 else ""
            lbl = f"{order_tag}{dist:.2f}A"
        dashed_flags.append(False)
        edge_items.append((b.i, b.j, lbl))
    return edge_items, dashed_flags


def _add_structure_graph_overlays(
    ax: maxes.Axes,
    atoms: structure.AtomsData,
    cam: geometry.Camera,
    pivot: np.ndarray,
    *,
    show_cell: bool,
    layout: str,
    has_dashed: bool,
    preset: str | theme.ThemePreset,
) -> None:
    """Adds optional unit-cell guide and periodic-edge legend to a graph."""
    ink_col = theme.get_preset(preset).ink_color
    if (
        show_cell
        and atoms.cell is not None
        and layout.strip().lower() == "camera"
    ):
        with theme.xkcd_style(preset=preset):
            corners = primitives.draw_unit_cell_box(
                ax,
                atoms.cell,
                cam,
                center=pivot,
                edgecolor="#94A3B8",
                linewidth=1.0,
                draw_back=True,
                draw_front=True,
                base_plane_color=None,
                cell_axes="none",
                front_zorder=1.2,
            )
            c_arr = np.vstack(corners)
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            ax.set_xlim(
                min(x0, float(np.min(c_arr[:, 0])) - 0.8),
                max(x1, float(np.max(c_arr[:, 0])) + 0.8),
            )
            ax.set_ylim(
                min(y0, float(np.min(c_arr[:, 1])) - 1.0),
                max(y1, float(np.max(c_arr[:, 1])) + 0.8),
            )

    if has_dashed:
        with theme.xkcd_style(preset=preset):
            handles = [
                mlines.Line2D([], [], color=ink_col, lw=2.0, label="Bond"),
                mlines.Line2D(
                    [],
                    [],
                    color="#0072B2",
                    lw=1.6,
                    linestyle="--",
                    label="Periodic edge",
                ),
            ]
            ax.legend(
                handles=handles,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.06),
                ncol=2,
                frameon=False,
                fontsize=8.8,
            )


def _load_graph_atoms(
    source: structure.AtomsData | str | pathlib.Path,
) -> structure.AtomsData:
    """Coerces a file path, sample name, or ``AtomsData`` into ``AtomsData``."""
    if isinstance(source, structure.AtomsData):
        return source
    loaded = loader.load(source)
    if not isinstance(loaded, structure.AtomsData):
        raise TypeError(
            f"Expected AtomsData for plot_structure_graph, got {loaded!r}."
        )
    return loaded


def _compute_structure_graph_coords(
    atoms: structure.AtomsData,
    layout: str,
    camera: geometry.Camera | str | None,
) -> tuple[np.ndarray | None, geometry.Camera, np.ndarray]:
    """Computes camera-projected or PCA 2D coordinates for a structure graph."""
    cam = geometry.resolve_camera(camera, cell=atoms.cell)
    pivot = (
        0.5 * (atoms.cell[0] + atoms.cell[1] + atoms.cell[2])
        if atoms.cell is not None
        else np.mean(atoms.positions, axis=0)
    )
    key = layout.strip().lower()
    if key == "camera" and len(atoms) >= 1:
        xy, _ = geometry.project_points(atoms.positions, cam, center=pivot)
        return xy, cam, pivot
    if key == "pca" and len(atoms) >= 2:
        centered = atoms.positions - np.mean(atoms.positions, axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        return centered @ vt[:2].T, cam, pivot
    return None, cam, pivot


def plot_structure_graph(
    source: structure.AtomsData | str | pathlib.Path,
    *,
    layout: str = "camera",
    camera: geometry.Camera | str | None = None,
    show_cell: bool = True,
    show_distances: bool = False,
    show_periodic_edges: bool = True,
    multi_edges: bool = False,
    show_shift_labels: bool = False,
    directed: bool = False,
    background_grid: str | None = None,
    palette: str | Mapping[str, str] = "xkcd",
    preset: str | theme.ThemePreset = "classic",
    ax: maxes.Axes | None = None,
    title: str | None = None,
) -> maxes.Axes:
    """Renders a crystal or molecule as an intuitive xkcd GNN structure graph.

    Supports both camera-matched 3D crystal graphs (``layout="camera"`` with
    faint unit-cell box guide) and flat planar 2D topological/quotient graphs
    (``layout="planar"``, optionally with ``multi_edges=True`` and
    ``show_shift_labels=True`` to display fanned periodic translation vectors).

    Args:
      source: ``AtomsData``, file path, or bundled sample filename.
      layout: ``"camera"`` (true 3D crystal view), ``"planar"`` (flat 2D),
        ``"pca"``, ``"spring"``, or ``"circular"``.
      camera: ``Camera`` instance or preset string when ``layout="camera"``.
      show_cell: Whether to draw a subtle unit-cell wireframe when periodic and
        ``layout="camera"``.
      show_distances: Whether to annotate intra-cell bond lengths in Angstrom.
      show_periodic_edges: Whether to include periodic-boundary image arcs.
      multi_edges: Whether to draw every distinct periodic image shift as a
        separate fanned arc (quotient multigraph view).
      show_shift_labels: Whether to label edges with integer lattice-shift
        vectors (e.g. ``"000"``, ``"100"``, ``"010"``, ``"001"``).
      directed: Whether to draw directional arrowheads on edges.
      background_grid: Optional background grid (default ``None``).
      palette: Element color scheme name or ``{symbol: hex}`` mapping.
      preset: Visual theme preset name or ``ThemePreset``.
      ax: Optional target Matplotlib ``Axes``.
      title: Optional heading above the graph.

    Returns:
      Matplotlib ``Axes`` containing the structure graph.
    """
    atoms = _load_graph_atoms(source)
    nodes = [f"{sym}{idx}" for idx, sym in enumerate(atoms.symbols)]
    node_cols = [
        palettes.get_element_color(sym, scheme=palette) for sym in atoms.symbols
    ]
    edge_items, dashed_flags = _extract_structure_edges(
        atoms,
        show_distances,
        show_periodic_edges,
        multi_edges=multi_edges,
        show_shift_labels=show_shift_labels,
    )
    pos_2d, cam, pivot = _compute_structure_graph_coords(atoms, layout, camera)
    ax = plot_graph_network(
        nodes,
        edge_items,
        pos=pos_2d,
        layout=layout if pos_2d is None else "camera",
        node_colors=node_cols,
        dashed_edges=dashed_flags,
        directed=directed,
        background_grid=background_grid,
        preset=preset,
        ax=ax,
        title=title or f"{atoms.formula()} Structure Graph",
    )
    _add_structure_graph_overlays(
        ax,
        atoms,
        cam,
        pivot,
        show_cell=show_cell,
        layout=layout,
        has_dashed=any(dashed_flags),
        preset=preset,
    )
    return ax
