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
from matplotlib import patches
from matplotlib import pyplot as plt
import numpy as np

from xkcd_matchem.core import bonding
from xkcd_matchem.core import exceptions
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
) -> None:
    """Draws a straight or curved xkcd graph edge with optional label."""
    ls = "--" if dashed else "-"
    if directed or abs(rad) > 1e-4:
        arrowstyle = "->,head_width=0.28,head_length=0.40" if directed else "-"
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
                "shrinkA": 15,
                "shrinkB": 15,
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
        perp = np.array([-vec[1], vec[0]], dtype=np.float64)
        p_norm = float(np.linalg.norm(perp))
        if p_norm > 1e-6 and abs(rad) > 1e-4:
            mid = mid + (perp / p_norm) * (0.26 * rad * p_norm)
        ax.text(
            float(mid[0]),
            float(mid[1]),
            label,
            ha="center",
            va="center",
            fontsize=8.0,
            color=color,
            bbox={
                "boxstyle": "round,pad=0.14",
                "facecolor": bg_color,
                "edgecolor": "none",
                "alpha": 0.88,
            },
            zorder=4.5,
        )


def _draw_graph_nodes(
    ax: maxes.Axes,
    nodes: Sequence[str],
    xy: np.ndarray,
    node_colors: Sequence[str] | Mapping[int, str] | None,
    series: Sequence[str],
    ink_col: str,
) -> float:
    """Draws shaded xkcd network nodes and returns the base node radius."""
    n_nodes = len(nodes)
    base_r = max(0.085, min(0.145, 0.52 / max(math.sqrt(n_nodes), 2.8)))
    for idx, raw_lbl in enumerate(nodes):
        if isinstance(node_colors, Mapping) and idx in node_colors:
            col = str(node_colors[idx])
        elif isinstance(node_colors, Sequence) and not isinstance(
            node_colors, str
        ):
            col = str(node_colors[idx % len(node_colors)])
        elif raw_lbl in elements.ELEMENTS_BY_SYMBOL:
            col = palettes.get_element_color(raw_lbl, scheme="xkcd")
        else:
            col = series[(idx + 1) % len(series)]
        ax.add_patch(
            patches.Circle(
                (float(xy[idx, 0]), float(xy[idx, 1])),
                radius=base_r,
                facecolor=col,
                edgecolor=ink_col,
                linewidth=1.8,
                zorder=5.0,
            )
        )
        ax.add_patch(
            patches.Circle(
                (
                    float(xy[idx, 0]) - 0.28 * base_r,
                    float(xy[idx, 1]) + 0.28 * base_r,
                ),
                radius=0.18 * base_r,
                facecolor="#FFFFFF",
                edgecolor="none",
                alpha=0.72,
                zorder=5.2,
            )
        )
        txt_col = primitives.contrast_text_color(mcolors.to_hex(col))
        ax.text(
            float(xy[idx, 0]),
            float(xy[idx, 1]),
            str(raw_lbl),
            ha="center",
            va="center",
            fontsize=max(7.5, min(10.0, 62.0 * base_r)),
            color=txt_col,
            zorder=6.0,
        )
    return base_r


def _draw_all_graph_edges(
    ax: maxes.Axes,
    n_nodes: int,
    norm_edges: Sequence[tuple[int, int, str | None]],
    xy: np.ndarray,
    dashed_set: set[tuple[int, int]],
    style_cfg: tuple[bool, bool, str, str],
) -> None:
    """Draws all network edges with automatic multi-edge arc separation."""
    directed, curved, ink_col, bg_color = style_cfg
    pair_seen: dict[tuple[int, int], int] = {}
    for u, v, lbl in norm_edges:
        if u >= n_nodes or v >= n_nodes:
            continue
        key = (min(u, v), max(u, v))
        mult = pair_seen.get(key, 0)
        pair_seen[key] = mult + 1
        is_dash = key in dashed_set
        rad = (
            0.22 * (1 if mult % 2 == 0 else -1) * (1 + mult // 2)
            if (mult > 0 or is_dash)
            else (0.14 if curved else 0.0)
        )
        _draw_graph_edge(
            ax,
            xy[u],
            xy[v],
            directed=directed,
            dashed=is_dash,
            rad=rad,
            color="#2A9D8F" if is_dash else ink_col,
            linewidth=1.6 if is_dash else 2.0,
            label=lbl,
            bg_color=bg_color,
        )


def plot_graph_network(
    nodes: Sequence[str],
    edges: Sequence[tuple[int, int] | tuple[int, int, str]],
    *,
    pos: np.ndarray | Sequence[tuple[float, float]] | None = None,
    layout: str = "spring",
    node_colors: Sequence[str] | Mapping[int, str] | None = None,
    dashed_edges: Sequence[tuple[int, int]] | None = None,
    directed: bool = False,
    curved: bool = False,
    background_grid: str | None = None,
    palette: str | Sequence[str] = "okabe_ito",
    preset: str | theme.ThemePreset = "classic",
    ax: maxes.Axes | None = None,
    title: str | None = None,
) -> maxes.Axes:
    """Plots a clean xkcd-style graph network for materials/chem informatics.

    Automatically styles node fills using element colors whenever node labels
    match a valid element symbol (or cycles ``palette`` otherwise).

    Args:
      nodes: Sequence of node labels (e.g. ``["LiFePO4", "FePO4", ...]`` or
        ``["Ti0", "O1", ...]``).
      edges: Sequence of ``(u, v)`` or ``(u, v, label)`` index tuples.
      pos: Optional explicit ``(N, 2)`` coordinates.
      layout: Layout mode when ``pos`` is ``None``: ``"spring"``,
        ``"circular"``, or ``"grid"``.
      node_colors: Optional per-node hex color list or ``{idx: color}``.
      dashed_edges: Optional edge pairs rendered as dashed arcs (e.g. periodic
        images or secondary interactions).
      directed: Whether to draw arrowheads along edges.
      curved: Whether to curve all edges slightly for comic flair.
      background_grid: Optional background grid (``"dots"``, ``"grid"``, etc.).
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
    dashed_set = {
        (min(int(a), int(b)), max(int(a), int(b)))
        for a, b in (dashed_edges or ())
    }

    with theme.xkcd_style(preset=preset):
        if ax is None:
            _, ax = plt.subplots(figsize=(6.4, 5.2))
        else:
            theme.style_existing_axes(ax, preset=preset)
        if background_grid:
            patterns.apply_background_grid(ax, style=background_grid)
        _draw_all_graph_edges(
            ax,
            n_nodes,
            norm_edges,
            xy,
            dashed_set,
            (directed, curved, ink_col, preset_obj.axes_facecolor),
        )
        base_r = _draw_graph_nodes(ax, nodes, xy, node_colors, series, ink_col)
        if n_nodes > 0:
            pad = base_r * 2.2 + 0.18
            ax.set_xlim(
                float(np.min(xy[:, 0])) - pad, float(np.max(xy[:, 0])) + pad
            )
            ax.set_ylim(
                float(np.min(xy[:, 1])) - pad, float(np.max(xy[:, 1])) + pad
            )
        ax.set_aspect("equal")
        ax.set_axis_off()
        if title:
            ax.set_title(title, color=ink_col)
    return ax


def _extract_structure_edges(
    atoms: structure.AtomsData,
    show_distances: bool,
    show_periodic_edges: bool,
) -> tuple[list[tuple[int, int, str]], list[tuple[int, int]]]:
    """Builds labeled solid and periodic dashed edges from an ``AtomsData``."""
    bonds = list(atoms.bonds) or bonding.infer_bonds(atoms)
    edge_items: list[tuple[int, int, str]] = []
    dashed: list[tuple[int, int]] = []
    seen_pbc: dict[tuple[int, int], int] = {}
    for b in bonds:
        is_pbc = b.shift != (0, 0, 0)
        if is_pbc and not show_periodic_edges:
            continue
        if is_pbc and atoms.cell is not None:
            pair = (min(b.i, b.j), max(b.i, b.j))
            seen_pbc[pair] = seen_pbc.get(pair, 0) + 1
            if seen_pbc[pair] > 1:
                continue
            dashed.append((b.i, b.j))
            edge_items.append((b.i, b.j, "pbc"))
            continue
        dist = float(
            np.linalg.norm(atoms.positions[b.j] - atoms.positions[b.i])
        )
        lbl = ""
        if show_distances:
            order_tag = "=" if abs(b.order - 2.0) < 0.25 else ""
            lbl = f"{order_tag}{dist:.2f}A"
        edge_items.append((b.i, b.j, lbl))
    return edge_items, dashed


def plot_structure_graph(
    source: structure.AtomsData | str | pathlib.Path,
    *,
    show_distances: bool = True,
    show_periodic_edges: bool = True,
    layout: str = "pca",
    background_grid: str | None = "dotgrid",
    palette: str | Mapping[str, str] = "xkcd",
    preset: str | theme.ThemePreset = "classic",
    ax: maxes.Axes | None = None,
    title: str | None = None,
) -> maxes.Axes:
    """Renders a crystal or molecule as an xkcd GNN graph (atoms + bonds).

    Ideal for illustrating Crystal Graph Convolutional Networks (CGCNN, MEGNet,
    MACE) and molecular message-passing graphs: intra-cell bonds draw as solid
    edges and periodic image bonds draw as curved dashed arcs.

    Args:
      source: ``AtomsData``, file path, or bundled sample filename.
      show_distances: Whether to label edges with bond lengths in Angstrom.
      show_periodic_edges: Whether to include periodic-boundary image edges.
      layout: ``"pca"`` (2D PCA of real atomic coordinates), ``"spring"``, or
        ``"circular"``.
      background_grid: Optional background grid (default ``"dotgrid"``).
      palette: Element color scheme name or ``{symbol: hex}`` mapping.
      preset: Visual theme preset name or ``ThemePreset``.
      ax: Optional target Matplotlib ``Axes``.
      title: Optional heading above the graph.

    Returns:
      Matplotlib ``Axes`` containing the structure graph.
    """
    if isinstance(source, structure.AtomsData):
        atoms = source
    else:
        loaded = loader.load(source)
        if not isinstance(loaded, structure.AtomsData):
            raise TypeError(
                f"Expected AtomsData for plot_structure_graph, got {loaded!r}."
            )
        atoms = loaded

    nodes = [f"{sym}{idx}" for idx, sym in enumerate(atoms.symbols)]
    node_cols = [
        palettes.get_element_color(sym, scheme=palette) for sym in atoms.symbols
    ]
    edge_items, dashed = _extract_structure_edges(
        atoms, show_distances, show_periodic_edges
    )

    pos_2d: np.ndarray | None = None
    if layout.strip().lower() == "pca" and len(atoms) >= 2:
        centered = atoms.positions - np.mean(atoms.positions, axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        proj = centered @ vt[:2].T
        span = max(float(np.max(np.abs(proj))), 1e-6)
        pos_2d = proj / span

    return plot_graph_network(
        nodes,
        edge_items,
        pos=pos_2d,
        layout="spring" if pos_2d is None else layout,
        node_colors=node_cols,
        dashed_edges=dashed,
        background_grid=background_grid,
        preset=preset,
        ax=ax,
        title=title or f"{atoms.formula()} Structure Graph",
    )
