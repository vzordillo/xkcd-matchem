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
"""Painter's-algorithm depth-sorted 3D structure and crystal renderer."""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence

from matplotlib import axes as maxes
from matplotlib import pyplot as plt
import numpy as np

from xkcd_matchem.core import bonding
from xkcd_matchem.core import exceptions
from xkcd_matchem.core import geometry
from xkcd_matchem.core import structure
from xkcd_matchem.data import elements
from xkcd_matchem.render import primitives
from xkcd_matchem.style import palettes


_VALID_STYLES = ("ball_and_stick", "spacefill", "stick", "polyhedra")


def _atom_visual_radius(symbol: str, style: str, scale: float) -> float:
    """Returns visual 2D sphere radius in Angstrom-scaled display units."""
    if style == "spacefill":
        return elements.get_vdw_radius(symbol) * 0.82 * scale
    if style == "stick":
        return 0.14 * scale
    if style == "polyhedra":
        return max(0.18, elements.get_covalent_radius(symbol) * 0.26) * scale
    r_cov = elements.get_covalent_radius(symbol)
    return max(0.24, min(0.58, 0.18 + 0.27 * r_cov)) * scale


_CORNER_AXES_LOCS = ("bottom_left", "bottom_right", "top_left", "top_right")


def _resolve_axes_override(
    crystal_axes: bool | str | None,
    draw_box: bool,
    default_axes: str,
) -> tuple[str, str]:
    """Resolves ``(box_axes, standalone_axes)`` from ``crystal_axes``."""
    if crystal_axes is None:
        return default_axes, "none"
    if isinstance(crystal_axes, bool):
        return "none", ("bottom_left" if crystal_axes else "none")
    ax_key = crystal_axes.strip().lower()
    mapping: dict[str, tuple[str, str]] = {
        "none": ("none", "none"),
        "off": ("none", "none"),
        "false": ("none", "none"),
        "corner": ("none", "bottom_left"),
        "bottom_left": ("none", "bottom_left"),
        "bottom_right": ("none", "bottom_right"),
        "top_left": ("none", "top_left"),
        "top_right": ("none", "top_right"),
        "origin": ("none", "origin"),
        "tripod": ("none", "origin"),
        "edges": ("edges" if draw_box else "none", "none"),
    }
    if ax_key in mapping:
        return mapping[ax_key]
    raise exceptions.StructureError(
        f"Unsupported crystal_axes mode {crystal_axes!r}. Use None, True, "
        "False, 'corner', 'bottom_left', 'bottom_right', 'top_left', "
        "'top_right', 'origin', or 'edges'."
    )


def _parse_cell_and_axes(
    show_cell: bool | str,
    crystal_axes: bool | str | None,
    has_cell: bool,
    has_miller: bool,
) -> tuple[bool, str | None, str, str]:
    """Resolves ``(draw_box, base_plane_color, box_axes, standalone_axes)``."""
    if not has_cell:
        return False, None, "none", "none"

    draw_box = True
    base_col: str | None = None if has_miller else palettes.VACUUM_TINT
    default_axes = "edges"
    if show_cell is False:
        draw_box, base_col, default_axes = False, None, "none"
    elif isinstance(show_cell, str):
        key = show_cell.strip().lower()
        if key in ("none", "off", "false"):
            draw_box, base_col, default_axes = False, None, "none"
        elif key == "wireframe":
            draw_box, base_col, default_axes = True, None, "edges"
        elif key == "clean":
            draw_box, base_col, default_axes = True, None, "none"
        elif key in ("tripod", "origin"):
            draw_box, base_col, default_axes = True, None, "origin"

    box_axes, standalone_axes = _resolve_axes_override(
        crystal_axes, draw_box, default_axes
    )
    return draw_box, base_col, box_axes, standalone_axes


def _polyhedra_skip_spec(
    style: str,
    polyhedra: tuple[str, str | None]
    | Sequence[bonding.CoordinationPolyhedron]
    | None,
) -> tuple[set[str], set[str] | None, set[int]] | None:
    """Returns ``(center_syms, ligand_syms, center_idxs)`` to suppress bonds."""
    if style != "polyhedra" or polyhedra is None:
        return None
    if (
        isinstance(polyhedra, tuple)
        and len(polyhedra) == 2
        and isinstance(polyhedra[0], str)
    ):
        ligs = None if polyhedra[1] is None else {polyhedra[1]}
        return {polyhedra[0]}, ligs, set()
    c_syms = {p.center_symbol for p in polyhedra}  # type: ignore[union-attr]
    c_idxs = {p.center_index for p in polyhedra}  # type: ignore[union-attr]
    return (c_syms, None, c_idxs) if c_syms else None


def _should_skip_poly_bond(
    i_idx: int,
    j_idx: int,
    atoms: structure.AtomsData,
    skip_spec: tuple[set[str], set[str] | None, set[int]] | None,
) -> bool:
    """Returns ``True`` when a bond is redundant with coordination polyhedra."""
    if skip_spec is None:
        return False
    c_syms, l_syms, c_idxs = skip_spec
    if i_idx in c_idxs or j_idx in c_idxs:
        return True
    s_i, s_j = atoms.symbols[i_idx], atoms.symbols[j_idx]
    if s_i in c_syms and s_j in c_syms:
        return True
    match_fwd = s_i in c_syms and (l_syms is None or s_j in l_syms)
    match_rev = s_j in c_syms and (l_syms is None or s_i in l_syms)
    return match_fwd or match_rev


def _build_interior_bond_item(
    atoms: structure.AtomsData,
    bond: structure.Bond,
    xy: np.ndarray,
    depth: np.ndarray,
    colors: Sequence[str],
    radii: Sequence[float],
) -> tuple[float, str, object]:
    """Builds one unified depth-sorted interior bond item."""
    i_idx, j_idx = bond.i, bond.j
    i_back, i_front = (
        (i_idx, j_idx) if depth[i_idx] <= depth[j_idx] else (j_idx, i_idx)
    )
    dz = float(depth[i_front] - depth[i_back])
    p_back = xy[i_back]
    p_front = xy[i_front]
    seg_2d = float(np.linalg.norm(p_front - p_back))
    if dz < 0.25 or seg_2d < 0.65 * radii[i_back]:
        bond_z = float(depth[i_back]) - 0.04
        p_start = p_back
        split_frac = 0.5
    else:
        d3 = float(
            np.linalg.norm(atoms.positions[i_front] - atoms.positions[i_back])
        )
        t_start = min(0.32, 0.55 * radii[i_back] / max(d3, 1e-6))
        p_start = p_back + t_start * (p_front - p_back)
        split_frac = (0.5 - t_start) / max(1.0 - t_start, 1e-6)
        bond_z = 0.5 * float(depth[i_back] + depth[i_front]) - 0.02
    return (
        bond_z,
        "bond",
        (
            p_start,
            p_front,
            colors[i_back],
            colors[i_front],
            bond.order,
            split_frac,
        ),
    )


def _add_bond_queue_items(
    items: list[tuple[float, str, object]],
    atoms: structure.AtomsData,
    bonds: Sequence[structure.Bond],
    cam: geometry.Camera,
    pivot: np.ndarray,
    xy_and_depth: tuple[np.ndarray, np.ndarray],
    colors: Sequence[str],
    radii: Sequence[float],
    skip_spec: tuple[set[str], set[str] | None, set[int]] | None = None,
) -> None:
    """Appends unified two-tone interior bonds and periodic edge stubs."""
    xy, depth = xy_and_depth
    for bond in bonds:
        i_idx, j_idx = bond.i, bond.j
        if i_idx >= len(atoms) or j_idx >= len(atoms):
            continue
        if _should_skip_poly_bond(i_idx, j_idx, atoms, skip_spec):
            continue
        if bond.shift == (0, 0, 0) or atoms.cell is None:
            items.append(
                _build_interior_bond_item(atoms, bond, xy, depth, colors, radii)
            )
            continue
        shift_vec = (
            bond.shift[0] * atoms.cell[0]
            + bond.shift[1] * atoms.cell[1]
            + bond.shift[2] * atoms.cell[2]
        )
        p_i = atoms.positions[i_idx]
        p_j_img = atoms.positions[j_idx] + shift_vec
        pts, _ = geometry.project_points(
            np.vstack(
                [
                    p_i + 0.42 * (p_j_img - p_i),
                    atoms.positions[j_idx]
                    + 0.42 * (p_i - shift_vec - atoms.positions[j_idx]),
                ]
            ),
            cam,
            center=pivot,
        )
        items.append(
            (
                float(depth[i_idx]) - 0.06,
                "half",
                (xy[i_idx], pts[0], colors[i_idx], bond.order),
            )
        )
        items.append(
            (
                float(depth[j_idx]) - 0.06,
                "half",
                (xy[j_idx], pts[1], colors[j_idx], bond.order),
            )
        )


def _add_polyhedra_items(
    items: list[tuple[float, str, object]],
    atoms: structure.AtomsData,
    polyhedra: tuple[str, str | None]
    | Sequence[bonding.CoordinationPolyhedron]
    | None,
    cam: geometry.Camera,
    pivot: np.ndarray,
    palette: str | Mapping[str, str],
) -> None:
    """Appends depth-sorted individual coordination triangle facets."""
    if polyhedra is None:
        return
    if (
        isinstance(polyhedra, tuple)
        and len(polyhedra) == 2
        and isinstance(polyhedra[0], str)
    ):
        poly_list = bonding.find_coordination_polyhedra(
            atoms, center_symbol=polyhedra[0], ligand_symbol=polyhedra[1]
        )
    else:
        poly_list = list(polyhedra)  # type: ignore[arg-type]
    for poly in poly_list:
        p_xy, p_depth = geometry.project_points(
            poly.vertices_3d, cam, center=pivot
        )
        c_col = palettes.get_element_color(poly.center_symbol, scheme=palette)
        for i0, i1, i2 in poly.faces:
            tri_xy = np.vstack([p_xy[i0], p_xy[i1], p_xy[i2]])
            mean_d = float((p_depth[i0] + p_depth[i1] + p_depth[i2]) / 3.0)
            items.append((mean_d - 0.03, "tri", (tri_xy, c_col)))


def _draw_queue_item(
    ax: maxes.Axes,
    kind: str,
    payload: object,
    z_layer: float,
    state: tuple[
        structure.AtomsData,
        np.ndarray,
        list[str],
        list[float],
        float,
        float,
        float,
        bool,
        str,
    ],
) -> None:
    """Paints a single depth-ranked atom sphere, bond, or polyhedron facet."""
    (
        atoms,
        xy,
        colors,
        radii,
        bond_w,
        bond_off,
        atom_lw,
        show_labels,
        style,
    ) = state
    if kind == "atom":
        idx = int(payload)  # type: ignore[arg-type]
        primitives.draw_atom_sphere(
            ax,
            float(xy[idx, 0]),
            float(xy[idx, 1]),
            radius=radii[idx],
            facecolor=colors[idx],
            linewidth=atom_lw,
            label=atoms.symbols[idx] if show_labels else None,
            highlight=style != "stick",
            zorder=z_layer,
        )
    elif kind == "bond":
        p_a, p_b, col_a, col_b, order, split_frac = payload  # type: ignore[misc]
        primitives.draw_two_tone_bond(
            ax,
            p_a,
            p_b,
            c1=col_a,
            c2=col_b,
            width=bond_w,
            order=order,
            offset_scale=bond_off,
            zorder=z_layer,
            split_frac=split_frac,
        )
    elif kind == "half":
        p_a, p_b, col, order = payload  # type: ignore[misc]
        primitives.draw_half_bond(
            ax,
            p_a,
            p_b,
            color=col,
            width=bond_w,
            order=order,
            offset_scale=bond_off,
            zorder=z_layer,
        )
    elif kind == "tri":
        primitives.draw_polyhedron_triangle(
            ax,
            payload[0],  # type: ignore[index]
            facecolor=payload[1],  # type: ignore[index]
            zorder=z_layer,
        )
    elif kind == "miller" and atoms.cell is not None:
        primitives.draw_miller_plane(
            ax,
            atoms.cell,
            payload[0],  # type: ignore[index]
            payload[3],  # type: ignore[index]
            center=payload[4],  # type: ignore[index]
            facecolor=payload[1],  # type: ignore[index]
            edgecolor=payload[2],  # type: ignore[index]
            show_label=False,
            zorder=z_layer,
        )
    elif kind == "cell_edge":
        ax.plot(
            [float(payload[0][0]), float(payload[1][0])],  # type: ignore[index]
            [float(payload[0][1]), float(payload[1][1])],  # type: ignore[index]
            color=str(plt.rcParams.get("axes.edgecolor", palettes.INK)),
            linestyle="-",
            linewidth=payload[2],  # type: ignore[index]
            solid_capstyle="round",
            zorder=z_layer,
        )


def _estimate_pt_per_angstrom(
    ax: maxes.Axes,
    bounds_2d: np.ndarray,
    margin: float,
) -> float:
    """Estimates screen points per Angstrom from axes size and bounds."""
    fig_w, fig_h = ax.figure.get_size_inches()
    pos = ax.get_position()
    ax_w_pt = max(float(pos.width * fig_w) * 72.0, 90.0)
    ax_h_pt = max(float(pos.height * fig_h) * 72.0, 90.0)
    x_span = max(float(np.ptp(bounds_2d[:, 0])) + 2.0 * margin, 2.0)
    y_span = max(float(np.ptp(bounds_2d[:, 1])) + 2.0 * margin, 2.0)
    return min(ax_w_pt / x_span, ax_h_pt / y_span)


def _miller_plane_colors(index: int) -> tuple[str, str]:
    """Returns ``(facecolor, edgecolor)`` pair for the ``index``-th plane."""
    fill_cycle = ("#3B82F6", "#10B981", "#F59E0B", "#EC4899", "#8B5CF6")
    edge_cycle = ("#1D4ED8", "#047857", "#B45309", "#BE185D", "#6D28D9")
    return fill_cycle[index % len(fill_cycle)], edge_cycle[
        index % len(edge_cycle)
    ]


def _add_miller_queue_items(
    items: list[tuple[float, str, object]],
    atoms: structure.AtomsData,
    miller_planes: Sequence[tuple[int, int, int]] | None,
    cam: geometry.Camera,
    pivot: np.ndarray,
) -> list[tuple[tuple[int, int, int], np.ndarray, str]]:
    """Adds depth-sorted Miller slices to queue and returns label positions."""
    labels_meta: list[tuple[tuple[int, int, int], np.ndarray, str]] = []
    if not miller_planes or atoms.cell is None:
        return labels_meta
    for idx_m, hkl in enumerate(miller_planes):
        poly_3d = geometry.miller_plane_polygon(atoms.cell, hkl)
        if poly_3d.shape[0] < 3:
            continue
        poly_2d, poly_depth = geometry.project_points(
            poly_3d, cam, center=pivot
        )
        face_col, edge_col = _miller_plane_colors(idx_m)
        items.append(
            (
                float(np.mean(poly_depth)) - 0.01,
                "miller",
                (hkl, face_col, edge_col, cam, pivot),
            )
        )
        labels_meta.append((hkl, np.mean(poly_2d, axis=0), edge_col))
    return labels_meta


def _draw_miller_labels(
    ax: maxes.Axes,
    labels_meta: Sequence[tuple[tuple[int, int, int], np.ndarray, str]],
    zorder: float,
) -> None:
    """Places crisp ``(hkl)`` badges above the rendered crystal."""
    for hkl, centroid, edge_col in labels_meta:
        parts = [str(int(v)) for v in hkl]
        tag = (
            f"({''.join(parts)})"
            if all(len(p) == 1 for p in parts)
            else f"({' '.join(parts)})"
        )
        ax.text(
            float(centroid[0]),
            float(centroid[1]),
            tag,
            fontsize=9.5,
            color=edge_col,
            ha="center",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": palettes.PAPER,
                "edgecolor": edge_col,
                "alpha": 0.92,
            },
            zorder=zorder,
        )


def _prepare_scene_geometry(
    atoms: structure.AtomsData,
    ax: maxes.Axes,
    cam: geometry.Camera,
    style: str,
    palette: str | Mapping[str, str],
    cell_cfg: tuple[bool, str | None, str, str],
    radius_scale: float,
    pad: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
    list[float],
    float,
    tuple[bool, str | None, str, str],
    tuple[float, float, float, float],
]:
    """Computes projected points, colors, radii, cell mode, and widths."""
    pivot = (
        0.5 * (atoms.cell[0] + atoms.cell[1] + atoms.cell[2])
        if atoms.cell is not None
        else np.mean(atoms.positions, axis=0)
    )
    xy, depth = geometry.project_points(atoms.positions, cam, center=pivot)
    colors = [
        palettes.get_element_color(sym, scheme=palette) for sym in atoms.symbols
    ]
    radii = [
        _atom_visual_radius(sym, style, radius_scale) for sym in atoms.symbols
    ]
    margin = max(radii, default=0.4) + pad

    bounds_pts: list[np.ndarray] = [xy]
    if cell_cfg[0] and atoms.cell is not None:
        corners_2d, _, _ = geometry.cell_corners_projected(
            atoms.cell, cam, center=pivot
        )
        bounds_pts.append(corners_2d)
    all_pts = np.vstack(bounds_pts)

    pt_per_a = _estimate_pt_per_angstrom(ax, all_pts, margin)
    bond_w = float(np.clip(0.26 * pt_per_a, 1.8, 5.2))
    if style == "stick":
        bond_w *= 1.35
    rail_outer = bond_w * 0.50 + max(1.4, 0.27 * bond_w)
    bond_off = float(
        np.clip(0.72 * rail_outer / max(pt_per_a, 1.0), 0.08, 0.135)
    )
    atom_lw = float(np.clip(0.11 * pt_per_a, 1.3, 2.2))
    cell_lw = float(np.clip(0.10 * pt_per_a, 1.15, 1.9))
    return (
        pivot,
        xy,
        depth,
        all_pts,
        colors,
        radii,
        margin,
        cell_cfg,
        (bond_w, bond_off, atom_lw, cell_lw),
    )


def render_structure_3d(
    atoms: structure.AtomsData,
    ax: maxes.Axes | None = None,
    camera: geometry.Camera | str | None = None,
    style: str = "ball_and_stick",
    palette: str | Mapping[str, str] = "xkcd",
    show_cell: bool | str = True,
    show_labels: bool = False,
    show_bonds: bool | None = None,
    radius_scale: float = 1.0,
    bond_scale: float = 1.20,
    polyhedra: tuple[str, str | None]
    | Sequence[bonding.CoordinationPolyhedron]
    | None = None,
    miller_planes: Sequence[tuple[int, int, int]] | None = None,
    pad: float = 0.85,
    *,
    crystal_axes: bool | str | None = None,
) -> maxes.Axes:
    """Renders a 3D molecule or crystal in clean xkcd-style depth-sorted view.

    Args:
      atoms: Input ``AtomsData`` structure.
      ax: Optional target Matplotlib ``Axes``. Created if ``None``.
      camera: ``Camera`` configuration or preset string (``"isometric"``,
        ``"perspective"``, ``"a"``, ``"b"``, ``"c"``, ``"top"``, etc.).
      style: Visual mode: ``"ball_and_stick"``, ``"spacefill"``,
        ``"stick"``, or ``"polyhedra"``.
      palette: Built-in or registered element scheme name, or inline
        ``{symbol: color}`` mapping overriding defaults.
      show_cell: Cell box display mode: ``True`` (base plane + edges),
        ``"wireframe"``, ``"clean"``, ``"tripod"``, or ``False``.
      crystal_axes: Independent crystallographic ``a, b, c`` axes option:
        ``None`` (follow ``show_cell``), ``True`` / ``"corner"`` /
        ``"bottom_left"`` / ``"bottom_right"`` / ``"top_left"`` /
        ``"top_right"`` (standalone VESTA-style triad separate from the cell
        box), ``"origin"`` (RGB tripod at cell origin), ``"edges"`` (box edge
        labels), or ``False`` / ``"none"`` (hidden).
      show_labels: Whether to print element symbols inside atom spheres.
      show_bonds: Whether to draw stick bonds. Defaults to ``None``, which
        draws bonds for ``"ball_and_stick"`` and ``"stick"`` while keeping
        ``"spacefill"`` and ``"polyhedra"`` views free of bond sticks. Pass
        ``True`` explicitly to overlay non-polyhedral bonds (for example
        organic linkers alongside coordination polyhedra).
      radius_scale: Global multiplier for atom sphere radii.
      bond_scale: Covalent-radius factor used when inferring missing bonds.
      polyhedra: ``(center_symbol, ligand_symbol)`` pair or sequence of
        ``CoordinationPolyhedron``.
      miller_planes: Optional crystallographic planes ``[(h, k, l), ...]``.
      pad: Viewport margin in Angstrom around all projected geometry.

    Returns:
      Matplotlib ``Axes`` containing the rendered structure.

    Raises:
      StructureError: If ``style``, ``crystal_axes``, or ``camera`` is invalid.
    """
    if style not in _VALID_STYLES:
        raise exceptions.StructureError(
            f"Unsupported 3D render style {style!r}. Valid: {_VALID_STYLES}"
        )
    if ax is None:
        _, ax = plt.subplots(figsize=(5.6, 4.8))
    if len(atoms) == 0:
        ax.set_axis_off()
        return ax

    cam = geometry.resolve_camera(camera, cell=atoms.cell)
    cell_cfg = _parse_cell_and_axes(
        show_cell, crystal_axes, atoms.cell is not None, bool(miller_planes)
    )
    scene = _prepare_scene_geometry(
        atoms, ax, cam, style, palette, cell_cfg, radius_scale, pad
    )
    draw_bonds = (
        style not in ("spacefill", "polyhedra")
        if show_bonds is None
        else show_bonds
    )
    opts = (style, palette, show_labels, draw_bonds, bond_scale)
    _render_prepared_3d(ax, atoms, cam, scene, opts, polyhedra, miller_planes)
    return ax


def _build_3d_painter_queue(
    atoms: structure.AtomsData,
    cam: geometry.Camera,
    pivot: np.ndarray,
    xy: np.ndarray,
    depth: np.ndarray,
    colors: list[str],
    radii: list[float],
    opts: tuple[str, str | Mapping[str, str], bool, bool, float],
    polyhedra: tuple[str, str | None]
    | Sequence[bonding.CoordinationPolyhedron]
    | None,
    miller_planes: Sequence[tuple[int, int, int]] | None,
) -> tuple[
    list[tuple[float, str, object]],
    list[tuple[tuple[int, int, int], np.ndarray, str]],
]:
    """Assembles and depth-sorts atoms, bonds, polyhedra, and Miller slices."""
    style, palette, _, show_bonds, bond_scale = opts
    items: list[tuple[float, str, object]] = [
        (float(depth[idx]), "atom", idx) for idx in range(len(atoms))
    ]
    if show_bonds and style != "spacefill":
        bonds = list(atoms.bonds) or (
            bonding.infer_bonds(atoms, scale=bond_scale)
            if len(atoms) <= 600
            else []
        )
        _add_bond_queue_items(
            items,
            atoms,
            bonds,
            cam,
            pivot,
            (xy, depth),
            colors,
            radii,
            skip_spec=_polyhedra_skip_spec(style, polyhedra),
        )
    _add_polyhedra_items(items, atoms, polyhedra, cam, pivot, palette)
    miller_labels = _add_miller_queue_items(
        items, atoms, miller_planes, cam, pivot
    )
    items.sort(key=lambda entry: entry[0])
    return items, miller_labels


def _finalize_3d_viewport(
    ax: maxes.Axes,
    atoms: structure.AtomsData,
    cam: geometry.Camera,
    pivot: np.ndarray,
    all_pts: np.ndarray,
    margin: float,
    standalone_axes: str,
    front_z: float,
) -> None:
    """Applies viewport bounds and draws standalone crystal axes when active."""
    x_lo = float(np.min(all_pts[:, 0])) - margin
    x_hi = float(np.max(all_pts[:, 0])) + margin
    y_lo = float(np.min(all_pts[:, 1])) - margin
    y_hi = float(np.max(all_pts[:, 1])) + margin
    x_span = max(x_hi - x_lo, 2.0)
    if standalone_axes in ("bottom_left", "top_left"):
        x_lo -= 0.28 * x_span
    elif standalone_axes in ("bottom_right", "top_right"):
        x_hi += 0.28 * x_span
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_aspect("equal")
    ax.set_axis_off()

    if atoms.cell is not None:
        if standalone_axes in _CORNER_AXES_LOCS:
            primitives.draw_crystal_axes(
                ax,
                atoms.cell,
                cam,
                location=standalone_axes,
                zorder=front_z + 2.5,
            )
        elif standalone_axes == "origin":
            primitives.draw_origin_tripod(
                ax,
                atoms.cell,
                cam,
                center=pivot,
                zorder=front_z + 2.5,
            )


_FRONT_CELL_EDGES: tuple[tuple[int, int], ...] = (
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


def _add_front_cell_edges(
    items: list[tuple[float, str, object]],
    atoms: structure.AtomsData,
    cam: geometry.Camera,
    pivot: np.ndarray,
    cell_lw: float,
) -> None:
    """Adds depth-sorted front cell edges so corner/edge atoms sit cleanly."""
    if atoms.cell is None:
        return
    corners_2d, c_depth, farthest_idx = geometry.cell_corners_projected(
        atoms.cell, cam, center=pivot
    )
    for i_idx, j_idx in _FRONT_CELL_EDGES:
        if farthest_idx in (i_idx, j_idx):
            continue
        edge_depth = 0.5 * float(c_depth[i_idx] + c_depth[j_idx]) - 0.05
        items.append(
            (
                edge_depth,
                "cell_edge",
                (corners_2d[i_idx], corners_2d[j_idx], cell_lw),
            )
        )


def _render_prepared_3d(
    ax: maxes.Axes,
    atoms: structure.AtomsData,
    cam: geometry.Camera,
    scene: tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list[str],
        list[float],
        float,
        tuple[bool, str | None, str, str],
        tuple[float, float, float, float],
    ],
    opts: tuple[str, str | Mapping[str, str], bool, bool, float],
    polyhedra: tuple[str, str | None]
    | Sequence[bonding.CoordinationPolyhedron]
    | None,
    miller_planes: Sequence[tuple[int, int, int]] | None,
) -> None:
    """Executes cell backdrop, depth-sorted painter pass, and front cell."""
    pivot, xy, depth, all_pts, colors = scene[:5]
    radii, margin, cell_cfg, widths = scene[5:]
    draw_box, base_col, box_axes, standalone_axes = cell_cfg

    if draw_box and atoms.cell is not None:
        primitives.draw_unit_cell_box(
            ax,
            atoms.cell,
            cam,
            center=pivot,
            linewidth=widths[3],
            draw_back=True,
            draw_front=False,
            base_plane_color=base_col,
            cell_axes=box_axes,
        )

    items, miller_labels = _build_3d_painter_queue(
        atoms,
        cam,
        pivot,
        xy,
        depth,
        colors,
        radii,
        opts,
        polyhedra,
        miller_planes,
    )
    if draw_box and atoms.cell is not None:
        _add_front_cell_edges(items, atoms, cam, pivot, widths[3])
        items.sort(key=lambda entry: entry[0])
    state = (
        atoms,
        xy,
        colors,
        radii,
        widths[0],
        widths[1],
        widths[2],
        opts[2],
        opts[0],
    )
    for rank, (_, kind, payload) in enumerate(items):
        _draw_queue_item(ax, kind, payload, 2.0 + 0.02 * rank, state)

    front_z = 2.0 + 0.02 * (len(items) + 4)
    if draw_box and atoms.cell is not None:
        primitives.draw_unit_cell_box(
            ax,
            atoms.cell,
            cam,
            center=pivot,
            linewidth=widths[3],
            draw_back=False,
            draw_front=True,
            draw_front_edges=False,
            base_plane_color=None,
            cell_axes=box_axes,
            front_zorder=front_z,
        )
    if miller_labels:
        _draw_miller_labels(ax, miller_labels, zorder=front_z + 1.5)
    _finalize_3d_viewport(
        ax, atoms, cam, pivot, all_pts, margin, standalone_axes, front_z
    )
