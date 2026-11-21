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


def _parse_cell_mode(
    show_cell: bool | str,
    has_cell: bool,
) -> tuple[bool, str | None, str]:
    """Resolves cell mode ``(enabled, base_plane_color, axes_mode)``."""
    if not has_cell or show_cell is False:
        return False, None, "none"
    if isinstance(show_cell, str):
        key = show_cell.strip().lower()
        if key in ("none", "off", "false"):
            return False, None, "none"
        if key == "wireframe":
            return True, None, "edges"
        if key == "clean":
            return True, None, "none"
        if key == "tripod":
            return True, None, "tripod"
    return True, palettes.VACUUM_TINT, "edges"


def _polyhedra_pair_filter(
    style: str,
    polyhedra: tuple[str, str | None]
    | Sequence[bonding.CoordinationPolyhedron]
    | None,
) -> tuple[str | None, str | None] | None:
    """Returns center/ligand pair to suppress redundant inner bonds."""
    if style != "polyhedra" or polyhedra is None:
        return None
    if (
        isinstance(polyhedra, tuple)
        and len(polyhedra) == 2
        and isinstance(polyhedra[0], str)
    ):
        return polyhedra[0], polyhedra[1]
    return None


def _add_bond_queue_items(
    items: list[tuple[float, str, object]],
    atoms: structure.AtomsData,
    bonds: Sequence[structure.Bond],
    cam: geometry.Camera,
    pivot: np.ndarray,
    xy_and_depth: tuple[np.ndarray, np.ndarray],
    colors: Sequence[str],
    skip_pair: tuple[str | None, str | None] | None = None,
) -> None:
    """Appends two depth-sorted half-bonds per interior or periodic bond."""
    xy, depth = xy_and_depth
    for bond in bonds:
        i_idx, j_idx = bond.i, bond.j
        if i_idx >= len(atoms) or j_idx >= len(atoms):
            continue
        if skip_pair is not None:
            c_sym, l_sym = skip_pair
            s_i, s_j = atoms.symbols[i_idx], atoms.symbols[j_idx]
            match_fwd = s_i == c_sym and (l_sym is None or s_j == l_sym)
            match_rev = s_j == c_sym and (l_sym is None or s_i == l_sym)
            if match_fwd or match_rev:
                continue
        if bond.shift == (0, 0, 0) or atoms.cell is None:
            mid_xy = 0.5 * (xy[i_idx] + xy[j_idx])
            items.append(
                (
                    float(depth[i_idx]) - 0.06,
                    "half",
                    (xy[i_idx], mid_xy, colors[i_idx], bond.order),
                )
            )
            items.append(
                (
                    float(depth[j_idx]) - 0.06,
                    "half",
                    (xy[j_idx], mid_xy, colors[j_idx], bond.order),
                )
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
        bool,
        str,
    ],
) -> None:
    """Paints a single depth-ranked atom sphere, half-bond, or poly facet."""
    atoms, xy, colors, radii, bond_w, atom_lw, show_labels, style = state
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
    elif kind == "half":
        p_a, p_b, col, order = payload  # type: ignore[misc]
        primitives.draw_half_bond(
            ax,
            p_a,
            p_b,
            color=col,
            width=bond_w,
            order=order,
            zorder=z_layer,
        )
    elif kind == "tri":
        tri_xy, c_col = payload  # type: ignore[misc]
        primitives.draw_polyhedron_triangle(
            ax,
            tri_xy,
            facecolor=c_col,
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


def _draw_miller_planes(
    ax: maxes.Axes,
    atoms: structure.AtomsData,
    cam: geometry.Camera,
    pivot: np.ndarray,
    miller_planes: Sequence[tuple[int, int, int]] | None,
) -> None:
    """Draws translucent crystallographic Miller slices when requested."""
    if not miller_planes or atoms.cell is None:
        return
    plane_palette = palettes.get_palette("okabe_ito")
    for idx_m, hkl in enumerate(miller_planes):
        col = plane_palette[(idx_m + 2) % len(plane_palette)]
        primitives.draw_miller_plane(
            ax,
            atoms.cell,
            hkl,
            cam,
            center=pivot,
            facecolor=col,
            edgecolor=col,
        )


def _prepare_scene_geometry(
    atoms: structure.AtomsData,
    ax: maxes.Axes,
    cam: geometry.Camera,
    style: str,
    palette: str | Mapping[str, str],
    show_cell: bool | str,
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
    tuple[bool, str | None, str],
    tuple[float, float, float],
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
    cell_cfg = _parse_cell_mode(show_cell, atoms.cell is not None)
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
        (bond_w, atom_lw, cell_lw),
    )


def render_structure_3d(
    atoms: structure.AtomsData,
    ax: maxes.Axes | None = None,
    camera: geometry.Camera | str | None = None,
    style: str = "ball_and_stick",
    palette: str | Mapping[str, str] = "xkcd",
    show_cell: bool | str = True,
    show_labels: bool = False,
    show_bonds: bool = True,
    radius_scale: float = 1.0,
    bond_scale: float = 1.20,
    polyhedra: tuple[str, str | None]
    | Sequence[bonding.CoordinationPolyhedron]
    | None = None,
    miller_planes: Sequence[tuple[int, int, int]] | None = None,
    pad: float = 0.85,
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
      show_cell: Cell display mode: ``True`` (base plane + edges + ``a,b,c``),
        ``"wireframe"``, ``"clean"``, ``"tripod"``, or ``False``.
      show_labels: Whether to print element symbols inside atom spheres.
      show_bonds: Whether to draw bonds (ignored in ``"spacefill"`` mode).
      radius_scale: Global multiplier for atom sphere radii.
      bond_scale: Covalent-radius factor used when inferring missing bonds.
      polyhedra: ``(center_symbol, ligand_symbol)`` pair or sequence of
        ``CoordinationPolyhedron``.
      miller_planes: Optional crystallographic planes ``[(h, k, l), ...]``.
      pad: Viewport margin in Angstrom around all projected geometry.

    Returns:
      Matplotlib ``Axes`` containing the rendered structure.

    Raises:
      StructureError: If ``style`` or ``camera`` preset is unsupported.
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
    scene = _prepare_scene_geometry(
        atoms, ax, cam, style, palette, show_cell, radius_scale, pad
    )
    opts = (style, palette, show_labels, show_bonds, bond_scale)
    _render_prepared_3d(ax, atoms, cam, scene, opts, polyhedra, miller_planes)
    return ax


def _build_3d_painter_queue(
    atoms: structure.AtomsData,
    cam: geometry.Camera,
    pivot: np.ndarray,
    xy: np.ndarray,
    depth: np.ndarray,
    colors: list[str],
    opts: tuple[str, str | Mapping[str, str], bool, bool, float],
    polyhedra: tuple[str, str | None]
    | Sequence[bonding.CoordinationPolyhedron]
    | None,
) -> list[tuple[float, str, object]]:
    """Assembles and depth-sorts atoms, half-bonds, and polyhedra faces."""
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
            skip_pair=_polyhedra_pair_filter(style, polyhedra),
        )
    _add_polyhedra_items(items, atoms, polyhedra, cam, pivot, palette)
    items.sort(key=lambda entry: entry[0])
    return items


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
        tuple[bool, str | None, str],
        tuple[float, float, float],
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

    if cell_cfg[0] and atoms.cell is not None:
        primitives.draw_unit_cell_box(
            ax,
            atoms.cell,
            cam,
            center=pivot,
            linewidth=widths[2],
            draw_back=True,
            draw_front=False,
            base_plane_color=cell_cfg[1],
            cell_axes=cell_cfg[2],
        )

    items = _build_3d_painter_queue(
        atoms, cam, pivot, xy, depth, colors, opts, polyhedra
    )
    state = (atoms, xy, colors, radii, widths[0], widths[1], opts[2], opts[0])
    for rank, (_, kind, payload) in enumerate(items):
        _draw_queue_item(ax, kind, payload, 2.0 + 0.02 * rank, state)

    _draw_miller_planes(ax, atoms, cam, pivot, miller_planes)
    if cell_cfg[0] and atoms.cell is not None:
        primitives.draw_unit_cell_box(
            ax,
            atoms.cell,
            cam,
            center=pivot,
            linewidth=widths[2],
            draw_back=False,
            draw_front=True,
            base_plane_color=None,
            cell_axes=cell_cfg[2],
            front_zorder=2.0 + 0.02 * (len(items) + 4),
        )

    ax.set_xlim(
        float(np.min(all_pts[:, 0])) - margin,
        float(np.max(all_pts[:, 0])) + margin,
    )
    ax.set_ylim(
        float(np.min(all_pts[:, 1])) - margin,
        float(np.max(all_pts[:, 1])) + margin,
    )
    ax.set_aspect("equal")
    ax.set_axis_off()
