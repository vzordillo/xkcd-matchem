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
    # Default ball_and_stick keeps light elements compact and heavy ones crisp.
    r_cov = elements.get_covalent_radius(symbol)
    return max(0.20, min(0.62, 0.14 + 0.28 * r_cov)) * scale


def _add_bond_queue_items(
    items: list[tuple[float, str, object]],
    atoms: structure.AtomsData,
    bonds: Sequence[structure.Bond],
    cam: geometry.Camera,
    pivot: np.ndarray,
    depth: np.ndarray,
    colors: Sequence[str],
) -> None:
    """Appends depth-sorted bond and periodic half-stub draw entries."""
    for bond in bonds:
        i_idx, j_idx = bond.i, bond.j
        if i_idx >= len(atoms) or j_idx >= len(atoms):
            continue
        if bond.shift == (0, 0, 0) or atoms.cell is None:
            mid_d = 0.5 * (float(depth[i_idx]) + float(depth[j_idx])) - 1e-3
            items.append((mid_d, "bond", (i_idx, j_idx, bond.order)))
            continue
        shift_vec = (
            bond.shift[0] * atoms.cell[0]
            + bond.shift[1] * atoms.cell[1]
            + bond.shift[2] * atoms.cell[2]
        )
        p_i = atoms.positions[i_idx]
        p_j_img = atoms.positions[j_idx] + shift_vec
        pts, d_s = geometry.project_points(
            np.vstack(
                [
                    p_i,
                    p_i + 0.44 * (p_j_img - p_i),
                    atoms.positions[j_idx],
                    atoms.positions[j_idx]
                    + 0.44 * (p_i - shift_vec - atoms.positions[j_idx]),
                ]
            ),
            cam,
            center=pivot,
        )
        items.append(
            (
                float(0.5 * (d_s[0] + d_s[1])) - 1e-3,
                "stub",
                (pts[0], pts[1], colors[i_idx], colors[j_idx], bond.order),
            )
        )
        items.append(
            (
                float(0.5 * (d_s[2] + d_s[3])) - 1e-3,
                "stub",
                (pts[2], pts[3], colors[j_idx], colors[i_idx], bond.order),
            )
        )


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
        bool,
        str,
        str,
    ],
) -> None:
    """Dispatches a single painter-sorted sphere, bond, or polyhedron."""
    atoms, xy, colors, radii, bond_w, show_labels, style, palette = state
    if kind == "atom":
        idx = int(payload)  # type: ignore[arg-type]
        primitives.draw_atom_sphere(
            ax,
            float(xy[idx, 0]),
            float(xy[idx, 1]),
            radius=radii[idx],
            facecolor=colors[idx],
            label=atoms.symbols[idx] if show_labels else None,
            highlight=style != "stick",
            zorder=z_layer,
        )
    elif kind == "bond":
        i_idx, j_idx, order = payload  # type: ignore[misc]
        primitives.draw_two_tone_bond(
            ax,
            xy[i_idx],
            xy[j_idx],
            colors[i_idx],
            colors[j_idx],
            width=bond_w,
            order=order,
            zorder=z_layer - 0.02,
        )
    elif kind == "stub":
        p_a, p_b, col_a, col_b, order = payload  # type: ignore[misc]
        primitives.draw_two_tone_bond(
            ax,
            p_a,
            p_b,
            col_a,
            col_b,
            width=bond_w * 0.88,
            order=order,
            zorder=z_layer - 0.02,
        )
    elif kind == "poly":
        poly_obj, p_xy, p_depth = payload  # type: ignore[misc]
        c_col = palettes.get_element_color(
            poly_obj.center_symbol, scheme=palette
        )
        primitives.draw_polyhedron_faces(
            ax,
            p_xy,
            poly_obj.faces,
            p_depth,
            facecolor=c_col,
            alpha=0.28,
            base_zorder=z_layer - 0.04,
        )


def _render_cell_and_planes(
    ax: maxes.Axes,
    atoms: structure.AtomsData,
    cam: geometry.Camera,
    pivot: np.ndarray,
    show_cell: bool,
    miller_planes: Sequence[tuple[int, int, int]] | None,
    bounds_pts: list[np.ndarray],
) -> None:
    """Draws optional periodic unit-cell frame and crystallographic planes."""
    if show_cell and atoms.cell is not None:
        cell_2d = primitives.draw_unit_cell_box(
            ax, atoms.cell, cam, center=pivot
        )
        if cell_2d:
            bounds_pts.append(np.vstack(cell_2d))
    if miller_planes and atoms.cell is not None:
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


def _add_polyhedra_items(
    items: list[tuple[float, str, object]],
    atoms: structure.AtomsData,
    polyhedra: tuple[str, str | None]
    | Sequence[bonding.CoordinationPolyhedron]
    | None,
    cam: geometry.Camera,
    pivot: np.ndarray,
    depth: np.ndarray,
) -> None:
    """Appends depth-sorted coordination polyhedra into the draw queue."""
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
        items.append(
            (
                float(depth[poly.center_index]) + 1e-4,
                "poly",
                (poly, p_xy, p_depth),
            )
        )


def _flush_draw_queue(
    ax: maxes.Axes,
    items: list[tuple[float, str, object]],
    state: tuple[
        structure.AtomsData,
        np.ndarray,
        list[str],
        list[float],
        float,
        bool,
        str,
        str,
    ],
) -> None:
    """Sorts draw items back-to-front and paints each artist in order."""
    items.sort(key=lambda entry: entry[0])
    min_d = items[0][0] if items else 0.0
    span_d = max((items[-1][0] - min_d) if items else 1.0, 1e-6)
    for item_d, kind, payload in items:
        _draw_queue_item(
            ax, kind, payload, 2.0 + 8.0 * (item_d - min_d) / span_d, state
        )


def render_structure_3d(
    atoms: structure.AtomsData,
    ax: maxes.Axes | None = None,
    camera: geometry.Camera | None = None,
    style: str = "ball_and_stick",
    palette: str = "xkcd",
    show_cell: bool = True,
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
    """Renders a 3D molecule or crystal in hand-drawn depth-sorted xkcd style.

    Args:
      atoms: Input ``AtomsData`` structure.
      ax: Optional target Matplotlib ``Axes``. Created if ``None``.
      camera: Viewing ``Camera`` configuration. Defaults to isometric view.
      style: Visual representation: ``"ball_and_stick"``, ``"spacefill"``,
        ``"stick"``, or ``"polyhedra"``.
      palette: Element coloring scheme (``"xkcd"``, ``"jmol"``,
        ``"colorblind"``).
      show_cell: Whether to render the periodic unit cell box when present.
      show_labels: Whether to draw element symbol labels inside atom spheres.
      show_bonds: Whether to draw chemical bonds (ignored in ``spacefill``).
      radius_scale: Global multiplier for atom sphere radii.
      bond_scale: Covalent-radius factor used when inferring missing bonds.
      polyhedra: Either ``(center_symbol, ligand_symbol)`` tuple for automatic
        coordination detection or a sequence of ``CoordinationPolyhedron``.
      miller_planes: Optional crystallographic planes ``[(h, k, l), ...]``.
      pad: Viewport margin in data units around all projected geometry.

    Returns:
      Matplotlib ``Axes`` containing the rendered structure.

    Raises:
      StructureError: If ``style`` is unsupported.
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

    cam = geometry.Camera() if camera is None else camera
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
    items: list[tuple[float, str, object]] = [
        (float(depth[idx]), "atom", idx) for idx in range(len(atoms))
    ]
    if show_bonds and style != "spacefill":
        bonds = list(atoms.bonds) or (
            bonding.infer_bonds(atoms, scale=bond_scale)
            if len(atoms) <= 600
            else []
        )
        _add_bond_queue_items(items, atoms, bonds, cam, pivot, depth, colors)
    _add_polyhedra_items(items, atoms, polyhedra, cam, pivot, depth)

    _flush_draw_queue(
        ax,
        items,
        (
            atoms,
            xy,
            colors,
            radii,
            5.2 if style == "stick" else 3.8,
            show_labels,
            style,
            palette,
        ),
    )

    bounds_pts: list[np.ndarray] = [xy]
    _render_cell_and_planes(
        ax, atoms, cam, pivot, show_cell, miller_planes, bounds_pts
    )
    all_pts = np.vstack(bounds_pts)
    margin = max(radii, default=0.4) + pad
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
    return ax
