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
"""Painter's-algorithm depth queue assembly for 3D structure rendering."""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
import warnings

from matplotlib import pyplot as plt
import numpy as np

from xkcd_matchem.core import bonding
from xkcd_matchem.core import geometry
from xkcd_matchem.core import structure
from xkcd_matchem.render import _structure3d_geometry as s3g
from xkcd_matchem.render import primitives
from xkcd_matchem.style import palettes


def polyhedra_skip_spec(
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
    poly_seq = [
        p
        for p in polyhedra
        if isinstance(p, bonding.CoordinationPolyhedron)
    ]
    c_syms = {p.center_symbol for p in poly_seq}
    c_idxs = {p.center_index for p in poly_seq}
    return (c_syms, None, c_idxs) if c_syms else None


def should_skip_poly_bond(
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


def build_interior_bond_item(
    atoms: structure.AtomsData,
    bond: structure.Bond,
    xy: np.ndarray,
    depth: np.ndarray,
    colors: Sequence[str],
    radii: Sequence[float],
) -> s3g.PainterItem:
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
    return s3g.PainterItem(
        depth=bond_z,
        kind="bond",
        payload=s3g.BondPayload(
            p_start=p_start,
            p_end=p_front,
            col_start=colors[i_back],
            col_end=colors[i_front],
            order=bond.order,
            split_frac=split_frac,
        ),
    )


def add_bond_queue_items(
    items: list[s3g.PainterItem],
    atoms: structure.AtomsData,
    bonds: Sequence[structure.Bond],
    cam: geometry.Camera,
    scene: s3g.SceneGeometry3D,
    skip_spec: tuple[set[str], set[str] | None, set[int]] | None = None,
) -> None:
    """Appends unified two-tone interior bonds and periodic edge stubs."""
    for bond in bonds:
        i_idx, j_idx = bond.i, bond.j
        if i_idx >= len(atoms) or j_idx >= len(atoms):
            continue
        if should_skip_poly_bond(i_idx, j_idx, atoms, skip_spec):
            continue
        if bond.shift == (0, 0, 0) or atoms.cell is None:
            items.append(
                build_interior_bond_item(
                    atoms,
                    bond,
                    scene.xy,
                    scene.depth,
                    scene.colors,
                    scene.radii,
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
            center=scene.pivot,
            span=scene.proj_span,
        )
        items.append(
            s3g.PainterItem(
                depth=float(scene.depth[i_idx]) - 0.06,
                kind="half",
                payload=s3g.HalfBondPayload(
                    p_start=scene.xy[i_idx],
                    p_end=pts[0],
                    color=scene.colors[i_idx],
                    order=bond.order,
                ),
            )
        )
        items.append(
            s3g.PainterItem(
                depth=float(scene.depth[j_idx]) - 0.06,
                kind="half",
                payload=s3g.HalfBondPayload(
                    p_start=scene.xy[j_idx],
                    p_end=pts[1],
                    color=scene.colors[j_idx],
                    order=bond.order,
                ),
            )
        )


def add_hbond_queue_items(
    items: list[s3g.PainterItem],
    atoms: structure.AtomsData,
    cam: geometry.Camera,
    scene: s3g.SceneGeometry3D,
    hbond_color: str | None,
    *,
    bg_color: str | None = None,
) -> None:
    """Appends depth-sorted dashed ``H...A`` hydrogen-bond guide strokes."""
    hbonds = bonding.infer_hbonds(atoms, use_pbc=atoms.cell is not None)
    if not hbonds:
        return
    bg = (
        str(plt.rcParams.get("axes.facecolor", palettes.PAPER))
        if bg_color is None
        else bg_color
    )
    default_col = "#56B4E9" if primitives.is_dark_color(bg) else "#0072B2"
    col = default_col if hbond_color is None else hbond_color
    lw = float(np.clip(0.38 * scene.bond_w, 0.95, 1.75))
    for hb in hbonds:
        h_idx, a_idx = hb.hydrogen_index, hb.acceptor_index
        if h_idx >= len(atoms) or a_idx >= len(atoms):
            continue
        if hb.shift == (0, 0, 0) or atoms.cell is None:
            depth_hb = 0.5 * float(scene.depth[h_idx] + scene.depth[a_idx])
            items.append(
                s3g.PainterItem(
                    depth=depth_hb - 0.05,
                    kind="hbond",
                    payload=s3g.HBondPayload(
                        p_start=scene.xy[h_idx],
                        p_end=scene.xy[a_idx],
                        color=col,
                        linewidth=lw,
                    ),
                )
            )
            continue
        shift_vec = (
            hb.shift[0] * atoms.cell[0]
            + hb.shift[1] * atoms.cell[1]
            + hb.shift[2] * atoms.cell[2]
        )
        p_h = atoms.positions[h_idx]
        p_a = atoms.positions[a_idx]
        stubs, _ = geometry.project_points(
            np.vstack(
                [
                    p_h + 0.45 * (p_a + shift_vec - p_h),
                    p_a + 0.45 * (p_h - shift_vec - p_a),
                ]
            ),
            cam,
            center=scene.pivot,
            span=scene.proj_span,
        )
        items.append(
            s3g.PainterItem(
                depth=float(scene.depth[h_idx]) - 0.07,
                kind="hbond",
                payload=s3g.HBondPayload(
                    p_start=scene.xy[h_idx],
                    p_end=stubs[0],
                    color=col,
                    linewidth=lw,
                ),
            )
        )
        items.append(
            s3g.PainterItem(
                depth=float(scene.depth[a_idx]) - 0.07,
                kind="hbond",
                payload=s3g.HBondPayload(
                    p_start=scene.xy[a_idx],
                    p_end=stubs[1],
                    color=col,
                    linewidth=lw,
                ),
            )
        )


def add_polyhedra_items(
    items: list[s3g.PainterItem],
    atoms: structure.AtomsData,
    polyhedra: tuple[str, str | None]
    | Sequence[bonding.CoordinationPolyhedron]
    | None,
    cam: geometry.Camera,
    scene: s3g.SceneGeometry3D,
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
        poly_list = [
            p
            for p in polyhedra
            if isinstance(p, bonding.CoordinationPolyhedron)
        ]
    for poly in poly_list:
        p_xy, p_depth = geometry.project_points(
            poly.vertices_3d, cam, center=scene.pivot, span=scene.proj_span
        )
        c_col = palettes.get_element_color(poly.center_symbol, scheme=palette)
        for i0, i1, i2 in poly.faces:
            tri_xy = np.vstack([p_xy[i0], p_xy[i1], p_xy[i2]])
            mean_d = float((p_depth[i0] + p_depth[i1] + p_depth[i2]) / 3.0)
            items.append(
                s3g.PainterItem(
                    depth=mean_d - 0.03,
                    kind="tri",
                    payload=s3g.TrianglePayload(
                        verts_2d=tri_xy, facecolor=c_col
                    ),
                )
            )


def add_vector_items(
    items: list[s3g.PainterItem],
    atoms: structure.AtomsData,
    vec_arr: np.ndarray | None,
    cam: geometry.Camera,
    scene: s3g.SceneGeometry3D,
    opts: s3g.RenderOptions3D,
    *,
    bg_color: str | None = None,
) -> None:
    """Enqueues depth-sorted 3D vector arrows for spins, forces, or modes."""
    if vec_arr is None or len(atoms) == 0:
        return
    bg = (
        str(plt.rcParams.get("axes.facecolor", palettes.PAPER))
        if bg_color is None
        else bg_color
    )
    neg_spin_col = "#56B4E9" if primitives.is_dark_color(bg) else "#0072B2"
    scaled = vec_arr * float(opts.vector_scale)
    tails_3d = atoms.positions - 0.25 * scaled
    tips_3d = atoms.positions + 0.75 * scaled
    tails_2d, _ = geometry.project_points(
        tails_3d, cam, center=scene.pivot, span=scene.proj_span
    )
    tips_2d, _ = geometry.project_points(
        tips_3d, cam, center=scene.pivot, span=scene.proj_span
    )
    for idx in range(len(atoms)):
        if float(np.linalg.norm(scaled[idx])) < 1e-5:
            continue
        col = opts.vector_color
        if opts.vector_color.lower() == "spin" and vec_arr[idx, 2] < 0.0:
            col = neg_spin_col
        elif opts.vector_color.lower() == "spin":
            col = "#D55E00"
        items.append(
            s3g.PainterItem(
                depth=float(scene.depth[idx]) + 0.015,
                kind="vec",
                payload=s3g.VectorPayload(
                    tail_2d=tails_2d[idx], tip_2d=tips_2d[idx], color=col
                ),
            )
        )


def miller_plane_colors(
    index: int,
    *,
    bg_color: str | None = None,
) -> tuple[str, str]:
    """Returns ``(facecolor, edgecolor)`` pair for the ``index``-th plane."""
    fill_cycle = ("#3B82F6", "#10B981", "#F59E0B", "#EC4899", "#8B5CF6")
    edge_cycle = ("#1D4ED8", "#047857", "#B45309", "#BE185D", "#6D28D9")
    face_col = fill_cycle[index % len(fill_cycle)]
    edge_col = edge_cycle[index % len(edge_cycle)]
    if bg_color is not None and primitives.is_dark_color(bg_color):
        edge_col = primitives.ensure_readable_text_color(edge_col, bg_color)
    return face_col, edge_col


def add_miller_queue_items(
    items: list[s3g.PainterItem],
    atoms: structure.AtomsData,
    miller_planes: Sequence[tuple[int, int, int]] | None,
    cam: geometry.Camera,
    scene: s3g.SceneGeometry3D,
    *,
    bg_color: str | None = None,
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
            poly_3d, cam, center=scene.pivot, span=scene.proj_span
        )
        face_col, edge_col = miller_plane_colors(idx_m, bg_color=bg_color)
        items.append(
            s3g.PainterItem(
                depth=float(np.mean(poly_depth)) - 0.01,
                kind="miller",
                payload=s3g.MillerPayload(
                    hkl=hkl,
                    facecolor=face_col,
                    edgecolor=edge_col,
                    camera=cam,
                    pivot=scene.pivot,
                    proj_span=scene.proj_span,
                ),
            )
        )
        labels_meta.append((hkl, np.mean(poly_2d, axis=0), edge_col))
    return labels_meta


def add_front_cell_edges(
    items: list[s3g.PainterItem],
    atoms: structure.AtomsData,
    cam: geometry.Camera,
    scene: s3g.SceneGeometry3D,
) -> None:
    """Adds depth-sorted front cell edges so corner/edge atoms sit cleanly."""
    if atoms.cell is None:
        return
    corners_2d, c_depth, farthest_idx = geometry.cell_corners_projected(
        atoms.cell, cam, center=scene.pivot, span=scene.proj_span
    )
    for i_idx, j_idx in s3g.FRONT_CELL_EDGES:
        if farthest_idx in (i_idx, j_idx):
            continue
        edge_depth = 0.5 * float(c_depth[i_idx] + c_depth[j_idx]) - 0.05
        items.append(
            s3g.PainterItem(
                depth=edge_depth,
                kind="cell_edge",
                payload=s3g.CellEdgePayload(
                    p0=corners_2d[i_idx],
                    p1=corners_2d[j_idx],
                    linewidth=scene.cell_lw,
                ),
            )
        )


def build_3d_painter_queue(
    atoms: structure.AtomsData,
    cam: geometry.Camera,
    scene: s3g.SceneGeometry3D,
    opts: s3g.RenderOptions3D,
    polyhedra: tuple[str, str | None]
    | Sequence[bonding.CoordinationPolyhedron]
    | None,
    miller_planes: Sequence[tuple[int, int, int]] | None,
    vec_arr: np.ndarray | None,
    *,
    bg_color: str | None = None,
) -> tuple[
    list[s3g.PainterItem],
    list[tuple[tuple[int, int, int], np.ndarray, str]],
]:
    """Assembles and depth-sorts atoms, bonds, polyhedra, vectors, & slices."""
    items: list[s3g.PainterItem] = [
        s3g.PainterItem(depth=float(scene.depth[idx]), kind="atom", payload=idx)
        for idx in range(len(atoms))
    ]
    if opts.show_bonds and opts.style != "spacefill":
        if atoms.bonds:
            bonds = list(atoms.bonds)
        elif len(atoms) <= s3g.MAX_AUTOBOND_ATOMS:
            bonds = bonding.infer_bonds(atoms, scale=opts.bond_scale)
        else:
            bonds = []
            warnings.warn(
                f"Structure has {len(atoms)} atoms (> {s3g.MAX_AUTOBOND_ATOMS})"
                "; skipping automatic bond inference. Pass show_bonds=True or "
                "precomputed atoms.bonds to draw bonds.",
                UserWarning,
                stacklevel=4,
            )
        add_bond_queue_items(
            items,
            atoms,
            bonds,
            cam,
            scene,
            skip_spec=polyhedra_skip_spec(opts.style, polyhedra),
        )
    if opts.show_hbonds and opts.style != "spacefill":
        add_hbond_queue_items(
            items, atoms, cam, scene, opts.hbond_color, bg_color=bg_color
        )
    add_polyhedra_items(items, atoms, polyhedra, cam, scene, opts.palette)
    add_vector_items(items, atoms, vec_arr, cam, scene, opts, bg_color=bg_color)
    miller_labels = add_miller_queue_items(
        items, atoms, miller_planes, cam, scene, bg_color=bg_color
    )
    items.sort(key=lambda entry: entry.depth)
    return items, miller_labels
