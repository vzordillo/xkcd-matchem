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
import dataclasses
from typing import Any
from typing import cast
import warnings

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
_CORNER_AXES_LOCS = ("bottom_left", "bottom_right", "top_left", "top_right")
_MAX_AUTOBOND_ATOMS = 2000
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


@dataclasses.dataclass(frozen=True)
class CellConfig3D:
    """Resolved unit-cell box and crystallographic axes configuration."""

    draw_box: bool
    base_plane_color: str | None
    box_axes: str
    standalone_axes: str


@dataclasses.dataclass(frozen=True)
class SceneGeometry3D:
    """Precomputed 2D projection, colors, radii, and stroke geometry."""

    pivot: np.ndarray
    proj_span: float
    xy: np.ndarray
    depth: np.ndarray
    all_pts: np.ndarray
    colors: list[str]
    radii: list[float]
    margin: float
    cell_cfg: CellConfig3D
    bond_w: float
    bond_off: float
    atom_lw: float
    cell_lw: float


@dataclasses.dataclass(frozen=True)
class RenderOptions3D:
    """High-level rendering options passed through the 3D pipeline."""

    style: str
    palette: str | Mapping[str, str]
    show_labels: bool | str
    show_bonds: bool
    bond_scale: float
    vector_scale: float = 1.0
    vector_color: str = "#D55E00"


@dataclasses.dataclass(frozen=True)
class RenderState3D:
    """Immutable shared context passed when painting sorted 3D queue items."""

    atoms: structure.AtomsData
    xy: np.ndarray
    colors: list[str]
    radii: list[float]
    bond_w: float
    bond_off: float
    atom_lw: float
    show_labels: bool | str
    style: str
    vector_color: str


@dataclasses.dataclass(frozen=True)
class BondPayload:
    """Painter item payload for a two-tone interior bond."""

    p_start: np.ndarray
    p_end: np.ndarray
    col_start: str
    col_end: str
    order: float
    split_frac: float


@dataclasses.dataclass(frozen=True)
class HalfBondPayload:
    """Painter item payload for a periodic boundary bond stub."""

    p_start: np.ndarray
    p_end: np.ndarray
    color: str
    order: float


@dataclasses.dataclass(frozen=True)
class TrianglePayload:
    """Painter item payload for a coordination polyhedron facet."""

    verts_2d: np.ndarray
    facecolor: str


@dataclasses.dataclass(frozen=True)
class MillerPayload:
    """Painter item payload for a crystallographic Miller slice."""

    hkl: tuple[int, int, int]
    facecolor: str
    edgecolor: str
    camera: geometry.Camera
    pivot: np.ndarray
    proj_span: float


@dataclasses.dataclass(frozen=True)
class CellEdgePayload:
    """Painter item payload for a front unit-cell wireframe edge."""

    p0: np.ndarray
    p1: np.ndarray
    linewidth: float


@dataclasses.dataclass(frozen=True)
class VectorPayload:
    """Painter item payload for an atomic spin, force, or velocity arrow."""

    tail_2d: np.ndarray
    tip_2d: np.ndarray
    color: str


@dataclasses.dataclass(frozen=True)
class PainterItem:
    """Depth-sorted 3D painter's-algorithm draw entry."""

    depth: float
    kind: str
    payload: (
        int
        | BondPayload
        | HalfBondPayload
        | TrianglePayload
        | MillerPayload
        | CellEdgePayload
        | VectorPayload
    )


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
) -> CellConfig3D:
    """Resolves cell box, plane tint, and crystallographic axes layout."""
    if not has_cell:
        return CellConfig3D(False, None, "none", "none")

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
    return CellConfig3D(draw_box, base_col, box_axes, standalone_axes)


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
    poly_seq = [
        p
        for p in polyhedra
        if isinstance(p, bonding.CoordinationPolyhedron)
    ]
    c_syms = {p.center_symbol for p in poly_seq}
    c_idxs = {p.center_index for p in poly_seq}
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
) -> PainterItem:
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
    return PainterItem(
        depth=bond_z,
        kind="bond",
        payload=BondPayload(
            p_start=p_start,
            p_end=p_front,
            col_start=colors[i_back],
            col_end=colors[i_front],
            order=bond.order,
            split_frac=split_frac,
        ),
    )


def _add_bond_queue_items(
    items: list[PainterItem],
    atoms: structure.AtomsData,
    bonds: Sequence[structure.Bond],
    cam: geometry.Camera,
    scene: SceneGeometry3D,
    skip_spec: tuple[set[str], set[str] | None, set[int]] | None = None,
) -> None:
    """Appends unified two-tone interior bonds and periodic edge stubs."""
    for bond in bonds:
        i_idx, j_idx = bond.i, bond.j
        if i_idx >= len(atoms) or j_idx >= len(atoms):
            continue
        if _should_skip_poly_bond(i_idx, j_idx, atoms, skip_spec):
            continue
        if bond.shift == (0, 0, 0) or atoms.cell is None:
            items.append(
                _build_interior_bond_item(
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
            PainterItem(
                depth=float(scene.depth[i_idx]) - 0.06,
                kind="half",
                payload=HalfBondPayload(
                    p_start=scene.xy[i_idx],
                    p_end=pts[0],
                    color=scene.colors[i_idx],
                    order=bond.order,
                ),
            )
        )
        items.append(
            PainterItem(
                depth=float(scene.depth[j_idx]) - 0.06,
                kind="half",
                payload=HalfBondPayload(
                    p_start=scene.xy[j_idx],
                    p_end=pts[1],
                    color=scene.colors[j_idx],
                    order=bond.order,
                ),
            )
        )


def _add_polyhedra_items(
    items: list[PainterItem],
    atoms: structure.AtomsData,
    polyhedra: tuple[str, str | None]
    | Sequence[bonding.CoordinationPolyhedron]
    | None,
    cam: geometry.Camera,
    scene: SceneGeometry3D,
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
                PainterItem(
                    depth=mean_d - 0.03,
                    kind="tri",
                    payload=TrianglePayload(verts_2d=tri_xy, facecolor=c_col),
                )
            )


def _resolve_vectors_3d(
    atoms: structure.AtomsData,
    vectors: (
        np.ndarray | Sequence[Sequence[float]] | Sequence[float] | str | None
    ),
) -> np.ndarray | None:
    """Resolves per-atom 3D vector array from keyword selector or array."""
    if vectors is None:
        return None
    if isinstance(vectors, str):
        key = vectors.strip().lower()
        if key == "forces":
            raw = atoms.forces
        elif key == "velocities":
            raw = atoms.velocities
        elif key in ("magmom", "magnetic_moments", "spins"):
            raw = atoms.magnetic_moments
        else:
            raise exceptions.StructureError(
                f"Unknown vector source {vectors!r}. Use 'forces', "
                "'velocities', 'magmom', or an explicit array."
            )
        if raw is None:
            return None
        arr = np.asarray(raw, dtype=np.float64)
    else:
        arr = np.asarray(vectors, dtype=np.float64)
    if arr.ndim == 1 and arr.shape[0] == len(atoms):
        out = np.zeros((len(atoms), 3), dtype=np.float64)
        out[:, 2] = arr
        return out
    if arr.shape != (len(atoms), 3):
        raise exceptions.StructureError(
            f"Expected vectors shape ({len(atoms)}, 3), got {arr.shape}."
        )
    return arr


def _add_vector_items(
    items: list[PainterItem],
    atoms: structure.AtomsData,
    vec_arr: np.ndarray | None,
    cam: geometry.Camera,
    scene: SceneGeometry3D,
    opts: RenderOptions3D,
) -> None:
    """Enqueues depth-sorted 3D vector arrows for spins, forces, or modes."""
    if vec_arr is None or len(atoms) == 0:
        return
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
            col = "#0072B2"
        elif opts.vector_color.lower() == "spin":
            col = "#D55E00"
        items.append(
            PainterItem(
                depth=float(scene.depth[idx]) + 0.015,
                kind="vec",
                payload=VectorPayload(
                    tail_2d=tails_2d[idx], tip_2d=tips_2d[idx], color=col
                ),
            )
        )


def _format_atom_label(
    atoms: structure.AtomsData,
    idx: int,
    show_labels: bool | str,
) -> str | None:
    """Resolves per-atom label for ``False``, ``True``, or ``'index'``."""
    if not show_labels:
        return None
    if isinstance(show_labels, str):
        key = show_labels.strip().lower()
        if key in ("none", "off", "false"):
            return None
        if key == "index":
            src_idxs = atoms.metadata.get("source_indices")
            site_idx = (
                int(src_idxs[idx])
                if isinstance(src_idxs, Sequence) and idx < len(src_idxs)
                else idx
            )
            return f"{atoms.symbols[idx]}{site_idx}"
    if atoms.atom_labels is not None:
        return str(atoms.atom_labels[idx])
    return atoms.symbols[idx]


def _draw_queue_item(
    ax: maxes.Axes,
    item: PainterItem,
    z_layer: float,
    state: RenderState3D,
) -> None:
    """Paints a single depth-ranked atom, bond, polyhedron, or vector item."""
    kind = item.kind
    payload = item.payload
    if kind == "atom" and isinstance(payload, int):
        primitives.draw_atom_sphere(
            ax,
            float(state.xy[payload, 0]),
            float(state.xy[payload, 1]),
            radius=state.radii[payload],
            facecolor=state.colors[payload],
            linewidth=state.atom_lw,
            label=_format_atom_label(state.atoms, payload, state.show_labels),
            highlight=state.style != "stick",
            zorder=z_layer,
        )
    elif kind == "bond" and isinstance(payload, BondPayload):
        primitives.draw_two_tone_bond(
            ax,
            payload.p_start,
            payload.p_end,
            c1=payload.col_start,
            c2=payload.col_end,
            width=state.bond_w,
            order=payload.order,
            offset_scale=state.bond_off,
            zorder=z_layer,
            split_frac=payload.split_frac,
        )
    elif kind == "half" and isinstance(payload, HalfBondPayload):
        primitives.draw_half_bond(
            ax,
            payload.p_start,
            payload.p_end,
            color=payload.color,
            width=state.bond_w,
            order=payload.order,
            offset_scale=state.bond_off,
            zorder=z_layer,
        )
    elif kind == "tri" and isinstance(payload, TrianglePayload):
        primitives.draw_polyhedron_triangle(
            ax,
            payload.verts_2d,
            facecolor=payload.facecolor,
            zorder=z_layer,
        )
    elif (
        kind == "miller"
        and isinstance(payload, MillerPayload)
        and state.atoms.cell is not None
    ):
        primitives.draw_miller_plane(
            ax,
            state.atoms.cell,
            payload.hkl,
            payload.camera,
            center=payload.pivot,
            facecolor=payload.facecolor,
            edgecolor=payload.edgecolor,
            show_label=False,
            zorder=z_layer,
            span=payload.proj_span,
        )
    elif kind == "cell_edge" and isinstance(payload, CellEdgePayload):
        ax.plot(
            [float(payload.p0[0]), float(payload.p1[0])],
            [float(payload.p0[1]), float(payload.p1[1])],
            color=str(plt.rcParams.get("axes.edgecolor", palettes.INK)),
            linestyle="-",
            linewidth=payload.linewidth,
            solid_capstyle="round",
            zorder=z_layer,
        )
    elif kind == "vec" and isinstance(payload, VectorPayload):
        primitives.draw_vector_arrow_2d(
            ax,
            payload.tail_2d,
            payload.tip_2d,
            color=payload.color,
            linewidth=max(1.8, 0.55 * state.bond_w),
            zorder=z_layer,
        )


def _estimate_pt_per_angstrom(
    ax: maxes.Axes,
    bounds_2d: np.ndarray,
    margin: float,
) -> float:
    """Estimates screen points per Angstrom from axes size and bounds."""
    fig_w, fig_h = cast(Any, ax.figure).get_size_inches()
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
    items: list[PainterItem],
    atoms: structure.AtomsData,
    miller_planes: Sequence[tuple[int, int, int]] | None,
    cam: geometry.Camera,
    scene: SceneGeometry3D,
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
        face_col, edge_col = _miller_plane_colors(idx_m)
        items.append(
            PainterItem(
                depth=float(np.mean(poly_depth)) - 0.01,
                kind="miller",
                payload=MillerPayload(
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


def _draw_miller_labels(
    ax: maxes.Axes,
    labels_meta: Sequence[tuple[tuple[int, int, int], np.ndarray, str]],
    zorder: float,
) -> None:
    """Places crisp ``(hkl)`` badges above the rendered crystal."""
    bg = str(plt.rcParams.get("axes.facecolor", palettes.PAPER))
    for hkl, centroid, edge_col in labels_meta:
        parts = [str(int(v)) for v in hkl]
        tag = (
            f"({''.join(parts)})"
            if all(len(p) == 1 for p in parts)
            else f"({' '.join(parts)})"
        )
        txt_col = primitives.ensure_readable_text_color(edge_col, bg)
        ax.text(
            float(centroid[0]),
            float(centroid[1]),
            tag,
            fontsize=9.5,
            color=txt_col,
            ha="center",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": bg,
                "edgecolor": txt_col,
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
    cell_cfg: CellConfig3D,
    radius_scale: float,
    pad: float,
) -> SceneGeometry3D:
    """Computes projected points, colors, radii, cell mode, and widths."""
    pivot = (
        0.5 * (atoms.cell[0] + atoms.cell[1] + atoms.cell[2])
        if atoms.cell is not None
        else np.mean(atoms.positions, axis=0)
    )
    proj_span = geometry.scene_projection_span(
        atoms.positions, cam, center=pivot, cell=atoms.cell
    )
    xy, depth = geometry.project_points(
        atoms.positions, cam, center=pivot, span=proj_span
    )
    colors = [
        palettes.get_element_color(sym, scheme=palette) for sym in atoms.symbols
    ]
    radii = [
        _atom_visual_radius(sym, style, radius_scale) for sym in atoms.symbols
    ]
    margin = max(radii, default=0.4) + pad

    bounds_pts: list[np.ndarray] = [xy]
    if cell_cfg.draw_box and atoms.cell is not None:
        corners_2d, _, _ = geometry.cell_corners_projected(
            atoms.cell, cam, center=pivot, span=proj_span
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
    return SceneGeometry3D(
        pivot=pivot,
        proj_span=proj_span,
        xy=xy,
        depth=depth,
        all_pts=all_pts,
        colors=colors,
        radii=radii,
        margin=margin,
        cell_cfg=cell_cfg,
        bond_w=bond_w,
        bond_off=bond_off,
        atom_lw=atom_lw,
        cell_lw=cell_lw,
    )


def _build_3d_painter_queue(
    atoms: structure.AtomsData,
    cam: geometry.Camera,
    scene: SceneGeometry3D,
    opts: RenderOptions3D,
    polyhedra: tuple[str, str | None]
    | Sequence[bonding.CoordinationPolyhedron]
    | None,
    miller_planes: Sequence[tuple[int, int, int]] | None,
    vec_arr: np.ndarray | None,
) -> tuple[
    list[PainterItem],
    list[tuple[tuple[int, int, int], np.ndarray, str]],
]:
    """Assembles and depth-sorts atoms, bonds, polyhedra, vectors, & slices."""
    items: list[PainterItem] = [
        PainterItem(depth=float(scene.depth[idx]), kind="atom", payload=idx)
        for idx in range(len(atoms))
    ]
    if opts.show_bonds and opts.style != "spacefill":
        if atoms.bonds:
            bonds = list(atoms.bonds)
        elif len(atoms) <= _MAX_AUTOBOND_ATOMS:
            bonds = bonding.infer_bonds(atoms, scale=opts.bond_scale)
        else:
            bonds = []
            warnings.warn(
                f"Structure has {len(atoms)} atoms (> {_MAX_AUTOBOND_ATOMS}); "
                "skipping automatic bond inference. Pass show_bonds=True or "
                "precomputed atoms.bonds to draw bonds.",
                UserWarning,
                stacklevel=4,
            )
        _add_bond_queue_items(
            items,
            atoms,
            bonds,
            cam,
            scene,
            skip_spec=_polyhedra_skip_spec(opts.style, polyhedra),
        )
    _add_polyhedra_items(items, atoms, polyhedra, cam, scene, opts.palette)
    _add_vector_items(items, atoms, vec_arr, cam, scene, opts)
    miller_labels = _add_miller_queue_items(
        items, atoms, miller_planes, cam, scene
    )
    items.sort(key=lambda entry: entry.depth)
    return items, miller_labels


def _add_front_cell_edges(
    items: list[PainterItem],
    atoms: structure.AtomsData,
    cam: geometry.Camera,
    scene: SceneGeometry3D,
) -> None:
    """Adds depth-sorted front cell edges so corner/edge atoms sit cleanly."""
    if atoms.cell is None:
        return
    corners_2d, c_depth, farthest_idx = geometry.cell_corners_projected(
        atoms.cell, cam, center=scene.pivot, span=scene.proj_span
    )
    for i_idx, j_idx in _FRONT_CELL_EDGES:
        if farthest_idx in (i_idx, j_idx):
            continue
        edge_depth = 0.5 * float(c_depth[i_idx] + c_depth[j_idx]) - 0.05
        items.append(
            PainterItem(
                depth=edge_depth,
                kind="cell_edge",
                payload=CellEdgePayload(
                    p0=corners_2d[i_idx],
                    p1=corners_2d[j_idx],
                    linewidth=scene.cell_lw,
                ),
            )
        )


def _finalize_3d_viewport(
    ax: maxes.Axes,
    atoms: structure.AtomsData,
    cam: geometry.Camera,
    scene: SceneGeometry3D,
    front_z: float,
) -> None:
    """Applies viewport bounds and draws standalone crystal axes when active."""
    x_lo = float(np.min(scene.all_pts[:, 0])) - scene.margin
    x_hi = float(np.max(scene.all_pts[:, 0])) + scene.margin
    y_lo = float(np.min(scene.all_pts[:, 1])) - scene.margin
    y_hi = float(np.max(scene.all_pts[:, 1])) + scene.margin
    x_span = max(x_hi - x_lo, 2.0)
    standalone_axes = scene.cell_cfg.standalone_axes
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
                center=scene.pivot,
                zorder=front_z + 2.5,
                span=scene.proj_span,
            )


def _render_prepared_3d(
    ax: maxes.Axes,
    atoms: structure.AtomsData,
    cam: geometry.Camera,
    scene: SceneGeometry3D,
    opts: RenderOptions3D,
    polyhedra: tuple[str, str | None]
    | Sequence[bonding.CoordinationPolyhedron]
    | None,
    miller_planes: Sequence[tuple[int, int, int]] | None,
    vec_arr: np.ndarray | None,
) -> None:
    """Executes cell backdrop, depth-sorted painter pass, and front cell."""
    cfg = scene.cell_cfg
    if cfg.draw_box and atoms.cell is not None:
        primitives.draw_unit_cell_box(
            ax,
            atoms.cell,
            cam,
            center=scene.pivot,
            linewidth=scene.cell_lw,
            draw_back=True,
            draw_front=False,
            base_plane_color=cfg.base_plane_color,
            cell_axes=cfg.box_axes,
            span=scene.proj_span,
        )

    items, miller_labels = _build_3d_painter_queue(
        atoms, cam, scene, opts, polyhedra, miller_planes, vec_arr
    )
    if cfg.draw_box and atoms.cell is not None:
        _add_front_cell_edges(items, atoms, cam, scene)
        items.sort(key=lambda entry: entry.depth)
    state = RenderState3D(
        atoms=atoms,
        xy=scene.xy,
        colors=scene.colors,
        radii=scene.radii,
        bond_w=scene.bond_w,
        bond_off=scene.bond_off,
        atom_lw=scene.atom_lw,
        show_labels=opts.show_labels,
        style=opts.style,
        vector_color=opts.vector_color,
    )
    for rank, item in enumerate(items):
        _draw_queue_item(ax, item, 2.0 + 0.02 * rank, state)

    front_z = 2.0 + 0.02 * (len(items) + 4)
    if cfg.draw_box and atoms.cell is not None:
        primitives.draw_unit_cell_box(
            ax,
            atoms.cell,
            cam,
            center=scene.pivot,
            linewidth=scene.cell_lw,
            draw_back=False,
            draw_front=True,
            draw_front_edges=False,
            base_plane_color=None,
            cell_axes=cfg.box_axes,
            front_zorder=front_z,
            span=scene.proj_span,
        )
    if miller_labels:
        _draw_miller_labels(ax, miller_labels, zorder=front_z + 1.5)
    _finalize_3d_viewport(ax, atoms, cam, scene, front_z)


def render_structure_3d(
    atoms: structure.AtomsData,
    ax: maxes.Axes | None = None,
    camera: geometry.Camera | str | None = None,
    style: str = "ball_and_stick",
    palette: str | Mapping[str, str] = "xkcd",
    show_cell: bool | str = True,
    show_labels: bool | str = False,
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
    vectors: (
        np.ndarray | Sequence[Sequence[float]] | Sequence[float] | str | None
    ) = None,
    vector_scale: float = 1.0,
    vector_color: str = "#D55E00",
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
        ``"spacefill"`` and ``"polyhedra"`` views free of bond sticks.
      radius_scale: Global multiplier for atom sphere radii.
      bond_scale: Covalent-radius factor used when inferring missing bonds.
      polyhedra: ``(center_symbol, ligand_symbol)`` pair or sequence of
        ``CoordinationPolyhedron``.
      miller_planes: Optional crystallographic planes ``[(h, k, l), ...]``.
      pad: Viewport margin in Angstrom around all projected geometry.
      vectors: Optional per-atom 3D vector overlay: ``"forces"``,
        ``"velocities"``, ``"magmom"``, or an ``(N, 3)`` / ``(N,)`` array.
      vector_scale: Multiplier for vector arrow length in Angstrom.
      vector_color: Arrow hex color or ``"spin"`` for red(+)/blue(-) z-spins.

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

    vec_arr = _resolve_vectors_3d(atoms, vectors)
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
    opts = RenderOptions3D(
        style=style,
        palette=palette,
        show_labels=show_labels,
        show_bonds=draw_bonds,
        bond_scale=bond_scale,
        vector_scale=vector_scale,
        vector_color=vector_color,
    )
    _render_prepared_3d(
        ax, atoms, cam, scene, opts, polyhedra, miller_planes, vec_arr
    )
    return ax
