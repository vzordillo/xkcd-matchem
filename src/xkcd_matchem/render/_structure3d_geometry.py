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
"""Data structures, projection geometry, and cell config for 3D rendering."""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
from typing import Any
from typing import cast

from matplotlib import axes as maxes
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
import numpy as np

from xkcd_matchem.core import exceptions
from xkcd_matchem.core import geometry
from xkcd_matchem.core import structure
from xkcd_matchem.data import elements
from xkcd_matchem.render import primitives
from xkcd_matchem.style import palettes


VALID_STYLES = ("ball_and_stick", "spacefill", "stick", "polyhedra")
CORNER_AXES_LOCS = ("bottom_left", "bottom_right", "top_left", "top_right")
MAX_AUTOBOND_ATOMS = 2000
FRONT_CELL_EDGES: tuple[tuple[int, int], ...] = (
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
    show_hbonds: bool = False
    hbond_color: str | None = None


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
class HBondPayload:
    """Painter item payload for a dashed geometric hydrogen bond."""

    p_start: np.ndarray
    p_end: np.ndarray
    color: str
    linewidth: float


@dataclasses.dataclass(frozen=True)
class PainterItem:
    """Depth-sorted 3D painter's-algorithm draw entry."""

    depth: float
    kind: str
    payload: (
        int
        | BondPayload
        | HalfBondPayload
        | HBondPayload
        | TrianglePayload
        | MillerPayload
        | CellEdgePayload
        | VectorPayload
    )


def resolve_axes_bg_ink(ax: maxes.Axes | None) -> tuple[str, str]:
    """Resolves active ``(background_hex, ink_hex)`` from axes and rcParams."""
    rc_bg = str(plt.rcParams.get("axes.facecolor", palettes.PAPER))
    rc_ink = str(plt.rcParams.get("axes.edgecolor", palettes.INK))
    if ax is None:
        return rc_bg, rc_ink
    try:
        bg_hex = mcolors.to_hex(ax.get_facecolor())
    except ValueError:
        bg_hex = rc_bg
    try:
        spine_col = ax.spines["bottom"].get_edgecolor()
        ink_hex = mcolors.to_hex(spine_col)
    except (KeyError, ValueError):
        ink_hex = rc_ink
    if primitives.is_dark_color(bg_hex) and primitives.is_dark_color(ink_hex):
        ink_hex = (
            rc_ink
            if not primitives.is_dark_color(rc_ink)
            else palettes.DARK_CHALK
        )
    return bg_hex, ink_hex


def atom_visual_radius(symbol: str, style: str, scale: float) -> float:
    """Returns visual 2D sphere radius in Angstrom-scaled display units."""
    if style == "spacefill":
        return elements.get_vdw_radius(symbol) * 0.82 * scale
    if style == "stick":
        return 0.14 * scale
    if style == "polyhedra":
        return max(0.18, elements.get_covalent_radius(symbol) * 0.26) * scale
    r_cov = elements.get_covalent_radius(symbol)
    return max(0.24, min(0.58, 0.18 + 0.27 * r_cov)) * scale


def resolve_axes_override(
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


def parse_cell_and_axes(
    show_cell: bool | str,
    crystal_axes: bool | str | None,
    has_cell: bool,
    has_miller: bool,
    *,
    bg_color: str | None = None,
) -> CellConfig3D:
    """Resolves cell box, plane tint, and crystallographic axes layout."""
    if not has_cell:
        return CellConfig3D(False, None, "none", "none")

    active_bg = (
        str(plt.rcParams.get("axes.facecolor", palettes.PAPER))
        if bg_color is None
        else bg_color
    )
    default_tint = (
        primitives.muted_surface_color(active_bg)
        if primitives.is_dark_color(active_bg)
        else palettes.VACUUM_TINT
    )
    draw_box = True
    base_col: str | None = None if has_miller else default_tint
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

    box_axes, standalone_axes = resolve_axes_override(
        crystal_axes, draw_box, default_axes
    )
    return CellConfig3D(draw_box, base_col, box_axes, standalone_axes)


def resolve_vectors_3d(
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


def estimate_pt_per_angstrom(
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


def prepare_scene_geometry(
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
        atom_visual_radius(sym, style, radius_scale) for sym in atoms.symbols
    ]
    margin = max(radii, default=0.4) + pad

    bounds_pts: list[np.ndarray] = [xy]
    if cell_cfg.draw_box and atoms.cell is not None:
        corners_2d, _, _ = geometry.cell_corners_projected(
            atoms.cell, cam, center=pivot, span=proj_span
        )
        bounds_pts.append(corners_2d)
    all_pts = np.vstack(bounds_pts)

    pt_per_a = estimate_pt_per_angstrom(ax, all_pts, margin)
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
