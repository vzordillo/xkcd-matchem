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
from xkcd_matchem.render import _structure3d_draw as s3d
from xkcd_matchem.render import _structure3d_geometry as s3g
from xkcd_matchem.render import _structure3d_queue as s3q


__all__ = [
    "BondPayload",
    "CellConfig3D",
    "CellEdgePayload",
    "HBondPayload",
    "HalfBondPayload",
    "MillerPayload",
    "PainterItem",
    "RenderOptions3D",
    "RenderState3D",
    "SceneGeometry3D",
    "TrianglePayload",
    "VectorPayload",
    "render_structure_3d",
]

# Re-exported public data classes and constants.
BondPayload = s3g.BondPayload
CellConfig3D = s3g.CellConfig3D
CellEdgePayload = s3g.CellEdgePayload
HBondPayload = s3g.HBondPayload
HalfBondPayload = s3g.HalfBondPayload
MillerPayload = s3g.MillerPayload
PainterItem = s3g.PainterItem
RenderOptions3D = s3g.RenderOptions3D
RenderState3D = s3g.RenderState3D
SceneGeometry3D = s3g.SceneGeometry3D
TrianglePayload = s3g.TrianglePayload
VectorPayload = s3g.VectorPayload

# Backwards-compatible internal symbol re-exports.
_VALID_STYLES = s3g.VALID_STYLES
_CORNER_AXES_LOCS = s3g.CORNER_AXES_LOCS
_MAX_AUTOBOND_ATOMS = s3g.MAX_AUTOBOND_ATOMS
_FRONT_CELL_EDGES = s3g.FRONT_CELL_EDGES

_atom_visual_radius = s3g.atom_visual_radius
_resolve_axes_override = s3g.resolve_axes_override
_parse_cell_and_axes = s3g.parse_cell_and_axes
_resolve_vectors_3d = s3g.resolve_vectors_3d
_estimate_pt_per_angstrom = s3g.estimate_pt_per_angstrom
_prepare_scene_geometry = s3g.prepare_scene_geometry
_polyhedra_skip_spec = s3q.polyhedra_skip_spec
_should_skip_poly_bond = s3q.should_skip_poly_bond
_build_interior_bond_item = s3q.build_interior_bond_item
_add_bond_queue_items = s3q.add_bond_queue_items
_add_hbond_queue_items = s3q.add_hbond_queue_items
_add_polyhedra_items = s3q.add_polyhedra_items
_add_vector_items = s3q.add_vector_items
_miller_plane_colors = s3q.miller_plane_colors
_add_miller_queue_items = s3q.add_miller_queue_items
_add_front_cell_edges = s3q.add_front_cell_edges
_build_3d_painter_queue = s3q.build_3d_painter_queue
_format_atom_label = s3d.format_atom_label
_draw_queue_item = s3d.draw_queue_item
_draw_miller_labels = s3d.draw_miller_labels
_finalize_3d_viewport = s3d.finalize_3d_viewport
_render_prepared_3d = s3d.render_prepared_3d


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
    show_hbonds: bool | str = False,
    hbond_color: str | None = None,
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
      show_hbonds: Whether to overlay dashed geometric ``D-H...A`` hydrogen
        bonds (pass ``True`` or a hex color string such as ``"#56B4E9"``).
      hbond_color: Optional hex color for dashed hydrogen-bond strokes
        (defaults to theme-contrasting blue).

    Returns:
      Matplotlib ``Axes`` containing the rendered structure.

    Raises:
      StructureError: If ``style``, ``crystal_axes``, or ``camera`` is invalid.
    """
    if style not in s3g.VALID_STYLES:
        raise exceptions.StructureError(
            f"Unsupported 3D render style {style!r}. Valid: {s3g.VALID_STYLES}"
        )
    if ax is None:
        _, ax = plt.subplots(figsize=(5.6, 4.8))
    if len(atoms) == 0:
        ax.set_axis_off()
        return ax

    bg_col, _ = s3g.resolve_axes_bg_ink(ax)
    vec_arr = s3g.resolve_vectors_3d(atoms, vectors)
    cam = geometry.resolve_camera(camera, cell=atoms.cell)
    cell_cfg = s3g.parse_cell_and_axes(
        show_cell,
        crystal_axes,
        atoms.cell is not None,
        bool(miller_planes),
        bg_color=bg_col,
    )
    scene = s3g.prepare_scene_geometry(
        atoms, ax, cam, style, palette, cell_cfg, radius_scale, pad
    )
    draw_bonds = (
        style not in ("spacefill", "polyhedra")
        if show_bonds is None
        else show_bonds
    )
    hb_flag = bool(show_hbonds) and (
        not isinstance(show_hbonds, str)
        or show_hbonds.strip().lower() not in ("", "none", "off", "false")
    )
    hb_col = (
        show_hbonds
        if (hb_flag and isinstance(show_hbonds, str) and hbond_color is None)
        else hbond_color
    )
    opts = s3g.RenderOptions3D(
        style=style,
        palette=palette,
        show_labels=show_labels,
        show_bonds=draw_bonds,
        bond_scale=bond_scale,
        vector_scale=vector_scale,
        vector_color=vector_color,
        show_hbonds=hb_flag,
        hbond_color=hb_col,
    )
    s3d.render_prepared_3d(
        ax, atoms, cam, scene, opts, polyhedra, miller_planes, vec_arr
    )
    return ax
