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
"""Painter dispatch, Miller badges, and viewport finalization for 3D views."""

from __future__ import annotations

from collections.abc import Sequence

from matplotlib import axes as maxes
import numpy as np

from xkcd_matchem.core import bonding
from xkcd_matchem.core import geometry
from xkcd_matchem.core import structure
from xkcd_matchem.render import _structure3d_geometry as s3g
from xkcd_matchem.render import _structure3d_queue as s3q
from xkcd_matchem.render import primitives


def format_atom_label(
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


def draw_queue_item(
    ax: maxes.Axes,
    item: s3g.PainterItem,
    z_layer: float,
    state: s3g.RenderState3D,
    *,
    ink_color: str | None = None,
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
            edgecolor=ink_color,
            linewidth=state.atom_lw,
            label=format_atom_label(state.atoms, payload, state.show_labels),
            highlight=state.style != "stick",
            zorder=z_layer,
        )
    elif kind == "bond" and isinstance(payload, s3g.BondPayload):
        primitives.draw_two_tone_bond(
            ax,
            payload.p_start,
            payload.p_end,
            c1=payload.col_start,
            c2=payload.col_end,
            edgecolor=ink_color,
            width=state.bond_w,
            order=payload.order,
            offset_scale=state.bond_off,
            zorder=z_layer,
            split_frac=payload.split_frac,
        )
    elif kind == "half" and isinstance(payload, s3g.HalfBondPayload):
        primitives.draw_half_bond(
            ax,
            payload.p_start,
            payload.p_end,
            color=payload.color,
            edgecolor=ink_color,
            width=state.bond_w,
            order=payload.order,
            offset_scale=state.bond_off,
            zorder=z_layer,
        )
    elif kind == "tri" and isinstance(payload, s3g.TrianglePayload):
        primitives.draw_polyhedron_triangle(
            ax,
            payload.verts_2d,
            facecolor=payload.facecolor,
            edgecolor=ink_color,
            zorder=z_layer,
        )
    elif (
        kind == "miller"
        and isinstance(payload, s3g.MillerPayload)
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
    elif kind == "cell_edge" and isinstance(payload, s3g.CellEdgePayload):
        _, resolved_ink = s3g.resolve_axes_bg_ink(ax)
        stroke = resolved_ink if ink_color is None else ink_color
        ax.plot(
            [float(payload.p0[0]), float(payload.p1[0])],
            [float(payload.p0[1]), float(payload.p1[1])],
            color=stroke,
            linestyle="-",
            linewidth=payload.linewidth,
            solid_capstyle="round",
            zorder=z_layer,
        )
    elif kind == "hbond" and isinstance(payload, s3g.HBondPayload):
        ln = ax.plot(
            [float(payload.p_start[0]), float(payload.p_end[0])],
            [float(payload.p_start[1]), float(payload.p_end[1])],
            color=payload.color,
            linestyle=(0, (3.2, 2.4)),
            linewidth=payload.linewidth,
            alpha=0.82,
            solid_capstyle="round",
            zorder=z_layer,
        )[0]
        ln.set_sketch_params(0.0, 0.0, 0.0)
    elif kind == "vec" and isinstance(payload, s3g.VectorPayload):
        primitives.draw_vector_arrow_2d(
            ax,
            payload.tail_2d,
            payload.tip_2d,
            color=payload.color,
            edgecolor=ink_color,
            linewidth=max(1.35, 0.44 * state.bond_w),
            zorder=z_layer,
        )


def draw_miller_labels(
    ax: maxes.Axes,
    labels_meta: Sequence[tuple[tuple[int, int, int], np.ndarray, str]],
    zorder: float,
) -> None:
    """Places crisp ``(hkl)`` badges above the rendered crystal."""
    bg, ink = s3g.resolve_axes_bg_ink(ax)
    for hkl, centroid, edge_col in labels_meta:
        parts = [str(int(v)) for v in hkl]
        tag = (
            f"({''.join(parts)})"
            if all(len(p) == 1 for p in parts)
            else f"({' '.join(parts)})"
        )
        edge_clean = primitives.ensure_readable_text_color(edge_col, bg)
        badge = ax.text(
            float(centroid[0]),
            float(centroid[1]),
            tag,
            fontsize=9.2,
            color=ink,
            ha="center",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.20",
                "facecolor": bg,
                "edgecolor": edge_clean,
                "linewidth": 1.0,
                "alpha": 0.92,
            },
            zorder=zorder,
        )
        patch = badge.get_bbox_patch()
        if patch is not None:
            patch.set_sketch_params(0.0, 0.0, 0.0)


def finalize_3d_viewport(
    ax: maxes.Axes,
    atoms: structure.AtomsData,
    cam: geometry.Camera,
    scene: s3g.SceneGeometry3D,
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
        bg, _ = s3g.resolve_axes_bg_ink(ax)
        axis_cols = (
            ("#F59E0B", "#34D399", "#56B4E9")
            if primitives.is_dark_color(bg)
            else ("#D55E00", "#009E73", "#0072B2")
        )
        if standalone_axes in s3g.CORNER_AXES_LOCS:
            primitives.draw_crystal_axes(
                ax,
                atoms.cell,
                cam,
                location=standalone_axes,
                colors=axis_cols,
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


def render_prepared_3d(
    ax: maxes.Axes,
    atoms: structure.AtomsData,
    cam: geometry.Camera,
    scene: s3g.SceneGeometry3D,
    opts: s3g.RenderOptions3D,
    polyhedra: tuple[str, str | None]
    | Sequence[bonding.CoordinationPolyhedron]
    | None,
    miller_planes: Sequence[tuple[int, int, int]] | None,
    vec_arr: np.ndarray | None,
) -> None:
    """Executes cell backdrop, depth-sorted painter pass, and front cell."""
    bg_col, ink_col = s3g.resolve_axes_bg_ink(ax)
    cfg = scene.cell_cfg
    if cfg.draw_box and atoms.cell is not None:
        primitives.draw_unit_cell_box(
            ax,
            atoms.cell,
            cam,
            center=scene.pivot,
            edgecolor=ink_col,
            linewidth=scene.cell_lw,
            draw_back=True,
            draw_front=False,
            base_plane_color=cfg.base_plane_color,
            cell_axes=cfg.box_axes,
            span=scene.proj_span,
        )

    items, miller_labels = s3q.build_3d_painter_queue(
        atoms,
        cam,
        scene,
        opts,
        polyhedra,
        miller_planes,
        vec_arr,
        bg_color=bg_col,
    )
    if cfg.draw_box and atoms.cell is not None:
        s3q.add_front_cell_edges(items, atoms, cam, scene)
        items.sort(key=lambda entry: entry.depth)
    state = s3g.RenderState3D(
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
        draw_queue_item(ax, item, 2.0 + 0.02 * rank, state, ink_color=ink_col)

    front_z = 2.0 + 0.02 * (len(items) + 4)
    if cfg.draw_box and atoms.cell is not None:
        primitives.draw_unit_cell_box(
            ax,
            atoms.cell,
            cam,
            center=scene.pivot,
            edgecolor=ink_col,
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
        draw_miller_labels(ax, miller_labels, zorder=front_z + 1.5)
    finalize_3d_viewport(ax, atoms, cam, scene, front_z)
