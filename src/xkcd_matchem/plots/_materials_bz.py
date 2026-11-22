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
"""3D first Brillouin zone polyhedron and high-symmetry k-path renderer."""

from __future__ import annotations

from collections.abc import Sequence
import math
import pathlib
from typing import Any
from typing import cast

from matplotlib import axes as maxes
from matplotlib import collections
from matplotlib import pyplot as plt
import numpy as np

from xkcd_matchem.core import brillouin
from xkcd_matchem.core import exceptions
from xkcd_matchem.core import geometry
from xkcd_matchem.core import structure
from xkcd_matchem.io import loader
from xkcd_matchem.plots import _materials_ternary as mat_tern
from xkcd_matchem.plots import _materials_wulff as mat_wulff
from xkcd_matchem.render import primitives
from xkcd_matchem.style import palettes
from xkcd_matchem.style import theme


def draw_bz_corner_axes(
    ax: maxes.Axes,
    bz: brillouin.BrillouinZone,
    cam: geometry.Camera,
    bounds: tuple[float, float, float, float],
    sw: theme.StrokeHierarchy,
    bg_color: str,
    *,
    ink_col: str | None = None,
) -> None:
    """Draws reciprocal basis ``b1, b2, b3`` triad cleanly in bottom-left."""
    basis = geometry.rotation_matrix(cam.azimuth, cam.elevation, cam.roll)
    unit_b = np.array(
        [
            v / max(float(np.linalg.norm(v)), 1e-12)
            for v in bz.reciprocal_cell
        ],
        dtype=np.float64,
    )
    proj = unit_b @ basis.T
    x_w = max(bounds[1] - bounds[0], 1.0)
    y_w = max(bounds[3] - bounds[2], 1.0)
    a_len = 0.125 * min(x_w, y_w)
    vecs = proj[:, :2] * a_len
    ox = bounds[0] + 0.085 * x_w - min(0.0, float(np.min(vecs[:, 0])))
    oy = bounds[2] + 0.095 * y_w - min(0.0, float(np.min(vecs[:, 1])))
    raw_cols = (
        ("#F59E0B", "#34D399", "#56B4E9")
        if primitives.is_dark_color(bg_color)
        else ("#D55E00", "#009E73", "#0072B2")
    )
    resolved_ink = (
        str(plt.rcParams.get("axes.edgecolor", palettes.INK))
        if ink_col is None
        else ink_col
    )
    ax.scatter([ox], [oy], s=14.0, color=resolved_ink, zorder=6.2)
    for rank, idx in enumerate(
        sorted(range(3), key=lambda k: float(proj[k, 2]))
    ):
        vx, vy = float(vecs[idx, 0]), float(vecs[idx, 1])
        seg_len = math.hypot(vx, vy)
        if seg_len < 1e-4:
            continue
        col_i = primitives.ensure_readable_text_color(raw_cols[idx], bg_color)
        ax.annotate(
            "",
            xy=(ox + vx, oy + vy),
            xytext=(ox, oy),
            arrowprops={
                "arrowstyle": "-|>,head_width=0.20,head_length=0.26",
                "color": col_i,
                "lw": sw.leader,
            },
            zorder=5.5 + 0.05 * rank,
        )
        pad_lbl = max(0.24 * a_len, 0.22)
        ax.text(
            ox + vx + pad_lbl * (vx / seg_len),
            oy + vy + pad_lbl * (vy / seg_len),
            f"$b_{idx + 1}$",
            fontsize=8.8,
            color=col_i,
            ha="center",
            va="center",
            zorder=6.4,
        )


def kpoint_label_offset(
    idx: int,
    q_xy: np.ndarray,
    bz_center: np.ndarray,
    span: float,
) -> tuple[float, float]:
    """Finds angular-gap outward direction so k-labels never touch segments."""
    pt = q_xy[idx]
    inc_angles: list[float] = []
    for k in range(len(q_xy) - 1):
        if idx in (k, k + 1):
            other = q_xy[k + 1] if k == idx else q_xy[k]
            vec = other - pt
            if float(np.linalg.norm(vec)) > 1e-6:
                inc_angles.append(math.atan2(float(vec[1]), float(vec[0])))
    rad_vec = pt - bz_center
    rad_ang = (
        math.atan2(float(rad_vec[1]), float(rad_vec[0]))
        if float(np.linalg.norm(rad_vec)) > 0.08 * span
        else 0.5 * math.pi
    )
    best_ang = rad_ang
    if inc_angles:
        cands = np.linspace(0.0, 2.0 * math.pi, 32, endpoint=False)
        best_score = -1.0
        for ang in cands:
            min_sep = min(
                abs(((ang - a + math.pi) % (2.0 * math.pi)) - math.pi)
                for a in inc_angles
            )
            align = math.cos(ang - rad_ang)
            score = min_sep + 0.22 * align
            if score > best_score:
                best_score = score
                best_ang = float(ang)
    step = 0.095 * span
    return step * math.cos(best_ang), step * math.sin(best_ang)


def draw_bz_kpath_and_axes(
    ax: maxes.Axes,
    bz: brillouin.BrillouinZone,
    cam: geometry.Camera,
    span: float,
    bounds: tuple[float, float, float, float],
    style_info: tuple[str, str, str, theme.StrokeHierarchy],
    *,
    kpath: Sequence[str] | None,
    show_kpoints: bool,
    show_axes: bool,
) -> None:
    """Draws reciprocal basis triad and subordinated high-symmetry k-path."""
    ink_col, path_color, bg_color, sw = style_info
    origin = np.zeros(3, dtype=np.float64)
    if show_axes:
        draw_bz_corner_axes(
            ax, bz, cam, bounds, sw, bg_color, ink_col=ink_col
        )

    tour = list(bz.default_path if kpath is None else kpath)
    valid = [name for name in tour if name in bz.high_symmetry_points]
    if not show_kpoints or not valid:
        return
    all_v, _ = geometry.project_points(
        bz.vertices, cam, center=origin, span=span
    )
    bz_center = np.mean(all_v, axis=0)
    q_arr = np.vstack([bz.high_symmetry_points[name] for name in valid])
    q_xy, _ = geometry.project_points(q_arr, cam, center=origin, span=span)
    if len(q_xy) >= 2:
        ax.plot(
            q_xy[:, 0],
            q_xy[:, 1],
            color=path_color,
            linewidth=sw.overlay,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=6.5,
        )
    ax.scatter(
        q_xy[:, 0],
        q_xy[:, 1],
        s=28.0,
        facecolor=path_color,
        edgecolor=ink_col,
        linewidth=sw.guide,
        zorder=7.5,
    )
    seen_pts: set[str] = set()
    for idx, name in enumerate(valid):
        if name in seen_pts:
            continue
        seen_pts.add(name)
        disp_lbl = "Γ" if name == "G" else name
        dx, dy = kpoint_label_offset(idx, q_xy, bz_center, span)
        theme.place_halo_text(
            ax,
            float(q_xy[idx, 0]) + dx,
            float(q_xy[idx, 1]) + dy,
            disp_lbl,
            color=ink_col,
            bg_color=bg_color,
            fontsize=9.2,
            pad=0.07,
            alpha=0.82,
            zorder=8.5,
        )


def project_bz_facets(
    bz: brillouin.BrillouinZone,
    cam: geometry.Camera,
    color: str,
    *,
    pad_left_axes: bool = False,
) -> tuple[
    list[np.ndarray],
    list[tuple[float, float, float]],
    list[np.ndarray],
    tuple[float, float, float, float],
    float,
]:
    """Projects, splits front/back, and Lambert-shades Brillouin zone faces."""
    basis = geometry.rotation_matrix(cam.azimuth, cam.elevation, cam.roll)
    light_dir = np.array([-0.45, 0.55, 0.70], dtype=np.float64)
    light_dir /= float(np.linalg.norm(light_dir))
    origin = np.zeros(3, dtype=np.float64)
    span = geometry.scene_projection_span(bz.vertices, cam, center=origin)

    records: list[
        tuple[float, np.ndarray, tuple[float, float, float], float]
    ] = []
    for facet in bz.facets:
        xy_f, dep_f = geometry.project_points(
            facet.vertices, cam, center=origin, span=span
        )
        n_rot = facet.normal @ basis.T
        shaded = mat_wulff.lambert_shade_rgb(
            color, float(np.dot(n_rot, light_dir))
        )
        records.append((float(np.mean(dep_f)), xy_f, shaded, float(n_rot[2])))
    records.sort(key=lambda item: item[0])

    front_polys = [it[1] for it in records if it[3] >= -1e-3]
    front_cols = [it[2] for it in records if it[3] >= -1e-3]
    back_polys = [it[1] for it in records if it[3] < -1e-3]

    all_xy, _ = geometry.project_points(
        bz.vertices, cam, center=origin, span=span
    )
    x_min, y_min = np.min(all_xy, axis=0)
    x_max, y_max = np.max(all_xy, axis=0)
    scene_w = max(float(x_max - x_min), float(y_max - y_min), 1e-3)
    pad = 0.24 * scene_w
    left_extra = 0.28 * scene_w if pad_left_axes else 0.0
    bounds = (
        float(x_min - pad - left_extra),
        float(x_max + pad),
        float(y_min - pad),
        float(y_max + pad),
    )
    return front_polys, front_cols, back_polys, bounds, span


def plot_brillouin_zone(
    source: (
        brillouin.BrillouinZone
        | structure.AtomsData
        | np.ndarray
        | Sequence[Sequence[float]]
        | str
        | pathlib.Path
    ),
    *,
    use_primitive: bool = True,
    kpath: Sequence[str] | None = None,
    show_kpoints: bool = True,
    show_axes: bool = True,
    camera: geometry.Camera | str | None = "isometric",
    ax: maxes.Axes | None = None,
    color: str = "#56B4E9",
    path_color: str = "#D55E00",
    alpha: float = 0.36,
    preset: str | theme.ThemePreset = "classic",
    title: str | None = None,
) -> maxes.Axes:
    """Plots 3D first Brillouin zone with high-symmetry k-path in xkcd style.

    Args:
      source: ``BrillouinZone``, ``AtomsData``, ``(3, 3)`` lattice matrix,
        file path, or bundled sample filename.
      use_primitive: Reduce centered F/I/C conventional cells to primitive form.
      kpath: Optional ordered sequence of high-symmetry point names to trace.
      show_kpoints: Whether to plot the high-symmetry k-path and markers.
      show_axes: Whether to draw reciprocal basis vectors ``b1, b2, b3``.
      camera: ``Camera`` object or preset string.
      ax: Optional target Matplotlib ``Axes``.
      color: Facet fill hex color string.
      path_color: High-symmetry k-path and marker accent hex color.
      alpha: Translucent polyhedron facet opacity.
      preset: Visual theme preset name or ``ThemePreset``.
      title: Optional plot heading.

    Returns:
      Matplotlib ``Axes`` containing the projected first Brillouin zone.
    """
    if isinstance(source, brillouin.BrillouinZone):
        bz = source
    elif isinstance(source, (str, pathlib.Path)):
        loaded = loader.load(source)
        if not isinstance(loaded, structure.AtomsData):
            raise exceptions.StructureError(
                f"Expected AtomsData from {source!r}."
            )
        bz = brillouin.compute_brillouin_zone(
            loaded, use_primitive=use_primitive
        )
    else:
        bz = brillouin.compute_brillouin_zone(
            source, use_primitive=use_primitive
        )

    cam = geometry.resolve_camera(camera)
    preset_obj = theme.get_preset(preset)
    ink_col = preset_obj.ink_color
    bg_col = preset_obj.axes_facecolor
    sw = theme.stroke_weights(preset)
    front_polys, front_cols, back_polys, bounds, span = project_bz_facets(
        bz, cam, color, pad_left_axes=show_axes
    )
    back_edge_col = mat_tern.theme_muted_guide(bg_col)

    with theme.xkcd_style(preset=preset):
        if ax is None:
            _, ax = plt.subplots(figsize=(5.8, 5.2))
        else:
            theme.style_existing_axes(ax, preset=preset)
        if back_polys:
            back_coll = collections.PolyCollection(
                cast(Any, back_polys),
                facecolors=[
                    mat_wulff.lambert_shade_rgb(color, 0.15)
                    for _ in back_polys
                ],
                edgecolors=back_edge_col,
                linewidths=round(0.85 * sw.guide, 2),
                linestyles="--",
                alpha=alpha * 0.40,
                zorder=1.5,
            )
            ax.add_collection(back_coll)
        if front_polys:
            front_coll = collections.PolyCollection(
                cast(Any, front_polys),
                facecolors=front_cols,
                edgecolors=ink_col,
                linewidths=sw.frame,
                alpha=alpha,
                joinstyle="round",
                zorder=3.0,
            )
            ax.add_collection(front_coll)
        draw_bz_kpath_and_axes(
            ax,
            bz,
            cam,
            span,
            bounds,
            (ink_col, path_color, bg_col, sw),
            kpath=kpath,
            show_kpoints=show_kpoints,
            show_axes=show_axes,
        )
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[2], bounds[3])
        ax.set_aspect("equal")
        ax.set_axis_off()
        if title:
            ax.set_title(title, color=ink_col)
    return ax
