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
"""2D volumetric slices and 3D artistic isosurface plots."""

from __future__ import annotations

import pathlib
from typing import Any
from typing import cast

from matplotlib import axes as maxes
from matplotlib import pyplot as plt
import numpy as np

from xkcd_matchem.core import electronic
from xkcd_matchem.core import exceptions
from xkcd_matchem.core import geometry
from xkcd_matchem.core import isosurface
from xkcd_matchem.io import loader
from xkcd_matchem.render import artistic3d
from xkcd_matchem.render import primitives
from xkcd_matchem.render import structure3d
from xkcd_matchem.style import palettes
from xkcd_matchem.style import theme


def draw_slice_atoms(
    ax: maxes.Axes,
    grid: electronic.VolumetricGrid,
    axis: str,
    slice_index: int | None,
    basis: np.ndarray,
    preset_obj: theme.ThemePreset,
) -> None:
    """Projects atoms onto a volumetric slice with element colors & depth."""
    if grid.atoms is None or len(grid.atoms) == 0:
        return
    ax_idx = {"x": 0, "y": 1, "z": 2}[axis.lower()]
    n_along = grid.data.shape[ax_idx]
    sl_idx = n_along // 2 if slice_index is None else int(slice_index)
    slice_origin = grid.origin + sl_idx * grid.voxel_vectors[ax_idx]
    normal = np.cross(basis[0], basis[1])
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    step_perp = max(
        abs(float(np.dot(grid.voxel_vectors[ax_idx], normal))), 0.25
    )
    d_perp = np.abs((grid.atoms.positions - slice_origin) @ normal)
    atom_uv = grid.atoms.positions @ basis.T
    colors = [
        palettes.get_element_color(sym, scheme="xkcd")
        for sym in grid.atoms.symbols
    ]
    in_plane = d_perp <= 1.5 * step_perp
    for mask, size, alpha in ((~in_plane, 45.0, 0.38), (in_plane, 85.0, 0.95)):
        if np.any(mask):
            cols = [c for c, flag in zip(colors, mask, strict=True) if flag]
            ax.scatter(
                atom_uv[mask, 0],
                atom_uv[mask, 1],
                s=size,
                facecolor=cols,
                edgecolor=preset_obj.ink_color,
                linewidth=1.5 if size > 60.0 else 1.1,
                alpha=alpha,
                zorder=6 if size > 60.0 else 5.5,
            )


def plot_volumetric_slice(
    grid: electronic.VolumetricGrid,
    ax: maxes.Axes | None = None,
    axis: str = "z",
    slice_index: int | None = None,
    levels: int = 9,
    cmap: str = "coolwarm",
    show_atoms: bool = True,
    preset: str | theme.ThemePreset = "classic",
    title: str | None = None,
) -> maxes.Axes:
    """Plots a 2D filled-plus-inked slice across a ``VolumetricGrid``."""
    u, v, plane, basis = grid.slice_mesh(axis=axis, index=slice_index)
    p_min, p_max = float(np.min(plane)), float(np.max(plane))
    vmax = max(float(np.max(np.abs(plane))), 1e-6)
    lo_lvl, hi_lvl = (-vmax, vmax) if p_min < -1e-9 * vmax else (0.0, vmax)
    is_flat = abs(p_max - p_min) < 1e-12
    lvl_arr = (
        np.array([p_min - 1e-6, p_max + 1e-6], dtype=np.float64)
        if is_flat
        else np.linspace(lo_lvl, hi_lvl, max(int(levels), 2))
    )
    preset_obj = theme.get_preset(preset)
    sw = theme.stroke_weights(preset)
    with theme.xkcd_style(preset=preset):
        if ax is None:
            _, ax = plt.subplots(figsize=(5.4, 4.6))
        else:
            theme.style_existing_axes(ax, preset=preset)
        ax.contourf(
            u,
            v,
            plane,
            levels=lvl_arr,
            cmap=cmap,
            vmin=lo_lvl,
            vmax=hi_lvl,
            alpha=0.82,
        )
        if not is_flat:
            c_lvls = lvl_arr[np.abs(lvl_arr) > 0.08 * vmax]
            if c_lvls.size >= 1:
                ax.contour(
                    u,
                    v,
                    plane,
                    levels=c_lvls,
                    colors=preset_obj.ink_color,
                    linewidths=round(0.85 * sw.guide, 2),
                    alpha=0.58,
                )
        if show_atoms:
            draw_slice_atoms(ax, grid, axis, slice_index, basis, preset_obj)
        ax.set_aspect("equal")
        axis_labels = {
            "x": ("y (A)", "z (A)"),
            "y": ("x (A)", "z (A)"),
            "z": ("x (A)", "y (A)"),
        }
        xl, yl = (
            axis_labels[axis.lower()]
            if (
                float(np.max(np.abs(basis[0]))) > 0.999
                and float(np.max(np.abs(basis[1]))) > 0.999
            )
            else ("u (A)", "v (A)")
        )
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        theme.apply_xkcd_spines(ax, preset=preset)
        if title:
            ax.set_title(title, color=preset_obj.ink_color)
    return ax


def coerce_volumetric_grid(
    source: electronic.VolumetricGrid | str | pathlib.Path,
) -> electronic.VolumetricGrid:
    """Coerces path, sample name, or ``VolumetricGrid`` into grid object."""
    if isinstance(source, electronic.VolumetricGrid):
        return source
    loaded = loader.load(source)
    if not isinstance(loaded, electronic.VolumetricGrid):
        raise exceptions.StructureError(
            f"Expected VolumetricGrid, got {type(loaded).__name__}."
        )
    return loaded


def expand_isosurface_limits(
    ax: maxes.Axes,
    all_xy: np.ndarray,
    has_atoms: bool,
) -> None:
    """Expands viewport limits cleanly around both atoms and isosurface."""
    mx0, my0 = np.min(all_xy, axis=0)
    mx1, my1 = np.max(all_xy, axis=0)
    if has_atoms:
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        ax.set_xlim(min(x0, float(mx0) - 0.5), max(x1, float(mx1) + 0.5))
        ax.set_ylim(min(y0, float(my0) - 0.5), max(y1, float(my1) + 0.5))
        return
    pad = 0.15 * max(float(mx1 - mx0), float(my1 - my0), 1.0)
    ax.set_xlim(float(mx0) - pad, float(mx1) + pad)
    ax.set_ylim(float(my0) - pad, float(my1) + pad)


def extract_isosurface_meshes(
    source: electronic.VolumetricGrid | str | pathlib.Path,
    level: float | None,
    symmetric: bool | None,
    colors: tuple[str, str],
) -> tuple[
    electronic.VolumetricGrid,
    list[tuple[isosurface.IsosurfaceMesh, str]],
]:
    """Coerces grid data and extracts positive and optional negative lobes."""
    raw_grid = coerce_volumetric_grid(source)
    grid = isosurface.upsample_coarse_grid(raw_grid, min_dim=36)
    max_abs = max(float(np.max(np.abs(grid.data))), 1e-12)
    iso_level = 0.35 * max_abs if level is None else float(level)
    if abs(iso_level) < 1e-15:
        raise exceptions.StructureError("Isosurface level must be non-zero.")
    sym_flag = (
        bool(float(np.min(grid.data)) < -0.05 * max_abs)
        if symmetric is None
        else bool(symmetric)
    )
    meshes: list[tuple[isosurface.IsosurfaceMesh, str]] = [
        (isosurface.extract_isosurface(grid, iso_level), colors[0])
    ]
    if sym_flag:
        meshes.append(
            (isosurface.extract_isosurface(grid, -abs(iso_level)), colors[1])
        )
    return grid, meshes


def draw_iso_art_batches(
    ax: maxes.Axes,
    art: artistic3d.ShadeArtData,
    sketch_scale: float,
) -> None:
    """Draws front wash, stipple dots, comic strokes, and silhouettes."""
    if art.front_rgba is not None and art.wash_extent is not None:
        ax.imshow(
            art.front_rgba,
            extent=art.wash_extent,
            origin="lower",
            interpolation="bicubic",
            zorder=40.0,
        )
    for pts, sz, rgba_s, z_s in art.dot_batches:
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=sz,
            c=cast(Any, rgba_s),
            edgecolors="none",
            zorder=z_s,
        )
    for segs, col_l, lw_l, al_l, z_l, sk in art.line_batches:
        for poly in segs:
            ln = ax.plot(
                poly[:, 0],
                poly[:, 1],
                color=col_l,
                linewidth=lw_l,
                alpha=al_l,
                solid_capstyle="round",
                zorder=z_l,
            )[0]
            if sk is not None and sketch_scale > 0.0:
                ln.set_sketch_params(sk[0] * sketch_scale, sk[1], sk[2])


def plot_isosurface(
    source: electronic.VolumetricGrid | str | pathlib.Path,
    level: float | None = None,
    *,
    symmetric: bool | None = None,
    camera: geometry.Camera | str | None = None,
    ax: maxes.Axes | None = None,
    colors: tuple[str, str] = ("#0072B2", "#D55E00"),
    alpha: float = 0.28,
    shade: str = "solid",
    show_atoms: bool = True,
    palette: str = "xkcd",
    preset: str | theme.ThemePreset = "classic",
    title: str | None = None,
) -> maxes.Axes:
    """Plots a 3D shaded orbital or electron-density isosurface.

    Args:
      source: ``VolumetricGrid`` or path/sample (``.cube``, ``CHGCAR``).
      level: Isovalue threshold. Defaults to ``0.35 * max(|data|)``.
      symmetric: If ``True``, extracts both ``+level`` and ``-level`` lobes.
      camera: ``Camera`` object or preset string.
      ax: Optional target Matplotlib ``Axes``.
      colors: ``(positive_hex, negative_hex)`` lobe colors.
      alpha: Surface translucency opacity.
      shade: Shading style: ``"solid"`` (smooth blending), ``"hatch"``
        (parallel pen hatching), ``"crosshatch"`` (comic cross-hatching),
        ``"contour"`` / ``"cross_contour"`` (3D curved form lines),
        ``"stipple"`` (felt-tip stippling), ``"scumble"`` (xkcd scribble),
        or ``"outline"``.
      show_atoms: Whether to draw embedded atomic ball-and-stick structure.
      palette: Atom element color palette when ``show_atoms=True``.
      preset: Visual theme preset name or ``ThemePreset``.
      title: Optional heading above the panel.

    Returns:
      Matplotlib ``Axes`` containing the rendered 3D isosurface.
    """
    preset_obj = theme.get_preset(preset)
    eff_colors = (
        ("#56B4E9", colors[1])
        if (
            primitives.is_dark_color(preset_obj.axes_facecolor)
            and colors[0].strip().lower() == "#0072b2"
        )
        else colors
    )
    grid, meshes = extract_isosurface_meshes(
        source, level, symmetric, eff_colors
    )
    if camera is None or (
        isinstance(camera, str) and camera.strip().lower() == "auto"
    ):
        cam = artistic3d.auto_orbital_camera(meshes, grid)
    else:
        cam = geometry.resolve_camera(
            camera, cell=grid.atoms.cell if grid.atoms is not None else None
        )
    art = artistic3d.build_isosurface_art(
        meshes, grid, cam, alpha, shade=shade
    )
    has_atoms = bool(
        show_atoms and grid.atoms is not None and len(grid.atoms) > 0
    )

    with theme.xkcd_style(preset=preset):
        if ax is None:
            _, ax = plt.subplots(figsize=(5.6, 4.8))
        else:
            theme.style_existing_axes(ax, preset=preset)
        if art.back_rgba is not None and art.wash_extent is not None:
            ax.imshow(
                art.back_rgba,
                extent=art.wash_extent,
                origin="lower",
                interpolation="bicubic",
                zorder=1.4,
            )
        if has_atoms and grid.atoms is not None:
            structure3d.render_structure_3d(
                grid.atoms,
                ax=ax,
                camera=cam,
                style="ball_and_stick",
                palette=palette,
                show_cell=False,
                crystal_axes=False,
                show_bonds=True,
                radius_scale=1.08,
            )
        draw_iso_art_batches(ax, art, preset_obj.sketch_scale)
        if art.all_xy is not None:
            expand_isosurface_limits(ax, art.all_xy, has_atoms)
        ax.set_aspect("equal")
        ax.set_axis_off()
        if title:
            ax.set_title(title, color=preset_obj.ink_color)
    return ax
