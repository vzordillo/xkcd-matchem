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
"""Animated GIF helpers for 3D structure orbits and orbital isosurfaces."""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
import importlib
from importlib import util
import io
import pathlib
from typing import Any
from typing import cast

from matplotlib import axes as maxes
from matplotlib import figure as mfigure
from matplotlib import pyplot as plt
import numpy as np

from xkcd_matchem.core import electronic
from xkcd_matchem.core import exceptions
from xkcd_matchem.core import geometry
from xkcd_matchem.core import isosurface
from xkcd_matchem.core import structure
from xkcd_matchem.io import loader
from xkcd_matchem.render import artistic3d
from xkcd_matchem.render import callouts
from xkcd_matchem.render import structure3d
from xkcd_matchem.style import theme


__all__ = [
    "animate_isosurface",
    "animate_orbit",
]


def _require_pillow() -> Any:
    """Imports ``PIL.Image`` lazily or raises ``OptionalDependencyError``."""
    if util.find_spec("PIL") is None:
        raise exceptions.OptionalDependencyError(
            "GIF animation requires 'pillow'. Install via: "
            "pip install 'xkcd-matchem[animation]'"
        )
    return importlib.import_module("PIL.Image")


def _figure_to_pil(fig: mfigure.Figure, dpi: int, pil_image: Any) -> Any:
    """Rasterizes a Matplotlib figure into an RGB ``PIL.Image`` frame."""
    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=dpi,
        facecolor=fig.get_facecolor(),
    )
    buf.seek(0)
    with pil_image.open(buf) as img:
        return img.convert("RGB")


def _save_gif_frames(
    frames: Sequence[Any],
    out_path: str | pathlib.Path,
    fps: int,
) -> pathlib.Path:
    """Writes RGB PIL frames to an optimized looping GIF."""
    if not frames:
        raise exceptions.StructureError("No frames were rendered for GIF.")
    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(20, int(round(1000.0 / max(1, fps))))
    frames[0].save(
        out,
        save_all=True,
        append_images=list(frames[1:]),
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    return out


def _coerce_atoms(
    source: structure.AtomsData | str | pathlib.Path,
) -> structure.AtomsData:
    """Coerces a path, sample name, or ``AtomsData`` into ``AtomsData``."""
    if isinstance(source, structure.AtomsData):
        return source
    loaded = loader.load(source)
    if hasattr(loaded, "atoms") and isinstance(
        loaded.atoms, structure.AtomsData
    ):
        return loaded.atoms
    if not isinstance(loaded, structure.AtomsData):
        raise exceptions.StructureError(
            f"Expected AtomsData from {source!r}, got {type(loaded).__name__}."
        )
    return loaded


def _coerce_volumetric_grid(
    source: electronic.VolumetricGrid | str | pathlib.Path,
) -> electronic.VolumetricGrid:
    """Coerces a path, sample name, or ``VolumetricGrid`` into grid data."""
    if isinstance(source, electronic.VolumetricGrid):
        return source
    loaded = loader.load(source)
    if not isinstance(loaded, electronic.VolumetricGrid):
        raise exceptions.StructureError(
            f"Expected VolumetricGrid, got {type(loaded).__name__}."
        )
    return loaded


def _trim_slab_vacuum(
    atoms: structure.AtomsData,
    vacuum_margin: float,
) -> structure.AtomsData:
    """Symmetrically trims tall c-axis slab vacuum for tight orbit framing."""
    if atoms.cell is None or len(atoms) == 0:
        return atoms
    cell = atoms.cell
    ez = np.cross(cell[0], cell[1])
    ez_norm = max(float(np.linalg.norm(ez)), 1e-12)
    ez = ez / ez_norm
    if ez[2] < 0.0:
        ez = -ez
    z_proj = atoms.positions @ ez
    z_min, z_max = float(np.min(z_proj)), float(np.max(z_proj))
    c_len = float(np.linalg.norm(cell[2]))
    true_vac = max(0.0, c_len - (z_max - z_min))
    if true_vac <= 2.0 * vacuum_margin + 0.5:
        return atoms
    trimmed = atoms.copy()
    trimmed.positions = trimmed.positions + (vacuum_margin - z_min) * ez
    new_cell = cell.copy()
    new_cell[2] = ez * ((z_max - z_min) + 2.0 * vacuum_margin)
    trimmed.cell = new_cell
    return trimmed


def _prepare_orbit_atoms(
    source: structure.AtomsData | str | pathlib.Path,
    supercell: tuple[int, int, int] | None,
    polyhedra: tuple[str, str | None] | None,
) -> structure.AtomsData:
    """Wraps, optionally tiles, and centers atoms for smooth 360-deg orbits."""
    atoms = _coerce_atoms(source)
    if len(atoms) == 0:
        raise exceptions.StructureError("Cannot animate an empty structure.")
    if atoms.cell is not None:
        atoms = atoms.wrap_positions()
        if supercell is not None and supercell != (1, 1, 1):
            atoms = atoms.repeat(supercell)
        if polyhedra is not None:
            atoms = _trim_slab_vacuum(atoms, vacuum_margin=2.2)
            atoms = atoms.center_in_cell(axes=(0, 1))
    return atoms


def _orbit_viewport(
    atoms: structure.AtomsData,
    cameras: Sequence[geometry.Camera],
    show_cell: bool | str,
    show_legend: bool,
) -> tuple[float, float, float, float]:
    """Computes a shared 2D viewport across all orbit cameras without jitter."""
    pivot = (
        0.5 * (atoms.cell[0] + atoms.cell[1] + atoms.cell[2])
        if atoms.cell is not None
        else np.mean(atoms.positions, axis=0)
    )
    chunks: list[np.ndarray] = []
    for cam in cameras:
        span = geometry.scene_projection_span(
            atoms.positions, cam, center=pivot, cell=atoms.cell
        )
        xy, _ = geometry.project_points(
            atoms.positions, cam, center=pivot, span=span
        )
        chunks.append(xy)
        if show_cell and atoms.cell is not None:
            c2d, _, _ = geometry.cell_corners_projected(
                atoms.cell, cam, center=pivot, span=span
            )
            chunks.append(c2d)
    all_pts = np.vstack(chunks)
    x_min, x_max = float(np.min(all_pts[:, 0])), float(np.max(all_pts[:, 0]))
    y_min, y_max = float(np.min(all_pts[:, 1])), float(np.max(all_pts[:, 1]))
    span_max = max(x_max - x_min, y_max - y_min, 1.0)
    pad = max(0.85, 0.11 * span_max)
    bottom_pad = pad + (0.16 * max(y_max - y_min, 1.0) if show_legend else 0.0)
    return (x_min - pad, x_max + pad, y_min - bottom_pad, y_max + pad)


def _orbit_cameras(
    start_azimuth: float,
    n_frames: int,
    elevation: float,
    perspective: float,
) -> list[geometry.Camera]:
    """Builds evenly spaced 360-degree camera viewpoints."""
    azimuths = start_azimuth + np.linspace(0.0, 360.0, n_frames, endpoint=False)
    return [
        geometry.Camera(
            azimuth=float(az),
            elevation=elevation,
            roll=0.0,
            perspective=perspective,
            zoom=1.0,
        )
        for az in azimuths
    ]


def _render_orbit_frame(
    atoms: structure.AtomsData,
    cam: geometry.Camera,
    viewport: tuple[float, float, float, float],
    *,
    render_style: str,
    polyhedra: tuple[str, str | None] | None,
    show_cell: bool | str,
    crystal_axes: bool | str | None,
    show_bonds: bool,
    show_hbonds: bool | str,
    show_legend: bool,
    palette: str | Mapping[str, str],
    preset: str | theme.ThemePreset,
    figsize: tuple[float, float],
    dpi: int,
    title: str | None,
    pil_image: Any,
) -> Any:
    """Renders one locked-viewport orbit frame into an RGB PIL image."""
    preset_obj = theme.get_preset(preset)
    x_lo, x_hi, y_lo, y_hi = viewport
    with theme.xkcd_style(preset=preset):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        try:
            fig.patch.set_facecolor(preset_obj.figure_facecolor)
            fig.patch.set_edgecolor(preset_obj.figure_facecolor)
            fig.patch.set_sketch_params(0.0, 0.0, 0.0)
            ax.patch.set_sketch_params(0.0, 0.0, 0.0)
            structure3d.render_structure_3d(
                atoms,
                ax=ax,
                camera=cam,
                style=render_style,
                palette=palette,
                show_cell=show_cell,
                crystal_axes=crystal_axes,
                show_bonds=show_bonds,
                show_hbonds=show_hbonds,
                polyhedra=polyhedra,
            )
            ax.set_xlim(x_lo, x_hi)
            ax.set_ylim(y_lo, y_hi)
            ax.set_aspect("equal", adjustable="box")
            if show_legend:
                callouts.draw_element_legend(ax, atoms.symbols, scheme=palette)
            if title:
                ax.set_title(title, color=preset_obj.ink_color)
            fig.tight_layout()
            return _figure_to_pil(fig, dpi=dpi, pil_image=pil_image)
        finally:
            plt.close(fig)


def animate_orbit(
    source: structure.AtomsData | str | pathlib.Path,
    out_path: str | pathlib.Path,
    *,
    n_frames: int = 24,
    fps: int = 10,
    elevation: float = 24.0,
    start_azimuth: float = -118.0,
    perspective: float = 0.35,
    style: str = "ball_and_stick",
    polyhedra: tuple[str, str | None] | None = None,
    supercell: tuple[int, int, int] | None = None,
    show_cell: bool | str = True,
    crystal_axes: bool | str | None = None,
    show_bonds: bool | None = None,
    show_hbonds: bool | str = False,
    show_legend: bool = True,
    palette: str | Mapping[str, str] = "xkcd",
    preset: str | theme.ThemePreset = "classic",
    figsize: tuple[float, float] = (5.2, 4.4),
    dpi: int = 110,
    title: str | None = None,
) -> pathlib.Path:
    """Renders a smooth 360-degree camera-orbit GIF of a molecule or crystal.

    Locks a shared projected viewport across all rotation angles so atoms,
    coordination polyhedra, and unit-cell edges orbit in place without zoom
    jitter.

    Args:
      source: ``AtomsData``, file path, or bundled sample filename (for
        example, ``"tio2_slab.poscar"`` or ``"aspirin.xyz"``).
      out_path: Destination ``.gif`` file path.
      n_frames: Number of evenly spaced azimuth steps across 360 degrees.
      fps: Output GIF playback rate in frames per second.
      elevation: Constant camera elevation angle in degrees.
      start_azimuth: Starting azimuth angle in degrees.
      perspective: Camera perspective factor (``0.0`` for orthographic).
      style: 3D rendering style (``"ball_and_stick"``, ``"spacefill"``,
        ``"stick"``, or ``"polyhedra"``).
      polyhedra: Optional ``(center_symbol, ligand_symbol)`` tuple to draw
        depth-sorted coordination polyhedra (automatically switches style to
        ``"polyhedra"`` when ``style == "ball_and_stick"``).
      supercell: Optional ``(na, nb, nc)`` periodic repeat tuple.
      show_cell: Whether to render the periodic unit-cell box.
      crystal_axes: Optional crystallographic axis triad placement (``False``,
        ``"corner"``, or ``"origin"``).
      show_bonds: Override covalent bond drawing (defaults to ``False`` when
        ``polyhedra`` is set, else ``True``).
      show_hbonds: Whether to overlay dashed geometric ``D-H...A`` hydrogen
        bonds on every frame (``True`` or a hex color string).
      show_legend: Whether to display the inline element color legend.
      palette: Element color palette name or custom ``{symbol: hex}`` mapping.
      preset: Visual theme preset name or ``ThemePreset``.
      figsize: Matplotlib figure size ``(width, height)`` in inches.
      dpi: Frame rasterization resolution in dots per inch.
      title: Optional heading above each frame.

    Returns:
      ``pathlib.Path`` pointing to the written looping GIF file.
    """
    pil_image = _require_pillow()
    if n_frames < 2:
        raise exceptions.StructureError("n_frames must be at least 2.")
    atoms = _prepare_orbit_atoms(source, supercell, polyhedra)
    render_style = (
        "polyhedra"
        if (polyhedra is not None and style == "ball_and_stick")
        else style
    )
    bonds_flag = (polyhedra is None) if show_bonds is None else show_bonds
    cameras = _orbit_cameras(start_azimuth, n_frames, elevation, perspective)
    viewport = _orbit_viewport(
        atoms, cameras, show_cell=show_cell, show_legend=show_legend
    )
    frames = [
        _render_orbit_frame(
            atoms,
            cam,
            viewport,
            render_style=render_style,
            polyhedra=polyhedra,
            show_cell=show_cell,
            crystal_axes=crystal_axes,
            show_bonds=bonds_flag,
            show_hbonds=show_hbonds,
            show_legend=show_legend,
            palette=palette,
            preset=preset,
            figsize=figsize,
            dpi=dpi,
            title=title,
            pil_image=pil_image,
        )
        for cam in cameras
    ]
    return _save_gif_frames(frames, out_path=out_path, fps=fps)


def _extract_iso_meshes(
    source: electronic.VolumetricGrid | str | pathlib.Path,
    level: float | None,
    symmetric: bool | None,
    colors: tuple[str, str],
) -> tuple[
    electronic.VolumetricGrid,
    list[tuple[isosurface.IsosurfaceMesh, str]],
]:
    """Upsamples a volumetric grid once and extracts orbital lobes."""
    raw_grid = _coerce_volumetric_grid(source)
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


def _iso_orbit_viewport(
    grid: electronic.VolumetricGrid,
    meshes: Sequence[tuple[isosurface.IsosurfaceMesh, str]],
    cameras: Sequence[geometry.Camera],
) -> tuple[float, float, float, float]:
    """Computes a shared 2D viewport across all isosurface orbit cameras."""
    chunks: list[np.ndarray] = []
    for cam in cameras:
        pivot, span, atom_xy, _ = artistic3d.iso_scene_frame(meshes, grid, cam)
        if atom_xy is not None and len(atom_xy) > 0:
            chunks.append(atom_xy)
        for mesh, _ in meshes:
            if len(mesh.vertices) > 0:
                xy_m, _ = geometry.project_points(
                    mesh.vertices, cam, center=pivot, span=span
                )
                chunks.append(xy_m)
    if not chunks:
        return (-2.0, 2.0, -2.0, 2.0)
    all_pts = np.vstack(chunks)
    x_min, x_max = float(np.min(all_pts[:, 0])), float(np.max(all_pts[:, 0]))
    y_min, y_max = float(np.min(all_pts[:, 1])), float(np.max(all_pts[:, 1]))
    pad = max(0.45, 0.10 * max(x_max - x_min, y_max - y_min, 1.0))
    return (x_min - pad, x_max + pad, y_min - pad, y_max + pad)


def _draw_iso_batches(
    ax: maxes.Axes,
    art: artistic3d.ShadeArtData,
    sketch_scale: float,
) -> None:
    """Draws front wash, stipple dots, comic hatching, and silhouettes."""
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


def _render_iso_frame(
    grid: electronic.VolumetricGrid,
    meshes: Sequence[tuple[isosurface.IsosurfaceMesh, str]],
    cam: geometry.Camera,
    viewport: tuple[float, float, float, float],
    *,
    shade_key: str,
    alpha: float,
    show_atoms: bool,
    palette: str | Mapping[str, str],
    preset: str | theme.ThemePreset,
    figsize: tuple[float, float],
    dpi: int,
    title: str | None,
    pil_image: Any,
) -> Any:
    """Renders one locked-viewport 3D isosurface frame into an RGB PIL image."""
    preset_obj = theme.get_preset(preset)
    x_lo, x_hi, y_lo, y_hi = viewport
    has_atoms = bool(
        show_atoms and grid.atoms is not None and len(grid.atoms) > 0
    )
    art = artistic3d.build_isosurface_art(
        meshes, grid, cam, alpha, shade=shade_key
    )
    with theme.xkcd_style(preset=preset):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        try:
            fig.patch.set_facecolor(preset_obj.figure_facecolor)
            fig.patch.set_edgecolor(preset_obj.figure_facecolor)
            fig.patch.set_sketch_params(0.0, 0.0, 0.0)
            ax.set_facecolor(preset_obj.axes_facecolor)
            ax.patch.set_sketch_params(0.0, 0.0, 0.0)
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
            _draw_iso_batches(ax, art, preset_obj.sketch_scale)
            ax.set_xlim(x_lo, x_hi)
            ax.set_ylim(y_lo, y_hi)
            ax.set_aspect("equal", adjustable="box")
            ax.set_axis_off()
            if title:
                ax.set_title(title, color=preset_obj.ink_color)
            fig.tight_layout()
            return _figure_to_pil(fig, dpi=dpi, pil_image=pil_image)
        finally:
            plt.close(fig)


def animate_isosurface(
    source: electronic.VolumetricGrid | str | pathlib.Path,
    out_path: str | pathlib.Path,
    *,
    level: float | None = None,
    symmetric: bool | None = None,
    n_frames: int = 18,
    fps: int = 8,
    elevation: float = 35.264,
    start_azimuth: float = 45.0,
    perspective: float = 0.0,
    shade: str = "crosshatch",
    colors: tuple[str, str] = ("#0072B2", "#D55E00"),
    alpha: float = 0.28,
    show_atoms: bool = True,
    palette: str | Mapping[str, str] = "xkcd",
    preset: str | theme.ThemePreset = "classic",
    figsize: tuple[float, float] = (5.2, 4.6),
    dpi: int = 110,
    title: str | None = None,
) -> pathlib.Path:
    """Renders a smooth 360-degree orbit GIF of a 3D orbital isosurface.

    Extracts the Marching Tetrahedra mesh once and orbits the camera across
    ``n_frames`` azimuth steps with locked viewport bounds and depth-occluded
    comic pen-and-ink shading.

    Args:
      source: ``VolumetricGrid``, file path, or bundled sample filename (for
        example, ``"water_homo.cube"``).
      out_path: Destination ``.gif`` file path.
      level: Isovalue threshold. Defaults to ``0.35 * max(|data|)``.
      symmetric: If ``True``, renders both ``+level`` and ``-level`` lobes.
      n_frames: Number of evenly spaced azimuth steps across 360 degrees.
      fps: Output GIF playback rate in frames per second.
      elevation: Constant camera elevation angle in degrees (default ``35.264``
        for an orthographic isometric orbit).
      start_azimuth: Starting azimuth angle in degrees.
      perspective: Camera perspective factor (``0.0`` for orthographic).
      shade: Pen-and-ink shading mode (``"solid"``, ``"hatch"``,
        ``"crosshatch"``, ``"contour"``, ``"cross_contour"``, ``"stipple"``,
        ``"scumble"``, or ``"outline"``).
      colors: ``(positive_hex, negative_hex)`` orbital lobe colors.
      alpha: Translucent lobe wash opacity.
      show_atoms: Whether to render the embedded ball-and-stick molecule.
      palette: Atom color palette name or ``{symbol: hex}`` mapping.
      preset: Visual theme preset name or ``ThemePreset``.
      figsize: Matplotlib figure size ``(width, height)`` in inches.
      dpi: Frame rasterization resolution in dots per inch.
      title: Optional heading above each frame.

    Returns:
      ``pathlib.Path`` pointing to the written looping GIF file.
    """
    pil_image = _require_pillow()
    if n_frames < 2:
        raise exceptions.StructureError("n_frames must be at least 2.")
    shade_key = artistic3d.normalize_iso_shade(shade)
    grid, meshes = _extract_iso_meshes(source, level, symmetric, colors)
    cameras = _orbit_cameras(start_azimuth, n_frames, elevation, perspective)
    viewport = _iso_orbit_viewport(grid, meshes, cameras)
    frames = [
        _render_iso_frame(
            grid,
            meshes,
            cam,
            viewport,
            shade_key=shade_key,
            alpha=alpha,
            show_atoms=show_atoms,
            palette=palette,
            preset=preset,
            figsize=figsize,
            dpi=dpi,
            title=title,
            pil_image=pil_image,
        )
        for cam in cameras
    ]
    return _save_gif_frames(frames, out_path=out_path, fps=fps)
