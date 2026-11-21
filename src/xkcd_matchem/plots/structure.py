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
"""High-level structural plots for molecules, crystals, slabs, and clusters."""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
import pathlib

from matplotlib import axes as maxes
from matplotlib import patches
from matplotlib import pyplot as plt
import numpy as np

from xkcd_matchem.core import bonding
from xkcd_matchem.core import geometry
from xkcd_matchem.core import structure
from xkcd_matchem.io import loader
from xkcd_matchem.render import callouts
from xkcd_matchem.render import primitives
from xkcd_matchem.render import structure2d
from xkcd_matchem.render import structure3d
from xkcd_matchem.style import palettes
from xkcd_matchem.style import theme


def _coerce_atoms(
    source: structure.AtomsData | str | pathlib.Path,
) -> structure.AtomsData:
    """Loads ``AtomsData`` when ``source`` is a path or sample name."""
    if isinstance(source, structure.AtomsData):
        return source
    loaded = loader.load(source)
    if hasattr(loaded, "atoms") and isinstance(
        loaded.atoms, structure.AtomsData
    ):
        return loaded.atoms
    if not isinstance(loaded, structure.AtomsData):
        raise TypeError(
            f"Expected AtomsData from {source!r}, got {type(loaded)}."
        )
    return loaded


def _pad_structure_view(
    ax: maxes.Axes,
    bottom_frac: float = 0.14,
    top_frac: float = 0.04,
) -> None:
    """Adds vertical headroom and expands data limits to fill the axes box."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    span = max(y1 - y0, 1.0)
    ax.update_datalim(
        [(x0, y0 - bottom_frac * span), (x1, y1 + top_frac * span)]
    )
    ax.margins(0.0)
    ax.set_autoscalex_on(True)
    ax.set_autoscaley_on(True)
    ax.set_aspect("equal", adjustable="datalim")
    ax.autoscale_view()


def _trim_slab_vacuum(
    atoms: structure.AtomsData,
    vacuum_margin: float | None,
) -> tuple[structure.AtomsData, float]:
    """Symmetrically tightens tall c-vacuum for display and returns true vac."""
    if atoms.cell is None or len(atoms) == 0:
        return atoms, 0.0
    m = atoms.cell
    ez = np.cross(m[0], m[1])
    ez_norm = max(float(np.linalg.norm(ez)), 1e-12)
    ez = ez / ez_norm
    if ez[2] < 0.0:
        ez = -ez
    z_proj = atoms.positions @ ez
    z_min = float(np.min(z_proj))
    z_max = float(np.max(z_proj))
    z_span = z_max - z_min
    c_len = float(np.linalg.norm(m[2]))
    true_vac = max(0.0, c_len - z_span)
    if vacuum_margin is None or true_vac <= 2.0 * vacuum_margin + 0.5:
        return atoms, true_vac
    trimmed = atoms.copy()
    new_c_len = z_span + 2.0 * vacuum_margin
    shift = (vacuum_margin - z_min) * ez
    trimmed.positions = trimmed.positions + shift
    new_cell = trimmed.cell.copy()
    new_cell[2] = ez * new_c_len
    trimmed.cell = new_cell
    return trimmed, true_vac


def _slab_side_coordinates(
    atoms: structure.AtomsData,
    vacuum_margin: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
]:
    """Projects slab coordinates onto the side-view ``(ey, ez)`` frame."""
    if atoms.cell is None:
        raise ValueError("Slab side view requires a unit cell.")
    m = atoms.cell
    ex = m[0] / max(float(np.linalg.norm(m[0])), 1e-12)
    ez = np.cross(m[0], m[1])
    ez /= max(float(np.linalg.norm(ez)), 1e-12)
    if ez[2] < 0.0:
        ez = -ez
    ey = np.cross(ez, ex)
    period_y = abs(float(np.dot(m[1], ey)))
    if float(np.dot(m[1], ey)) < 0.0:
        ey = -ey

    pos = atoms.positions
    x_coord = pos @ ex
    y_raw = pos @ ey
    z_raw = pos @ ez
    y_shift = float(
        geometry.best_fractional_shift((y_raw / period_y)[:, None], axes=(0,))[
            0
        ]
    )
    y_coord = np.mod(y_raw + y_shift * period_y, period_y)
    z_min = float(np.min(z_raw))
    z_max = float(np.max(z_raw))
    z_coord = z_raw - z_min + vacuum_margin
    height_c = (z_max - z_min) + 2.0 * vacuum_margin
    return (
        x_coord,
        np.column_stack([y_coord, z_coord]),
        z_coord,
        ey,
        ez,
        period_y,
        height_c,
    )


def _draw_slab_backdrop(
    ax: maxes.Axes,
    z_coord: np.ndarray,
    period_y: float,
    height_c: float,
) -> patches.Rectangle:
    """Draws shaded vacuum and slab bands inside the unit cell frame."""
    clip_rect = patches.Rectangle(
        (0.0, 0.0), period_y, height_c, transform=ax.transData
    )
    ax.add_patch(
        patches.Rectangle(
            (0.0, 0.0),
            period_y,
            height_c,
            facecolor=palettes.VACUUM_TINT,
            edgecolor="none",
            zorder=0.5,
        )
    )
    pad_band = 0.65
    slab_lo = max(float(np.min(z_coord)) - pad_band, 0.0)
    slab_hi = min(float(np.max(z_coord)) + pad_band, height_c)
    ax.add_patch(
        patches.Rectangle(
            (0.0, slab_lo),
            period_y,
            max(0.1, slab_hi - slab_lo),
            facecolor=palettes.SLAB_TINT,
            edgecolor="none",
            zorder=0.7,
        )
    )
    ax.add_patch(
        patches.Rectangle(
            (0.0, 0.0),
            period_y,
            height_c,
            facecolor="none",
            edgecolor=palettes.INK,
            linewidth=2.0,
            zorder=600.0,
        )
    )
    return clip_rect


def _build_slab_side_items(
    atoms: structure.AtomsData,
    x_coord: np.ndarray,
    xy_side: np.ndarray,
    ey: np.ndarray,
    ez: np.ndarray,
    colors: list[str],
) -> list[tuple[float, str, object]]:
    """Builds depth-sorted half-bond and atom items for slab side views."""
    if atoms.cell is None:
        raise ValueError("Slab side view requires a unit cell.")
    bonds = list(atoms.bonds) or bonding.infer_bonds(atoms)
    frac = atoms.fractional_positions()
    items: list[tuple[float, str, object]] = [
        (float(x_coord[i]), "atom", i) for i in range(len(atoms))
    ]
    for b in bonds:
        vec = (
            frac[b.j] + np.asarray(b.shift, dtype=np.float64) - frac[b.i]
        ) @ atoms.cell
        half = np.array(
            [0.5 * float(np.dot(vec, ey)), 0.5 * float(np.dot(vec, ez))],
            dtype=np.float64,
        )
        for idx, sign in ((b.i, 1.0), (b.j, -1.0)):
            p0 = xy_side[idx]
            items.append(
                (
                    float(x_coord[idx]) - 0.06,
                    "half",
                    (p0, p0 + sign * half, colors[idx]),
                )
            )
    items.sort(key=lambda entry: entry[0])
    return items


def _estimate_slab_widths(
    ax: maxes.Axes,
    period_y: float,
    height_c: float,
) -> tuple[float, float]:
    """Estimates responsive bond and atom stroke widths for slab side views."""
    fig_w, fig_h = ax.figure.get_size_inches()
    pos_ax = ax.get_position()
    pt_per_a = min(
        max(float(pos_ax.width * fig_w) * 72.0, 90.0) / max(period_y, 2.0),
        max(float(pos_ax.height * fig_h) * 72.0, 90.0) / max(height_c, 2.0),
    )
    return (
        float(np.clip(0.24 * pt_per_a, 1.8, 4.8)),
        float(np.clip(0.11 * pt_per_a, 1.3, 2.1)),
    )


def _draw_slab_atoms_and_bonds(
    ax: maxes.Axes,
    atoms: structure.AtomsData,
    side_geom: tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        float,
        float,
    ],
    palette: str | Mapping[str, str],
    clip_rect: patches.Rectangle,
) -> int:
    """Depth-sorts and draws clipped bonds and spheres in slab side view."""
    x_coord, xy_side, ey, ez, period_y, height_c = side_geom
    colors = [
        palettes.get_element_color(sym, scheme=palette) for sym in atoms.symbols
    ]
    items = _build_slab_side_items(atoms, x_coord, xy_side, ey, ez, colors)
    bond_w, atom_lw = _estimate_slab_widths(ax, period_y, height_c)
    before = len(ax.patches)
    for rank, (_, kind, payload) in enumerate(items):
        z_ord = 2.0 + 0.02 * rank
        if kind == "atom":
            idx = int(payload)  # type: ignore[arg-type]
            r_cov = 0.34 if atoms.symbols[idx] in ("H", "O", "C", "N") else 0.44
            primitives.draw_atom_sphere(
                ax,
                float(xy_side[idx, 0]),
                float(xy_side[idx, 1]),
                radius=r_cov,
                facecolor=colors[idx],
                linewidth=atom_lw,
                zorder=z_ord,
            )
        else:
            p0, p1, col = payload  # type: ignore[misc]
            primitives.draw_half_bond(
                ax, p0, p1, color=col, width=bond_w, zorder=z_ord
            )
    for patch in ax.patches[before:]:
        patch.set_clip_path(clip_rect)
    for line in ax.lines:
        line.set_clip_path(clip_rect)
    return len(items)


def _render_slab_side_view(
    atoms: structure.AtomsData,
    ax: maxes.Axes,
    palette: str | Mapping[str, str],
    vacuum_margin: float,
    vacuum_callout: bool,
    true_vac: float,
) -> None:
    """Draws a crisp 2D orthographic slab side view looking along ``a``."""
    x_coord, xy_side, z_coord, ey, ez, period_y, height_c = (
        _slab_side_coordinates(atoms, vacuum_margin)
    )
    clip_rect = _draw_slab_backdrop(ax, z_coord, period_y, height_c)
    n_items = _draw_slab_atoms_and_bonds(
        ax,
        atoms,
        (x_coord, xy_side, ey, ez, period_y, height_c),
        palette,
        clip_rect,
    )

    ax.text(
        0.5 * period_y,
        height_c - 0.45 * vacuum_margin,
        "vacuum",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#2E4A5A",
        zorder=2.0 + 0.02 * (n_items + 2),
    )
    ink = str(plt.rcParams.get("axes.edgecolor", palettes.INK))
    if vacuum_callout:
        callouts.draw_callout(
            ax,
            f"Vacuum ~ {true_vac:.1f} A",
            xy=(0.74 * period_y, height_c - 0.35 * vacuum_margin),
            xytext=(0.74 * period_y, height_c + 0.065 * height_c),
            color="#0072B2",
            fontsize=9.0,
        )
    ax.text(
        0.5 * period_y,
        -0.06 * height_c,
        "b",
        ha="center",
        va="top",
        fontsize=11.0,
        color=ink,
    )
    ax.text(
        -0.06 * period_y,
        0.5 * height_c,
        "c",
        ha="right",
        va="center",
        fontsize=11.0,
        color=ink,
    )
    x_pad = max(0.9, 0.14 * period_y)
    y_pad_bot = max(1.0, 0.16 * height_c)
    y_pad_top = max(1.4, 0.25 * height_c) if vacuum_callout else y_pad_bot
    ax.set_xlim(-x_pad, period_y + x_pad)
    ax.set_ylim(-y_pad_bot - 0.12 * height_c, height_c + y_pad_top)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_axis_off()


def plot_molecule(
    source: structure.AtomsData | str | pathlib.Path,
    ax: maxes.Axes | None = None,
    mode: str = "3d",
    style: str = "ball_and_stick",
    camera: geometry.Camera | str | None = None,
    palette: str | Mapping[str, str] = "xkcd",
    show_labels: bool = False,
    show_legend: bool = True,
    preset: str | theme.ThemePreset = "classic",
    title: str | None = None,
) -> maxes.Axes:
    """Plots a molecule in xkcd-style 3D depth-sorted or 2D skeletal view.

    Args:
      source: ``AtomsData``, file path, or bundled sample filename.
      ax: Optional target Matplotlib ``Axes``.
      mode: ``"3d"`` or ``"2d"``.
      style: 3D rendering mode (``"ball_and_stick"``, ``"spacefill"``,
        ``"stick"``).
      camera: ``Camera`` object or preset string (``"isometric"``, ``"front"``,
        ``"top"``, etc.).
      palette: Scheme name or custom ``{symbol: color}`` dictionary override.
      show_labels: Whether to print element symbols on 3D spheres.
      show_legend: Whether to draw an inline bottom element legend.
      preset: Visual theme preset name or custom ``ThemePreset``.
      title: Optional title shown above the panel.

    Returns:
      Matplotlib ``Axes`` containing the rendered molecule.
    """
    atoms = _coerce_atoms(source)
    with theme.xkcd_style(preset=preset):
        if ax is None:
            _, ax = plt.subplots(figsize=(5.4, 4.4))
        else:
            theme.style_existing_axes(ax, preset=preset)
        if mode.lower() == "2d":
            structure2d.render_structure_2d(atoms, ax=ax, palette=palette)
        else:
            structure3d.render_structure_3d(
                atoms,
                ax=ax,
                camera=camera,
                style=style,
                palette=palette,
                show_cell=False,
                show_labels=show_labels,
            )
            if show_legend:
                _pad_structure_view(ax, bottom_frac=0.14, top_frac=0.04)
                callouts.draw_element_legend(ax, atoms.symbols, scheme=palette)
            else:
                _pad_structure_view(ax, bottom_frac=0.04, top_frac=0.04)
        if title:
            ax.set_title(title)
    return ax


def plot_crystal(
    source: structure.AtomsData | str | pathlib.Path,
    ax: maxes.Axes | None = None,
    supercell: tuple[int, int, int] = (1, 1, 1),
    style: str = "ball_and_stick",
    camera: geometry.Camera | str | None = None,
    palette: str | Mapping[str, str] = "xkcd",
    show_cell: bool | str = True,
    show_labels: bool = False,
    show_legend: bool = True,
    center_in_cell: bool = True,
    miller_planes: Sequence[tuple[int, int, int]] | None = None,
    preset: str | theme.ThemePreset = "classic",
    title: str | None = None,
) -> maxes.Axes:
    """Plots a periodic crystal or supercell with clean unit cell & axes."""
    atoms = _coerce_atoms(source).wrap_positions()
    if center_in_cell:
        atoms = atoms.center_in_cell()
    if supercell != (1, 1, 1) and atoms.cell is not None:
        atoms = atoms.repeat(supercell)
    with theme.xkcd_style(preset=preset):
        if ax is None:
            _, ax = plt.subplots(figsize=(5.8, 5.0))
        else:
            theme.style_existing_axes(ax, preset=preset)
        structure3d.render_structure_3d(
            atoms,
            ax=ax,
            camera=camera,
            style=style,
            palette=palette,
            show_cell=show_cell,
            show_labels=show_labels,
            miller_planes=miller_planes,
        )
        if show_legend:
            _pad_structure_view(ax, bottom_frac=0.17, top_frac=0.04)
            callouts.draw_element_legend(ax, atoms.symbols, scheme=palette)
        else:
            _pad_structure_view(ax, bottom_frac=0.05, top_frac=0.05)
        if title:
            ax.set_title(title)
    return ax


def plot_slab(
    source: structure.AtomsData | str | pathlib.Path,
    ax: maxes.Axes | None = None,
    supercell: tuple[int, int, int] = (2, 1, 1),
    view: str = "side",
    camera: geometry.Camera | str | None = None,
    palette: str | Mapping[str, str] = "xkcd",
    vacuum_callout: bool = True,
    vacuum_margin: float | None = 2.6,
    preset: str | theme.ThemePreset = "classic",
    title: str | None = None,
) -> maxes.Axes:
    """Plots a surface slab in shaded side view or trimmed isometric view."""
    atoms = _coerce_atoms(source).wrap_positions()
    if supercell != (1, 1, 1) and atoms.cell is not None:
        atoms = atoms.repeat(supercell)
    trimmed, true_vac = _trim_slab_vacuum(atoms, vacuum_margin=vacuum_margin)
    with theme.xkcd_style(preset=preset):
        if ax is None:
            _, ax = plt.subplots(figsize=(5.8, 5.2))
        else:
            theme.style_existing_axes(ax, preset=preset)
        if view.lower() == "side" and trimmed.cell is not None:
            margin_v = 2.6 if vacuum_margin is None else vacuum_margin
            _render_slab_side_view(
                trimmed,
                ax=ax,
                palette=palette,
                vacuum_margin=margin_v,
                vacuum_callout=vacuum_callout,
                true_vac=true_vac,
            )
            callouts.draw_element_legend(ax, atoms.symbols, scheme=palette)
        else:
            cam = (
                geometry.Camera.from_preset("isometric", cell=trimmed.cell)
                if camera is None
                else geometry.resolve_camera(camera, cell=trimmed.cell)
            )
            structure3d.render_structure_3d(
                trimmed,
                ax=ax,
                camera=cam,
                style="ball_and_stick",
                palette=palette,
                show_cell=True,
            )
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            _pad_structure_view(ax, bottom_frac=0.17, top_frac=0.14)
            if vacuum_callout and len(trimmed) > 0:
                x_span = max(xlim[1] - xlim[0], 1.0)
                y_span = max(ylim[1] - ylim[0], 1.0)
                callouts.draw_callout(
                    ax,
                    f"Vacuum ~ {true_vac:.1f} A",
                    xy=(xlim[0] + 0.62 * x_span, ylim[0] + 0.72 * y_span),
                    xytext=(xlim[0] - 0.12 * x_span, ylim[0] + 0.94 * y_span),
                    color="#0072B2",
                )
            callouts.draw_element_legend(ax, atoms.symbols, scheme=palette)
        if title:
            ax.set_title(title)
    return ax


def plot_coordination(
    source: structure.AtomsData | str | pathlib.Path,
    center: str,
    ligand: str | None = None,
    ax: maxes.Axes | None = None,
    supercell: tuple[int, int, int] = (1, 1, 1),
    camera: geometry.Camera | str | None = None,
    palette: str | Mapping[str, str] = "xkcd",
    preset: str | theme.ThemePreset = "classic",
    title: str | None = None,
) -> maxes.Axes:
    """Plots a crystal with translucent ConvexHull coordination polyhedra."""
    atoms = _coerce_atoms(source).wrap_positions()
    if supercell != (1, 1, 1) and atoms.cell is not None:
        atoms = atoms.repeat(supercell)
    trimmed, _ = _trim_slab_vacuum(atoms, vacuum_margin=2.2)
    trimmed = trimmed.center_in_cell(axes=(0, 1))
    with theme.xkcd_style(preset=preset):
        if ax is None:
            _, ax = plt.subplots(figsize=(5.8, 5.0))
        else:
            theme.style_existing_axes(ax, preset=preset)
        structure3d.render_structure_3d(
            trimmed,
            ax=ax,
            camera=camera,
            style="polyhedra",
            palette=palette,
            polyhedra=(center, ligand),
            show_cell=True,
        )
        _pad_structure_view(ax, bottom_frac=0.17, top_frac=0.06)
        callouts.draw_element_legend(ax, atoms.symbols, scheme=palette)
        if title:
            ax.set_title(title)
    return ax
