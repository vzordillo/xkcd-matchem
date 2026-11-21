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

from collections.abc import Sequence
import pathlib

from matplotlib import axes as maxes
from matplotlib import pyplot as plt
import numpy as np

from xkcd_matchem.core import geometry
from xkcd_matchem.core import structure
from xkcd_matchem.io import loader
from xkcd_matchem.render import callouts
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


def plot_molecule(
    source: structure.AtomsData | str | pathlib.Path,
    ax: maxes.Axes | None = None,
    mode: str = "3d",
    style: str = "ball_and_stick",
    camera: geometry.Camera | None = None,
    palette: str = "xkcd",
    show_labels: bool = False,
    show_legend: bool = True,
    preset: str = "classic",
    title: str | None = None,
) -> maxes.Axes:
    """Plots a molecule in hand-drawn 3D depth-sorted or 2D skeletal view.

    Args:
      source: ``AtomsData``, file path, or bundled sample filename.
      ax: Optional target Matplotlib ``Axes``.
      mode: ``"3d"`` or ``"2d"``.
      style: 3D rendering mode (``"ball_and_stick"``, ``"spacefill"``,
        ``"stick"``).
      camera: Optional 3D camera viewpoint.
      palette: Element palette name (``"xkcd"``, ``"jmol"``, ``"colorblind"``).
      show_labels: Whether to print element symbols on 3D spheres.
      show_legend: Whether to include a compact element legend swatch box.
      preset: Visual theme preset name.
      title: Optional title shown above the panel.

    Returns:
      Matplotlib ``Axes`` containing the rendered molecule.
    """
    atoms = _coerce_atoms(source)
    with theme.xkcd_style(preset=preset):
        if ax is None:
            _, ax = plt.subplots(figsize=(5.4, 4.4))
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
    camera: geometry.Camera | None = None,
    palette: str = "xkcd",
    show_cell: bool = True,
    show_labels: bool = False,
    show_legend: bool = True,
    miller_planes: Sequence[tuple[int, int, int]] | None = None,
    preset: str = "classic",
    title: str | None = None,
) -> maxes.Axes:
    """Plots a periodic crystal or supercell with unit cell edges and planes."""
    atoms = _coerce_atoms(source).wrap_positions()
    if supercell != (1, 1, 1) and atoms.cell is not None:
        atoms = atoms.repeat(supercell)
    with theme.xkcd_style(preset=preset):
        if ax is None:
            _, ax = plt.subplots(figsize=(5.8, 5.0))
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
            _pad_structure_view(ax, bottom_frac=0.15, top_frac=0.04)
            callouts.draw_element_legend(ax, atoms.symbols, scheme=palette)
        else:
            _pad_structure_view(ax, bottom_frac=0.04, top_frac=0.04)
        if title:
            ax.set_title(title)
    return ax


def plot_slab(
    source: structure.AtomsData | str | pathlib.Path,
    ax: maxes.Axes | None = None,
    supercell: tuple[int, int, int] = (2, 1, 1),
    camera: geometry.Camera | None = None,
    palette: str = "xkcd",
    vacuum_callout: bool = True,
    preset: str = "classic",
    title: str | None = None,
) -> maxes.Axes:
    """Plots a side-view surface slab highlighting vacuum gap thickness."""
    atoms = _coerce_atoms(source).wrap_positions()
    if supercell != (1, 1, 1) and atoms.cell is not None:
        atoms = atoms.repeat(supercell)
    cam = (
        geometry.Camera(azimuth=85.0, elevation=8.0)
        if camera is None
        else camera
    )
    with theme.xkcd_style(preset=preset):
        if ax is None:
            _, ax = plt.subplots(figsize=(5.8, 5.2))
        structure3d.render_structure_3d(
            atoms,
            ax=ax,
            camera=cam,
            style="ball_and_stick",
            palette=palette,
            show_cell=True,
        )
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        _pad_structure_view(ax, bottom_frac=0.16, top_frac=0.14)
        if vacuum_callout and atoms.cell is not None and len(atoms) > 0:
            z_span = float(
                np.max(atoms.positions[:, 2]) - np.min(atoms.positions[:, 2])
            )
            c_len = float(np.linalg.norm(atoms.cell[2]))
            vac = max(0.0, c_len - z_span)
            x_span = max(xlim[1] - xlim[0], 1.0)
            y_span = max(ylim[1] - ylim[0], 1.0)
            callouts.draw_callout(
                ax,
                f"Vacuum ~ {vac:.1f} A",
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
    camera: geometry.Camera | None = None,
    palette: str = "xkcd",
    preset: str = "classic",
    title: str | None = None,
) -> maxes.Axes:
    """Plots a crystal with translucent ConvexHull coordination polyhedra."""
    atoms = _coerce_atoms(source).wrap_positions()
    if supercell != (1, 1, 1) and atoms.cell is not None:
        atoms = atoms.repeat(supercell)
    with theme.xkcd_style(preset=preset):
        if ax is None:
            _, ax = plt.subplots(figsize=(5.8, 5.0))
        structure3d.render_structure_3d(
            atoms,
            ax=ax,
            camera=camera,
            style="polyhedra",
            palette=palette,
            polyhedra=(center, ligand),
            show_cell=True,
        )
        _pad_structure_view(ax, bottom_frac=0.16, top_frac=0.16)
        callouts.draw_element_legend(
            ax, atoms.symbols, scheme=palette, loc="lower right"
        )
        col = palettes.get_element_color(center, scheme=palette)
        ax.text(
            0.03,
            0.96,
            f"Polyhedra: {center}" + (f"-{ligand}" if ligand else ""),
            transform=ax.transAxes,
            va="top",
            fontsize=10.0,
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": col,
                "alpha": 0.28,
            },
        )
        if title:
            ax.set_title(title)
    return ax
