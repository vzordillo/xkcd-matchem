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
"""Phonon band dispersion and LOBSTER pCOHP/ICOHP bonding plots."""

from __future__ import annotations

from collections.abc import Sequence
import pathlib

from matplotlib import axes as maxes
from matplotlib import pyplot as plt
import numpy as np

from xkcd_matchem.core import electronic
from xkcd_matchem.core import exceptions
from xkcd_matchem.io import loader
from xkcd_matchem.plots import _electronic_bands_dos as elec_bd
from xkcd_matchem.plots import _materials_ternary as mat_tern
from xkcd_matchem.render import primitives
from xkcd_matchem.style import palettes
from xkcd_matchem.style import patterns
from xkcd_matchem.style import theme


def _draw_phonon_branches(
    ax: maxes.Axes,
    q_distances: np.ndarray,
    freqs: np.ndarray,
    highlight_imaginary: bool,
    cols: tuple[str, str],
    sw: theme.StrokeHierarchy,
) -> bool:
    """Draws phonon dispersion branches and returns True if imaginary exist."""
    mode_col, imag_col = cols
    imag_labeled = False
    for mode_idx in range(freqs.shape[1]):
        qx, qy = elec_bd.split_kpath_breaks(q_distances, freqs[:, mode_idx])
        ax.plot(
            qx,
            qy,
            color=mode_col,
            linewidth=sw.primary,
            label="Acoustic / Optical" if mode_idx == 0 else None,
        )
        if highlight_imaginary and bool(np.any(qy < -1e-2)):
            imag_mask = qy < -1e-2
            dilated = imag_mask.copy()
            dilated[1:] |= imag_mask[:-1]
            dilated[:-1] |= imag_mask[1:]
            ax.plot(
                qx,
                np.where(dilated, qy, np.nan),
                color=imag_col,
                linewidth=sw.frame,
                zorder=4.0,
                label="Imaginary Modes" if not imag_labeled else None,
            )
            imag_labeled = True
    return imag_labeled


def plot_phonon_bands(
    source: electronic.PhononBandStructure | str | pathlib.Path,
    *,
    ax: maxes.Axes | None = None,
    unit: str = "THz",
    ylim: tuple[float, float] | None = None,
    highlight_imaginary: bool = True,
    palette: str | Sequence[str] = "okabe_ito",
    preset: str | theme.ThemePreset = "classic",
    title: str | None = None,
) -> maxes.Axes:
    """Plots phonon dispersion branches along a reciprocal-space q-path.

    Args:
      source: ``PhononBandStructure`` or Phonopy ``band.yaml`` path/sample.
      ax: Optional target Matplotlib ``Axes``.
      unit: Display unit (``"THz"``, ``"cm-1"``, or ``"meV"``).
      ylim: Optional vertical axis limits in ``unit``.
      highlight_imaginary: Highlight soft/imaginary modes (< 0) in vermilion.
      palette: Series palette name or hex color sequence.
      preset: Visual theme preset name or ``ThemePreset``.
      title: Optional plot heading.

    Returns:
      Matplotlib ``Axes`` containing the phonon dispersion diagram.
    """
    if isinstance(source, electronic.PhononBandStructure):
        pbs = source
    else:
        loaded = loader.load(source)
        if not isinstance(loaded, electronic.PhononBandStructure):
            raise exceptions.StructureError(
                f"Expected PhononBandStructure, got {type(loaded).__name__}."
            )
        pbs = loaded

    freqs = pbs.frequencies_in(unit)
    series = palettes.get_palette(palette)
    preset_obj = theme.get_preset(preset)
    ink_col = preset_obj.ink_color
    bg_col = preset_obj.axes_facecolor
    sw = theme.stroke_weights(preset)
    mode_col = mat_tern.theme_series_color(
        series[5 % len(series)], bg_col, ink_col
    )
    imag_col = mat_tern.theme_series_color("#D55E00", bg_col, ink_col)

    with theme.xkcd_style(preset=preset):
        if ax is None:
            _, ax = plt.subplots(figsize=(6.2, 4.4))
        else:
            theme.style_existing_axes(ax, preset=preset)
        imag_labeled = _draw_phonon_branches(
            ax,
            pbs.q_distances,
            freqs,
            highlight_imaginary,
            (mode_col, imag_col),
            sw,
        )
        z_line = ax.axhline(
            0.0,
            color=mat_tern.theme_muted_guide(bg_col),
            linestyle="--",
            lw=sw.guide,
        )
        z_line.set_sketch_params(0.0, 0.0, 0.0)
        if pbs.q_ticks and len(pbs.q_ticks) == len(pbs.q_labels):
            for tick_q in pbs.q_ticks:
                t_line = ax.axvline(
                    tick_q, color=ink_col, linewidth=sw.guide, alpha=0.40
                )
                t_line.set_sketch_params(0.0, 0.0, 0.0)
            ax.set_xticks(pbs.q_ticks)
            ax.set_xticklabels(elec_bd.format_k_labels(pbs.q_labels))
        ax.set_xlim(float(pbs.q_distances[0]), float(pbs.q_distances[-1]))
        if ylim is not None:
            ax.set_ylim(ylim)
        disp_unit = (
            "$\\mathrm{cm}^{-1}$" if unit.strip().lower() == "cm-1" else unit
        )
        ax.set_xlabel("Wavevector Path")
        ax.set_ylabel(f"Frequency ({disp_unit})")
        if imag_labeled:
            mat_tern.style_legend(ax, ink_col, bg_col, loc="best", frameon=True)
        theme.apply_xkcd_spines(ax, preset=preset)
        if title:
            ax.set_title(title, color=ink_col)
    return ax


def draw_cohp_series(
    ax: maxes.Axes,
    e_grid: np.ndarray,
    vals: np.ndarray,
    *,
    col: str,
    label: str,
    vertical: bool,
    shade_lobes: bool,
    spin_dn: np.ndarray | None,
    series_idx: int = 0,
    sw: theme.StrokeHierarchy | None = None,
    bg_color: str = palettes.PAPER,
) -> None:
    """Plots one COHP/ICOHP interaction curve and optional lobe shading."""
    weights = sw if sw is not None else theme.stroke_weights("classic")
    lw_curve = (
        weights.primary if series_idx == 0 else weights.overlay + 0.10
    )
    ls_curve = "-" if series_idx == 0 else "--"
    z_curve = 4.0 + 0.25 * min(series_idx, 3)
    bond_fill = "#56B4E9" if primitives.is_dark_color(bg_color) else "#0072B2"
    anti_fill = "#D55E00"
    if vertical:
        ax.plot(
            vals,
            e_grid,
            color=col,
            lw=lw_curve,
            linestyle=ls_curve,
            label=label,
            zorder=z_curve,
        )
        if shade_lobes and series_idx == 0:
            pos = np.where(vals >= 0.0, vals, 0.0)
            neg = np.where(vals < 0.0, vals, 0.0)
            patterns.shade_between(
                ax,
                e_grid,
                0.0,
                pos,
                color=bond_fill,
                shade="solid",
                alpha=0.20,
                vertical=True,
            )
            patterns.shade_between(
                ax,
                e_grid,
                0.0,
                neg,
                color=anti_fill,
                shade="solid",
                alpha=0.20,
                vertical=True,
            )
        if spin_dn is not None:
            ax.plot(
                -spin_dn, e_grid, color=col, lw=weights.overlay, linestyle=":"
            )
    else:
        ax.plot(
            e_grid,
            vals,
            color=col,
            lw=lw_curve,
            linestyle=ls_curve,
            label=label,
            zorder=z_curve,
        )
        if shade_lobes and series_idx == 0:
            pos = np.where(vals >= 0.0, vals, 0.0)
            neg = np.where(vals < 0.0, vals, 0.0)
            patterns.shade_between(
                ax,
                e_grid,
                0.0,
                pos,
                color=bond_fill,
                shade="solid",
                alpha=0.20,
                vertical=False,
            )
            patterns.shade_between(
                ax,
                e_grid,
                0.0,
                neg,
                color=anti_fill,
                shade="solid",
                alpha=0.20,
                vertical=False,
            )
        if spin_dn is not None:
            ax.plot(
                e_grid, -spin_dn, color=col, lw=weights.overlay, linestyle=":"
            )


def configure_cohp_axes(
    ax: maxes.Axes,
    *,
    vertical: bool,
    pop_label: str,
    ink_col: str,
    xlim: tuple[float, float] | None,
    ylim: tuple[float, float] | None,
    sw: theme.StrokeHierarchy | None = None,
    bg_color: str = palettes.PAPER,
) -> None:
    """Applies Fermi reference lines, limits, and labels to a COHP panel."""
    weights = sw if sw is not None else theme.stroke_weights("classic")
    guide_col = mat_tern.theme_muted_guide(bg_color)
    if vertical:
        ef_ln = ax.axhline(
            0.0, color=guide_col, linestyle="--", lw=weights.guide
        )
        z_ln = ax.axvline(0.0, color=ink_col, lw=weights.guide, alpha=0.50)
        ef_ln.set_sketch_params(0.0, 0.0, 0.0)
        z_ln.set_sketch_params(0.0, 0.0, 0.0)
        ax.set_xlabel(pop_label)
        ax.set_ylabel("$E - E_F$ (eV)")
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
    else:
        ef_ln = ax.axvline(
            0.0, color=guide_col, linestyle="--", lw=weights.guide
        )
        z_ln = ax.axhline(0.0, color=ink_col, lw=weights.guide, alpha=0.50)
        ef_ln.set_sketch_params(0.0, 0.0, 0.0)
        z_ln.set_sketch_params(0.0, 0.0, 0.0)
        ax.set_xlabel("$E - E_F$ (eV)")
        ax.set_ylabel(pop_label)
        if ylim is not None:
            ax.set_xlim(ylim)
        if xlim is not None:
            ax.set_ylim(xlim)


def resolve_cohp_inputs(
    source: electronic.CohpData | str | pathlib.Path,
    interactions: Sequence[str] | None,
    integrated: bool,
) -> tuple[
    electronic.CohpData,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    list[str],
]:
    """Loads COHP data and resolves spin tables and selected interactions."""
    if isinstance(source, electronic.CohpData):
        data = source
    else:
        loaded = loader.load(source)
        if not isinstance(loaded, electronic.CohpData):
            raise exceptions.StructureError(
                f"Expected CohpData, got {type(loaded).__name__}."
            )
        data = loaded
    table = data.icohp if integrated else data.cohp
    dn_table = data.spin_down_icohp if integrated else data.spin_down_cohp
    if not table:
        raise exceptions.StructureError("COHP dataset contains no data table.")
    selected = (
        list(table.keys()) if interactions is None else list(interactions)
    )
    missing = [lbl for lbl in selected if lbl not in table]
    if missing:
        raise exceptions.StructureError(
            f"Unknown COHP interaction labels: {missing!r}."
        )
    return data, table, dn_table, selected


def plot_cohp(
    source: electronic.CohpData | str | pathlib.Path,
    *,
    interactions: Sequence[str] | None = None,
    minus_cohp: bool = True,
    integrated: bool = False,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = (-8.0, 6.0),
    orientation: str = "vertical",
    shade_lobes: bool = True,
    palette: str | Sequence[str] = "okabe_ito",
    preset: str | theme.ThemePreset = "classic",
    ax: maxes.Axes | None = None,
    title: str | None = None,
) -> maxes.Axes:
    """Plots LOBSTER ``-pCOHP`` or ``-ICOHP`` bonding curves in xkcd style.

    By chemical convention (``minus_cohp=True``), positive ``-pCOHP`` values
    denote bonding states (shaded cerulean) while negative values denote
    antibonding states (shaded vermilion).

    Args:
      source: ``CohpData`` instance or path/sample name to ``COHPCAR.lobster``.
      interactions: Optional subset of interaction labels to plot.
      minus_cohp: Negate ``pCOHP``/``ICOHP`` so bonding appears positive.
      integrated: Plot cumulative ``ICOHP`` instead of differential ``pCOHP``.
      xlim: Optional population-axis bounds.
      ylim: Optional energy window around ``E_F`` in eV.
      orientation: ``"vertical"`` (E - E_F on y-axis) or ``"horizontal"``.
      shade_lobes: Shade bonding vs. antibonding lobes when ``not integrated``.
      palette: Series palette name or hex color list.
      preset: Visual theme preset name or ``ThemePreset``.
      ax: Optional target Matplotlib ``Axes``.
      title: Optional plot heading.

    Returns:
      Matplotlib ``Axes`` containing the COHP/ICOHP plot.

    Raises:
      StructureError: If input cannot be loaded or requested labels are absent.
    """
    data, table, dn_table, selected = resolve_cohp_inputs(
        source, interactions, integrated
    )
    sign = -1.0 if (minus_cohp and not data.is_coop) else 1.0
    prefix = "ICOHP" if integrated else "COOP" if data.is_coop else "pCOHP"
    vertical = orientation.strip().lower() != "horizontal"
    series = palettes.get_palette(palette)
    preset_obj = theme.get_preset(preset)
    sw = theme.stroke_weights(preset)

    with theme.xkcd_style(preset=preset):
        if ax is None:
            _, ax = plt.subplots(
                figsize=(5.2, 5.4) if vertical else (6.8, 4.2)
            )
        else:
            theme.style_existing_axes(ax, preset=preset)
        for idx, lbl in enumerate(selected):
            draw_cohp_series(
                ax,
                data.energies - data.fermi_energy,
                sign * table[lbl],
                col=mat_tern.theme_series_color(
                    series[(idx + 1) % len(series)],
                    preset_obj.axes_facecolor,
                    preset_obj.ink_color,
                ),
                label=lbl,
                vertical=vertical,
                shade_lobes=bool(shade_lobes and not integrated),
                spin_dn=(sign * dn_table[lbl] if lbl in dn_table else None),
                series_idx=idx,
                sw=sw,
                bg_color=preset_obj.axes_facecolor,
            )
        configure_cohp_axes(
            ax,
            vertical=vertical,
            pop_label=f"-{prefix} (eV)" if sign < 0.0 else f"{prefix} (eV)",
            ink_col=preset_obj.ink_color,
            xlim=xlim,
            ylim=ylim,
            sw=sw,
            bg_color=preset_obj.axes_facecolor,
        )
        mat_tern.style_legend(
            ax,
            preset_obj.ink_color,
            preset_obj.axes_facecolor,
            loc="best",
            frameon=True,
            fontsize=8.5,
        )
        theme.apply_xkcd_spines(ax, preset=preset)
        if title:
            ax.set_title(title, color=preset_obj.ink_color)
    return ax
