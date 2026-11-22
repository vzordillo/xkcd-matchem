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
"""Powder XRD, molecular IR/UV-Vis, 1D NMR multiplet, and XPS plots."""

from __future__ import annotations

from collections.abc import Sequence
import pathlib

from matplotlib import axes as maxes
from matplotlib import pyplot as plt
import numpy as np

from xkcd_matchem.core import spectroscopy
from xkcd_matchem.core import structure
from xkcd_matchem.data import elements
from xkcd_matchem.io import loader
from xkcd_matchem.plots import _materials_ternary as mat_tern
from xkcd_matchem.plots import _spectroscopy_nmr as spec_nmr
from xkcd_matchem.plots import _spectroscopy_nmr_core as nmr_core
from xkcd_matchem.plots import _spectroscopy_peaks as spec_peaks
from xkcd_matchem.plots import _spectroscopy_xps as spec_xps
from xkcd_matchem.style import palettes
from xkcd_matchem.style import patterns
from xkcd_matchem.style import theme


__all__ = [
    "compute_shirley_background",
    "expand_nmr_multiplet",
    "plot_nmr",
    "plot_spectrum",
    "plot_xps",
    "plot_xrd",
]

# Re-exported public NMR & XPS plotting/calculation API.
compute_shirley_background = spec_xps.compute_shirley_background
expand_nmr_multiplet = nmr_core.expand_nmr_multiplet
plot_nmr = spec_nmr.plot_nmr
plot_xps = spec_xps.plot_xps

# Backwards-compatible internal symbol re-exports.
_NMR_WORD_MAP = nmr_core.NMR_WORD_MAP
_NMR_CHAR_NEIGHBORS = nmr_core.NMR_CHAR_NEIGHBORS

_resolve_stick_scale = spec_peaks.resolve_stick_scale
_peak_label_offset = spec_peaks.peak_label_offset
_annotate_spectrum_peaks = spec_peaks.annotate_spectrum_peaks

_format_nucleus_symbol = nmr_core.format_nucleus_symbol
_parse_multiplicity_stages = nmr_core.parse_multiplicity_stages
_parse_resonance_entry = nmr_core.parse_resonance_entry
_coerce_nmr_resonances = nmr_core.coerce_nmr_resonances
_cumulative_trapezoid = nmr_core.cumulative_trapezoid
_spectrum_to_nmr_bundle = nmr_core.spectrum_to_nmr_bundle
_prepare_nmr_data = nmr_core.prepare_nmr_data

_draw_straight_seg = spec_nmr.draw_straight_seg
_draw_nmr_splitting_tree = spec_nmr.draw_nmr_splitting_tree
_draw_nmr_integrals = spec_nmr.draw_nmr_integrals
_render_nmr_axes = spec_nmr.render_nmr_axes

_compute_xps_background = spec_xps.compute_xps_background
_resolve_comp_dict_center_amp = spec_xps.resolve_comp_dict_center_amp
_evaluate_one_xps_component = spec_xps.evaluate_one_xps_component
_annotate_xps_components = spec_xps.annotate_xps_components
_prepare_xps_arrays = spec_xps.prepare_xps_arrays
_draw_xps_traces = spec_xps.draw_xps_traces


def plot_spectrum(
    spectrum: spectroscopy.Spectrum,
    ax: maxes.Axes | None = None,
    invert_x: bool | None = None,
    show_sticks: bool = True,
    annotate_peaks: int = 5,
    color: str | None = None,
    fill_under: bool = True,
    shade: str | None = None,
    background_grid: str | None = None,
    palette: str | Sequence[str] = "okabe_ito",
    preset: str | theme.ThemePreset = "classic",
    title: str | None = None,
) -> maxes.Axes:
    """Plots a broadened 1D spectrum (IR, Raman, UV-Vis, NMR, or XVG curve)."""
    series = palettes.get_palette(palette)
    preset_obj = theme.get_preset(preset)
    ink_col = preset_obj.ink_color
    bg_col = preset_obj.axes_facecolor
    line_col = (
        mat_tern.theme_series_color(series[5 % len(series)], bg_col, ink_col)
        if color is None
        else color
    )
    stick_col = mat_tern.theme_series_color(
        series[6 % len(series)], bg_col, ink_col
    )
    sw = theme.stroke_weights(preset)
    should_invert = (
        spectrum.kind.lower() in ("ir", "nmr") if invert_x is None else invert_x
    )
    scale = spec_peaks.resolve_stick_scale(spectrum)
    x_arr = np.asarray(spectrum.x, dtype=np.float64)
    y_arr = np.asarray(spectrum.y, dtype=np.float64)
    with theme.xkcd_style(preset=preset):
        if ax is None:
            _, ax = plt.subplots(figsize=(6.4, 4.0))
        else:
            theme.style_existing_axes(ax, preset=preset)
        if background_grid:
            patterns.apply_background_grid(ax, style=background_grid)
        if x_arr.size and y_arr.size:
            ax.plot(x_arr, y_arr, color=line_col, lw=sw.primary, zorder=3.5)
            if fill_under:
                patterns.shade_between(
                    ax,
                    x_arr,
                    0.0,
                    y_arr,
                    color=line_col,
                    shade=shade,
                    alpha=0.22 if shade else 0.18,
                )
        top_y = float(np.max(y_arr)) if y_arr.size else 1.0
        if show_sticks and spectrum.peaks:
            for pk in spectrum.peaks:
                pk_y = pk.intensity * scale
                top_y = max(top_y, pk_y)
                ax.vlines(
                    pk.position,
                    0.0,
                    pk_y,
                    color=stick_col,
                    lw=sw.guide + 0.10,
                    alpha=0.72,
                    zorder=2.2,
                )
        ann_top = spec_peaks.annotate_spectrum_peaks(
            ax,
            spectrum.peaks,
            x_arr,
            y_arr,
            scale,
            annotate_peaks,
            (ink_col, bg_col, should_invert),
            leader_lw=sw.leader,
        )
        top_y = max(top_y, ann_top)
        y_low = min(0.0, float(np.min(y_arr)) if y_arr.size else 0.0)
        ax.set_ylim(y_low, max(top_y * 1.15, y_low + 1.0))
        ax.set_xlabel(spectrum.x_label)
        ax.set_ylabel(spectrum.y_label)
        if should_invert:
            ax.invert_xaxis()
        theme.apply_xkcd_spines(ax, preset=preset)
        if title:
            ax.set_title(title, color=ink_col)
    return ax


def plot_xrd(
    source: spectroscopy.Spectrum | structure.AtomsData | str | pathlib.Path,
    ax: maxes.Axes | None = None,
    wavelength: float = elements.CODATA2022.cu_kalpha1_angstrom,
    two_theta_range: tuple[float, float] = (10.0, 90.0),
    fwhm: float = 0.4,
    label_peaks: int = 5,
    color: str | None = None,
    shade: str | None = None,
    background_grid: str | None = None,
    palette: str | Sequence[str] = "okabe_ito",
    preset: str | theme.ThemePreset = "classic",
    title: str | None = None,
) -> maxes.Axes:
    """Computes (when given a crystal) and plots an xkcd-style powder XRD."""
    if isinstance(source, spectroscopy.Spectrum):
        spec = source
    else:
        atoms = (
            source
            if isinstance(source, structure.AtomsData)
            else loader.load(source)
        )
        if not isinstance(atoms, structure.AtomsData):
            raise TypeError("Expected AtomsData or Spectrum for plot_xrd.")
        spec = spectroscopy.compute_xrd(
            atoms,
            wavelength=wavelength,
            two_theta_range=two_theta_range,
            fwhm=fwhm,
        )
    series = palettes.get_palette(palette)
    preset_obj = theme.get_preset(preset)
    line_col = (
        mat_tern.theme_series_color(
            series[5 % len(series)],
            preset_obj.axes_facecolor,
            preset_obj.ink_color,
        )
        if color is None
        else color
    )
    with theme.xkcd_style(preset=preset):
        if ax is None:
            _, ax = plt.subplots(figsize=(6.8, 4.2))
        else:
            theme.style_existing_axes(ax, preset=preset)
        ax = plot_spectrum(
            spec,
            ax=ax,
            invert_x=False,
            show_sticks=True,
            annotate_peaks=label_peaks,
            color=line_col,
            shade=shade,
            background_grid=background_grid,
            palette=palette,
            preset=preset,
            title=title,
        )
        ax.set_xlim(two_theta_range)
        _, cur_hi = ax.get_ylim()
        ax.set_ylim(0.0, max(118.0, float(cur_hi)))
    return ax
