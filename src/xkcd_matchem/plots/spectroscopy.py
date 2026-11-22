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
"""Powder X-ray diffraction and molecular spectroscopy plotting functions."""

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
from xkcd_matchem.style import palettes
from xkcd_matchem.style import patterns
from xkcd_matchem.style import theme


def _resolve_stick_scale(spectrum: spectroscopy.Spectrum) -> float:
    """Computes stick scaling without distorting already-matched intensities."""
    if not spectrum.peaks or spectrum.y.size == 0:
        return 1.0
    y_max_curve = max(float(np.max(spectrum.y)), 1e-6)
    p_max = max((p.intensity for p in spectrum.peaks), default=1.0)
    if 0.25 * y_max_curve <= p_max <= 4.0 * y_max_curve:
        return 1.0
    return y_max_curve / max(p_max, 1e-6)


def _annotate_spectrum_peaks(
    ax: maxes.Axes,
    peaks: list[spectroscopy.Peak],
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    scale: float,
    annotate_peaks: int,
    ink_col: str,
) -> float:
    """Annotates top spectral peaks above sticks/curve and returns max y."""
    if annotate_peaks <= 0 or not peaks:
        return 0.0
    top_peaks = sorted(peaks, key=lambda p: p.intensity, reverse=True)[
        :annotate_peaks
    ]
    if x_arr.size >= 2:
        order = np.argsort(x_arr)
        xs_s, ys_s = x_arr[order], y_arr[order]
    else:
        xs_s, ys_s = x_arr, y_arr
    top_y = 0.0
    for pk in top_peaks:
        y_curve = (
            float(np.interp(pk.position, xs_s, ys_s))
            if xs_s.size >= 2 and xs_s[0] <= pk.position <= xs_s[-1]
            else 0.0
        )
        y_loc = max(pk.intensity * scale, y_curve)
        top_y = max(top_y, y_loc)
        lbl = pk.label if pk.label else f"{pk.position:.1f}"
        ax.annotate(
            lbl,
            xy=(pk.position, y_loc),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            color=ink_col,
        )
    return top_y


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
    line_col = series[5 % len(series)] if color is None else color
    stick_col = series[6 % len(series)]
    ink_col = theme.get_preset(preset).ink_color
    sw = theme.stroke_weights(preset)
    should_invert = (
        spectrum.kind.lower() in ("ir", "nmr") if invert_x is None else invert_x
    )
    scale = _resolve_stick_scale(spectrum)
    x_arr = np.asarray(spectrum.x, dtype=np.float64)
    y_arr = np.asarray(spectrum.y, dtype=np.float64)
    with theme.xkcd_style(preset=preset):
        if ax is None:
            _, ax = plt.subplots(figsize=(6.4, 4.0))
        else:
            theme.style_existing_axes(ax, preset=preset)
        if background_grid:
            patterns.apply_background_grid(ax, style=background_grid)
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
        ann_top = _annotate_spectrum_peaks(
            ax,
            spectrum.peaks,
            x_arr,
            y_arr,
            scale,
            annotate_peaks,
            ink_col,
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
            ax.set_title(title)
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
    line_col = series[5 % len(series)] if color is None else color
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
        ax.set_ylim(0.0, 118.0)
    return ax
