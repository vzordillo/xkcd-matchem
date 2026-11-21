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

import pathlib

from matplotlib import axes as maxes
from matplotlib import pyplot as plt

from xkcd_matchem.core import spectroscopy
from xkcd_matchem.core import structure
from xkcd_matchem.data import elements
from xkcd_matchem.io import loader
from xkcd_matchem.style import palettes
from xkcd_matchem.style import theme


def plot_spectrum(
    spectrum: spectroscopy.Spectrum,
    ax: maxes.Axes | None = None,
    invert_x: bool | None = None,
    show_sticks: bool = True,
    annotate_peaks: int = 5,
    color: str | None = None,
    fill_under: bool = True,
    preset: str = "classic",
    title: str | None = None,
) -> maxes.Axes:
    """Plots a broadened 1D spectrum (IR, Raman, UV-Vis, NMR, or XVG curve)."""
    series = palettes.get_palette("okabe_ito")
    line_col = series[5] if color is None else color
    should_invert = (
        spectrum.kind.lower() in ("ir", "nmr") if invert_x is None else invert_x
    )
    with theme.xkcd_style(preset=preset):
        if ax is None:
            _, ax = plt.subplots(figsize=(6.4, 4.0))
        ax.plot(spectrum.x, spectrum.y, color=line_col, lw=2.0)
        if fill_under:
            ax.fill_between(
                spectrum.x, 0.0, spectrum.y, color=line_col, alpha=0.18
            )
        if show_sticks and spectrum.peaks:
            y_max_curve = max(float(max(spectrum.y, default=1.0)), 1e-6)
            p_max = max((p.intensity for p in spectrum.peaks), default=1.0)
            scale = y_max_curve / max(p_max, 1e-6)
            for pk in spectrum.peaks:
                ax.vlines(
                    pk.position,
                    0.0,
                    pk.intensity * scale * 0.9,
                    color=series[6],
                    lw=1.3,
                    alpha=0.85,
                )
        if annotate_peaks > 0 and spectrum.peaks:
            top_peaks = sorted(
                spectrum.peaks, key=lambda p: p.intensity, reverse=True
            )[:annotate_peaks]
            y_max_curve = max(float(max(spectrum.y, default=1.0)), 1e-6)
            p_max = max((p.intensity for p in spectrum.peaks), default=1.0)
            for pk in top_peaks:
                y_loc = pk.intensity * (y_max_curve / max(p_max, 1e-6))
                lbl = pk.label if pk.label else f"{pk.position:.1f}"
                ax.annotate(
                    lbl,
                    xy=(pk.position, y_loc),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8.5,
                    color=palettes.INK,
                )
        ax.set_xlabel(spectrum.x_label)
        ax.set_ylabel(spectrum.y_label)
        if should_invert:
            ax.invert_xaxis()
        theme.apply_xkcd_spines(ax)
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
    preset: str = "classic",
    title: str | None = None,
) -> maxes.Axes:
    """Computes (if crystal given) and plots a hand-drawn powder XRD pattern."""
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
    with theme.xkcd_style(preset=preset):
        if ax is None:
            _, ax = plt.subplots(figsize=(6.8, 4.2))
        ax = plot_spectrum(
            spec,
            ax=ax,
            invert_x=False,
            show_sticks=True,
            annotate_peaks=label_peaks,
            color="#0072B2",
            preset=preset,
            title=title,
        )
        ax.set_xlim(two_theta_range)
        ax.set_ylim(0.0, 118.0)
    return ax
