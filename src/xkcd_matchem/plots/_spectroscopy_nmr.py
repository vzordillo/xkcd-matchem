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
"""1D NMR spectrum plotting with multiplet trees and step integrals."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from matplotlib import axes as maxes
from matplotlib import pyplot as plt
import numpy as np

from xkcd_matchem.core import spectroscopy
from xkcd_matchem.plots import _materials_ternary as mat_tern
from xkcd_matchem.plots import _spectroscopy_nmr_core as nmr_core
from xkcd_matchem.style import palettes
from xkcd_matchem.style import theme


def draw_straight_seg(
    ax: maxes.Axes,
    xs: list[float],
    ys: list[float],
    color: str,
    lw: float,
) -> None:
    """Draws a crisp un-sketched tree segment."""
    ln = ax.plot(xs, ys, color=color, lw=lw, zorder=4.8)[0]
    ln.set_sketch_params(0.0, 0.0, 0.0)


def draw_nmr_splitting_tree(
    ax: maxes.Axes,
    res: tuple[float, float, str, float | tuple[float, ...], str],
    scales: tuple[float, float, float, float],
    colors: tuple[str, str],
    sw: theme.StrokeHierarchy,
) -> float:
    """Draws crisp first-order multiplet splitting tree rungs above a peak."""
    shift, _, mult, j_val, lbl = res
    mhz, y_base, y_max, x_span = scales
    stages = nmr_core.parse_multiplicity_stages(mult, j_val)
    if not stages:
        if lbl:
            theme.place_halo_text(
                ax,
                shift,
                y_base + 0.05 * y_max,
                lbl,
                color=colors[0],
                bg_color=colors[1],
                fontsize=8.3,
                va="bottom",
                pad=0.06,
            )
            return y_base + 0.15 * y_max
        return y_base

    step_h, tick_h = 0.068 * y_max, 0.022 * y_max
    y_top = y_base + 0.05 * y_max + len(stages) * step_h + tick_h
    draw_straight_seg(
        ax, [shift, shift], [y_top, y_top - tick_h], colors[0], sw.leader
    )
    min_step_ppm = 0.012 * x_span / max(sum(n for n, _ in stages), 2)
    nodes = [shift]
    y_cur = y_top - tick_h
    for n_eq, jk in stages:
        d_ppm = max(jk / mhz, min_step_ppm)
        y_nxt = y_cur - step_h
        nxt_nodes: list[float] = []
        for xp in nodes:
            for m_idx in range(n_eq + 1):
                xc = xp + (float(m_idx) - 0.5 * float(n_eq)) * d_ppm
                nxt_nodes.append(xc)
                draw_straight_seg(
                    ax,
                    [xp, xc],
                    [y_cur, y_nxt + tick_h],
                    colors[0],
                    sw.leader * 0.90,
                )
                draw_straight_seg(
                    ax,
                    [xc, xc],
                    [y_nxt + tick_h, y_nxt],
                    colors[0],
                    sw.leader * 0.90,
                )
        nodes = sorted({round(v, 9) for v in nxt_nodes})
        y_cur = y_nxt

    theme.place_halo_text(
        ax,
        shift,
        y_top + 0.02 * y_max,
        f"{lbl} ({mult})" if lbl else mult,
        color=colors[0],
        bg_color=colors[1],
        fontsize=8.2,
        va="bottom",
        pad=0.06,
        zorder=6.5,
    )
    return y_top + 0.13 * y_max


def draw_nmr_integrals(
    ax: maxes.Axes,
    ppm_grid: np.ndarray,
    y_data: np.ndarray,
    resonances: Sequence[tuple[float, float, str, Any, str]],
    y_max: float,
    colors: tuple[str, str, str],
    sw: theme.StrokeHierarchy,
) -> np.ndarray:
    """Draws step-integral trace and clean sub-baseline ``nH`` badges."""
    int_col, ink_col, bg_col = colors
    x_rev = ppm_grid[::-1]
    cum: np.ndarray = nmr_core.cumulative_trapezoid(x_rev, y_data[::-1])
    tot_h = sum(float(r[1]) for r in resonances) if resonances else 1.0
    if float(cum[-1]) > 1e-12 and tot_h > 0.0:
        cum = cum * (tot_h / float(cum[-1]))

    y_plot = (0.06 * y_max) + (0.42 * y_max) * (cum / max(tot_h, 1e-12))
    ax.plot(
        x_rev,
        y_plot,
        color=int_col,
        lw=sw.overlay,
        alpha=0.82,
        zorder=2.8,
        gid="nmr_step_integral",
    )
    for shift, integ, _, _, _ in resonances:
        if integ <= 0.0:
            continue
        int_txt = (
            f"{int(round(integ))}H"
            if abs(integ - round(integ)) < 0.05
            else f"{integ:.1f}H"
        )
        theme.place_halo_text(
            ax,
            shift,
            -0.045 * y_max,
            f"$\\int$ {int_txt}",
            color=ink_col,
            bg_color=bg_col,
            fontsize=8.2,
            va="top",
            pad=0.05,
            alpha=0.90,
            zorder=6.2,
        )
    return np.asarray(y_plot[::-1], dtype=np.float64)


def _draw_nmr_trees_above_peaks(
    ax: maxes.Axes,
    ppm_grid: np.ndarray,
    y_curve: np.ndarray,
    y_int_asc: np.ndarray | None,
    resonances: Sequence[
        tuple[float, float, str, float | tuple[float, ...], str]
    ],
    scales: tuple[float, float, float],
    colors: tuple[str, str],
    sw: theme.StrokeHierarchy,
) -> float:
    """Draws splitting trees above all resonances and returns required top y."""
    mhz, y_max, x_span = scales
    top_needed = 1.18 * y_max
    for res in resonances:
        win = (ppm_grid >= res[0] - 0.25) & (ppm_grid <= res[0] + 0.25)
        y_pk = float(np.max(y_curve[win])) if np.any(win) else 0.8 * y_max
        if y_int_asc is not None and np.any(win):
            y_pk = max(y_pk, float(np.max(y_int_asc[win])))
        top_needed = max(
            top_needed,
            draw_nmr_splitting_tree(
                ax, res, (mhz, y_pk, y_max, x_span), colors, sw
            ),
        )
    return top_needed


def _draw_tms_marker(
    ax: maxes.Axes,
    y_max: float,
    guide_col: str,
    colors: tuple[str, str],
    sw: theme.StrokeHierarchy,
) -> None:
    """Draws internal 0.00 ppm TMS reference marker and halo label."""
    draw_straight_seg(
        ax, [0.0, 0.0], [0.0, 0.20 * y_max], guide_col, sw.guide + 0.15
    )
    theme.place_halo_text(
        ax,
        0.0,
        0.23 * y_max,
        "TMS",
        color=colors[0],
        bg_color=colors[1],
        fontsize=8.0,
        va="bottom",
        pad=0.05,
    )


def render_nmr_axes(
    ax: maxes.Axes,
    bundle: tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list[tuple[float, float, str, float | tuple[float, ...], str]],
        tuple[float, float],
    ],
    opts: tuple[float, bool, bool, bool],
    style_cfg: tuple[
        str, Sequence[str], tuple[str, str], theme.StrokeHierarchy
    ],
) -> None:
    """Draws NMR sticks, trace, TMS marker, step integrals, and trees."""
    ppm_grid, y_curve, stick_pos, stick_wt, resonances, (lo_p, hi_p) = bundle
    mhz, show_integrals, show_trees, show_tms = opts
    line_col, series, colors, sw = style_cfg
    guide_col = mat_tern.theme_muted_guide(colors[1])
    y_max = max(float(np.max(y_curve)) if y_curve.size else 1.0, 1e-6)

    base_ln = ax.axhline(
        0.0, color=guide_col, lw=sw.guide, alpha=0.55, zorder=1.5
    )
    base_ln.set_sketch_params(0.0, 0.0, 0.0)
    if stick_pos.size > 0:
        w_max = max(float(np.max(stick_wt)), 1e-9)
        ax.vlines(
            stick_pos,
            0.0,
            stick_wt * (0.85 * y_max / w_max),
            color=mat_tern.theme_series_color(
                series[6 % len(series)], colors[1], colors[0]
            ),
            lw=sw.guide,
            alpha=0.62,
            zorder=2.2,
        )
    ax.plot(ppm_grid, y_curve, color=line_col, lw=sw.primary, zorder=3.6)

    if show_tms and lo_p <= 0.0 <= hi_p:
        _draw_tms_marker(ax, y_max, guide_col, colors, sw)

    y_int_asc: np.ndarray | None = None
    if show_integrals and ppm_grid.size >= 2:
        int_col = mat_tern.theme_series_color(
            series[3 % len(series)], colors[1], colors[0]
        )
        y_int_asc = draw_nmr_integrals(
            ax,
            ppm_grid,
            y_curve,
            resonances,
            y_max,
            (int_col, colors[0], colors[1]),
            sw,
        )

    top_needed = (
        _draw_nmr_trees_above_peaks(
            ax,
            ppm_grid,
            y_curve,
            y_int_asc,
            resonances,
            (mhz, y_max, hi_p - lo_p),
            colors,
            sw,
        )
        if (show_trees and resonances)
        else 1.18 * y_max
    )

    ax.set_xlim(lo_p, hi_p)
    ax.invert_xaxis()
    y_bot = (
        -0.16 * y_max if (show_integrals and resonances) else -0.05 * y_max
    )
    ax.set_ylim(y_bot, top_needed)


def plot_nmr(
    shifts_ppm: spectroscopy.Spectrum | Sequence[Any] | np.ndarray,
    couplings_hz: Sequence[Any] | float | None = None,
    multiplicities: Sequence[str] | None = None,
    integrals: Sequence[float] | np.ndarray | None = None,
    spectrometer_mhz: float = 400.0,
    ppm_range: tuple[float, float] | None = None,
    fwhm_ppm: float = 0.018,
    *,
    labels: Sequence[str | None] | None = None,
    show_integrals: bool = True,
    show_trees: bool = True,
    show_tms: bool = True,
    nucleus: str = "1H",
    ax: maxes.Axes | None = None,
    color: str | None = None,
    palette: str | Sequence[str] = "okabe_ito",
    preset: str | theme.ThemePreset = "classic",
    title: str | None = None,
) -> maxes.Axes:
    """Synthesizes and plots a 1D NMR spectrum with trees and integrals.

    Args:
      shifts_ppm: Pre-built ``Spectrum``, sequence of resonance dicts/tuples
        ``(shift, integral, multiplicity, j_hz[, label])``, or 1D chemical
        shifts in ppm.
      couplings_hz: Optional scalar coupling constants ``J`` in Hz matching
        ``shifts_ppm``.
      multiplicities: Optional splitting codes (``"s"``, ``"d"``, ``"t"``,
        ``"q"``, ``"dd"``, ``"m"``) matching ``shifts_ppm``.
      integrals: Optional relative proton/spin counts matching ``shifts_ppm``.
      spectrometer_mhz: Spectrometer Larmor frequency in MHz.
      ppm_range: Optional ``(min_ppm, max_ppm)`` spectral window.
      fwhm_ppm: Lorentzian line FWHM in ppm.
      labels: Optional assignment labels when ``shifts_ppm`` is a 1D float list.
      show_integrals: Overlay cumulative step-integral curve and ``nH`` tags.
      show_trees: Draw first-order multiplet splitting trees above peaks.
      show_tms: Add internal ``0.00 ppm`` TMS reference line and tag.
      nucleus: Observed nucleus label (e.g. ``"1H"``, ``"13C"``).
      ax: Optional target Matplotlib ``Axes``.
      color: Optional primary trace color override.
      palette: Series palette name or hex sequence.
      preset: Visual theme preset name or ``ThemePreset``.
      title: Optional plot heading.

    Returns:
      Styled Matplotlib ``Axes`` with inverted chemical-shift ``delta (ppm)``.
    """
    bundle = nmr_core.prepare_nmr_data(
        shifts_ppm,
        couplings_hz,
        multiplicities,
        integrals,
        labels,
        (spectrometer_mhz, ppm_range, fwhm_ppm, show_tms),
    )
    series = palettes.get_palette(palette)
    preset_obj = theme.get_preset(preset)
    ink_col = preset_obj.ink_color
    bg_col = preset_obj.axes_facecolor
    line_col = (
        mat_tern.theme_series_color(series[5 % len(series)], bg_col, ink_col)
        if color is None
        else color
    )

    with theme.xkcd_style(preset=preset):
        if ax is None:
            _, ax = plt.subplots(figsize=(6.8, 4.3))
        else:
            theme.style_existing_axes(ax, preset=preset)

        render_nmr_axes(
            ax,
            bundle,
            (spectrometer_mhz, show_integrals, show_trees, show_tms),
            (
                line_col,
                series,
                (ink_col, bg_col),
                theme.stroke_weights(preset),
            ),
        )
        nuc_lbl = nmr_core.format_nucleus_symbol(nucleus)
        ax.set_xlabel(f"{nuc_lbl} Chemical Shift $\\delta$ (ppm)")
        ax.set_ylabel("Intensity (arb. units)")
        theme.apply_xkcd_spines(ax, preset=preset)
        if title:
            ax.set_title(title, color=ink_col)
    return ax
