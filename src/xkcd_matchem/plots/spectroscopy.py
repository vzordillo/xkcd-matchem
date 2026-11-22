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

from collections.abc import Mapping
from collections.abc import Sequence
import math
import pathlib
from typing import Any

from matplotlib import axes as maxes
from matplotlib import pyplot as plt
import numpy as np

from xkcd_matchem.core import exceptions
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


def _peak_label_offset(
    pos: float,
    y_loc: float,
    xs_s: np.ndarray,
    ys_s: np.ndarray,
    y_max: float,
    should_invert: bool,
) -> tuple[int, int, str]:
    """Computes peak tag offset (pt) nudged clear of taller nearby slopes."""
    if xs_s.size < 2:
        return 0, 8, "center"
    x_span = max(float(xs_s[-1] - xs_s[0]), 1e-6)
    win = 0.045 * x_span
    lo_mask = (xs_s >= pos - win) & (xs_s < pos)
    hi_mask = (xs_s > pos) & (xs_s <= pos + win)
    tall_lo = float(np.max(ys_s[lo_mask])) if np.any(lo_mask) else y_loc
    tall_hi = float(np.max(ys_s[hi_mask])) if np.any(hi_mask) else y_loc
    thresh = y_loc + 0.10 * max(y_max, 1e-6)
    if tall_lo > thresh and tall_lo >= tall_hi:
        dx = -6 if should_invert else 6
        return dx, 9, ("right" if dx < 0 else "left")
    if tall_hi > thresh and tall_hi > tall_lo:
        dx = 6 if should_invert else -6
        return dx, 9, ("right" if dx < 0 else "left")
    return 0, 8, "center"


def _annotate_spectrum_peaks(
    ax: maxes.Axes,
    peaks: list[spectroscopy.Peak],
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    scale: float,
    annotate_peaks: int,
    style_cfg: tuple[str, str, bool],
) -> float:
    """Annotates top spectral peaks above sticks/curve and returns max y."""
    if annotate_peaks <= 0 or not peaks:
        return 0.0
    ink_col, bg_color, should_invert = style_cfg
    top_peaks = sorted(peaks, key=lambda p: p.intensity, reverse=True)[
        :annotate_peaks
    ]
    if x_arr.size >= 2:
        order = np.argsort(x_arr)
        xs_s, ys_s = x_arr[order], y_arr[order]
    else:
        xs_s, ys_s = x_arr, y_arr
    y_max = float(np.max(ys_s)) if ys_s.size else 1.0
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
        dx_pt, dy_pt, ha = _peak_label_offset(
            pk.position, y_loc, xs_s, ys_s, y_max, should_invert
        )
        ann = ax.annotate(
            lbl,
            xy=(pk.position, y_loc),
            xytext=(dx_pt, dy_pt),
            textcoords="offset points",
            ha=ha,
            fontsize=8.5,
            color=ink_col,
            bbox=theme.label_halo_bbox(bg_color, pad=0.05, alpha=0.84),
            zorder=6.0,
        )
        patch = ann.get_bbox_patch()
        if patch is not None:
            patch.set_sketch_params(0.0, 0.0, 0.0)
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
    preset_obj = theme.get_preset(preset)
    ink_col = preset_obj.ink_color
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
        ann_top = _annotate_spectrum_peaks(
            ax,
            spectrum.peaks,
            x_arr,
            y_arr,
            scale,
            annotate_peaks,
            (ink_col, preset_obj.axes_facecolor, should_invert),
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


_NMR_WORD_MAP: dict[str, str] = {
    "singlet": "s",
    "doublet": "d",
    "triplet": "t",
    "quartet": "q",
    "quintet": "quint",
    "pentet": "quint",
    "sextet": "sext",
    "septet": "sept",
    "heptet": "sept",
    "multiplet": "m",
}

_NMR_CHAR_NEIGHBORS: dict[str, int] = {
    "d": 1,
    "t": 2,
    "q": 3,
    "p": 4,
    "h": 5,
}


def _format_nucleus_symbol(nucleus: str) -> str:
    """Formats leading mass numbers as mathtext superscripts (e.g. ``1H``)."""
    tok = nucleus.strip()
    if "$" in tok or not tok:
        return tok
    idx = 0
    while idx < len(tok) and tok[idx].isdigit():
        idx += 1
    if 0 < idx < len(tok) and tok[idx:].isalpha():
        return f"$^{{{tok[:idx]}}}{tok[idx:]}$"
    return tok


def _parse_multiplicity_stages(
    multiplicity: str,
    j_hz: float | Sequence[float] | None,
) -> list[tuple[int, float]]:
    """Converts an NMR multiplicity token and J values into splitting stages."""
    raw_code = multiplicity.strip().lower()
    code = _NMR_WORD_MAP.get(raw_code, raw_code)
    if j_hz is None:
        j_list: list[float] = []
    elif isinstance(j_hz, (int, float, np.integer, np.floating)):
        j_list = [float(j_hz)]
    else:
        j_list = [float(val) for val in j_hz]

    if code in ("", "s", "br", "brs"):
        return []
    named_n = {"quint": 4, "pent": 4, "sext": 5, "sept": 6, "hept": 6}
    if code in named_n:
        j0 = j_list[0] if j_list else 7.0
        return [(named_n[code], j0)] if j0 > 0.0 else []
    if code == "m":
        if len(j_list) >= 2:
            return [(1, jk) for jk in j_list if jk > 0.0]
        j0 = j_list[0] if (j_list and j_list[0] > 0.0) else 6.5
        return [(4, j0)]
    if all(ch in _NMR_CHAR_NEIGHBORS for ch in code):
        stages: list[tuple[int, float]] = []
        for idx, ch in enumerate(code):
            if idx < len(j_list):
                j_val = j_list[idx]
            elif j_list:
                j_val = j_list[-1] * (0.5 ** (idx - len(j_list) + 1))
            else:
                j_val = 7.0
            if j_val > 0.0:
                stages.append((_NMR_CHAR_NEIGHBORS[ch], j_val))
        return stages
    raise exceptions.StructureError(
        f"Unsupported NMR multiplicity {multiplicity!r}."
    )


def expand_nmr_multiplet(
    shift_ppm: float,
    integral: float = 1.0,
    multiplicity: str = "s",
    j_hz: float | Sequence[float] | None = None,
    spectrometer_mhz: float = 400.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthesizes exact first-order Pascal/doublet NMR stick lines.

    Args:
      shift_ppm: Resonance chemical shift center in ppm.
      integral: Total integrated proton/spin count for the multiplet.
      multiplicity: Splitting code (``"s"``, ``"d"``, ``"t"``, ``"q"``,
        ``"quint"``, ``"sext"``, ``"sept"``, ``"dd"``, ``"dt"``, ``"td"``,
        ``"ddd"``, ``"m"``).
      j_hz: Scalar coupling constant(s) ``J`` in Hz.
      spectrometer_mhz: Larmor frequency ``nu_0`` in MHz (``> 0``).

    Returns:
      Tuple ``(positions_ppm, stick_areas)`` of sorted 1D float64 arrays whose
      weights sum to ``integral``.
    """
    mhz = float(spectrometer_mhz)
    if mhz <= 0.0:
        raise exceptions.StructureError("spectrometer_mhz must be > 0.")
    area = float(integral)
    if area < 0.0:
        raise exceptions.StructureError("NMR integral must be non-negative.")
    stages = _parse_multiplicity_stages(multiplicity, j_hz)
    sticks: list[tuple[float, float]] = [(float(shift_ppm), area)]
    for n_eq, j_val in stages:
        delta_ppm = float(j_val) / mhz
        denom = float(2**n_eq)
        nxt: list[tuple[float, float]] = []
        for pos0, w0 in sticks:
            for m_idx in range(n_eq + 1):
                offset = (float(m_idx) - 0.5 * float(n_eq)) * delta_ppm
                wt = w0 * float(math.comb(n_eq, m_idx)) / denom
                nxt.append((pos0 + offset, wt))
        sticks = nxt

    sticks.sort(key=lambda item: item[0])
    merged_pos: list[float] = []
    merged_wt: list[float] = []
    for pos, wt in sticks:
        if merged_pos and abs(pos - merged_pos[-1]) <= 1e-10:
            merged_wt[-1] += wt
        else:
            merged_pos.append(pos)
            merged_wt.append(wt)
    return (
        np.asarray(merged_pos, dtype=np.float64),
        np.asarray(merged_wt, dtype=np.float64),
    )


def _parse_resonance_entry(
    entry: Any,
) -> tuple[float, float, str, float | tuple[float, ...], str]:
    """Parses one dict or tuple NMR resonance specification."""
    if isinstance(entry, Mapping):
        shift = float(
            entry.get("shift", entry.get("ppm", entry.get("delta", 0.0)))
        )
        integ = float(
            entry.get("integral", entry.get("n_h", entry.get("protons", 1.0)))
        )
        mult = str(entry.get("multiplicity", entry.get("mult", "s")))
        raw_j = entry.get("j_hz", entry.get("couplings_hz", entry.get("j")))
        lbl = str(entry.get("label", ""))
    elif isinstance(entry, Sequence) and not isinstance(entry, str):
        items = list(entry)
        if len(items) < 2:
            raise exceptions.StructureError(
                "NMR resonance tuple requires at least (shift, integral)."
            )
        shift = float(items[0])
        if isinstance(items[1], str):
            integ, mult = 1.0, items[1]
            raw_j = items[2] if len(items) >= 3 else 0.0
            lbl = str(items[3]) if len(items) >= 4 else ""
        else:
            integ = float(items[1])
            mult = str(items[2]) if len(items) >= 3 else "s"
            raw_j = items[3] if len(items) >= 4 else 0.0
            lbl = str(items[4]) if len(items) >= 5 else ""
    else:
        raise exceptions.StructureError(
            f"Invalid NMR resonance item {entry!r}."
        )

    if raw_j is None:
        j_norm: float | tuple[float, ...] = 0.0
    elif isinstance(raw_j, (int, float, np.integer, np.floating)):
        j_norm = float(raw_j)
    else:
        j_norm = tuple(float(v) for v in raw_j)
    return shift, integ, mult, j_norm, lbl


def _coerce_nmr_resonances(
    shifts_ppm: Sequence[Any] | np.ndarray,
    couplings_hz: Sequence[Any] | float | None,
    multiplicities: Sequence[str] | None,
    integrals: Sequence[float] | np.ndarray | None,
    labels: Sequence[str | None] | None,
) -> list[tuple[float, float, str, float | tuple[float, ...], str]]:
    """Normalizes structured or parallel-array NMR resonance inputs."""
    seq = list(shifts_ppm)
    if not seq:
        raise exceptions.StructureError("shifts_ppm cannot be empty.")
    first = seq[0]
    if isinstance(first, Mapping) or (
        isinstance(first, Sequence) and not isinstance(first, str)
    ):
        return [_parse_resonance_entry(item) for item in seq]

    n_res = len(seq)
    mult_list = (
        [str(m) for m in multiplicities]
        if multiplicities is not None
        else ["s"] * n_res
    )
    int_list = (
        [float(v) for v in integrals]
        if integrals is not None
        else [1.0] * n_res
    )
    lbl_list = (
        ["" if v is None else str(v) for v in labels]
        if labels is not None
        else [""] * n_res
    )
    if couplings_hz is None:
        j_seq: list[Any] = [0.0] * n_res
    elif isinstance(couplings_hz, (int, float, np.integer, np.floating)):
        j_seq = [float(couplings_hz)] * n_res
    else:
        j_seq = list(couplings_hz)
    same_len = (
        len(mult_list) == n_res
        and len(int_list) == n_res
        and len(lbl_list) == n_res
        and len(j_seq) == n_res
    )
    if not same_len:
        raise exceptions.StructureError(
            "NMR shifts, couplings, multiplicities, and integrals must match."
        )
    out: list[tuple[float, float, str, float | tuple[float, ...], str]] = []
    for k in range(n_res):
        rj = j_seq[k]
        j_val: float | tuple[float, ...] = (
            0.0
            if rj is None
            else (
                float(rj)
                if isinstance(rj, (int, float, np.integer, np.floating))
                else tuple(float(v) for v in rj)
            )
        )
        out.append(
            (float(seq[k]), int_list[k], mult_list[k], j_val, lbl_list[k])
        )
    return out


def _cumulative_trapezoid(x_arr: np.ndarray, y_arr: np.ndarray) -> np.ndarray:
    """Computes exact 1D cumulative trapezoidal integral starting at 0."""
    if x_arr.size < 2:
        return np.zeros_like(y_arr, dtype=np.float64)
    dx = np.abs(np.diff(x_arr))
    seg = 0.5 * (y_arr[:-1] + y_arr[1:]) * dx
    return np.concatenate([np.zeros(1, dtype=np.float64), np.cumsum(seg)])


def _draw_straight_seg(
    ax: maxes.Axes,
    xs: list[float],
    ys: list[float],
    color: str,
    lw: float,
) -> None:
    """Draws a crisp un-sketched tree segment."""
    ln = ax.plot(xs, ys, color=color, lw=lw, zorder=4.8)[0]
    ln.set_sketch_params(0.0, 0.0, 0.0)


def _draw_nmr_splitting_tree(
    ax: maxes.Axes,
    res: tuple[float, float, str, float | tuple[float, ...], str],
    scales: tuple[float, float, float, float],
    colors: tuple[str, str],
    sw: theme.StrokeHierarchy,
) -> float:
    """Draws crisp first-order multiplet splitting tree rungs above a peak."""
    shift, _, mult, j_val, lbl = res
    mhz, y_base, y_max, x_span = scales
    stages = _parse_multiplicity_stages(mult, j_val)
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
    _draw_straight_seg(
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
                _draw_straight_seg(
                    ax,
                    [xp, xc],
                    [y_cur, y_nxt + tick_h],
                    colors[0],
                    sw.leader * 0.90,
                )
                _draw_straight_seg(
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


def _draw_nmr_integrals(
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
    cum: np.ndarray = _cumulative_trapezoid(x_rev, y_data[::-1])
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


def _spectrum_to_nmr_bundle(
    spec: spectroscopy.Spectrum,
    ppm_range: tuple[float, float] | None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[tuple[float, float, str, float | tuple[float, ...], str]],
    tuple[float, float],
]:
    """Extracts sorted NMR arrays and ppm bounds from a ``Spectrum``."""
    order = np.argsort(spec.x)
    ppm_grid = np.asarray(spec.x[order], dtype=np.float64)
    y_curve = np.asarray(spec.y[order], dtype=np.float64)
    stick_pos = np.asarray([p.position for p in spec.peaks], dtype=np.float64)
    stick_wt = np.asarray([p.intensity for p in spec.peaks], dtype=np.float64)
    lo_p = float(ppm_grid[0]) if ppm_grid.size else -0.2
    hi_p = float(ppm_grid[-1]) if ppm_grid.size else 10.0
    if ppm_range is not None:
        lo_p = min(float(ppm_range[0]), float(ppm_range[1]))
        hi_p = max(float(ppm_range[0]), float(ppm_range[1]))
    return ppm_grid, y_curve, stick_pos, stick_wt, [], (lo_p, hi_p)


def _prepare_nmr_data(
    shifts_ppm: spectroscopy.Spectrum | Sequence[Any] | np.ndarray,
    couplings_hz: Sequence[Any] | float | None,
    multiplicities: Sequence[str] | None,
    integrals: Sequence[float] | np.ndarray | None,
    labels: Sequence[str | None] | None,
    cfg: tuple[float, tuple[float, float] | None, float, bool],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[tuple[float, float, str, float | tuple[float, ...], str]],
    tuple[float, float],
]:
    """Builds NMR grid, Lorentzian trace, stick lines, and ppm bounds."""
    mhz, ppm_range, fwhm_ppm, show_tms = cfg
    if isinstance(shifts_ppm, spectroscopy.Spectrum):
        return _spectrum_to_nmr_bundle(shifts_ppm, ppm_range)

    resonances = _coerce_nmr_resonances(
        shifts_ppm, couplings_hz, multiplicities, integrals, labels
    )
    pairs = [
        expand_nmr_multiplet(sh, integ, mult, jv, mhz)
        for sh, integ, mult, jv, _ in resonances
    ]
    stick_pos = np.concatenate([p[0] for p in pairs])
    stick_wt = np.concatenate([p[1] for p in pairs])
    if ppm_range is not None:
        lo_p = min(float(ppm_range[0]), float(ppm_range[1]))
        hi_p = max(float(ppm_range[0]), float(ppm_range[1]))
    else:
        min_s, max_s = float(np.min(stick_pos)), float(np.max(stick_pos))
        pad = max(0.35, 0.12 * (max_s - min_s))
        lo_p = min(-0.25, min_s - pad) if show_tms else (min_s - pad)
        hi_p = max_s + pad
    raw_n = int(round((hi_p - lo_p) / max(fwhm_ppm * 0.22, 1e-4)))
    ppm_grid = np.linspace(lo_p, hi_p, min(max(1600, raw_n), 8000))
    y_curve = spectroscopy.broaden_peaks(
        stick_pos,
        stick_wt,
        ppm_grid,
        fwhm=fwhm_ppm,
        profile="lorentzian",
        normalize="area",
    )
    return ppm_grid, y_curve, stick_pos, stick_wt, resonances, (lo_p, hi_p)


def _render_nmr_axes(
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
    y_max = max(float(np.max(y_curve)) if y_curve.size else 1.0, 1e-6)

    base_ln = ax.axhline(
        0.0, color=palettes.MUTED_GRAY, lw=sw.guide, alpha=0.55, zorder=1.5
    )
    base_ln.set_sketch_params(0.0, 0.0, 0.0)
    if stick_pos.size > 0:
        w_max = max(float(np.max(stick_wt)), 1e-9)
        ax.vlines(
            stick_pos,
            0.0,
            stick_wt * (0.85 * y_max / w_max),
            color=series[6 % len(series)],
            lw=sw.guide,
            alpha=0.62,
            zorder=2.2,
        )
    ax.plot(ppm_grid, y_curve, color=line_col, lw=sw.primary, zorder=3.6)

    if show_tms and lo_p <= 0.0 <= hi_p:
        _draw_straight_seg(
            ax,
            [0.0, 0.0],
            [0.0, 0.20 * y_max],
            palettes.MUTED_GRAY,
            sw.guide + 0.15,
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

    y_int_asc: np.ndarray | None = None
    if show_integrals and ppm_grid.size >= 2:
        y_int_asc = _draw_nmr_integrals(
            ax,
            ppm_grid,
            y_curve,
            resonances,
            y_max,
            (series[3 % len(series)], colors[0], colors[1]),
            sw,
        )

    top_needed = 1.18 * y_max
    if show_trees and resonances:
        for res in resonances:
            win = (ppm_grid >= res[0] - 0.25) & (ppm_grid <= res[0] + 0.25)
            y_pk = (
                float(np.max(y_curve[win])) if np.any(win) else 0.8 * y_max
            )
            if y_int_asc is not None and np.any(win):
                y_pk = max(y_pk, float(np.max(y_int_asc[win])))
            top_needed = max(
                top_needed,
                _draw_nmr_splitting_tree(
                    ax,
                    res,
                    (mhz, y_pk, y_max, hi_p - lo_p),
                    colors,
                    sw,
                ),
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
    bundle = _prepare_nmr_data(
        shifts_ppm,
        couplings_hz,
        multiplicities,
        integrals,
        labels,
        (spectrometer_mhz, ppm_range, fwhm_ppm, show_tms),
    )
    series = palettes.get_palette(palette)
    line_col = series[5 % len(series)] if color is None else color
    preset_obj = theme.get_preset(preset)

    with theme.xkcd_style(preset=preset):
        if ax is None:
            _, ax = plt.subplots(figsize=(6.8, 4.3))
        else:
            theme.style_existing_axes(ax, preset=preset)

        _render_nmr_axes(
            ax,
            bundle,
            (spectrometer_mhz, show_integrals, show_trees, show_tms),
            (
                line_col,
                series,
                (preset_obj.ink_color, preset_obj.axes_facecolor),
                theme.stroke_weights(preset),
            ),
        )
        nuc_lbl = _format_nucleus_symbol(nucleus)
        ax.set_xlabel(f"{nuc_lbl} Chemical Shift $\\delta$ (ppm)")
        ax.set_ylabel("Intensity (arb. units)")
        theme.apply_xkcd_spines(ax, preset=preset)
        if title:
            ax.set_title(title)
    return ax


def compute_shirley_background(
    binding_energy: Sequence[float] | np.ndarray,
    intensity: Sequence[float] | np.ndarray,
    max_iter: int = 60,
    tol: float = 1e-7,
) -> np.ndarray:
    """Computes an iterative Shirley inelastic XPS background curve.

    Evaluates ``B(E)`` in ascending binding energy so ``B(E_min) = I_low``,
    ``B(E_max) = I_high``, and ``B(E_i) - I_low`` is proportional to the
    integrated net intensity ``max(I - B, 0)`` at lower binding energies
    ``[E_min, E_i]``.

    Args:
      binding_energy: 1D binding energy grid in eV (ascending or descending).
      intensity: 1D raw photoemission intensity array matching
        ``binding_energy``.
      max_iter: Maximum fixed-point iterations.
      tol: Relative convergence tolerance.

    Returns:
      1D float64 background array matching the order of ``binding_energy``.
    """
    be = np.asarray(binding_energy, dtype=np.float64)
    inten = np.asarray(intensity, dtype=np.float64)
    valid = (
        be.ndim == 1
        and inten.ndim == 1
        and be.shape == inten.shape
        and be.size >= 2
    )
    if not valid:
        raise exceptions.StructureError(
            "Shirley background requires matching 1D arrays with >= 2 points."
        )
    order = np.argsort(be)
    e_asc = be[order]
    y_asc = inten[order]
    i_low = float(y_asc[0])
    i_high = float(y_asc[-1])
    span_e = max(float(e_asc[-1] - e_asc[0]), 1e-12)
    b_cur: np.ndarray = i_low + (i_high - i_low) * (
        (e_asc - e_asc[0]) / span_e
    )
    conv_thresh = float(tol) * max(float(np.ptp(y_asc)), 1.0)

    for _ in range(max(1, int(max_iter))):
        net = np.maximum(y_asc - b_cur, 0.0)
        cum_area = _cumulative_trapezoid(e_asc, net)
        tot_area = float(cum_area[-1])
        if tot_area <= 1e-12:
            break
        b_target = i_low + (i_high - i_low) * (cum_area / tot_area)
        b_next = 0.65 * b_target + 0.35 * b_cur
        b_next[0] = i_low
        b_next[-1] = i_high
        if float(np.max(np.abs(b_next - b_cur))) <= conv_thresh:
            b_cur = b_next
            break
        b_cur = b_next

    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(be.size)
    return np.asarray(b_cur[inv_order], dtype=np.float64)


def _compute_xps_background(
    be_asc: np.ndarray,
    y_asc: np.ndarray,
    mode: str | None,
) -> np.ndarray:
    """Computes Shirley, linear, or zero XPS baseline on ascending BE."""
    if mode is None:
        return np.zeros_like(y_asc, dtype=np.float64)
    key = mode.strip().lower()
    if key in ("none", "zero", "off"):
        return np.zeros_like(y_asc, dtype=np.float64)
    if key == "linear":
        span = max(float(be_asc[-1] - be_asc[0]), 1e-12)
        return np.asarray(
            y_asc[0] + (y_asc[-1] - y_asc[0]) * ((be_asc - be_asc[0]) / span),
            dtype=np.float64,
        )
    if key == "shirley":
        return compute_shirley_background(be_asc, y_asc)
    raise exceptions.StructureError(
        f"Unsupported XPS background {mode!r}; expected 'shirley', 'linear', "
        "or None."
    )


def _resolve_comp_dict_center_amp(
    comp: Mapping[str, Any],
) -> tuple[float, float, str]:
    """Extracts center BE, amplitude/area, and normalization mode from dict."""
    c_raw = comp.get("center", comp.get("position", comp.get("be")))
    c_be = float(comp.get("binding_energy", 0.0) if c_raw is None else c_raw)
    if "area" in comp and "amplitude" not in comp and "height" not in comp:
        return c_be, float(comp["area"]), "area"
    a_raw = comp.get(
        "amplitude", comp.get("height", comp.get("intensity", 1.0))
    )
    return c_be, float(a_raw), "height"


def _evaluate_one_xps_component(
    be_asc: np.ndarray,
    comp: Any,
    default_fwhm: float,
    order: np.ndarray,
) -> list[tuple[str, float, np.ndarray]]:
    """Evaluates a single XPS component dict or tuple into ``(lbl, be0, y)``."""
    if isinstance(comp, Mapping):
        c_be, amp, norm_m = _resolve_comp_dict_center_amp(comp)
        w_fwhm = float(comp.get("fwhm", default_fwhm))
        prof = str(comp.get("profile", "pseudo_voigt"))
        lbl = str(comp.get("label", f"{c_be:.1f} eV"))
        y_c = spectroscopy.broaden_peaks(
            [c_be], [amp], be_asc, fwhm=w_fwhm, profile=prof, normalize=norm_m
        )
        out = [(lbl, c_be, y_c)]
        if "splitting" in comp or "delta_be" in comp:
            split = float(comp.get("splitting", comp.get("delta_be", 0.0)))
            ratio = float(comp.get("branching_ratio", 0.5))
            lbl2 = str(comp.get("doublet_label", f"{lbl} (SO)"))
            y_c2 = spectroscopy.broaden_peaks(
                [c_be + split],
                [amp * ratio],
                be_asc,
                fwhm=w_fwhm,
                profile=prof,
                normalize=norm_m,
            )
            out.append((lbl2, c_be + split, y_c2))
        return out

    if isinstance(comp, Sequence) and not isinstance(comp, str):
        items = list(comp)
        if len(items) == 2 and isinstance(items[0], str):
            arr = np.asarray(items[1], dtype=np.float64)[order]
            pk_be = float(be_asc[int(np.argmax(arr))])
            return [(items[0], pk_be, arr)]
        if isinstance(items[0], str) and len(items) >= 3:
            lbl, c_be, amp = items[0], float(items[1]), float(items[2])
            w_fwhm = float(items[3]) if len(items) >= 4 else default_fwhm
        elif len(items) >= 2:
            c_be, amp = float(items[0]), float(items[1])
            if len(items) == 3 and isinstance(items[2], str):
                w_fwhm, lbl = default_fwhm, items[2]
            else:
                w_fwhm = float(items[2]) if len(items) >= 3 else default_fwhm
                lbl = str(items[3]) if len(items) >= 4 else f"{c_be:.1f} eV"
        else:
            raise exceptions.StructureError(f"Invalid XPS component {comp!r}.")
        y_c = spectroscopy.broaden_peaks(
            [c_be], [amp], be_asc, fwhm=w_fwhm, profile="pseudo_voigt"
        )
        return [(lbl, c_be, y_c)]

    raise exceptions.StructureError(f"Unsupported XPS component {comp!r}.")


def _annotate_xps_components(
    ax: maxes.Axes,
    be_asc: np.ndarray,
    envelope: np.ndarray,
    comp_curves: Sequence[tuple[str, float, np.ndarray]],
    colors: tuple[str, str],
) -> float:
    """Places non-overlapping haloed ink labels above XPS component peaks."""
    ink_col, bg_col = colors
    y_span = max(float(np.ptp(envelope)), 1.0)
    be_span = max(float(be_asc[-1] - be_asc[0]), 1.0)
    placed: list[tuple[float, float]] = []
    top_y = float(np.max(envelope))

    for lbl, c_be, _ in sorted(
        comp_curves, key=lambda item: float(np.max(item[2])), reverse=True
    ):
        if not lbl:
            continue
        y_lbl = float(np.interp(c_be, be_asc, envelope)) + 0.055 * y_span
        for px, py in placed:
            near_x = abs(c_be - px) < 0.14 * be_span
            near_y = abs(y_lbl - py) < 0.09 * y_span
            if near_x and near_y:
                y_lbl = max(y_lbl, py + 0.095 * y_span)
        placed.append((c_be, y_lbl))
        top_y = max(top_y, y_lbl + 0.07 * y_span)
        theme.place_halo_text(
            ax,
            c_be,
            y_lbl,
            lbl,
            color=ink_col,
            bg_color=bg_col,
            fontsize=8.4,
            va="bottom",
            pad=0.07,
            alpha=0.88,
            zorder=6.5,
        )
    return top_y


def _prepare_xps_arrays(
    binding_energy: Sequence[float] | np.ndarray | spectroscopy.Spectrum,
    intensity: Sequence[float] | np.ndarray | None,
    components: Sequence[Any] | None,
    fwhm: float,
    background: str | None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[tuple[str, float, np.ndarray]],
    bool,
]:
    """Coerces XPS energy/intensity grids, baseline, and component curves."""
    be_raw: np.ndarray
    y_raw: np.ndarray | None
    if isinstance(binding_energy, spectroscopy.Spectrum) and intensity is None:
        be_raw = np.asarray(binding_energy.x, dtype=np.float64)
        y_raw = np.asarray(binding_energy.y, dtype=np.float64)
    else:
        be_raw = np.asarray(binding_energy, dtype=np.float64)
        y_raw = (
            None
            if intensity is None
            else np.asarray(intensity, dtype=np.float64)
        )
    if be_raw.ndim != 1 or be_raw.size < 2:
        raise exceptions.StructureError(
            "XPS binding_energy must be a 1D array with >= 2 points."
        )
    order = np.argsort(be_raw)
    be_asc = be_raw[order]
    comp_curves: list[tuple[str, float, np.ndarray]] = []
    if components is not None:
        for comp in components:
            comp_curves.extend(
                _evaluate_one_xps_component(be_asc, comp, float(fwhm), order)
            )

    has_raw = y_raw is not None
    y_asc: np.ndarray
    if y_raw is not None:
        if y_raw.ndim != 1 or y_raw.shape != be_raw.shape:
            raise exceptions.StructureError(
                "XPS intensity must be a 1D array matching binding_energy."
            )
        y_asc = y_raw[order]
    elif comp_curves:
        y_asc = np.asarray(
            np.sum([c[2] for c in comp_curves], axis=0), dtype=np.float64
        )
    else:
        raise exceptions.StructureError(
            "XPS plot requires intensity data or non-empty components."
        )

    bg_asc = _compute_xps_background(be_asc, y_asc, background)
    env_asc: np.ndarray = (
        bg_asc + np.sum([c[2] for c in comp_curves], axis=0)
        if comp_curves
        else y_asc.copy()
    )
    return be_asc, y_asc, bg_asc, env_asc, comp_curves, has_raw


def _draw_xps_traces(
    ax: maxes.Axes,
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    comp_curves: Sequence[tuple[str, float, np.ndarray]],
    background: str | None,
    has_raw: bool,
    series: Sequence[str],
    colors: tuple[str, str],
    sw: theme.StrokeHierarchy,
) -> None:
    """Renders XPS baseline, shaded components, raw scatter, and envelope."""
    be_asc, y_asc, bg_asc, env_asc = arrays
    ink_col, bg_col = colors
    if background is not None and background.strip().lower() not in (
        "none",
        "zero",
        "off",
    ):
        ax.plot(
            be_asc,
            bg_asc,
            color=palettes.MUTED_GRAY,
            linestyle="--",
            lw=sw.guide + 0.15,
            alpha=0.85,
            label=f"{background.strip().title()} BG",
            zorder=2.0,
        )

    for idx, (_, _, y_c) in enumerate(comp_curves):
        c_col = series[(idx + 1) % len(series)]
        if c_col == "#F0E442":
            c_col = series[(idx + 2) % len(series)]
        y_top_c = bg_asc + y_c
        ax.plot(be_asc, y_top_c, color=c_col, lw=sw.overlay, zorder=3.0)
        patterns.shade_between(
            ax,
            be_asc,
            bg_asc,
            y_top_c,
            color=c_col,
            shade=None,
            alpha=0.22,
            zorder=2.3,
        )

    if comp_curves and has_raw:
        stride = max(1, be_asc.size // 90)
        ax.scatter(
            be_asc[::stride],
            y_asc[::stride],
            s=22.0,
            facecolor=bg_col,
            edgecolor=ink_col,
            linewidth=1.0,
            alpha=0.82,
            label="Raw Data",
            zorder=4.2,
        )
        ax.plot(
            be_asc,
            env_asc,
            color=series[5 % len(series)],
            lw=sw.primary,
            label="Total Envelope",
            zorder=4.0,
        )
    else:
        ax.plot(
            be_asc,
            y_asc,
            color=series[5 % len(series)],
            lw=sw.primary,
            label="XPS Spectrum",
            zorder=4.0,
        )


def plot_xps(
    binding_energy: Sequence[float] | np.ndarray | spectroscopy.Spectrum,
    intensity: Sequence[float] | np.ndarray | None = None,
    components: Sequence[Any] | None = None,
    background: str | None = "shirley",
    invert_x: bool = True,
    show_residuals: bool = False,
    fwhm: float = 1.2,
    ax: maxes.Axes | None = None,
    palette: str | Sequence[str] = "okabe_ito",
    preset: str | theme.ThemePreset = "classic",
    title: str | None = None,
) -> maxes.Axes:
    """Plots an XPS core-level spectrum with Shirley background & Voigt fits.

    Args:
      binding_energy: 1D binding energy array in eV (or ``Spectrum`` when
        ``intensity`` is ``None``).
      intensity: 1D photoemission intensity array (optional if ``components``
        are supplied to synthesize the envelope).
      components: Optional sequence of Voigt/GL component dicts or tuples.
      background: ``"shirley"`` (iterative inelastic background), ``"linear"``,
        or ``None``.
      invert_x: Invert horizontal binding-energy axis (standard XPS convention).
      show_residuals: Overlay ``I_raw - I_envelope`` residual strip at top.
      fwhm: Default pseudo-Voigt component FWHM in eV.
      ax: Optional target Matplotlib ``Axes``.
      palette: Series palette name or hex sequence.
      preset: Visual theme preset name or ``ThemePreset``.
      title: Optional plot heading.

    Returns:
      Styled Matplotlib ``Axes`` containing the XPS core-level deconvolution.
    """
    be_asc, y_asc, bg_asc, env_asc, comp_curves, has_raw = _prepare_xps_arrays(
        binding_energy, intensity, components, fwhm, background
    )
    series = palettes.get_palette(palette)
    preset_obj = theme.get_preset(preset)
    colors = (preset_obj.ink_color, preset_obj.axes_facecolor)
    sw = theme.stroke_weights(preset)

    with theme.xkcd_style(preset=preset):
        if ax is None:
            _, ax = plt.subplots(figsize=(6.4, 4.4))
        else:
            theme.style_existing_axes(ax, preset=preset)

        _draw_xps_traces(
            ax,
            (be_asc, y_asc, bg_asc, env_asc),
            comp_curves,
            background,
            has_raw,
            series,
            colors,
            sw,
        )
        top_y = max(float(np.max(y_asc)), float(np.max(env_asc)))
        y_lo = min(float(np.min(y_asc)), float(np.min(bg_asc)))
        if comp_curves:
            top_y = max(
                top_y,
                _annotate_xps_components(
                    ax, be_asc, env_asc, comp_curves, colors
                ),
            )

        y_span = max(top_y - y_lo, 1.0)
        if show_residuals and comp_curves and has_raw:
            resid = y_asc - env_asc
            r_base = top_y + 0.14 * y_span
            r_ref = ax.axhline(
                r_base,
                color=palettes.MUTED_GRAY,
                linestyle=":",
                lw=sw.guide,
                zorder=2.0,
            )
            r_ref.set_sketch_params(0.0, 0.0, 0.0)
            ax.plot(
                be_asc,
                r_base + resid,
                color=series[6 % len(series)],
                lw=sw.guide + 0.15,
                label="Residual",
                zorder=3.8,
            )
            r_top = r_base + float(np.max(np.abs(resid)))
            top_y = max(r_top, top_y + 0.22 * y_span) + 0.28 * y_span

        ax.set_xlim(float(be_asc[0]), float(be_asc[-1]))
        if invert_x:
            ax.invert_xaxis()
        ax.set_ylim(y_lo - 0.06 * y_span, top_y + 0.10 * y_span)
        ax.set_xlabel("Binding Energy (eV)")
        ax.set_ylabel("Intensity (counts / arb. units)")
        if comp_curves or background is not None:
            ax.legend(loc="upper left", frameon=True, fontsize=8.3)
        theme.apply_xkcd_spines(ax, preset=preset)
        if title:
            ax.set_title(title)
    return ax
