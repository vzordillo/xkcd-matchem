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
"""Electrochemistry and battery characterization plots (CV, Tafel, dQ/dV)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from matplotlib import axes as maxes
from matplotlib import pyplot as plt
import numpy as np

from xkcd_matchem.core import exceptions
from xkcd_matchem.style import palettes
from xkcd_matchem.style import patterns
from xkcd_matchem.style import theme


def _format_echem_label(text: str) -> str:
    """Replaces Unicode superscripts lacking xkcd glyphs with mathtext."""
    return text.replace("²", "$^2$").replace("³", "$^3$")


def _as_1d_pair(
    xs_raw: Any,
    ys_raw: Any,
    context: str,
    min_len: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Coerces and validates a matching pair of 1D float64 arrays."""
    xs = np.asarray(xs_raw, dtype=np.float64)
    ys = np.asarray(ys_raw, dtype=np.float64)
    if (
        xs.ndim != 1
        or ys.ndim != 1
        or xs.shape != ys.shape
        or xs.size < min_len
    ):
        raise exceptions.StructureError(
            f"{context} requires matching 1D arrays with >= {min_len} points."
        )
    return xs, ys


def _coerce_cv_scans(
    potential: Any,
    current: Any,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Normalizes single-trace or multi-scan CV inputs into ``(E, j)`` pairs."""
    if current is None:
        if not isinstance(potential, Sequence) or len(potential) == 0:
            raise exceptions.StructureError(
                "Cyclic voltammetry requires a non-empty (E, j) scan sequence."
            )
        scans: list[tuple[np.ndarray, np.ndarray]] = []
        for item in potential:
            if not isinstance(item, Sequence) or len(item) != 2:
                raise exceptions.StructureError(
                    "Each CV scan item must be a (potential, current) pair."
                )
            scans.append(_as_1d_pair(item[0], item[1], "CV scan", min_len=3))
        return scans

    pot_arr = np.asarray(potential, dtype=object)
    if pot_arr.ndim == 1 and pot_arr.size > 0 and np.ndim(pot_arr[0]) == 1:
        cur_seq = list(current)
        pot_seq = list(potential)
        if len(pot_seq) != len(cur_seq):
            raise exceptions.StructureError(
                "Multi-scan potential and current lists must match in length."
            )
        return [
            _as_1d_pair(p_i, c_i, "CV scan", min_len=3)
            for p_i, c_i in zip(pot_seq, cur_seq, strict=True)
        ]

    if (
        isinstance(potential, np.ndarray)
        and potential.ndim == 2
        and isinstance(current, np.ndarray)
        and current.ndim == 2
    ):
        if potential.shape != current.shape or potential.shape[0] == 0:
            raise exceptions.StructureError(
                "2D CV potential and current arrays must have matching shapes."
            )
        return [
            _as_1d_pair(potential[k], current[k], "CV scan", min_len=3)
            for k in range(potential.shape[0])
        ]

    return [_as_1d_pair(potential, current, "Cyclic voltammetry", min_len=3)]


def _detect_cv_peaks(
    pot: np.ndarray,
    cur: np.ndarray,
) -> tuple[float, float, float, float, float]:
    """Detects anodic/cathodic peak extrema and half-wave potential E_1/2."""
    idx_pa = int(np.argmax(cur))
    idx_pc = int(np.argmin(cur))
    e_pa = float(pot[idx_pa])
    j_pa = float(cur[idx_pa])
    e_pc = float(pot[idx_pc])
    j_pc = float(cur[idx_pc])
    e_half = 0.5 * (e_pa + e_pc)
    return e_pa, j_pa, e_pc, j_pc, e_half


def _draw_cv_scan_arrows(
    ax: maxes.Axes,
    pot: np.ndarray,
    cur: np.ndarray,
    color: str,
    sw: theme.StrokeHierarchy,
) -> None:
    """Places subtle forward and reverse sweep direction arrows along a CV."""
    n_pts = pot.size
    if n_pts < 8:
        return
    for frac in (0.18, 0.68):
        idx = min(max(1, int(round(frac * (n_pts - 1)))), n_pts - 2)
        if abs(float(pot[idx + 1] - pot[idx - 1])) < 1e-9:
            continue
        ax.annotate(
            "",
            xy=(float(pot[idx + 1]), float(cur[idx + 1])),
            xytext=(float(pot[idx - 1]), float(cur[idx - 1])),
            arrowprops={
                "arrowstyle": "->",
                "color": color,
                "lw": sw.leader,
            },
            zorder=4.8,
        )


def _series_color_for_theme(
    series: Sequence[str],
    index: int,
    preset_obj: theme.ThemePreset,
) -> str:
    """Resolves a series color, replacing near-black traces on dark themes."""
    col = str(series[index % len(series)])
    if (
        theme.is_dark_background(preset_obj.axes_facecolor)
        and theme.relative_luminance(col) < 0.04
    ):
        return preset_obj.ink_color
    return col


def _annotate_cv_peaks_and_ehalf(
    ax: maxes.Axes,
    peaks: tuple[float, float, float, float, float],
    flags: tuple[bool, bool],
    colors: tuple[str, str, str, str],
    sw: theme.StrokeHierarchy,
) -> None:
    """Draws straight E_1/2 guide and haloed ink peak/separation callouts."""
    e_pa, j_pa, e_pc, j_pc, e_half = peaks
    annotate_peaks, show_e_half = flags
    trace_col, ink_col, bg_col, guide_col = colors

    if show_e_half:
        eh_line = ax.axvline(
            e_half,
            color=guide_col,
            linestyle="--",
            lw=sw.guide,
            alpha=0.80,
            zorder=2.0,
        )
        eh_line.set_sketch_params(0.0, 0.0, 0.0)
        theme.place_halo_text(
            ax,
            e_half,
            0.5 * (j_pa + j_pc),
            f"$E_{{1/2}}$ = {e_half:.3f} V",
            color=ink_col,
            bg_color=bg_col,
            fontsize=8.6,
            pad=0.10,
            alpha=0.90,
            zorder=6.5,
        )

    if not annotate_peaks:
        return

    ax.scatter(
        [e_pa, e_pc],
        [j_pa, j_pc],
        s=48.0,
        facecolor=trace_col,
        edgecolor=ink_col,
        linewidth=1.25,
        zorder=5.5,
    )
    for px, py, lbl, dy_pt in (
        (e_pa, j_pa, f"$E_{{pa}}$ ({e_pa:.2f} V)", 10),
        (e_pc, j_pc, f"$E_{{pc}}$ ({e_pc:.2f} V)", -12),
    ):
        ann = ax.annotate(
            lbl,
            xy=(px, py),
            xytext=(0, dy_pt),
            textcoords="offset points",
            ha="center",
            va="bottom" if dy_pt > 0 else "top",
            fontsize=8.6,
            color=ink_col,
            bbox=theme.label_halo_bbox(bg_col, pad=0.08, alpha=0.88),
            zorder=6.6,
        )
        patch = ann.get_bbox_patch()
        if patch is not None:
            patch.set_sketch_params(0.0, 0.0, 0.0)

    dep_mv = abs(e_pa - e_pc) * 1000.0
    dep_txt = (
        f"$\\Delta E_p$ = {dep_mv:.0f} mV"
        if dep_mv < 1000.0
        else f"$\\Delta E_p$ = {abs(e_pa - e_pc):.2f} V"
    )
    badge = ax.text(
        0.03,
        0.95,
        dep_txt,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.8,
        color=ink_col,
        bbox=theme.label_halo_bbox(bg_col, pad=0.10, alpha=0.88),
        zorder=6.6,
    )
    b_patch = badge.get_bbox_patch()
    if b_patch is not None:
        b_patch.set_sketch_params(0.0, 0.0, 0.0)


def _set_cv_limits(
    ax: maxes.Axes,
    scans: Sequence[tuple[np.ndarray, np.ndarray]],
    invert_x: bool,
) -> None:
    """Applies padded potential/current bounds and optional US inversion."""
    all_e = np.concatenate([p for p, _ in scans])
    all_j = np.concatenate([c for _, c in scans])
    e_span = max(float(np.ptp(all_e)), 0.1)
    j_span = max(float(np.ptp(all_j)), 1e-3)
    ax.set_xlim(
        float(np.min(all_e)) - 0.06 * e_span,
        float(np.max(all_e)) + 0.06 * e_span,
    )
    ax.set_ylim(
        float(np.min(all_j)) - 0.22 * j_span,
        float(np.max(all_j)) + 0.24 * j_span,
    )
    if invert_x:
        ax.invert_xaxis()


def plot_cyclic_voltammetry(
    potential: Any,
    current: Any = None,
    convention: str = "iupac",
    annotate_peaks: bool = True,
    show_e_half: bool = True,
    show_scan_arrows: bool = True,
    scan_labels: Sequence[str] | None = None,
    potential_label: str = "Potential E (V vs. RHE)",
    current_label: str = "Current Density j (mA/cm²)",
    ax: maxes.Axes | None = None,
    palette: str | Sequence[str] = "okabe_ito",
    preset: str | theme.ThemePreset = "classic",
    title: str | None = None,
) -> maxes.Axes:
    """Plots single- or multi-scan cyclic voltammetry under IUPAC or US style.

    Args:
      potential: 1D potential array or sequence of ``(potential, current)``
        scans.
      current: 1D current array (or sequence matching ``potential``); ``None``
        when ``potential`` is already a sequence of ``(E, j)`` pairs.
      convention: ``"iupac"`` (positive potential right) or ``"us"`` (inverted
        polarographic potential axis).
      annotate_peaks: Label anodic/cathodic extrema ``E_pa``, ``E_pc``, and
        ``Delta E_p``.
      show_e_half: Draw straight vertical ``E_{1/2}`` guide and label.
      show_scan_arrows: Draw directional sweep arrows along the primary trace.
      scan_labels: Optional legend labels for multi-scan voltammograms.
      potential_label: Horizontal axis label.
      current_label: Vertical axis label.
      ax: Optional target Matplotlib ``Axes``.
      palette: Series palette name or hex sequence.
      preset: Visual theme preset name or ``ThemePreset``.
      title: Optional plot heading.

    Returns:
      Styled Matplotlib ``Axes`` containing the cyclic voltammogram.
    """
    conv = convention.strip().lower()
    if conv not in ("iupac", "us"):
        raise exceptions.StructureError(
            f"Unsupported CV convention {convention!r}; use 'iupac' or 'us'."
        )
    scans = _coerce_cv_scans(potential, current)
    if scan_labels is not None and len(scan_labels) != len(scans):
        raise exceptions.StructureError(
            "scan_labels length must match the number of CV scans."
        )
    series = palettes.get_palette(palette)
    preset_obj = theme.get_preset(preset)
    guide_col = theme.guide_color(preset_obj)
    sw = theme.stroke_weights(preset)
    ref_idx = max(range(len(scans)), key=lambda i: float(np.ptp(scans[i][1])))
    peaks = _detect_cv_peaks(scans[ref_idx][0], scans[ref_idx][1])

    with theme.xkcd_style(preset=preset):
        if ax is None:
            _, ax = plt.subplots(figsize=(6.2, 4.4))
        else:
            theme.style_existing_axes(ax, preset=preset)

        z_ln = ax.axhline(
            0.0,
            color=guide_col,
            linestyle=":",
            lw=sw.guide,
            alpha=0.65,
            zorder=1.5,
        )
        z_ln.set_sketch_params(0.0, 0.0, 0.0)

        for idx, (pot_s, cur_s) in enumerate(scans):
            ax.plot(
                pot_s,
                cur_s,
                color=_series_color_for_theme(series, 5 + idx, preset_obj),
                lw=sw.primary if idx == ref_idx else sw.overlay + 0.15,
                label=scan_labels[idx] if scan_labels is not None else None,
                zorder=3.5 + idx * 0.1,
            )

        if show_scan_arrows:
            _draw_cv_scan_arrows(
                ax,
                scans[ref_idx][0],
                scans[ref_idx][1],
                preset_obj.ink_color,
                sw,
            )
        _annotate_cv_peaks_and_ehalf(
            ax,
            peaks,
            (annotate_peaks, show_e_half),
            (
                _series_color_for_theme(series, 5 + ref_idx, preset_obj),
                preset_obj.ink_color,
                preset_obj.axes_facecolor,
                guide_col,
            ),
            sw,
        )
        _set_cv_limits(ax, scans, conv == "us")
        ax.set_xlabel(_format_echem_label(potential_label))
        ax.set_ylabel(_format_echem_label(current_label))
        if scan_labels is not None:
            ax.legend(loc="lower right", frameon=True, fontsize=8.5)
        theme.apply_xkcd_spines(ax, preset=preset)
        if title:
            ax.set_title(title, color=preset_obj.ink_color)
    return ax


def _select_tafel_mask(
    log_j: np.ndarray,
    eta: np.ndarray,
    fit_range: tuple[float, float] | None,
) -> np.ndarray:
    """Resolves linear Tafel fit window along overpotential or log10(|j|)."""
    if fit_range is None:
        return np.ones(log_j.shape, dtype=bool)
    lo = min(float(fit_range[0]), float(fit_range[1]))
    hi = max(float(fit_range[0]), float(fit_range[1]))
    eta_in = (
        lo >= float(np.min(eta)) - 1e-9 and hi <= float(np.max(eta)) + 1e-9
    )
    lj_in = (
        lo >= float(np.min(log_j)) - 1e-9 and hi <= float(np.max(log_j)) + 1e-9
    )
    mask_eta = (eta >= lo) & (eta <= hi)
    mask_lj = (log_j >= lo) & (log_j <= hi)
    if eta_in and not lj_in and int(np.sum(mask_eta)) >= 2:
        return mask_eta
    if lj_in and not eta_in and int(np.sum(mask_lj)) >= 2:
        return mask_lj
    if int(np.sum(mask_eta)) >= 2:
        return mask_eta
    if int(np.sum(mask_lj)) >= 2:
        return mask_lj
    raise exceptions.StructureError(
        f"Tafel fit_range {fit_range!r} selects fewer than 2 data points."
    )


def _fit_tafel_ols(
    log_j: np.ndarray,
    eta: np.ndarray,
    unit: str,
) -> tuple[float, float, float, float]:
    """Computes OLS Tafel slope (mV/dec) and exchange current log10(j0)."""
    coeffs = np.polyfit(log_j, eta, 1)
    slope = float(coeffs[0])
    intercept = float(coeffs[1])
    scale = 1000.0 if unit.strip().lower() == "v" else 1.0
    b_mv_dec = abs(slope) * scale
    log_j0 = -intercept / slope if abs(slope) > 1e-12 else float("nan")
    return slope, intercept, b_mv_dec, log_j0


def _prepare_tafel_fit(
    overpotential: Sequence[float] | np.ndarray,
    current_density: Sequence[float] | np.ndarray,
    unit: str,
    fit_range: tuple[float, float] | None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[float, float, float, float],
]:
    """Validates Tafel inputs and evaluates OLS linear slope parameters."""
    unit_key = unit.strip()
    if unit_key.lower() not in ("mv", "v"):
        raise exceptions.StructureError(
            f"Unsupported Tafel overpotential unit {unit!r}; use 'mV' or 'V'."
        )
    eta_raw, j_raw = _as_1d_pair(
        overpotential, current_density, "Tafel plot", min_len=2
    )
    pos_mask = np.abs(j_raw) > 0.0
    if int(np.sum(pos_mask)) < 2:
        raise exceptions.StructureError(
            "Tafel plot requires at least 2 non-zero current_density points."
        )
    eta = eta_raw[pos_mask]
    log_j = np.log10(np.abs(j_raw[pos_mask]))
    fit_mask = _select_tafel_mask(log_j, eta, fit_range)
    fit_vals = _fit_tafel_ols(log_j[fit_mask], eta[fit_mask], unit_key)
    return log_j, eta, fit_mask, fit_vals


def _draw_tafel_slope_badge(
    ax: maxes.Axes,
    b_mv_dec: float,
    log_j0: float,
    ink_col: str,
    bg_col: str,
) -> None:
    """Places a haloed ink Tafel slope and exchange-current summary box."""
    summary = f"b = {b_mv_dec:.1f} mV/dec"
    if np.isfinite(log_j0):
        summary += f"\nlog10(j0) = {log_j0:.2f}"
    txt = ax.text(
        0.05,
        0.93,
        summary,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.0,
        color=ink_col,
        bbox=theme.label_halo_bbox(bg_col, pad=0.14, alpha=0.90),
        zorder=6.5,
    )
    patch = txt.get_bbox_patch()
    if patch is not None:
        patch.set_sketch_params(0.0, 0.0, 0.0)


def plot_tafel(
    overpotential: Sequence[float] | np.ndarray,
    current_density: Sequence[float] | np.ndarray,
    unit: str = "mV",
    fit_range: tuple[float, float] | None = None,
    annotate_slope: bool = True,
    current_unit: str = "mA/cm²",
    ax: maxes.Axes | None = None,
    color: str | None = None,
    palette: str | Sequence[str] = "okabe_ito",
    preset: str | theme.ThemePreset = "classic",
    title: str | None = None,
) -> maxes.Axes:
    """Plots overpotential ``eta`` vs ``log10(|j|)`` with an OLS Tafel fit.

    Args:
      overpotential: Overpotential ``eta`` values in ``unit`` (``"mV"`` or
        ``"V"``).
      current_density: Measured current density ``j`` (non-zero entries used).
      unit: Overpotential unit (``"mV"`` or ``"V"``).
      fit_range: Optional ``(min, max)`` window in ``eta`` or ``log10(|j|)``
        restricting the linear Tafel regression.
      annotate_slope: Place haloed ink ``b = ... mV/dec`` & ``log10(j0)`` box.
      current_unit: Current density unit displayed in the x-axis label.
      ax: Optional target Matplotlib ``Axes``.
      color: Optional data marker/trace color override.
      palette: Series palette name or hex sequence.
      preset: Visual theme preset name or ``ThemePreset``.
      title: Optional plot heading.

    Returns:
      Styled Matplotlib ``Axes`` containing the Tafel plot.
    """
    log_j, eta, fit_mask, fit_vals = _prepare_tafel_fit(
        overpotential, current_density, unit, fit_range
    )
    series = palettes.get_palette(palette)
    preset_obj = theme.get_preset(preset)
    sw = theme.stroke_weights(preset)

    with theme.xkcd_style(preset=preset):
        if ax is None:
            _, ax = plt.subplots(figsize=(6.0, 4.3))
        else:
            theme.style_existing_axes(ax, preset=preset)

        ax.scatter(
            log_j,
            eta,
            s=58.0,
            facecolor=series[5 % len(series)] if color is None else color,
            edgecolor=preset_obj.ink_color,
            linewidth=1.25,
            zorder=4.5,
            label="Data",
        )
        x_fit = np.linspace(
            float(np.min(log_j[fit_mask])),
            float(np.max(log_j[fit_mask])),
            80,
        )
        ax.plot(
            x_fit,
            fit_vals[0] * x_fit + fit_vals[1],
            color=series[6 % len(series)],
            linestyle="--",
            lw=sw.overlay + 0.20,
            zorder=5.0,
            label="Tafel Fit",
        )
        if annotate_slope:
            _draw_tafel_slope_badge(
                ax,
                fit_vals[2],
                fit_vals[3],
                preset_obj.ink_color,
                preset_obj.axes_facecolor,
            )

        x_span = max(float(np.ptp(log_j)), 0.5)
        y_span = max(float(np.ptp(eta)), 1e-3)
        ax.set_xlim(
            float(np.min(log_j)) - 0.08 * x_span,
            float(np.max(log_j)) + 0.08 * x_span,
        )
        ax.set_ylim(
            float(np.min(eta)) - 0.14 * y_span,
            float(np.max(eta)) + 0.24 * y_span,
        )
        ax.set_xlabel(
            f"$\\log_{{10}}|j|$ ({_format_echem_label(current_unit)})"
        )
        ax.set_ylabel(f"Overpotential $\\eta$ ({unit.strip()})")
        ax.legend(loc="lower right", frameon=True, fontsize=8.5)
        theme.apply_xkcd_spines(ax, preset=preset)
        if title:
            ax.set_title(title, color=preset_obj.ink_color)
    return ax


def _coerce_battery_traces(
    capacity: Any,
    voltage: Any,
    charge_voltage: Any,
    charge_capacity: Any,
    cycle_labels: Sequence[str] | None,
) -> list[tuple[np.ndarray, np.ndarray, str, str]]:
    """Normalizes single, charge/discharge, or multi-cycle battery curves."""
    traces: list[tuple[np.ndarray, np.ndarray, str, str]] = []
    if voltage is None:
        if not isinstance(capacity, Sequence) or len(capacity) == 0:
            raise exceptions.StructureError(
                "Battery profile requires non-empty (capacity, voltage) data."
            )
        for idx, item in enumerate(capacity):
            if not isinstance(item, Sequence) or len(item) < 2:
                raise exceptions.StructureError(
                    "Multi-cycle items must be (capacity, voltage) tuples."
                )
            q_i, v_i = _as_1d_pair(item[0], item[1], "Battery cycle")
            lbl = (
                cycle_labels[idx]
                if cycle_labels is not None and idx < len(cycle_labels)
                else f"Cycle {idx + 1}"
            )
            traces.append((q_i, v_i, lbl, "-"))
        return traces

    cap_obj = np.asarray(capacity, dtype=object)
    if cap_obj.ndim == 1 and cap_obj.size > 0 and np.ndim(cap_obj[0]) == 1:
        cap_seq = list(capacity)
        vol_seq = list(voltage)
        if len(cap_seq) != len(vol_seq):
            raise exceptions.StructureError(
                "Multi-cycle capacity and voltage sequences must match."
            )
        paired = zip(cap_seq, vol_seq, strict=True)
        for idx, (q_raw, v_raw) in enumerate(paired):
            q_i, v_i = _as_1d_pair(q_raw, v_raw, "Battery cycle")
            lbl = (
                cycle_labels[idx]
                if cycle_labels is not None and idx < len(cycle_labels)
                else f"Cycle {idx + 1}"
            )
            traces.append((q_i, v_i, lbl, "-"))
        return traces

    q_main, v_main = _as_1d_pair(capacity, voltage, "Battery profile")
    if charge_voltage is None:
        lbl0 = cycle_labels[0] if cycle_labels else ""
        return [(q_main, v_main, lbl0, "-")]

    q_chg_raw = capacity if charge_capacity is None else charge_capacity
    q_chg, v_chg = _as_1d_pair(q_chg_raw, charge_voltage, "Battery charge")
    lbl_dis = cycle_labels[0] if cycle_labels else "Discharge"
    lbl_chg = (
        cycle_labels[1]
        if cycle_labels is not None and len(cycle_labels) >= 2
        else "Charge"
    )
    return [(q_main, v_main, lbl_dis, "-"), (q_chg, v_chg, lbl_chg, "--")]


def _apply_plateau_shade(
    ax: maxes.Axes,
    traces: Sequence[tuple[np.ndarray, np.ndarray, str, str]],
    plateau_shade: tuple[float, float] | str | bool | None,
    color: str,
) -> None:
    """Applies optional voltage-band or hysteresis shading on battery plots."""
    if plateau_shade is None or plateau_shade is False:
        return
    if (
        isinstance(plateau_shade, Sequence)
        and not isinstance(plateau_shade, str)
        and len(plateau_shade) == 2
    ):
        v0, v1 = float(plateau_shade[0]), float(plateau_shade[1])
        ax.axhspan(
            min(v0, v1), max(v0, v1), color=color, alpha=0.16, zorder=1.0
        )
        return

    shade_key = None if plateau_shade is True else str(plateau_shade)
    if len(traces) >= 2:
        q_a, v_a = traces[0][0], traces[0][1]
        q_b, v_b = traces[1][0], traces[1][1]
        ord_a, ord_b = np.argsort(q_a), np.argsort(q_b)
        q_lo = max(float(q_a[ord_a[0]]), float(q_b[ord_b[0]]))
        q_hi = min(float(q_a[ord_a[-1]]), float(q_b[ord_b[-1]]))
        if q_hi > q_lo:
            q_grid = np.linspace(q_lo, q_hi, 120)
            va_g = np.interp(q_grid, q_a[ord_a], v_a[ord_a])
            vb_g = np.interp(q_grid, q_b[ord_b], v_b[ord_b])
            patterns.shade_between(
                ax,
                q_grid,
                va_g,
                vb_g,
                color=color,
                shade=shade_key,
                alpha=0.18,
                zorder=1.2,
            )
            return

    v_ref = traces[0][1]
    v_med = float(np.median(v_ref))
    band = max(0.04 * float(np.ptp(v_ref)), 0.04)
    ax.axhspan(v_med - band, v_med + band, color=color, alpha=0.16, zorder=1.0)


def plot_battery_profile(
    capacity: Any,
    voltage: Any = None,
    charge_voltage: Sequence[float] | np.ndarray | None = None,
    charge_capacity: Sequence[float] | np.ndarray | None = None,
    cycle_labels: Sequence[str] | None = None,
    plateau_shade: tuple[float, float] | str | bool | None = None,
    capacity_unit: str = "mAh/g",
    ax: maxes.Axes | None = None,
    palette: str | Sequence[str] = "okabe_ito",
    preset: str | theme.ThemePreset = "classic",
    title: str | None = None,
) -> maxes.Axes:
    """Plots galvanostatic battery charge/discharge voltage-capacity profiles.

    Args:
      capacity: 1D specific capacity array, multi-cycle list of capacity arrays,
        or sequence of ``(capacity, voltage)`` tuples when ``voltage`` is None.
      voltage: 1D voltage array (or sequence of voltage arrays).
      charge_voltage: Optional companion charge voltage curve.
      charge_capacity: Optional companion charge capacity curve (defaults to
        ``capacity`` when ``charge_voltage`` is given).
      cycle_labels: Optional legend labels for cycles or discharge/charge legs.
      plateau_shade: Optional ``(v_min, v_max)`` tuple, ``True``, or shade style
        highlighting the redox plateau / voltage hysteresis.
      capacity_unit: Specific capacity unit string (default ``"mAh/g"``).
      ax: Optional target Matplotlib ``Axes``.
      palette: Series palette name or hex sequence.
      preset: Visual theme preset name or ``ThemePreset``.
      title: Optional plot heading.

    Returns:
      Styled Matplotlib ``Axes`` containing the galvanostatic profile.
    """
    traces = _coerce_battery_traces(
        capacity, voltage, charge_voltage, charge_capacity, cycle_labels
    )
    series = palettes.get_palette(palette)
    preset_obj = theme.get_preset(preset)
    sw = theme.stroke_weights(preset)

    with theme.xkcd_style(preset=preset):
        if ax is None:
            _, ax = plt.subplots(figsize=(6.2, 4.3))
        else:
            theme.style_existing_axes(ax, preset=preset)

        _apply_plateau_shade(
            ax, traces, plateau_shade, series[2 % len(series)]
        )
        has_labels = False
        for idx, (q_arr, v_arr, lbl, ls) in enumerate(traces):
            col = _series_color_for_theme(series, 5 + idx, preset_obj)
            ax.plot(
                q_arr,
                v_arr,
                color=col,
                linestyle=ls,
                lw=sw.primary,
                label=lbl if lbl else None,
                zorder=3.5 + idx * 0.1,
            )
            if lbl:
                has_labels = True

        all_q = np.concatenate([t[0] for t in traces])
        all_v = np.concatenate([t[1] for t in traces])
        q_span = max(float(np.ptp(all_q)), 1.0)
        v_span = max(float(np.ptp(all_v)), 0.2)
        ax.set_xlim(
            min(0.0, float(np.min(all_q))) - 0.03 * q_span,
            float(np.max(all_q)) + 0.06 * q_span,
        )
        ax.set_ylim(
            float(np.min(all_v)) - 0.12 * v_span,
            float(np.max(all_v)) + 0.16 * v_span,
        )
        u_fmt = _format_echem_label(capacity_unit)
        ax.set_xlabel(f"Specific Capacity ({u_fmt})")
        ax.set_ylabel("Voltage (V)")
        if has_labels:
            ax.legend(loc="best", frameon=True, fontsize=8.5)
        theme.apply_xkcd_spines(ax, preset=preset)
        if title:
            ax.set_title(title, color=preset_obj.ink_color)
    return ax


def _smooth_gaussian_1d(vals: np.ndarray, window: int) -> np.ndarray:
    """Applies a symmetric zero-phase reflected-edge Gaussian kernel."""
    win = int(window)
    if win <= 1 or vals.size < 3:
        return np.asarray(vals.copy(), dtype=np.float64)
    if win % 2 == 0:
        win += 1
    win = min(win, vals.size if vals.size % 2 == 1 else vals.size - 1)
    if win <= 1:
        return np.asarray(vals.copy(), dtype=np.float64)
    half = win // 2
    sigma = max(0.30 * float(win - 1), 0.65)
    offsets = np.arange(-half, half + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= float(np.sum(kernel))
    padded = np.pad(vals, pad_width=half, mode="reflect")
    conv = np.convolve(padded, kernel, mode="valid")
    return np.asarray(conv, dtype=np.float64)


def _collapse_monotonic_v(
    v_arr: np.ndarray,
    q_arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sorts by ascending voltage and merges duplicate voltage samples."""
    order = np.argsort(v_arr, kind="mergesort")
    vs = v_arr[order]
    qs = q_arr[order]
    tol = 1e-9 * max(float(np.ptp(vs)), 1.0)
    v_out: list[float] = [float(vs[0])]
    q_out: list[float] = [float(qs[0])]
    counts: list[int] = [1]
    for k in range(1, vs.size):
        vk = float(vs[k])
        qk = float(qs[k])
        if vk - v_out[-1] <= tol:
            n_prev = counts[-1]
            v_out[-1] = (v_out[-1] * n_prev + vk) / (n_prev + 1)
            q_out[-1] = (q_out[-1] * n_prev + qk) / (n_prev + 1)
            counts[-1] = n_prev + 1
        else:
            v_out.append(vk)
            q_out.append(qk)
            counts.append(1)
    if len(v_out) < 3:
        raise exceptions.StructureError(
            "dQ/dV requires at least 3 distinct voltage points."
        )
    return (
        np.asarray(v_out, dtype=np.float64),
        np.asarray(q_out, dtype=np.float64),
    )


def _compute_dqdv_trace(
    voltage: Sequence[float] | np.ndarray,
    capacity: Sequence[float] | np.ndarray,
    smooth_window: int = 5,
    force_sign: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Computes central-difference smoothed dQ/dV on an ascending V grid."""
    v_in, q_in = _as_1d_pair(voltage, capacity, "dQ/dV curve", min_len=3)
    v_asc, q_asc = _collapse_monotonic_v(v_in, q_in)
    raw_dqdv = np.asarray(np.gradient(q_asc, v_asc), dtype=np.float64)
    dqdv: np.ndarray = _smooth_gaussian_1d(raw_dqdv, smooth_window)
    if force_sign > 0:
        dqdv = np.abs(dqdv)
    elif force_sign < 0:
        dqdv = -np.abs(dqdv)
    return v_asc, dqdv


def _find_local_peak_indices(
    v_arr: np.ndarray,
    y_abs: np.ndarray,
    max_peaks: int,
) -> list[int]:
    """Returns indices of top prominent local maxima along ``y_abs``."""
    if max_peaks <= 0 or y_abs.size < 3:
        return []
    cands: list[int] = [
        k
        for k in range(1, y_abs.size - 1)
        if y_abs[k] >= y_abs[k - 1] and y_abs[k] >= y_abs[k + 1]
    ]
    if not cands:
        cands = [int(np.argmax(y_abs))]
    cands.sort(key=lambda idx: float(y_abs[idx]), reverse=True)
    min_sep = 0.045 * max(float(np.ptp(v_arr)), 1e-6)
    chosen: list[int] = []
    for idx in cands:
        if all(abs(float(v_arr[idx] - v_arr[c])) >= min_sep for c in chosen):
            chosen.append(idx)
            if len(chosen) >= max_peaks:
                break
    return chosen


def _annotate_dqdv_peaks(
    ax: maxes.Axes,
    v_arr: np.ndarray,
    dqdv: np.ndarray,
    annotate_peaks: int,
    colors: tuple[str, str, str],
) -> None:
    """Marks and labels redox plateau peaks on a differential capacity curve."""
    trace_col, ink_col, bg_col = colors
    indices = _find_local_peak_indices(v_arr, np.abs(dqdv), annotate_peaks)
    for idx in indices:
        vp = float(v_arr[idx])
        yp = float(dqdv[idx])
        ax.scatter(
            [vp],
            [yp],
            s=44.0,
            facecolor=trace_col,
            edgecolor=ink_col,
            linewidth=1.2,
            zorder=5.5,
        )
        dy_pt = 9 if yp >= 0.0 else -11
        ann = ax.annotate(
            f"{vp:.2f} V",
            xy=(vp, yp),
            xytext=(0, dy_pt),
            textcoords="offset points",
            ha="center",
            va="bottom" if dy_pt > 0 else "top",
            fontsize=8.6,
            color=ink_col,
            bbox=theme.label_halo_bbox(bg_col, pad=0.08, alpha=0.88),
            zorder=6.5,
        )
        patch = ann.get_bbox_patch()
        if patch is not None:
            patch.set_sketch_params(0.0, 0.0, 0.0)


def _draw_dqdv_branch(
    ax: maxes.Axes,
    v_arr: np.ndarray,
    dqdv: np.ndarray,
    label: str | None,
    annotate_peaks: int,
    colors: tuple[str, str, str],
    lw: float,
) -> None:
    """Plots, shades, and peak-labels one differential capacity branch."""
    ax.plot(v_arr, dqdv, color=colors[0], lw=lw, label=label, zorder=3.6)
    patterns.shade_between(
        ax, v_arr, 0.0, dqdv, color=colors[0], shade=None, alpha=0.16
    )
    _annotate_dqdv_peaks(ax, v_arr, dqdv, annotate_peaks, colors)


def plot_dqdv(
    voltage: Sequence[float] | np.ndarray,
    capacity: Sequence[float] | np.ndarray,
    charge_voltage: Sequence[float] | np.ndarray | None = None,
    charge_capacity: Sequence[float] | np.ndarray | None = None,
    smooth_window: int = 5,
    annotate_peaks: int = 2,
    capacity_unit: str = "mAh/g",
    ax: maxes.Axes | None = None,
    palette: str | Sequence[str] = "okabe_ito",
    preset: str | theme.ThemePreset = "classic",
    title: str | None = None,
) -> maxes.Axes:
    """Computes and plots smoothed differential capacity ``dQ/dV`` vs ``V``.

    Args:
      voltage: 1D voltage array (single curve, or discharge curve when
        ``charge_voltage`` is also supplied).
      capacity: 1D capacity array matching ``voltage``.
      charge_voltage: Optional companion charge voltage array.
      charge_capacity: Optional companion charge capacity array (defaults to
        ``capacity`` when ``charge_voltage`` is provided).
      smooth_window: Symmetric Gaussian kernel window length (``<= 1`` disables
        smoothing).
      annotate_peaks: Maximum number of redox peaks per branch to label in ink.
      capacity_unit: Capacity unit string for the ``dQ/dV`` axis label.
      ax: Optional target Matplotlib ``Axes``.
      palette: Series palette name or hex sequence.
      preset: Visual theme preset name or ``ThemePreset``.
      title: Optional plot heading.

    Returns:
      Styled Matplotlib ``Axes`` containing the ``dQ/dV`` spectrum.
    """
    has_pair = charge_voltage is not None
    v_main, dqdv_main = _compute_dqdv_trace(
        voltage,
        capacity,
        smooth_window=smooth_window,
        force_sign=-1 if has_pair else 0,
    )
    chg_pair: tuple[np.ndarray, np.ndarray] | None = None
    if charge_voltage is not None:
        q_chg_raw = capacity if charge_capacity is None else charge_capacity
        chg_pair = _compute_dqdv_trace(
            charge_voltage,
            q_chg_raw,
            smooth_window=smooth_window,
            force_sign=1,
        )

    series = palettes.get_palette(palette)
    preset_obj = theme.get_preset(preset)
    sw = theme.stroke_weights(preset)

    with theme.xkcd_style(preset=preset):
        if ax is None:
            _, ax = plt.subplots(figsize=(6.2, 4.3))
        else:
            theme.style_existing_axes(ax, preset=preset)

        z_ln = ax.axhline(
            0.0,
            color=theme.guide_color(preset_obj),
            linestyle=":",
            lw=sw.guide,
            alpha=0.68,
            zorder=1.5,
        )
        z_ln.set_sketch_params(0.0, 0.0, 0.0)

        if chg_pair is not None:
            _draw_dqdv_branch(
                ax,
                chg_pair[0],
                chg_pair[1],
                "Charge",
                annotate_peaks,
                (
                    series[6 % len(series)],
                    preset_obj.ink_color,
                    preset_obj.axes_facecolor,
                ),
                sw.primary,
            )
        _draw_dqdv_branch(
            ax,
            v_main,
            dqdv_main,
            "Discharge" if has_pair else None,
            annotate_peaks,
            (
                series[5 % len(series)],
                preset_obj.ink_color,
                preset_obj.axes_facecolor,
            ),
            sw.primary,
        )

        all_v = (
            np.concatenate([v_main, chg_pair[0]])
            if chg_pair is not None
            else v_main
        )
        all_y = (
            np.concatenate([dqdv_main, chg_pair[1]])
            if chg_pair is not None
            else dqdv_main
        )
        v_span = max(float(np.ptp(all_v)), 0.2)
        y_hi = max(float(np.max(all_y)), 0.0)
        y_lo = min(float(np.min(all_y)), 0.0)
        y_span = max(y_hi - y_lo, 1.0)
        ax.set_xlim(
            float(np.min(all_v)) - 0.06 * v_span,
            float(np.max(all_v)) + 0.06 * v_span,
        )
        ax.set_ylim(
            y_lo - (0.24 * y_span if y_lo < 0.0 else 0.04 * y_span),
            y_hi + (0.24 * y_span if y_hi > 0.0 else 0.04 * y_span),
        )
        ax.set_xlabel("Voltage (V)")
        u_fmt = _format_echem_label(capacity_unit)
        ax.set_ylabel(f"dQ/dV ({u_fmt}/V)")
        if has_pair:
            ax.legend(loc="best", frameon=True, fontsize=8.5)
        theme.apply_xkcd_spines(ax, preset=preset)
        if title:
            ax.set_title(title, color=preset_obj.ink_color)
    return ax
