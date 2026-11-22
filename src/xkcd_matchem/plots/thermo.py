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
"""Thermodynamic plots: 2D binary hulls, Birch-Murnaghan EOS, and Pourbaix."""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
import math

from matplotlib import axes as maxes
from matplotlib import patches
from matplotlib import pyplot as plt
import numpy as np
from scipy import optimize

from xkcd_matchem.core import exceptions
from xkcd_matchem.core import wulff
from xkcd_matchem.data import elements
from xkcd_matchem.style import palettes
from xkcd_matchem.style import patterns
from xkcd_matchem.style import theme


_EV_PER_A3_TO_GPA = elements.CODATA2022.ev_to_joule * 1e21


def _birch_murnaghan(
    vol: np.ndarray,
    e0: float,
    v0: float,
    b0_ev_a3: float,
    b0_prime: float,
) -> np.ndarray:
    """Evaluates third-order Birch-Murnaghan isothermal equation of state."""
    eta = (v0 / vol) ** (2.0 / 3.0)
    f = eta - 1.0
    return e0 + (9.0 * v0 * b0_ev_a3 / 16.0) * (
        (f**3) * b0_prime + (f**2) * (6.0 - 4.0 * eta)
    )


def _lower_hull_indices(xs: np.ndarray, ys: np.ndarray) -> set[int]:
    """Returns lower 2D convex-hull vertex indices via Andrew's chain."""
    if len(xs) == 0:
        return set()
    order = sorted(range(len(xs)), key=lambda i: (float(xs[i]), float(ys[i])))
    uniq: list[int] = []
    for idx in order:
        if uniq and abs(float(xs[idx]) - float(xs[uniq[-1]])) <= 1e-9:
            if float(ys[idx]) < float(ys[uniq[-1]]):
                uniq[-1] = idx
        else:
            uniq.append(idx)
    if len(uniq) <= 2:
        return set(uniq)
    lower: list[int] = []
    for idx in uniq:
        x2, y2 = float(xs[idx]), float(ys[idx])
        while len(lower) >= 2:
            i0, i1 = lower[-2], lower[-1]
            x0, y0 = float(xs[i0]), float(ys[i0])
            x1, y1 = float(xs[i1]), float(ys[i1])
            cross = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
            if cross <= 1e-12:
                lower.pop()
            else:
                break
        lower.append(idx)
    return set(lower)


def _annotate_phase_labels(
    ax: maxes.Axes,
    xs: np.ndarray,
    ys: np.ndarray,
    labels: Sequence[str],
    lower_indices: set[int],
    ink_col: str,
    bg_color: str,
) -> None:
    """Places stable phase labels below hull nodes and unstable above."""
    y_span = max(float(np.ptp(ys)), 0.25)
    for k, lbl in enumerate(labels):
        if k in lower_indices:
            dx, dy, ha = 0, -12, "center"
        elif -0.18 * y_span <= ys[k] < 0.02 * y_span:
            dx, dy, ha = 8, -2, "left"
        else:
            dx, dy, ha = 0, 8, "center"
        ann = ax.annotate(
            lbl,
            xy=(xs[k], ys[k]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            fontsize=9.5,
            color=ink_col,
            bbox=theme.label_halo_bbox(bg_color, pad=0.07, alpha=0.84),
        )
        patch = ann.get_bbox_patch()
        if patch is not None:
            patch.set_sketch_params(0.0, 0.0, 0.0)


def plot_phase_diagram_2d(
    compositions: Sequence[float] | np.ndarray,
    energies: Sequence[float] | np.ndarray,
    labels: Sequence[str] | None = None,
    ax: maxes.Axes | None = None,
    xlabel: str = "Mole Fraction $x_B$",
    palette: str | Sequence[str] = "okabe_ito",
    preset: str | theme.ThemePreset = "classic",
    title: str | None = None,
) -> maxes.Axes:
    """Plots a binary A-B convex hull formation-energy phase diagram."""
    xs = np.asarray(compositions, dtype=np.float64)
    ys = np.asarray(energies, dtype=np.float64)
    if xs.ndim != 1 or ys.ndim != 1 or xs.shape != ys.shape or xs.size == 0:
        raise exceptions.StructureError(
            "Binary phase diagram requires non-empty matching 1D arrays."
        )
    lower_indices = _lower_hull_indices(xs, ys)
    hull_sorted = sorted(lower_indices, key=lambda idx: float(xs[idx]))
    hull_x = xs[hull_sorted]
    hull_y = ys[hull_sorted]
    series = palettes.get_palette(palette)
    hull_col = series[5 % len(series)]
    preset_obj = theme.get_preset(preset)
    sw = theme.stroke_weights(preset)

    with theme.xkcd_style(preset=preset):
        if ax is None:
            _, ax = plt.subplots(figsize=(6.2, 4.4))
        else:
            theme.style_existing_axes(ax, preset=preset)
        ax.plot(
            hull_x, hull_y, color=hull_col, lw=sw.primary, label="Convex Hull"
        )
        ax.scatter(
            hull_x,
            hull_y,
            s=80,
            facecolor=series[3 % len(series)],
            edgecolor=preset_obj.ink_color,
            linewidth=1.35,
            zorder=6,
            label="Stable",
        )
        unstable = [k for k in range(len(xs)) if k not in lower_indices]
        if unstable:
            above_col = series[6 % len(series)]
            ax.scatter(
                xs[unstable],
                ys[unstable],
                s=70,
                marker="D",
                facecolor=preset_obj.axes_facecolor,
                edgecolor=above_col,
                linewidth=1.35,
                zorder=5,
                label="Above Hull",
            )
            for k in unstable:
                y_hull = float(np.interp(xs[k], hull_x, hull_y))
                if abs(float(ys[k]) - y_hull) > 1e-9:
                    drop = ax.plot(
                        [xs[k], xs[k]],
                        [y_hull, ys[k]],
                        color=above_col,
                        linestyle=":",
                        lw=sw.guide,
                    )[0]
                    drop.set_sketch_params(0.0, 0.0, 0.0)
        if labels:
            _annotate_phase_labels(
                ax,
                xs,
                ys,
                labels,
                lower_indices,
                preset_obj.ink_color,
                preset_obj.axes_facecolor,
            )
        z_line = ax.axhline(
            0.0, color=palettes.MUTED_GRAY, linestyle="--", lw=sw.guide
        )
        z_line.set_sketch_params(0.0, 0.0, 0.0)
        y_span = max(float(np.ptp(ys)), 0.25)
        x_lo, x_hi = float(np.min(xs)), float(np.max(xs))
        if x_lo >= -1e-9 and x_hi <= 1.0 + 1e-9:
            ax.set_xlim(-0.06, 1.06)
        else:
            x_pad = 0.06 * max(x_hi - x_lo, 1.0)
            ax.set_xlim(x_lo - x_pad, x_hi + x_pad)
        ax.set_ylim(
            float(np.min(ys)) - 0.18 * y_span,
            float(np.max(ys)) + 0.46 * y_span,
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Formation Energy (eV/atom)")
        ax.legend(loc="upper center", ncol=2, frameon=True, fontsize=8.5)
        theme.apply_xkcd_spines(ax, preset=preset)
        if title:
            ax.set_title(title)
    return ax


def _fit_birch_murnaghan(vs: np.ndarray, es: np.ndarray) -> np.ndarray:
    """Fits 3rd-order Birch-Murnaghan EOS with quadratic initialization."""
    v_min, v_max = float(np.min(vs)), float(np.max(vs))
    min_idx = int(np.argmin(es))
    v0_init = float(vs[min_idx])
    e0_init = float(es[min_idx])
    b0_init = 0.6
    if len(vs) >= 3:
        coeffs = np.polyfit(vs, es, 2)
        if coeffs[0] > 1e-12:
            v_q = -coeffs[1] / (2.0 * coeffs[0])
            v0_init = float(np.clip(v_q, 0.8 * v_min, 1.2 * v_max))
            b0_init = float(np.clip(2.0 * coeffs[0] * v0_init, 1e-3, 10.0))
            e0_init = float(np.polyval(coeffs, v0_init))
    lower = [-np.inf, 0.5 * v_min, 1e-5, -5.0]
    upper = [np.inf, 2.0 * v_max, 50.0, 25.0]
    p0 = [
        e0_init,
        float(np.clip(v0_init, 0.55 * v_min, 1.9 * v_max)),
        float(np.clip(b0_init, 1e-4, 20.0)),
        4.0,
    ]
    popt, _ = optimize.curve_fit(
        _birch_murnaghan,
        vs,
        es,
        p0=p0,
        bounds=(lower, upper),
        maxfev=10000,
    )
    return np.asarray(popt, dtype=np.float64)


_EOS_YLABELS: dict[str, str] = {
    "cell": "Energy (eV/cell)",
    "formula_unit": "Energy (eV/f.u.)",
    "atom": "Energy (eV/atom)",
}


def _format_eos_rmse(rmse: float) -> str:
    """Formats Birch-Murnaghan RMSE concisely in eV."""
    if rmse == 0.0 or rmse >= 1e-2:
        return f"{rmse:.3f} eV"
    if rmse >= 1e-3:
        return f"{rmse:.4f} eV"
    return f"{rmse:.2e} eV"


def plot_eos(
    volumes: Sequence[float] | np.ndarray,
    energies: Sequence[float] | np.ndarray,
    ax: maxes.Axes | None = None,
    fit_birch_murnaghan: bool = True,
    palette: str | Sequence[str] = "okabe_ito",
    preset: str | theme.ThemePreset = "classic",
    title: str | None = None,
    energy_basis: str = "cell",
) -> maxes.Axes:
    """Plots energy vs volume with an optional 3rd-order Birch-Murnaghan fit."""
    canon_basis = wulff.normalize_eos_basis(energy_basis)
    vs = np.asarray(volumes, dtype=np.float64)
    es = np.asarray(energies, dtype=np.float64)
    if vs.ndim != 1 or es.ndim != 1 or vs.shape != es.shape or vs.size == 0:
        raise exceptions.StructureError(
            "EOS plot requires non-empty matching 1D volume & energy arrays."
        )
    series = palettes.get_palette(palette)
    pts_col = series[1 % len(series)]
    fit_col = series[5 % len(series)]
    preset_obj = theme.get_preset(preset)
    sw = theme.stroke_weights(preset)
    with theme.xkcd_style(preset=preset):
        if ax is None:
            _, ax = plt.subplots(figsize=(6.0, 4.3))
        else:
            theme.style_existing_axes(ax, preset=preset)
        ax.scatter(
            vs,
            es,
            s=68,
            facecolor=pts_col,
            edgecolor=preset_obj.ink_color,
            linewidth=1.35,
            zorder=5,
            label="DFT Points",
        )
        if fit_birch_murnaghan and len(vs) >= 4:
            fit_res = wulff.fit_birch_murnaghan(
                vs,
                es,
                energy_basis=canon_basis,
                allow_extrapolation=True,
            )
            v_fine = np.linspace(float(np.min(vs)), float(np.max(vs)), 160)
            e_fine = wulff.birch_murnaghan_energy(
                v_fine,
                fit_res.e0,
                fit_res.v0,
                fit_res.b0_ev_a3,
                fit_res.b0_prime,
            )
            ax.plot(
                v_fine,
                e_fine,
                color=fit_col,
                lw=sw.overlay + 0.25,
                zorder=3,
                label="Birch-Murnaghan",
            )
            b0_line = (
                f"B0 = {fit_res.b0_gpa:.1f} GPa \u00b1 {fit_res.b0_gpa_err:.1f}"
                if fit_res.b0_gpa_err is not None
                else f"B0 = {fit_res.b0_gpa:.1f} GPa"
            )
            summary = (
                f"V0 = {fit_res.v0:.2f} A^3\n"
                f"{b0_line}\n"
                f"B0' = {fit_res.b0_prime:.2f}\n"
                f"RMSE = {_format_eos_rmse(fit_res.rmse_ev)}"
            )
            box_txt = ax.text(
                0.50,
                0.57,
                summary,
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=8.6,
                color=preset_obj.ink_color,
                bbox={
                    "boxstyle": "round,pad=0.3",
                    "facecolor": preset_obj.axes_facecolor,
                    "edgecolor": palettes.MUTED_GRAY,
                    "linewidth": 1.1,
                },
            )
            patch = box_txt.get_bbox_patch()
            if patch is not None:
                patch.set_sketch_params(0.0, 0.0, 0.0)
        e_span = max(float(np.ptp(es)), 1e-3)
        ax.set_ylim(
            float(np.min(es)) - 0.14 * e_span,
            float(np.max(es)) + 0.38 * e_span,
        )
        ax.set_xlabel("Cell Volume ($\\mathrm{\\AA}^3$)")
        ax.set_ylabel(_EOS_YLABELS[canon_basis])
        ax.legend(loc="upper center", ncol=2, frameon=True, fontsize=8.5)
        theme.apply_xkcd_spines(ax, preset=preset)
        if title:
            ax.set_title(title)
    return ax


def _polygon_centroid(pts: np.ndarray) -> np.ndarray:
    """Computes the exact shoelace area centroid of a 2D polygon."""
    if len(pts) < 3:
        return np.asarray(np.mean(pts, axis=0), dtype=np.float64)
    x, y = pts[:, 0], pts[:, 1]
    x_nxt, y_nxt = np.roll(x, -1), np.roll(y, -1)
    cross = x * y_nxt - x_nxt * y
    area2 = float(np.sum(cross))
    if abs(area2) < 1e-12:
        return np.asarray(np.mean(pts, axis=0), dtype=np.float64)
    cx = float(np.sum((x + x_nxt) * cross)) / (3.0 * area2)
    cy = float(np.sum((y + y_nxt) * cross)) / (3.0 * area2)
    return np.array([cx, cy], dtype=np.float64)


def plot_pourbaix(
    regions: Sequence[tuple[str, Sequence[tuple[float, float]]]],
    ax: maxes.Axes | None = None,
    ph_range: tuple[float, float] = (0.0, 14.0),
    potential_range: tuple[float, float] = (-1.2, 2.0),
    show_water_window: bool = True,
    shade: str | Sequence[str | None] | None = None,
    background_grid: str | None = None,
    palette: str | Sequence[str] = "okabe_ito",
    preset: str | theme.ThemePreset = "classic",
    title: str | None = None,
) -> maxes.Axes:
    """Plots a schematic or DFT Pourbaix pH vs E_SHE stability diagram."""
    series = palettes.get_palette(palette)
    preset_obj = theme.get_preset(preset)
    ink_col = preset_obj.ink_color
    sw = theme.stroke_weights(preset)
    with theme.xkcd_style(preset=preset):
        if ax is None:
            _, ax = plt.subplots(figsize=(6.2, 4.8))
        else:
            theme.style_existing_axes(ax, preset=preset)
        if background_grid:
            patterns.apply_background_grid(ax, style=background_grid)
        for idx, (label, vertices) in enumerate(regions):
            pts = np.asarray(vertices, dtype=np.float64)
            col = series[(idx + 1) % len(series)]
            hatch = patterns.resolve_hatch(shade, index=idx)
            poly = patches.Polygon(
                pts,
                closed=True,
                facecolor=col,
                edgecolor=ink_col,
                linewidth=round(0.80 * sw.primary, 2),
                alpha=0.28,
                hatch=hatch,
                zorder=2,
            )
            if hatch:
                poly.set_sketch_params(0.0, 0.0, 0.0)
            ax.add_patch(poly)
            ctr = _polygon_centroid(pts)
            theme.place_halo_text(
                ax,
                float(ctr[0]),
                float(ctr[1]),
                label,
                color=ink_col,
                bg_color=preset_obj.axes_facecolor,
                fontsize=10.5,
                pad=0.10,
                alpha=0.68 if hatch else 0.0,
                zorder=4,
            )
        p_span = max(potential_range[1] - potential_range[0], 0.5)
        y_hi = potential_range[1] + (
            0.24 * p_span if show_water_window else 0.0
        )
        if show_water_window:
            phs = np.linspace(ph_range[0], ph_range[1], 50)
            ax.plot(
                phs,
                -0.05916 * phs,
                color=series[5 % len(series)],
                linestyle="--",
                lw=sw.overlay,
                label="H2/H+",
            )
            ax.plot(
                phs,
                1.229 - 0.05916 * phs,
                color=series[6 % len(series)],
                linestyle="--",
                lw=sw.overlay,
                label="O2/H2O",
            )
            ax.legend(loc="upper center", ncol=2, frameon=True, fontsize=8.5)
        ax.set_xlim(ph_range)
        ax.set_ylim(potential_range[0], y_hi)
        ax.set_xlabel("pH")
        ax.set_ylabel("Potential vs SHE (V)")
        theme.apply_xkcd_spines(ax, preset=preset)
        if title:
            ax.set_title(title)
    return ax


def _resolve_volcano_legs(
    xs: np.ndarray,
    ys: np.ndarray,
    left_leg: tuple[float, float] | None,
    right_leg: tuple[float, float] | None,
    fit_legs: bool,
    invert_y: bool,
) -> tuple[tuple[float, float] | None, tuple[float, float] | None, float]:
    """Computes left/right linear scaling legs and volcano apex coordinate."""
    opt_idx = int(np.argmin(ys) if invert_y else np.argmax(ys))
    x_opt = float(xs[opt_idx])
    l_leg = left_leg
    r_leg = right_leg
    if fit_legs and l_leg is None:
        l_mask = xs <= x_opt + 1e-12
        if int(np.sum(l_mask)) >= 2 and float(np.ptp(xs[l_mask])) > 1e-9:
            c_l = np.polyfit(xs[l_mask], ys[l_mask], 1)
            l_leg = (float(c_l[0]), float(c_l[1]))
    if fit_legs and r_leg is None:
        r_mask = xs >= x_opt - 1e-12
        if int(np.sum(r_mask)) >= 2 and float(np.ptp(xs[r_mask])) > 1e-9:
            c_r = np.polyfit(xs[r_mask], ys[r_mask], 1)
            r_leg = (float(c_r[0]), float(c_r[1]))

    x_apex = x_opt
    if l_leg is not None and r_leg is not None:
        denom = l_leg[0] - r_leg[0]
        if abs(denom) > 1e-9:
            cand = (r_leg[1] - l_leg[1]) / denom
            span = max(float(np.ptp(xs)), 0.5)
            lo_bound = float(np.min(xs)) - 0.35 * span
            hi_bound = float(np.max(xs)) + 0.35 * span
            if lo_bound <= cand <= hi_bound:
                x_apex = cand
    return l_leg, r_leg, x_apex


def _volcano_label_shift(
    xi: float,
    yi: float,
    l_leg: tuple[float, float] | None,
    r_leg: tuple[float, float] | None,
    x_apex: float,
    x_span: float,
    y_span: float,
    invert_y: bool,
) -> tuple[int, int]:
    """Computes outward normal label shift (pt) clear of scaling legs."""
    sign_up = -1.0 if invert_y else 1.0
    if abs(xi - x_apex) <= 0.08 * x_span:
        return 0, int(round(12.0 * sign_up))
    leg = l_leg if (xi < x_apex and l_leg is not None) else r_leg
    if leg is None:
        return 0, int(round(11.0 * sign_up))
    y_line = leg[0] * xi + leg[1]
    above = (
        yi <= y_line + 0.02 * y_span
        if invert_y
        else yi >= y_line - 0.02 * y_span
    )
    side = 1.0 if above else -1.0
    nx, ny = -leg[0] * (x_span / max(y_span, 1e-6)), 1.0
    n_len = max(math.hypot(nx, ny), 1e-6)
    return (
        int(round(12.5 * (nx / n_len) * side)),
        int(round(12.5 * (ny / n_len) * side * sign_up)),
    )


def _annotate_volcano_labels(
    ax: maxes.Axes,
    xs: np.ndarray,
    ys: np.ndarray,
    labels: Sequence[str] | None,
    geom: tuple[
        tuple[float, float] | None,
        tuple[float, float] | None,
        float,
        float,
        float,
        bool,
    ],
    ink_col: str,
    bg_color: str,
) -> None:
    """Places ink-first volcano catalyst labels away from scaling legs."""
    if labels is None:
        return
    l_leg, r_leg, x_apex, x_span, y_span, invert_y = geom
    for idx, lbl in enumerate(labels):
        xi, yi = float(xs[idx]), float(ys[idx])
        dx_pt, dy_pt = _volcano_label_shift(
            xi, yi, l_leg, r_leg, x_apex, x_span, y_span, invert_y
        )
        ann = ax.annotate(
            lbl,
            xy=(xi, yi),
            xytext=(dx_pt, dy_pt),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=9.0,
            color=ink_col,
            bbox=theme.label_halo_bbox(bg_color, pad=0.07, alpha=0.86),
            zorder=6.0,
        )
        patch = ann.get_bbox_patch()
        if patch is not None:
            patch.set_sketch_params(0.0, 0.0, 0.0)


def _draw_volcano_legs(
    ax: maxes.Axes,
    l_leg: tuple[float, float] | None,
    r_leg: tuple[float, float] | None,
    x_bounds: tuple[float, float, float],
    series: Sequence[str],
    sw: theme.StrokeHierarchy,
) -> None:
    """Draws subordinate strong- and weak-binding scaling legs."""
    x_lo, x_apex, x_hi = x_bounds
    if l_leg is not None and x_lo < x_apex:
        x_l = np.linspace(x_lo, x_apex, 40)
        ax.plot(
            x_l,
            l_leg[0] * x_l + l_leg[1],
            color=series[5 % len(series)],
            lw=sw.overlay + 0.12,
            alpha=0.88,
            zorder=2.5,
            label="Strong-binding leg",
        )
    if r_leg is not None and x_apex < x_hi:
        x_r = np.linspace(x_apex, x_hi, 40)
        ax.plot(
            x_r,
            r_leg[0] * x_r + r_leg[1],
            color=series[6 % len(series)],
            lw=sw.overlay + 0.12,
            alpha=0.88,
            zorder=2.5,
            label="Weak-binding leg",
        )


def plot_volcano(
    descriptors: Sequence[float] | np.ndarray,
    activities: Sequence[float] | np.ndarray,
    *,
    labels: Sequence[str] | None = None,
    left_leg: tuple[float, float] | None = None,
    right_leg: tuple[float, float] | None = None,
    fit_legs: bool = True,
    peak_shade: bool = True,
    xlabel: str = "Descriptor $\\Delta G_{\\mathrm{ads}}$ (eV)",
    ylabel: str = "Activity / Limiting Potential (V)",
    invert_y: bool = False,
    palette: str | Sequence[str] = "okabe_ito",
    preset: str | theme.ThemePreset = "classic",
    ax: maxes.Axes | None = None,
    title: str | None = None,
) -> maxes.Axes:
    """Plots a Sabatier catalysis volcano relation in xkcd style.

    Args:
      descriptors: Catalytic descriptor values (e.g. adsorption free energy).
      activities: Corresponding activity or limiting-potential metrics.
      labels: Optional point labels (e.g. catalyst metal symbols).
      left_leg: Optional explicit ``(slope, intercept)`` strong-binding leg.
      right_leg: Optional explicit ``(slope, intercept)`` weak-binding leg.
      fit_legs: Automatically fit linear scaling legs around the optimum.
      peak_shade: Highlight the Sabatier optimum region with a soft band.
      xlabel: Horizontal axis label.
      ylabel: Vertical axis label.
      invert_y: Invert vertical axis (e.g. for overpotential plots).
      palette: Series palette name or hex sequence.
      preset: Visual theme preset name or ``ThemePreset``.
      ax: Optional target Matplotlib ``Axes``.
      title: Optional heading.

    Returns:
      Matplotlib ``Axes`` containing the Sabatier volcano plot.

    Raises:
      StructureError: If input arrays mismatch or have fewer than 2 points.
    """
    xs = np.asarray(descriptors, dtype=np.float64)
    ys = np.asarray(activities, dtype=np.float64)
    if xs.ndim != 1 or ys.ndim != 1 or xs.shape != ys.shape or xs.size < 2:
        raise exceptions.StructureError(
            "Volcano plot requires matching 1D arrays with at least 2 points."
        )
    if labels is not None and len(labels) != xs.size:
        raise exceptions.StructureError(
            "Volcano labels length must match descriptors."
        )

    l_leg, r_leg, x_apex = _resolve_volcano_legs(
        xs, ys, left_leg, right_leg, fit_legs, invert_y
    )
    series = palettes.get_palette(palette)
    preset_obj = theme.get_preset(preset)
    sw = theme.stroke_weights(preset)
    x_span = max(float(np.ptp(xs)), 0.4)
    y_span = max(float(np.ptp(ys)), 0.4)
    x_lo = float(np.min(xs)) - 0.12 * x_span
    x_hi = float(np.max(xs)) + 0.12 * x_span

    with theme.xkcd_style(preset=preset):
        if ax is None:
            _, ax = plt.subplots(figsize=(6.4, 4.4))
        else:
            theme.style_existing_axes(ax, preset=preset)
        if peak_shade:
            ax.axvspan(
                x_apex - 0.08 * x_span,
                x_apex + 0.08 * x_span,
                color=series[2 % len(series)],
                alpha=0.14,
                zorder=1.0,
            )
        _draw_volcano_legs(ax, l_leg, r_leg, (x_lo, x_apex, x_hi), series, sw)
        ax.scatter(
            xs,
            ys,
            s=72.0,
            facecolor=series[1 % len(series)],
            edgecolor=preset_obj.ink_color,
            linewidth=1.35,
            zorder=5.0,
        )
        _annotate_volcano_labels(
            ax,
            xs,
            ys,
            labels,
            (l_leg, r_leg, x_apex, x_span, y_span, invert_y),
            preset_obj.ink_color,
            preset_obj.axes_facecolor,
        )
        y_lo = float(np.min(ys)) - 0.18 * y_span
        y_hi = float(np.max(ys)) + 0.18 * y_span
        ax.set_ylim(y_lo, y_hi)
        if invert_y:
            ax.invert_yaxis()
        ax.set_xlim(x_lo, x_hi)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if l_leg is not None or r_leg is not None:
            ax.legend(loc="best", frameon=True, fontsize=8.5)
        theme.apply_xkcd_spines(ax, preset=preset)
        if title:
            ax.set_title(title)
    return ax


def _fmt_charge(q: int) -> str:
    """Formats an integer defect charge state concisely."""
    return "0" if q == 0 else f"{q:+d}"


def _defect_breakpoints(
    cand: Sequence[tuple[int, float]],
    e_lo: float,
    e_hi: float,
) -> list[float]:
    """Collects sorted unique pairwise transition points inside [e_lo, e_hi]."""
    bps: list[float] = [e_lo, e_hi]
    for i_idx, (q_a, e_a) in enumerate(cand):
        for q_b, e_b in cand[i_idx + 1 :]:
            if q_a != q_b:
                x_int = -(e_a - e_b) / float(q_a - q_b)
                if e_lo < x_int < e_hi:
                    bps.append(x_int)
    bps.sort()
    unique_bps: list[float] = []
    for bp in bps:
        if not unique_bps or bp - unique_bps[-1] > 1e-8:
            unique_bps.append(bp)
    return unique_bps


def _lowest_charge_state(
    cand: Sequence[tuple[int, float]],
    e_fermi: float,
) -> tuple[int, float]:
    """Returns the lowest-energy ``(q, E0)`` charge state at ``e_fermi``."""
    best_key = (float("inf"), 0)
    win_q, win_e = cand[0]
    for q_c, e_c in cand:
        score = (e_c + q_c * e_fermi, -q_c)
        if score < best_key:
            best_key = score
            win_q, win_e = q_c, e_c
    return win_q, win_e


def _lower_defect_envelope(
    states: Sequence[tuple[int, float]],
    e_lo: float,
    e_hi: float,
) -> tuple[list[tuple[float, float]], list[tuple[float, float, int, int]]]:
    """Computes exact piecewise-linear E^f(E_F) and transitions."""
    if not states:
        raise exceptions.StructureError(
            "Each defect must have at least one (charge, E_f) state."
        )
    cand = [(int(q), float(e0)) for q, e0 in states]
    unique_bps = _defect_breakpoints(cand, e_lo, e_hi)

    segs: list[tuple[float, float, int, float]] = []
    for k in range(len(unique_bps) - 1):
        x0, x1 = unique_bps[k], unique_bps[k + 1]
        win_q, win_e = _lowest_charge_state(cand, 0.5 * (x0 + x1))
        if segs and segs[-1][2] == win_q and abs(segs[-1][3] - win_e) < 1e-8:
            segs[-1] = (segs[-1][0], x1, win_q, win_e)
        else:
            segs.append((x0, x1, win_q, win_e))

    curve: list[tuple[float, float]] = [
        (segs[0][0], segs[0][3] + segs[0][2] * segs[0][0])
    ]
    transitions: list[tuple[float, float, int, int]] = []
    for k, seg in enumerate(segs):
        x_end, q_val, e0_val = seg[1], seg[2], seg[3]
        y_end = e0_val + q_val * x_end
        curve.append((x_end, y_end))
        if k + 1 < len(segs):
            transitions.append((x_end, y_end, q_val, segs[k + 1][2]))
    return curve, transitions


def _format_defect_label(name: str) -> str:
    """Formats simple ``X_Y`` defect names using mathtext subscripts."""
    if "$" in name or "_" not in name:
        return name
    base, sub = name.split("_", 1)
    if base and sub and base.isalnum() and sub.isalnum():
        return f"$\\mathrm{{{base}}}_{{\\mathrm{{{sub}}}}}$"
    return name


def _defect_bisector_offset(
    q1: int,
    q2: int,
    x_span: float,
    y_span: float,
) -> tuple[int, int]:
    """Computes outward upward bisector offset (pt) clear of both rays."""
    sx = max(x_span, 1e-6)
    sy = max(y_span, 1e-6)
    a1 = math.hypot(1.0 / sx, float(q1) / sy)
    a2 = math.hypot(1.0 / sx, float(q2) / sy)
    n1x, n1y = -(float(q1) / sy) / a1, (1.0 / sx) / a1
    n2x, n2y = -(float(q2) / sy) / a2, (1.0 / sx) / a2
    bx, by = n1x + n2x, n1y + n2y
    bn = max(math.hypot(bx, by), 1e-6)
    return int(round(13.0 * bx / bn)), int(round(12.0 * by / bn))


def _plot_single_defect(
    ax: maxes.Axes,
    d_name: str,
    curve: list[tuple[float, float]],
    trans: list[tuple[float, float, int, int]],
    col: str,
    style_cfg: tuple[str, str, theme.StrokeHierarchy],
    spans: tuple[float, float],
    show_transitions: bool,
) -> None:
    """Plots one defect envelope curve and ink-first kink labels."""
    ink_col, bg_color, sw = style_cfg
    ax.plot(
        [pt[0] for pt in curve],
        [pt[1] for pt in curve],
        color=col,
        lw=sw.primary,
        label=_format_defect_label(d_name),
        zorder=4.0,
    )
    if show_transitions and trans:
        ax.scatter(
            [t[0] for t in trans],
            [t[1] for t in trans],
            s=46.0,
            facecolor=col,
            edgecolor=ink_col,
            linewidth=1.15,
            zorder=6.0,
        )
        for xf, yf, q1, q2 in trans:
            dx_pt, dy_pt = _defect_bisector_offset(q1, q2, spans[0], spans[1])
            lbl = f"({_fmt_charge(q1)}/{_fmt_charge(q2)})"
            ann = ax.annotate(
                lbl,
                xy=(xf, yf),
                xytext=(dx_pt, dy_pt),
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=8.2,
                color=ink_col,
                bbox=theme.label_halo_bbox(bg_color, pad=0.07, alpha=0.86),
                zorder=7.0,
            )
            patch = ann.get_bbox_patch()
            if patch is not None:
                patch.set_sketch_params(0.0, 0.0, 0.0)


def _draw_defect_band_guides(
    ax: maxes.Axes,
    e_lo: float,
    e_hi: float,
    gap: float,
    ink_col: str,
    guide_lw: float,
) -> None:
    """Draws VBM/CBM straight guides and valence/conduction shading bands."""
    if e_lo < 0.0:
        ax.axvspan(e_lo, 0.0, color=palettes.MUTED_GRAY, alpha=0.14, zorder=1.0)
    if e_hi > gap:
        ax.axvspan(gap, e_hi, color=palettes.MUTED_GRAY, alpha=0.14, zorder=1.0)
    for edge_x in (0.0, gap):
        b_line = ax.axvline(
            edge_x,
            color=ink_col,
            linestyle=":",
            lw=guide_lw,
            alpha=0.55,
        )
        b_line.set_sketch_params(0.0, 0.0, 0.0)


def plot_defect_formation(
    defects: Mapping[str, Sequence[tuple[int, float]]],
    *,
    band_gap: float = 1.12,
    fermi_range: tuple[float, float] | None = None,
    show_transitions: bool = True,
    xlabel: str = "Fermi Level $E_F - E_{\\mathrm{VBM}}$ (eV)",
    ylabel: str = "Formation Energy $E^f$ (eV)",
    palette: str | Sequence[str] = "okabe_ito",
    preset: str | theme.ThemePreset = "classic",
    ax: maxes.Axes | None = None,
    title: str | None = None,
) -> maxes.Axes:
    """Plots defect formation energies vs. Fermi level in xkcd style.

    Evaluates the exact lower convex envelope ``min(E0 + q * E_F)`` across all
    charge states ``(q, E0)`` supplied per defect and labels thermodynamic
    charge-transition kinks ``(q1/q2)``.

    Args:
      defects: Mapping ``{defect_name: [(charge, Ef_at_VBM), ...]}``.
      band_gap: Host semiconductor band gap in eV.
      fermi_range: Optional ``(min_EF, max_EF)`` display window in eV.
      show_transitions: Mark and label active charge-transition levels.
      xlabel: Horizontal axis label.
      ylabel: Vertical axis label.
      palette: Series palette name or hex color list.
      preset: Visual theme preset name or ``ThemePreset``.
      ax: Optional target Matplotlib ``Axes``.
      title: Optional heading.

    Returns:
      Matplotlib ``Axes`` containing the defect formation energy diagram.

    Raises:
      StructureError: If ``defects`` is empty or ``band_gap <= 0``.
    """
    if not defects:
        raise exceptions.StructureError(
            "plot_defect_formation requires at least one defect entry."
        )
    gap = float(band_gap)
    if gap <= 0.0:
        raise exceptions.StructureError("band_gap must be strictly positive.")
    e_lo, e_hi = (
        (-0.04 * gap, 1.04 * gap)
        if fermi_range is None
        else (float(fermi_range[0]), float(fermi_range[1]))
    )
    if e_hi <= e_lo:
        raise exceptions.StructureError("fermi_range must satisfy max > min.")

    envelopes = [
        (d_name, _lower_defect_envelope(states, e_lo, e_hi))
        for d_name, states in defects.items()
    ]
    all_ys = [pt[1] for _, (curve, _) in envelopes for pt in curve]
    y_min, y_max = float(min(all_ys)), float(max(all_ys))
    y_span = max(y_max - y_min, 0.4)
    series = palettes.get_palette(palette)
    preset_obj = theme.get_preset(preset)
    sw = theme.stroke_weights(preset)
    with theme.xkcd_style(preset=preset):
        if ax is None:
            _, ax = plt.subplots(figsize=(6.4, 4.5))
        else:
            theme.style_existing_axes(ax, preset=preset)
        _draw_defect_band_guides(
            ax, e_lo, e_hi, gap, preset_obj.ink_color, sw.guide
        )
        for idx, (d_name, (curve, trans)) in enumerate(envelopes):
            _plot_single_defect(
                ax,
                d_name,
                curve,
                trans,
                series[(idx + 1) % len(series)],
                (preset_obj.ink_color, preset_obj.axes_facecolor, sw),
                (max(e_hi - e_lo, 0.2), y_span),
                show_transitions,
            )
        ax.set_xlim(e_lo, e_hi)
        ax.set_ylim(y_min - 0.08 * y_span, y_max + 0.16 * y_span)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(loc="best", frameon=True, fontsize=8.8)
        theme.apply_xkcd_spines(ax, preset=preset)
        if title:
            ax.set_title(title)
    return ax
