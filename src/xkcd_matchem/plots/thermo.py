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

from collections.abc import Sequence

from matplotlib import axes as maxes
from matplotlib import patches
from matplotlib import pyplot as plt
import numpy as np
from scipy import optimize

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
        ax.annotate(
            lbl,
            xy=(xs[k], ys[k]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            fontsize=9.5,
        )


def plot_phase_diagram_2d(
    compositions: Sequence[float] | np.ndarray,
    energies: Sequence[float] | np.ndarray,
    labels: Sequence[str] | None = None,
    ax: maxes.Axes | None = None,
    xlabel: str = "Mole Fraction x_B",
    palette: str | Sequence[str] = "okabe_ito",
    preset: str | theme.ThemePreset = "classic",
    title: str | None = None,
) -> maxes.Axes:
    """Plots a binary A-B convex hull formation-energy phase diagram."""
    xs = np.asarray(compositions, dtype=np.float64)
    ys = np.asarray(energies, dtype=np.float64)
    lower_indices = _lower_hull_indices(xs, ys)
    hull_sorted = sorted(lower_indices, key=lambda idx: float(xs[idx]))
    hull_x = xs[hull_sorted]
    hull_y = ys[hull_sorted]
    series = palettes.get_palette(palette)
    hull_col = series[5 % len(series)]
    stable_col = series[3 % len(series)]
    above_col = series[6 % len(series)]
    preset_obj = theme.get_preset(preset)

    with theme.xkcd_style(preset=preset):
        if ax is None:
            _, ax = plt.subplots(figsize=(6.2, 4.4))
        else:
            theme.style_existing_axes(ax, preset=preset)
        ax.plot(hull_x, hull_y, color=hull_col, lw=2.3, label="Convex Hull")
        ax.scatter(
            hull_x,
            hull_y,
            s=85,
            facecolor=stable_col,
            edgecolor=preset_obj.ink_color,
            linewidth=1.5,
            zorder=6,
            label="Stable",
        )
        unstable = [k for k in range(len(xs)) if k not in lower_indices]
        if unstable:
            ax.scatter(
                xs[unstable],
                ys[unstable],
                s=75,
                marker="D",
                facecolor=preset_obj.axes_facecolor,
                edgecolor=above_col,
                linewidth=1.6,
                zorder=5,
                label="Above Hull",
            )
            for k in unstable:
                y_hull = float(np.interp(xs[k], hull_x, hull_y))
                if abs(float(ys[k]) - y_hull) > 1e-9:
                    ax.plot(
                        [xs[k], xs[k]],
                        [y_hull, ys[k]],
                        color=above_col,
                        linestyle=":",
                        lw=1.3,
                    )
        if labels:
            _annotate_phase_labels(ax, xs, ys, labels, lower_indices)
        ax.axhline(0.0, color=palettes.MUTED_GRAY, linestyle="--", lw=1.1)
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


def plot_eos(
    volumes: Sequence[float] | np.ndarray,
    energies: Sequence[float] | np.ndarray,
    ax: maxes.Axes | None = None,
    fit_birch_murnaghan: bool = True,
    palette: str | Sequence[str] = "okabe_ito",
    preset: str | theme.ThemePreset = "classic",
    title: str | None = None,
) -> maxes.Axes:
    """Plots energy vs volume with an optional 3rd-order Birch-Murnaghan fit."""
    vs = np.asarray(volumes, dtype=np.float64)
    es = np.asarray(energies, dtype=np.float64)
    series = palettes.get_palette(palette)
    pts_col = series[1 % len(series)]
    fit_col = series[5 % len(series)]
    preset_obj = theme.get_preset(preset)
    with theme.xkcd_style(preset=preset):
        if ax is None:
            _, ax = plt.subplots(figsize=(6.0, 4.3))
        else:
            theme.style_existing_axes(ax, preset=preset)
        ax.scatter(
            vs,
            es,
            s=70,
            facecolor=pts_col,
            edgecolor=preset_obj.ink_color,
            linewidth=1.5,
            zorder=5,
            label="DFT Points",
        )
        if fit_birch_murnaghan and len(vs) >= 4:
            popt = _fit_birch_murnaghan(vs, es)
            v_fine = np.linspace(float(np.min(vs)), float(np.max(vs)), 160)
            e_fine = _birch_murnaghan(v_fine, *popt)
            ax.plot(
                v_fine, e_fine, color=fit_col, lw=2.2, label="Birch-Murnaghan"
            )
            b0_gpa = float(popt[2]) * _EV_PER_A3_TO_GPA
            summary = (
                f"V0 = {popt[1]:.2f} A^3\n"
                f"B0 = {b0_gpa:.1f} GPa\n"
                f"B0' = {popt[3]:.2f}"
            )
            ax.text(
                0.50,
                0.56,
                summary,
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=9.0,
                bbox={
                    "boxstyle": "round,pad=0.3",
                    "facecolor": preset_obj.axes_facecolor,
                    "edgecolor": fit_col,
                },
            )
        e_span = max(float(np.ptp(es)), 1e-3)
        ax.set_ylim(
            float(np.min(es)) - 0.14 * e_span,
            float(np.max(es)) + 0.38 * e_span,
        )
        ax.set_xlabel("Cell Volume (A^3)")
        ax.set_ylabel("Total Energy (eV)")
        ax.legend(loc="upper center", ncol=2, frameon=True, fontsize=8.5)
        theme.apply_xkcd_spines(ax, preset=preset)
        if title:
            ax.set_title(title)
    return ax


def _polygon_centroid(pts: np.ndarray) -> np.ndarray:
    """Computes the exact shoelace area centroid of a 2D polygon."""
    if len(pts) < 3:
        return np.mean(pts, axis=0)
    x, y = pts[:, 0], pts[:, 1]
    x_nxt, y_nxt = np.roll(x, -1), np.roll(y, -1)
    cross = x * y_nxt - x_nxt * y
    area2 = float(np.sum(cross))
    if abs(area2) < 1e-12:
        return np.mean(pts, axis=0)
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
    ink_col = theme.get_preset(preset).ink_color
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
                linewidth=1.7,
                alpha=0.28,
                hatch=hatch,
                zorder=2,
            )
            if hatch:
                poly.set_sketch_params(0.0, 0.0, 0.0)
            ax.add_patch(poly)
            ctr = _polygon_centroid(pts)
            ax.text(
                float(ctr[0]),
                float(ctr[1]),
                label,
                ha="center",
                va="center",
                fontsize=10.5,
                color=ink_col,
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
                lw=1.6,
                label="H2/H+",
            )
            ax.plot(
                phs,
                1.229 - 0.05916 * phs,
                color=series[6 % len(series)],
                linestyle="--",
                lw=1.6,
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
