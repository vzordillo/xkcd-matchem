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
"""Reaction coordinate energy diagrams and NEB barrier visualization."""

from __future__ import annotations

from collections.abc import Sequence

from matplotlib import axes as maxes
from matplotlib import pyplot as plt
import numpy as np
from scipy import interpolate

from xkcd_matchem.core import trajectory
from xkcd_matchem.style import palettes
from xkcd_matchem.style import theme


def _is_local_peak(y_vals: np.ndarray, idx: int) -> bool:
    """Returns whether state ``idx`` is a local energy maximum."""
    if len(y_vals) < 2:
        return False
    if idx == 0:
        return bool(y_vals[0] > y_vals[1])
    if idx == len(y_vals) - 1:
        return bool(y_vals[-1] > y_vals[-2])
    return bool(
        y_vals[idx] >= y_vals[idx - 1] and y_vals[idx] >= y_vals[idx + 1]
    )


def plot_reaction_profile(
    energies: Sequence[float] | np.ndarray,
    labels: Sequence[str] | None = None,
    ax: maxes.Axes | None = None,
    unit: str = "eV",
    barriers: bool = True,
    smooth_connect: bool = True,
    color: str = "#0072B2",
    preset: str = "classic",
    title: str | None = None,
) -> maxes.Axes:
    """Plots a stationary-state reaction energy diagram with barriers."""
    y_vals = np.asarray(energies, dtype=np.float64)
    n_pts = len(y_vals)
    span = max(float(np.ptp(y_vals)), 0.5)
    with theme.xkcd_style(preset=preset):
        if ax is None:
            _, ax = plt.subplots(figsize=(6.4, 4.2))
        xs = np.arange(n_pts, dtype=np.float64)
        if smooth_connect and n_pts >= 2:
            for k in range(n_pts - 1):
                t = np.linspace(0.0, 1.0, 40)
                sx = (xs[k] + 0.22) + (0.56) * t
                sy = y_vals[k] + (y_vals[k + 1] - y_vals[k]) * (
                    3.0 * t * t - 2.0 * t * t * t
                )
                ax.plot(sx, sy, color=color, linestyle="--", lw=1.4, alpha=0.8)

        for k in range(n_pts):
            ax.plot(
                [xs[k] - 0.22, xs[k] + 0.22],
                [y_vals[k], y_vals[k]],
                color=color,
                lw=3.2,
                solid_capstyle="round",
            )
            lbl_txt = labels[k] if (labels and k < len(labels)) else f"S{k}"
            above = _is_local_peak(y_vals, k)
            y_text = (
                y_vals[k] + 0.05 * span if above else y_vals[k] - 0.05 * span
            )
            ax.text(
                xs[k],
                y_text,
                f"{lbl_txt}\n({y_vals[k]:+.2f})",
                ha="center",
                va="bottom" if above else "top",
                fontsize=9.2,
                color=palettes.INK,
            )

        if barriers and n_pts >= 3:
            peak_idx = int(np.argmax(y_vals))
            if 0 < peak_idx < n_pts - 1:
                e_act = float(y_vals[peak_idx] - y_vals[0])
                ax.annotate(
                    "",
                    xy=(xs[peak_idx], y_vals[peak_idx]),
                    xytext=(xs[peak_idx], y_vals[0]),
                    arrowprops={
                        "arrowstyle": "<->",
                        "color": "#D55E00",
                        "lw": 1.6,
                    },
                )
                ax.plot(
                    [xs[0] + 0.22, xs[peak_idx] + 0.10],
                    [y_vals[0], y_vals[0]],
                    color="#D55E00",
                    linestyle=":",
                    lw=1.1,
                )
                ax.text(
                    xs[peak_idx],
                    0.5 * (y_vals[0] + y_vals[peak_idx]),
                    f"Ea = {e_act:.2f} {unit}",
                    color="#D55E00",
                    ha="center",
                    va="center",
                    fontsize=9.0,
                    bbox={
                        "boxstyle": "round,pad=0.18",
                        "facecolor": palettes.PAPER,
                        "edgecolor": "none",
                        "alpha": 0.9,
                    },
                    zorder=6,
                )

        ax.set_ylim(
            float(np.min(y_vals)) - 0.30 * span,
            float(np.max(y_vals)) + 0.32 * span,
        )
        ax.set_xlim(-0.55, n_pts - 0.35)
        ax.set_xticks([])
        ax.set_xlabel("Reaction Coordinate")
        ax.set_ylabel(f"Relative Energy ({unit})")
        theme.apply_xkcd_spines(ax)
        if title:
            ax.set_title(title)
    return ax


def plot_neb(
    source: trajectory.Trajectory | Sequence[float] | np.ndarray,
    ax: maxes.Axes | None = None,
    unit: str = "eV",
    annotate_ts: bool = True,
    color: str = "#D55E00",
    preset: str = "classic",
    title: str | None = None,
) -> maxes.Axes:
    """Plots a continuous spline-interpolated NEB reaction barrier curve."""
    if isinstance(source, trajectory.Trajectory):
        energies = source.energy_profile(relative=True, unit=unit)
    else:
        raw = np.asarray(source, dtype=np.float64)
        energies = raw - raw[0]

    n_img = len(energies)
    xi = np.linspace(0.0, 1.0, n_img)
    with theme.xkcd_style(preset=preset):
        if ax is None:
            _, ax = plt.subplots(figsize=(6.4, 4.2))
        if n_img >= 4:
            spline = interpolate.CubicSpline(xi, energies, bc_type="natural")
            x_fine = np.linspace(0.0, 1.0, 120)
            y_fine = spline(x_fine)
            ax.plot(x_fine, y_fine, color=color, lw=2.2, label="NEB MEP")
        else:
            ax.plot(xi, energies, color=color, lw=2.2)
        ax.scatter(
            xi,
            energies,
            s=65,
            facecolor=palettes.PAPER,
            edgecolor=color,
            linewidth=2.0,
            zorder=5,
            label="NEB Images",
        )
        if annotate_ts and n_img >= 3:
            ts_idx = int(np.argmax(energies))
            ax.annotate(
                f"TS: +{energies[ts_idx]:.2f} {unit}",
                xy=(xi[ts_idx], energies[ts_idx]),
                xytext=(
                    xi[ts_idx],
                    energies[ts_idx] + 0.14 * max(float(np.ptp(energies)), 0.2),
                ),
                ha="center",
                color="#0072B2",
                arrowprops={"arrowstyle": "->", "color": "#0072B2", "lw": 1.4},
            )
        ax.axhline(0.0, color=palettes.MUTED_GRAY, linestyle="--", lw=1.1)
        span = max(float(np.ptp(energies)), 0.4)
        ax.set_ylim(
            float(np.min(energies)) - 0.15 * span,
            float(np.max(energies)) + 0.28 * span,
        )
        ax.set_xlabel("Normalized Reaction Coordinate")
        ax.set_ylabel(f"Relative Energy ({unit})")
        theme.apply_xkcd_spines(ax)
        if title:
            ax.set_title(title)
    return ax
