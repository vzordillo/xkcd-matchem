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
"""ML parity plots, bar benchmarks, and xkcd-style pipeline diagrams."""

from __future__ import annotations

from collections.abc import Sequence

from matplotlib import axes as maxes
from matplotlib import patches
from matplotlib import pyplot as plt
import numpy as np

from xkcd_matchem.core import exceptions
from xkcd_matchem.style import palettes
from xkcd_matchem.style import patterns
from xkcd_matchem.style import theme


def plot_parity(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
    ax: maxes.Axes | None = None,
    unit: str = "eV",
    show_metrics: bool = True,
    color: str = "#0072B2",
    preset: str | theme.ThemePreset = "classic",
    title: str | None = None,
) -> maxes.Axes:
    """Plots predicted vs reference parity scatter with MAE, RMSE, and R^2."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if yt.shape != yp.shape:
        raise exceptions.StructureError(
            "y_true and y_pred must have identical shapes."
        )
    if yt.size > 0:
        lo = min(float(np.min(yt)), float(np.min(yp)))
        hi = max(float(np.max(yt)), float(np.max(yp)))
    else:
        lo, hi = 0.0, 1.0
    pad = 0.08 * max(hi - lo, 1.0)
    preset_obj = theme.get_preset(preset)
    with theme.xkcd_style(preset=preset):
        if ax is None:
            _, ax = plt.subplots(figsize=(5.0, 4.8))
        else:
            theme.style_existing_axes(ax, preset=preset)
        ax.plot(
            [lo - pad, hi + pad],
            [lo - pad, hi + pad],
            color=preset_obj.ink_color,
            linestyle="--",
            lw=1.5,
            label="y = x",
        )
        ax.scatter(
            yt,
            yp,
            s=52,
            facecolor=color,
            edgecolor=preset_obj.ink_color,
            linewidth=1.1,
            alpha=0.82,
            zorder=4,
        )
        if show_metrics and yt.size > 0:
            err = yp - yt
            mae = float(np.mean(np.abs(err)))
            rmse = float(np.sqrt(np.mean(err * err)))
            ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
            r2_str = (
                "N/A"
                if ss_tot <= 1e-12
                else f"{1.0 - float(np.sum(err * err)) / ss_tot:.3f}"
            )
            metrics_txt = (
                f"MAE = {mae:.3f} {unit}\n"
                f"RMSE = {rmse:.3f} {unit}\n"
                f"R^2 = {r2_str}"
            )
            ax.text(
                0.05,
                0.95,
                metrics_txt,
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=9.5,
                bbox={
                    "boxstyle": "round,pad=0.3",
                    "facecolor": preset_obj.axes_facecolor,
                    "edgecolor": color,
                },
            )
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel(f"Reference ({unit})")
        ax.set_ylabel(f"Predicted ({unit})")
        ax.set_aspect("equal")
        theme.apply_xkcd_spines(ax, preset=preset)
        if title:
            ax.set_title(title)
    return ax


def _bar_error_extents(
    errors: Sequence[float] | np.ndarray | None,
    n_bars: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalizes None, scalar, 1D, or 2xN error inputs to (lower, upper)."""
    if errors is None or n_bars == 0:
        zeros = np.zeros(n_bars, dtype=np.float64)
        return zeros, zeros
    arr = np.asarray(errors, dtype=np.float64)
    if arr.ndim == 0:
        vec = np.full(n_bars, abs(float(arr)), dtype=np.float64)
        return vec, vec
    if arr.ndim == 1:
        vec = np.abs(arr)
        return vec, vec
    return np.abs(arr[0]), np.abs(arr[1])


def _annotate_bar_values(
    ax: maxes.Axes,
    bars: Sequence[patches.Rectangle],
    vals: np.ndarray,
    err_lo: np.ndarray,
    err_hi: np.ndarray,
    ink_col: str,
) -> None:
    """Places signed bar-value annotations cleanly beyond bar and error caps."""
    for rect, v, e_lo, e_hi in zip(bars, vals, err_lo, err_hi, strict=True):
        is_neg = float(v) < 0.0
        tip_y = float(v - e_lo) if is_neg else float(v + e_hi)
        ax.annotate(
            f"{v:.2f}",
            xy=(rect.get_x() + rect.get_width() * 0.5, tip_y),
            xytext=(0, -4 if is_neg else 4),
            textcoords="offset points",
            ha="center",
            va="top" if is_neg else "bottom",
            fontsize=9.5,
            color=ink_col,
        )


def plot_bar_comparison(
    labels: Sequence[str],
    values: Sequence[float] | np.ndarray,
    ax: maxes.Axes | None = None,
    errors: Sequence[float] | np.ndarray | None = None,
    ylabel: str = "Value",
    annotate_values: bool = True,
    shade: str | Sequence[str | None] | None = None,
    background_grid: str | None = None,
    palette: str | Sequence[str] = "okabe_ito",
    preset: str | theme.ThemePreset = "classic",
    title: str | None = None,
) -> maxes.Axes:
    """Plots an xkcd-style categorical bar comparison with customizable fill."""
    vals = np.asarray(values, dtype=np.float64)
    err_lo, err_hi = _bar_error_extents(errors, len(vals))
    series = palettes.get_palette(palette)
    colors = [series[(i + 1) % len(series)] for i in range(len(labels))]
    ink_col = theme.get_preset(preset).ink_color
    with theme.xkcd_style(preset=preset):
        if ax is None:
            _, ax = plt.subplots(figsize=(6.2, 4.2))
        else:
            theme.style_existing_axes(ax, preset=preset)
        if background_grid:
            patterns.apply_background_grid(ax, style=background_grid)
        xs = np.arange(len(labels))
        bars = ax.bar(
            xs,
            vals,
            yerr=errors,
            color=colors,
            edgecolor=ink_col,
            linewidth=1.5,
            width=0.62,
            capsize=4.0,
            alpha=0.88,
            zorder=3.0,
        )
        if shade is not None:
            for idx, rect in enumerate(bars):
                hatch = patterns.resolve_hatch(shade, index=idx)
                if hatch:
                    rect.set_hatch(hatch)
                    rect.set_sketch_params(0.0, 0.0, 0.0)
        if annotate_values:
            _annotate_bar_values(ax, bars, vals, err_lo, err_hi, ink_col)
        if vals.size:
            y_hi = float(np.max(vals + err_hi))
            y_lo = float(np.min(vals - err_lo))
            span = max(y_hi - min(0.0, y_lo), abs(y_hi), abs(y_lo), 1.0)
            top = y_hi + 0.16 * span if y_hi >= 0.0 else 0.06 * span
            bot = y_lo - 0.16 * span if y_lo < 0.0 else 0.0
            ax.set_ylim(bot, top)
            if y_lo < 0.0:
                ax.axhline(0.0, color=ink_col, linewidth=1.0, alpha=0.6)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel)
        theme.apply_xkcd_spines(ax, preset=preset)
        if title:
            ax.set_title(title)
    return ax


def plot_pipeline_diagram(
    steps: Sequence[tuple[str, str]],
    ax: maxes.Axes | None = None,
    orientation: str = "horizontal",
    palette: str | Sequence[str] = "okabe_ito",
    preset: str | theme.ThemePreset = "classic",
    title: str | None = None,
) -> maxes.Axes:
    """Plots a multi-stage workflow flowchart with xkcd-style rounded boxes.

    Args:
      steps: Sequence of ``(stage_title, stage_detail)`` pairs.
      ax: Optional target ``Axes``.
      orientation: ``"horizontal"`` or ``"vertical"``.
      palette: Series palette name or custom sequence of colors.
      preset: Visual theme preset name or ``ThemePreset``.
      title: Optional title above the flowchart.

    Returns:
      Matplotlib ``Axes`` containing the flowchart.
    """
    series = palettes.get_palette(palette)
    ink_col = theme.get_preset(preset).ink_color
    n_steps = len(steps)
    is_horiz = orientation.lower() == "horizontal"
    with theme.xkcd_style(preset=preset):
        if ax is None:
            figsize = (
                (max(6.5, 2.2 * n_steps), 3.0)
                if is_horiz
                else (4.6, max(4.0, 1.4 * n_steps))
            )
            _, ax = plt.subplots(figsize=figsize)
        else:
            theme.style_existing_axes(ax, preset=preset)
        for idx, (heading, detail) in enumerate(steps):
            col = series[(idx + 1) % len(series)]
            cx = idx * 1.4 if is_horiz else 0.0
            cy = 0.0 if is_horiz else -idx * 1.25
            w, h = (1.05, 0.72) if is_horiz else (2.4, 0.75)
            box = patches.FancyBboxPatch(
                (cx - w * 0.5, cy - h * 0.5),
                w,
                h,
                boxstyle="round,pad=0.06,rounding_size=0.12",
                facecolor=col,
                edgecolor=ink_col,
                linewidth=1.6,
                alpha=0.25,
                zorder=2,
            )
            ax.add_patch(box)
            ax.text(
                cx,
                cy + 0.13,
                heading,
                ha="center",
                va="center",
                fontsize=10.5,
                color=ink_col,
                zorder=4,
            )
            ax.text(
                cx,
                cy - 0.14,
                detail,
                ha="center",
                va="center",
                fontsize=8.5,
                color=palettes.MUTED_GRAY,
                zorder=4,
            )
            if idx < n_steps - 1:
                if is_horiz:
                    p0 = (cx + w * 0.5 + 0.06, cy)
                    p1 = (cx + 1.4 - w * 0.5 - 0.06, cy)
                else:
                    p0 = (cx, cy - h * 0.5 - 0.06)
                    p1 = (cx, cy - 1.25 + h * 0.5 + 0.06)
                ax.annotate(
                    "",
                    xy=p1,
                    xytext=p0,
                    arrowprops={
                        "arrowstyle": "->",
                        "color": ink_col,
                        "lw": 1.8,
                    },
                    zorder=5,
                )
        if is_horiz:
            ax.set_xlim(-0.7, max(0.7, (n_steps - 1) * 1.4 + 0.7))
            ax.set_ylim(-0.65, 0.65)
        else:
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(min(-0.65, -(n_steps - 1) * 1.25 - 0.65), 0.65)
        ax.set_axis_off()
        if title:
            ax.set_title(title)
    return ax
