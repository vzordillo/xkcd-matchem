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
"""ML parity plots, bar benchmarks, and hand-drawn pipeline diagrams."""

from __future__ import annotations

from collections.abc import Sequence

from matplotlib import axes as maxes
from matplotlib import patches
from matplotlib import pyplot as plt
import numpy as np

from xkcd_matchem.style import palettes
from xkcd_matchem.style import theme


def plot_parity(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
    ax: maxes.Axes | None = None,
    unit: str = "eV",
    show_metrics: bool = True,
    color: str = "#0072B2",
    preset: str = "classic",
    title: str | None = None,
) -> maxes.Axes:
    """Plots predicted vs reference parity scatter with MAE, RMSE, and R^2."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    lo = min(float(np.min(yt)), float(np.min(yp)))
    hi = max(float(np.max(yt)), float(np.max(yp)))
    pad = 0.08 * max(hi - lo, 1.0)
    with theme.xkcd_style(preset=preset):
        if ax is None:
            _, ax = plt.subplots(figsize=(5.0, 4.8))
        ax.plot(
            [lo - pad, hi + pad],
            [lo - pad, hi + pad],
            color=palettes.INK,
            linestyle="--",
            lw=1.5,
            label="y = x",
        )
        ax.scatter(
            yt,
            yp,
            s=52,
            facecolor=color,
            edgecolor=palettes.INK,
            linewidth=1.1,
            alpha=0.82,
            zorder=4,
        )
        if show_metrics and yt.size > 0:
            err = yp - yt
            mae = float(np.mean(np.abs(err)))
            rmse = float(np.sqrt(np.mean(err * err)))
            ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
            r2 = 1.0 - float(np.sum(err * err)) / max(ss_tot, 1e-12)
            metrics_txt = (
                f"MAE = {mae:.3f} {unit}\n"
                f"RMSE = {rmse:.3f} {unit}\n"
                f"R^2 = {r2:.3f}"
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
                    "facecolor": palettes.PAPER,
                    "edgecolor": color,
                },
            )
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel(f"Reference ({unit})")
        ax.set_ylabel(f"Predicted ({unit})")
        ax.set_aspect("equal")
        theme.apply_xkcd_spines(ax)
        if title:
            ax.set_title(title)
    return ax


def plot_bar_comparison(
    labels: Sequence[str],
    values: Sequence[float] | np.ndarray,
    ax: maxes.Axes | None = None,
    errors: Sequence[float] | np.ndarray | None = None,
    ylabel: str = "Value",
    annotate_values: bool = True,
    preset: str = "classic",
    title: str | None = None,
) -> maxes.Axes:
    """Plots a hand-drawn categorical bar comparison with Okabe-Ito fills."""
    vals = np.asarray(values, dtype=np.float64)
    series = palettes.get_palette("okabe_ito")
    colors = [series[(i + 1) % len(series)] for i in range(len(labels))]
    with theme.xkcd_style(preset=preset):
        if ax is None:
            _, ax = plt.subplots(figsize=(6.2, 4.2))
        xs = np.arange(len(labels))
        bars = ax.bar(
            xs,
            vals,
            yerr=errors,
            color=colors,
            edgecolor=palettes.INK,
            linewidth=1.5,
            width=0.62,
            capsize=4.0,
            alpha=0.88,
        )
        if annotate_values:
            for rect, v in zip(bars, vals, strict=True):
                ax.text(
                    rect.get_x() + rect.get_width() * 0.5,
                    rect.get_height(),
                    f"{v:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9.5,
                )
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel)
        theme.apply_xkcd_spines(ax)
        if title:
            ax.set_title(title)
    return ax


def plot_pipeline_diagram(
    steps: Sequence[tuple[str, str]],
    ax: maxes.Axes | None = None,
    orientation: str = "horizontal",
    preset: str = "classic",
    title: str | None = None,
) -> maxes.Axes:
    """Plots a multi-stage scientific workflow flowchart with hand-drawn boxes.

    Args:
      steps: Sequence of ``(stage_title, stage_detail)`` pairs.
      ax: Optional target ``Axes``.
      orientation: ``"horizontal"`` or ``"vertical"``.
      preset: Visual theme preset name.
      title: Optional title above the flowchart.

    Returns:
      Matplotlib ``Axes`` containing the flowchart.
    """
    series = palettes.get_palette("okabe_ito")
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
                edgecolor=palettes.INK,
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
                color=palettes.INK,
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
                        "color": palettes.INK,
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
