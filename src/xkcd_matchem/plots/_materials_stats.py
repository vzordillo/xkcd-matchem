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
"""Space-group crystal-system breakdown and hatched property histograms."""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
import math
from typing import Any
from typing import cast

from matplotlib import axes as maxes
from matplotlib import colors as mcolors
from matplotlib import patches
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats

from xkcd_matchem.core import exceptions
from xkcd_matchem.plots import _materials_ternary as mat_tern
from xkcd_matchem.render import primitives
from xkcd_matchem.style import palettes
from xkcd_matchem.style import patterns
from xkcd_matchem.style import theme


CRYSTAL_SYSTEM_RANGES: tuple[tuple[str, int, int], ...] = (
    ("Triclinic", 1, 2),
    ("Monoclinic", 3, 15),
    ("Orthorhombic", 16, 74),
    ("Tetragonal", 75, 142),
    ("Trigonal", 143, 167),
    ("Hexagonal", 168, 194),
    ("Cubic", 195, 230),
)


def _blend_alpha_hex(fg_color: str, bg_color: str, alpha: float) -> str:
    """Computes effective opaque hex for ``fg_color`` over ``bg_color``."""
    fr, fg, fb = mcolors.to_rgb(fg_color)
    br, bg, bb = mcolors.to_rgb(bg_color)
    a = max(0.0, min(1.0, float(alpha)))
    return mcolors.to_hex(
        (
            a * fr + (1.0 - a) * br,
            a * fg + (1.0 - a) * bg,
            a * fb + (1.0 - a) * bb,
        )
    )


def spacegroup_to_crystal_system(sg_number: int) -> str:
    """Maps an International Tables space-group number (1-230) to its system."""
    for name, lo, hi in CRYSTAL_SYSTEM_RANGES:
        if lo <= sg_number <= hi:
            return name
    raise exceptions.StructureError(
        f"Space group number {sg_number} out of valid range 1..230."
    )


def aggregate_spacegroups(
    spacegroups: Sequence[int] | Mapping[int, int],
) -> tuple[dict[str, int], dict[str, list[tuple[int, int]]]]:
    """Aggregates space-group counts by crystal system."""
    sg_counts: dict[int, int] = {}
    if isinstance(spacegroups, Mapping):
        for sg, cnt in spacegroups.items():
            sg_counts[int(sg)] = sg_counts.get(int(sg), 0) + int(cnt)
    else:
        for sg in spacegroups:
            sg_int = int(sg)
            sg_counts[sg_int] = sg_counts.get(sg_int, 0) + 1

    sys_totals: dict[str, int] = {}
    sys_members: dict[str, list[tuple[int, int]]] = {}
    for sg_num, cnt in sg_counts.items():
        if cnt <= 0:
            continue
        sys_name = spacegroup_to_crystal_system(sg_num)
        sys_totals[sys_name] = sys_totals.get(sys_name, 0) + cnt
        sys_members.setdefault(sys_name, []).append((sg_num, cnt))
    for members in sys_members.values():
        members.sort(key=lambda pair: (-pair[1], pair[0]))
    return sys_totals, sys_members


def draw_sunburst_rings(
    ax: maxes.Axes,
    sys_totals: dict[str, int],
    sys_members: dict[str, list[tuple[int, int]]],
    series: Sequence[str],
    ink_col: str,
    shade: str | Sequence[str | None] | None,
    bg_color: str = palettes.PAPER,
    sw: theme.StrokeHierarchy | None = None,
) -> None:
    """Draws inner crystal-system ring and outer space-group ring wedges."""
    total = float(sum(sys_totals.values()))
    if total <= 0.0:
        return
    weights = sw if sw is not None else theme.stroke_weights("classic")
    ordered = [
        (name, sys_totals[name])
        for name, _, _ in CRYSTAL_SYSTEM_RANGES
        if name in sys_totals
    ]
    angle = 90.0
    for idx, (sys_name, sys_cnt) in enumerate(ordered):
        span = 360.0 * (sys_cnt / total)
        raw_col = series[(idx + 1) % len(series)]
        col = mat_tern.theme_series_color(raw_col, bg_color, ink_col)
        hatch = patterns.resolve_hatch(shade, index=idx)
        inner = patches.Wedge(
            (0.0, 0.0),
            0.58,
            angle,
            angle + span,
            width=0.33,
            facecolor=col,
            edgecolor=ink_col,
            linewidth=round(0.85 * weights.primary, 2),
            alpha=0.86,
            hatch=hatch,
            zorder=3.0,
        )
        if hatch:
            inner.set_sketch_params(0.0, 0.0, 0.0)
        ax.add_patch(inner)
        mid_rad = math.radians(angle + 0.5 * span)
        if span >= 22.0:
            theme.place_halo_text(
                ax,
                0.415 * math.cos(mid_rad),
                0.415 * math.sin(mid_rad),
                sys_name[:5] if span < 42.0 else sys_name,
                color=ink_col,
                bg_color=bg_color,
                fontsize=8.0,
                pad=0.14,
                alpha=0.85,
                zorder=5.0,
            )

        sub_ang = angle
        outer_txt_col = primitives.contrast_text_color(
            _blend_alpha_hex(col, bg_color, 0.52)
        )
        for sg_num, cnt in sys_members.get(sys_name, []):
            sub_span = 360.0 * (cnt / total)
            outer = patches.Wedge(
                (0.0, 0.0),
                0.96,
                sub_ang,
                sub_ang + sub_span,
                width=0.34,
                facecolor=col,
                edgecolor=ink_col,
                linewidth=weights.overlay,
                alpha=0.52,
                zorder=2.5,
            )
            ax.add_patch(outer)
            if sub_span >= 15.0:
                s_rad = math.radians(sub_ang + 0.5 * sub_span)
                ax.text(
                    0.79 * math.cos(s_rad),
                    0.79 * math.sin(s_rad),
                    f"#{sg_num}",
                    ha="center",
                    va="center",
                    fontsize=7.8,
                    color=outer_txt_col,
                    zorder=5.0,
                )
            sub_ang += sub_span
        angle += span

    ax.text(
        0.0,
        0.0,
        f"N={int(total)}",
        ha="center",
        va="center",
        fontsize=11.0,
        color=ink_col,
        zorder=6.0,
    )


def plot_spacegroup_distribution(
    spacegroups: Sequence[int] | Mapping[int, int],
    *,
    mode: str = "sunburst",
    ax: maxes.Axes | None = None,
    shade: str | Sequence[str | None] | None = None,
    background_grid: str | None = None,
    palette: str | Sequence[str] = "okabe_ito",
    preset: str | theme.ThemePreset = "classic",
    title: str | None = None,
) -> maxes.Axes:
    """Plots crystal-system & space-group counts as a sunburst or bar chart.

    Inspired by ``pymatviz.spacegroup_sunburst`` and ``spacegroup_bar``,
    automatically grouping International Tables space groups ``1..230`` into
    the 7 crystal systems.

    Args:
      spacegroups: Sequence of space-group numbers or ``{sg_num: count}`` dict.
      mode: ``"sunburst"`` (two-ring radial chart) or ``"bar"`` (crystal-system
        count bars).
      ax: Optional target Matplotlib ``Axes``.
      shade: Optional hatched/stippled shading (e.g., ``"cycle"``, ``"hatch"``).
      background_grid: Optional background grid style (applies in ``"bar"``).
      palette: Series palette name or hex color list.
      preset: Visual theme preset name or ``ThemePreset``.
      title: Optional plot heading.

    Returns:
      Matplotlib ``Axes`` containing the space-group breakdown.
    """
    view_mode = mode.strip().lower()
    if view_mode not in ("sunburst", "bar"):
        raise exceptions.StructureError(
            f"Unsupported mode {mode!r}. Use 'sunburst' or 'bar'."
        )
    sys_totals, sys_members = aggregate_spacegroups(spacegroups)
    series = palettes.get_palette(palette)
    preset_obj = theme.get_preset(preset)
    ink_col = preset_obj.ink_color
    bg_col = preset_obj.axes_facecolor
    sw = theme.stroke_weights(preset)

    with theme.xkcd_style(preset=preset):
        if ax is None:
            figsize = (5.8, 5.4) if view_mode == "sunburst" else (7.0, 4.3)
            _, ax = plt.subplots(figsize=figsize)
        else:
            theme.style_existing_axes(ax, preset=preset)

        if view_mode == "sunburst":
            draw_sunburst_rings(
                ax,
                sys_totals,
                sys_members,
                series,
                ink_col,
                shade,
                bg_col,
                sw=sw,
            )
            ax.set_xlim(-1.12, 1.12)
            ax.set_ylim(-1.12, 1.12)
            ax.set_aspect("equal")
            ax.set_axis_off()
        else:
            if background_grid:
                patterns.apply_background_grid(ax, style=background_grid)
            ordered = [
                (name, sys_totals[name])
                for name, _, _ in CRYSTAL_SYSTEM_RANGES
                if name in sys_totals
            ]
            names = [item[0] for item in ordered]
            counts = [item[1] for item in ordered]
            xs = np.arange(len(names))
            for idx, (cnt, x_pos) in enumerate(zip(counts, xs, strict=True)):
                raw_col = series[(idx + 1) % len(series)]
                col = mat_tern.theme_series_color(raw_col, bg_col, ink_col)
                hatch = patterns.resolve_hatch(shade, index=idx)
                bar_list = ax.bar(
                    [x_pos],
                    [cnt],
                    color=col,
                    edgecolor=ink_col,
                    linewidth=round(0.85 * sw.primary, 2),
                    width=0.64,
                    alpha=0.88,
                    hatch=hatch,
                    zorder=3.0,
                )
                if hatch:
                    bar_list[0].set_sketch_params(0.0, 0.0, 0.0)
                ax.text(
                    float(x_pos),
                    float(cnt),
                    str(cnt),
                    ha="center",
                    va="bottom",
                    fontsize=9.5,
                    color=ink_col,
                )
            ax.set_xticks(xs)
            ax.set_xticklabels(names, rotation=22, ha="right")
            ax.set_ylabel("Structure Count")
            theme.apply_xkcd_spines(ax, preset=preset)
        if title:
            ax.set_title(title, color=ink_col)
    return ax


def draw_hist_series(
    ax: maxes.Axes,
    lbl: str,
    vals: np.ndarray,
    bins: int | np.ndarray,
    style_cfg: tuple[str, str, str | None, bool, bool, bool, int],
    *,
    bg_color: str = palettes.PAPER,
    sw: theme.StrokeHierarchy | None = None,
) -> None:
    """Draws one histogram series with optional hatch, KDE, and mean line."""
    col, ink_col, hatch, multi, show_kde, show_mean, series_idx = style_cfg
    clean = vals[np.isfinite(vals)]
    if clean.size == 0:
        return
    weights = sw if sw is not None else theme.stroke_weights("classic")
    bins_arg: Any = bins.tolist() if isinstance(bins, np.ndarray) else bins
    _, edges, bar_patches = ax.hist(
        clean,
        bins=bins_arg,
        color=col,
        edgecolor=ink_col,
        linewidth=1.15,
        alpha=0.58 if multi else 0.72,
        hatch=hatch,
        label=lbl if multi else None,
        zorder=3.0,
    )
    if hatch:
        for patch in cast(Any, bar_patches):
            patch.set_sketch_params(0.0, 0.0, 0.0)
    if show_kde and clean.size >= 4 and float(np.std(clean)) > 1e-9:
        kde = stats.gaussian_kde(clean)
        x_grid = np.linspace(float(edges[0]), float(edges[-1]), 140)
        bin_w = float(edges[1] - edges[0])
        ax.plot(
            x_grid,
            kde(x_grid) * clean.size * bin_w,
            color=col,
            linewidth=weights.overlay + 0.25,
            zorder=4.5,
        )
    if show_mean:
        mu = float(np.mean(clean))
        m_line = ax.axvline(
            mu,
            color=ink_col,
            linestyle="--",
            linewidth=weights.guide,
            alpha=0.82,
            zorder=4.2,
        )
        m_line.set_sketch_params(0.0, 0.0, 0.0)
        txt = f" {lbl}: {mu:.2f}" if multi else f" mean={mu:.2f}"
        y_frac = max(0.18, 0.96 - 0.10 * series_idx)
        artist = ax.text(
            mu,
            y_frac,
            txt,
            transform=ax.get_xaxis_transform(),
            ha="left",
            va="top",
            fontsize=8.6,
            color=ink_col,
            bbox=theme.label_halo_bbox(bg_color, pad=0.06, alpha=0.84),
            zorder=5.0,
        )
        patch = artist.get_bbox_patch()
        if patch is not None:
            patch.set_sketch_params(0.0, 0.0, 0.0)


def _resolve_hist_bins(
    series_map: dict[str, np.ndarray],
    bins: int,
) -> int | np.ndarray:
    """Computes shared bin edges across all finite histogram series."""
    non_empty = [v[np.isfinite(v)] for v in series_map.values()]
    non_empty = [c for c in non_empty if c.size > 0]
    if not non_empty:
        return bins
    pooled = np.concatenate(non_empty)
    lo, hi = float(np.min(pooled)), float(np.max(pooled))
    if hi - lo < 1e-12:
        lo -= 0.5
        hi += 0.5
    return np.linspace(lo, hi, max(int(bins), 1) + 1)


def plot_histogram(
    data: Sequence[float]
    | np.ndarray
    | Mapping[str, Sequence[float] | np.ndarray],
    *,
    bins: int = 18,
    ax: maxes.Axes | None = None,
    xlabel: str = "Property Value",
    ylabel: str = "Count",
    show_kde: bool = True,
    show_mean: bool = True,
    shade: str | Sequence[str | None] | None = "hatch",
    background_grid: str | None = None,
    palette: str | Sequence[str] = "okabe_ito",
    preset: str | theme.ThemePreset = "classic",
    title: str | None = None,
) -> maxes.Axes:
    """Plots an xkcd-style property histogram with hatched fills and KDE.

    Args:
      data: 1D array/sequence of values or ``{series_label: values}`` mapping.
      bins: Number of histogram bins.
      ax: Optional target Matplotlib ``Axes``.
      xlabel: Horizontal axis label.
      ylabel: Vertical axis label.
      show_kde: Whether to overlay a smooth SciPy Gaussian KDE curve.
      show_mean: Whether to mark each series mean with a vertical dashed line.
      shade: Bar fill pattern (``"hatch"``, ``"crosshatch"``, ``"dots"``,
        ``"cycle"``, ``"solid"``, or raw Matplotlib hatch string).
      background_grid: Optional background grid (``"grid"``, ``"dots"``, etc.).
      palette: Series palette name or hex color sequence.
      preset: Visual theme preset name or ``ThemePreset``.
      title: Optional heading above the histogram.

    Returns:
      Matplotlib ``Axes`` containing the histogram.
    """
    series_map: dict[str, np.ndarray] = (
        {str(k): np.asarray(v, dtype=np.float64) for k, v in data.items()}
        if isinstance(data, Mapping)
        else {"Distribution": np.asarray(data, dtype=np.float64)}
    )
    series_cols = palettes.get_palette(palette)
    preset_obj = theme.get_preset(preset)
    ink_col = preset_obj.ink_color
    bg_col = preset_obj.axes_facecolor
    sw = theme.stroke_weights(preset)
    multi = len(series_map) > 1
    bin_spec = _resolve_hist_bins(series_map, bins)

    with theme.xkcd_style(preset=preset):
        if ax is None:
            _, ax = plt.subplots(figsize=(6.4, 4.2))
        else:
            theme.style_existing_axes(ax, preset=preset)
        if background_grid:
            patterns.apply_background_grid(ax, style=background_grid)

        for idx, (lbl, vals) in enumerate(series_map.items()):
            raw_col = series_cols[(idx + 1) % len(series_cols)]
            col = mat_tern.theme_series_color(raw_col, bg_col, ink_col)
            hatch = patterns.resolve_hatch(shade, index=idx)
            draw_hist_series(
                ax,
                lbl,
                vals,
                bin_spec,
                (col, ink_col, hatch, multi, show_kde, show_mean, idx),
                bg_color=bg_col,
                sw=sw,
            )

        if multi:
            mat_tern.style_legend(
                ax, ink_col, bg_col, loc="best", frameon=True, fontsize=8.8
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        theme.apply_xkcd_spines(ax, preset=preset)
        if title:
            ax.set_title(title, color=ink_col)
    return ax
