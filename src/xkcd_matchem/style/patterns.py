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
"""Background grid papers and hatched/stippled shading styles for xkcd plots."""

from __future__ import annotations

from collections.abc import Sequence

from matplotlib import axes as maxes
from matplotlib import lines as mlines
import numpy as np

from xkcd_matchem.core import exceptions


BLUEPRINT_BG = "#163A5F"
BLUEPRINT_INK = "#F0F6FC"
BLUEPRINT_GRID = "#64B5F6"

BACKGROUND_STYLES: tuple[str, ...] = (
    "none",
    "lines",
    "ruled",
    "grid",
    "graph",
    "dots",
    "dotgrid",
    "isometric",
    "blueprint",
)

SHADE_STYLES: dict[str, str | None] = {
    "solid": None,
    "none": None,
    "hatch": "///",
    "diagonal": "///",
    "backslash": "\\\\\\",
    "crosshatch": "xxx",
    "vertical": "|||",
    "horizontal": "---",
    "dots": "...",
    "stipple": "...",
    "circles": "ooo",
    "plus": "+++",
    "stars": "***",
}

_SHADE_CYCLE: tuple[str, ...] = (
    "///",
    "...",
    "xxx",
    "\\\\\\",
    "|||",
    "---",
)


def list_background_styles() -> tuple[str, ...]:
    """Returns supported background grid/paper style keys."""
    return BACKGROUND_STYLES


def list_shade_styles() -> tuple[str, ...]:
    """Returns supported named hatched/stippled shading keys."""
    return (*sorted(SHADE_STYLES.keys()), "cycle")


def resolve_hatch(
    shade: str | Sequence[str | None] | None,
    index: int = 0,
) -> str | None:
    """Resolves a named shade style, cycle, or raw Matplotlib hatch string.

    Args:
      shade: Named style (``"solid"``, ``"hatch"``, ``"crosshatch"``,
        ``"dots"``, ``"stipple"``, ``"vertical"``, ``"horizontal"``,
        ``"backslash"``, ``"circles"``, ``"plus"``, ``"stars"``, ``"cycle"``),
        sequence of styles/hatches, raw Matplotlib hatch string, or ``None``.
      index: Category index used when ``shade="cycle"`` or a sequence is given.

    Returns:
      Matplotlib hatch pattern string or ``None`` for solid fill.
    """
    if shade is None:
        return None
    if isinstance(shade, str):
        key = shade.strip().lower()
        if key == "cycle":
            return _SHADE_CYCLE[index % len(_SHADE_CYCLE)]
        if key in SHADE_STYLES:
            return SHADE_STYLES[key]
        return shade
    seq = list(shade)
    if not seq:
        return None
    return resolve_hatch(seq[index % len(seq)], index=index)


def _draw_axes_rule_lines(
    ax: maxes.Axes,
    xs: Sequence[tuple[float, float]],
    ys: Sequence[tuple[float, float]],
    color: str,
    alpha: float,
    linewidth: float,
) -> None:
    """Draws background lines in axes-fraction coordinates behind plot art."""
    for (x0, x1), (y0, y1) in zip(xs, ys, strict=True):
        line = mlines.Line2D(
            [x0, x1],
            [y0, y1],
            transform=ax.transAxes,
            color=color,
            alpha=alpha,
            linewidth=linewidth,
            zorder=0.2,
            clip_on=True,
        )
        line.set_sketch_params(0.0, 0.0, 0.0)
        ax.add_line(line)


def _draw_isometric_grid(
    ax: maxes.Axes,
    color: str,
    alpha: float,
    step: float,
) -> None:
    """Draws 60-degree isometric triangular lines in axes coordinates."""
    xs_out: list[tuple[float, float]] = []
    ys_out: list[tuple[float, float]] = []
    tan60 = 1.7320508075688772
    dx = max(0.04, step)
    for x0 in np.arange(-1.2, 2.2, dx):
        xs_out.append((float(x0), float(x0)))
        ys_out.append((0.0, 1.0))
        xs_out.append((float(x0), float(x0 + 1.0 / tan60)))
        ys_out.append((0.0, 1.0))
        xs_out.append((float(x0), float(x0 - 1.0 / tan60)))
        ys_out.append((0.0, 1.0))
    _draw_axes_rule_lines(ax, xs_out, ys_out, color, alpha, 0.75)


def apply_background_grid(
    ax: maxes.Axes,
    style: str = "grid",
    *,
    color: str | None = None,
    alpha: float | None = None,
    spacing: float = 0.08,
    margin_line: bool = True,
) -> maxes.Axes:
    """Applies a customizable notebook, graph, dot, isometric, or blueprint bg.

    All patterns render in axes-fraction coordinates at low ``zorder`` so they
    remain cleanly aligned regardless of data range or axis scaling.

    Args:
      ax: Target Matplotlib ``Axes``.
      style: One of ``"none"``, ``"lines"``/``"ruled"``, ``"grid"``/``"graph"``,
        ``"dots"``/``"dotgrid"``, ``"isometric"``, or ``"blueprint"``.
      color: Optional grid line/dot color override.
      alpha: Optional grid opacity override in ``[0, 1]``.
      spacing: Base grid spacing in axes-fraction units (default ``0.08``).
      margin_line: Whether ``"ruled"``/``"lines"`` draws a subtle left margin.

    Returns:
      The styled ``Axes`` instance.

    Raises:
      StyleError: If ``style`` is not a recognized background grid key.
    """
    key = style.strip().lower()
    if key not in BACKGROUND_STYLES:
        raise exceptions.StyleError(
            f"Unknown background_grid {style!r}. Valid: {BACKGROUND_STYLES}"
        )
    if key == "none":
        return ax

    step = max(0.03, min(0.25, float(spacing)))
    ax.patch.set_sketch_params(0.0, 0.0, 0.0)

    if key == "blueprint":
        ax.set_facecolor(BLUEPRINT_BG)
        g_col = BLUEPRINT_GRID if color is None else color
        g_alpha = 0.30 if alpha is None else alpha
        major = np.arange(step, 1.0, step)
        minor = np.arange(0.5 * step, 1.0, step)
        _draw_axes_rule_lines(
            ax,
            [(float(x), float(x)) for x in minor] + [(0.0, 1.0) for _ in minor],
            [(0.0, 1.0) for _ in minor] + [(float(y), float(y)) for y in minor],
            g_col,
            g_alpha * 0.55,
            0.55,
        )
        _draw_axes_rule_lines(
            ax,
            [(float(x), float(x)) for x in major] + [(0.0, 1.0) for _ in major],
            [(0.0, 1.0) for _ in major] + [(float(y), float(y)) for y in major],
            g_col,
            g_alpha,
            0.95,
        )
        return ax

    g_col = "#8FA3B8" if color is None else color
    if key in ("lines", "ruled"):
        g_alpha = 0.34 if alpha is None else alpha
        ys = np.arange(step, 1.0, step)
        _draw_axes_rule_lines(
            ax,
            [(0.0, 1.0) for _ in ys],
            [(float(y), float(y)) for y in ys],
            g_col,
            g_alpha,
            0.85,
        )
        if margin_line:
            _draw_axes_rule_lines(
                ax,
                [(0.11, 0.11)],
                [(0.0, 1.0)],
                "#E07A5F",
                min(0.55, g_alpha + 0.12),
                1.05,
            )
    elif key in ("grid", "graph"):
        g_alpha = 0.28 if alpha is None else alpha
        coords = np.arange(step, 1.0, step)
        _draw_axes_rule_lines(
            ax,
            [(float(x), float(x)) for x in coords]
            + [(0.0, 1.0) for _ in coords],
            [(0.0, 1.0) for _ in coords]
            + [(float(y), float(y)) for y in coords],
            g_col,
            g_alpha,
            0.75,
        )
    elif key in ("dots", "dotgrid"):
        g_alpha = 0.45 if alpha is None else alpha
        xs = np.arange(0.5 * step, 1.0, step)
        ys = np.arange(0.5 * step, 1.0, step)
        gx, gy = np.meshgrid(xs, ys)
        pts = ax.scatter(
            gx.ravel(),
            gy.ravel(),
            s=8.5,
            color=g_col,
            alpha=g_alpha,
            edgecolors="none",
            transform=ax.transAxes,
            zorder=0.2,
            clip_on=True,
        )
        pts.set_sketch_params(0.0, 0.0, 0.0)
    elif key == "isometric":
        g_alpha = 0.24 if alpha is None else alpha
        _draw_isometric_grid(ax, g_col, g_alpha, step * 1.4)
    return ax


def shade_between(
    ax: maxes.Axes,
    x: Sequence[float] | np.ndarray,
    y1: Sequence[float] | np.ndarray,
    y2: float | Sequence[float] | np.ndarray = 0.0,
    *,
    color: str = "#56B4E9",
    shade: str | None = "hatch",
    alpha: float = 0.26,
    edgecolor: str | None = None,
    vertical: bool = False,
    zorder: float = 1.5,
) -> object:
    """Draws a solid or hatched/stippled shaded region between two curves."""
    hatch = resolve_hatch(shade)
    edge = color if edgecolor is None else edgecolor
    fill_fn = ax.fill_betweenx if vertical else ax.fill_between
    if hatch:
        poly = fill_fn(
            x,
            y1,
            y2,
            facecolor=color,
            edgecolor=edge,
            alpha=alpha,
            hatch=hatch,
            linewidth=0.6,
            zorder=zorder,
        )
        poly.set_sketch_params(0.0, 0.0, 0.0)
        return poly
    return fill_fn(
        x,
        y1,
        y2,
        facecolor=color,
        edgecolor="none",
        alpha=alpha,
        zorder=zorder,
    )
