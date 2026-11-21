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
"""Old-parchment scroll card templates in portrait and landscape views."""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses

from matplotlib import axes as maxes
from matplotlib import figure
from matplotlib import patches
from matplotlib import pyplot as plt
import numpy as np

from xkcd_matchem.core import exceptions
from xkcd_matchem.style import patterns
from xkcd_matchem.style import theme


PARCHMENT_LIGHT = "#F6ECD4"
PARCHMENT_ROLLER = "#E2C896"
PARCHMENT_SHADOW = "#C9A86C"
PARCHMENT_INK = "#3A2618"
_GRID_LINE_STYLES = (
    "lines",
    "ruled",
    "grid",
    "graph",
    "dots",
    "dotgrid",
    "blueprint",
)
_SQUARE_GRID_STYLES = ("grid", "graph", "dots", "dotgrid", "blueprint")


def _normalize_grid_style(style: str | None) -> str | None:
    """Returns canonical lowercase grid style or ``None`` when disabled."""
    if style is None:
        return None
    key = style.strip().lower()
    if not key or key == "none":
        return None
    return key


def _resolve_scroll_grid_spacing(
    style: str | None,
    inner_size_mm: tuple[float, float],
    grid_spacing: float | None,
    min_rows: int = 0,
    square_grid: bool = True,
) -> tuple[float | None, float | None]:
    """Computes notebook-scale ``(spacing_y, spacing_x)`` for a scroll Axes."""
    key = _normalize_grid_style(style)
    if key is None or key not in _GRID_LINE_STYLES:
        return None, None
    ax_w_mm, ax_h_mm = (
        max(float(inner_size_mm[0]), 10.0),
        max(float(inner_size_mm[1]), 10.0),
    )
    if grid_spacing is not None:
        step_y = max(0.03, min(0.25, float(grid_spacing)))
    else:
        target_mm = 7.4 if key in ("lines", "ruled") else 6.0
        n_slots = max(5, min_rows + 1, round(ax_h_mm / target_mm))
        n_slots = min(n_slots, 24)
        step_y = 1.0 / float(n_slots)

    step_x: float | None = None
    if key in _SQUARE_GRID_STYLES and square_grid:
        pitch_mm = max(step_y * ax_h_mm, 1.5)
        n_cols = max(4, min(36, round(ax_w_mm / pitch_mm)))
        step_x = 1.0 / float(n_cols)
    return step_y, step_x


def _writable_line_ys(
    raw_ys: np.ndarray,
    step_y: float | None,
) -> tuple[float, ...]:
    """Drops cramped top-edge slivers so every notebook line has headroom."""
    if raw_ys.size == 0:
        return ()
    if step_y is None:
        return tuple(float(y) for y in raw_ys)
    min_headroom = 0.65 * float(step_y)
    kept = [float(y) for y in raw_ys if (1.0 - float(y)) >= min_headroom]
    return tuple(kept) if kept else tuple(float(y) for y in raw_ys)


def _baseline_lift(line_ys: Sequence[float], offset: float | None) -> float:
    """Computes subtle upward baseline lift above a notebook rule line."""
    if offset is not None:
        return float(offset)
    if len(line_ys) >= 2:
        step_y = abs(float(line_ys[0]) - float(line_ys[1]))
    elif len(line_ys) == 1:
        step_y = min(float(line_ys[0]), 1.0 - float(line_ys[0]))
    else:
        step_y = 0.08
    return min(0.014, max(0.005, 0.08 * step_y))


def _resolve_horizontal_columns(
    grid_xs: Sequence[float],
    margin_line: bool,
    has_bullet: bool,
    col: int = 0,
) -> tuple[float, float]:
    """Computes ``(x_prefix, x_text)`` snapped to grid columns or margins."""
    if grid_xs:
        xs = [float(x) for x in grid_xs]
        idx = max(0, min(int(col), len(xs) - 1))
        x_start = xs[idx]
        if not has_bullet:
            return x_start, x_start
        x_pref = x_start if x_start <= 0.14 or col > 0 else 0.025
        min_text_x = x_pref + 0.095
        x_txt = next((x for x in xs if x >= min_text_x), min_text_x)
        return x_pref, x_txt
    if margin_line:
        x_base = 0.135 + max(0, int(col)) * 0.08
        if has_bullet:
            return x_base, x_base + 0.115
        return x_base, x_base
    indent = max(0, int(col)) * 0.08
    if has_bullet:
        return 0.03 + indent, 0.13 + indent
    return 0.04 + indent, 0.04 + indent


def _prepare_text_axes(ax: maxes.Axes) -> None:
    """Configures a scroll Axes cleanly for notebook text and list entries."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


@dataclasses.dataclass
class ScrollCard:
    """Old-parchment scroll card owning an inner Matplotlib ``Axes``.

    Provides notebook-style line and grid helpers (``line_y``, ``grid_x``,
    ``write_line``, ``write_lines``) plus dynamic grid management
    (``set_background_grid``, ``clear_background_grid``).

    Attributes:
      axes: Child Matplotlib ``Axes`` inside the unrolled parchment area.
      x_mm: Left position in millimeters.
      y_mm: Top position in millimeters.
      w_mm: Total scroll width in millimeters.
      h_mm: Total scroll height in millimeters.
      orientation: ``"portrait"`` (top/bottom rollers) or ``"landscape"``
        (left/right rollers).
      title: Optional banner heading inside the scroll.
      background_grid: Active background pattern key or ``None`` for plain
        parchment.
      grid_spacing: Active vertical grid/rule spacing in axes-fraction units.
      line_ys: Top-to-bottom horizontal line coordinates in axes fraction.
      grid_xs: Left-to-right vertical grid coordinates in axes fraction.
      margin_line: Whether a ruled notebook left margin line is active.
      inner_size_mm: Inner ``Axes`` physical ``(width_mm, height_mm)``.
      ink_color: Primary text and rule ink color.
    """

    axes: maxes.Axes
    x_mm: float
    y_mm: float
    w_mm: float
    h_mm: float
    orientation: str = "portrait"
    title: str | None = None
    background_grid: str | None = None
    grid_spacing: float | None = None
    line_ys: tuple[float, ...] = ()
    grid_xs: tuple[float, ...] = ()
    margin_line: bool = False
    inner_size_mm: tuple[float, float] = (100.0, 100.0)
    ink_color: str = PARCHMENT_INK

    def line_y(self, row: int) -> float:
        """Returns the axes-fraction ``y`` coordinate of notebook line ``row``.

        Row ``0`` is the topmost writable ruled/grid line, increasing downward.
        """
        idx = int(row)
        if self.line_ys:
            if 0 <= idx < len(self.line_ys):
                return self.line_ys[idx]
            step = (
                abs(self.line_ys[0] - self.line_ys[1])
                if len(self.line_ys) >= 2
                else (self.grid_spacing or 0.10)
            )
            if idx < 0:
                return self.line_ys[0] - idx * step
            return self.line_ys[-1] - (idx - len(self.line_ys) + 1) * step
        step = self.grid_spacing or 0.12
        return max(0.05, 0.88 - idx * step)

    def grid_x(self, col: int = 0) -> float:
        """Returns the axes-fraction ``x`` coordinate of grid column ``col``."""
        idx = max(0, int(col))
        if self.grid_xs:
            if idx < len(self.grid_xs):
                return self.grid_xs[idx]
            step = (
                self.grid_xs[1] - self.grid_xs[0]
                if len(self.grid_xs) >= 2
                else 0.08
            )
            return min(
                0.96, self.grid_xs[-1] + (idx - len(self.grid_xs) + 1) * step
            )
        base = 0.135 if self.margin_line else 0.04
        return min(0.96, base + idx * 0.08)

    def set_background_grid(
        self,
        style: str | None,
        *,
        spacing: float | None = None,
        color: str = "#A68452",
        alpha: float = 0.38,
        margin_line: bool = False,
        square_grid: bool = True,
        sketch: tuple[float, float, float] | bool = True,
    ) -> ScrollCard:
        """Applies, switches, or removes (``None`` / ``"none"``) the grid."""
        key = _normalize_grid_style(style)
        if key is None:
            patterns.clear_background_grid(self.axes)
            self.background_grid = None
            self.grid_spacing = None
            self.line_ys = ()
            self.grid_xs = ()
            self.margin_line = False
            return self
        step_y, step_x = _resolve_scroll_grid_spacing(
            key,
            self.inner_size_mm,
            spacing,
            square_grid=square_grid,
        )
        eff_y = 0.08 if step_y is None else step_y
        patterns.apply_background_grid(
            self.axes,
            style=key,
            color=color,
            alpha=alpha,
            spacing=eff_y,
            spacing_x=step_x,
            margin_line=margin_line,
            sketch=sketch,
        )
        raw_xs, raw_ys = patterns.compute_background_grid_coords(
            key, spacing=eff_y, spacing_x=step_x
        )
        self.background_grid = key
        self.grid_spacing = step_y
        self.line_ys = _writable_line_ys(raw_ys, step_y)
        self.grid_xs = tuple(float(x) for x in raw_xs)
        self.margin_line = bool(margin_line and key in ("lines", "ruled"))
        return self

    def clear_background_grid(self) -> ScrollCard:
        """Removes all background lines, grids, or dots from the scroll."""
        return self.set_background_grid(None)

    def write_line(
        self,
        row: int,
        text: str,
        *,
        col: int = 0,
        x: float | None = None,
        prefix: str | None = None,
        color: str | None = None,
        accent_color: str = "#9E3A26",
        fontsize: float = 10.5,
        align_to_grid: bool = True,
        baseline_offset: float | None = None,
    ) -> maxes.Axes:
        """Writes one text line resting cleanly on notebook line ``row``."""
        _prepare_text_axes(self.axes)
        use_baseline = align_to_grid and bool(self.line_ys)
        y_base = self.line_y(row)
        y_pos = (
            y_base + _baseline_lift(self.line_ys, baseline_offset)
            if use_baseline
            else y_base
        )
        va = "baseline" if use_baseline else "center"
        has_pref = bool(prefix)
        x_pref, x_txt = _resolve_horizontal_columns(
            self.grid_xs if align_to_grid else (),
            self.margin_line,
            has_pref,
            col=col,
        )
        if x is not None:
            x_txt = float(x)
            x_pref = float(x)
        ink = self.ink_color if color is None else color
        if prefix:
            self.axes.text(
                x_pref,
                y_pos,
                prefix,
                fontsize=fontsize,
                color=accent_color,
                va=va,
                ha="left",
            )
            if x is not None:
                x_txt = float(x) + 0.10
        self.axes.text(
            x_txt,
            y_pos,
            text,
            fontsize=fontsize,
            color=ink,
            va=va,
            ha="left",
        )
        return self.axes

    def write_lines(
        self,
        items: Sequence[str | tuple[str, str]],
        *,
        start_line: int = 0,
        line_stride: int = 1,
        col: int = 0,
        bullet: str | None = "check",
        ink_color: str | None = None,
        accent_color: str = "#9E3A26",
        fontsize: float = 10.5,
        align_to_grid: bool = True,
    ) -> maxes.Axes:
        """Writes notebook lines or checklist entries on the scroll."""
        return render_scroll_list(
            self.axes,
            items,
            bullet=bullet,
            ink_color=self.ink_color if ink_color is None else ink_color,
            accent_color=accent_color,
            fontsize=fontsize,
            align_to_grid=align_to_grid,
            line_ys=self.line_ys if align_to_grid else None,
            grid_xs=self.grid_xs if align_to_grid else None,
            start_line=start_line,
            line_stride=line_stride,
            col=col,
            margin_line=self.margin_line,
        )


def _draw_spiral_cap(
    ax: maxes.Axes,
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    ink_color: str,
    roller_color: str,
    zorder: float,
) -> None:
    """Draws a rolled spiral end-cap on one tip of a parchment cylinder."""
    cap = patches.Ellipse(
        (cx, cy),
        width=2.0 * rx,
        height=2.0 * ry,
        facecolor=roller_color,
        edgecolor=ink_color,
        linewidth=1.4,
        zorder=zorder,
    )
    ax.add_patch(cap)
    theta = np.linspace(0.0, 3.6 * np.pi, 36)
    frac = theta / (3.6 * np.pi)
    sx = cx + 0.72 * rx * frac * np.cos(theta)
    sy = cy + 0.72 * ry * frac * np.sin(theta)
    ax.plot(sx, sy, color=ink_color, linewidth=1.05, zorder=zorder + 0.2)


def _draw_portrait_scroll_art(
    overlay: maxes.Axes,
    geom: tuple[float, float, float, float, float],
    colors: tuple[str, str, str, str],
) -> tuple[float, float, float, float]:
    """Draws portrait scroll body + top/bottom rollers, returning inner box."""
    x_mm, y_mm, w_mm, h_mm, roll_t = geom
    parch, roll_col, shad_col, ink = colors
    inset = min(4.5, 0.055 * w_mm)
    px = x_mm + inset
    pw = w_mm - 2.0 * inset
    py = y_mm + 0.6 * roll_t
    ph = h_mm - 1.2 * roll_t

    # Parchment sheet with gentle vintage side-edge wave.
    ys_edge = np.linspace(py, py + ph, 24)
    wave_l = px + 0.55 * np.sin(np.linspace(0.0, 2.5 * np.pi, 24))
    wave_r = (px + pw) + 0.55 * np.cos(np.linspace(0.0, 2.2 * np.pi, 24))
    poly_pts = np.vstack(
        [
            np.column_stack([wave_l, ys_edge]),
            np.column_stack([wave_r[::-1], ys_edge[::-1]]),
        ]
    )
    overlay.add_patch(
        patches.Polygon(
            poly_pts,
            closed=True,
            facecolor=parch,
            edgecolor=ink,
            linewidth=1.6,
            zorder=2,
        )
    )
    # Inner fold shadows under top roller and above bottom roller.
    for sy in (py + 0.4 * roll_t, py + ph - 0.85 * roll_t):
        overlay.add_patch(
            patches.Rectangle(
                (px + 0.8, sy),
                pw - 1.6,
                0.48 * roll_t,
                facecolor=shad_col,
                edgecolor="none",
                alpha=0.35,
                zorder=2.4,
            )
        )
    # Top and bottom rolled cylinders + spiral tips.
    for ry_top in (y_mm, y_mm + h_mm - roll_t):
        overlay.add_patch(
            patches.FancyBboxPatch(
                (x_mm + 1.8, ry_top),
                w_mm - 3.6,
                roll_t,
                boxstyle="round,pad=0.1,rounding_size=1.4",
                facecolor=roll_col,
                edgecolor=ink,
                linewidth=1.6,
                zorder=3,
            )
        )
        cy = ry_top + 0.5 * roll_t
        cap_rx = min(2.4, 0.34 * roll_t)
        cap_ry = 0.5 * roll_t
        _draw_spiral_cap(
            overlay, x_mm + 1.8, cy, cap_rx, cap_ry, ink, roll_col, 3.3
        )
        _draw_spiral_cap(
            overlay,
            x_mm + w_mm - 1.8,
            cy,
            cap_rx,
            cap_ry,
            ink,
            roll_col,
            3.3,
        )
    return (px + 2.5, y_mm + roll_t + 2.2, pw - 5.0, h_mm - 2.0 * roll_t - 4.4)


def _draw_landscape_scroll_art(
    overlay: maxes.Axes,
    geom: tuple[float, float, float, float, float],
    colors: tuple[str, str, str, str],
) -> tuple[float, float, float, float]:
    """Draws landscape scroll body + left/right rollers, returning inner box."""
    x_mm, y_mm, w_mm, h_mm, roll_t = geom
    parch, roll_col, shad_col, ink = colors
    inset = min(4.2, 0.065 * h_mm)
    px = x_mm + 0.6 * roll_t
    pw = w_mm - 1.2 * roll_t
    py = y_mm + inset
    ph = h_mm - 2.0 * inset

    xs_edge = np.linspace(px, px + pw, 26)
    wave_t = py + 0.50 * np.sin(np.linspace(0.0, 2.8 * np.pi, 26))
    wave_b = (py + ph) + 0.50 * np.cos(np.linspace(0.0, 2.4 * np.pi, 26))
    poly_pts = np.vstack(
        [
            np.column_stack([xs_edge, wave_t]),
            np.column_stack([xs_edge[::-1], wave_b[::-1]]),
        ]
    )
    overlay.add_patch(
        patches.Polygon(
            poly_pts,
            closed=True,
            facecolor=parch,
            edgecolor=ink,
            linewidth=1.6,
            zorder=2,
        )
    )
    for sx in (px + 0.4 * roll_t, px + pw - 0.85 * roll_t):
        overlay.add_patch(
            patches.Rectangle(
                (sx, py + 0.8),
                0.48 * roll_t,
                ph - 1.6,
                facecolor=shad_col,
                edgecolor="none",
                alpha=0.35,
                zorder=2.4,
            )
        )
    for rx_left in (x_mm, x_mm + w_mm - roll_t):
        overlay.add_patch(
            patches.FancyBboxPatch(
                (rx_left, y_mm + 1.8),
                roll_t,
                h_mm - 3.6,
                boxstyle="round,pad=0.1,rounding_size=1.4",
                facecolor=roll_col,
                edgecolor=ink,
                linewidth=1.6,
                zorder=3,
            )
        )
        cx = rx_left + 0.5 * roll_t
        cap_rx = 0.5 * roll_t
        cap_ry = min(2.4, 0.34 * roll_t)
        _draw_spiral_cap(
            overlay, cx, y_mm + 1.8, cap_rx, cap_ry, ink, roll_col, 3.3
        )
        _draw_spiral_cap(
            overlay,
            cx,
            y_mm + h_mm - 1.8,
            cap_rx,
            cap_ry,
            ink,
            roll_col,
            3.3,
        )
    return (x_mm + roll_t + 2.5, py + 2.2, w_mm - 2.0 * roll_t - 5.0, ph - 4.4)


def _compute_list_ys(
    n_items: int,
    line_ys: Sequence[float] | None,
    align_to_grid: bool,
    start_line: int,
    line_stride: int,
    baseline_offset: float | None,
) -> tuple[list[float], str]:
    """Computes row y-coordinates and vertical alignment for scroll entries."""
    if align_to_grid and line_ys:
        lines = [float(y) for y in line_ys]
        stride = max(1, int(line_stride))
        step = (
            abs(lines[0] - lines[1])
            if len(lines) >= 2
            else max(0.08, 1.0 / max(n_items + 1, 2))
        )
        lift = _baseline_lift(lines, baseline_offset)
        ys_out: list[float] = []
        for idx in range(n_items):
            row = int(start_line) + idx * stride
            if 0 <= row < len(lines):
                y_rule = lines[row]
            elif row < 0:
                y_rule = lines[0] - row * step
            else:
                y_rule = lines[-1] - (row - len(lines) + 1) * step
            ys_out.append(y_rule + lift)
        return ys_out, "baseline"
    if n_items > 1:
        return [float(y) for y in np.linspace(0.86, 0.14, n_items)], "center"
    return [0.75], "center"


_BULLET_MARKERS: dict[str, str] = {
    "check": "[ ]",
    "checked": "[x]",
    "dot": "*",
    "star": "+",
}


def _format_list_entry(entry: str | tuple[str, str]) -> str:
    """Formats a scroll list item string or ``(heading, detail)`` pair."""
    if isinstance(entry, tuple):
        return f"{entry[0]}:  {entry[1]}"
    return str(entry)


def render_scroll_list(
    ax: maxes.Axes,
    items: Sequence[str | tuple[str, str]],
    *,
    bullet: str | None = "check",
    ink_color: str = PARCHMENT_INK,
    accent_color: str = "#9E3A26",
    fontsize: float = 10.5,
    align_to_grid: bool = True,
    line_ys: Sequence[float] | None = None,
    grid_xs: Sequence[float] | None = None,
    start_line: int = 0,
    line_stride: int = 1,
    col: int = 0,
    margin_line: bool = False,
    baseline_offset: float | None = None,
) -> maxes.Axes:
    """Renders formatted checklist or notebook entries onto a scroll's Axes.

    When ``align_to_grid=True`` and ``line_ys`` are supplied (automatically
    wired by ``plot_scroll_card`` and ``Board.add_scroll_card``), each entry
    rests with ``va="baseline"`` right on top of its ruled/grid line—and snaps
    horizontally to grid columns on graph/dot-grid backgrounds—just like
    writing in a normal lined or graph notebook.

    Args:
      ax: Target scroll ``Axes``.
      items: List of strings or ``(label, detail)`` pairs.
      bullet: Marker style: ``"check"``, ``"checked"``, ``"dot"``, ``"number"``,
        ``"star"``, or ``"none"`` / ``None`` for plain notebook lines.
      ink_color: Primary text color.
      accent_color: Highlight color for item headings / markers.
      fontsize: Entry font size in points.
      align_to_grid: Whether to snap text baselines to ``line_ys`` and columns
        to ``grid_xs`` when present.
      line_ys: Optional top-to-bottom horizontal rule/grid coordinates.
      grid_xs: Optional left-to-right vertical grid column coordinates.
      start_line: Zero-based top line index at which to begin writing.
      line_stride: Integer rule step between consecutive items (``1`` single-
        spaced, ``2`` double-spaced).
      col: Zero-based vertical grid column / indent level.
      margin_line: Whether a ruled notebook left margin rule is visible.
      baseline_offset: Optional axes-fraction lift above each horizontal rule.

    Returns:
      The populated ``Axes``.
    """
    _prepare_text_axes(ax)
    if not items:
        return ax

    ys, va = _compute_list_ys(
        len(items),
        line_ys,
        align_to_grid,
        start_line,
        line_stride,
        baseline_offset,
    )
    bullet_key = None if bullet is None else bullet.strip().lower()
    has_bullet = bullet_key not in (None, "", "none")
    x_pref, x_txt = _resolve_horizontal_columns(
        grid_xs if (align_to_grid and grid_xs) else (),
        margin_line,
        has_bullet,
        col=col,
    )
    for idx, (entry, y_pos) in enumerate(zip(items, ys, strict=True)):
        if has_bullet:
            prefix = (
                f"{idx + 1}."
                if bullet_key == "number"
                else _BULLET_MARKERS.get(bullet_key or "", "*")
            )
            ax.text(
                x_pref,
                y_pos,
                prefix,
                fontsize=fontsize,
                color=accent_color,
                va=va,
                ha="left",
            )
        ax.text(
            x_txt,
            y_pos,
            _format_list_entry(entry),
            fontsize=fontsize,
            color=ink_color,
            va=va,
            ha="left",
        )
    return ax


def _draw_scroll_heading(
    overlay: maxes.Axes,
    inner_box: tuple[float, float, float, float],
    title: str | None,
    subtitle: str | None,
    ink_color: str,
) -> float:
    """Draws optional scroll title banner and subtitle, returning top offset."""
    ix, iy, iw, _ = inner_box
    title_h = 0.0
    if title:
        overlay.text(
            ix + 0.5 * iw,
            iy + 3.2,
            title,
            fontsize=12.5,
            color=ink_color,
            va="top",
            ha="center",
            zorder=5,
        )
        overlay.plot(
            [ix + 0.14 * iw, ix + 0.86 * iw],
            [iy + 10.0, iy + 10.0],
            color=ink_color,
            linewidth=1.1,
            alpha=0.65,
            zorder=5,
        )
        title_h = 11.5
    if subtitle:
        overlay.text(
            ix + 0.5 * iw,
            iy + title_h + 0.8,
            subtitle,
            fontsize=9.2,
            color="#6D5039",
            va="top",
            ha="center",
            zorder=5,
        )
        title_h += 5.5
    return title_h


def _apply_initial_scroll_grid(
    child_ax: maxes.Axes,
    inner_size_mm: tuple[float, float],
    grid_spec: tuple[str | None, float | None, int, bool],
) -> tuple[
    str | None,
    float | None,
    tuple[float, ...],
    tuple[float, ...],
    bool,
]:
    """Applies initial parchment grid and returns resolved grid metadata."""
    background_grid, grid_spacing, min_rows, margin_line = grid_spec
    bg_key = _normalize_grid_style(background_grid)
    step_y, step_x = _resolve_scroll_grid_spacing(
        bg_key, inner_size_mm, grid_spacing, min_rows=min_rows
    )
    if bg_key is None:
        return None, None, (), (), False
    eff_y = 0.08 if step_y is None else step_y
    patterns.apply_background_grid(
        child_ax,
        style=bg_key,
        color="#A68452",
        alpha=0.38,
        spacing=eff_y,
        spacing_x=step_x,
        margin_line=margin_line,
        sketch=True,
    )
    raw_xs, raw_ys = patterns.compute_background_grid_coords(
        bg_key, spacing=eff_y, spacing_x=step_x
    )
    return (
        bg_key,
        step_y,
        _writable_line_ys(raw_ys, step_y),
        tuple(float(x) for x in raw_xs),
        bool(margin_line and bg_key in ("lines", "ruled")),
    )


def _setup_scroll_axes(
    fig: figure.Figure,
    board_dims: tuple[float, float],
    inner_box: tuple[float, float, float, float],
    pad_mm: tuple[float, float, float, float],
    parch_color: str,
) -> tuple[maxes.Axes, tuple[float, float]]:
    """Creates the inner scroll ``Axes`` and returns ``(ax, size_mm)``."""
    ax_w_mm = max(5.0, inner_box[2] - pad_mm[0] - pad_mm[1])
    ax_h_mm = max(5.0, inner_box[3] - pad_mm[2] - pad_mm[3])
    child_ax = fig.add_axes(
        [
            (inner_box[0] + pad_mm[0]) / board_dims[0],
            (board_dims[1] - (inner_box[1] + inner_box[3] - pad_mm[3]))
            / board_dims[1],
            max(0.05, ax_w_mm / board_dims[0]),
            max(0.05, ax_h_mm / board_dims[1]),
        ]
    )
    child_ax.set_facecolor(parch_color)
    child_ax.patch.set_sketch_params(0.0, 0.0, 0.0)
    return child_ax, (ax_w_mm, ax_h_mm)


def draw_scroll_card_on_board(
    fig: figure.Figure,
    overlay: maxes.Axes,
    board_dims: tuple[float, float],
    rect_mm: tuple[float, float, float, float],
    *,
    orientation: str = "portrait",
    title: str | None = None,
    subtitle: str | None = None,
    items: Sequence[str | tuple[str, str]] | None = None,
    bullet: str | None = "check",
    scroll_colors: tuple[str, str, str] | None = None,
    background_grid: str | None = None,
    align_to_grid: bool = True,
    grid_spacing: float | None = None,
    preset: str | theme.ThemePreset = "classic",
    inner_pad_mm: tuple[float, float, float, float] | None = None,
    notebook_opts: tuple[int, bool] = (1, False),
) -> ScrollCard:
    """Draws an old-parchment scroll card on a millimeter overlay canvas."""
    orient = orientation.strip().lower()
    if orient not in ("portrait", "landscape"):
        raise exceptions.StyleError(
            f"Unsupported scroll orientation {orientation!r}. "
            "Use 'portrait' or 'landscape'."
        )
    colors = (
        (PARCHMENT_LIGHT, PARCHMENT_ROLLER, PARCHMENT_INK)
        if scroll_colors is None
        else scroll_colors
    )
    roll_t = (
        max(5.5, min(11.0, 0.085 * rect_mm[3]))
        if orient == "portrait"
        else max(5.5, min(11.5, 0.075 * rect_mm[2]))
    )
    min_rows = (
        (len(items) - 1) * max(1, int(notebook_opts[0])) + 1 if items else 0
    )
    with theme.xkcd_style(preset=preset):
        draw_fn = (
            _draw_portrait_scroll_art
            if orient == "portrait"
            else _draw_landscape_scroll_art
        )
        inner_box = draw_fn(
            overlay,
            (*rect_mm, roll_t),
            (colors[0], colors[1], PARCHMENT_SHADOW, colors[2]),
        )
        title_h = _draw_scroll_heading(
            overlay, inner_box, title, subtitle, colors[2]
        )
        pad_mm = (
            (3.0, 3.0, title_h + 1.0, 2.5)
            if inner_pad_mm is None
            else inner_pad_mm
        )
        child_ax, inner_size_mm = _setup_scroll_axes(
            fig, board_dims, inner_box, pad_mm, colors[0]
        )
        grid_meta = _apply_initial_scroll_grid(
            child_ax,
            inner_size_mm,
            (background_grid, grid_spacing, min_rows, notebook_opts[1]),
        )
        if items is not None:
            render_scroll_list(
                child_ax,
                items,
                bullet=bullet,
                ink_color=colors[2],
                align_to_grid=align_to_grid,
                line_ys=grid_meta[2],
                grid_xs=grid_meta[3],
                line_stride=notebook_opts[0],
                margin_line=grid_meta[4],
            )
    return ScrollCard(
        axes=child_ax,
        x_mm=rect_mm[0],
        y_mm=rect_mm[1],
        w_mm=rect_mm[2],
        h_mm=rect_mm[3],
        orientation=orient,
        title=title,
        background_grid=grid_meta[0],
        grid_spacing=grid_meta[1],
        line_ys=grid_meta[2],
        grid_xs=grid_meta[3],
        margin_line=grid_meta[4],
        inner_size_mm=inner_size_mm,
        ink_color=colors[2],
    )


def plot_scroll_card(
    title: str | None = None,
    *,
    subtitle: str | None = None,
    items: Sequence[str | tuple[str, str]] | None = None,
    orientation: str = "portrait",
    bullet: str | None = "check",
    parchment_color: str = PARCHMENT_LIGHT,
    roller_color: str = PARCHMENT_ROLLER,
    ink_color: str = PARCHMENT_INK,
    background_grid: str | None = None,
    align_to_grid: bool = True,
    grid_spacing: float | None = None,
    line_stride: int = 1,
    margin_line: bool = False,
    figsize_mm: tuple[float, float] | None = None,
    dpi: int = 160,
    preset: str | theme.ThemePreset = "classic",
) -> tuple[figure.Figure, ScrollCard]:
    """Creates a standalone old-parchment scroll card figure.

    Ideal for checklists, lab notes, synthesis recipes, maps, or hosting any
    ``xkcd_matchem`` plot on ``card.axes`` in portrait or landscape format.
    When ``background_grid`` is set to a notebook/grid paper style (``"ruled"``,
    ``"lines"``, ``"grid"``, ``"graph"``, ``"dots"``, ``"dotgrid"``) and
    ``align_to_grid=True``, text baselines sit cleanly on top of the drawn
    lines (and align to vertical grid columns on graph/dot-grid backgrounds).
    Pass ``background_grid=None`` or ``"none"`` (or call
    ``card.clear_background_grid()``) for clean plain parchment without lines.

    Args:
      title: Optional heading centered under the top scroll edge.
      subtitle: Optional secondary line below ``title``.
      items: Optional sequence of strings or ``(heading, detail)`` entries.
      orientation: ``"portrait"`` or ``"landscape"``.
      bullet: List bullet marker (``"check"``, ``"checked"``, ``"dot"``,
        ``"number"``, ``"star"``, or ``"none"`` / ``None``).
      parchment_color: Fill color for the parchment body.
      roller_color: Fill color for the rolled cylinders and end-caps.
      ink_color: Stroke and text color.
      background_grid: Optional subtle paper grid on the parchment
        (``"ruled"``, ``"graph"``, ``"dotgrid"``, or ``None`` / ``"none"``
        for plain parchment without lines).
      align_to_grid: Whether to align text baselines/columns to the active
        notebook lines or grid (default ``True``).
      grid_spacing: Optional vertical rule/grid spacing in axes-fraction units
        (defaults to notebook-scale physical line pitch).
      line_stride: Rule step between consecutive list items (``1`` single-
        spaced, ``2`` double-spaced).
      margin_line: Whether ``"ruled"`` / ``"lines"`` draws a classic left
        margin rule.
      figsize_mm: Optional ``(width_mm, height_mm)`` override.
      dpi: Output figure DPI.
      preset: Visual theme preset name or ``ThemePreset``.

    Returns:
      Tuple ``(fig, scroll_card)`` where ``scroll_card`` exposes ``.axes``,
      ``.write_line()``, ``.write_lines()``, and ``.clear_background_grid()``.
    """
    orient = orientation.strip().lower()
    if figsize_mm is None:
        w_mm, h_mm = (125.0, 165.0) if orient == "portrait" else (185.0, 118.0)
    else:
        w_mm, h_mm = float(figsize_mm[0]), float(figsize_mm[1])
    preset_obj = theme.get_preset(preset)
    with theme.xkcd_style(preset=preset):
        fig = plt.figure(
            figsize=(w_mm / 25.4, h_mm / 25.4),
            dpi=dpi,
            facecolor=preset_obj.figure_facecolor,
        )
        fig.patch.set_sketch_params(0.0, 0.0, 0.0)
        overlay = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        overlay.set_xlim(0.0, w_mm)
        overlay.set_ylim(h_mm, 0.0)
        overlay.set_axis_off()
        card = draw_scroll_card_on_board(
            fig,
            overlay,
            (w_mm, h_mm),
            (4.0, 4.0, w_mm - 8.0, h_mm - 8.0),
            orientation=orient,
            title=title,
            subtitle=subtitle,
            items=items,
            bullet=bullet,
            scroll_colors=(parchment_color, roller_color, ink_color),
            background_grid=background_grid,
            align_to_grid=align_to_grid,
            grid_spacing=grid_spacing,
            preset=preset,
            notebook_opts=(line_stride, margin_line),
        )
    return fig, card
