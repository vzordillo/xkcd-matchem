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


@dataclasses.dataclass
class ScrollCard:
    """Old-parchment scroll card owning an inner Matplotlib ``Axes``.

    Attributes:
      axes: Child Matplotlib ``Axes`` inside the unrolled parchment area.
      x_mm: Left position in millimeters.
      y_mm: Top position in millimeters.
      w_mm: Total scroll width in millimeters.
      h_mm: Total scroll height in millimeters.
      orientation: ``"portrait"`` (top/bottom rollers) or ``"landscape"``
        (left/right rollers).
      title: Optional banner heading inside the scroll.
    """

    axes: maxes.Axes
    x_mm: float
    y_mm: float
    w_mm: float
    h_mm: float
    orientation: str = "portrait"
    title: str | None = None


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


def render_scroll_list(
    ax: maxes.Axes,
    items: Sequence[str | tuple[str, str]],
    *,
    bullet: str = "check",
    ink_color: str = PARCHMENT_INK,
    accent_color: str = "#9E3A26",
    fontsize: float = 10.5,
) -> maxes.Axes:
    """Renders formatted checklist or bullet entries onto a scroll's Axes.

    Args:
      ax: Target scroll ``Axes``.
      items: List of strings or ``(label, detail)`` pairs.
      bullet: Marker style: ``"check"``, ``"checked"``, ``"dot"``, ``"number"``,
        or ``"star"``.
      ink_color: Primary text color.
      accent_color: Highlight color for item headings / markers.
      fontsize: Entry font size in points.

    Returns:
      The populated ``Axes``.
    """
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    n_items = max(len(items), 1)
    ys = np.linspace(0.90, 0.12, n_items) if n_items > 1 else np.array([0.75])
    marker_map = {
        "check": "[ ]",
        "checked": "[x]",
        "dot": "*",
        "star": "+",
    }
    for idx, (entry, y_pos) in enumerate(zip(items, ys, strict=True)):
        prefix = (
            f"{idx + 1}." if bullet == "number" else marker_map.get(bullet, "*")
        )
        if isinstance(entry, tuple):
            heading, detail = entry
            text = f"{heading}:  {detail}"
        else:
            text = str(entry)
        ax.text(
            0.03,
            float(y_pos),
            prefix,
            fontsize=fontsize,
            color=accent_color,
            va="center",
            ha="left",
        )
        ax.text(
            0.13,
            float(y_pos),
            text,
            fontsize=fontsize,
            color=ink_color,
            va="center",
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


def _setup_scroll_axes(
    fig: figure.Figure,
    board_dims: tuple[float, float],
    inner_box: tuple[float, float, float, float],
    pad_mm: tuple[float, float, float, float],
    parch_color: str,
    background_grid: str | None,
) -> maxes.Axes:
    """Creates and textures the inner Matplotlib ``Axes`` inside a scroll."""
    board_w, board_h = board_dims
    ix, iy, iw, ih = inner_box
    pad_l, pad_r, pad_t, pad_b = pad_mm
    child_ax = fig.add_axes(
        [
            (ix + pad_l) / board_w,
            (board_h - (iy + ih - pad_b)) / board_h,
            max(0.05, (iw - pad_l - pad_r) / board_w),
            max(0.05, (ih - pad_t - pad_b) / board_h),
        ]
    )
    child_ax.set_facecolor(parch_color)
    child_ax.patch.set_sketch_params(0.0, 0.0, 0.0)
    if background_grid:
        patterns.apply_background_grid(
            child_ax,
            style=background_grid,
            color="#B09060",
            alpha=0.28,
            margin_line=False,
        )
    return child_ax


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
    bullet: str = "check",
    scroll_colors: tuple[str, str, str] | None = None,
    background_grid: str | None = None,
    preset: str | theme.ThemePreset = "classic",
    inner_pad_mm: tuple[float, float, float, float] | None = None,
) -> ScrollCard:
    """Draws an old-parchment scroll card on a millimeter overlay canvas."""
    orient = orientation.strip().lower()
    if orient not in ("portrait", "landscape"):
        raise exceptions.StyleError(
            f"Unsupported scroll orientation {orientation!r}. "
            "Use 'portrait' or 'landscape'."
        )
    parch, roll_col, ink_col = (
        (PARCHMENT_LIGHT, PARCHMENT_ROLLER, PARCHMENT_INK)
        if scroll_colors is None
        else scroll_colors
    )
    roll_t = (
        max(5.5, min(11.0, 0.085 * rect_mm[3]))
        if orient == "portrait"
        else max(5.5, min(11.5, 0.075 * rect_mm[2]))
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
            (parch, roll_col, PARCHMENT_SHADOW, ink_col),
        )
        title_h = _draw_scroll_heading(
            overlay, inner_box, title, subtitle, ink_col
        )
        pad_mm = (
            (3.0, 3.0, title_h + 1.0, 2.5)
            if inner_pad_mm is None
            else inner_pad_mm
        )
        child_ax = _setup_scroll_axes(
            fig, board_dims, inner_box, pad_mm, parch, background_grid
        )
        if items is not None:
            render_scroll_list(
                child_ax, items, bullet=bullet, ink_color=ink_col
            )
    return ScrollCard(
        axes=child_ax,
        x_mm=rect_mm[0],
        y_mm=rect_mm[1],
        w_mm=rect_mm[2],
        h_mm=rect_mm[3],
        orientation=orient,
        title=title,
    )


def plot_scroll_card(
    title: str | None = None,
    *,
    subtitle: str | None = None,
    items: Sequence[str | tuple[str, str]] | None = None,
    orientation: str = "portrait",
    bullet: str = "check",
    parchment_color: str = PARCHMENT_LIGHT,
    roller_color: str = PARCHMENT_ROLLER,
    ink_color: str = PARCHMENT_INK,
    background_grid: str | None = None,
    figsize_mm: tuple[float, float] | None = None,
    dpi: int = 160,
    preset: str | theme.ThemePreset = "classic",
) -> tuple[figure.Figure, ScrollCard]:
    """Creates a standalone old-parchment scroll card figure.

    Ideal for checklists, lab notes, synthesis recipes, maps, or hosting any
    ``xkcd_matchem`` plot on ``card.axes`` in portrait or landscape format.

    Args:
      title: Optional heading centered under the top scroll edge.
      subtitle: Optional secondary line below ``title``.
      items: Optional sequence of strings or ``(heading, detail)`` entries.
      orientation: ``"portrait"`` or ``"landscape"``.
      bullet: List bullet marker (``"check"``, ``"checked"``, ``"dot"``,
        ``"number"``, ``"star"``).
      parchment_color: Fill color for the parchment body.
      roller_color: Fill color for the rolled cylinders and end-caps.
      ink_color: Stroke and text color.
      background_grid: Optional subtle grid on the parchment (e.g. ``"ruled"``,
        ``"dots"``, ``"grid"``).
      figsize_mm: Optional ``(width_mm, height_mm)`` override.
      dpi: Output figure DPI.
      preset: Visual theme preset name or ``ThemePreset``.

    Returns:
      Tuple ``(fig, scroll_card)`` where ``scroll_card.axes`` accepts plots.
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
            preset=preset,
        )
    return fig, card
