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
"""Turnkey science-communication layout recipes for papers, talks & comics."""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import math
import pathlib
import textwrap

from matplotlib import axes as maxes
from matplotlib import figure
from matplotlib import patches
from matplotlib import pyplot as plt

from xkcd_matchem.core import exceptions
from xkcd_matchem.layout import canvas
from xkcd_matchem.layout import comic
from xkcd_matchem.layout import export
from xkcd_matchem.render import primitives
from xkcd_matchem.style import palettes
from xkcd_matchem.style import theme


_POSTER_SIZES_MM: dict[str, tuple[float, float]] = {
    "a0_landscape": (1189.0, 841.0),
    "a0_portrait": (841.0, 1189.0),
    "compact_board": (360.0, 240.0),
}

_POSTER_HEADER_COLORS: tuple[str, ...] = (
    "#56B4E9",
    "#009E73",
    "#E69F00",
    "#CC79A7",
    "#0072B2",
    "#D55E00",
)

_ARROW_ONLY_TOKENS = frozenset({"->", "=>", "\u2192"})


def _resolve_banner_colors(
    preset_obj: theme.ThemePreset,
) -> tuple[str, str, str]:
    """Returns ``(accent_fill, accent_rule, muted_ink)`` for light or dark."""
    if primitives.is_dark_color(preset_obj.figure_facecolor):
        return "#23364D", "#56B4E9", "#B8C2CC"
    return "#EDF4FA", "#0072B2", palettes.MUTED_GRAY


def _draw_rounded_strip(
    ax: maxes.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    facecolor: str,
    edgecolor: str,
    linewidth: float,
    rounding: float,
    pad: float = 0.0,
    zorder: float = 2.0,
) -> patches.FancyBboxPatch:
    """Draws a rounded strip or banner patch onto ``ax``."""
    box = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad={pad},rounding_size={rounding}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
    )
    ax.add_patch(box)
    return box


def _draw_gap_arrow(
    ax: maxes.Axes,
    label: str,
    preset_obj: theme.ThemePreset,
    sw: theme.StrokeHierarchy,
    fontsize: float,
) -> None:
    """Draws a crisp workflow arrow and optional tag inside a gap axes."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_axis_off()
    ax.patch.set_visible(False)
    ax.annotate(
        "",
        xy=(0.92, 0.50),
        xytext=(0.08, 0.50),
        arrowprops={
            "arrowstyle": "-|>,head_width=0.34,head_length=0.45",
            "color": preset_obj.ink_color,
            "lw": sw.primary * 1.15,
            "shrinkA": 0.0,
            "shrinkB": 0.0,
        },
        zorder=5.0,
    )
    clean_lbl = label.strip()
    if clean_lbl and clean_lbl not in _ARROW_ONLY_TOKENS:
        theme.place_halo_text(
            ax,
            0.50,
            0.64,
            clean_lbl,
            color=preset_obj.ink_color,
            bg_color=preset_obj.figure_facecolor,
            fontsize=fontsize,
            va="bottom",
            zorder=6.0,
        )


@dataclasses.dataclass(frozen=True)
class GraphicalAbstractLayout:
    """Turnkey journal Table-of-Contents / graphical abstract canvas.

    Attributes:
      figure: Underlying Matplotlib ``Figure`` sized in millimeters.
      axes: Ordered dictionary mapping panel keys (e.g. ``"left"``,
        ``"right"``) to Matplotlib ``Axes``.
      left_ax: Primary left-most content panel ``Axes``.
      right_ax: Right-most content panel ``Axes`` when ``panels >= 2``.
      arrow_ax: Dedicated central workflow-arrow ``Axes`` when present.
      center_ax: Center panel ``Axes`` when ``panels == 3``.
      overlay_ax: Full-sheet millimeter overlay ``Axes`` holding title/strip.
      size_mm: Canvas ``(width_mm, height_mm)`` dimensions.
      dpi: Default export resolution.
      preset: Visual theme preset applied to the layout.
    """

    figure: figure.Figure
    axes: dict[str, maxes.Axes]
    left_ax: maxes.Axes
    right_ax: maxes.Axes | None = None
    arrow_ax: maxes.Axes | None = None
    center_ax: maxes.Axes | None = None
    overlay_ax: maxes.Axes | None = None
    size_mm: tuple[float, float] = (130.0, 55.0)
    dpi: int = 300
    preset: str | theme.ThemePreset = "graphical_abstract"

    def save(
        self,
        path: str | pathlib.Path,
        dpi: int | None = None,
        *,
        exact_mm: bool = True,
    ) -> pathlib.Path:
        """Exports the graphical abstract figure to PNG, SVG, or PDF."""
        return export.save_figure(
            self.figure,
            path,
            dpi=self.dpi if dpi is None else dpi,
            tight=not exact_mm,
        )


def _ga_panel_keys(n_panels: int) -> list[str]:
    """Returns canonical dictionary keys for ``n_panels`` GA panels."""
    if n_panels == 1:
        return ["main"]
    if n_panels == 2:
        return ["left", "right"]
    if n_panels == 3:
        return ["left", "center", "right"]
    return [f"panel_{i + 1}" for i in range(n_panels)]


def _draw_ga_chrome(
    overlay: maxes.Axes,
    w_mm: float,
    h_mm: float,
    title: str | None,
    takeaway: str | None,
    preset_obj: theme.ThemePreset,
    sw: theme.StrokeHierarchy,
) -> tuple[float, float]:
    """Draws optional GA top title and bottom takeaway strip, returning pads."""
    accent_fill, _, _ = _resolve_banner_colors(preset_obj)
    top_mm = min(14.0, max(11.8, 0.215 * h_mm)) if title else 4.5
    strip_h_mm = min(8.6, max(6.0, 0.11 * h_mm)) if takeaway else 0.0
    bot_mm = (
        strip_h_mm + max(14.6, 0.265 * h_mm)
        if takeaway
        else max(7.0, 0.125 * h_mm)
    )

    if title:
        title_fs = min(preset_obj.title_size, max(8.5, 0.175 * h_mm))
        overlay.text(
            0.5 * w_mm,
            1.2,
            title,
            fontsize=title_fs,
            color=preset_obj.ink_color,
            ha="center",
            va="top",
            zorder=5.0,
        )
    if takeaway:
        pad_x = 4.5
        strip_y = h_mm - strip_h_mm - 1.0
        _draw_rounded_strip(
            overlay,
            pad_x,
            strip_y,
            w_mm - 2.0 * pad_x,
            strip_h_mm,
            facecolor=accent_fill,
            edgecolor=preset_obj.ink_color,
            linewidth=sw.overlay,
            rounding=1.6,
            zorder=2.0,
        )
        tk_fs = min(preset_obj.font_size * 0.84, max(6.8, 0.14 * h_mm))
        overlay.text(
            0.5 * w_mm,
            strip_y + 0.5 * strip_h_mm,
            takeaway,
            fontsize=tk_fs,
            color=preset_obj.ink_color,
            ha="center",
            va="center",
            zorder=5.0,
        )
    return top_mm, bot_mm


def _populate_ga_panels(
    fig: figure.Figure,
    *,
    panels: int,
    w_mm: float,
    h_mm: float,
    top_mm: float,
    bot_mm: float,
    arrow_label: str | None,
    preset: str | theme.ThemePreset,
) -> tuple[list[maxes.Axes], maxes.Axes | None]:
    """Creates panel axes and inter-panel workflow arrows for GA canvas."""
    preset_obj = theme.get_preset(preset)
    sw = theme.stroke_weights(preset)
    left_mm = min(11.0, max(6.5, 0.07 * w_mm))
    right_mm = min(6.5, max(4.0, 0.038 * w_mm))
    usable_w = max(20.0, w_mm - left_mm - right_mm)
    has_arrow = panels >= 2 and bool(arrow_label)
    gap_mm = (
        min(24.0 if has_arrow else 11.0, 0.30 * usable_w / (panels - 1))
        if panels >= 2
        else 0.0
    )
    panel_w_mm = (usable_w - (panels - 1) * gap_mm) / panels
    panel_h_mm = max(8.0, h_mm - top_mm - bot_mm)
    tick_fs = max(6.8, min(preset_obj.font_size * 0.72, 0.145 * h_mm))
    sub_fs = max(8.0, min(preset_obj.title_size * 0.74, 0.165 * h_mm))

    panel_axes: list[maxes.Axes] = []
    for idx in range(panels):
        x0 = left_mm + idx * (panel_w_mm + gap_mm)
        ax_p = fig.add_axes(
            (x0 / w_mm, bot_mm / h_mm, panel_w_mm / w_mm, panel_h_mm / h_mm)
        )
        theme.style_existing_axes(ax_p, preset=preset)
        ax_p.tick_params(labelsize=tick_fs)
        ax_p.xaxis.label.set_fontsize(tick_fs)
        ax_p.yaxis.label.set_fontsize(tick_fs)
        ax_p.title.set_fontsize(sub_fs)
        panel_axes.append(ax_p)

    arrow_ax: maxes.Axes | None = None
    if has_arrow and arrow_label is not None:
        arr_fs = max(6.8, min(preset_obj.font_size * 0.72, 0.145 * h_mm))
        for idx in range(panels - 1):
            gx0 = left_mm + (idx + 1) * panel_w_mm + idx * gap_mm
            arr_w = max(6.0, 0.44 * gap_mm)
            ax_gap = fig.add_axes(
                (
                    (gx0 + 0.05 * gap_mm) / w_mm,
                    (bot_mm + 0.12 * panel_h_mm) / h_mm,
                    arr_w / w_mm,
                    (0.76 * panel_h_mm) / h_mm,
                )
            )
            _draw_gap_arrow(ax_gap, arrow_label, preset_obj, sw, arr_fs)
            if arrow_ax is None:
                arrow_ax = ax_gap
    return panel_axes, arrow_ax


def create_graphical_abstract(
    title: str | None = None,
    takeaway: str | None = None,
    panels: int = 2,
    size_mm: tuple[float, float] = (130.0, 55.0),
    arrow_label: str | None = "\u2192",
    preset: str = "graphical_abstract",
    dpi: int = 300,
) -> GraphicalAbstractLayout:
    """Creates a millimeter-sized journal TOC / graphical abstract layout.

    Args:
      title: Optional centered header string along the top margin.
      takeaway: Optional bold ink summary strip along the bottom edge.
      panels: Number of horizontal content panels (``1``, ``2``, ``3``, ...).
      size_mm: Physical ``(width_mm, height_mm)`` sheet size.
      arrow_label: Workflow/reaction arrow between adjacent panels (``"→"``
        for an unlabeled arrow, any custom text for a labeled arrow, or
        ``None`` to omit the central arrow).
      preset: Visual theme preset (default ``"graphical_abstract"``).
      dpi: Figure resolution in dots per inch (default ``300``).

    Returns:
      Configured ``GraphicalAbstractLayout`` ready for ``xkcd_matchem`` plots.

    Raises:
      StyleError: If ``panels < 1`` or ``size_mm`` has non-positive sides.
    """
    if panels < 1:
        raise exceptions.StyleError("Graphical abstract requires panels >= 1.")
    w_mm, h_mm = float(size_mm[0]), float(size_mm[1])
    if w_mm <= 0.0 or h_mm <= 0.0:
        raise exceptions.StyleError(
            "size_mm dimensions must both be positive."
        )
    preset_obj = theme.get_preset(preset)
    sw = theme.stroke_weights(preset)

    with theme.xkcd_style(preset=preset, overrides={"axes.titlepad": 5.0}):
        fig = plt.figure(
            figsize=(canvas.mm_to_inches(w_mm), canvas.mm_to_inches(h_mm)),
            dpi=int(dpi),
            facecolor=preset_obj.figure_facecolor,
        )
        fig.patch.set_sketch_params(0.0, 0.0, 0.0)
        overlay = fig.add_axes((0.0, 0.0, 1.0, 1.0))
        overlay.set_xlim(0.0, w_mm)
        overlay.set_ylim(h_mm, 0.0)
        overlay.set_axis_off()
        overlay.patch.set_visible(False)

        top_mm, bot_mm = _draw_ga_chrome(
            overlay, w_mm, h_mm, title, takeaway, preset_obj, sw
        )
        panel_axes, arrow_ax = _populate_ga_panels(
            fig,
            panels=panels,
            w_mm=w_mm,
            h_mm=h_mm,
            top_mm=top_mm,
            bot_mm=bot_mm,
            arrow_label=arrow_label,
            preset=preset,
        )

    axes_map = dict(zip(_ga_panel_keys(panels), panel_axes, strict=True))
    return GraphicalAbstractLayout(
        figure=fig,
        axes=axes_map,
        left_ax=panel_axes[0],
        right_ax=panel_axes[-1] if panels >= 2 else None,
        arrow_ax=arrow_ax,
        center_ax=panel_axes[1] if panels == 3 else None,
        overlay_ax=overlay,
        size_mm=(w_mm, h_mm),
        dpi=int(dpi),
        preset=preset,
    )


@dataclasses.dataclass(frozen=True)
class TalkSlideLayout:
    """16:9 widescreen presentation slide with optional presenter commentary.

    Attributes:
      figure: Underlying 16:9 Matplotlib ``Figure``.
      axes: Dictionary mapping semantic slot names (e.g. ``"left"``,
        ``"right"``, ``"hero"``, ``"notes"``, ``"center"``) to ``Axes``.
      header_ax: Slide title and subtitle banner ``Axes``.
      footer_ax: Optional bottom takeaway bar ``Axes``.
      presenter_ax: Optional corner stick-scientist ``Axes``.
      preset: Visual theme preset applied to the slide.
      dpi: Default export resolution.
    """

    figure: figure.Figure
    axes: dict[str, maxes.Axes]
    header_ax: maxes.Axes
    footer_ax: maxes.Axes | None = None
    presenter_ax: maxes.Axes | None = None
    preset: str | theme.ThemePreset = "talk_16x9"
    dpi: int = 180

    def save(
        self,
        path: str | pathlib.Path,
        dpi: int | None = None,
        *,
        exact_ratio: bool = True,
    ) -> pathlib.Path:
        """Exports the 16:9 presentation slide to PNG, SVG, or PDF."""
        return export.save_figure(
            self.figure,
            path,
            dpi=self.dpi if dpi is None else dpi,
            tight=not exact_ratio,
        )


def _talk_slide_rects(
    layout_key: str,
    y_bot: float,
    y_top: float,
    has_presenter: bool,
    has_takeaway: bool,
) -> dict[str, tuple[float, float, float, float]]:
    """Computes normalized ``(left, bottom, width, height)`` panel rects."""
    panel_h = y_top - y_bot
    if layout_key == "two_column":
        return {
            "left": (0.065, y_bot, 0.405, panel_h),
            "right": (0.545, y_bot, 0.415, panel_h),
        }
    if layout_key == "hero_right_notes":
        hero_bot = 0.205 if has_takeaway else 0.105
        notes_bot = y_bot + (0.02 if has_presenter else 0.0)
        return {
            "hero": (0.065, hero_bot, 0.560, y_top - hero_bot),
            "notes": (0.665, notes_bot, 0.295, y_top - notes_bot),
        }
    if layout_key == "three_panel":
        return {
            "left": (0.060, y_bot, 0.265, panel_h),
            "center": (0.380, y_bot, 0.265, panel_h),
            "right": (0.700, y_bot, 0.265, panel_h),
        }
    raise exceptions.StyleError(
        f"Unsupported talk slide layout {layout_key!r}. "
        "Use 'two_column', 'hero_right_notes', or 'three_panel'."
    )


def create_talk_slide(
    title: str,
    subtitle: str | None = None,
    takeaway: str | None = None,
    layout: str = "two_column",
    preset: str = "talk_16x9",
    character: str | None = None,
    speech: str | None = None,
    dpi: int = 180,
) -> TalkSlideLayout:
    """Creates a 16:9 widescreen presentation slide with optional presenter.

    Args:
      title: Main slide title rendered across the top header bar.
      subtitle: Optional secondary subtitle line beneath ``title``.
      takeaway: Optional bottom takeaway highlight strip.
      layout: Panel arrangement: ``"two_column"``, ``"hero_right_notes"``, or
        ``"three_panel"``.
      preset: Theme preset (default ``"talk_16x9"``; try ``"talk_dark"``).
      character: Optional stick-scientist cast name (``"cueball"``,
        ``"ponytail"``, ``"megan"``, ``"beret"``, ``"goggles"``) placed in the
        bottom-right corner.
      speech: Optional commentary line spoken by the corner presenter.
      dpi: Figure resolution in dots per inch (default ``180``).

    Returns:
      Configured ``TalkSlideLayout`` ready for ``xkcd_matchem`` plots.

    Raises:
      StyleError: If ``layout`` or ``character`` is invalid.
    """
    layout_key = layout.strip().lower()
    preset_obj = theme.get_preset(preset)
    sw = theme.stroke_weights(preset)
    accent_fill, accent_rule, muted_ink = _resolve_banner_colors(preset_obj)
    has_presenter = (character is not None) or (speech is not None)
    y_bot = 0.250 if has_presenter else (0.205 if takeaway else 0.105)
    rects = _talk_slide_rects(
        layout_key, y_bot, 0.79, has_presenter, bool(takeaway)
    )

    with theme.xkcd_style(preset=preset):
        fig = plt.figure(
            figsize=(12.8, 7.2),
            dpi=int(dpi),
            facecolor=preset_obj.figure_facecolor,
        )
        fig.patch.set_sketch_params(0.0, 0.0, 0.0)

        header_ax = fig.add_axes((0.03, 0.845, 0.94, 0.13))
        header_ax.set_xlim(0.0, 1.0)
        header_ax.set_ylim(0.0, 1.0)
        header_ax.set_axis_off()
        header_ax.patch.set_visible(False)
        header_ax.text(
            0.01,
            0.66 if subtitle else 0.50,
            title,
            fontsize=preset_obj.title_size + 2.5,
            color=preset_obj.ink_color,
            ha="left",
            va="center",
        )
        if subtitle:
            header_ax.text(
                0.01,
                0.24,
                subtitle,
                fontsize=preset_obj.font_size * 0.92,
                color=muted_ink,
                ha="left",
                va="center",
            )
        header_ax.plot(
            [0.0, 1.0],
            [0.04, 0.04],
            color=accent_rule,
            linewidth=sw.primary * 1.2,
        )

        axes_map: dict[str, maxes.Axes] = {}
        for name, rect in rects.items():
            ax_panel = fig.add_axes(rect)
            theme.style_existing_axes(ax_panel, preset=preset)
            axes_map[name] = ax_panel

        footer_ax: maxes.Axes | None = None
        if takeaway:
            footer_w = 0.68 if has_presenter else 0.94
            footer_ax = fig.add_axes((0.03, 0.025, footer_w, 0.088))
            footer_ax.set_xlim(0.0, 1.0)
            footer_ax.set_ylim(0.0, 1.0)
            footer_ax.set_axis_off()
            footer_ax.patch.set_visible(False)
            _draw_rounded_strip(
                footer_ax,
                0.006,
                0.10,
                0.988,
                0.80,
                facecolor=accent_fill,
                edgecolor=preset_obj.ink_color,
                linewidth=sw.overlay,
                rounding=0.14,
                zorder=2.0,
            )
            footer_ax.text(
                0.025 if has_presenter else 0.50,
                0.50,
                takeaway,
                fontsize=preset_obj.font_size * 0.88,
                color=preset_obj.ink_color,
                ha="left" if has_presenter else "center",
                va="center",
                zorder=4.0,
            )

        presenter_ax: maxes.Axes | None = None
        if has_presenter:
            presenter_ax = fig.add_axes((0.72, 0.015, 0.265, 0.220))
            presenter_ax.set_xlim(0.0, 1.0)
            presenter_ax.set_ylim(0.0, 1.0)
            presenter_ax.set_axis_off()
            presenter_ax.patch.set_visible(False)
            wrapped = (
                "\n".join(textwrap.wrap(speech, width=22))
                if speech and "\n" not in speech
                else speech
            )
            comic.draw_stick_scientist(
                presenter_ax,
                xy=(0.85, 0.04),
                scale=0.52,
                pose="point",
                character=character or "cueball",
                facing="left",
                prop="pointer",
                speech=wrapped,
                speech_offset=(-0.44, 0.14),
                color=preset_obj.ink_color,
            )

    return TalkSlideLayout(
        figure=fig,
        axes=axes_map,
        header_ax=header_ax,
        footer_ax=footer_ax,
        presenter_ax=presenter_ax,
        preset=preset,
        dpi=int(dpi),
    )


@dataclasses.dataclass(frozen=True)
class ConferencePosterLayout:
    """Millimeter-accurate multi-card research poster built on ``Board``.

    Attributes:
      board: Underlying physical-millimeter ``Board`` canvas.
      cards: Ordered dictionary mapping each section title to its ``Card``.
      preset: Visual theme preset applied to the poster.
      size: Poster size preset key (``"A0_landscape"``, ``"A0_portrait"``, or
        ``"compact_board"``).
    """

    board: canvas.Board
    cards: dict[str, canvas.Card]
    preset: str | theme.ThemePreset = "poster"
    size: str = "A0_landscape"

    def save(
        self,
        path: str | pathlib.Path,
        dpi: int | None = None,
        *,
        exact_mm: bool = True,
    ) -> pathlib.Path:
        """Exports the completed poster with exact millimeter dimensions."""
        return self.board.save(path, dpi=dpi, exact_mm=exact_mm)


def _poster_grid_dims(n_sections: int, is_landscape: bool) -> tuple[int, int]:
    """Computes balanced ``(ncols, nrows)`` poster grid dimensions."""
    dims = (1, 1)
    if n_sections > 1:
        if is_landscape:
            if n_sections <= 3:
                dims = (n_sections, 1)
            elif n_sections == 4:
                dims = (2, 2)
            elif n_sections <= 6:
                dims = (3, 2)
            elif n_sections <= 8:
                dims = (4, 2)
            else:
                dims = (3, int(math.ceil(n_sections / 3)))
        elif n_sections <= 3:
            dims = (1, n_sections)
        elif n_sections == 4:
            dims = (2, 2)
        else:
            dims = (2, int(math.ceil(n_sections / 2)))
    return dims


def _draw_poster_header(
    board: canvas.Board,
    *,
    title: str,
    authors: str,
    affiliation: str,
    margin_mm: float,
    header_h_mm: float,
    scale_f: float,
    preset_obj: theme.ThemePreset,
    sw: theme.StrokeHierarchy,
) -> None:
    """Draws the scaled poster title banner, authors, and badge onto overlay."""
    accent_fill, _, muted_ink = _resolve_banner_colors(preset_obj)
    banner_w = board.width_mm - 2.0 * margin_mm
    with theme.xkcd_style(preset=board.preset):
        _draw_rounded_strip(
            board.overlay,
            margin_mm,
            margin_mm,
            banner_w,
            header_h_mm,
            facecolor=accent_fill,
            edgecolor=preset_obj.ink_color,
            linewidth=sw.frame * min(scale_f, 1.8),
            rounding=2.4 * scale_f,
            pad=0.3 * scale_f,
            zorder=2.0,
        )
        has_meta = bool(authors or affiliation)
        title_fs = min(42.0, max(14.5, 14.5 * scale_f))
        meta_fs = min(24.0, max(9.5, 9.5 * scale_f))
        title_y = margin_mm + (0.30 if has_meta else 0.50) * header_h_mm
        board.overlay.text(
            margin_mm + 6.0 * scale_f,
            title_y,
            title,
            fontsize=title_fs,
            color=preset_obj.ink_color,
            ha="left",
            va="center",
            zorder=5.0,
        )
        if authors and affiliation:
            board.overlay.text(
                margin_mm + 6.0 * scale_f,
                margin_mm + 0.62 * header_h_mm,
                authors,
                fontsize=meta_fs,
                color=preset_obj.ink_color,
                ha="left",
                va="center",
                zorder=5.0,
            )
            board.overlay.text(
                margin_mm + 6.0 * scale_f,
                margin_mm + 0.84 * header_h_mm,
                affiliation,
                fontsize=meta_fs * 0.90,
                color=muted_ink,
                ha="left",
                va="center",
                zorder=5.0,
            )
        elif authors or affiliation:
            board.overlay.text(
                margin_mm + 6.0 * scale_f,
                margin_mm + 0.72 * header_h_mm,
                authors or affiliation,
                fontsize=meta_fs,
                color=preset_obj.ink_color if authors else muted_ink,
                ha="left",
                va="center",
                zorder=5.0,
            )
    badge_fs = min(22.0, max(8.5, 8.5 * scale_f))
    board.add_badge(
        board.width_mm - margin_mm - 20.0 * scale_f,
        margin_mm + 0.50 * header_h_mm,
        "xkcd-matchem",
        facecolor="#F0E442",
        edgecolor=preset_obj.ink_color,
        fontsize=badge_fs,
    )


def _populate_poster_cards(
    board: canvas.Board,
    section_titles: Sequence[str],
    *,
    margin_mm: float,
    gap_mm: float,
    grid_y0: float,
    scale_f: float,
    preset: str | theme.ThemePreset,
) -> dict[str, canvas.Card]:
    """Creates the millimeter card grid on ``board`` and returns keyed map."""
    preset_obj = theme.get_preset(preset)
    ncols, nrows = _poster_grid_dims(
        len(section_titles), board.width_mm >= board.height_mm
    )
    card_w = (board.width_mm - 2.0 * margin_mm - (ncols - 1) * gap_mm) / ncols
    card_h = (board.height_mm - grid_y0 - margin_mm - (nrows - 1) * gap_mm) / (
        nrows
    )
    use_scaled_hdr = scale_f > 1.5
    inner_pad = (
        19.0 * scale_f,
        6.0 * scale_f,
        13.5 * scale_f,
        17.5 * scale_f,
    )

    cards: dict[str, canvas.Card] = {}
    for idx, sec_title in enumerate(section_titles):
        row, col = divmod(idx, ncols)
        cx = margin_mm + col * (card_w + gap_mm)
        cy = grid_y0 + row * (card_h + gap_mm)
        h_col = _POSTER_HEADER_COLORS[idx % len(_POSTER_HEADER_COLORS)]
        card = board.add_card(
            cx,
            cy,
            card_w,
            card_h,
            title=None if use_scaled_hdr else sec_title,
            header_color=h_col,
            header_band=not use_scaled_hdr,
            inner_pad_mm=inner_pad,
        )
        if use_scaled_hdr:
            with theme.xkcd_style(preset=preset):
                band = patches.FancyBboxPatch(
                    (cx + 0.8 * scale_f, cy + 0.8 * scale_f),
                    card_w - 1.6 * scale_f,
                    8.2 * scale_f,
                    boxstyle=f"round,pad={0.1 * scale_f},"
                    f"rounding_size={1.4 * scale_f}",
                    facecolor=h_col,
                    edgecolor=preset_obj.ink_color,
                    linewidth=1.1 * min(scale_f, 1.8),
                    alpha=0.28,
                    zorder=3.0,
                )
                board.overlay.add_patch(band)
                board.overlay.text(
                    cx + 4.0 * scale_f,
                    cy + 4.9 * scale_f,
                    sec_title,
                    fontsize=min(28.0, 11.0 * scale_f),
                    color=preset_obj.ink_color,
                    va="center",
                    ha="left",
                    zorder=4.0,
                )
            card.title = sec_title
            card.axes.tick_params(labelsize=min(18.0, 11.0 * scale_f * 0.55))
        cards[sec_title] = card
    return cards


def create_conference_poster(
    title: str,
    authors: str = "",
    affiliation: str = "",
    section_titles: Sequence[str] = (
        "1. Motivation",
        "2. Structure & Method",
        "3. Electronic & Bonding",
        "4. Catalysis / Spectra",
        "5. Thermodynamics",
        "6. Key Takeaways",
    ),
    size: str = "A0_landscape",
    preset: str = "poster",
    dpi: int = 150,
) -> ConferencePosterLayout:
    """Creates a millimeter-accurate multi-card research poster on ``Board``.

    Args:
      title: Main poster title rendered in the top banner.
      authors: Optional author list rendered beneath ``title``.
      affiliation: Optional institution or conference affiliation line.
      section_titles: Ordered sequence of card headings to populate the grid.
      size: Physical canvas preset: ``"A0_landscape"`` (``1189x841 mm``),
        ``"A0_portrait"`` (``841x1189 mm``), or ``"compact_board"``
        (``360x240 mm`` for quick previews).
      preset: Visual theme preset (default ``"poster"``).
      dpi: Export resolution in dots per inch (default ``150``).

    Returns:
      ``ConferencePosterLayout`` wrapping the ``Board`` and keyed ``Card``s.

    Raises:
      StyleError: If ``size`` is unknown or ``section_titles`` is empty.
    """
    size_key = size.strip().lower()
    if size_key not in _POSTER_SIZES_MM:
        available = ", ".join(sorted(_POSTER_SIZES_MM))
        raise exceptions.StyleError(
            f"Unsupported poster size {size!r}. Available: {available}"
        )
    if not section_titles:
        raise exceptions.StyleError(
            "Conference poster requires at least one section title."
        )

    w_mm, h_mm = _POSTER_SIZES_MM[size_key]
    scale_f = min(w_mm, h_mm) / 240.0
    preset_obj = theme.get_preset(preset)
    sw = theme.stroke_weights(preset)
    board = canvas.Board(
        width_mm=w_mm,
        height_mm=h_mm,
        dpi=int(dpi),
        title=None,
        subtitle=None,
        preset=preset,
    )

    margin_mm = 7.0 * scale_f
    gap_mm = 5.5 * scale_f
    has_meta = bool(authors or affiliation)
    header_h_mm = (31.0 if has_meta else 22.0) * scale_f
    _draw_poster_header(
        board,
        title=title,
        authors=authors,
        affiliation=affiliation,
        margin_mm=margin_mm,
        header_h_mm=header_h_mm,
        scale_f=scale_f,
        preset_obj=preset_obj,
        sw=sw,
    )
    cards = _populate_poster_cards(
        board,
        section_titles,
        margin_mm=margin_mm,
        gap_mm=gap_mm,
        grid_y0=margin_mm + header_h_mm + gap_mm,
        scale_f=scale_f,
        preset=preset,
    )
    return ConferencePosterLayout(
        board=board,
        cards=cards,
        preset=preset,
        size=size,
    )


@dataclasses.dataclass(frozen=True)
class MechanismComicLayout:
    """Turnkey N-panel research comic strip with pre-positioned plot insets.

    Attributes:
      strip: Underlying ``ComicStrip`` object.
      figure: Underlying Matplotlib ``Figure``.
      panel_axes: Outer comic-panel ``Axes`` (holding frame, character, and
        speech leader).
      inset_axes: Pre-positioned upper-right plot ``Axes`` inside each panel.
      preset: Visual theme preset applied to the comic strip.
    """

    strip: comic.ComicStrip
    figure: figure.Figure
    panel_axes: list[maxes.Axes]
    inset_axes: list[maxes.Axes]
    preset: str | theme.ThemePreset = "classic"

    def save(
        self,
        path: str | pathlib.Path,
        dpi: int = 200,
    ) -> pathlib.Path:
        """Exports the mechanism comic strip to PNG, SVG, or PDF."""
        return self.strip.save(path, dpi=dpi)


def create_mechanism_comic(
    captions: Sequence[str] = (
        "1. Reactants Approach",
        "2. Transition State!",
        "3. Products Formed",
    ),
    speeches: Sequence[str | None] = (
        "Looking stable...",
        "Watch that barrier!",
        "Downhill from here!",
    ),
    characters: Sequence[str] = ("cueball", "ponytail", "goggles"),
    poses: Sequence[str] = ("stand", "point", "cheer"),
    title: str | None = None,
    preset: str = "classic",
    figsize: tuple[float, float] = (13.5, 4.4),
) -> MechanismComicLayout:
    """Creates a turnkey N-panel research comic with stick scientists & insets.

    Args:
      captions: Panel header captions defining the number of comic panels.
      speeches: Speech commentary strings per panel (``None`` omits bubble).
      characters: Sequence of stick-scientist cast keys cycled across panels.
      poses: Sequence of stick-scientist poses cycled across panels.
      title: Optional centered super-title above the comic strip.
      preset: Visual theme preset (default ``"classic"``).
      figsize: Overall ``(width_in, height_in)`` figure dimensions.

    Returns:
      Configured ``MechanismComicLayout`` with ``panel_axes`` and
      ``inset_axes`` ready for reaction/structure plots.

    Raises:
      StyleError: If ``captions``, ``characters``, or ``poses`` is empty.
    """
    if not captions:
        raise exceptions.StyleError(
            "Mechanism comic requires at least one panel caption."
        )
    if not characters or not poses:
        raise exceptions.StyleError(
            "Mechanism comic requires non-empty characters and poses."
        )

    n_panels = len(captions)
    cols = min(3, n_panels)
    rows = int(math.ceil(n_panels / cols))
    panel_size = (float(figsize[0]) / cols, float(figsize[1]) / rows)
    panel_specs = [comic.ComicPanel(title=cap) for cap in captions]
    strip = comic.plot_comic_strip(
        panel_specs,
        ncols=cols,
        panel_size=panel_size,
        preset=preset,
        title=title,
    )
    if title:
        strip.figure.subplots_adjust(top=0.81 if rows == 1 else 0.85)

    insets: list[maxes.Axes] = []
    with theme.xkcd_style(preset=preset):
        for idx, p_ax in enumerate(strip.axes):
            inset = p_ax.inset_axes((0.42, 0.25, 0.54, 0.67))
            theme.style_existing_axes(inset, preset=preset)
            inset.tick_params(labelsize=7.8)
            inset.xaxis.label.set_fontsize(8.2)
            inset.yaxis.label.set_fontsize(8.2)
            inset.title.set_fontsize(9.2)
            insets.append(inset)

            char_k = characters[idx % len(characters)]
            pose_k = poses[idx % len(poses)]
            raw_sp = speeches[idx] if idx < len(speeches) else None
            sp_txt = (
                "\n".join(textwrap.wrap(raw_sp, width=12))
                if raw_sp and "\n" not in raw_sp
                else raw_sp
            )
            prop_k = "pointer" if pose_k.strip().lower() == "point" else None
            strip.add_stick_scientist(
                idx,
                xy=(0.135, 0.05),
                scale=0.28,
                pose=pose_k,
                character=char_k,
                facing="right",
                prop=prop_k,
                speech=sp_txt,
                speech_offset=(0.01, 0.24),
            )

    return MechanismComicLayout(
        strip=strip,
        figure=strip.figure,
        panel_axes=list(strip.axes),
        inset_axes=insets,
        preset=preset,
    )
