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
"""Physical-millimeter multi-panel Board, Container, Card, and Badge layout."""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import pathlib

from matplotlib import axes as maxes
from matplotlib import figure
from matplotlib import patches
from matplotlib import pyplot as plt

from xkcd_matchem.layout import export
from xkcd_matchem.layout import scroll
from xkcd_matchem.style import palettes
from xkcd_matchem.style import patterns
from xkcd_matchem.style import theme


def mm_to_inches(mm: float) -> float:
    """Converts millimeters to inches."""
    return float(mm) / 25.4


@dataclasses.dataclass
class Badge:
    """Small xkcd-style pill badge rendered on the overlay axes."""

    text: str
    x_mm: float
    y_mm: float
    facecolor: str
    edgecolor: str
    textcolor: str


@dataclasses.dataclass
class Container:
    """Grouped dashed or tinted outer container frame in millimeters."""

    x_mm: float
    y_mm: float
    w_mm: float
    h_mm: float
    title: str | None = None
    facecolor: str = palettes.PASTEL_BG
    edgecolor: str = palettes.INK


@dataclasses.dataclass
class Card:
    """Framed content card owning a child Matplotlib ``Axes`` for plots."""

    axes: maxes.Axes
    x_mm: float
    y_mm: float
    w_mm: float
    h_mm: float
    title: str | None = None
    header_color: str | None = None


class Board:
    """Physical millimeter canvas for multi-panel posters and figures.

    Uses normalized top-down coordinates ``(x_mm, y_mm)`` with ``(0, 0)`` at the
    top-left corner of the sheet so you can group and stack multiple ``Card``
    panels vertically or side by side inside ``Container`` boxes.
    """

    def __init__(
        self,
        width_mm: float = 280.0,
        height_mm: float = 180.0,
        dpi: int = 180,
        title: str | None = None,
        subtitle: str | None = None,
        preset: str | theme.ThemePreset = "classic",
        background_style: str = "none",
    ) -> None:
        """Initializes the millimeter canvas and background overlay axes."""
        self.width_mm = float(width_mm)
        self.height_mm = float(height_mm)
        self.dpi = int(dpi)
        self.preset = preset
        self.theme = theme.get_preset(preset)
        with theme.xkcd_style(preset=preset):
            self.figure: figure.Figure = plt.figure(
                figsize=(
                    mm_to_inches(self.width_mm),
                    mm_to_inches(self.height_mm),
                ),
                dpi=self.dpi,
                facecolor=self.theme.figure_facecolor,
            )
            self.figure.patch.set_sketch_params(0.0, 0.0, 0.0)
            self.overlay: maxes.Axes = self.figure.add_axes(
                (0.0, 0.0, 1.0, 1.0)
            )
            self.overlay.set_xlim(0.0, self.width_mm)
            self.overlay.set_ylim(self.height_mm, 0.0)
            self.overlay.set_facecolor(self.theme.figure_facecolor)
            self.overlay.patch.set_sketch_params(0.0, 0.0, 0.0)
            if background_style and background_style.lower() != "none":
                patterns.apply_background_grid(
                    self.overlay,
                    style=background_style,
                    spacing=0.05,
                    margin_line=False,
                )
            self.overlay.set_axis_off()
            if title:
                self.overlay.text(
                    10.0,
                    9.5,
                    title,
                    fontsize=17.0,
                    color=self.theme.ink_color,
                    va="top",
                    ha="left",
                    zorder=5,
                )
            if subtitle:
                self.overlay.text(
                    10.0,
                    18.5,
                    subtitle,
                    fontsize=10.5,
                    color=palettes.MUTED_GRAY,
                    va="top",
                    ha="left",
                    zorder=5,
                )
        self.cards: list[Card] = []
        self.scroll_cards: list[scroll.ScrollCard] = []

    def add_container(
        self,
        x_mm: float,
        y_mm: float,
        w_mm: float,
        h_mm: float,
        title: str | None = None,
        facecolor: str = palettes.PASTEL_BG,
        edgecolor: str | None = None,
        solid_header: bool = True,
        header_color: str = "#3E6E8E",
    ) -> Container:
        """Adds a section box behind child cards with solid or dashed header."""
        edge = (
            header_color
            if solid_header
            else (self.theme.ink_color if edgecolor is None else edgecolor)
        )
        fill = (
            "#FFFFFF"
            if solid_header and facecolor == palettes.PASTEL_BG
            else facecolor
        )
        with theme.xkcd_style(preset=self.preset):
            box = patches.FancyBboxPatch(
                (x_mm, y_mm),
                w_mm,
                h_mm,
                boxstyle="round,pad=0.6,rounding_size=2.5",
                facecolor=fill,
                edgecolor=edge,
                linestyle="-" if solid_header else "--",
                linewidth=1.8 if solid_header else 1.3,
                alpha=0.95 if solid_header else 0.8,
                zorder=1,
            )
            self.overlay.add_patch(box)
            if title and solid_header:
                header_patch = patches.FancyBboxPatch(
                    (x_mm + 0.3, y_mm + 0.3),
                    w_mm - 0.6,
                    8.6,
                    boxstyle="round,pad=0.3,rounding_size=1.8",
                    facecolor=header_color,
                    edgecolor="none",
                    zorder=2,
                )
                self.overlay.add_patch(header_patch)
                self.overlay.text(
                    x_mm + 0.5 * w_mm,
                    y_mm + 4.7,
                    title,
                    fontsize=12.0,
                    color="#FFFFFF",
                    va="center",
                    ha="center",
                    zorder=3,
                )
            elif title:
                self.overlay.text(
                    x_mm + 4.0,
                    y_mm + 4.0,
                    title,
                    fontsize=11.0,
                    color=edge,
                    va="top",
                    ha="left",
                    zorder=2,
                )
        return Container(
            x_mm=x_mm,
            y_mm=y_mm,
            w_mm=w_mm,
            h_mm=h_mm,
            title=title,
            facecolor=fill,
            edgecolor=edge,
        )

    def add_card(
        self,
        x_mm: float,
        y_mm: float,
        w_mm: float,
        h_mm: float,
        title: str | None = None,
        subtitle: str | None = None,
        header_color: str = "#56B4E9",
        header_band: bool = False,
        inner_pad_mm: tuple[float, float, float, float] | None = None,
        preset: str | theme.ThemePreset | None = None,
    ) -> Card:
        """Adds a framed card and returns its child plotting ``Card.axes``.

        Defaults to a clean borderless white card header (``header_band=False``)
        and switches to a tinted top banner when ``header_band=True``. Pass
        ``preset="blueprint"`` or ``"dark_chalk"`` to style an individual card
        in a contrasting theme.

        Args:
          x_mm: Left position in millimeters.
          y_mm: Top position in millimeters.
          w_mm: Card width in millimeters.
          h_mm: Card height in millimeters.
          title: Optional centered card title.
          subtitle: Optional secondary subtitle line beneath ``title``.
          header_color: Fill tint when ``header_band=True``.
          header_band: Whether to draw a tinted header box behind ``title``.
          inner_pad_mm: ``(left, right, top, bottom)`` padding inside the card.
          preset: Optional per-card theme preset overriding the board preset.

        Returns:
          ``Card`` whose ``.axes`` accepts ``xkcd_matchem`` plots.
        """
        card_preset = self.preset if preset is None else preset
        card_theme = theme.get_preset(card_preset)
        default_top = 17.0 if (title and subtitle) else (13.0 if title else 6.5)
        pad_l, pad_r, pad_t, pad_b = (
            (18.5, 6.5, default_top, 15.0)
            if inner_pad_mm is None
            else inner_pad_mm
        )
        with theme.xkcd_style(preset=card_preset):
            card_box = patches.FancyBboxPatch(
                (x_mm, y_mm),
                w_mm,
                h_mm,
                boxstyle="round,pad=0.3,rounding_size=2.2",
                facecolor=card_theme.axes_facecolor,
                edgecolor=card_theme.ink_color,
                linewidth=1.5,
                zorder=2,
            )
            self.overlay.add_patch(card_box)
            if title and header_band:
                band = patches.FancyBboxPatch(
                    (x_mm + 0.8, y_mm + 0.8),
                    w_mm - 1.6,
                    8.2,
                    boxstyle="round,pad=0.1,rounding_size=1.4",
                    facecolor=header_color,
                    edgecolor=card_theme.ink_color,
                    linewidth=1.1,
                    alpha=0.28,
                    zorder=3,
                )
                self.overlay.add_patch(band)
                self.overlay.text(
                    x_mm + 4.0,
                    y_mm + 4.9,
                    title,
                    fontsize=11.0,
                    color=card_theme.ink_color,
                    va="center",
                    ha="left",
                    zorder=4,
                )
            elif title:
                self.overlay.text(
                    x_mm + 0.5 * w_mm,
                    y_mm + 4.8,
                    title,
                    fontsize=11.5,
                    color=card_theme.ink_color,
                    va="center",
                    ha="center",
                    zorder=4,
                )
            if subtitle:
                self.overlay.text(
                    x_mm + 0.5 * w_mm,
                    y_mm + 10.2,
                    subtitle,
                    fontsize=9.2,
                    color=palettes.MUTED_GRAY,
                    va="center",
                    ha="center",
                    zorder=4,
                )
            ax_left = (x_mm + pad_l) / self.width_mm
            ax_w = max(0.05, (w_mm - pad_l - pad_r) / self.width_mm)
            ax_bottom = (
                self.height_mm - (y_mm + h_mm - pad_b)
            ) / self.height_mm
            ax_h = max(0.05, (h_mm - pad_t - pad_b) / self.height_mm)
            child_ax = self.figure.add_axes((ax_left, ax_bottom, ax_w, ax_h))
        card = Card(
            axes=child_ax,
            x_mm=x_mm,
            y_mm=y_mm,
            w_mm=w_mm,
            h_mm=h_mm,
            title=title,
            header_color=header_color,
        )
        self.cards.append(card)
        return card

    def add_scroll_card(
        self,
        x_mm: float,
        y_mm: float,
        w_mm: float,
        h_mm: float,
        *,
        orientation: str = "portrait",
        title: str | None = None,
        subtitle: str | None = None,
        items: Sequence[str | tuple[str, str]] | None = None,
        bullet: str | None = "check",
        parchment_color: str = scroll.PARCHMENT_LIGHT,
        roller_color: str = scroll.PARCHMENT_ROLLER,
        background_grid: str | None = None,
        align_to_grid: bool = True,
        grid_spacing: float | None = None,
        inner_pad_mm: tuple[float, float, float, float] | None = None,
    ) -> scroll.ScrollCard:
        """Adds a parchment scroll card in portrait or landscape view."""
        sc = scroll.draw_scroll_card_on_board(
            self.figure,
            self.overlay,
            (self.width_mm, self.height_mm),
            (x_mm, y_mm, w_mm, h_mm),
            orientation=orientation,
            title=title,
            subtitle=subtitle,
            items=items,
            bullet=bullet,
            scroll_colors=(
                parchment_color,
                roller_color,
                scroll.PARCHMENT_INK,
            ),
            background_grid=background_grid,
            align_to_grid=align_to_grid,
            grid_spacing=grid_spacing,
            preset=self.preset,
            inner_pad_mm=inner_pad_mm,
        )
        self.scroll_cards.append(sc)
        return sc

    def add_badge(
        self,
        x_mm: float,
        y_mm: float,
        text: str,
        facecolor: str = "#F0E442",
        edgecolor: str | None = None,
        fontsize: float = 9.0,
    ) -> Badge:
        """Places a rounded annotation badge anywhere on the canvas."""
        edge = self.theme.ink_color if edgecolor is None else edgecolor
        with theme.xkcd_style(preset=self.preset):
            self.overlay.text(
                x_mm,
                y_mm,
                text,
                fontsize=fontsize,
                color=edge,
                va="center",
                ha="center",
                bbox={
                    "boxstyle": "round,pad=0.35",
                    "facecolor": facecolor,
                    "edgecolor": edge,
                    "lw": 1.3,
                    "alpha": 0.9,
                },
                zorder=10,
            )
        return Badge(
            text=text,
            x_mm=x_mm,
            y_mm=y_mm,
            facecolor=facecolor,
            edgecolor=edge,
            textcolor=edge,
        )

    def save(
        self,
        path: str | pathlib.Path,
        dpi: int | None = None,
        *,
        exact_mm: bool = True,
    ) -> pathlib.Path:
        """Exports the completed board with exact millimeter dimensions."""
        return export.save_figure(
            self.figure,
            path,
            dpi=self.dpi if dpi is None else dpi,
            tight=not exact_mm,
        )
