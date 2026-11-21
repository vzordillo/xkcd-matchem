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

import dataclasses
import pathlib

from matplotlib import axes as maxes
from matplotlib import figure
from matplotlib import patches
from matplotlib import pyplot as plt

from xkcd_matchem.layout import export
from xkcd_matchem.style import palettes
from xkcd_matchem.style import theme


def mm_to_inches(mm: float) -> float:
    """Converts millimeters to inches."""
    return float(mm) / 25.4


@dataclasses.dataclass
class Badge:
    """Small hand-drawn pill badge rendered on the overlay axes."""

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
    top-left corner of the sheet for predictable layout composition.
    """

    def __init__(
        self,
        width_mm: float = 280.0,
        height_mm: float = 180.0,
        dpi: int = 180,
        title: str | None = None,
        subtitle: str | None = None,
        preset: str = "classic",
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
            self.overlay: maxes.Axes = self.figure.add_axes(
                [0.0, 0.0, 1.0, 1.0]
            )
            self.overlay.set_xlim(0.0, self.width_mm)
            self.overlay.set_ylim(self.height_mm, 0.0)
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

    def add_container(
        self,
        x_mm: float,
        y_mm: float,
        w_mm: float,
        h_mm: float,
        title: str | None = None,
        facecolor: str = palettes.PASTEL_BG,
        edgecolor: str | None = None,
    ) -> Container:
        """Adds a dashed pastel section box behind child cards."""
        edge = self.theme.ink_color if edgecolor is None else edgecolor
        with theme.xkcd_style(preset=self.preset):
            box = patches.FancyBboxPatch(
                (x_mm, y_mm),
                w_mm,
                h_mm,
                boxstyle="round,pad=0.6,rounding_size=2.5",
                facecolor=facecolor,
                edgecolor=edge,
                linestyle="--",
                linewidth=1.3,
                alpha=0.8,
                zorder=1,
            )
            self.overlay.add_patch(box)
            if title:
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
            facecolor=facecolor,
            edgecolor=edge,
        )

    def add_card(
        self,
        x_mm: float,
        y_mm: float,
        w_mm: float,
        h_mm: float,
        title: str | None = None,
        header_color: str = "#56B4E9",
        inner_pad_mm: tuple[float, float, float, float] | None = None,
    ) -> Card:
        """Adds a framed card and returns its child plotting ``Card.axes``.

        Args:
          x_mm: Left position in millimeters.
          y_mm: Top position in millimeters.
          w_mm: Card width in millimeters.
          h_mm: Card height in millimeters.
          title: Optional header banner title.
          header_color: Fill tint for the card header band.
          inner_pad_mm: ``(left, right, top, bottom)`` padding inside the card.

        Returns:
          ``Card`` whose ``.axes`` accepts ``xkcd_matchem`` plots.
        """
        pad_l, pad_r, pad_t, pad_b = (
            (19.5, 6.5, 14.5 if title else 7.0, 15.5)
            if inner_pad_mm is None
            else inner_pad_mm
        )
        with theme.xkcd_style(preset=self.preset):
            card_box = patches.FancyBboxPatch(
                (x_mm, y_mm),
                w_mm,
                h_mm,
                boxstyle="round,pad=0.3,rounding_size=2.2",
                facecolor=self.theme.axes_facecolor,
                edgecolor=self.theme.ink_color,
                linewidth=1.6,
                zorder=2,
            )
            self.overlay.add_patch(card_box)
            if title:
                band = patches.FancyBboxPatch(
                    (x_mm + 0.8, y_mm + 0.8),
                    w_mm - 1.6,
                    8.2,
                    boxstyle="round,pad=0.1,rounding_size=1.4",
                    facecolor=header_color,
                    edgecolor=self.theme.ink_color,
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
                    color=self.theme.ink_color,
                    va="center",
                    ha="left",
                    zorder=4,
                )
            ax_left = (x_mm + pad_l) / self.width_mm
            ax_w = max(0.05, (w_mm - pad_l - pad_r) / self.width_mm)
            ax_bottom = (
                self.height_mm - (y_mm + h_mm - pad_b)
            ) / self.height_mm
            ax_h = max(0.05, (h_mm - pad_t - pad_b) / self.height_mm)
            child_ax = self.figure.add_axes([ax_left, ax_bottom, ax_w, ax_h])
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
    ) -> pathlib.Path:
        """Exports the completed board to PNG, SVG, or PDF."""
        return export.save_figure(
            self.figure, path, dpi=self.dpi if dpi is None else dpi
        )
