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
"""Multi-panel xkcd comic strip composition, speech bubbles, and stick figs."""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import math
import pathlib

from matplotlib import axes as maxes
from matplotlib import figure
from matplotlib import patches
from matplotlib import pyplot as plt

from xkcd_matchem.core import exceptions
from xkcd_matchem.layout import export
from xkcd_matchem.render import primitives
from xkcd_matchem.style import palettes
from xkcd_matchem.style import theme


@dataclasses.dataclass
class ComicPanel:
    """Declarative specification for one panel in a ``ComicStrip``.

    Attributes:
      title: Optional header badge centered at the top of the panel.
      caption: Optional narrative caption strip along the bottom edge.
      speech: Optional speech or thought bubble text drawn inside the panel.
      speech_xy: Normalized ``(x, y)`` anchor for the speech bubble box.
      tail_xy: Optional pointer target ``(x, y)`` in axes-fraction coords.
      bubble_kind: ``"speech"`` (pointed tail) or ``"thought"`` (dots trail).
    """

    title: str | None = None
    caption: str | None = None
    speech: str | None = None
    speech_xy: tuple[float, float] = (0.66, 0.80)
    tail_xy: tuple[float, float] | None = (0.50, 0.54)
    bubble_kind: str = "speech"


def draw_speech_bubble(
    ax: maxes.Axes,
    text: str,
    xy: tuple[float, float] = (0.66, 0.80),
    tail_xy: tuple[float, float] | None = (0.50, 0.54),
    *,
    kind: str = "speech",
    facecolor: str | None = None,
    edgecolor: str | None = None,
    fontsize: float = 9.5,
) -> None:
    """Draws an xkcd-style speech or thought bubble in axes-fraction coords."""
    kind_key = kind.strip().lower()
    if kind_key not in ("speech", "thought"):
        raise exceptions.StyleError(
            f"Unsupported bubble kind {kind!r}. Use 'speech' or 'thought'."
        )
    bg = (
        facecolor
        if facecolor is not None
        else str(plt.rcParams.get("axes.facecolor", palettes.PAPER))
    )
    ink = edgecolor if edgecolor is not None else primitives.resolve_ink()
    txt_col = primitives.ensure_readable_text_color(ink, bg)

    if kind_key == "speech":
        target = xy if tail_xy is None else tail_xy
        arrow_cfg = (
            {
                "arrowstyle": "->,head_width=0.26,head_length=0.35",
                "color": ink,
                "lw": 1.6,
                "connectionstyle": "arc3,rad=0.14",
            }
            if tail_xy is not None
            else None
        )
        ax.annotate(
            text,
            xy=target,
            xytext=xy,
            xycoords="axes fraction",
            textcoords="axes fraction",
            ha="center",
            va="center",
            fontsize=fontsize,
            color=txt_col,
            bbox={
                "boxstyle": "round,pad=0.36,rounding_size=0.30",
                "facecolor": bg,
                "edgecolor": ink,
                "linewidth": 1.7,
            },
            arrowprops=arrow_cfg,
            zorder=20.0,
        )
        return

    if tail_xy is not None:
        for frac, rad in ((0.22, 0.012), (0.45, 0.020)):
            cx = tail_xy[0] + frac * (xy[0] - tail_xy[0])
            cy = tail_xy[1] + frac * (xy[1] - tail_xy[1])
            ax.add_patch(
                patches.Circle(
                    (cx, cy),
                    radius=rad,
                    transform=ax.transAxes,
                    facecolor=bg,
                    edgecolor=ink,
                    linewidth=1.4,
                    zorder=19.0,
                )
            )
    ax.text(
        xy[0],
        xy[1],
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=txt_col,
        bbox={
            "boxstyle": "round,pad=0.38,rounding_size=0.45",
            "facecolor": bg,
            "edgecolor": ink,
            "linewidth": 1.6,
            "linestyle": "--",
        },
        zorder=20.0,
    )


def draw_stick_scientist(
    ax: maxes.Axes,
    xy: tuple[float, float] = (0.20, 0.12),
    *,
    scale: float = 0.26,
    pose: str = "point",
    prop: str | None = None,
    color: str | None = None,
) -> None:
    """Draws an xkcd-style stick-figure scientist in axes-fraction coords."""
    pose_key = pose.strip().lower()
    if pose_key not in ("point", "shrug", "cheer"):
        raise exceptions.StyleError(
            f"Unsupported pose {pose!r}. Use 'point', 'shrug', or 'cheer'."
        )
    ink = color if color is not None else primitives.resolve_ink()
    bg = str(plt.rcParams.get("axes.facecolor", palettes.PAPER))
    x0, y0 = float(xy[0]), float(xy[1])
    h = float(scale)

    head = patches.Ellipse(
        (x0, y0 + 0.85 * h),
        width=0.22 * h,
        height=0.25 * h,
        transform=ax.transAxes,
        facecolor=bg,
        edgecolor=ink,
        linewidth=1.8,
        zorder=18.0,
    )
    ax.add_patch(head)

    def _seg(xs: Sequence[float], ys: Sequence[float]) -> None:
        ax.plot(
            xs,
            ys,
            transform=ax.transAxes,
            color=ink,
            linewidth=1.9,
            solid_capstyle="round",
            zorder=18.0,
        )

    _seg([x0, x0], [y0 + 0.72 * h, y0 + 0.34 * h])
    _seg([x0 - 0.13 * h, x0, x0 + 0.13 * h], [y0, y0 + 0.34 * h, y0])

    shoulder_y = y0 + 0.62 * h
    if pose_key == "point":
        hand_x, hand_y = x0 + 0.25 * h, y0 + 0.72 * h
        _seg([x0 - 0.16 * h, x0, hand_x], [y0 + 0.48 * h, shoulder_y, hand_y])
    elif pose_key == "shrug":
        hand_x, hand_y = x0 + 0.18 * h, y0 + 0.70 * h
        _seg(
            [
                x0 - 0.18 * h,
                x0 - 0.12 * h,
                x0,
                x0 + 0.12 * h,
                hand_x,
            ],
            [
                y0 + 0.70 * h,
                shoulder_y - 0.04 * h,
                shoulder_y,
                shoulder_y - 0.04 * h,
                hand_y,
            ],
        )
    else:
        hand_x, hand_y = x0 + 0.18 * h, y0 + 0.90 * h
        _seg([x0 - 0.18 * h, x0, hand_x], [y0 + 0.90 * h, shoulder_y, hand_y])

    if prop is not None:
        prop_key = prop.strip().lower()
        if prop_key == "pointer":
            _seg(
                [hand_x, hand_x + 0.34 * h],
                [hand_y, hand_y + 0.20 * h],
            )
        elif prop_key == "flask":
            tri = [
                (hand_x - 0.04 * h, hand_y + 0.10 * h),
                (hand_x + 0.04 * h, hand_y + 0.10 * h),
                (hand_x + 0.09 * h, hand_y - 0.04 * h),
                (hand_x - 0.09 * h, hand_y - 0.04 * h),
            ]
            ax.add_patch(
                patches.Polygon(
                    tri,
                    closed=True,
                    transform=ax.transAxes,
                    facecolor="#8ECAE6",
                    edgecolor=ink,
                    linewidth=1.4,
                    zorder=19.0,
                )
            )


@dataclasses.dataclass
class ComicStrip:
    """Multi-panel comic layout with speech bubbles and stick figures."""

    panels: tuple[ComicPanel, ...]
    figure: figure.Figure
    axes: tuple[maxes.Axes, ...]
    preset: str | theme.ThemePreset = "classic"

    def panel(self, index: int) -> maxes.Axes:
        """Returns Matplotlib ``Axes`` for comic panel ``index``."""
        if not 0 <= index < len(self.axes):
            raise exceptions.StyleError(
                f"Panel index {index} out of range [0, {len(self.axes)})."
            )
        return self.axes[index]

    def add_speech_bubble(
        self,
        index: int,
        text: str,
        xy: tuple[float, float] = (0.66, 0.80),
        tail_xy: tuple[float, float] | None = (0.50, 0.54),
        *,
        kind: str = "speech",
        fontsize: float = 9.5,
    ) -> None:
        """Draws a speech or thought bubble onto panel ``index``."""
        ax = self.panel(index)
        with theme.xkcd_style(preset=self.preset):
            draw_speech_bubble(
                ax,
                text,
                xy=xy,
                tail_xy=tail_xy,
                kind=kind,
                fontsize=fontsize,
            )

    def add_stick_scientist(
        self,
        index: int,
        xy: tuple[float, float] = (0.20, 0.12),
        *,
        scale: float = 0.26,
        pose: str = "point",
        prop: str | None = None,
    ) -> None:
        """Draws a stick-figure scientist onto panel ``index``."""
        ax = self.panel(index)
        with theme.xkcd_style(preset=self.preset):
            draw_stick_scientist(ax, xy=xy, scale=scale, pose=pose, prop=prop)

    def save(
        self,
        path: str | pathlib.Path,
        dpi: int = 200,
    ) -> pathlib.Path:
        """Exports the complete comic strip to PNG, SVG, or PDF."""
        return export.save_figure(self.figure, path, dpi=dpi)


def plot_comic_strip(
    panels: int | Sequence[ComicPanel | str],
    *,
    ncols: int | None = None,
    panel_size: tuple[float, float] = (4.1, 3.9),
    preset: str | theme.ThemePreset = "classic",
    title: str | None = None,
) -> ComicStrip:
    """Creates an xkcd-style multi-panel narrative comic strip figure."""
    if isinstance(panels, int):
        if panels < 1:
            raise exceptions.StyleError("ComicStrip requires at least 1 panel.")
        spec_list = [ComicPanel() for _ in range(panels)]
    else:
        spec_list = [
            ComicPanel(title=item) if isinstance(item, str) else item
            for item in panels
        ]
    if not spec_list:
        raise exceptions.StyleError("ComicStrip requires at least 1 panel.")

    n_tot = len(spec_list)
    cols = min(3, n_tot) if ncols is None else max(1, int(ncols))
    rows = int(math.ceil(n_tot / cols))
    preset_obj = theme.get_preset(preset)

    with theme.xkcd_style(preset=preset):
        fig, grid_axes = plt.subplots(
            rows,
            cols,
            figsize=(cols * panel_size[0], rows * panel_size[1]),
            squeeze=False,
        )
        fig.subplots_adjust(
            left=0.04,
            right=0.96,
            bottom=0.06,
            top=0.88 if title else 0.94,
            wspace=0.14,
            hspace=0.22,
        )
        flat_axes = list(grid_axes.ravel())
        used_axes: list[maxes.Axes] = []
        for idx, ax in enumerate(flat_axes):
            if idx >= n_tot:
                ax.set_visible(False)
                continue
            spec = spec_list[idx]
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(2.1)
                spine.set_color(preset_obj.ink_color)
            if spec.title:
                ax.set_title(
                    spec.title,
                    fontsize=11.0,
                    color=preset_obj.ink_color,
                    pad=7.0,
                )
            if spec.caption:
                ax.text(
                    0.5,
                    -0.06,
                    spec.caption,
                    transform=ax.transAxes,
                    ha="center",
                    va="top",
                    fontsize=9.2,
                    color=preset_obj.ink_color,
                )
            if spec.speech:
                draw_speech_bubble(
                    ax,
                    spec.speech,
                    xy=spec.speech_xy,
                    tail_xy=spec.tail_xy,
                    kind=spec.bubble_kind,
                )
            used_axes.append(ax)
        if title:
            fig.suptitle(title, fontsize=14.0, color=preset_obj.ink_color)

    return ComicStrip(
        panels=tuple(spec_list),
        figure=fig,
        axes=tuple(used_axes),
        preset=preset,
    )
