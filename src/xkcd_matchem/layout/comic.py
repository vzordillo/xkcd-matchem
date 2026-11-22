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
from typing import Any
from typing import cast

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
    bubble_kind: str = "line"


def _axes_aspect_factor(ax: maxes.Axes) -> float:
    """Returns axes ``height / width`` ratio so stick heads stay circular."""
    fig = ax.figure
    if fig is not None:
        fw, fh = cast(Any, fig).get_size_inches()
        pos = ax.get_position()
        w_in = max(float(pos.width) * float(fw), 1e-4)
        h_in = max(float(pos.height) * float(fh), 1e-4)
        return max(0.30, min(2.20, h_in / w_in))
    return 1.0


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
    """Draws an xkcd speech, thought, or borderless line-leader callout."""
    kind_key = kind.strip().lower()
    if kind_key not in ("line", "speech", "thought"):
        raise exceptions.StyleError(
            f"Unsupported bubble kind {kind!r}. "
            "Use 'line', 'speech', or 'thought'."
        )
    bg = (
        facecolor
        if facecolor is not None
        else str(plt.rcParams.get("axes.facecolor", palettes.PAPER))
    )
    ink = edgecolor if edgecolor is not None else primitives.resolve_ink()
    txt_col = primitives.ensure_readable_text_color(ink, bg)

    if kind_key == "line":
        target = xy if tail_xy is None else tail_xy
        arrow_cfg = (
            {
                "arrowstyle": "-",
                "color": ink,
                "lw": 1.05,
                "shrinkA": 3.0,
                "shrinkB": 3.0,
                "connectionstyle": "arc3,rad=0.12",
            }
            if tail_xy is not None
            else None
        )
        ann = ax.annotate(
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
                "boxstyle": "round,pad=0.14",
                "facecolor": bg,
                "edgecolor": "none",
                "alpha": 0.90,
            },
            arrowprops=arrow_cfg,
            zorder=20.0,
        )
        patch = ann.get_bbox_patch()
        if patch is not None:
            patch.set_sketch_params(0.0, 0.0, 0.0)
        return

    if kind_key == "speech":
        target = xy if tail_xy is None else tail_xy
        arrow_cfg = (
            {
                "arrowstyle": "->,head_width=0.22,head_length=0.30",
                "color": ink,
                "lw": 1.25,
                "connectionstyle": "arc3,rad=0.14",
            }
            if tail_xy is not None
            else None
        )
        ann = ax.annotate(
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
                "boxstyle": "round,pad=0.32,rounding_size=0.28",
                "facecolor": bg,
                "edgecolor": ink,
                "linewidth": 1.35,
            },
            arrowprops=arrow_cfg,
            zorder=20.0,
        )
        patch = ann.get_bbox_patch()
        if patch is not None:
            patch.set_sketch_params(0.0, 0.0, 0.0)
        return

    if tail_xy is not None:
        asp = _axes_aspect_factor(ax)
        for frac, rad in ((0.22, 0.011), (0.45, 0.018)):
            cx = tail_xy[0] + frac * (xy[0] - tail_xy[0])
            cy = tail_xy[1] + frac * (xy[1] - tail_xy[1])
            ax.add_patch(
                patches.Ellipse(
                    (cx, cy),
                    width=2.0 * rad * asp,
                    height=2.0 * rad,
                    transform=ax.transAxes,
                    facecolor=bg,
                    edgecolor=ink,
                    linewidth=1.25,
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
            "boxstyle": "round,pad=0.34,rounding_size=0.42",
            "facecolor": bg,
            "edgecolor": ink,
            "linewidth": 1.35,
            "linestyle": "--",
        },
        zorder=20.0,
    )


def _draw_character_head_detail(
    ax: maxes.Axes,
    character: str,
    head_xy: tuple[float, float],
    radii: tuple[float, float],
    flip: float,
    ink: str,
    bg: str,
) -> None:
    """Draws xkcd character hair, beret, or goggles onto the round head."""
    hx, hy = head_xy
    rx, ry = radii

    def _stk(
        xs: Sequence[float], ys: Sequence[float], lw: float = 1.65
    ) -> None:
        ax.plot(
            xs,
            ys,
            transform=ax.transAxes,
            color=ink,
            linewidth=lw,
            solid_capstyle="round",
            zorder=18.4,
        )

    if character == "ponytail":
        _stk(
            [
                hx + flip * 0.72 * rx,
                hx + flip * 0.1 * rx,
                hx - flip * 0.86 * rx,
            ],
            [hy + 0.56 * ry, hy + 0.96 * ry, hy + 0.35 * ry],
            lw=1.7,
        )
        _stk(
            [
                hx - flip * 0.85 * rx,
                hx - flip * 1.48 * rx,
                hx - flip * 1.72 * rx,
                hx - flip * 1.32 * rx,
            ],
            [hy + 0.32 * ry, hy + 0.25 * ry, hy - 0.32 * ry, hy - 0.86 * ry],
            lw=1.9,
        )
    elif character == "megan":
        _stk(
            [
                hx + flip * 0.80 * rx,
                hx + flip * 0.25 * rx,
                hx - flip * 0.45 * rx,
                hx - flip * 0.96 * rx,
                hx - flip * 1.02 * rx,
            ],
            [
                hy + 0.48 * ry,
                hy + 1.02 * ry,
                hy + 0.98 * ry,
                hy + 0.30 * ry,
                hy - 0.78 * ry,
            ],
            lw=1.9,
        )
        _stk(
            [hx + flip * 0.76 * rx, hx + flip * 0.94 * rx],
            [hy + 0.50 * ry, hy - 0.55 * ry],
            lw=1.75,
        )
    elif character == "beret":
        beret = patches.Ellipse(
            (hx - flip * 0.12 * rx, hy + 0.92 * ry),
            width=2.35 * rx,
            height=0.82 * ry,
            angle=flip * -10.0,
            transform=ax.transAxes,
            facecolor=ink,
            edgecolor=ink,
            linewidth=1.2,
            zorder=18.6,
        )
        ax.add_patch(beret)
        _stk(
            [hx - flip * 0.12 * rx, hx - flip * 0.22 * rx],
            [hy + 1.28 * ry, hy + 1.50 * ry],
            lw=1.5,
        )
    elif character == "goggles":
        _stk(
            [hx - 0.95 * rx, hx + 0.95 * rx],
            [hy + 0.14 * ry, hy + 0.14 * ry],
            lw=1.2,
        )
        lens = patches.FancyBboxPatch(
            (hx + flip * 0.08 * rx - 0.42 * rx, hy - 0.14 * ry),
            0.84 * rx,
            0.52 * ry,
            boxstyle="round,pad=0.0,rounding_size=0.008",
            transform=ax.transAxes,
            facecolor=bg,
            edgecolor=ink,
            linewidth=1.35,
            zorder=18.6,
        )
        ax.add_patch(lens)


def _draw_stick_prop(
    ax: maxes.Axes,
    prop: str | None,
    hand_xy: tuple[float, float],
    scale: float,
    aspect: float,
    flip: float,
    ink: str,
) -> None:
    """Draws optional handheld lab props (pointer, flask, clipboard)."""
    if prop is None:
        return
    hx, hy = hand_xy
    prop_key = prop.strip().lower()
    if prop_key == "pointer":
        ax.plot(
            [hx, hx + flip * 0.32 * scale * aspect],
            [hy, hy + 0.19 * scale],
            transform=ax.transAxes,
            color=ink,
            linewidth=1.55,
            solid_capstyle="round",
            zorder=18.2,
        )
    elif prop_key == "flask":
        tri = [
            (hx - 0.045 * scale * aspect, hy + 0.10 * scale),
            (hx + 0.045 * scale * aspect, hy + 0.10 * scale),
            (hx + 0.095 * scale * aspect, hy - 0.04 * scale),
            (hx - 0.095 * scale * aspect, hy - 0.04 * scale),
        ]
        ax.add_patch(
            patches.Polygon(
                tri,
                closed=True,
                transform=ax.transAxes,
                facecolor="#8ECAE6",
                edgecolor=ink,
                linewidth=1.25,
                zorder=19.0,
            )
        )
    elif prop_key == "clipboard":
        board = patches.Rectangle(
            (
                hx - 0.055 * scale * aspect,
                hy - 0.04 * scale,
            ),
            0.11 * scale * aspect,
            0.14 * scale,
            transform=ax.transAxes,
            facecolor="#FAF6EE",
            edgecolor=ink,
            linewidth=1.25,
            zorder=19.0,
        )
        ax.add_patch(board)


def _stick_arm_polylines(
    pose_key: str,
    x0: float,
    y0: float,
    scale: float,
    aspect: float,
    flip: float,
) -> tuple[list[float], list[float], tuple[float, float]]:
    """Computes clean xkcd-proportioned arm vertices and active hand xy."""
    dx_s = flip * scale * aspect
    shoulder_y = y0 + 0.62 * scale
    if pose_key == "point":
        hx, hy = x0 + 0.25 * dx_s, y0 + 0.72 * scale
        xs = [x0 - 0.14 * dx_s, x0, hx]
        ys = [y0 + 0.45 * scale, shoulder_y, hy]
    elif pose_key == "shrug":
        hx, hy = x0 + 0.19 * dx_s, y0 + 0.70 * scale
        xs = [
            x0 - 0.19 * dx_s,
            x0 - 0.12 * dx_s,
            x0,
            x0 + 0.12 * dx_s,
            hx,
        ]
        ys = [
            y0 + 0.70 * scale,
            shoulder_y - 0.04 * scale,
            shoulder_y,
            shoulder_y - 0.04 * scale,
            hy,
        ]
    elif pose_key == "cheer":
        hx, hy = x0 + 0.26 * dx_s, y0 + 0.92 * scale
        xs = [x0 - 0.26 * dx_s, x0, hx]
        ys = [y0 + 0.92 * scale, shoulder_y, hy]
    elif pose_key == "think":
        hx, hy = x0 + 0.06 * dx_s, y0 + 0.72 * scale
        xs = [
            x0 - 0.11 * dx_s,
            x0 - 0.09 * dx_s,
            x0,
            x0 + 0.14 * dx_s,
            hx,
        ]
        ys = [
            y0 + 0.43 * scale,
            y0 + 0.54 * scale,
            shoulder_y,
            y0 + 0.54 * scale,
            hy,
        ]
    else:
        hx, hy = x0 + 0.11 * dx_s, y0 + 0.43 * scale
        xs = [x0 - 0.11 * dx_s, x0, hx]
        ys = [y0 + 0.43 * scale, shoulder_y, hy]
    return xs, ys, (hx, hy)


def draw_stick_scientist(
    ax: maxes.Axes,
    xy: tuple[float, float] = (0.20, 0.12),
    *,
    scale: float = 0.26,
    pose: str = "point",
    character: str = "cueball",
    facing: str = "right",
    prop: str | None = None,
    speech: str | None = None,
    speech_offset: tuple[float, float] = (0.02, 0.22),
    color: str | None = None,
) -> None:
    """Draws an aspect-invariant xkcd.com stick scientist on any Axes."""
    pose_key = pose.strip().lower()
    if pose_key not in ("point", "shrug", "cheer", "stand", "think"):
        raise exceptions.StyleError(
            f"Unsupported pose {pose!r}. "
            "Use 'stand', 'point', 'think', 'shrug', or 'cheer'."
        )
    char_key = character.strip().lower()
    if char_key not in ("cueball", "ponytail", "megan", "beret", "goggles"):
        raise exceptions.StyleError(
            f"Unsupported character {character!r}. "
            "Use 'cueball', 'ponytail', 'megan', 'beret', or 'goggles'."
        )
    face_key = facing.strip().lower()
    if face_key not in ("right", "left"):
        raise exceptions.StyleError(
            f"Unsupported facing {facing!r}. Use 'right' or 'left'."
        )
    flip = -1.0 if face_key == "left" else 1.0
    ink = color if color is not None else primitives.resolve_ink()
    bg = str(plt.rcParams.get("axes.facecolor", palettes.PAPER))
    x0, y0 = float(xy[0]), float(xy[1])
    h = float(scale)
    asp = _axes_aspect_factor(ax)
    rx = 0.115 * h * asp
    ry = 0.120 * h
    hy = y0 + 0.85 * h

    head = patches.Ellipse(
        (x0, hy),
        width=2.0 * rx,
        height=2.0 * ry,
        transform=ax.transAxes,
        facecolor=bg,
        edgecolor=ink,
        linewidth=1.75,
        zorder=18.0,
    )
    ax.add_patch(head)
    _draw_character_head_detail(ax, char_key, (x0, hy), (rx, ry), flip, ink, bg)

    def _seg(xs: Sequence[float], ys: Sequence[float]) -> None:
        ax.plot(
            xs,
            ys,
            transform=ax.transAxes,
            color=ink,
            linewidth=1.75,
            solid_capstyle="round",
            zorder=18.0,
        )

    _seg([x0, x0], [y0 + 0.72 * h, y0 + 0.34 * h])
    _seg(
        [x0 - 0.12 * h * asp, x0, x0 + 0.12 * h * asp],
        [y0, y0 + 0.34 * h, y0],
    )
    arm_xs, arm_ys, hand_xy = _stick_arm_polylines(
        pose_key, x0, y0, h, asp, flip
    )
    _seg(arm_xs, arm_ys)
    _draw_stick_prop(ax, prop, hand_xy, h, asp, flip, ink)

    if speech:
        draw_speech_bubble(
            ax,
            speech,
            xy=(x0 + speech_offset[0], y0 + h + speech_offset[1]),
            tail_xy=(x0 + flip * 0.02 * h * asp, y0 + h + 0.02 * h),
            kind="line",
            edgecolor=ink,
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
        character: str = "cueball",
        facing: str = "right",
        prop: str | None = None,
        speech: str | None = None,
        speech_offset: tuple[float, float] = (0.02, 0.22),
    ) -> None:
        """Draws a stick-figure scientist onto panel ``index``."""
        ax = self.panel(index)
        with theme.xkcd_style(preset=self.preset):
            draw_stick_scientist(
                ax,
                xy=xy,
                scale=scale,
                pose=pose,
                character=character,
                facing=facing,
                prop=prop,
                speech=speech,
                speech_offset=speech_offset,
            )

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
