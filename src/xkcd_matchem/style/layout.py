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
"""Label anti-collision engine, layout auditing, and perceptual regression."""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import io
import math
from typing import Any

from matplotlib import axes as maxes
from matplotlib import colors as mcolors
from matplotlib import figure as mfigure
import matplotlib.backends.backend_agg as mpl_agg
import matplotlib.image as mimage
import matplotlib.text as mtext
import numpy as np

from xkcd_matchem.core import exceptions


@dataclasses.dataclass(frozen=True)
class LayoutAuditReport:
    """Automated visual layout and theme integrity report for a ``Figure``.

    Attributes:
      text_overlaps: Colliding visible text pairs ``(text_a, text_b, ioa)``.
      clipped_texts: Visible text strings extending meaningfully off-canvas.
      border_leak_ratio: Fraction of outer 2-pixel rim pixels deviating from
        ``fig.get_facecolor()`` in ``[0.0, 1.0]``.
      low_contrast_texts: Visible ``(text, contrast_ratio)`` entries falling
        below ``min_contrast``.
      ink_ratio: Fraction of rendered canvas pixels differing noticeably from
        the figure background.
    """

    text_overlaps: tuple[tuple[str, str, float], ...]
    clipped_texts: tuple[str, ...]
    border_leak_ratio: float
    low_contrast_texts: tuple[tuple[str, float], ...]
    ink_ratio: float = 0.0

    @property
    def passed(self) -> bool:
        """Returns True when overlap, clipping, border, and contrast pass."""
        return (
            not self.text_overlaps
            and not self.clipped_texts
            and self.border_leak_ratio <= 0.01
            and not self.low_contrast_texts
        )


@dataclasses.dataclass(frozen=True)
class VisualFingerprint:
    """Compact deterministic structural and chromatic figure fingerprint.

    Attributes:
      luminance_grid: Block-averaged ``(H, W)`` perceptual luminance grid.
      rgb_grid: Block-averaged ``(H, W, 3)`` sRGB grid in ``[0.0, 1.0]``.
      bg_rgb: Normalized ``(r, g, b)`` figure background color.
      ink_ratio: Fraction of non-background pixels on the rendered canvas.
      size: Spatial grid resolution ``(height, width)``.
    """

    luminance_grid: np.ndarray
    rgb_grid: np.ndarray
    bg_rgb: tuple[float, float, float]
    ink_ratio: float
    size: tuple[int, int]


def relative_luminance(color: Any) -> float:
    """Computes WCAG 2.1 relative luminance for any Matplotlib color."""
    r_s, g_s, b_s = mcolors.to_rgb(color)
    lin = [
        c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
        for c in (r_s, g_s, b_s)
    ]
    return float(0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2])


def wcag_contrast_ratio(fg_color: Any, bg_color: Any) -> float:
    """Computes the WCAG 2.1 contrast ratio in ``[1.0, 21.0]``."""
    lum_a = relative_luminance(fg_color)
    lum_b = relative_luminance(bg_color)
    hi, lo = max(lum_a, lum_b), min(lum_a, lum_b)
    return float((hi + 0.05) / (lo + 0.05))


def ensure_wcag_contrast(
    fg_color: Any,
    bg_color: Any,
    *,
    min_contrast: float = 4.5,
) -> str:
    """Adjusts ``fg_color`` minimally toward white/ink to meet WCAG contrast."""
    base_rgb = mcolors.to_rgb(fg_color)
    if wcag_contrast_ratio(base_rgb, bg_color) >= min_contrast:
        return mcolors.to_hex(base_rgb)
    lum_bg = relative_luminance(bg_color)
    pole = (1.0, 1.0, 1.0) if lum_bg < 0.18 else (0.067, 0.067, 0.067)
    if wcag_contrast_ratio(pole, bg_color) <= min_contrast:
        return mcolors.to_hex(pole)
    lo_t, hi_t = 0.0, 1.0
    for _ in range(18):
        mid = 0.5 * (lo_t + hi_t)
        cand = (
            base_rgb[0] + (pole[0] - base_rgb[0]) * mid,
            base_rgb[1] + (pole[1] - base_rgb[1]) * mid,
            base_rgb[2] + (pole[2] - base_rgb[2]) * mid,
        )
        if wcag_contrast_ratio(cand, bg_color) >= min_contrast:
            hi_t = mid
        else:
            lo_t = mid
    final_rgb = (
        base_rgb[0] + (pole[0] - base_rgb[0]) * hi_t,
        base_rgb[1] + (pole[1] - base_rgb[1]) * hi_t,
        base_rgb[2] + (pole[2] - base_rgb[2]) * hi_t,
    )
    return mcolors.to_hex(final_rgb)


def _relax_tier_xs(
    xs_tier: np.ndarray,
    min_dx: float,
    x_bounds: tuple[float, float] | None,
    relax_iters: int,
) -> np.ndarray:
    """Relaxes 1D horizontal coordinates within one vertical stagger tier."""
    out: np.ndarray = np.asarray(xs_tier, dtype=np.float64).copy()
    n_pts = out.size
    if n_pts <= 1 or min_dx <= 0.0:
        if x_bounds is not None and n_pts > 0:
            out = np.asarray(
                np.clip(out, x_bounds[0], x_bounds[1]), dtype=np.float64
            )
        return out
    eff_dx = min_dx
    if x_bounds is not None and x_bounds[1] > x_bounds[0]:
        eff_dx = min(min_dx, (x_bounds[1] - x_bounds[0]) / float(n_pts - 1))
    for _ in range(max(1, relax_iters)):
        moved = False
        for k in range(1, n_pts):
            gap = float(out[k] - out[k - 1])
            if gap < eff_dx - 1e-9:
                push = 0.52 * (eff_dx - gap)
                out[k - 1] -= push
                out[k] += push
                moved = True
        if x_bounds is not None and x_bounds[1] >= x_bounds[0]:
            out = np.asarray(
                np.clip(out, x_bounds[0], x_bounds[1]), dtype=np.float64
            )
        if not moved:
            break
    return out


def resolve_1d_label_positions(
    xs: Sequence[float] | np.ndarray,
    ys: Sequence[float] | np.ndarray,
    min_dx: float,
    y_step: float,
    *,
    max_tiers: int = 4,
    x_bounds: tuple[float, float] | None = None,
    relax_iters: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    """Computes staggered anti-colliding ``(x, y)`` positions for 1D labels.

    Args:
      xs: Anchor horizontal coordinates of length ``N``.
      ys: Base vertical coordinates of length ``N``.
      min_dx: Minimum desired horizontal spacing within the same tier.
      y_step: Signed vertical stagger increment per tier.
      max_tiers: Maximum number of vertical stagger tiers (``>= 1``).
      x_bounds: Optional ``(xmin, xmax)`` horizontal clamp interval.
      relax_iters: Number of 1D spring-relaxation iterations per tier.

    Returns:
      Tuple ``(adj_xs, adj_ys)`` of adjusted 1D ``float64`` arrays matching
      the input element order.

    Raises:
      StyleError: If ``xs`` and ``ys`` are not matching 1D sequences.
    """
    x_arr = np.asarray(xs, dtype=np.float64)
    y_arr = np.asarray(ys, dtype=np.float64)
    if x_arr.ndim != 1 or y_arr.ndim != 1 or x_arr.shape != y_arr.shape:
        raise exceptions.StyleError(
            "resolve_1d_label_positions requires matching 1D xs and ys arrays."
        )
    n_items = x_arr.size
    if n_items == 0:
        return x_arr.copy(), y_arr.copy()
    n_tiers = max(1, int(max_tiers))
    dx_req = max(0.0, float(min_dx))
    order = sorted(
        range(n_items), key=lambda idx: (float(x_arr[idx]), float(y_arr[idx]))
    )

    last_x = [float("-inf")] * n_tiers
    tier_buckets: list[list[int]] = [[] for _ in range(n_tiers)]
    for sorted_pos, orig_idx in enumerate(order):
        xv = float(x_arr[orig_idx])
        chosen = -1
        for t_idx in range(n_tiers):
            if xv - last_x[t_idx] >= dx_req - 1e-9:
                chosen = t_idx
                break
        if chosen < 0:
            chosen = min(range(n_tiers), key=lambda t_i: last_x[t_i])
            if dx_req <= 0.0:
                chosen = sorted_pos % n_tiers
        last_x[chosen] = xv
        tier_buckets[chosen].append(orig_idx)

    adj_x = x_arr.copy()
    adj_y = y_arr.copy()
    for t_idx, idxs in enumerate(tier_buckets):
        if not idxs:
            continue
        relaxed = _relax_tier_xs(
            x_arr[idxs], dx_req, x_bounds=x_bounds, relax_iters=relax_iters
        )
        for loc_i, orig_idx in enumerate(idxs):
            adj_x[orig_idx] = float(relaxed[loc_i])
            adj_y[orig_idx] = float(y_arr[orig_idx]) + float(t_idx) * float(
                y_step
            )
    return adj_x, adj_y


def _ensure_agg_draw(fig: mfigure.Figure) -> Any:
    """Ensures ``fig`` has a rendered Agg canvas and returns its renderer."""
    canvas: Any = fig.canvas
    if canvas is None or not hasattr(canvas, "buffer_rgba"):
        canvas = mpl_agg.FigureCanvasAgg(fig)
    canvas.draw()
    return canvas.get_renderer()


def _artist_pixel_bbox(
    artist: mtext.Text,
    renderer: Any,
) -> list[float]:
    """Returns ``[x0, y0, x1, y1]`` display bbox including any halo patch."""
    bb = artist.get_window_extent(renderer=renderer)
    x0, y0, x1, y1 = float(bb.x0), float(bb.y0), float(bb.x1), float(bb.y1)
    patch = artist.get_bbox_patch()
    if patch is not None and patch.get_visible():
        pb = patch.get_window_extent(renderer=renderer)
        x0 = min(x0, float(pb.x0))
        y0 = min(y0, float(pb.y0))
        x1 = max(x1, float(pb.x1))
        y1 = max(y1, float(pb.y1))
    return [x0, y0, x1, y1]


def _shift_text_artist_px(
    artist: mtext.Text,
    dx_px: float,
    dy_px: float,
    ax: maxes.Axes,
    dpi: float,
) -> None:
    """Translates a ``Text`` or ``Annotation`` by ``(dx_px, dy_px)`` pixels."""
    if abs(dx_px) < 0.05 and abs(dy_px) < 0.05:
        return
    if isinstance(artist, mtext.Annotation):
        ann_coords = artist.anncoords
        ox, oy = float(artist.xyann[0]), float(artist.xyann[1])
        if ann_coords == "offset points":
            pt_per_px = 72.0 / max(dpi, 1.0)
            artist.xyann = (ox + dx_px * pt_per_px, oy + dy_px * pt_per_px)
            return
        if ann_coords == "offset pixels":
            artist.xyann = (ox + dx_px, oy + dy_px)
            return
        if ann_coords == "data":
            px, py = ax.transData.transform((ox, oy))
            nx, ny = ax.transData.inverted().transform(
                (float(px) + dx_px, float(py) + dy_px)
            )
            artist.xyann = (float(nx), float(ny))
            return
        if ann_coords == "axes fraction":
            px, py = ax.transAxes.transform((ox, oy))
            nx, ny = ax.transAxes.inverted().transform(
                (float(px) + dx_px, float(py) + dy_px)
            )
            artist.xyann = (float(nx), float(ny))
            return
    trans = artist.get_transform()
    pos_x, pos_y = artist.get_position()
    px, py = trans.transform((float(pos_x), float(pos_y)))
    inv = trans.inverted()
    nx, ny = inv.transform((float(px) + dx_px, float(py) + dy_px))
    artist.set_position((float(nx), float(ny)))


def _clamp_box_to_bounds(
    box: list[float],
    bounds: tuple[float, float, float, float],
) -> tuple[float, float]:
    """Computes ``(dx, dy)`` needed to clamp ``box`` inside ``bounds``."""
    bx0, by0, bx1, by1 = bounds
    w = box[2] - box[0]
    h = box[3] - box[1]
    avail_w = bx1 - bx0
    avail_h = by1 - by0
    dx, dy = 0.0, 0.0
    if avail_w > 0.0:
        if w >= avail_w:
            dx = 0.5 * (bx0 + bx1) - 0.5 * (box[0] + box[2])
        elif box[0] < bx0:
            dx = bx0 - box[0]
        elif box[2] > bx1:
            dx = bx1 - box[2]
    if avail_h > 0.0:
        if h >= avail_h:
            dy = 0.5 * (by0 + by1) - 0.5 * (box[1] + box[3])
        elif box[1] < by0:
            dy = by0 - box[1]
        elif box[3] > by1:
            dy = by1 - box[3]
    return dx, dy


def _pair_repulsion_step(
    box_a: list[float],
    box_b: list[float],
    pad_px: float,
    tie_x: float,
) -> tuple[float, float]:
    """Computes separation delta ``(dx_b, dy_b)`` for two colliding boxes."""
    ov_x = min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]) + 2.0 * pad_px
    ov_y = min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]) + 2.0 * pad_px
    if ov_x <= 0.0 or ov_y <= 0.0:
        return 0.0, 0.0
    cx_a = 0.5 * (box_a[0] + box_a[2])
    cy_a = 0.5 * (box_a[1] + box_a[3])
    cx_b = 0.5 * (box_b[0] + box_b[2])
    cy_b = 0.5 * (box_b[1] + box_b[3])
    dir_x = (
        1.0 if cx_b > cx_a + 0.25 else (-1.0 if cx_b < cx_a - 0.25 else tie_x)
    )
    dir_y = 1.0 if cy_b >= cy_a - 0.25 else -1.0
    if ov_y <= 1.35 * ov_x:
        return 0.12 * dir_x * min(ov_x, 6.0), 0.54 * dir_y * ov_y
    return 0.54 * dir_x * ov_x, 0.12 * dir_y * min(ov_y, 6.0)


def _collect_static_obstacles(
    ax: maxes.Axes,
    movable_set: set[int],
    renderer: Any,
) -> list[list[float]]:
    """Collects fixed obstacle bboxes (legend and non-movable texts)."""
    obs: list[list[float]] = []
    leg = ax.get_legend()
    if leg is not None and leg.get_visible():
        lb = leg.get_window_extent(renderer=renderer)
        if lb.width > 1.0 and lb.height > 1.0:
            obs.append([float(lb.x0), float(lb.y0), float(lb.x1), float(lb.y1)])
    for other in ax.texts:
        if (
            id(other) not in movable_set
            and other.get_visible()
            and other.get_text().strip()
        ):
            obs.append(_artist_pixel_bbox(other, renderer))
    return obs


def _relax_boxes_in_pixel_space(
    boxes: list[list[float]],
    obstacles: list[list[float]],
    bounds_per_item: list[tuple[float, float, float, float]],
    max_iter: int,
    pad_px: float,
) -> list[tuple[float, float]]:
    """Iteratively resolves pixel-bbox overlaps against peers and obstacles."""
    n_items = len(boxes)
    orig_centers = [
        (0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3])) for b in boxes
    ]
    for _ in range(max(1, max_iter)):
        any_overlap = False
        deltas = [[0.0, 0.0] for _ in range(n_items)]
        for i in range(n_items):
            for j in range(i + 1, n_items):
                tie_x = 1.0 if (j - i) % 2 == 1 else -1.0
                dx, dy = _pair_repulsion_step(boxes[i], boxes[j], pad_px, tie_x)
                if dx != 0.0 or dy != 0.0:
                    any_overlap = True
                    deltas[i][0] -= dx
                    deltas[i][1] -= dy
                    deltas[j][0] += dx
                    deltas[j][1] += dy
            for obs_box in obstacles:
                dx_o, dy_o = _pair_repulsion_step(
                    obs_box, boxes[i], pad_px, 1.0 if i % 2 == 0 else -1.0
                )
                if dx_o != 0.0 or dy_o != 0.0:
                    any_overlap = True
                    deltas[i][0] += 1.65 * dx_o
                    deltas[i][1] += 1.65 * dy_o
        if not any_overlap:
            break
        for i in range(n_items):
            sx = float(np.clip(deltas[i][0], -28.0, 28.0))
            sy = float(np.clip(deltas[i][1], -28.0, 28.0))
            boxes[i][0] += sx
            boxes[i][2] += sx
            boxes[i][1] += sy
            boxes[i][3] += sy
            cdx, cdy = _clamp_box_to_bounds(boxes[i], bounds_per_item[i])
            boxes[i][0] += cdx
            boxes[i][2] += cdx
            boxes[i][1] += cdy
            boxes[i][3] += cdy
    return [
        (
            0.5 * (boxes[i][0] + boxes[i][2]) - orig_centers[i][0],
            0.5 * (boxes[i][1] + boxes[i][3]) - orig_centers[i][1],
        )
        for i in range(n_items)
    ]


def _resolve_clamp_bounds(
    box: list[float],
    ax_bounds: tuple[float, float, float, float],
    fig_bounds: tuple[float, float, float, float],
    clamp_to_axes: bool,
) -> tuple[float, float, float, float]:
    """Selects axes or figure clamping rectangle appropriate for ``box``."""
    if not clamp_to_axes:
        return fig_bounds
    w = box[2] - box[0]
    h = box[3] - box[1]
    ax_w = ax_bounds[2] - ax_bounds[0]
    ax_h = ax_bounds[3] - ax_bounds[1]
    started_in_ax = (
        box[0] >= ax_bounds[0] - 16.0
        and box[2] <= ax_bounds[2] + 16.0
        and box[1] >= ax_bounds[1] - 16.0
        and box[3] <= ax_bounds[3] + 16.0
    )
    if started_in_ax and w <= 0.96 * ax_w and h <= 0.96 * ax_h:
        return ax_bounds
    if w <= 0.92 * ax_w and h <= 0.92 * ax_h:
        return ax_bounds
    return fig_bounds


def _axes_and_fig_pixel_bounds(
    ax: maxes.Axes,
    fig: mfigure.Figure,
    renderer: Any,
) -> tuple[
    tuple[float, float, float, float], tuple[float, float, float, float]
]:
    """Computes padded display-pixel clamping boxes for ``ax`` and ``fig``."""
    ax_bb = ax.get_window_extent(renderer=renderer)
    fig_bb = fig.bbox
    ax_bounds = (
        float(ax_bb.x0) + 3.0,
        float(ax_bb.y0) + 3.0,
        float(ax_bb.x1) - 3.0,
        float(ax_bb.y1) - 3.0,
    )
    fig_bounds = (
        float(fig_bb.x0) + 4.0,
        float(fig_bb.y0) + 4.0,
        float(fig_bb.x1) - 4.0,
        float(fig_bb.y1) - 4.0,
    )
    return ax_bounds, fig_bounds


def _maybe_attach_leader(
    ax: maxes.Axes,
    artist: mtext.Text,
    box_final: list[float],
    shift_px: tuple[float, float],
    leader_color: str,
    leader_lw: float,
) -> None:
    """Draws a thin un-sketched data-space leader segment to displaced text."""
    if not isinstance(artist, mtext.Annotation):
        return
    if artist.arrow_patch is not None or not ax.axison:
        return
    if math.hypot(shift_px[0], shift_px[1]) < 15.0:
        return
    if artist.xycoords != "data":
        return
    anchor_px = ax.transData.transform(
        (float(artist.xy[0]), float(artist.xy[1]))
    )
    cx = float(np.clip(anchor_px[0], box_final[0], box_final[2]))
    cy = float(np.clip(anchor_px[1], box_final[1], box_final[3]))
    if math.hypot(cx - float(anchor_px[0]), cy - float(anchor_px[1])) < 6.0:
        return
    inv = ax.transData.inverted()
    p0 = inv.transform((float(anchor_px[0]), float(anchor_px[1])))
    p1 = inv.transform((cx, cy))
    ldr = ax.plot(
        [float(p0[0]), float(p1[0])],
        [float(p0[1]), float(p1[1])],
        color=leader_color,
        lw=leader_lw,
        alpha=0.65,
        zorder=max(float(artist.get_zorder()) - 0.5, 2.0),
    )[0]
    ldr.set_sketch_params(0.0, 0.0, 0.0)


def avoid_text_overlaps(
    ax: maxes.Axes,
    texts: Sequence[mtext.Text] | None = None,
    *,
    max_iter: int = 30,
    pad_px: float = 2.0,
    clamp_to_axes: bool = True,
    draw_leaders: bool = False,
    leader_color: str | None = None,
    leader_lw: float = 0.9,
) -> int:
    """Iteratively nudges overlapping ``Text``/``Annotation`` artists apart.

    Uses rendered Matplotlib window extents (``fig.canvas.draw()`` +
    ``get_window_extent()``) to separate colliding annotations along vertical
    and outward directions while clamping inside axes/figure bounds.

    Args:
      ax: Target Matplotlib ``Axes``.
      texts: Optional subset of ``Text`` or ``Annotation`` artists to adjust;
        defaults to all visible non-empty ``ax.texts``.
      max_iter: Maximum relaxation iterations in display pixel coordinates.
      pad_px: Minimum padding margin (pixels) around each text bounding box.
      clamp_to_axes: Clamp displaced labels inside ``ax`` (or ``fig`` fallback).
      draw_leaders: Attach thin leader segments when ``Annotation`` labels are
        displaced significantly from their anchor point.
      leader_color: Stroke color for auto-attached leader lines.
      leader_lw: Linewidth in points for auto-attached leader lines.

    Returns:
      Number of text artists whose positions were adjusted.
    """
    fig_raw = ax.get_figure()
    if not isinstance(fig_raw, mfigure.Figure):
        return 0
    candidates = list(ax.texts) if texts is None else list(texts)
    movable = [
        t for t in candidates if t.get_visible() and bool(t.get_text().strip())
    ]
    if not movable:
        return 0

    renderer = _ensure_agg_draw(fig_raw)
    dpi = float(fig_raw.dpi)
    ax_bounds, fig_bounds = _axes_and_fig_pixel_bounds(ax, fig_raw, renderer)
    use_ax_clamp = bool(clamp_to_axes and ax.axison)
    boxes = [_artist_pixel_bbox(t, renderer) for t in movable]
    bounds_list = [
        _resolve_clamp_bounds(b, ax_bounds, fig_bounds, use_ax_clamp)
        for b in boxes
    ]
    moved_flags = [False] * len(movable)
    for idx, box in enumerate(boxes):
        cdx, cdy = _clamp_box_to_bounds(box, bounds_list[idx])
        if abs(cdx) > 0.25 or abs(cdy) > 0.25:
            box[0] += cdx
            box[2] += cdx
            box[1] += cdy
            box[3] += cdy
            _shift_text_artist_px(movable[idx], cdx, cdy, ax, dpi)
            moved_flags[idx] = True

    obstacles = _collect_static_obstacles(
        ax, {id(t) for t in movable}, renderer
    )
    shifts = _relax_boxes_in_pixel_space(
        boxes, obstacles, bounds_list, max_iter=max_iter, pad_px=pad_px
    )
    ldr_col = (
        mcolors.to_hex(ax.spines["left"].get_edgecolor())
        if leader_color is None and "left" in ax.spines
        else (leader_color or "#111111")
    )
    for idx, (dx_px, dy_px) in enumerate(shifts):
        if math.hypot(dx_px, dy_px) >= 0.4:
            _shift_text_artist_px(movable[idx], dx_px, dy_px, ax, dpi)
            moved_flags[idx] = True
            if draw_leaders:
                _maybe_attach_leader(
                    ax,
                    movable[idx],
                    boxes[idx],
                    (dx_px, dy_px),
                    ldr_col,
                    leader_lw,
                )
    return sum(1 for flag in moved_flags if flag)


def _render_figure_rgb(fig: mfigure.Figure) -> np.ndarray:
    """Renders ``fig`` to a tight composited float64 RGB array in ``[0, 1]``."""
    bio = io.BytesIO()
    fig.savefig(
        bio,
        format="png",
        dpi=float(fig.dpi),
        facecolor=fig.get_facecolor(),
        edgecolor=fig.get_facecolor(),
        bbox_inches="tight",
        pad_inches=0.06,
    )
    bio.seek(0)
    arr = np.asarray(mimage.imread(bio, format="png"), dtype=np.float64)
    if arr.ndim == 3 and arr.shape[2] >= 4:
        alpha = arr[..., 3:4]
        rgb = arr[..., :3] * alpha + (1.0 - alpha)
        return np.asarray(rgb, dtype=np.float64)
    return np.asarray(arr[..., :3], dtype=np.float64)


def _compute_border_leak_and_ink(
    fig: mfigure.Figure,
    check_dark_border: bool,
) -> tuple[float, float]:
    """Computes outer 2-pixel rim leak ratio and canvas ink coverage ratio."""
    rgb = _render_figure_rgb(fig)
    fc_rgb = np.asarray(mcolors.to_rgb(fig.get_facecolor()), dtype=np.float64)
    canvas_diff = np.max(np.abs(rgb - fc_rgb[None, None, :]), axis=-1)
    ink_ratio = float(np.mean(canvas_diff > 0.06))
    if not check_dark_border or rgb.shape[0] < 5 or rgb.shape[1] < 5:
        return 0.0, ink_ratio
    top = rgb[:2, :, :].reshape(-1, 3)
    bot = rgb[-2:, :, :].reshape(-1, 3)
    left = rgb[2:-2, :2, :].reshape(-1, 3)
    right = rgb[2:-2, -2:, :].reshape(-1, 3)
    rim = np.concatenate([top, bot, left, right], axis=0)
    rim_diff = np.max(np.abs(rim - fc_rgb[None, :]), axis=1)
    leak_ratio = float(np.mean(rim_diff > 0.08))
    return leak_ratio, ink_ratio


def _collect_visible_audit_texts(
    fig: mfigure.Figure,
) -> list[tuple[mtext.Text, str, Any]]:
    """Enumerates visible ``(artist, role, host_legend)`` entries on ``fig``."""
    collected: list[tuple[mtext.Text, str, Any]] = []
    seen: set[int] = set()

    def _add(t_obj: mtext.Text, role: str, leg_obj: Any = None) -> None:
        if id(t_obj) in seen or not t_obj.get_visible():
            return
        if not t_obj.get_text().strip():
            return
        seen.add(id(t_obj))
        collected.append((t_obj, role, leg_obj))

    for f_txt in fig.texts:
        _add(f_txt, "figure")
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        for a_txt in ax.texts:
            _add(a_txt, "annotation")
        _add(ax.title, "axis_label")
        if ax.axison:
            _add(ax.xaxis.label, "axis_label")
            _add(ax.yaxis.label, "axis_label")
            x_lo, x_hi = sorted(float(v) for v in ax.get_xlim())
            x_tol = 1e-5 * max(x_hi - x_lo, 1.0)
            for loc, lbl in zip(
                ax.get_xticks(), ax.get_xticklabels(), strict=False
            ):
                if x_lo - x_tol <= float(loc) <= x_hi + x_tol:
                    _add(lbl, "xtick")
            y_lo, y_hi = sorted(float(v) for v in ax.get_ylim())
            y_tol = 1e-5 * max(y_hi - y_lo, 1.0)
            for loc, lbl in zip(
                ax.get_yticks(), ax.get_yticklabels(), strict=False
            ):
                if y_lo - y_tol <= float(loc) <= y_hi + y_tol:
                    _add(lbl, "ytick")
        leg = ax.get_legend()
        if leg is not None and leg.get_visible():
            for l_txt in leg.get_texts():
                _add(l_txt, "legend", leg)
            _add(leg.get_title(), "legend", leg)
    return collected


def _shrunk_glyph_box(
    bb: Any,
) -> tuple[float, float, float, float]:
    """Trims TrueType line-height padding slightly for glyph collision tests."""
    w = float(bb.width)
    h = float(bb.height)
    dx = min(1.0, 0.08 * w)
    dy = min(1.8, 0.14 * h)
    return (
        float(bb.x0) + dx,
        float(bb.y0) + dy,
        float(bb.x1) - dx,
        float(bb.y1) - dy,
    )


def _parallel_tilted_separated(
    t_a: mtext.Text,
    box_a: tuple[float, float, float, float],
    t_b: mtext.Text,
    box_b: tuple[float, float, float, float],
    dpi: float,
) -> bool:
    """Checks oriented separation when two texts share a non-axis tilt angle."""
    rot_a = float(t_a.get_rotation()) % 180.0
    rot_b = float(t_b.get_rotation()) % 180.0
    if abs(rot_a - rot_b) > 8.0:
        return False
    tilt = min(rot_a % 90.0, 90.0 - (rot_a % 90.0))
    if tilt < 12.0:
        return False
    theta = math.radians(rot_a)
    cx_a = 0.5 * (box_a[0] + box_a[2])
    cy_a = 0.5 * (box_a[1] + box_a[3])
    cx_b = 0.5 * (box_b[0] + box_b[2])
    cy_b = 0.5 * (box_b[1] + box_b[3])
    perp_dist = abs(
        -(cx_b - cx_a) * math.sin(theta) + (cy_b - cy_a) * math.cos(theta)
    )
    h_a = float(t_a.get_fontsize()) * (dpi / 72.0) * 1.05
    h_b = float(t_b.get_fontsize()) * (dpi / 72.0) * 1.05
    return perp_dist >= 0.5 * (h_a + h_b)


_BoxItem = tuple[mtext.Text, str, tuple[float, float, float, float]]


def _pair_overlap_ioa(
    t_a: mtext.Text,
    role_a: str,
    box_a: tuple[float, float, float, float],
    t_b: mtext.Text,
    role_b: str,
    box_b: tuple[float, float, float, float],
    dpi: float,
) -> float | None:
    """Returns overlap IoA when ``t_a`` and ``t_b`` collide meaningfully."""
    ix0, iy0 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    ix1, iy1 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return None
    inter = (ix1 - ix0) * (iy1 - iy0)
    if inter <= 12.0 or _parallel_tilted_separated(t_a, box_a, t_b, box_b, dpi):
        return None
    area_a = max((box_a[2] - box_a[0]) * (box_a[3] - box_a[1]), 1e-6)
    area_b = max((box_b[2] - box_b[0]) * (box_b[3] - box_b[1]), 1e-6)
    ioa = inter / min(area_a, area_b)
    rot_a = float(t_a.get_rotation()) % 180.0
    rot_b = float(t_b.get_rotation()) % 180.0
    rot_diff = min(abs(rot_a - rot_b), 180.0 - abs(rot_a - rot_b))
    thresh = 0.16
    if {role_a, role_b} == {"xtick", "ytick"}:
        thresh = 0.36
    elif rot_diff > 15.0:
        thresh = 0.45
    elif (
        min(len(t_a.get_text().strip()), len(t_b.get_text().strip())) <= 2
        and role_a != role_b
    ):
        thresh = 0.38
    return round(float(ioa), 3) if ioa > thresh else None


def _detect_text_overlaps(
    items: Sequence[tuple[mtext.Text, str, Any]],
    renderer: Any,
    dpi: float,
) -> tuple[tuple[str, str, float], ...]:
    """Finds meaningfully colliding visible text pairs across ``fig``."""
    measured: list[_BoxItem] = []
    for t_obj, role, _ in items:
        win_bb = t_obj.get_window_extent(renderer=renderer)
        if win_bb.width <= 1.0 or win_bb.height <= 1.0:
            continue
        measured.append((t_obj, role, _shrunk_glyph_box(win_bb)))

    overlaps: list[tuple[str, str, float]] = []
    for i, (t_a, role_a, box_a) in enumerate(measured):
        txt_a = t_a.get_text().strip()
        for t_b, role_b, box_b in measured[i + 1 :]:
            ioa = _pair_overlap_ioa(
                t_a, role_a, box_a, t_b, role_b, box_b, dpi
            )
            if ioa is not None:
                overlaps.append((txt_a, t_b.get_text().strip(), ioa))
    return tuple(overlaps)


def _detect_clipped_texts(
    items: Sequence[tuple[mtext.Text, str, Any]],
    fig: mfigure.Figure,
    renderer: Any,
) -> tuple[str, ...]:
    """Finds visible texts extending meaningfully outside the figure canvas."""
    fig_w = float(fig.bbox.width)
    fig_h = float(fig.bbox.height)
    tol_inner = max(8.0, 0.02 * min(fig_w, fig_h))
    tol_outer = max(52.0, 0.14 * max(fig_w, fig_h))
    clipped: list[str] = []
    for t_obj, role, _ in items:
        bb = t_obj.get_window_extent(renderer=renderer)
        if bb.width <= 1.0 or bb.height <= 1.0:
            continue
        tol = (
            tol_outer
            if role in ("axis_label", "xtick", "ytick")
            else tol_inner
        )
        if (
            float(bb.x0) < -tol
            or float(bb.y0) < -tol
            or float(bb.x1) > fig_w + tol
            or float(bb.y1) > fig_h + tol
        ):
            clipped.append(t_obj.get_text().strip())
    return tuple(clipped)


def _blend_rgba_layer(
    bg: np.ndarray,
    color: Any,
) -> np.ndarray:
    """Alpha-composites ``color`` onto ``bg`` RGB in ``[0, 1]``."""
    rgba = mcolors.to_rgba(color)
    alpha = float(rgba[3])
    if alpha <= 0.02:
        return bg
    fg = np.asarray(rgba[:3], dtype=np.float64)
    return np.asarray(alpha * fg + (1.0 - alpha) * bg, dtype=np.float64)


def _effective_text_background(
    t_obj: mtext.Text,
    leg_obj: Any,
    fig_bg: np.ndarray,
    renderer: Any,
) -> np.ndarray:
    """Resolves the composited RGB background directly behind ``t_obj``."""
    bg: np.ndarray = fig_bg.copy()
    ax = t_obj.axes
    bb = t_obj.get_window_extent(renderer=renderer)
    cx = 0.5 * (float(bb.x0) + float(bb.x1))
    cy = 0.5 * (float(bb.y0) + float(bb.y1))

    if ax is not None and ax.patch.get_visible():
        bg = _blend_rgba_layer(bg, ax.get_facecolor())
        t_z = float(t_obj.get_zorder())
        bg_patches = sorted(
            (
                p
                for p in ax.patches
                if p.get_visible() and float(p.get_zorder()) <= t_z
            ),
            key=lambda p: float(p.get_zorder()),
        )
        for patch in bg_patches:
            p_rgba = mcolors.to_rgba(patch.get_facecolor())
            if float(p_rgba[3]) > 0.02 and patch.contains_point((cx, cy)):
                bg = _blend_rgba_layer(bg, p_rgba)

    if leg_obj is not None and leg_obj.get_frame_on():
        frame = leg_obj.get_frame()
        if frame is not None and frame.get_visible():
            bg = _blend_rgba_layer(bg, frame.get_facecolor())

    bp = t_obj.get_bbox_patch()
    if bp is not None and bp.get_visible():
        bg = _blend_rgba_layer(bg, bp.get_facecolor())
    return bg


def _detect_low_contrast_texts(
    items: Sequence[tuple[mtext.Text, str, Any]],
    fig: mfigure.Figure,
    renderer: Any,
    min_contrast: float,
) -> tuple[tuple[str, float], ...]:
    """Finds visible texts failing WCAG ``min_contrast`` against local bg."""
    fig_bg = np.asarray(mcolors.to_rgb(fig.get_facecolor()), dtype=np.float64)
    low_contrast: list[tuple[str, float]] = []
    for t_obj, _, leg_obj in items:
        bg_rgb = _effective_text_background(t_obj, leg_obj, fig_bg, renderer)
        fg_rgba = mcolors.to_rgba(t_obj.get_color())
        t_alpha = t_obj.get_alpha()
        eff_alpha = float(fg_rgba[3]) * (
            1.0 if t_alpha is None else float(t_alpha)
        )
        fg_rgb = (
            eff_alpha * np.asarray(fg_rgba[:3], dtype=np.float64)
            + (1.0 - eff_alpha) * bg_rgb
        )
        ratio = wcag_contrast_ratio(
            (float(fg_rgb[0]), float(fg_rgb[1]), float(fg_rgb[2])),
            (float(bg_rgb[0]), float(bg_rgb[1]), float(bg_rgb[2])),
        )
        if ratio < min_contrast:
            low_contrast.append((t_obj.get_text().strip(), round(ratio, 3)))
    return tuple(low_contrast)


def audit_figure_layout(
    fig: mfigure.Figure,
    *,
    check_dark_border: bool = True,
    min_contrast: float = 3.5,
) -> LayoutAuditReport:
    """Renders ``fig`` via Agg and audits text overlaps, clipping, and contrast.

    Args:
      fig: Target Matplotlib ``Figure`` to inspect.
      check_dark_border: Inspect outer 2-pixel rim for background color leaks.
      min_contrast: Minimum acceptable WCAG 2.1 contrast ratio for visible text.

    Returns:
      Populated ``LayoutAuditReport`` instance.
    """
    renderer = _ensure_agg_draw(fig)
    items = _collect_visible_audit_texts(fig)
    overlaps = _detect_text_overlaps(items, renderer, dpi=float(fig.dpi))
    clipped = _detect_clipped_texts(items, fig, renderer)
    low_contrast = _detect_low_contrast_texts(
        items, fig, renderer, min_contrast=min_contrast
    )
    leak_ratio, ink_ratio = _compute_border_leak_and_ink(
        fig, check_dark_border=check_dark_border
    )
    return LayoutAuditReport(
        text_overlaps=overlaps,
        clipped_texts=clipped,
        border_leak_ratio=round(leak_ratio, 5),
        low_contrast_texts=low_contrast,
        ink_ratio=round(ink_ratio, 5),
    )


def _downsample_area_rgb(
    rgb: np.ndarray,
    size: tuple[int, int],
) -> np.ndarray:
    """Deterministically downsamples ``(H, W, 3)`` float RGB to ``size``."""
    target_h, target_w = int(size[0]), int(size[1])
    if target_h < 4 or target_w < 4:
        raise exceptions.StyleError(
            f"Fingerprint size must be at least (4, 4), got {size!r}."
        )
    src_h, src_w, _ = rgb.shape
    row_edges = np.linspace(0, src_h, target_h + 1, dtype=np.int64)
    col_edges = np.linspace(0, src_w, target_w + 1, dtype=np.int64)
    out = np.zeros((target_h, target_w, 3), dtype=np.float64)
    for r_i in range(target_h):
        r0 = int(row_edges[r_i])
        r1 = min(max(r0 + 1, int(row_edges[r_i + 1])), src_h)
        for c_i in range(target_w):
            c0 = int(col_edges[c_i])
            c1 = min(max(c0 + 1, int(col_edges[c_i + 1])), src_w)
            patch = rgb[r0:r1, c0:c1, :]
            out[r_i, c_i, 0] = float(np.mean(patch[..., 0]))
            out[r_i, c_i, 1] = float(np.mean(patch[..., 1]))
            out[r_i, c_i, 2] = float(np.mean(patch[..., 2]))
    return out


def figure_visual_fingerprint(
    fig: mfigure.Figure,
    size: tuple[int, int] = (64, 64),
) -> VisualFingerprint:
    """Computes a deterministic structural/perceptual fingerprint for ``fig``.

    Args:
      fig: Target Matplotlib ``Figure``.
      size: Downsampled ``(height, width)`` spatial grid resolution.

    Returns:
      ``VisualFingerprint`` with block-averaged luminance, RGB, and ink ratio.
    """
    rgb = _render_figure_rgb(fig)
    fc_rgb = mcolors.to_rgb(fig.get_facecolor())
    fc_arr = np.asarray(fc_rgb, dtype=np.float64)
    diff = np.max(np.abs(rgb - fc_arr[None, None, :]), axis=-1)
    ink_ratio = float(np.mean(diff > 0.06))
    rgb_small = _downsample_area_rgb(rgb, size)
    lum_small = (
        0.2126 * rgb_small[..., 0]
        + 0.7152 * rgb_small[..., 1]
        + 0.0722 * rgb_small[..., 2]
    )
    return VisualFingerprint(
        luminance_grid=np.asarray(lum_small, dtype=np.float64),
        rgb_grid=rgb_small,
        bg_rgb=(float(fc_rgb[0]), float(fc_rgb[1]), float(fc_rgb[2])),
        ink_ratio=round(ink_ratio, 6),
        size=(int(size[0]), int(size[1])),
    )


def _windowed_ssim_2d(
    grid_a: np.ndarray,
    grid_b: np.ndarray,
    win: int = 8,
) -> float:
    """Computes mean local-window SSIM across 2D float grids in ``[0, 1]``."""
    h, w = grid_a.shape
    step_h = max(1, min(win, h))
    step_w = max(1, min(win, w))
    c1 = 0.01**2
    c2 = 0.03**2
    scores: list[float] = []
    for r0 in range(0, h, step_h):
        r1 = min(h, r0 + step_h)
        for c0 in range(0, w, step_w):
            c1_idx = min(w, c0 + step_w)
            pa = grid_a[r0:r1, c0:c1_idx]
            pb = grid_b[r0:r1, c0:c1_idx]
            mu_a = float(np.mean(pa))
            mu_b = float(np.mean(pb))
            da = pa - mu_a
            db = pb - mu_b
            var_a = float(np.mean(da * da))
            var_b = float(np.mean(db * db))
            cov_ab = float(np.mean(da * db))
            num = (2.0 * mu_a * mu_b + c1) * (2.0 * cov_ab + c2)
            den = (mu_a * mu_a + mu_b * mu_b + c1) * (var_a + var_b + c2)
            scores.append(num / max(den, 1e-12))
    return float(np.mean(np.asarray(scores, dtype=np.float64)))


def visual_similarity(
    fp_a: VisualFingerprint,
    fp_b: VisualFingerprint,
) -> float:
    """Computes perceptual SSIM + ink-coverage similarity in ``[0.0, 1.0]``."""
    common_size = (
        min(fp_a.size[0], fp_b.size[0]),
        min(fp_a.size[1], fp_b.size[1]),
    )
    rgb_a = (
        fp_a.rgb_grid
        if fp_a.size == common_size
        else _downsample_area_rgb(fp_a.rgb_grid, common_size)
    )
    rgb_b = (
        fp_b.rgb_grid
        if fp_b.size == common_size
        else _downsample_area_rgb(fp_b.rgb_grid, common_size)
    )
    lum_a = (
        fp_a.luminance_grid
        if fp_a.size == common_size
        else (
            0.2126 * rgb_a[..., 0]
            + 0.7152 * rgb_a[..., 1]
            + 0.0722 * rgb_a[..., 2]
        )
    )
    lum_b = (
        fp_b.luminance_grid
        if fp_b.size == common_size
        else (
            0.2126 * rgb_b[..., 0]
            + 0.7152 * rgb_b[..., 1]
            + 0.0722 * rgb_b[..., 2]
        )
    )
    ssim_lum = max(0.0, _windowed_ssim_2d(lum_a, lum_b))
    ssim_chroma = max(
        0.0,
        (
            _windowed_ssim_2d(rgb_a[..., 0], rgb_b[..., 0])
            + _windowed_ssim_2d(rgb_a[..., 1], rgb_b[..., 1])
            + _windowed_ssim_2d(rgb_a[..., 2], rgb_b[..., 2])
        )
        / 3.0,
    )
    denom_ink = max(fp_a.ink_ratio, fp_b.ink_ratio, 0.015)
    ink_sim = max(0.0, 1.0 - abs(fp_a.ink_ratio - fp_b.ink_ratio) / denom_ink)
    bg_diff = max(
        abs(fp_a.bg_rgb[0] - fp_b.bg_rgb[0]),
        abs(fp_a.bg_rgb[1] - fp_b.bg_rgb[1]),
        abs(fp_a.bg_rgb[2] - fp_b.bg_rgb[2]),
    )
    bg_sim = max(0.0, 1.0 - 1.6 * bg_diff)
    combined = (
        0.40 * ssim_lum + 0.32 * ssim_chroma + 0.16 * ink_sim + 0.12 * bg_sim
    )
    return float(np.clip(combined, 0.0, 1.0))
