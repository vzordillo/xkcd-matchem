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
"""xkcd-style 2D/3D atomic, bond, polyhedron, and unit-cell primitives."""

from __future__ import annotations

import math

import matplotlib
from matplotlib import axes as maxes
from matplotlib import colors as mcolors
from matplotlib import patches
import numpy as np

from xkcd_matchem.core import geometry
from xkcd_matchem.style import palettes


_CELL_EDGE_PAIRS: tuple[tuple[int, int, str], ...] = (
    (0, 1, "a"),
    (2, 4, "a"),
    (3, 5, "a"),
    (6, 7, "a"),
    (0, 2, "b"),
    (1, 4, "b"),
    (3, 6, "b"),
    (5, 7, "b"),
    (0, 3, "c"),
    (1, 5, "c"),
    (2, 6, "c"),
    (4, 7, "c"),
)


def lighten_hex(color: str, amount: float = 0.55) -> str:
    """Blends a hex or named color toward white by ``amount`` in ``[0, 1]``."""
    r, g, b = mcolors.to_rgb(color)
    mix = max(0.0, min(1.0, amount))
    return mcolors.to_hex(
        (r + (1.0 - r) * mix, g + (1.0 - g) * mix, b + (1.0 - b) * mix)
    )


def darken_hex(color: str, amount: float = 0.35) -> str:
    """Blends a hex or named color toward black by ``amount`` in ``[0, 1]``."""
    r, g, b = mcolors.to_rgb(color)
    keep = 1.0 - max(0.0, min(1.0, amount))
    return mcolors.to_hex((r * keep, g * keep, b * keep))


def contrast_text_color(bg_color: str) -> str:
    """Returns dark ink or pure white depending on background luminance."""
    r, g, b = mcolors.to_rgb(bg_color)
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return palettes.INK if lum > 0.45 else "#FFFFFF"


def is_dark_color(color: str) -> bool:
    """Returns True when a color has dark perceptual luminance (< 0.35)."""
    r, g, b = mcolors.to_rgb(color)
    return (0.2126 * r + 0.7152 * g + 0.0722 * b) < 0.35


def muted_surface_color(bg_color: str) -> str:
    """Returns a neutral tile/badge background adapted to ``bg_color``."""
    return "#252C3A" if is_dark_color(bg_color) else "#EAECEF"


def ensure_readable_text_color(fg_color: str, bg_color: str) -> str:
    """Adjusts ``fg_color`` to maintain high contrast against ``bg_color``."""
    rf, gf, bf = mcolors.to_rgb(fg_color)
    lum_fg = 0.2126 * rf + 0.7152 * gf + 0.0722 * bf
    if is_dark_color(bg_color):
        if lum_fg < 0.32:
            return lighten_hex(fg_color, 0.58)
        return fg_color
    if lum_fg > 0.70:
        return darken_hex(fg_color, 0.48)
    return fg_color


def resolve_ink(color: str | None = None) -> str:
    """Resolves explicit ink color or active theme ``axes.edgecolor``."""
    if color is not None:
        return color
    return str(matplotlib.rcParams.get("axes.edgecolor", palettes.INK))


def _resolve_ink(color: str | None = None) -> str:
    """Internal alias for ``resolve_ink``."""
    return resolve_ink(color)


def draw_vector_arrow_2d(
    ax: maxes.Axes,
    tail_2d: np.ndarray,
    tip_2d: np.ndarray,
    color: str = "#D55E00",
    edgecolor: str | None = None,
    linewidth: float = 2.0,
    zorder: float = 4.0,
) -> None:
    """Draws an xkcd ink-outlined 2D-projected vector arrow (spins/forces)."""
    dx = float(tip_2d[0] - tail_2d[0])
    dy = float(tip_2d[1] - tail_2d[1])
    if math.hypot(dx, dy) < 1e-4:
        return
    stroke = resolve_ink(edgecolor)
    ax.annotate(
        "",
        xy=(float(tip_2d[0]), float(tip_2d[1])),
        xytext=(float(tail_2d[0]), float(tail_2d[1])),
        arrowprops={
            "arrowstyle": "-|>,head_length=0.42,head_width=0.25",
            "color": stroke,
            "lw": linewidth + 0.90,
            "shrinkA": 0.0,
            "shrinkB": 0.0,
        },
        zorder=zorder,
    )
    ax.annotate(
        "",
        xy=(float(tip_2d[0]), float(tip_2d[1])),
        xytext=(float(tail_2d[0]), float(tail_2d[1])),
        arrowprops={
            "arrowstyle": "-|>,head_length=0.38,head_width=0.22",
            "color": color,
            "lw": linewidth,
            "shrinkA": 0.0,
            "shrinkB": 0.0,
        },
        zorder=zorder + 0.004,
    )


def draw_atom_sphere(
    ax: maxes.Axes,
    x: float,
    y: float,
    radius: float,
    facecolor: str,
    edgecolor: str | None = None,
    linewidth: float = 1.6,
    label: str | None = None,
    highlight: bool = True,
    alpha: float = 1.0,
    zorder: float = 3.0,
) -> None:
    """Draws an xkcd-style shaded atomic sphere with a clean specular dot.

    Args:
      ax: Target Matplotlib ``Axes``.
      x: Projected horizontal center coordinate.
      y: Projected vertical center coordinate.
      radius: Visual circle radius in data units.
      facecolor: Fill color string.
      edgecolor: Outline stroke color; defaults to active theme ink.
      linewidth: Outline width in points.
      label: Optional symbol label centered inside the sphere.
      highlight: Whether to draw a crisp white upper-left specular highlight.
      alpha: Artist opacity in ``[0, 1]``.
      zorder: Matplotlib layer depth ordering.
    """
    circle = patches.Circle(
        (x, y),
        radius=radius,
        facecolor=facecolor,
        edgecolor=_resolve_ink(edgecolor),
        linewidth=linewidth,
        alpha=alpha,
        zorder=zorder,
    )
    ax.add_patch(circle)
    if highlight and radius > 0.02:
        spot = patches.Circle(
            (x - 0.28 * radius, y + 0.28 * radius),
            radius=0.24 * radius,
            facecolor="#FFFFFF",
            edgecolor="none",
            alpha=0.82 * alpha,
            zorder=zorder + 0.004,
        )
        ax.add_patch(spot)
    if label:
        scale_f = 16.5 if len(label) >= 3 else 24.0
        fontsize = max(6.5, min(11.0, radius * scale_f))
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            color=contrast_text_color(facecolor),
            zorder=zorder + 0.008,
        )


def _active_sketch_factor() -> float:
    """Returns active xkcd wiggle multiplier from rcParams path.sketch."""
    sk = matplotlib.rcParams.get("path.sketch")
    if isinstance(sk, (tuple, list)) and len(sk) >= 1:
        scale_val = float(sk[0])
        if scale_val > 0.0:
            return float(np.clip(scale_val, 0.45, 1.8))
    return 0.0


def _bond_phases(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Computes deterministic wiggle phases from unshifted bond endpoints."""
    seed_f = (
        997.0 * float(a[0])
        + 613.0 * float(a[1])
        + 353.0 * float(b[0])
        + 179.0 * float(b[1])
    )
    two_pi = 2.0 * math.pi
    return (
        math.fmod(abs(seed_f), two_pi),
        math.fmod(abs(seed_f * 1.6180339), two_pi),
    )


def _wiggled_bond_curve(
    p_start: np.ndarray,
    p_end: np.ndarray,
    phases: tuple[float, float],
) -> np.ndarray:
    """Builds coaxial endpoint-pinned squiggly centerline points for xkcd."""
    vec = p_end - p_start
    length = float(np.linalg.norm(vec))
    factor = _active_sketch_factor()
    if length < 1e-6 or factor <= 0.0:
        return np.vstack([p_start, p_end])
    perp = np.array([-vec[1], vec[0]], dtype=np.float64) / length
    ts = np.linspace(0.0, 1.0, 17)
    env = np.sin(math.pi * ts)
    amp = min(0.022, 0.018 * length) * factor
    wave = 0.75 * np.sin(2.0 * math.pi * 0.65 * ts + phases[0]) + 0.25 * np.sin(
        2.0 * math.pi * 1.25 * ts + phases[1]
    )
    return np.asarray(
        p_start[None, :]
        + ts[:, None] * vec[None, :]
        + (amp * env * wave)[:, None] * perp[None, :],
        dtype=np.float64,
    )


def _draw_casing_segment(
    ax: maxes.Axes,
    p_start: np.ndarray,
    p_end: np.ndarray,
    color: str,
    edgecolor: str,
    width: float,
    zorder: float,
    phases: tuple[float, float] = (0.4, 1.1),
) -> None:
    """Draws one coaxial ink-cased segment with zero inter-layer drift."""
    pts = _wiggled_bond_curve(p_start, p_end, phases)
    xs = pts[:, 0]
    ys = pts[:, 1]
    outer_w = width + max(1.4, 0.54 * width)
    (outer_line,) = ax.plot(
        xs,
        ys,
        color=edgecolor,
        linewidth=outer_w,
        solid_capstyle="butt",
        solid_joinstyle="round",
        zorder=zorder,
    )
    outer_line.set_sketch_params(0.0, 0.0, 0.0)
    (inner_line,) = ax.plot(
        xs,
        ys,
        color=color,
        linewidth=max(0.8, width),
        solid_capstyle="butt",
        solid_joinstyle="round",
        zorder=zorder + 0.004,
    )
    inner_line.set_sketch_params(0.0, 0.0, 0.0)


def _draw_two_tone_rail(
    ax: maxes.Axes,
    p_start: np.ndarray,
    p_end: np.ndarray,
    c1: str,
    c2: str,
    edgecolor: str,
    width: float,
    zorder: float,
    split_frac: float = 0.5,
    phases: tuple[float, float] = (0.4, 1.1),
) -> None:
    """Draws one continuous ink casing around seamlessly joined color halves."""
    pts = _wiggled_bond_curve(p_start, p_end, phases)
    outer_w = width + max(1.4, 0.54 * width)
    (outer_line,) = ax.plot(
        pts[:, 0],
        pts[:, 1],
        color=edgecolor,
        linewidth=outer_w,
        solid_capstyle="butt",
        solid_joinstyle="round",
        zorder=zorder,
    )
    outer_line.set_sketch_params(0.0, 0.0, 0.0)
    inner_w = max(0.8, width)
    if c1 == c2:
        (core_line,) = ax.plot(
            pts[:, 0],
            pts[:, 1],
            color=c1,
            linewidth=inner_w,
            solid_capstyle="butt",
            solid_joinstyle="round",
            zorder=zorder + 0.004,
        )
        core_line.set_sketch_params(0.0, 0.0, 0.0)
        return

    frac = float(np.clip(split_frac, 0.05, 0.95))
    if len(pts) <= 2:
        vec = p_end - p_start
        mid = p_start + frac * vec
        overlap = p_start + min(0.98, frac + 0.025) * vec
        pts_a = np.vstack([p_start, overlap])
        pts_b = np.vstack([mid, p_end])
    else:
        n_last = len(pts) - 1
        split_idx = int(np.clip(round(frac * n_last), 1, n_last - 1))
        end_a = min(len(pts), split_idx + 2)
        pts_a = pts[:end_a]
        pts_b = pts[split_idx:]
    (half_a,) = ax.plot(
        pts_a[:, 0],
        pts_a[:, 1],
        color=c1,
        linewidth=inner_w,
        solid_capstyle="butt",
        solid_joinstyle="round",
        zorder=zorder + 0.004,
    )
    half_a.set_sketch_params(0.0, 0.0, 0.0)
    (half_b,) = ax.plot(
        pts_b[:, 0],
        pts_b[:, 1],
        color=c2,
        linewidth=inner_w,
        solid_capstyle="butt",
        solid_joinstyle="round",
        zorder=zorder + 0.005,
    )
    half_b.set_sketch_params(0.0, 0.0, 0.0)


def draw_half_bond(
    ax: maxes.Axes,
    p_start: np.ndarray,
    p_end: np.ndarray,
    color: str,
    width: float = 3.6,
    order: float = 1.0,
    edgecolor: str | None = None,
    offset_scale: float = 0.085,
    zorder: float = 2.0,
) -> None:
    """Draws a single-color ink-outlined half-bond segment using butt caps."""
    stroke = _resolve_ink(edgecolor)
    a = np.asarray(p_start, dtype=np.float64)
    b = np.asarray(p_end, dtype=np.float64)
    vec = b - a
    length = float(np.linalg.norm(vec))
    if length < 1e-6:
        return
    perp = np.array([-vec[1], vec[0]], dtype=np.float64) / length
    phases = _bond_phases(a, b)

    if abs(order - 2.0) < 0.25:
        shift = perp * offset_scale
        for sgn in (-1.0, 1.0):
            _draw_casing_segment(
                ax,
                a + sgn * shift,
                b + sgn * shift,
                color,
                stroke,
                width * 0.50,
                zorder,
                phases=phases,
            )
        return

    if abs(order - 3.0) < 0.25:
        shift = perp * (offset_scale * 1.15)
        for sgn in (-1.0, 0.0, 1.0):
            _draw_casing_segment(
                ax,
                a + sgn * shift,
                b + sgn * shift,
                color,
                stroke,
                width * 0.38,
                zorder,
                phases=phases,
            )
        return

    _draw_casing_segment(
        ax, a, b, color, stroke, width, zorder, phases=phases
    )
    if abs(order - 1.5) < 0.2:
        shift = perp * (offset_scale * 1.1)
        dash_pts = _wiggled_bond_curve(a + 0.1 * vec + shift, b + shift, phases)
        (dash_line,) = ax.plot(
            dash_pts[:, 0],
            dash_pts[:, 1],
            color=stroke,
            linewidth=max(1.0, width * 0.45),
            linestyle="--",
            zorder=zorder + 0.005,
        )
        dash_line.set_sketch_params(0.0, 0.0, 0.0)


def draw_two_tone_bond(
    ax: maxes.Axes,
    p1: np.ndarray,
    p2: np.ndarray,
    c1: str,
    c2: str,
    width: float = 3.6,
    order: float = 1.0,
    edgecolor: str | None = None,
    offset_scale: float = 0.085,
    zorder: float = 2.0,
    split_frac: float = 0.5,
) -> None:
    """Draws a unified two-tone bond with continuous aligned ink casing."""
    stroke = _resolve_ink(edgecolor)
    a = np.asarray(p1, dtype=np.float64)
    b = np.asarray(p2, dtype=np.float64)
    vec = b - a
    length = float(np.linalg.norm(vec))
    if length < 1e-6:
        return
    perp = np.array([-vec[1], vec[0]], dtype=np.float64) / length
    phases = _bond_phases(a, b)

    if abs(order - 2.0) < 0.25:
        shift = perp * offset_scale
        for sgn in (-1.0, 1.0):
            _draw_two_tone_rail(
                ax,
                a + sgn * shift,
                b + sgn * shift,
                c1,
                c2,
                stroke,
                width * 0.50,
                zorder,
                split_frac=split_frac,
                phases=phases,
            )
        return

    if abs(order - 3.0) < 0.25:
        shift = perp * (offset_scale * 1.15)
        for sgn in (-1.0, 0.0, 1.0):
            _draw_two_tone_rail(
                ax,
                a + sgn * shift,
                b + sgn * shift,
                c1,
                c2,
                stroke,
                width * 0.38,
                zorder,
                split_frac=split_frac,
                phases=phases,
            )
        return

    _draw_two_tone_rail(
        ax,
        a,
        b,
        c1,
        c2,
        stroke,
        width,
        zorder,
        split_frac=split_frac,
        phases=phases,
    )
    if abs(order - 1.5) < 0.2:
        shift = perp * (offset_scale * 1.1)
        dash_pts = _wiggled_bond_curve(
            a + 0.1 * vec + shift, b - 0.1 * vec + shift, phases
        )
        (dash_line,) = ax.plot(
            dash_pts[:, 0],
            dash_pts[:, 1],
            color=stroke,
            linewidth=max(1.0, width * 0.45),
            linestyle="--",
            zorder=zorder + 0.006,
        )
        dash_line.set_sketch_params(0.0, 0.0, 0.0)


def draw_polyhedron_triangle(
    ax: maxes.Axes,
    pts_2d: np.ndarray,
    facecolor: str,
    edgecolor: str | None = None,
    alpha: float = 0.34,
    linewidth: float = 0.85,
    zorder: float = 2.0,
) -> None:
    """Draws one translucent coordination-polyhedron triangular facet."""
    edge = darken_hex(facecolor, 0.38) if edgecolor is None else edgecolor
    poly = patches.Polygon(
        pts_2d,
        closed=True,
        facecolor=facecolor,
        edgecolor=edge,
        linewidth=linewidth,
        alpha=alpha,
        joinstyle="round",
        zorder=zorder,
    )
    ax.add_patch(poly)


def draw_polyhedron_faces(
    ax: maxes.Axes,
    verts_2d: np.ndarray,
    faces: tuple[tuple[int, int, int], ...],
    depths: np.ndarray,
    facecolor: str,
    edgecolor: str | None = None,
    alpha: float = 0.34,
    base_zorder: float = 1.5,
) -> None:
    """Draws depth-sorted translucent triangular polyhedron facets."""
    if not faces:
        return
    ordered = sorted(
        faces,
        key=lambda tri: float(
            (depths[tri[0]] + depths[tri[1]] + depths[tri[2]]) / 3.0
        ),
    )
    for step, (i0, i1, i2) in enumerate(ordered):
        pts = np.vstack([verts_2d[i0], verts_2d[i1], verts_2d[i2]])
        draw_polyhedron_triangle(
            ax,
            pts,
            facecolor=facecolor,
            edgecolor=edgecolor,
            alpha=alpha,
            zorder=base_zorder + step * 1e-4,
        )


def _draw_edge_axis_labels(
    ax: maxes.Axes,
    corners_2d: np.ndarray,
    farthest_idx: int,
    zorder: float,
) -> None:
    """Places crisp black ``a``, ``b``, ``c`` labels outside cell edges."""
    center_2d = np.mean(corners_2d, axis=0)
    diag = float(
        np.linalg.norm(np.max(corners_2d, axis=0) - np.min(corners_2d, axis=0))
    )
    offset = max(0.38, 0.048 * diag)
    for axis_name in ("a", "b", "c"):
        candidates = []
        for i_idx, j_idx, tag in _CELL_EDGE_PAIRS:
            if tag != axis_name:
                continue
            p_i = corners_2d[i_idx]
            p_j = corners_2d[j_idx]
            edge_len = float(np.linalg.norm(p_j - p_i))
            if edge_len >= 0.25:
                candidates.append((i_idx, j_idx, p_i, p_j))
        if not candidates:
            continue
        front = [c for c in candidates if farthest_idx not in (c[0], c[1])]
        pool = front if front else candidates
        if axis_name == "c":
            chosen = max(
                pool,
                key=lambda c: abs(
                    0.5 * (float(c[2][0]) + float(c[3][0]))
                    - float(center_2d[0])
                ),
            )
        else:
            chosen = min(
                pool,
                key=lambda c: 0.5 * (float(c[2][1]) + float(c[3][1])),
            )
        p_a, p_b = chosen[2], chosen[3]
        mid = 0.5 * (p_a + p_b)
        vec = p_b - p_a
        norm_vec = np.array([-vec[1], vec[0]], dtype=np.float64)
        norm_vec /= max(float(np.linalg.norm(norm_vec)), 1e-9)
        if float(np.dot(norm_vec, mid - center_2d)) < 0.0:
            norm_vec = -norm_vec
        pos = mid + offset * norm_vec
        ax.text(
            float(pos[0]),
            float(pos[1]),
            axis_name,
            fontsize=11.0,
            color=_resolve_ink(),
            ha="center",
            va="center",
            zorder=zorder,
        )


def draw_origin_tripod(
    ax: maxes.Axes,
    cell: np.ndarray,
    camera: geometry.Camera,
    center: np.ndarray | None = None,
    origin: np.ndarray | None = None,
    zorder: float = 500.0,
    *,
    span: float | None = None,
) -> None:
    """Draws compact RGB ``a, b, c`` arrows anchored at the cell origin."""
    orig_3d = (
        np.zeros(3, dtype=np.float64)
        if origin is None
        else np.asarray(origin, dtype=np.float64)
    )
    axis_tips = np.vstack(
        [
            orig_3d,
            orig_3d + cell[0] * 0.30,
            orig_3d + cell[1] * 0.30,
            orig_3d + cell[2] * 0.30,
        ]
    )
    proj_ctr = (
        orig_3d + 0.5 * (cell[0] + cell[1] + cell[2])
        if center is None
        else center
    )
    tips_2d, depths = geometry.project_points(
        axis_tips, camera, center=proj_ctr, span=span
    )
    axis_colors = ("#D55E00", "#009E73", "#0072B2")
    axis_names = ("a", "b", "c")
    ax.scatter(
        [float(tips_2d[0][0])],
        [float(tips_2d[0][1])],
        s=18.0,
        color=_resolve_ink(),
        zorder=zorder + 0.3,
    )
    order = sorted(range(3), key=lambda k: float(depths[k + 1]))
    leader_lw = round(
        max(
            1.05,
            0.70 * float(matplotlib.rcParams.get("lines.linewidth", 1.8)),
        ),
        2,
    )
    for rank, k in enumerate(order):
        dx = float(tips_2d[k + 1][0] - tips_2d[0][0])
        dy = float(tips_2d[k + 1][1] - tips_2d[0][1])
        seg_len = math.hypot(dx, dy)
        if seg_len > 0.12:
            ax.annotate(
                "",
                xy=(float(tips_2d[k + 1][0]), float(tips_2d[k + 1][1])),
                xytext=(float(tips_2d[0][0]), float(tips_2d[0][1])),
                arrowprops={
                    "arrowstyle": "-|>,head_length=0.35,head_width=0.22",
                    "color": axis_colors[k],
                    "lw": leader_lw,
                },
                zorder=zorder + 0.05 * rank,
            )
            ux, uy = dx / seg_len, dy / seg_len
            pad_lbl = max(0.28, 0.20 * seg_len)
            ax.text(
                float(tips_2d[k + 1][0]) + pad_lbl * ux,
                float(tips_2d[k + 1][1]) + pad_lbl * uy,
                axis_names[k],
                fontsize=10.0,
                color=axis_colors[k],
                ha="center",
                va="center",
                zorder=zorder + 0.6,
            )


def _crystal_axes_origin(
    ax: maxes.Axes,
    vecs: np.ndarray,
    location: str,
) -> tuple[float, float]:
    """Computes clean non-clipping corner anchor ``(ox, oy)`` for axes triad."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    pad_x = 0.065 * max(x1 - x0, 1.0)
    pad_y = 0.065 * max(y1 - y0, 1.0)
    loc = location.strip().lower()
    ox = (
        x1 - pad_x - max(0.0, float(np.max(vecs[:, 0])))
        if loc in ("bottom_right", "top_right")
        else x0 + pad_x - min(0.0, float(np.min(vecs[:, 0])))
    )
    oy = (
        y1 - pad_y - max(0.0, float(np.max(vecs[:, 1])))
        if loc in ("top_left", "top_right")
        else y0 + pad_y - min(0.0, float(np.min(vecs[:, 1])))
    )
    return ox, oy


def draw_crystal_axes(
    ax: maxes.Axes,
    cell: np.ndarray,
    camera: geometry.Camera,
    *,
    location: str = "bottom_left",
    size_frac: float = 0.13,
    colors: tuple[str, str, str] = ("#D55E00", "#009E73", "#0072B2"),
    labels: tuple[str, str, str] = ("a", "b", "c"),
    zorder: float = 600.0,
) -> None:
    """Draws a standalone crystallographic ``a, b, c`` orientation triad.

    Positioned cleanly in an axes corner separate from the unit-cell box while
    preserving exact 3D camera parallelism with the rendered crystal.

    Args:
      ax: Target Matplotlib ``Axes`` (with data limits already set).
      cell: Lattice matrix ``(3, 3)`` whose rows are ``a, b, c`` vectors.
      camera: Active viewing ``Camera`` instance.
      location: Corner placement: ``"bottom_left"`` (or ``"corner"``),
        ``"bottom_right"``, ``"top_left"``, or ``"top_right"``.
      size_frac: Arrow length as a fraction of the smaller axis span.
      colors: Color triple for ``(a, b, c)``.
      labels: Label triple for ``(a, b, c)``.
      zorder: Base z-order above the structure and unit cell.
    """
    cell_arr = np.asarray(cell, dtype=np.float64)
    row_norms = np.maximum(
        np.linalg.norm(cell_arr, axis=1, keepdims=True), 1e-12
    )
    unit_vecs = cell_arr / row_norms
    basis = geometry.rotation_matrix(
        camera.azimuth, camera.elevation, camera.roll
    )
    proj = unit_vecs @ basis.T
    x_span = max(ax.get_xlim()[1] - ax.get_xlim()[0], 1.0)
    y_span = max(ax.get_ylim()[1] - ax.get_ylim()[0], 1.0)
    arrow_len = size_frac * min(x_span, y_span)
    vecs = proj[:, :2] * arrow_len
    ox, oy = _crystal_axes_origin(ax, vecs, location)
    leader_lw = round(
        max(
            1.05,
            0.70 * float(matplotlib.rcParams.get("lines.linewidth", 1.8)),
        ),
        2,
    )

    ax.scatter([ox], [oy], s=18.0, color=_resolve_ink(), zorder=zorder + 0.4)
    for rank, k in enumerate(sorted(range(3), key=lambda i: float(proj[i, 2]))):
        vx, vy = float(vecs[k, 0]), float(vecs[k, 1])
        seg_len = math.hypot(vx, vy)
        if seg_len < 1e-4:
            continue
        tx, ty = ox + vx, oy + vy
        ax.annotate(
            "",
            xy=(tx, ty),
            xytext=(ox, oy),
            arrowprops={
                "arrowstyle": "-|>,head_length=0.35,head_width=0.22",
                "color": colors[k],
                "lw": leader_lw,
            },
            zorder=zorder + 0.05 * rank,
        )
        lbl_pad = max(0.24 * arrow_len, 0.25)
        ax.text(
            tx + lbl_pad * (vx / seg_len),
            ty + lbl_pad * (vy / seg_len),
            labels[k],
            fontsize=10.5,
            color=colors[k],
            ha="center",
            va="center",
            zorder=zorder + 0.6,
        )


def draw_unit_cell_box(
    ax: maxes.Axes,
    cell: np.ndarray,
    camera: geometry.Camera,
    center: np.ndarray | None = None,
    origin: np.ndarray | None = None,
    edgecolor: str | None = None,
    linewidth: float = 1.35,
    draw_back: bool = True,
    draw_front: bool = True,
    draw_front_edges: bool = True,
    base_plane_color: str | None = palettes.VACUUM_TINT,
    cell_axes: str = "edges",
    front_zorder: float = 500.0,
    *,
    span: float | None = None,
) -> list[np.ndarray]:
    """Draws a clean crystallographic unit cell with back/front depth splitting.

    Returns:
      List of projected corner 2D coordinates for bounding-box sizing.
    """
    stroke = _resolve_ink(edgecolor)
    corners_2d, _, farthest_idx = geometry.cell_corners_projected(
        cell, camera, center=center, origin=origin, span=span
    )
    if draw_back:
        if base_plane_color is not None:
            base_quad = corners_2d[[0, 1, 4, 2]]
            ax.add_patch(
                patches.Polygon(
                    base_quad,
                    closed=True,
                    facecolor=base_plane_color,
                    edgecolor="none",
                    alpha=0.72,
                    zorder=0.55,
                )
            )
        for i_idx, j_idx, _ in _CELL_EDGE_PAIRS:
            if farthest_idx in (i_idx, j_idx):
                ax.plot(
                    [float(corners_2d[i_idx][0]), float(corners_2d[j_idx][0])],
                    [float(corners_2d[i_idx][1]), float(corners_2d[j_idx][1])],
                    color="#777777",
                    linestyle=(0, (4, 3)),
                    linewidth=max(0.75, round(0.65 * linewidth, 2)),
                    zorder=1.0,
                )
    if draw_front:
        if draw_front_edges:
            for i_idx, j_idx, _ in _CELL_EDGE_PAIRS:
                if farthest_idx not in (i_idx, j_idx):
                    ax.plot(
                        [
                            float(corners_2d[i_idx][0]),
                            float(corners_2d[j_idx][0]),
                        ],
                        [
                            float(corners_2d[i_idx][1]),
                            float(corners_2d[j_idx][1]),
                        ],
                        color=stroke,
                        linestyle="-",
                        linewidth=linewidth,
                        solid_capstyle="round",
                        zorder=front_zorder,
                    )
        axes_mode = cell_axes.lower()
        if axes_mode == "edges":
            _draw_edge_axis_labels(
                ax, corners_2d, farthest_idx, zorder=front_zorder + 1.0
            )
        elif axes_mode in ("tripod", "origin"):
            draw_origin_tripod(
                ax,
                cell,
                camera,
                center=center,
                origin=origin,
                zorder=front_zorder + 1.0,
                span=span,
            )
    return [corners_2d[k] for k in range(8)]


def _format_hkl_label(hkl: tuple[int, int, int]) -> str:
    """Formats Miller indices cleanly, adding spaces when multi-char."""
    parts = [str(int(v)) for v in hkl]
    if all(len(p) == 1 for p in parts):
        return f"({''.join(parts)})"
    return f"({' '.join(parts)})"


def draw_miller_plane(
    ax: maxes.Axes,
    cell: np.ndarray,
    hkl: tuple[int, int, int],
    camera: geometry.Camera,
    center: np.ndarray | None = None,
    facecolor: str = "#56B4E9",
    edgecolor: str = "#0072B2",
    alpha: float = 0.30,
    show_label: bool = True,
    zorder: float = 2.5,
    label_zorder: float | None = None,
    *,
    span: float | None = None,
) -> None:
    """Draws a translucent crystallographic ``(hkl)`` slice inside the cell."""
    poly_3d = geometry.miller_plane_polygon(cell, hkl)
    if poly_3d.shape[0] < 3:
        return
    proj_ctr = 0.5 * (cell[0] + cell[1] + cell[2]) if center is None else center
    poly_2d, _ = geometry.project_points(
        poly_3d, camera, center=proj_ctr, span=span
    )
    patch = patches.Polygon(
        poly_2d,
        closed=True,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.25,
        alpha=alpha,
        joinstyle="round",
        zorder=zorder,
    )
    ax.add_patch(patch)
    ax.scatter(
        poly_2d[:, 0],
        poly_2d[:, 1],
        s=14.0,
        color=edgecolor,
        edgecolor=palettes.PAPER,
        linewidth=0.65,
        zorder=zorder + 0.008,
    )
    if show_label:
        centroid = np.mean(poly_2d, axis=0)
        lbl_z = zorder + 0.2 if label_zorder is None else label_zorder
        bg = str(matplotlib.rcParams.get("axes.facecolor", palettes.PAPER))
        ink = resolve_ink()
        edge_clean = ensure_readable_text_color(edgecolor, bg)
        badge = ax.text(
            float(centroid[0]),
            float(centroid[1]),
            _format_hkl_label(hkl),
            fontsize=9.5,
            color=ink,
            ha="center",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": bg,
                "edgecolor": edge_clean,
                "alpha": 0.90,
            },
            zorder=lbl_z,
        )
        bbox_patch = badge.get_bbox_patch()
        if bbox_patch is not None:
            bbox_patch.set_sketch_params(0.0, 0.0, 0.0)
