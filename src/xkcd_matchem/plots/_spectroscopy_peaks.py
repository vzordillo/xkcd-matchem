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
"""Stick scaling and dense-peak anti-collision annotation for 1D spectra."""

from __future__ import annotations

import dataclasses

from matplotlib import axes as maxes
import numpy as np

from xkcd_matchem.core import spectroscopy
from xkcd_matchem.style import theme


@dataclasses.dataclass(frozen=True)
class _PeakCandidate:
    """Normalized peak descriptor used during anti-collision label layout."""

    pos: float
    y_loc: float
    label: str
    u_peak: float
    width_u: float


@dataclasses.dataclass(frozen=True)
class _PeakContext:
    """Shared axis geometry and styling context for spectrum peak labels."""

    xs_s: np.ndarray
    ys_s: np.ndarray
    x_lo: float
    x_hi: float
    y_max: float
    ink_col: str
    bg_color: str
    should_invert: bool
    lw_leader: float

    @property
    def x_span(self) -> float:
        """Returns positive horizontal span of the spectrum."""
        return max(self.x_hi - self.x_lo, 1e-6)

    def u_to_x(self, u_val: float) -> float:
        """Maps normalized visual left-to-right ``u`` back to data ``x``."""
        if self.should_invert:
            return self.x_hi - u_val * self.x_span
        return self.x_lo + u_val * self.x_span


def resolve_stick_scale(spectrum: spectroscopy.Spectrum) -> float:
    """Computes stick scaling without distorting already-matched intensities."""
    if not spectrum.peaks or spectrum.y.size == 0:
        return 1.0
    y_max_curve = max(float(np.max(spectrum.y)), 1e-6)
    p_max = max((p.intensity for p in spectrum.peaks), default=1.0)
    if 0.25 * y_max_curve <= p_max <= 4.0 * y_max_curve:
        return 1.0
    return y_max_curve / max(p_max, 1e-6)


def peak_label_offset(
    pos: float,
    y_loc: float,
    xs_s: np.ndarray,
    ys_s: np.ndarray,
    y_max: float,
    should_invert: bool,
) -> tuple[int, int, str]:
    """Computes peak tag offset (pt) nudged clear of taller nearby slopes."""
    if xs_s.size < 2:
        return 0, 8, "center"
    x_span = max(float(xs_s[-1] - xs_s[0]), 1e-6)
    win = 0.045 * x_span
    lo_mask = (xs_s >= pos - win) & (xs_s < pos)
    hi_mask = (xs_s > pos) & (xs_s <= pos + win)
    tall_lo = float(np.max(ys_s[lo_mask])) if np.any(lo_mask) else y_loc
    tall_hi = float(np.max(ys_s[hi_mask])) if np.any(hi_mask) else y_loc
    thresh = y_loc + 0.10 * max(y_max, 1e-6)
    if tall_lo > thresh and tall_lo >= tall_hi:
        dx = -6 if should_invert else 6
        return dx, 9, ("right" if dx < 0 else "left")
    if tall_hi > thresh and tall_hi > tall_lo:
        dx = 6 if should_invert else -6
        return dx, 9, ("right" if dx < 0 else "left")
    return 0, 8, "center"


def _build_peak_candidates(
    top_peaks: list[spectroscopy.Peak],
    ctx: _PeakContext,
    scale: float,
) -> tuple[list[_PeakCandidate], float]:
    """Converts top peaks into visual left-to-right sorted candidates."""
    candidates: list[_PeakCandidate] = []
    top_y = 0.0
    for pk in top_peaks:
        in_range = (
            ctx.xs_s.size >= 2 and ctx.xs_s[0] <= pk.position <= ctx.xs_s[-1]
        )
        y_curve = (
            float(np.interp(pk.position, ctx.xs_s, ctx.ys_s))
            if in_range
            else 0.0
        )
        y_loc = max(pk.intensity * scale, y_curve)
        top_y = max(top_y, y_loc)
        lbl = pk.label if pk.label else f"{pk.position:.1f}"
        u_pk = (
            (ctx.x_hi - pk.position) / ctx.x_span
            if ctx.should_invert
            else (pk.position - ctx.x_lo) / ctx.x_span
        )
        w_u = max(0.072, 0.016 * float(len(lbl)) + 0.026)
        candidates.append(
            _PeakCandidate(
                pos=pk.position,
                y_loc=y_loc,
                label=lbl,
                u_peak=u_pk,
                width_u=w_u,
            )
        )
    candidates.sort(key=lambda item: item.u_peak)
    return candidates, top_y


def _cluster_peak_candidates(
    candidates: list[_PeakCandidate],
    gap_u: float = 0.016,
) -> list[list[_PeakCandidate]]:
    """Groups horizontally colliding peaks into contiguous clusters."""
    if not candidates:
        return []
    clusters: list[list[_PeakCandidate]] = [[candidates[0]]]
    for cand in candidates[1:]:
        prev = clusters[-1][-1]
        min_sep = 0.5 * (prev.width_u + cand.width_u) + gap_u
        if cand.u_peak - prev.u_peak < min_sep:
            clusters[-1].append(cand)
        else:
            clusters.append([cand])
    return clusters


def _repel_cluster_horizontal(
    cluster: list[_PeakCandidate],
    bounds_u: tuple[float, float],
    gap_u: float = 0.014,
) -> list[float]:
    """Repels horizontal label slots monotonically inside ``bounds_u``."""
    n_c = len(cluster)
    lo_u, hi_u = bounds_u
    max_drift = min(0.18, 0.030 * float(n_c))
    u_pos = [float(np.clip(c.u_peak, lo_u, hi_u)) for c in cluster]
    if n_c <= 1:
        return u_pos

    for _ in range(20):
        moved = False
        for idx in range(n_c - 1):
            req = 0.5 * (cluster[idx].width_u + cluster[idx + 1].width_u)
            sep = u_pos[idx + 1] - u_pos[idx]
            if sep < req + gap_u:
                delta = 0.45 * (req + gap_u - sep)
                u_pos[idx] -= delta
                u_pos[idx + 1] += delta
                moved = True
        for idx, cand in enumerate(cluster):
            low_lim = max(lo_u, cand.u_peak - max_drift)
            high_lim = min(hi_u, cand.u_peak + max_drift)
            u_pos[idx] = float(np.clip(u_pos[idx], low_lim, high_lim))
        for idx in range(1, n_c):
            if u_pos[idx] < u_pos[idx - 1]:
                mid = 0.5 * (u_pos[idx - 1] + u_pos[idx])
                u_pos[idx - 1] = mid
                u_pos[idx] = mid
        if not moved:
            break

    span_now = u_pos[-1] - u_pos[0]
    min_fan = min(hi_u - lo_u, 0.028 * float(n_c - 1))
    if span_now < min_fan and n_c >= 2:
        ctr = 0.5 * (u_pos[0] + u_pos[-1])
        f_lo = max(lo_u, ctr - 0.5 * min_fan)
        f_hi = min(hi_u, f_lo + min_fan)
        f_lo = max(lo_u, f_hi - min_fan)
        u_pos = [
            float(val) for val in np.linspace(f_lo, f_hi, n_c, dtype=np.float64)
        ]
    return u_pos


def _envelope_under_label(
    x_txt: float,
    u_txt: float,
    width_u: float,
    cluster: list[_PeakCandidate],
    ctx: _PeakContext,
) -> float:
    """Computes max spectral/stick height underneath a label's x footprint."""
    half_x = 0.65 * width_u * ctx.x_span
    y_env = 0.0
    if ctx.xs_s.size >= 2:
        mask = (ctx.xs_s >= x_txt - half_x) & (ctx.xs_s <= x_txt + half_x)
        if np.any(mask):
            y_env = max(y_env, float(np.max(ctx.ys_s[mask])))
    for other in cluster:
        if abs(other.u_peak - u_txt) <= 0.65 * width_u:
            y_env = max(y_env, other.y_loc)
    return y_env


def _resolve_staggered_y(
    u_txt: float,
    width_u: float,
    base_y: float,
    tier_h: float,
    placed_boxes: list[tuple[float, float, float]],
) -> float:
    """Iteratively raises ``y`` until clear of all placed label boxes."""
    y_lbl = base_y
    y_tol = tier_h * (1.0 - 1e-6)
    for _ in range(len(placed_boxes) + 2):
        bumped = False
        for up, yp, wp in placed_boxes:
            near_x = abs(u_txt - up) < 0.5 * (width_u + wp) + 0.018
            near_y = abs(y_lbl - yp) < y_tol
            if near_x and near_y:
                new_y = yp + tier_h
                if new_y > y_lbl:
                    y_lbl = new_y
                    bumped = True
        if not bumped:
            break
    return y_lbl


def _place_singleton_peak(
    ax: maxes.Axes,
    cand: _PeakCandidate,
    ctx: _PeakContext,
) -> tuple[float, tuple[float, float, float]]:
    """Places one isolated peak annotation using slope-aware point offsets."""
    dx_pt, dy_pt, ha = peak_label_offset(
        cand.pos,
        cand.y_loc,
        ctx.xs_s,
        ctx.ys_s,
        ctx.y_max,
        ctx.should_invert,
    )
    ann = ax.annotate(
        cand.label,
        xy=(cand.pos, cand.y_loc),
        xytext=(dx_pt, dy_pt),
        textcoords="offset points",
        ha=ha,
        fontsize=8.5,
        color=ctx.ink_col,
        bbox=theme.label_halo_bbox(ctx.bg_color, pad=0.05, alpha=0.84),
        zorder=6.0,
    )
    patch = ann.get_bbox_patch()
    if patch is not None:
        patch.set_sketch_params(0.0, 0.0, 0.0)
    y_box = cand.y_loc + 0.055 * ctx.y_max
    return cand.y_loc + 0.08 * ctx.y_max, (cand.u_peak, y_box, cand.width_u)


def _place_dense_cluster(
    ax: maxes.Axes,
    cluster: list[_PeakCandidate],
    bounds_u: tuple[float, float],
    placed_boxes: list[tuple[float, float, float]],
    ctx: _PeakContext,
) -> float:
    """Staggers clustered peak labels vertically and horizontally w/ leaders."""
    u_slots = _repel_cluster_horizontal(cluster, bounds_u)
    tier_h = 0.125 * ctx.y_max
    top_y = 0.0

    for cand, u_txt in zip(cluster, u_slots, strict=True):
        x_txt = ctx.u_to_x(u_txt)
        y_env = _envelope_under_label(x_txt, u_txt, cand.width_u, cluster, ctx)
        base_y = max(cand.y_loc, y_env) + 0.08 * ctx.y_max
        y_lbl = _resolve_staggered_y(
            u_txt, cand.width_u, base_y, tier_h, placed_boxes
        )
        placed_boxes.append((u_txt, y_lbl, cand.width_u))
        top_y = max(top_y, y_lbl + 0.095 * ctx.y_max)
        ann = ax.annotate(
            cand.label,
            xy=(cand.pos, cand.y_loc),
            xytext=(x_txt, y_lbl),
            textcoords="data",
            ha="center",
            va="bottom",
            fontsize=8.3,
            color=ctx.ink_col,
            bbox=theme.label_halo_bbox(ctx.bg_color, pad=0.05, alpha=0.88),
            arrowprops={
                "arrowstyle": "-",
                "color": ctx.ink_col,
                "lw": ctx.lw_leader,
                "alpha": 0.72,
                "shrinkA": 1.5,
                "shrinkB": 1.5,
            },
            zorder=6.0,
        )
        patch = ann.get_bbox_patch()
        if patch is not None:
            patch.set_sketch_params(0.0, 0.0, 0.0)
        if ann.arrow_patch is not None:
            ann.arrow_patch.set_sketch_params(0.0, 0.0, 0.0)
    return top_y


def _cluster_bounds_u(
    clusters: list[list[_PeakCandidate]],
    cl_idx: int,
) -> tuple[float, float]:
    """Computes horizontal normalized bounds allocated to cluster ``cl_idx``."""
    cluster = clusters[cl_idx]
    left_lim = 0.055
    if cl_idx > 0:
        prev_edge = clusters[cl_idx - 1][-1].u_peak + 0.04
        left_lim = max(left_lim, min(cluster[0].u_peak, prev_edge))
    right_lim = 0.945
    if cl_idx + 1 < len(clusters):
        next_edge = clusters[cl_idx + 1][0].u_peak - 0.04
        right_lim = min(right_lim, max(cluster[-1].u_peak, next_edge))
    if right_lim < left_lim:
        return 0.055, 0.945
    return left_lim, right_lim


def _make_peak_context(
    top_peaks: list[spectroscopy.Peak],
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    scale: float,
    style_cfg: tuple[str, str, bool],
    leader_lw: float | None,
) -> _PeakContext:
    """Constructs sorted curve bounds and style parameters for peak labels."""
    ink_col, bg_color, should_invert = style_cfg
    if x_arr.size >= 2:
        order = np.argsort(x_arr)
        xs_s, ys_s = x_arr[order], y_arr[order]
        x_lo, x_hi = float(xs_s[0]), float(xs_s[-1])
    else:
        xs_s, ys_s = x_arr, y_arr
        p_xs = [pk.position for pk in top_peaks]
        x_lo, x_hi = min(p_xs) - 1.0, max(p_xs) + 1.0
    y_max = max(
        float(np.max(ys_s)) if ys_s.size else 0.0,
        max((pk.intensity * scale for pk in top_peaks), default=0.0),
        1e-6,
    )
    lw_val = (
        theme.stroke_weights("classic").leader
        if leader_lw is None
        else float(leader_lw)
    )
    return _PeakContext(
        xs_s=xs_s,
        ys_s=ys_s,
        x_lo=x_lo,
        x_hi=x_hi,
        y_max=y_max,
        ink_col=ink_col,
        bg_color=bg_color,
        should_invert=should_invert,
        lw_leader=lw_val,
    )


def annotate_spectrum_peaks(
    ax: maxes.Axes,
    peaks: list[spectroscopy.Peak],
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    scale: float,
    annotate_peaks: int,
    style_cfg: tuple[str, str, bool],
    *,
    leader_lw: float | None = None,
) -> float:
    """Annotates top spectral peaks with dense-cluster anti-collision."""
    if annotate_peaks <= 0 or not peaks:
        return 0.0
    top_peaks = sorted(peaks, key=lambda p: p.intensity, reverse=True)[
        :annotate_peaks
    ]
    ctx = _make_peak_context(
        top_peaks, x_arr, y_arr, scale, style_cfg, leader_lw
    )
    candidates, top_y = _build_peak_candidates(top_peaks, ctx, scale)
    clusters = _cluster_peak_candidates(candidates)
    placed_boxes: list[tuple[float, float, float]] = []
    for cl_idx, cluster in enumerate(clusters):
        if len(cluster) == 1:
            s_top, s_box = _place_singleton_peak(ax, cluster[0], ctx)
            placed_boxes.append(s_box)
            top_y = max(top_y, s_top)
            continue
        bounds_u = _cluster_bounds_u(clusters, cl_idx)
        cl_top = _place_dense_cluster(ax, cluster, bounds_u, placed_boxes, ctx)
        top_y = max(top_y, cl_top)
    return top_y
