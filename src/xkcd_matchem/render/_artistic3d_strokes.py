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
"""Pen-and-ink silhouettes, hatching, contours, scumbles, & stipples."""

from __future__ import annotations

from collections.abc import Sequence
import math

from matplotlib import colors as mcolors
import numpy as np

from xkcd_matchem.core import isosurface
from xkcd_matchem.render import _artistic3d_raster as raster


LineBatch = tuple[
    list[np.ndarray],
    str,
    float,
    float,
    float,
    tuple[float, float, float] | None,
]


def sample_mesh_surface(
    mesh: isosurface.IsosurfaceMesh,
    count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministically samples stratified R2 blue-noise mesh barycentrics."""
    if len(mesh.faces) == 0 or count <= 0:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 4), dtype=np.float64),
        )
    v0 = mesh.vertices[mesh.faces[:, 0]]
    e1 = mesh.vertices[mesh.faces[:, 1]] - v0
    e2 = mesh.vertices[mesh.faces[:, 2]] - v0
    areas = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
    tot = float(np.sum(areas))
    if tot <= 1e-12:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 4), dtype=np.float64),
        )
    cdf = np.cumsum(areas) / tot
    off = np.random.default_rng(seed).random(4)
    idx = np.arange(count, dtype=np.float64)
    u_cdf = ((idx + float(off[0])) / float(count)) % 1.0
    f_idx = np.clip(np.searchsorted(cdf, u_cdf), 0, len(areas) - 1)
    r1 = (idx * 0.7548776662466927 + float(off[1])) % 1.0
    r2 = (idx * 0.5698402909980532 + float(off[2])) % 1.0
    fold = (r1 + r2) > 1.0
    r1 = np.where(fold, 1.0 - r1, r1)
    r2 = np.where(fold, 1.0 - r2, r2)
    bary = np.column_stack([1.0 - r1 - r2, r1, r2])
    attrs = np.column_stack(
        [
            (idx * 0.6180339887498949 + float(off[3])) % 1.0,
            (idx * 0.4142135623730951 + float(off[1])) % 1.0,
            (idx * 0.7071067811865476 + float(off[2])) % 1.0,
            (idx * 0.3027756377319946 + float(off[0])) % 1.0,
        ]
    )
    return f_idx, bary, attrs


def smooth_polyline(pts: np.ndarray, passes: int = 2) -> np.ndarray:
    """Applies Chaikin-like fairing to remove voxel stair-step jaggedness."""
    if len(pts) < 3:
        return pts
    arr = np.asarray(pts, dtype=np.float64).copy()
    is_closed = bool(np.allclose(arr[0], arr[-1], atol=1e-6))
    for _ in range(passes):
        if is_closed and len(arr) >= 4:
            core = arr[:-1]
            sm = (
                0.25 * np.roll(core, 1, axis=0)
                + 0.50 * core
                + 0.25 * np.roll(core, -1, axis=0)
            )
            arr = np.vstack([sm, sm[0]])
        else:
            arr[1:-1] = 0.25 * arr[:-2] + 0.50 * arr[1:-1] + 0.25 * arr[2:]
    return np.asarray(arr, dtype=np.float64)


def _chain_edges(
    edges: np.ndarray | Sequence[tuple[int, int]],
    pts: np.ndarray,
    min_len: int = 3,
) -> list[np.ndarray]:
    """Chains unordered vertex-index edges into continuous N-D polylines."""
    if len(edges) == 0:
        return []
    adj: dict[int, list[int]] = {}
    for pair in edges:
        u, v = int(pair[0]), int(pair[1])
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
    visited: set[tuple[int, int]] = set()
    paths: list[np.ndarray] = []
    for start, nbrs in adj.items():
        for nxt in nbrs:
            e_key = (start, nxt) if start < nxt else (nxt, start)
            if e_key in visited:
                continue
            chain = [start]
            cur = nxt
            visited.add(e_key)
            while True:
                chain.append(cur)
                if cur == start:
                    break
                moved = False
                for cand in adj.get(cur, ()):
                    c_key = (cur, cand) if cur < cand else (cand, cur)
                    if c_key not in visited:
                        visited.add(c_key)
                        cur = cand
                        moved = True
                        break
                if not moved:
                    break
            if len(chain) >= min_len:
                paths.append(pts[np.asarray(chain, dtype=np.int64)])
    return paths


def extract_silhouette_3d(
    mesh: isosurface.IsosurfaceMesh,
    xy_v: np.ndarray,
    depth_v: np.ndarray,
    front_mask: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Extracts smoothed silhouette ``xy`` loops and matching ``z`` arrays."""
    if len(mesh.faces) == 0 or len(xy_v) == 0:
        return [], []
    faces = mesh.faces
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges_sorted = np.sort(edges, axis=1)
    tri_front = np.concatenate([front_mask, front_mask, front_mask])
    stride = max(int(len(xy_v)), 1)
    keys = (
        edges_sorted[:, 0].astype(np.int64) * stride
        + edges_sorted[:, 1].astype(np.int64)
    )
    order = np.argsort(keys, kind="mergesort")
    k_s = keys[order]
    f_s = tri_front[order].astype(np.int64)
    is_new = np.empty(len(k_s), dtype=bool)
    is_new[0] = True
    is_new[1:] = k_s[1:] != k_s[:-1]
    starts = np.flatnonzero(is_new)
    counts = np.diff(np.append(starts, len(k_s)))
    front_sums = np.add.reduceat(f_s, starts)
    sil_mask = ((counts == 2) & (front_sums == 1)) | (
        (counts == 1) & (front_sums == 1)
    )
    if not np.any(sil_mask):
        return [], []
    sil_pairs = edges_sorted[order[starts[sil_mask]]]
    xyz_v = np.column_stack([xy_v, depth_v])
    loops = [
        smooth_polyline(p, passes=3)
        for p in _chain_edges(sil_pairs, xyz_v, min_len=3)
    ]
    return [lp[:, :2] for lp in loops], [lp[:, 2] for lp in loops]


def _row_intervals(
    u_tri: np.ndarray,
    v_tri: np.ndarray,
    v_val: float,
) -> list[tuple[float, float]]:
    """Computes merged 1D intervals intersected by ``v = v_val``."""
    min_v = np.min(v_tri, axis=1)
    max_v = np.max(v_tri, axis=1)
    hits = np.flatnonzero((min_v < v_val) & (max_v > v_val))
    if hits.size == 0:
        return []
    vh = v_tri[hits]
    uh = u_tri[hits]
    vb = np.roll(vh, -1, axis=1)
    ub = np.roll(uh, -1, axis=1)
    denom = vb - vh
    cross = ((vh <= v_val) & (v_val < vb)) | ((vb <= v_val) & (v_val < vh))
    safe = cross & (np.abs(denom) > 1e-12)
    frac = np.where(safe, (v_val - vh) / np.where(safe, denom, 1.0), 0.0)
    u_int = uh + frac * (ub - uh)
    has2 = np.sum(safe, axis=1) >= 2
    if not np.any(has2):
        return []
    u_sel = u_int[has2]
    s_sel = safe[has2]
    lo_arr = np.min(np.where(s_sel, u_sel, np.inf), axis=1)
    hi_arr = np.max(np.where(s_sel, u_sel, -np.inf), axis=1)
    keep = np.isfinite(lo_arr) & np.isfinite(hi_arr)
    if not np.any(keep):
        return []
    lo_arr, hi_arr = lo_arr[keep], hi_arr[keep]
    order = np.argsort(lo_arr)
    merged: list[tuple[float, float]] = [
        (float(lo_arr[order[0]]), float(hi_arr[order[0]]))
    ]
    for idx in order[1:]:
        lo, hi = float(lo_arr[idx]), float(hi_arr[idx])
        if lo <= merged[-1][1] + 2e-4:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def _clamp_to_bounds(
    sa: float,
    sb: float,
    bounds: Sequence[tuple[float, float]],
    inset: float,
) -> tuple[float, float] | None:
    """Clamps a shadow stroke interval inside its enclosing lobe span."""
    for ba, bb in bounds:
        if sb >= ba and sa <= bb:
            lo = max(sa, ba + inset)
            hi = min(sb, bb - inset)
            if hi > lo:
                return (lo, hi)
    return None


def _clip_row_chords(
    segs: Sequence[tuple[float, float]],
    bnds: Sequence[tuple[float, float]],
    *,
    vk: float,
    inset: float,
    min_len: float,
    spacing: float,
    span: float,
    rng: np.random.Generator,
) -> list[tuple[float, float, float, float, float]]:
    """Feathers interior shadow chord ends and leaves reflected-light gap."""
    out: list[tuple[float, float, float, float, float]] = []
    for sa, sb in segs:
        clamped = _clamp_to_bounds(sa, sb, bnds, inset)
        if clamped is None:
            continue
        ca, cb = clamped
        for ba, bb in bnds:
            if ca >= ba - 1e-4 and cb <= bb + 1e-4:
                if ca > ba + inset + 0.015 * span:
                    ca += float(rng.uniform(-0.30, 0.55) * spacing)
                if cb < bb - inset - 0.015 * span:
                    cb -= float(rng.uniform(-0.30, 0.55) * spacing)
                ca = max(ca, ba + inset)
                cb = min(cb, bb - inset)
                if cb - ca >= min_len:
                    out.append((ca, cb, vk, ba + inset, bb - inset))
                break
    return out


def _clip_xkcd_chords(
    sub_tris: np.ndarray,
    bound_tris: np.ndarray,
    *,
    angle_deg: float,
    spacing: float,
    span: float,
    offset_frac: float = 0.0,
    seed: int = 11,
) -> list[tuple[float, float, float, float, float]]:
    """Returns feathered ``(ua, ub, vk, ba, bb)`` shadow pen chords."""
    if len(sub_tris) == 0 or len(bound_tris) == 0 or spacing <= 1e-9:
        return []
    rad = math.radians(angle_deg)
    e_u = np.array([math.cos(rad), math.sin(rad)], dtype=np.float64)
    e_v = np.array([-math.sin(rad), math.cos(rad)], dtype=np.float64)
    u_sub, v_sub = sub_tris @ e_u, sub_tris @ e_v
    u_bnd, v_bnd = bound_tris @ e_u, bound_tris @ e_v

    k_lo = int(math.ceil((float(np.min(v_sub)) + 0.30 * spacing) / spacing))
    k_hi = int(math.floor((float(np.max(v_sub)) - 0.15 * spacing) / spacing))
    if k_hi < k_lo:
        return []

    inset = 0.024 * max(span, 1.0)
    min_len = 0.28 * spacing
    chords: list[tuple[float, float, float, float, float]] = []
    for k_idx in range(k_lo, k_hi + 1):
        row_seed = (
            seed * 1315423911 + (k_idx + 4096) * 2654435761
        ) & 0xFFFFFFFF
        row_rng = np.random.default_rng(row_seed)
        vk = (float(k_idx) + offset_frac) * spacing + float(
            row_rng.uniform(-0.12, 0.12) * spacing
        )
        segs = _row_intervals(u_sub, v_sub, vk)
        bnds = _row_intervals(u_bnd, v_bnd, vk) if segs else []
        if segs and bnds:
            chords.extend(
                _clip_row_chords(
                    segs,
                    bnds,
                    vk=vk,
                    inset=inset,
                    min_len=min_len,
                    spacing=spacing,
                    span=span,
                    rng=row_rng,
                )
            )
    return chords


def _split_chord_intervals(
    ca: float,
    cb: float,
    lims: tuple[float, float, float, float],
    rng: np.random.Generator,
) -> list[tuple[float, float]]:
    """Breaks long halftone chords into staggered natural pen flick spans."""
    max_flick, flick_step, gap, span_ref = lims
    if cb - ca <= max_flick:
        return [(ca, cb)]
    out: list[tuple[float, float]] = []
    pos = ca + float(rng.uniform(0.0, 0.04) * span_ref)
    while pos < cb - 0.05 * span_ref:
        end_p = min(cb, pos + flick_step * float(rng.uniform(0.80, 1.15)))
        if end_p - pos >= 0.045 * span_ref:
            out.append((pos, end_p))
        pos = end_p + gap * float(rng.uniform(0.75, 1.35))
    return out


def _chords_to_comic_flicks(
    chords: Sequence[tuple[float, float, float, float, float]],
    angle_deg: float,
    span: float,
    seed: int,
    curve_ref: tuple[float, float] = (0.0, 0.5),
) -> list[np.ndarray]:
    """Turns shadow chords into 3D form-wrapping curved pen flicks."""
    if not chords:
        return []
    rad = math.radians(angle_deg)
    e_u = np.array([math.cos(rad), math.sin(rad)], dtype=np.float64)
    e_v = np.array([-math.sin(rad), math.cos(rad)], dtype=np.float64)
    rng = np.random.default_rng(seed)
    span_ref = max(span, 1.0)
    lims = (0.34 * span_ref, 0.24 * span_ref, 0.032 * span_ref, span_ref)
    c_v, l_r = curve_ref[0], max(curve_ref[1], 0.15)

    flick_rows: list[tuple[float, float, float, float]] = []
    for ca, cb, vk, _, _ in chords:
        rel_v = float(np.clip((vk - c_v) / l_r, -1.0, 1.0))
        for sa, sb in _split_chord_intervals(ca, cb, lims, rng):
            bow = (sb - sa) * (
                0.14 * rel_v - 0.03 + float(rng.uniform(-0.015, 0.015))
            )
            flick_rows.append((sa, sb, vk, bow))
    if not flick_rows:
        return []

    arr = np.asarray(flick_rows, dtype=np.float64)
    ts = np.linspace(0.0, 1.0, 11, dtype=np.float64)
    sin_t = np.sin(math.pi * ts)
    p0 = arr[:, 0:1] * e_u[None, :] + arr[:, 2:3] * e_v[None, :]
    dp = (arr[:, 1:2] - arr[:, 0:1]) * e_u[None, :]
    pts = (
        p0[:, None, :]
        + ts[None, :, None] * dp[:, None, :]
        + (arr[:, 3:4] * sin_t[None, :])[:, :, None] * e_v[None, None, :]
    )
    return [pts[k] for k in range(len(arr))]


def _slice_single_plane(
    tris_hit: np.ndarray,
    s_verts: np.ndarray,
    xy_v: np.ndarray,
    sk: float,
) -> list[np.ndarray]:
    """Intersects straddling 3D triangles at ``s = sk`` & chains 2D arcs."""
    edge_pts: dict[tuple[int, int], np.ndarray] = {}
    edge_ids: dict[tuple[int, int], int] = {}
    segs: list[tuple[int, int]] = []
    for tri in tris_hit:
        crossed: list[int] = []
        for a, b in (
            (int(tri[0]), int(tri[1])),
            (int(tri[1]), int(tri[2])),
            (int(tri[2]), int(tri[0])),
        ):
            sa = float(s_verts[a])
            sb = float(s_verts[b])
            if (sa <= sk < sb) or (sb <= sk < sa):
                e_key = (a, b) if a < b else (b, a)
                if e_key not in edge_ids:
                    denom = sb - sa
                    frac = (sk - sa) / denom if abs(denom) > 1e-12 else 0.5
                    edge_ids[e_key] = len(edge_pts)
                    edge_pts[e_key] = xy_v[a] + frac * (xy_v[b] - xy_v[a])
                crossed.append(edge_ids[e_key])
        if len(crossed) == 2 and crossed[0] != crossed[1]:
            segs.append((crossed[0], crossed[1]))
    if not edge_pts or not segs:
        return []
    return _chain_edges(segs, np.vstack(list(edge_pts.values())), min_len=2)


def _append_filtered_contour_arcs(
    sub_faces: np.ndarray,
    s_verts: np.ndarray,
    xy_v: np.ndarray,
    station: float,
    min_len: float,
    all_arcs: list[np.ndarray],
) -> None:
    """Slices one plane station and appends trimmed, smoothed contour arcs."""
    for raw_arc in _slice_single_plane(sub_faces, s_verts, xy_v, station):
        trimmed = raw_arc[1:-1] if len(raw_arc) >= 6 else raw_arc
        if len(trimmed) < 2:
            continue
        arc_l = float(np.sum(np.linalg.norm(np.diff(trimmed, axis=0), axis=1)))
        if arc_l >= min_len:
            all_arcs.append(smooth_polyline(trimmed, passes=2))


def _slice_3d_contour_arcs(
    mesh: isosurface.IsosurfaceMesh,
    verts_cam: np.ndarray,
    xy_v: np.ndarray,
    sub_face_indices: np.ndarray,
    plane_normal: np.ndarray,
    *,
    spacing: float,
    span: float,
    offset_frac: float = 0.0,
    seed: int = 11,
) -> list[np.ndarray]:
    """Slices 3D shadow mesh faces at world-anchored plane stations."""
    if len(sub_face_indices) == 0 or spacing <= 1e-9:
        return []
    p_vec = np.asarray(plane_normal, dtype=np.float64)
    p_vec /= max(float(np.linalg.norm(p_vec)), 1e-12)
    s_verts = verts_cam @ p_vec
    sub_faces = mesh.faces[sub_face_indices]
    s_tris = s_verts[sub_faces]
    s_start = float(np.min(s_verts)) + (0.42 + offset_frac) * spacing
    s_stop = float(np.max(s_verts)) - 0.22 * spacing
    if s_stop <= s_start:
        return []

    tri_lo, tri_hi = float(np.min(s_tris)), float(np.max(s_tris))
    s_min_f, s_max_f = np.min(s_tris, axis=1), np.max(s_tris, axis=1)
    all_arcs: list[np.ndarray] = []
    min_len = 0.022 * max(span, 1.0)
    n_steps = int(math.floor((s_stop - s_start) / spacing)) + 1
    for k_idx in range(n_steps):
        k_rng = np.random.default_rng(
            (seed * 2246822519 + k_idx * 3266489917) & 0xFFFFFFFF
        )
        sk = (
            s_start
            + k_idx * spacing
            + float(k_rng.uniform(-0.06, 0.06) * spacing)
        )
        if sk <= tri_lo or sk >= tri_hi:
            continue
        mask = (s_min_f < sk) & (s_max_f >= sk)
        if np.any(mask):
            _append_filtered_contour_arcs(
                sub_faces[mask], s_verts, xy_v, sk, min_len, all_arcs
            )
    return all_arcs


def _make_woven_scumble_skein(
    anchor_2d_z: tuple[np.ndarray, float],
    vn_pt: np.ndarray,
    axes_ctx: tuple[np.ndarray, np.ndarray],
    skein_cfg: tuple[float, float, float, float, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Builds one loose overlapping circulism skein along 3D shadow band."""
    c_xy, c_z = anchor_2d_z
    light_dir, fallback_ax = axes_ctx
    e_a = np.cross(vn_pt, light_dir)
    n1 = float(np.linalg.norm(e_a))
    if n1 < 1e-3:
        e_a = fallback_ax - float(np.dot(fallback_ax, vn_pt)) * vn_pt
        n1 = float(np.linalg.norm(e_a))
    e_a = e_a / max(n1, 1e-12)
    e_b = np.cross(vn_pt, e_a)
    rx, ry, drift_len, n_turns, curl_seed = skein_cfg
    rng = np.random.default_rng(curl_seed)
    phi = float(rng.uniform(0.0, 2.0 * math.pi))
    wob = float(rng.uniform(0.0, 2.0 * math.pi))
    ts = np.linspace(-0.5, 0.5, 44)
    omega = 2.0 * math.pi * n_turns
    ang = omega * (ts + 0.5) + phi
    du = (
        drift_len * ts
        + rx * np.cos(ang)
        + 0.18 * rx * np.sin(2.1 * omega * ts)
    )
    dv = 0.22 * drift_len * np.sin(math.pi * ts + wob) + ry * np.sin(
        ang + 0.35 * np.sin(math.pi * ts)
    )
    pts_xy = (
        c_xy[None, :]
        + du[:, None] * e_a[:2][None, :]
        + dv[:, None] * e_b[:2][None, :]
    )
    pts_z = c_z + du * float(e_a[2]) + dv * float(e_b[2])
    return pts_xy, pts_z


def _split_visible_runs(
    xy_pts: np.ndarray,
    occ_mask: np.ndarray,
    min_len: float,
) -> list[np.ndarray]:
    """Splits a polyline at occluded indices into visible segments."""
    if len(xy_pts) < 2 or np.all(occ_mask):
        return []
    if not np.any(occ_mask):
        return [xy_pts]
    vis = ~occ_mask
    padded = np.concatenate([[False], vis, [False]])
    starts = np.flatnonzero(padded[1:] & ~padded[:-1])
    ends = np.flatnonzero(~padded[1:] & padded[:-1])
    runs: list[np.ndarray] = []
    for st, en in zip(starts, ends, strict=True):
        seg = xy_pts[st:en]
        if len(seg) >= 2:
            arc_len = float(
                np.sum(np.linalg.norm(np.diff(seg, axis=0), axis=1))
            )
            if arc_len >= min_len:
                runs.append(seg)
    return runs


def _clip_strokes_occlusion(
    strokes: Sequence[np.ndarray],
    lobe_idx: int,
    occ_ctx: raster.OcclusionContext,
    min_len: float,
) -> list[np.ndarray]:
    """Clips strokes against foreground lobes and atom sphere borders."""
    if not strokes:
        return []
    lens = [len(p) for p in strokes]
    stacked = np.vstack(strokes)
    occ_all = raster.is_occluded_mask(
        stacked, None, lobe_idx, occ_ctx, cull_over_atoms=True
    )
    clipped: list[np.ndarray] = []
    pos = 0
    for poly, n_pts in zip(strokes, lens, strict=True):
        clipped.extend(
            _split_visible_runs(poly, occ_all[pos : pos + n_pts], min_len)
        )
        pos += n_pts
    return clipped


def _clip_scumble_skeins(
    layer: raster.MeshLayerInput,
    anchor_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    span_ref: float,
    lobe_idx: int,
    occ_ctx: raster.OcclusionContext,
    web_cfg: tuple[int, int, int, float, float, float, float, float, float],
) -> list[np.ndarray]:
    """Builds woven scumble loops and clips occluded segments in batch."""
    tris_chosen, w_chosen, vn_chosen, sel_chosen = anchor_data
    pts_2d = np.sum(layer.xy_v[tris_chosen] * w_chosen[:, :, None], axis=1)
    pts_z = np.sum(layer.depth_v[tris_chosen] * w_chosen, axis=1)
    light_d, _ = raster.light_vectors()
    skein_xy: list[np.ndarray] = []
    skein_z: list[np.ndarray] = []
    for pos, k_idx in enumerate(sel_chosen):
        px_arr, pz_arr = _make_woven_scumble_skein(
            (pts_2d[pos], float(pts_z[pos])),
            vn_chosen[pos],
            (light_d, layer.lobe_axis),
            (
                web_cfg[5] * span_ref,
                web_cfg[6] * span_ref,
                web_cfg[7] * span_ref,
                web_cfg[8],
                int(web_cfg[0]) * 1009 + int(k_idx),
            ),
        )
        skein_xy.append(px_arr)
        skein_z.append(pz_arr)

    all_xy = np.vstack(skein_xy)
    occ_all = raster.is_occluded_mask(
        all_xy,
        np.concatenate(skein_z),
        lobe_idx,
        occ_ctx,
        cull_over_atoms=True,
    ) | raster.outside_own_lobe_mask(all_xy, lobe_idx, occ_ctx)
    curls: list[np.ndarray] = []
    pos = 0
    min_vis = 0.015 * span_ref
    for px_arr in skein_xy:
        n_p = len(px_arr)
        curls.extend(
            _split_visible_runs(px_arr, occ_all[pos : pos + n_p], min_vis)
        )
        pos += n_p
    return curls


def _sample_wandering_scumbles(
    layer: raster.MeshLayerInput,
    sub_mask: np.ndarray,
    span: float,
    lobe_idx: int,
    occ_ctx: raster.OcclusionContext,
    web_cfg: tuple[int, int, int, float, float, float, float, float, float],
) -> list[np.ndarray]:
    """Generates woven surface-anchored circulism loops across shadows."""
    f_idx, bary, attrs = sample_mesh_surface(
        layer.mesh, int(web_cfg[1]), int(web_cfg[0])
    )
    if len(f_idx) == 0:
        return []
    active_f = np.zeros(len(layer.mesh.faces), dtype=bool)
    active_f[np.flatnonzero(layer.front_mask)[sub_mask]] = True
    sel = np.flatnonzero(active_f[f_idx])
    if len(sel) == 0:
        return []

    tris, w = layer.mesh.faces[f_idx[sel]], bary[sel]
    vn = np.sum(layer.vn_rot[tris] * w[:, :, None], axis=1)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
    light_d, half_d = raster.light_vectors()
    sh = np.clip(
        (0.62 - raster.compute_front_tones(vn, light_d, half_d)) / 0.62,
        0.0,
        1.0,
    )
    nz_fac = np.clip(vn[:, 2], 0.04, 1.0) ** 0.30
    chosen = np.flatnonzero(
        (sh > web_cfg[3])
        & (attrs[sel, 0] < nz_fac * (web_cfg[4] + 0.72 * (sh**1.25)))
    )[: int(web_cfg[2])]
    if len(chosen) == 0:
        return []
    return _clip_scumble_skeins(
        layer,
        (tris[chosen], w[chosen], vn[chosen], sel[chosen]),
        max(span, 1.0),
        lobe_idx,
        occ_ctx,
        web_cfg,
    )


def _stipple_rgba_and_sizes(
    attrs_uv: np.ndarray,
    sh_k: np.ndarray,
    ink_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Computes shadow-graded fine-liner micro-dot sizes and crisp RGBA."""
    sizes = 0.36 + 2.35 * (sh_k**1.05) * (0.68 + 0.64 * attrs_uv[:, 0])
    if len(sizes) >= 4:
        std_v = float(np.std(sizes))
        if 1e-6 < std_v < 0.50:
            mean_v = float(np.mean(sizes))
            sizes = np.clip(
                mean_v + (sizes - mean_v) * (0.50 / std_v), 0.35, 4.20
            )
    rgb = np.asarray(
        mcolors.to_rgb(raster.ink_hex(ink_col, charcoal_mix=0.22)),
        dtype=np.float64,
    )
    alphas = np.clip(0.30 + 0.46 * sh_k + 0.10 * attrs_uv[:, 1], 0.28, 0.78)
    rgba = np.empty((len(sh_k), 4), dtype=np.float64)
    rgba[:, :3] = rgb[None, :]
    rgba[:, 3] = alphas
    return np.asarray(sizes, dtype=np.float64), rgba


def _sample_stipple_marks(
    layer: raster.MeshLayerInput,
    sub_mask: np.ndarray,
    cfg_s: tuple[str, int],
    occ_ctx: raster.OcclusionContext,
    seed_lobe: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates blue-noise surface-anchored fine-liner stipple micro-dots."""
    ink_col, lobe_idx = cfg_s
    empty: tuple[np.ndarray, np.ndarray, np.ndarray] = (
        np.zeros((0, 2), dtype=np.float64),
        np.zeros((0,), dtype=np.float64),
        np.zeros((0, 4), dtype=np.float64),
    )
    f_idx, bary, attrs = sample_mesh_surface(layer.mesh, 6800, seed_lobe)
    if len(f_idx) == 0:
        return empty
    active_f = np.zeros(len(layer.mesh.faces), dtype=bool)
    active_f[np.flatnonzero(layer.front_mask)[sub_mask]] = True
    sel = np.flatnonzero(active_f[f_idx])
    if len(sel) == 0:
        return empty

    tris, w = layer.mesh.faces[f_idx[sel]], bary[sel]
    vn = np.sum(layer.vn_rot[tris] * w[:, :, None], axis=1)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
    raw_nl = vn @ raster.light_vectors()[0]
    nz = np.clip(vn[:, 2], 0.0, 1.0)
    sh = np.clip(
        np.clip((0.30 - raw_nl) / 0.96, 0.0, 1.0)
        + 0.26 * np.exp(-(((raw_nl + 0.22) / 0.30) ** 2))
        - 0.20 * ((1.0 - nz) ** 2.4) * np.clip(-raw_nl, 0.0, 1.0),
        0.0,
        1.0,
    )
    nz_fac = np.clip(nz, 0.04, 1.0) ** 0.28
    keep = (sh > 0.05) & (attrs[sel, 0] < nz_fac * (0.95 * (sh**1.75)))
    if not np.any(keep):
        return empty

    pts_2d = np.sum(layer.xy_v[tris[keep]] * w[keep, :, None], axis=1)
    pts_z = np.sum(layer.depth_v[tris[keep]] * w[keep], axis=1)
    occ = raster.is_occluded_mask(
        pts_2d, pts_z, lobe_idx, occ_ctx, cull_over_atoms=True
    )
    pts_2d = pts_2d[~occ]
    if len(pts_2d) == 0:
        return empty
    sizes, rgba = _stipple_rgba_and_sizes(
        attrs[sel][keep][~occ, 1:3], sh[keep][~occ], ink_col
    )
    return pts_2d, sizes, rgba


def _append_tapered_strokes(
    arcs: Sequence[np.ndarray],
    ink_col: str,
    lw: float,
    alpha: float,
    z_stroke: float,
    sketch: tuple[float, float, float] | None,
    line_batches: list[LineBatch],
) -> None:
    """Splits halftone stroke tips into lighter, thinner tapered nib ends."""
    bodies: list[np.ndarray] = []
    tips: list[np.ndarray] = []
    for poly in arcs:
        if len(poly) >= 9:
            bodies.append(poly[:-2])
            tips.append(poly[-3:])
        else:
            bodies.append(poly)
    if bodies:
        line_batches.append((bodies, ink_col, lw, alpha, z_stroke, sketch))
    if tips:
        line_batches.append(
            (tips, ink_col, lw * 0.70, alpha * 0.52, z_stroke, sketch)
        )


def _hatch_curve_ref(
    tris_2d: np.ndarray, angle_deg: float
) -> tuple[float, float]:
    """Computes transverse projection center and radius for form hatching."""
    rad = math.radians(angle_deg)
    e_v = np.array([-math.sin(rad), math.cos(rad)], dtype=np.float64)
    v_all = tris_2d.reshape(-1, 2) @ e_v
    return float(np.mean(v_all)), max(0.5 * float(np.ptp(v_all)), 0.15)


def _emit_hatch_strokes(
    layer: raster.MeshLayerInput,
    sub_mask: np.ndarray,
    bound_tris: np.ndarray,
    *,
    shade_key: str,
    span: float,
    ink_col: str,
    s_alpha: float,
    z_stroke: float,
    lobe_idx: int,
    occ_ctx: raster.OcclusionContext,
    line_batches: list[LineBatch],
    seed_base: int,
) -> None:
    """Appends feathered 3D form-wrapping pen flicks for parallel/crosshatch."""
    xkcd_sketch = (0.80, 90.0, 2.0)
    sp = 0.048 * max(span, 1.0)
    min_vis = 0.020 * max(span, 1.0)
    passes: tuple[tuple[float, float, float, float, int, float, float], ...]
    if shade_key == "hatch":
        passes = (
            (38.0, 0.62, 1.18, 0.0, 1, 0.62, min(0.46, 0.72 * s_alpha)),
            (38.0, 0.38, 1.00, 0.5, 3, 0.76, min(0.48, s_alpha)),
        )
    else:
        passes = (
            (38.0, 0.62, 1.18, 0.0, 1, 0.62, min(0.46, 0.72 * s_alpha)),
            (-50.0, 0.45, 1.00, 0.0, 5, 0.68, min(0.46, 0.88 * s_alpha)),
            (-4.0, 0.26, 1.12, 0.0, 7, 0.60, min(0.44, 0.78 * s_alpha)),
        )
    for ang, t_max, sp_m, off_f, sd_off, lw_h, al_h in passes:
        ck = _clip_xkcd_chords(
            layer.tris_2d[sub_mask & (layer.tones < t_max)],
            bound_tris,
            angle_deg=ang,
            spacing=sp * sp_m,
            span=span,
            offset_frac=off_f,
            seed=seed_base + sd_off,
        )
        fk = _clip_strokes_occlusion(
            _chords_to_comic_flicks(
                ck,
                ang,
                span,
                seed=seed_base + sd_off + 1,
                curve_ref=_hatch_curve_ref(layer.tris_2d, ang),
            ),
            lobe_idx,
            occ_ctx,
            min_vis,
        )
        _append_tapered_strokes(
            fk, ink_col, lw_h, al_h, z_stroke, xkcd_sketch, line_batches
        )


def _emit_contour_strokes(
    layer: raster.MeshLayerInput,
    sub_mask: np.ndarray,
    *,
    shade_key: str,
    span: float,
    ink_col: str,
    s_alpha: float,
    z_stroke: float,
    lobe_idx: int,
    occ_ctx: raster.OcclusionContext,
    line_batches: list[LineBatch],
    seed_base: int,
) -> None:
    """Appends shadow-swelling coaxial 3D latitude & meridian contour rings."""
    xkcd_sketch = (0.80, 90.0, 2.0)
    sp = 0.056 * max(span, 1.0)
    min_vis = 0.022 * max(span, 1.0)
    front_ids = np.flatnonzero(layer.front_mask)

    for t_max, sp_m, off_f, lw_t, al_t, sd_off in (
        (0.70, 1.15, 0.0, 0.48, min(0.46, 0.58 * s_alpha), 9),
        (0.48, 1.15, 0.0, 0.68, min(0.48, 0.85 * s_alpha), 9),
        (0.30, 1.00, 0.5, 0.80, min(0.49, 0.98 * s_alpha), 10),
    ):
        arcs = _clip_strokes_occlusion(
            _slice_3d_contour_arcs(
                layer.mesh,
                layer.verts_cam,
                layer.xy_v,
                front_ids[sub_mask & (layer.tones < t_max)],
                layer.lobe_axis,
                spacing=sp * sp_m,
                span=span,
                offset_frac=off_f,
                seed=seed_base + sd_off,
            ),
            lobe_idx,
            occ_ctx,
            min_vis,
        )
        _append_tapered_strokes(
            arcs, ink_col, lw_t, al_t, z_stroke, xkcd_sketch, line_batches
        )

    if shade_key == "cross_contour":
        u_mer2 = np.cross(layer.lobe_axis, layer.meridian_axis)
        mer_n = 0.80 * layer.meridian_axis + 0.60 * u_mer2
        mer_n /= max(float(np.linalg.norm(mer_n)), 1e-12)
        for t_max, lw_t, al_t in (
            (0.56, 0.50, min(0.45, 0.68 * s_alpha)),
            (0.34, 0.72, min(0.48, 0.90 * s_alpha)),
        ):
            arcs2 = _clip_strokes_occlusion(
                _slice_3d_contour_arcs(
                    layer.mesh,
                    layer.verts_cam,
                    layer.xy_v,
                    front_ids[sub_mask & (layer.tones < t_max)],
                    mer_n,
                    spacing=sp * 1.08,
                    span=span,
                    seed=seed_base + 11,
                ),
                lobe_idx,
                occ_ctx,
                min_vis,
            )
            _append_tapered_strokes(
                arcs2, ink_col, lw_t, al_t, z_stroke, xkcd_sketch, line_batches
            )


def _append_lobe_art_strokes(
    layer: raster.MeshLayerInput,
    sub_mask: np.ndarray,
    *,
    shade_key: str,
    span: float,
    ink_col: str,
    alpha: float,
    z_stroke: float,
    lobe_idx: int,
    occ_ctx: raster.OcclusionContext,
    line_batches: list[LineBatch],
    dot_batches: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]],
    seed_base: int,
) -> None:
    """Appends light, occluded shading strokes/stipples for one depth side."""
    if not np.any(sub_mask):
        return
    s_alpha = float(np.clip(alpha + 0.16, 0.26, 0.50))
    if shade_key in ("hatch", "crosshatch"):
        _emit_hatch_strokes(
            layer,
            sub_mask,
            layer.tris_2d[sub_mask],
            shade_key=shade_key,
            span=span,
            ink_col=ink_col,
            s_alpha=s_alpha,
            z_stroke=z_stroke,
            lobe_idx=lobe_idx,
            occ_ctx=occ_ctx,
            line_batches=line_batches,
            seed_base=seed_base,
        )
    elif shade_key in ("contour", "cross_contour"):
        _emit_contour_strokes(
            layer,
            sub_mask,
            shade_key=shade_key,
            span=span,
            ink_col=ink_col,
            s_alpha=s_alpha,
            z_stroke=z_stroke,
            lobe_idx=lobe_idx,
            occ_ctx=occ_ctx,
            line_batches=line_batches,
            seed_base=seed_base,
        )
    elif shade_key == "stipple":
        pts, sz, rgba = _sample_stipple_marks(
            layer,
            sub_mask,
            (ink_col, lobe_idx),
            occ_ctx,
            seed_lobe=seed_base + 500,
        )
        if len(pts) > 0:
            dot_batches.append((pts, sz, rgba, z_stroke))
    elif shade_key == "scumble":
        scum_layers = (
            (
                (
                    seed_base + 700, 520, 52, 0.18, 0.18,
                    0.056, 0.038, 0.19, 2.35,
                ),
                0.54,
                min(0.42, 0.72 * s_alpha),
            ),
            (
                (
                    seed_base + 731, 420, 38, 0.44, 0.00,
                    0.040, 0.028, 0.14, 2.85,
                ),
                0.68,
                min(0.48, 0.94 * s_alpha),
            ),
        )
        for cfg_w, lw_s, al_s in scum_layers:
            curls = _sample_wandering_scumbles(
                layer, sub_mask, span, lobe_idx, occ_ctx, cfg_w
            )
            if curls:
                line_batches.append(
                    (curls, ink_col, lw_s, al_s, z_stroke, None)
                )


def _append_highlight_arc(
    seg: np.ndarray,
    lit_val: np.ndarray,
    c_xy: np.ndarray,
    cfg: tuple[str, str, float, float, int],
    occ_ctx: raster.OcclusionContext,
    line_batches: list[LineBatch],
) -> None:
    """Adds an occluded sunlit specular crescent for outline/solid modes."""
    hex_col, shade_key, span, z_sil, lobe_idx = cfg
    if shade_key not in ("outline", "solid"):
        return
    min_run = 0.08 * max(span, 1.0)
    for arc_seg in _split_visible_runs(seg, ~(lit_val > 0.58), min_run):
        core = arc_seg[1:-1] if len(arc_seg) >= 6 else arc_seg
        inset_arc = c_xy[None, :] + 0.82 * (core - c_xy[None, :])
        occ = raster.is_occluded_mask(
            inset_arc, None, lobe_idx, occ_ctx, cull_over_atoms=True
        )
        if occ_ctx.atom_xy is not None and occ_ctx.atom_radii is not None:
            d2 = (inset_arc[:, 0:1] - occ_ctx.atom_xy[None, :, 0]) ** 2 + (
                inset_arc[:, 1:2] - occ_ctx.atom_xy[None, :, 1]
            ) ** 2
            occ |= np.any(
                d2 < (1.22 * occ_ctx.atom_radii[None, :]) ** 2, axis=1
            )
        for vis_arc in _split_visible_runs(inset_arc, occ, min_run):
            smoothed = smooth_polyline(vis_arc, passes=2)
            if shade_key == "outline":
                entry: LineBatch = (
                    [smoothed],
                    raster.rim_hex(hex_col),
                    1.22,
                    0.34,
                    z_sil,
                    (0.5, 80.0, 2.0),
                )
            else:
                entry = ([smoothed], "#FFFFFF", 1.35, 0.62, z_sil, None)
            line_batches.append(entry)


def _append_single_rim_tiers(
    seg: np.ndarray,
    c_xy: np.ndarray,
    cfg: tuple[str, str, float, float, float, int],
    occ_ctx: raster.OcclusionContext,
    line_batches: list[LineBatch],
) -> None:
    """Splits one silhouette loop into sunlit, mid, and shadow-belly strokes."""
    hex_col, shade_key, span, alpha, z_sil, lobe_idx = cfg
    diff_v = seg - c_xy[None, :]
    norms = np.maximum(np.linalg.norm(diff_v, axis=1, keepdims=True), 1e-12)
    lit_val = (diff_v / norms) @ np.array([-0.668, 0.744], dtype=np.float64)
    rim_col = raster.rim_hex(hex_col)
    sh_col = raster.ink_hex(hex_col, charcoal_mix=0.35)
    min_run = 0.015 * max(span, 1.0)
    tiers = (
        (
            lit_val > 0.22,
            1.22,
            float(np.clip(alpha + 0.10, 0.26, 0.40)),
            rim_col,
        ),
        (
            (lit_val >= -0.22) & (lit_val <= 0.28),
            1.50,
            float(np.clip(alpha + 0.30, 0.42, 0.60)),
            rim_col,
        ),
        (
            lit_val < -0.16,
            1.92 if z_sil > 20.0 else 1.72,
            float(np.clip(alpha + 0.48, 0.58, 0.80)),
            sh_col,
        ),
    )
    added = False
    for mask_t, lw_t, al_t, col_t in tiers:
        runs = _split_visible_runs(seg, ~mask_t, min_run)
        if runs:
            line_batches.append((runs, col_t, lw_t, al_t, z_sil, None))
            added = True
    if not added:
        al_fb = float(np.clip(alpha + 0.34, 0.30, 0.64))
        line_batches.append(([seg], rim_col, 1.40, al_fb, z_sil, None))
    _append_highlight_arc(
        seg,
        lit_val,
        c_xy,
        (hex_col, shade_key, span, z_sil, lobe_idx),
        occ_ctx,
        line_batches,
    )


def _append_modulated_silhouettes(
    ly: raster.MeshLayerInput,
    lobe_idx: int,
    vis_sils: Sequence[np.ndarray],
    cfg: tuple[str, float, float],
    occ_ctx: raster.OcclusionContext,
    line_batches: list[LineBatch],
) -> None:
    """Appends light- and depth-weighted lobe silhouettes and rim highlights."""
    shade_key, span, alpha = cfg
    z_sil = 41.0
    c_xy = np.mean(ly.tris_2d.reshape(-1, 2), axis=0)
    if occ_ctx.atom_xy is not None and occ_ctx.atom_depth is not None:
        near = int(np.argmin(np.sum((occ_ctx.atom_xy - c_xy) ** 2, axis=1)))
        if occ_ctx.mean_z < float(occ_ctx.atom_depth[near]) - 0.03:
            z_sil = 1.5
    rim_cfg = (ly.hex_col, shade_key, span, alpha, z_sil, lobe_idx)
    for seg in vis_sils:
        _append_single_rim_tiers(seg, c_xy, rim_cfg, occ_ctx, line_batches)


def build_lobe_batches(
    ly: raster.MeshLayerInput,
    idx: int,
    occ_ctx: raster.OcclusionContext,
    cfg: tuple[str, float, float],
    line_batches: list[LineBatch],
    dot_batches: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]],
) -> None:
    """Appends shading marks and depth-occluded silhouettes for one lobe."""
    shade_key, span, alpha = cfg
    if shade_key not in ("solid", "outline"):
        ink_col = raster.ink_hex(ly.hex_col)
        for mask, z_st in (
            (ly.behind_mask, 1.46),
            (~ly.behind_mask, 40.4),
        ):
            _append_lobe_art_strokes(
                ly,
                mask,
                shade_key=shade_key,
                span=span,
                ink_col=ink_col,
                alpha=alpha,
                z_stroke=z_st,
                lobe_idx=idx,
                occ_ctx=occ_ctx,
                line_batches=line_batches,
                dot_batches=dot_batches,
                seed_base=100 + 40 * idx,
            )
    vis_sils: list[np.ndarray] = []
    for s_xy, s_z in zip(ly.silhouettes, ly.sil_z, strict=True):
        occ = raster.is_occluded_mask(
            s_xy, s_z, idx, occ_ctx, cull_over_atoms=False
        )
        vis_sils.extend(
            _split_visible_runs(s_xy, occ, 0.025 * max(span, 1.0))
        )
    if vis_sils:
        _append_modulated_silhouettes(
            ly, idx, vis_sils, cfg, occ_ctx, line_batches
        )
