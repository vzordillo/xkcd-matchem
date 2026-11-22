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
"""Pure-NumPy pen-and-ink comic/xkcd 3D surface shading helpers."""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import math

from matplotlib import colors as mcolors
import numpy as np

from xkcd_matchem.core import electronic
from xkcd_matchem.core import exceptions
from xkcd_matchem.core import geometry
from xkcd_matchem.core import isosurface


_SHADE_ALIASES: dict[str, str] = {
    "solid": "solid",
    "smooth": "solid",
    "blending": "solid",
    "hatch": "hatch",
    "hatching": "hatch",
    "diagonal": "hatch",
    "crosshatch": "crosshatch",
    "cross_hatching": "crosshatch",
    "contour": "contour",
    "contour_hatch": "contour",
    "cross_contour": "cross_contour",
    "stipple": "stipple",
    "stippling": "stipple",
    "dots": "stipple",
    "scumble": "scumble",
    "scribble": "scumble",
    "circulism": "scumble",
    "outline": "outline",
    "none": "outline",
}

VALID_ISO_SHADES: tuple[str, ...] = (
    "solid",
    "hatch",
    "crosshatch",
    "contour",
    "cross_contour",
    "stipple",
    "scumble",
    "outline",
)


@dataclasses.dataclass(frozen=True)
class MeshLayerInput:
    """Projected front-facing triangles and smooth normals for one lobe."""

    mesh: isosurface.IsosurfaceMesh
    xy_v: np.ndarray
    depth_v: np.ndarray
    vn_rot: np.ndarray
    verts_cam: np.ndarray
    front_mask: np.ndarray
    tris_2d: np.ndarray
    depths: np.ndarray
    tones: np.ndarray
    behind_mask: np.ndarray
    silhouettes: list[np.ndarray]
    hex_col: str


@dataclasses.dataclass(frozen=True)
class ShadeArtData:
    """Assembled seam-free RGBA washes, pen strokes, stipples, and bounds."""

    back_rgba: np.ndarray | None
    front_rgba: np.ndarray | None
    wash_extent: tuple[float, float, float, float] | None
    line_batches: list[
        tuple[
            list[np.ndarray],
            str,
            float,
            float,
            float,
            tuple[float, float, float] | None,
        ]
    ]
    dot_batches: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]]
    all_xy: np.ndarray | None


def normalize_iso_shade(shade: str) -> str:
    """Validates and normalizes a 3D isosurface shading mode name."""
    key = shade.strip().lower().replace("-", "_")
    if key not in _SHADE_ALIASES:
        valid = ", ".join(VALID_ISO_SHADES)
        raise exceptions.StyleError(
            f"Unknown isosurface shade {shade!r}. Valid options: {valid}."
        )
    return _SHADE_ALIASES[key]


def _ink_hex(base_hex: str, lift: float = 0.04) -> str:
    """Softens a lobe phase color slightly toward paper for clean ink."""
    r, g, b = mcolors.to_rgb(base_hex)
    return mcolors.to_hex(
        (
            float(np.clip(lift + (1.0 - lift) * r, 0.0, 1.0)),
            float(np.clip(lift + (1.0 - lift) * g, 0.0, 1.0)),
            float(np.clip(lift + (1.0 - lift) * b, 0.0, 1.0)),
        )
    )


def _chain_edges(
    edges: Sequence[tuple[int, int]],
    pts: np.ndarray,
    min_len: int = 3,
) -> list[np.ndarray]:
    """Chains unordered vertex-index edges into continuous 2D polylines."""
    if not edges:
        return []
    adj: dict[int, list[int]] = {}
    for u, v in edges:
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


def extract_silhouette_polylines(
    mesh: isosurface.IsosurfaceMesh,
    xy_v: np.ndarray,
    front_mask: np.ndarray,
) -> list[np.ndarray]:
    """Finds horizon silhouette edges separating front and back faces."""
    edge_flags: dict[tuple[int, int], list[bool]] = {}
    for t_idx, tri in enumerate(mesh.faces):
        is_f = bool(front_mask[t_idx])
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        for u, v in ((i0, i1), (i1, i2), (i2, i0)):
            key = (u, v) if u < v else (v, u)
            edge_flags.setdefault(key, []).append(is_f)
    sil_edges = [
        key
        for key, flags in edge_flags.items()
        if (len(flags) == 2 and flags[0] != flags[1])
        or (len(flags) == 1 and flags[0])
    ]
    return [
        _smooth_polyline(p, passes=3)
        for p in _chain_edges(sil_edges, xy_v, min_len=3)
    ]


def iso_scene_frame(
    meshes: Sequence[tuple[isosurface.IsosurfaceMesh, str]],
    grid: electronic.VolumetricGrid,
    cam: geometry.Camera,
) -> tuple[np.ndarray, float, np.ndarray | None, np.ndarray | None]:
    """Resolves common pivot, projection span, and projected atom depths."""
    if grid.atoms is not None and len(grid.atoms) > 0:
        pivot = (
            0.5 * (grid.atoms.cell[0] + grid.atoms.cell[1] + grid.atoms.cell[2])
            if grid.atoms.cell is not None
            else np.mean(grid.atoms.positions, axis=0)
        )
        span = geometry.scene_projection_span(
            grid.atoms.positions, cam, center=pivot, cell=grid.atoms.cell
        )
        atom_xy, atom_depth = geometry.project_points(
            grid.atoms.positions, cam, center=pivot, span=span
        )
        return pivot, span, atom_xy, atom_depth

    ext = np.asarray(grid.data.shape, dtype=np.float64)[:, None]
    pivot = grid.origin + 0.5 * np.sum(ext * grid.voxel_vectors, axis=0)
    pts = [mesh.vertices for mesh, _ in meshes if len(mesh.vertices)]
    box_pts = np.vstack(pts) if pts else np.zeros((0, 3), dtype=np.float64)
    span = (
        geometry.scene_projection_span(box_pts, cam, center=pivot)
        if len(box_pts)
        else 5.0
    )
    return pivot, span, None, None


def _row_intervals(
    u_tri: np.ndarray,
    v_tri: np.ndarray,
    v_val: float,
) -> list[tuple[float, float]]:
    """Computes merged 1D intervals intersected by ``v = v_val``."""
    raw: list[tuple[float, float]] = []
    min_v = np.min(v_tri, axis=1)
    max_v = np.max(v_tri, axis=1)
    hits = np.flatnonzero((min_v < v_val) & (max_v > v_val))
    for t_idx in hits:
        us: list[float] = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            va = float(v_tri[t_idx, a])
            vb = float(v_tri[t_idx, b])
            if (va <= v_val < vb) or (vb <= v_val < va):
                denom = vb - va
                if abs(denom) > 1e-12:
                    frac = (v_val - va) / denom
                    ua = float(u_tri[t_idx, a])
                    ub = float(u_tri[t_idx, b])
                    us.append(ua + frac * (ub - ua))
        if len(us) >= 2:
            raw.append((min(us), max(us)))
    if not raw:
        return []
    raw.sort(key=lambda seg: seg[0])
    merged: list[tuple[float, float]] = [raw[0]]
    for lo, hi in raw[1:]:
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
    """Feathers interior shadow chord ends and clamps inside lobe span."""
    out: list[tuple[float, float, float, float, float]] = []
    for sa, sb in segs:
        clamped = _clamp_to_bounds(sa, sb, bnds, inset)
        if clamped is None:
            continue
        ca, cb = clamped
        for ba, bb in bnds:
            if ca >= ba - 1e-4 and cb <= bb + 1e-4:
                if ca > ba + inset + 0.015 * span:
                    ca += float(rng.uniform(-0.28, 0.45) * spacing)
                if cb < bb - inset - 0.015 * span:
                    cb -= float(rng.uniform(-0.28, 0.45) * spacing)
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

    v_lo = float(np.min(v_sub)) + 0.45 * spacing
    v_hi = float(np.max(v_sub)) - 0.15 * spacing
    if v_hi <= v_lo:
        return []

    rng = np.random.default_rng(seed)
    inset = 0.008 * max(span, 1.0)
    min_len = 0.25 * spacing
    chords: list[tuple[float, float, float, float, float]] = []
    for base_v in np.arange(v_lo, v_hi, spacing):
        vk = float(base_v + rng.uniform(-0.14, 0.14) * spacing)
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
                    rng=rng,
                )
            )
    return chords


def _chords_to_scumbles(
    chords: Sequence[tuple[float, float, float, float, float]],
    angle_deg: float,
    spacing: float,
    seed: int,
) -> list[np.ndarray]:
    """Turns shadow chords into wandering circular xkcd pen scribbles."""
    if not chords:
        return []
    rad = math.radians(angle_deg)
    e_u = np.array([math.cos(rad), math.sin(rad)], dtype=np.float64)
    e_v = np.array([-math.sin(rad), math.cos(rad)], dtype=np.float64)
    rng = np.random.default_rng(seed)
    out: list[np.ndarray] = []
    for ca, cb, vk, ba, bb in chords:
        length = cb - ca
        n_loops = max(1.3, length / max(0.88 * spacing, 1e-4))
        n_pts = min(96, max(14, int(math.ceil(n_loops * 12))))
        ts = np.linspace(0.0, 1.0, n_pts)
        phi = float(rng.uniform(0.0, 2.0 * math.pi))
        psi = float(rng.uniform(0.0, 2.0 * math.pi))
        omega = 2.0 * math.pi * n_loops
        u_pts = ca + ts * length + 0.30 * spacing * np.sin(omega * ts + phi)
        v_pts = vk + 0.36 * spacing * np.cos(1.13 * omega * ts + psi)
        u_pts = np.clip(u_pts, ba, bb)
        xy_pts = u_pts[:, None] * e_u[None, :] + v_pts[:, None] * e_v[None, :]
        out.append(xy_pts)
    return out


def _smooth_polyline(pts: np.ndarray, passes: int = 2) -> np.ndarray:
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
            arr[1:-1] = (
                0.25 * arr[:-2] + 0.50 * arr[1:-1] + 0.25 * arr[2:]
            )
    return np.asarray(arr, dtype=np.float64)


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
    """Slices 3D shadow mesh faces into faired form-hugging arcs."""
    if len(sub_face_indices) == 0 or spacing <= 1e-9:
        return []
    n_vec = np.asarray(plane_normal, dtype=np.float64)
    n_vec /= max(float(np.linalg.norm(n_vec)), 1e-12)
    s_verts = verts_cam @ n_vec
    sub_faces = mesh.faces[sub_face_indices]
    s_tris = s_verts[sub_faces]
    s_lo = float(np.min(s_tris)) + (0.40 + offset_frac) * spacing
    s_hi = float(np.max(s_tris)) - 0.20 * spacing
    if s_hi <= s_lo:
        return []

    rng = np.random.default_rng(seed)
    all_arcs: list[np.ndarray] = []
    min_len = 0.018 * max(span, 1.0)
    for base_s in np.arange(s_lo, s_hi, spacing):
        sk = float(base_s + rng.uniform(-0.10, 0.10) * spacing)
        mask = (np.min(s_tris, axis=1) < sk) & (np.max(s_tris, axis=1) >= sk)
        if not np.any(mask):
            continue
        for raw_arc in _slice_single_plane(sub_faces[mask], s_verts, xy_v, sk):
            if len(raw_arc) >= 2:
                arc_len = float(
                    np.sum(np.linalg.norm(np.diff(raw_arc, axis=0), axis=1))
                )
                if arc_len >= min_len:
                    all_arcs.append(_smooth_polyline(raw_arc, passes=2))
    return all_arcs


def _point_in_tris_tones(
    cands: np.ndarray,
    sub_tris: np.ndarray,
    sub_tones: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluates barycentric hit mask and tonal intensity in chunks."""
    p0 = sub_tris[:, 0, :]
    v0 = sub_tris[:, 2, :] - p0
    v1 = sub_tris[:, 1, :] - p0
    d00 = np.sum(v0 * v0, axis=1)
    d01 = np.sum(v0 * v1, axis=1)
    d11 = np.sum(v1 * v1, axis=1)
    denom = d00 * d11 - d01 * d01
    inv_d = 1.0 / np.where(np.abs(denom) < 1e-18, 1e-18, denom)

    hit_chunks: list[np.ndarray] = []
    tone_chunks: list[np.ndarray] = []
    for start in range(0, len(cands), 512):
        chunk = cands[start : start + 512]
        v2 = chunk[:, None, :] - p0[None, :, :]
        d20 = np.sum(v2 * v0[None, :, :], axis=2)
        d21 = np.sum(v2 * v1[None, :, :], axis=2)
        u = (d11[None, :] * d20 - d01[None, :] * d21) * inv_d[None, :]
        v = (d00[None, :] * d21 - d01[None, :] * d20) * inv_d[None, :]
        inside = (u >= -1e-4) & (v >= -1e-4) & (u + v <= 1.0 + 1e-4)
        has_tri = np.any(inside, axis=1)
        if np.any(has_tri):
            tri_idx = np.argmax(inside[has_tri], axis=1)
            hit_chunks.append(chunk[has_tri])
            tone_chunks.append(sub_tones[tri_idx])
    if not hit_chunks:
        return (
            np.zeros((0, 2), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
        )
    return np.vstack(hit_chunks), np.concatenate(tone_chunks)


def _sample_stipple_marks(
    sub_tris: np.ndarray,
    sub_tones: np.ndarray,
    span: float,
    ink_col: str,
    s_alpha: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates light random non-uniform fine-liner stipple points."""
    empty: tuple[np.ndarray, np.ndarray, np.ndarray] = (
        np.zeros((0, 2), dtype=np.float64),
        np.zeros((0,), dtype=np.float64),
        np.zeros((0, 4), dtype=np.float64),
    )
    if len(sub_tris) == 0:
        return empty

    flat = sub_tris.reshape(-1, 2)
    x0, y0 = np.min(flat, axis=0)
    x1, y1 = np.max(flat, axis=0)
    area = max(float((x1 - x0) * (y1 - y0)), 1e-6)
    cell_area = (0.017 * max(span, 1.0)) ** 2
    n_cands = int(np.clip(area / max(cell_area, 1e-6), 320, 4200))

    rng = np.random.default_rng(seed)
    cands = np.column_stack(
        [rng.uniform(x0, x1, size=n_cands), rng.uniform(y0, y1, size=n_cands)]
    )
    hit_pts, pt_tones = _point_in_tris_tones(cands, sub_tris, sub_tones)
    if len(hit_pts) == 0:
        return empty

    sh = np.clip((0.66 - pt_tones) / 0.66, 0.0, 1.0)
    keep = rng.random(len(sh)) < (0.04 + 0.92 * (sh**1.75))
    kept_pts = hit_pts[keep]
    if len(kept_pts) == 0:
        return empty

    sh_k = sh[keep]
    skew = rng.random(len(kept_pts)) ** 1.55
    sizes = (0.50 + 4.20 * skew) * (0.72 + 0.38 * sh_k)
    rgb = np.asarray(mcolors.to_rgb(ink_col), dtype=np.float64)
    alphas = np.clip(
        s_alpha * rng.uniform(0.72, 1.18, size=len(kept_pts))
        * (0.65 + 0.45 * sh_k),
        0.20,
        0.52,
    )
    rgba = np.empty((len(kept_pts), 4), dtype=np.float64)
    rgba[:, :3] = rgb[None, :]
    rgba[:, 3] = alphas
    return kept_pts, sizes, rgba


def _smooth_vertex_normals(mesh: isosurface.IsosurfaceMesh) -> np.ndarray:
    """Averages outward triangle normals onto vertices for Phong shading."""
    v_norms = np.zeros_like(mesh.vertices, dtype=np.float64)
    if len(mesh.faces) == 0:
        return v_norms
    for col in range(3):
        np.add.at(v_norms, mesh.faces[:, col], mesh.normals)
    lens = np.linalg.norm(v_norms, axis=1, keepdims=True)
    return np.asarray(v_norms / np.maximum(lens, 1e-12), dtype=np.float64)


def _compute_front_tones(
    n_front: np.ndarray,
    light_dir: np.ndarray,
    half_dir: np.ndarray,
) -> np.ndarray:
    """Computes smooth 3D form tones with highlight and reflected rim."""
    ndotl = np.clip(n_front @ light_dir, 0.0, 1.0)
    spec = np.clip(n_front @ half_dir, 0.0, 1.0) ** 12
    nz_f = np.clip(n_front[:, 2], 0.0, 1.0)
    rim_bounce = 0.22 * ((1.0 - nz_f) ** 2.2) * (1.0 - ndotl)
    return np.asarray(
        np.clip(0.10 + 0.74 * ndotl + 0.22 * spec + rim_bounce, 0.0, 1.0),
        dtype=np.float64,
    )


def _project_single_lobe(
    mesh: isosurface.IsosurfaceMesh,
    hex_col: str,
    cam: geometry.Camera,
    frame: tuple[np.ndarray, float, np.ndarray | None, np.ndarray | None],
    basis: np.ndarray,
    light_dir: np.ndarray,
    half_dir: np.ndarray,
) -> MeshLayerInput | None:
    """Projects one lobe, computes 3D tones & smooth normals, splits depths."""
    pivot, span, atom_xy, atom_depth = frame
    if len(mesh.faces) == 0:
        return None
    xy_v, depth_v = geometry.project_points(
        mesh.vertices, cam, center=pivot, span=span
    )
    n_rot = mesh.normals @ basis.T
    front_mask = n_rot[:, 2] >= -0.02
    sils = extract_silhouette_polylines(mesh, xy_v, front_mask)
    if not np.any(front_mask):
        return None

    vis_faces = mesh.faces[front_mask]
    tris_2d = xy_v[vis_faces]
    deps = np.mean(depth_v[vis_faces], axis=1)
    tones = _compute_front_tones(n_rot[front_mask], light_dir, half_dir)

    if atom_xy is not None and atom_depth is not None:
        tri_xy = np.mean(tris_2d, axis=1)
        nearest = np.argmin(
            np.sum((tri_xy[:, None, :] - atom_xy[None, :, :]) ** 2, axis=2),
            axis=1,
        )
        behind_mask = deps < (atom_depth[nearest] - 0.03)
    else:
        behind_mask = np.zeros(len(tris_2d), dtype=bool)
    vn_rot = _smooth_vertex_normals(mesh) @ basis.T
    return MeshLayerInput(
        mesh=mesh,
        xy_v=xy_v,
        depth_v=depth_v,
        vn_rot=vn_rot,
        verts_cam=(mesh.vertices - pivot) @ basis.T,
        front_mask=front_mask,
        tris_2d=tris_2d,
        depths=deps,
        tones=tones,
        behind_mask=behind_mask,
        silhouettes=sils,
        hex_col=hex_col,
    )


def project_lobe_layers(
    meshes: Sequence[tuple[isosurface.IsosurfaceMesh, str]],
    grid: electronic.VolumetricGrid,
    cam: geometry.Camera,
    shade_key: str,
) -> tuple[list[MeshLayerInput], float, np.ndarray | None, np.ndarray | None]:
    """Projects lobes, computes 3D tones, and splits front/back of atoms."""
    del shade_key
    frame = iso_scene_frame(meshes, grid, cam)
    basis = geometry.rotation_matrix(cam.azimuth, cam.elevation, cam.roll)
    light_dir = np.array([-0.52, 0.58, 0.63], dtype=np.float64)
    light_dir /= float(np.linalg.norm(light_dir))
    half_dir = light_dir + np.array([0.0, 0.0, 1.0], dtype=np.float64)
    half_dir /= float(np.linalg.norm(half_dir))
    layers: list[MeshLayerInput] = []
    for mesh, hex_col in meshes:
        layer = _project_single_lobe(
            mesh, hex_col, cam, frame, basis, light_dir, half_dir
        )
        if layer is not None:
            layers.append(layer)
    return layers, frame[1], frame[2], frame[3]


def _tri_pixel_box(
    pts_2d: np.ndarray,
    grid_spec: tuple[np.ndarray, np.ndarray, float, float, float, float, int],
) -> tuple[int, int, int, int, float] | None:
    """Computes clipped pixel bounding box and determinant for a triangle."""
    _, _, x0, y0, dx, dy, res = grid_spec
    det = (pts_2d[1, 1] - pts_2d[2, 1]) * (pts_2d[0, 0] - pts_2d[2, 0]) + (
        pts_2d[2, 0] - pts_2d[1, 0]
    ) * (pts_2d[0, 1] - pts_2d[2, 1])
    if abs(float(det)) < 1e-16:
        return None
    ix0 = max(0, int(math.floor((float(np.min(pts_2d[:, 0])) - x0) / dx)))
    ix1 = min(res - 1, int(math.ceil((float(np.max(pts_2d[:, 0])) - x0) / dx)))
    iy0 = max(0, int(math.floor((float(np.min(pts_2d[:, 1])) - y0) / dy)))
    iy1 = min(res - 1, int(math.ceil((float(np.max(pts_2d[:, 1])) - y0) / dy)))
    if ix1 < ix0 or iy1 < iy0:
        return None
    return ix0, ix1, iy0, iy1, float(det)


def _rasterize_triangle(
    z_buf: np.ndarray,
    n_buf: np.ndarray,
    pts_2d: np.ndarray,
    tri_z: np.ndarray,
    tri_vn: np.ndarray,
    grid_spec: tuple[np.ndarray, np.ndarray, float, float, float, float, int],
) -> None:
    """Rasterizes one front triangle into z-buffer & smooth normal buffer."""
    box = _tri_pixel_box(pts_2d, grid_spec)
    if box is None:
        return
    ix0, ix1, iy0, iy1, det = box
    gx = grid_spec[0][ix0 : ix1 + 1][None, :] - pts_2d[2, 0]
    gy = grid_spec[1][iy0 : iy1 + 1][:, None] - pts_2d[2, 1]
    inv_d = 1.0 / det
    w0 = (
        (pts_2d[1, 1] - pts_2d[2, 1]) * gx + (pts_2d[2, 0] - pts_2d[1, 0]) * gy
    ) * inv_d
    w1 = (
        (pts_2d[2, 1] - pts_2d[0, 1]) * gx + (pts_2d[0, 0] - pts_2d[2, 0]) * gy
    ) * inv_d
    w2 = 1.0 - w0 - w1
    hit = (w0 >= -1e-5) & (w1 >= -1e-5) & (w2 >= -1e-5)
    if not np.any(hit):
        return
    z_px = w0 * tri_z[0] + w1 * tri_z[1] + w2 * tri_z[2]
    sub_z = z_buf[iy0 : iy1 + 1, ix0 : ix1 + 1]
    upd = hit & (z_px > sub_z)
    if np.any(upd):
        sub_z[upd] = z_px[upd]
        n_px = (
            w0[..., None] * tri_vn[0]
            + w1[..., None] * tri_vn[1]
            + w2[..., None] * tri_vn[2]
        )
        n_buf[iy0 : iy1 + 1, ix0 : ix1 + 1][upd] = n_px[upd]


def _shade_lobe_hits(
    n_hit: np.ndarray,
    hex_col: str,
    alpha: float,
    shade_key: str,
    light_dir: np.ndarray,
    half_dir: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluates per-pixel Phong 3D color and volumetric alpha."""
    norms = np.linalg.norm(n_hit, axis=1, keepdims=True)
    n_unit = n_hit / np.maximum(norms, 1e-12)
    ndotl = np.clip(n_unit @ light_dir, 0.0, 1.0)
    spec = np.clip(n_unit @ half_dir, 0.0, 1.0) ** 16
    nz_h = np.clip(n_unit[:, 2], 0.0, 1.0)
    rim = 0.24 * ((1.0 - nz_h) ** 2.2) * (1.0 - ndotl)

    rgb_base = np.asarray(mcolors.to_rgb(hex_col), dtype=np.float64)
    lit = 0.16 + 0.84 * rgb_base[None, :] * (
        (0.48 + 0.52 * ndotl + 0.20 * rim)[:, None]
    )
    px_rgb = np.clip(lit + (0.68 * spec)[:, None] * (1.0 - lit), 0.0, 1.0)

    if shade_key == "solid":
        base_a = alpha
    elif shade_key == "outline":
        base_a = alpha * 0.10
    else:
        base_a = alpha * 0.52
    core_band = (1.0 - ndotl) * nz_h
    spec_boost = (0.14 if shade_key != "outline" else 0.0) * spec
    px_a = np.clip(
        base_a * (0.62 + 0.62 * core_band) + spec_boost,
        0.0,
        min(0.80, alpha + 0.16),
    )
    return px_rgb, px_a


def _composite_pm(
    dst_pm: np.ndarray,
    iy: np.ndarray,
    ix: np.ndarray,
    rgb: np.ndarray,
    px_a: np.ndarray,
) -> None:
    """Alpha-over composites premultiplied pixel hits into ``dst_pm``."""
    if len(iy) == 0:
        return
    src_pm = np.empty((len(iy), 4), dtype=np.float64)
    src_pm[:, :3] = rgb * px_a[:, None]
    src_pm[:, 3] = px_a
    dst_pm[iy, ix] = src_pm + dst_pm[iy, ix] * (1.0 - px_a[:, None])


def _unpremultiply_pm(pm: np.ndarray) -> np.ndarray | None:
    """Converts premultiplied RGBA buffer into straight float RGBA."""
    if float(np.max(pm[..., 3])) <= 1e-6:
        return None
    alpha = np.clip(pm[..., 3], 0.0, 1.0)
    rgb = np.zeros_like(pm[..., :3], dtype=np.float64)
    mask = alpha > 1e-6
    rgb[mask] = np.clip(pm[..., :3][mask] / alpha[mask, None], 0.0, 1.0)
    out = np.empty_like(pm, dtype=np.float64)
    out[..., :3] = rgb
    out[..., 3] = alpha
    return out


def _composite_single_lobe(
    ly: MeshLayerInput,
    back_pm: np.ndarray,
    front_pm: np.ndarray,
    grid_spec: tuple[np.ndarray, np.ndarray, float, float, float, float, int],
    atom_ref: tuple[np.ndarray | None, np.ndarray | None],
    shade_cfg: tuple[float, str, np.ndarray, np.ndarray],
) -> None:
    """Rasterizes one lobe into back-of-atom and front-of-atom PM buffers."""
    res = grid_spec[6]
    z_buf = np.full((res, res), -np.inf, dtype=np.float64)
    n_buf = np.zeros((res, res, 3), dtype=np.float64)
    vis_faces = ly.mesh.faces[ly.front_mask]
    tri_z = ly.depth_v[vis_faces]
    tri_vn = ly.vn_rot[vis_faces]
    for idx in range(len(vis_faces)):
        _rasterize_triangle(
            z_buf, n_buf, ly.tris_2d[idx], tri_z[idx], tri_vn[idx], grid_spec
        )
    iy_h, ix_h = np.nonzero(np.isfinite(z_buf))
    if len(iy_h) == 0:
        return
    alpha, shade_key, light_dir, half_dir = shade_cfg
    px_rgb, px_a = _shade_lobe_hits(
        n_buf[iy_h, ix_h], ly.hex_col, alpha, shade_key, light_dir, half_dir
    )
    atom_xy, atom_depth = atom_ref
    if atom_xy is not None and atom_depth is not None:
        px_xy = np.column_stack([grid_spec[0][ix_h], grid_spec[1][iy_h]])
        near = np.argmin(
            np.sum((px_xy[:, None, :] - atom_xy[None, :, :]) ** 2, axis=2),
            axis=1,
        )
        is_back = z_buf[iy_h, ix_h] < (atom_depth[near] - 0.03)
    else:
        is_back = np.zeros(len(iy_h), dtype=bool)
    _composite_pm(
        back_pm, iy_h[is_back], ix_h[is_back], px_rgb[is_back], px_a[is_back]
    )
    _composite_pm(
        front_pm,
        iy_h[~is_back],
        ix_h[~is_back],
        px_rgb[~is_back],
        px_a[~is_back],
    )


def _rasterize_wash_buffers(
    layers: Sequence[MeshLayerInput],
    all_xy: np.ndarray,
    atom_xy: np.ndarray | None,
    atom_depth: np.ndarray | None,
    span: float,
    alpha: float,
    shade_key: str,
) -> tuple[
    np.ndarray | None,
    np.ndarray | None,
    tuple[float, float, float, float],
]:
    """Rasterizes seam-free depth-split Phong 3D RGBA wash buffers."""
    res = 200
    pad = 0.02 * max(span, 1.0)
    x0 = float(np.min(all_xy[:, 0])) - pad
    x1 = float(np.max(all_xy[:, 0])) + pad
    y0 = float(np.min(all_xy[:, 1])) - pad
    y1 = float(np.max(all_xy[:, 1])) + pad
    grid_spec = (
        np.linspace(x0, x1, res),
        np.linspace(y0, y1, res),
        x0,
        y0,
        (x1 - x0) / (res - 1),
        (y1 - y0) / (res - 1),
        res,
    )
    light_dir = np.array([-0.52, 0.58, 0.63], dtype=np.float64)
    light_dir /= float(np.linalg.norm(light_dir))
    half_dir = light_dir + np.array([0.0, 0.0, 1.0], dtype=np.float64)
    half_dir /= float(np.linalg.norm(half_dir))

    back_pm = np.zeros((res, res, 4), dtype=np.float64)
    front_pm = np.zeros((res, res, 4), dtype=np.float64)
    for ly in sorted(layers, key=lambda item: float(np.mean(item.depths))):
        _composite_single_lobe(
            ly,
            back_pm,
            front_pm,
            grid_spec,
            (atom_xy, atom_depth),
            (alpha, shade_key, light_dir, half_dir),
        )

    return (
        _unpremultiply_pm(back_pm),
        _unpremultiply_pm(front_pm),
        (x0, x1, y0, y1),
    )


def _emit_hatch_strokes(
    layer: MeshLayerInput,
    sub_mask: np.ndarray,
    *,
    shade_key: str,
    span: float,
    ink_col: str,
    s_alpha: float,
    z_stroke: float,
    line_batches: list[
        tuple[
            list[np.ndarray],
            str,
            float,
            float,
            float,
            tuple[float, float, float] | None,
        ]
    ],
    seed_base: int,
) -> None:
    """Appends light, curved, native-xkcd 3D parallel or cross-hatch lines."""
    xkcd_sketch = (0.85, 90.0, 2.0)
    sp = 0.056 * max(span, 1.0)
    front_ids = np.flatnonzero(layer.front_mask)

    arcs1 = _slice_3d_contour_arcs(
        layer.mesh,
        layer.verts_cam,
        layer.xy_v,
        front_ids[sub_mask & (layer.tones < 0.68)],
        np.array([-0.55, 0.65, 0.52]),
        spacing=sp * 1.25,
        span=span,
        seed=seed_base + 1,
    )
    if arcs1:
        line_batches.append(
            (arcs1, ink_col, 0.66, 0.72 * s_alpha, z_stroke, xkcd_sketch)
        )
    if shade_key == "hatch":
        arcs_d = _slice_3d_contour_arcs(
            layer.mesh,
            layer.verts_cam,
            layer.xy_v,
            front_ids[sub_mask & (layer.tones < 0.38)],
            np.array([-0.55, 0.65, 0.52]),
            spacing=sp,
            span=span,
            offset_frac=0.5,
            seed=seed_base + 2,
        )
        if arcs_d:
            line_batches.append(
                (arcs_d, ink_col, 0.76, s_alpha, z_stroke, xkcd_sketch)
            )
        return

    arcs2 = _slice_3d_contour_arcs(
        layer.mesh,
        layer.verts_cam,
        layer.xy_v,
        front_ids[sub_mask & (layer.tones < 0.44)],
        np.array([0.65, 0.55, 0.52]),
        spacing=sp,
        span=span,
        seed=seed_base + 3,
    )
    if arcs2:
        line_batches.append(
            (arcs2, ink_col, 0.70, 0.88 * s_alpha, z_stroke, xkcd_sketch)
        )
    arcs3 = _slice_3d_contour_arcs(
        layer.mesh,
        layer.verts_cam,
        layer.xy_v,
        front_ids[sub_mask & (layer.tones < 0.25)],
        np.array([0.08, 0.86, 0.50]),
        spacing=sp * 1.05,
        span=span,
        seed=seed_base + 4,
    )
    if arcs3:
        line_batches.append(
            (arcs3, ink_col, 0.64, 0.82 * s_alpha, z_stroke, xkcd_sketch)
        )


def _emit_contour_strokes(
    layer: MeshLayerInput,
    sub_mask: np.ndarray,
    *,
    shade_key: str,
    span: float,
    ink_col: str,
    s_alpha: float,
    z_stroke: float,
    line_batches: list[
        tuple[
            list[np.ndarray],
            str,
            float,
            float,
            float,
            tuple[float, float, float] | None,
        ]
    ],
    seed_base: int,
) -> None:
    """Appends light, curved native-xkcd 3D latitude & meridian arcs."""
    xkcd_sketch = (0.85, 90.0, 2.0)
    sp = 0.058 * max(span, 1.0)
    front_ids = np.flatnonzero(layer.front_mask)

    arcs1 = _slice_3d_contour_arcs(
        layer.mesh,
        layer.verts_cam,
        layer.xy_v,
        front_ids[sub_mask & (layer.tones < 0.70)],
        np.array([-0.42, 0.50, 0.76]),
        spacing=sp * 1.15,
        span=span,
        seed=seed_base + 5,
    )
    if arcs1:
        line_batches.append(
            (arcs1, ink_col, 0.68, 0.76 * s_alpha, z_stroke, xkcd_sketch)
        )
    arcs_core = _slice_3d_contour_arcs(
        layer.mesh,
        layer.verts_cam,
        layer.xy_v,
        front_ids[sub_mask & (layer.tones < 0.36)],
        np.array([-0.42, 0.50, 0.76]),
        spacing=sp,
        span=span,
        offset_frac=0.5,
        seed=seed_base + 12,
    )
    if arcs_core:
        line_batches.append(
            (arcs_core, ink_col, 0.76, s_alpha, z_stroke, xkcd_sketch)
        )
    if shade_key == "cross_contour":
        arcs2 = _slice_3d_contour_arcs(
            layer.mesh,
            layer.verts_cam,
            layer.xy_v,
            front_ids[sub_mask & (layer.tones < 0.48)],
            np.array([0.62, 0.34, 0.71]),
            spacing=sp * 1.05,
            span=span,
            seed=seed_base + 6,
        )
        if arcs2:
            line_batches.append(
                (arcs2, ink_col, 0.68, 0.86 * s_alpha, z_stroke, xkcd_sketch)
            )


def _emit_scumble_strokes(
    layer: MeshLayerInput,
    sub_mask: np.ndarray,
    bound_tris: np.ndarray,
    *,
    span: float,
    ink_col: str,
    s_alpha: float,
    z_stroke: float,
    line_batches: list[
        tuple[
            list[np.ndarray],
            str,
            float,
            float,
            float,
            tuple[float, float, float] | None,
        ]
    ],
    seed_base: int,
) -> None:
    """Appends light wandering circular xkcd pen-scribble loops."""
    sp_scum = 0.072 * max(span, 1.0)
    scum_sketch = None
    c1 = _clip_xkcd_chords(
        layer.tris_2d[sub_mask & (layer.tones < 0.62)],
        bound_tris,
        angle_deg=32.0,
        spacing=sp_scum,
        span=span,
        seed=seed_base + 8,
    )
    s1 = _chords_to_scumbles(c1, 32.0, sp_scum, seed=seed_base + 9)
    if s1:
        line_batches.append(
            (s1, ink_col, 0.64, 0.72 * s_alpha, z_stroke, scum_sketch)
        )
    c2 = _clip_xkcd_chords(
        layer.tris_2d[sub_mask & (layer.tones < 0.34)],
        bound_tris,
        angle_deg=-52.0,
        spacing=sp_scum * 1.05,
        span=span,
        seed=seed_base + 10,
    )
    s2 = _chords_to_scumbles(c2, -52.0, sp_scum * 1.05, seed=seed_base + 11)
    if s2:
        line_batches.append(
            (s2, ink_col, 0.60, 0.75 * s_alpha, z_stroke, scum_sketch)
        )


def _append_lobe_art_strokes(
    layer: MeshLayerInput,
    sub_mask: np.ndarray,
    *,
    shade_key: str,
    span: float,
    ink_col: str,
    alpha: float,
    z_stroke: float,
    line_batches: list[
        tuple[
            list[np.ndarray],
            str,
            float,
            float,
            float,
            tuple[float, float, float] | None,
        ]
    ],
    dot_batches: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]],
    seed_base: int,
) -> None:
    """Appends light shading strokes/stipples for one depth side."""
    if not np.any(sub_mask):
        return
    bound_tris = layer.tris_2d[sub_mask]
    s_alpha = float(np.clip(alpha + 0.15, 0.24, 0.48))
    if shade_key in ("hatch", "crosshatch"):
        _emit_hatch_strokes(
            layer,
            sub_mask,
            shade_key=shade_key,
            span=span,
            ink_col=ink_col,
            s_alpha=s_alpha,
            z_stroke=z_stroke,
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
            line_batches=line_batches,
            seed_base=seed_base,
        )
    elif shade_key == "stipple":
        shadow = sub_mask & (layer.tones < 0.66)
        pts, sz, rgba = _sample_stipple_marks(
            layer.tris_2d[shadow],
            layer.tones[shadow],
            span,
            ink_col,
            s_alpha,
            seed=seed_base + 7,
        )
        if len(pts) > 0:
            dot_batches.append((pts, sz, rgba, z_stroke))
    elif shade_key == "scumble":
        _emit_scumble_strokes(
            layer,
            sub_mask,
            bound_tris,
            span=span,
            ink_col=ink_col,
            s_alpha=s_alpha,
            z_stroke=z_stroke,
            line_batches=line_batches,
            seed_base=seed_base,
        )


def build_isosurface_art(
    meshes: Sequence[tuple[isosurface.IsosurfaceMesh, str]],
    grid: electronic.VolumetricGrid,
    cam: geometry.Camera,
    alpha: float,
    shade: str = "solid",
) -> ShadeArtData:
    """Builds seam-free 3D RGBA washes, light squiggly strokes, & rims."""
    shade_key = normalize_iso_shade(shade)
    layers, span, atom_xy, atom_depth = project_lobe_layers(
        meshes, grid, cam, shade_key
    )
    if not layers:
        return ShadeArtData(None, None, None, [], [], None)

    all_xy = np.vstack([ly.xy_v for ly in layers])
    back_rgba, front_rgba, wash_extent = _rasterize_wash_buffers(
        layers, all_xy, atom_xy, atom_depth, span, alpha, shade_key
    )

    line_batches: list[
        tuple[
            list[np.ndarray],
            str,
            float,
            float,
            float,
            tuple[float, float, float] | None,
        ]
    ] = []
    dot_batches: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
    sil_lw = 1.40
    sil_alpha = float(np.clip(alpha + 0.32, 0.28, 0.62))
    sil_sketch = None

    for idx, ly in enumerate(layers):
        z_sil = 41.0
        if atom_xy is not None and atom_depth is not None:
            c_xy = np.mean(ly.tris_2d.reshape(-1, 2), axis=0)
            near = int(np.argmin(np.sum((atom_xy - c_xy) ** 2, axis=1)))
            if float(np.mean(ly.depths)) < float(atom_depth[near]) - 0.03:
                z_sil = 1.5
        ink_col = _ink_hex(ly.hex_col)
        if shade_key not in ("solid", "outline"):
            for mask, z_st, off in (
                (ly.behind_mask, 1.46, 100),
                (~ly.behind_mask, 40.4, 200),
            ):
                _append_lobe_art_strokes(
                    ly,
                    mask,
                    shade_key=shade_key,
                    span=span,
                    ink_col=ink_col,
                    alpha=alpha,
                    z_stroke=z_st,
                    line_batches=line_batches,
                    dot_batches=dot_batches,
                    seed_base=off + 30 * idx,
                )
        if ly.silhouettes:
            line_batches.append(
                (
                    ly.silhouettes,
                    ly.hex_col,
                    sil_lw,
                    sil_alpha,
                    z_sil,
                    sil_sketch,
                )
            )

    return ShadeArtData(
        back_rgba, front_rgba, wash_extent, line_batches, dot_batches, all_xy
    )
