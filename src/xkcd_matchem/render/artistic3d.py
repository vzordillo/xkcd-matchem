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
from xkcd_matchem.data import elements


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

LineBatch = tuple[
    list[np.ndarray],
    str,
    float,
    float,
    float,
    tuple[float, float, float] | None,
]


@dataclasses.dataclass(frozen=True)
class MeshLayerInput:
    """Projected front-facing triangles, normals, and axis for one lobe."""

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
    sil_z: list[np.ndarray]
    lobe_axis: np.ndarray
    meridian_axis: np.ndarray
    hex_col: str


@dataclasses.dataclass(frozen=True)
class ShadeArtData:
    """Assembled seam-free RGBA washes, pen strokes, stipples, and bounds."""

    back_rgba: np.ndarray | None
    front_rgba: np.ndarray | None
    wash_extent: tuple[float, float, float, float] | None
    line_batches: list[LineBatch]
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


def _ink_hex(base_hex: str, charcoal_mix: float = 0.18) -> str:
    """Mixes comic charcoal into lobe color for pen-and-ink harmony."""
    rgb = np.asarray(mcolors.to_rgb(base_hex), dtype=np.float64)
    charcoal = np.array([0.13, 0.15, 0.17], dtype=np.float64)
    mixed = (1.0 - charcoal_mix) * rgb + charcoal_mix * charcoal
    clipped = np.clip(mixed, 0.0, 1.0)
    return mcolors.to_hex(
        (float(clipped[0]), float(clipped[1]), float(clipped[2]))
    )


def _rim_hex(base_hex: str) -> str:
    """Produces charcoal-tinted comic ink for outer orbital rims."""
    return _ink_hex(base_hex, charcoal_mix=0.26)


def _light_vectors() -> tuple[np.ndarray, np.ndarray]:
    """Returns normalized key-light and Blinn-Phong half directions."""
    light_dir = np.array([-0.52, 0.58, 0.63], dtype=np.float64)
    light_dir /= float(np.linalg.norm(light_dir))
    half_dir = light_dir + np.array([0.0, 0.0, 1.0], dtype=np.float64)
    half_dir /= float(np.linalg.norm(half_dir))
    return light_dir, half_dir


def _sample_mesh_surface(
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


def auto_orbital_camera(
    meshes: Sequence[tuple[isosurface.IsosurfaceMesh, str]],
    grid: electronic.VolumetricGrid,
) -> geometry.Camera:
    """Orients orbital dumbbells upright with atoms clearly separated."""
    if (
        grid.atoms is None
        or len(grid.atoms) < 2
        or grid.atoms.is_periodic
    ):
        return geometry.Camera()

    pivot = np.mean(grid.atoms.positions, axis=0)
    pts = [m.vertices for m, _ in meshes if len(m.vertices)]
    if pts:
        all_pts = np.vstack(pts)
        first_pts = pts[0]
    else:
        vals = np.abs(grid.data)
        max_abs = max(float(np.max(vals)), 1e-12)
        idx = np.argwhere(vals >= 0.35 * max_abs)
        if len(idx) < 4:
            return geometry.Camera()
        all_pts = grid.origin[None, :] + idx @ grid.voxel_vectors
        pos_mask = grid.data[idx[:, 0], idx[:, 1], idx[:, 2]] >= 0.0
        first_pts = all_pts[pos_mask] if np.any(pos_mask) else all_pts

    _, _, vt_orb = np.linalg.svd(all_pts - pivot, full_matrices=False)
    u_orb = vt_orb[0].copy()
    if float(np.dot(np.mean(first_pts, axis=0) - pivot, u_orb)) < 0.0:
        u_orb = -u_orb

    _, _, vt_mol = np.linalg.svd(
        grid.atoms.positions - pivot, full_matrices=False
    )
    u_mol = vt_mol[0]
    right_0 = u_mol - float(np.dot(u_mol, u_orb)) * u_orb
    if float(np.linalg.norm(right_0)) < 1e-4:
        right_0 = vt_orb[1]
    right_0 /= max(float(np.linalg.norm(right_0)), 1e-12)
    fwd_0 = np.cross(right_0, u_orb)
    fwd_0 /= max(float(np.linalg.norm(fwd_0)), 1e-12)

    fwd = 0.80 * fwd_0 - 0.30 * right_0 - 0.44 * u_orb
    fwd /= max(float(np.linalg.norm(fwd)), 1e-12)
    el = math.degrees(math.asin(float(np.clip(fwd[2], -1.0, 1.0))))
    az = math.degrees(math.atan2(float(fwd[1]), float(fwd[0])))
    b0 = geometry.rotation_matrix(az, el, 0.0)
    roll = math.degrees(
        math.atan2(-float(np.dot(b0[0], u_orb)), float(np.dot(b0[1], u_orb)))
    )
    return geometry.Camera(azimuth=az, elevation=el, roll=roll)


def _atom_visual_radii(grid: electronic.VolumetricGrid) -> np.ndarray | None:
    """Returns 2D visual radii matching ``structure3d.render_structure_3d``."""
    if grid.atoms is None or len(grid.atoms) == 0:
        return None
    radii = [
        max(0.24, min(0.58, 0.18 + 0.27 * elements.get_covalent_radius(sym)))
        * 1.08
        for sym in grid.atoms.symbols
    ]
    return np.asarray(radii, dtype=np.float64)


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


def _chain_edges(
    edges: Sequence[tuple[int, int]],
    pts: np.ndarray,
    min_len: int = 3,
) -> list[np.ndarray]:
    """Chains unordered vertex-index edges into continuous N-D polylines."""
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
    """Finds smoothed 2D horizon silhouette loops separating front and back."""
    xy_paths, _ = _extract_silhouette_3d(
        mesh, xy_v, np.zeros(len(xy_v), dtype=np.float64), front_mask
    )
    return xy_paths


def _extract_silhouette_3d(
    mesh: isosurface.IsosurfaceMesh,
    xy_v: np.ndarray,
    depth_v: np.ndarray,
    front_mask: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Extracts smoothed silhouette ``xy`` loops and matching ``z`` arrays."""
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
    xyz_v = np.column_stack([xy_v, depth_v])
    loops = [
        _smooth_polyline(p, passes=3)
        for p in _chain_edges(sil_edges, xyz_v, min_len=3)
    ]
    return [lp[:, :2] for lp in loops], [lp[:, 2] for lp in loops]


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
    c_v, l_r = curve_ref
    l_r = max(l_r, 0.15)
    ts = np.linspace(0.0, 1.0, 11)
    out: list[np.ndarray] = []

    for ca, cb, vk, _, _ in chords:
        rel_v = float(np.clip((vk - c_v) / l_r, -1.0, 1.0))
        for sa, sb in _split_chord_intervals(ca, cb, lims, rng):
            length = sb - sa
            bow = length * (
                0.14 * rel_v - 0.03 + float(rng.uniform(-0.015, 0.015))
            )
            p0 = sa * e_u + vk * e_v
            p1 = sb * e_u + vk * e_v
            pts = (
                p0[None, :]
                + ts[:, None] * (p1 - p0)[None, :]
                + (bow * np.sin(math.pi * ts))[:, None] * e_v[None, :]
            )
            out.append(pts)
    return out


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

    tri_lo = float(np.min(s_tris))
    tri_hi = float(np.max(s_tris))
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
        mask = (np.min(s_tris, axis=1) < sk) & (np.max(s_tris, axis=1) >= sk)
        if not np.any(mask):
            continue
        for raw_arc in _slice_single_plane(sub_faces[mask], s_verts, xy_v, sk):
            trimmed = raw_arc[1:-1] if len(raw_arc) >= 6 else raw_arc
            if len(trimmed) < 2:
                continue
            if (
                float(np.sum(np.linalg.norm(np.diff(trimmed, axis=0), axis=1)))
                >= min_len
            ):
                all_arcs.append(_smooth_polyline(trimmed, passes=2))
    return all_arcs


def _scumble_light_tangent(
    vn_pt: np.ndarray,
    light_dir: np.ndarray,
    fallback_ax: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Builds 3D surface tangent basis aligned along shadow terminator."""
    e_along = np.cross(vn_pt, light_dir)
    n1 = float(np.linalg.norm(e_along))
    if n1 < 1e-3:
        e_along = fallback_ax - float(np.dot(fallback_ax, vn_pt)) * vn_pt
        n1 = float(np.linalg.norm(e_along))
    e_along = e_along / max(n1, 1e-12)
    e_across = np.cross(vn_pt, e_along)
    return e_along, e_across


def _make_woven_scumble_skein(
    anchor_2d_z: tuple[np.ndarray, float],
    vn_pt: np.ndarray,
    axes_ctx: tuple[np.ndarray, np.ndarray],
    skein_cfg: tuple[float, float, float, float, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Builds one loose overlapping circulism skein along 3D shadow band."""
    c_xy, c_z = anchor_2d_z
    e_a, e_b = _scumble_light_tangent(vn_pt, axes_ctx[0], axes_ctx[1])
    rx, ry, drift_len, n_turns, curl_seed = skein_cfg
    rng = np.random.default_rng(curl_seed)
    phi = float(rng.uniform(0.0, 2.0 * math.pi))
    wob = float(rng.uniform(0.0, 2.0 * math.pi))
    ts = np.linspace(-0.5, 0.5, 44)
    sp_u = drift_len * ts
    sp_v = 0.22 * drift_len * np.sin(math.pi * ts + wob)
    omega = 2.0 * math.pi * n_turns
    ang = omega * (ts + 0.5) + phi
    du = sp_u + rx * np.cos(ang) + 0.18 * rx * np.sin(2.1 * omega * ts)
    dv = sp_v + ry * np.sin(ang + 0.35 * np.sin(math.pi * ts))
    pts_xy = (
        c_xy[None, :]
        + du[:, None] * e_a[:2][None, :]
        + dv[:, None] * e_b[:2][None, :]
    )
    pts_z = c_z + du * float(e_a[2]) + dv * float(e_b[2])
    return pts_xy, pts_z


def _outside_own_lobe_mask(
    xy_pts: np.ndarray,
    lobe_idx: int,
    occ_ctx: tuple[
        list[np.ndarray],
        tuple[np.ndarray, np.ndarray, float, float, float, float, int],
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        float,
    ],
) -> np.ndarray:
    """Returns True for points falling outside the lobe's projected zbuf."""
    z_buf = occ_ctx[0][lobe_idx]
    _, _, x0, y0, dx, dy, res = occ_ctx[1]
    out_box = (
        (xy_pts[:, 0] < x0)
        | (xy_pts[:, 0] > x0 + (res - 1) * dx)
        | (xy_pts[:, 1] < y0)
        | (xy_pts[:, 1] > y0 + (res - 1) * dy)
    )
    ix = np.clip(
        np.round((xy_pts[:, 0] - x0) / dx).astype(np.int64), 0, res - 1
    )
    iy = np.clip(
        np.round((xy_pts[:, 1] - y0) / dy).astype(np.int64), 0, res - 1
    )
    return np.asarray(out_box | (~np.isfinite(z_buf[iy, ix])), dtype=bool)


def _clip_chosen_scumbles(
    layer: MeshLayerInput,
    samples: tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ],
    chosen: np.ndarray,
    cfg: tuple[
        float,
        int,
        tuple[
            list[np.ndarray],
            tuple[np.ndarray, np.ndarray, float, float, float, float, int],
            np.ndarray | None,
            np.ndarray | None,
            np.ndarray | None,
            float,
        ],
        tuple[float, float, float, float],
        int,
    ],
) -> list[np.ndarray]:
    """Builds and clips 3D surface circulism skeins for chosen anchors."""
    tris, w, vn, _, sel = samples
    span, lobe_idx, occ_ctx, skein_frac, seed_lobe = cfg
    light_d = _light_vectors()[0]
    pts_2d = np.sum(layer.xy_v[tris[chosen]] * w[chosen, :, None], axis=1)
    pts_z = np.sum(layer.depth_v[tris[chosen]] * w[chosen], axis=1)
    span_ref = max(span, 1.0)
    min_vis = 0.015 * span_ref
    curls: list[np.ndarray] = []
    for pos, k_idx in enumerate(chosen):
        px_arr, pz_arr = _make_woven_scumble_skein(
            (pts_2d[pos], float(pts_z[pos])),
            vn[k_idx],
            (light_d, layer.lobe_axis),
            (
                skein_frac[0] * span_ref,
                skein_frac[1] * span_ref,
                skein_frac[2] * span_ref,
                skein_frac[3],
                seed_lobe * 1009 + int(sel[k_idx]),
            ),
        )
        occ = _is_occluded_mask(
            px_arr, pz_arr, lobe_idx, occ_ctx, cull_over_atoms=True
        ) | _outside_own_lobe_mask(px_arr, lobe_idx, occ_ctx)
        curls.extend(_split_visible_runs(px_arr, occ, min_vis))
    return curls


def _sample_wandering_scumbles(
    layer: MeshLayerInput,
    sub_mask: np.ndarray,
    span: float,
    lobe_idx: int,
    occ_ctx: tuple[
        list[np.ndarray],
        tuple[np.ndarray, np.ndarray, float, float, float, float, int],
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        float,
    ],
    web_cfg: tuple[
        int, int, int, float, float, float, float, float, float
    ],
) -> list[np.ndarray]:
    """Generates woven surface-anchored circulism loops across shadows."""
    f_idx, bary, attrs = _sample_mesh_surface(
        layer.mesh, int(web_cfg[1]), int(web_cfg[0])
    )
    if len(f_idx) == 0:
        return []
    active_f = np.zeros(len(layer.mesh.faces), dtype=bool)
    active_f[np.flatnonzero(layer.front_mask)[sub_mask]] = True
    sel = np.flatnonzero(active_f[f_idx])
    if len(sel) == 0:
        return []

    tris = layer.mesh.faces[f_idx[sel]]
    w = bary[sel]
    vn = np.sum(layer.vn_rot[tris] * w[:, :, None], axis=1)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
    sh = np.clip(
        (0.62 - _compute_front_tones(vn, *_light_vectors())) / 0.62,
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
    return _clip_chosen_scumbles(
        layer,
        (tris, w, vn, sh, sel),
        chosen,
        (
            span,
            lobe_idx,
            occ_ctx,
            (web_cfg[5], web_cfg[6], web_cfg[7], web_cfg[8]),
            int(web_cfg[0]),
        ),
    )


def _is_occluded_mask(
    xy_pts: np.ndarray,
    z_pts: np.ndarray | None,
    lobe_idx: int,
    occ_ctx: tuple[
        list[np.ndarray],
        tuple[np.ndarray, np.ndarray, float, float, float, float, int],
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        float,
    ],
    *,
    cull_over_atoms: bool,
) -> np.ndarray:
    """Returns True for points occluded by foreground lobes or front atoms."""
    lobe_z_bufs, gspec, atom_xy, atom_depth, atom_radii, mean_z = occ_ctx
    _, _, x0, y0, dx, dy, res = gspec
    ix = np.clip(
        np.round((xy_pts[:, 0] - x0) / dx).astype(np.int64), 0, res - 1
    )
    iy = np.clip(
        np.round((xy_pts[:, 1] - y0) / dy).astype(np.int64), 0, res - 1
    )
    if z_pts is None:
        z_eff = lobe_z_bufs[lobe_idx][iy, ix].copy()
        z_eff = np.where(np.isfinite(z_eff), z_eff, mean_z)
    else:
        z_eff = z_pts

    occ = np.zeros(len(xy_pts), dtype=bool)
    for o_idx, o_buf in enumerate(lobe_z_bufs):
        if o_idx != lobe_idx:
            occ |= o_buf[iy, ix] > (z_eff + 0.04)

    if (
        atom_xy is not None
        and atom_depth is not None
        and atom_radii is not None
    ):
        d2 = (xy_pts[:, 0:1] - atom_xy[None, :, 0]) ** 2 + (
            xy_pts[:, 1:2] - atom_xy[None, :, 1]
        ) ** 2
        r_mul = 1.02 if cull_over_atoms else 1.01
        z_off = 0.58 if cull_over_atoms else 0.45
        in_disk = d2 < (atom_radii[None, :] * r_mul) ** 2
        atom_in_front = z_eff[:, None] < (
            atom_depth[None, :] + z_off * atom_radii[None, :]
        )
        occ |= np.any(in_disk & atom_in_front, axis=1)
    return occ


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
    occ_ctx: tuple[
        list[np.ndarray],
        tuple[np.ndarray, np.ndarray, float, float, float, float, int],
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        float,
    ],
    min_len: float,
) -> list[np.ndarray]:
    """Clips strokes against foreground lobes and atom sphere borders."""
    clipped: list[np.ndarray] = []
    for poly in strokes:
        occ = _is_occluded_mask(
            poly, None, lobe_idx, occ_ctx, cull_over_atoms=True
        )
        clipped.extend(_split_visible_runs(poly, occ, min_len))
    return clipped


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
        mcolors.to_rgb(_ink_hex(ink_col, charcoal_mix=0.22)),
        dtype=np.float64,
    )
    alphas = np.clip(0.30 + 0.46 * sh_k + 0.10 * attrs_uv[:, 1], 0.28, 0.78)
    rgba = np.empty((len(sh_k), 4), dtype=np.float64)
    rgba[:, :3] = rgb[None, :]
    rgba[:, 3] = alphas
    return np.asarray(sizes, dtype=np.float64), rgba


def _stipple_shadow_weights(
    vn: np.ndarray,
    light_dir: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Computes 3-zone shadow intensity and gentle limb foreshortening."""
    raw_nl = vn @ light_dir
    nz = np.clip(vn[:, 2], 0.0, 1.0)
    base_sh = np.clip((0.30 - raw_nl) / 0.96, 0.0, 1.0)
    core_b = 0.26 * np.exp(-(((raw_nl + 0.22) / 0.30) ** 2))
    rim_e = 0.20 * ((1.0 - nz) ** 2.4) * np.clip(-raw_nl, 0.0, 1.0)
    sh = np.clip(base_sh + core_b - rim_e, 0.0, 1.0)
    nz_fac = np.clip(nz, 0.04, 1.0) ** 0.28
    return sh, nz_fac


def _sample_stipple_marks(
    layer: MeshLayerInput,
    sub_mask: np.ndarray,
    cfg_s: tuple[str, int],
    occ_ctx: tuple[
        list[np.ndarray],
        tuple[np.ndarray, np.ndarray, float, float, float, float, int],
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        float,
    ],
    seed_lobe: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates blue-noise surface-anchored fine-liner stipple micro-dots."""
    ink_col, lobe_idx = cfg_s
    empty: tuple[np.ndarray, np.ndarray, np.ndarray] = (
        np.zeros((0, 2), dtype=np.float64),
        np.zeros((0,), dtype=np.float64),
        np.zeros((0, 4), dtype=np.float64),
    )
    f_idx, bary, attrs = _sample_mesh_surface(layer.mesh, 6800, seed_lobe)
    if len(f_idx) == 0:
        return empty
    active_f = np.zeros(len(layer.mesh.faces), dtype=bool)
    active_f[np.flatnonzero(layer.front_mask)[sub_mask]] = True
    sel = np.flatnonzero(active_f[f_idx])
    if len(sel) == 0:
        return empty

    tris = layer.mesh.faces[f_idx[sel]]
    w = bary[sel]
    vn = np.sum(layer.vn_rot[tris] * w[:, :, None], axis=1)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
    sh, nz_fac = _stipple_shadow_weights(vn, _light_vectors()[0])
    keep = (sh > 0.05) & (attrs[sel, 0] < nz_fac * (0.95 * (sh**1.75)))
    if not np.any(keep):
        return empty

    pts_2d = np.sum(layer.xy_v[tris[keep]] * w[keep, :, None], axis=1)
    pts_z = np.sum(layer.depth_v[tris[keep]] * w[keep], axis=1)
    occ = _is_occluded_mask(
        pts_2d, pts_z, lobe_idx, occ_ctx, cull_over_atoms=True
    )
    pts_2d = pts_2d[~occ]
    if len(pts_2d) == 0:
        return empty
    sizes, rgba = _stipple_rgba_and_sizes(
        attrs[sel][keep][~occ, 1:3], sh[keep][~occ], ink_col
    )
    return pts_2d, sizes, rgba


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
    """Computes 3-zone 3D form tones with core shadow & reflected rim."""
    raw_nl = n_front @ light_dir
    ndotl = np.clip((raw_nl + 0.12) / 1.12, 0.0, 1.0)
    spec = np.clip(n_front @ half_dir, 0.0, 1.0) ** 18
    nz_f = np.clip(n_front[:, 2], 0.0, 1.0)
    core_dip = 0.16 * np.exp(-(((raw_nl + 0.22) / 0.32) ** 2)) * (nz_f**0.4)
    rim_bounce = 0.24 * ((1.0 - nz_f) ** 2.1) * np.clip(-raw_nl, 0.0, 1.0)
    return np.asarray(
        np.clip(
            0.08 + 0.78 * ndotl + 0.24 * spec + rim_bounce - core_dip,
            0.0,
            1.0,
        ),
        dtype=np.float64,
    )


def _compute_lobe_axes(
    mesh: isosurface.IsosurfaceMesh,
    grid: electronic.VolumetricGrid,
    pivot: np.ndarray,
    basis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Computes world-locked lobe axis and orthogonal meridian in cam space."""
    lobe_mean_w = np.mean(mesh.vertices, axis=0)
    u_ax_w: np.ndarray | None = None
    if grid.atoms is not None and len(grid.atoms) > 0:
        dists = np.linalg.norm(
            grid.atoms.positions - lobe_mean_w[None, :], axis=1
        )
        vec_w = lobe_mean_w - grid.atoms.positions[int(np.argmin(dists))]
        n_ax = float(np.linalg.norm(vec_w))
        if n_ax > 1e-3:
            u_ax_w = vec_w / n_ax
    if u_ax_w is None:
        _, _, vt = np.linalg.svd(
            mesh.vertices - lobe_mean_w, full_matrices=False
        )
        u_ax_w = vt[0] / max(float(np.linalg.norm(vt[0])), 1e-12)
        if float(np.dot(lobe_mean_w - pivot, u_ax_w)) < 0.0:
            u_ax_w = -u_ax_w

    ref_w = (
        np.linalg.svd(grid.atoms.positions - pivot, full_matrices=False)[2][0]
        if grid.atoms is not None and len(grid.atoms) >= 2
        else np.array([0.0, 0.0, 1.0], dtype=np.float64)
    )
    if abs(float(np.dot(ref_w, u_ax_w))) > 0.92:
        ref_w = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(ref_w, u_ax_w))) > 0.92:
        ref_w = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    mer_w = ref_w - float(np.dot(ref_w, u_ax_w)) * u_ax_w
    u_mer_w = mer_w / max(float(np.linalg.norm(mer_w)), 1e-12)
    return (
        np.asarray(basis @ u_ax_w, dtype=np.float64),
        np.asarray(basis @ u_mer_w, dtype=np.float64),
    )


def _project_single_lobe(
    mesh: isosurface.IsosurfaceMesh,
    hex_col: str,
    grid: electronic.VolumetricGrid,
    cam: geometry.Camera,
    frame: tuple[np.ndarray, float, np.ndarray | None, np.ndarray | None],
    basis: np.ndarray,
    light_pair: tuple[np.ndarray, np.ndarray],
) -> MeshLayerInput | None:
    """Projects one lobe, computes 3D tones & intrinsic axes, splits depth."""
    pivot, span, atom_xy, atom_depth = frame
    if len(mesh.faces) == 0:
        return None
    xy_v, depth_v = geometry.project_points(
        mesh.vertices, cam, center=pivot, span=span
    )
    n_rot = mesh.normals @ basis.T
    front_mask = n_rot[:, 2] >= -0.02
    sils, sil_z = _extract_silhouette_3d(mesh, xy_v, depth_v, front_mask)
    if not np.any(front_mask):
        return None

    vis_faces = mesh.faces[front_mask]
    tris_2d = xy_v[vis_faces]
    deps = np.mean(depth_v[vis_faces], axis=1)
    tones = _compute_front_tones(
        n_rot[front_mask], light_pair[0], light_pair[1]
    )

    if atom_xy is not None and atom_depth is not None:
        tri_xy = np.mean(tris_2d, axis=1)
        nearest = np.argmin(
            np.sum((tri_xy[:, None, :] - atom_xy[None, :, :]) ** 2, axis=2),
            axis=1,
        )
        behind_mask = deps < (atom_depth[nearest] - 0.03)
    else:
        behind_mask = np.zeros(len(tris_2d), dtype=bool)
    verts_cam = (mesh.vertices - pivot) @ basis.T
    vn_rot = _smooth_vertex_normals(mesh) @ basis.T
    lobe_ax, mer_ax = _compute_lobe_axes(mesh, grid, pivot, basis)
    return MeshLayerInput(
        mesh=mesh,
        xy_v=xy_v,
        depth_v=depth_v,
        vn_rot=vn_rot,
        verts_cam=verts_cam,
        front_mask=front_mask,
        tris_2d=tris_2d,
        depths=deps,
        tones=tones,
        behind_mask=behind_mask,
        silhouettes=sils,
        sil_z=sil_z,
        lobe_axis=lobe_ax,
        meridian_axis=mer_ax,
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
    light_pair = _light_vectors()
    layers: list[MeshLayerInput] = []
    for mesh, hex_col in meshes:
        layer = _project_single_lobe(
            mesh, hex_col, grid, cam, frame, basis, light_pair
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
    """Evaluates 3-zone Blinn-Phong color, core shadow, and specular wash."""
    norms = np.linalg.norm(n_hit, axis=1, keepdims=True)
    n_unit = n_hit / np.maximum(norms, 1e-12)
    raw_nl = n_unit @ light_dir
    ndotl = np.clip((raw_nl + 0.12) / 1.12, 0.0, 1.0)
    nh = np.clip(n_unit @ half_dir, 0.0, 1.0)
    spec = (nh**14) * 0.55 + (nh**52) * 0.75
    nz_h = np.clip(n_unit[:, 2], 0.0, 1.0)
    core_sh = np.exp(-(((raw_nl + 0.22) / 0.34) ** 2)) * (nz_h**0.4)
    rim = 0.26 * ((1.0 - nz_h) ** 2.0) * np.clip(-raw_nl, 0.0, 1.0)

    rgb_base = np.asarray(mcolors.to_rgb(hex_col), dtype=np.float64)
    shade_f = np.clip(
        0.30 + 0.72 * ndotl + 0.22 * rim - 0.16 * core_sh, 0.18, 1.08
    )
    lit = np.clip(
        rgb_base[None, :] * shade_f[:, None] + (0.12 * (ndotl**2))[:, None],
        0.0,
        1.0,
    )
    px_rgb = np.clip(
        lit + np.clip(spec, 0.0, 0.92)[:, None] * (1.0 - lit), 0.0, 1.0
    )

    if shade_key == "solid":
        base_a = alpha * 1.06
    elif shade_key == "outline":
        base_a = alpha * 0.14
    else:
        base_a = alpha * 0.62
    form_a = 0.48 + 0.68 * ((1.0 - ndotl) * (nz_h**0.65)) + 0.24 * core_sh
    spec_a = (0.18 if shade_key != "outline" else 0.08) * (nh**44)
    px_a = np.clip(
        base_a * form_a + spec_a,
        0.0,
        min(0.80, alpha + 0.21),
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
) -> np.ndarray:
    """Rasterizes one lobe into PM buffers and returns its ``(H, W)`` zbuf."""
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
        return z_buf
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
    return z_buf


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
    list[np.ndarray],
    tuple[np.ndarray, np.ndarray, float, float, float, float, int],
]:
    """Rasterizes seam-free depth-split Phong 3D washes and lobe z-buffers."""
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
    light_dir, half_dir = _light_vectors()

    back_pm = np.zeros((res, res, 4), dtype=np.float64)
    front_pm = np.zeros((res, res, 4), dtype=np.float64)
    z_bufs: list[np.ndarray] = [
        np.empty((0, 0), dtype=np.float64) for _ in layers
    ]
    order = sorted(
        range(len(layers)), key=lambda i: float(np.mean(layers[i].depths))
    )
    for idx in order:
        z_bufs[idx] = _composite_single_lobe(
            layers[idx],
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
        z_bufs,
        grid_spec,
    )


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
    layer: MeshLayerInput,
    sub_mask: np.ndarray,
    bound_tris: np.ndarray,
    *,
    shade_key: str,
    span: float,
    ink_col: str,
    s_alpha: float,
    z_stroke: float,
    lobe_idx: int,
    occ_ctx: tuple[
        list[np.ndarray],
        tuple[np.ndarray, np.ndarray, float, float, float, float, int],
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        float,
    ],
    line_batches: list[LineBatch],
    seed_base: int,
) -> None:
    """Appends feathered 3D form-wrapping pen flicks for parallel/crosshatch."""
    xkcd_sketch = (0.80, 90.0, 2.0)
    sp = 0.048 * max(span, 1.0)
    min_vis = 0.020 * max(span, 1.0)

    c1 = _clip_xkcd_chords(
        layer.tris_2d[sub_mask & (layer.tones < 0.62)],
        bound_tris,
        angle_deg=38.0,
        spacing=sp * 1.18,
        span=span,
        seed=seed_base + 1,
    )
    f1 = _clip_strokes_occlusion(
        _chords_to_comic_flicks(
            c1,
            38.0,
            span,
            seed=seed_base + 2,
            curve_ref=_hatch_curve_ref(layer.tris_2d, 38.0),
        ),
        lobe_idx,
        occ_ctx,
        min_vis,
    )
    _append_tapered_strokes(
        f1,
        ink_col,
        0.62,
        min(0.46, 0.72 * s_alpha),
        z_stroke,
        xkcd_sketch,
        line_batches,
    )
    if shade_key == "hatch":
        cd = _clip_xkcd_chords(
            layer.tris_2d[sub_mask & (layer.tones < 0.38)],
            bound_tris,
            angle_deg=38.0,
            spacing=sp,
            span=span,
            offset_frac=0.5,
            seed=seed_base + 3,
        )
        fd = _clip_strokes_occlusion(
            _chords_to_comic_flicks(
                cd,
                38.0,
                span,
                seed=seed_base + 4,
                curve_ref=_hatch_curve_ref(layer.tris_2d, 38.0),
            ),
            lobe_idx,
            occ_ctx,
            min_vis,
        )
        _append_tapered_strokes(
            fd,
            ink_col,
            0.76,
            min(0.48, s_alpha),
            z_stroke,
            xkcd_sketch,
            line_batches,
        )
        return

    for ang, t_max, sp_m, sd_off, lw_h, al_h in (
        (-50.0, 0.45, 1.0, 5, 0.68, min(0.46, 0.88 * s_alpha)),
        (-4.0, 0.26, 1.12, 7, 0.60, min(0.44, 0.78 * s_alpha)),
    ):
        ck = _clip_xkcd_chords(
            layer.tris_2d[sub_mask & (layer.tones < t_max)],
            bound_tris,
            angle_deg=ang,
            spacing=sp * sp_m,
            span=span,
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
    layer: MeshLayerInput,
    sub_mask: np.ndarray,
    *,
    shade_key: str,
    span: float,
    ink_col: str,
    s_alpha: float,
    z_stroke: float,
    lobe_idx: int,
    occ_ctx: tuple[
        list[np.ndarray],
        tuple[np.ndarray, np.ndarray, float, float, float, float, int],
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        float,
    ],
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
    layer: MeshLayerInput,
    sub_mask: np.ndarray,
    *,
    shade_key: str,
    span: float,
    ink_col: str,
    alpha: float,
    z_stroke: float,
    lobe_idx: int,
    occ_ctx: tuple[
        list[np.ndarray],
        tuple[np.ndarray, np.ndarray, float, float, float, float, int],
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        float,
    ],
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
    cfg: tuple[str, str, float, float],
    line_batches: list[LineBatch],
) -> None:
    """Adds a sunlit inset specular crescent for clean outline/solid modes."""
    hex_col, shade_key, span, z_sil = cfg
    if shade_key not in ("outline", "solid"):
        return
    for arc_seg in _split_visible_runs(
        seg, ~(lit_val > 0.58), 0.08 * max(span, 1.0)
    ):
        core = arc_seg[1:-1] if len(arc_seg) >= 6 else arc_seg
        inset_arc = c_xy[None, :] + 0.82 * (core - c_xy[None, :])
        smoothed = _smooth_polyline(inset_arc, passes=2)
        if shade_key == "outline":
            entry: LineBatch = (
                [smoothed], _rim_hex(hex_col), 1.22, 0.34, z_sil,
                (0.5, 80.0, 2.0),
            )
        else:
            entry = ([smoothed], "#FFFFFF", 1.35, 0.62, z_sil, None)
        line_batches.append(entry)


def _append_single_rim_tiers(
    seg: np.ndarray,
    c_xy: np.ndarray,
    cfg: tuple[str, str, float, float, float],
    line_batches: list[LineBatch],
) -> None:
    """Splits one silhouette loop into sunlit, mid, and shadow-belly strokes."""
    hex_col, shade_key, span, alpha, z_sil = cfg
    diff_v = seg - c_xy[None, :]
    norms = np.maximum(np.linalg.norm(diff_v, axis=1, keepdims=True), 1e-12)
    lit_val = (diff_v / norms) @ np.array([-0.668, 0.744], dtype=np.float64)
    rim_col = _rim_hex(hex_col)
    sh_col = _ink_hex(hex_col, charcoal_mix=0.35)
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
        seg, lit_val, c_xy, (hex_col, shade_key, span, z_sil), line_batches
    )


def _append_modulated_silhouettes(
    ly: MeshLayerInput,
    vis_sils: Sequence[np.ndarray],
    cfg: tuple[str, float, float],
    occ_ctx: tuple[
        list[np.ndarray],
        tuple[np.ndarray, np.ndarray, float, float, float, float, int],
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        float,
    ],
    line_batches: list[LineBatch],
) -> None:
    """Appends light- and depth-weighted lobe silhouettes and rim highlights."""
    shade_key, span, alpha = cfg
    z_sil = 41.0
    c_xy = np.mean(ly.tris_2d.reshape(-1, 2), axis=0)
    _, _, atom_xy, atom_depth, _, mean_z = occ_ctx
    if atom_xy is not None and atom_depth is not None:
        near = int(np.argmin(np.sum((atom_xy - c_xy) ** 2, axis=1)))
        if mean_z < float(atom_depth[near]) - 0.03:
            z_sil = 1.5
    rim_cfg = (ly.hex_col, shade_key, span, alpha, z_sil)
    for seg in vis_sils:
        _append_single_rim_tiers(seg, c_xy, rim_cfg, line_batches)


def _build_lobe_batches(
    ly: MeshLayerInput,
    idx: int,
    occ_ctx: tuple[
        list[np.ndarray],
        tuple[np.ndarray, np.ndarray, float, float, float, float, int],
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        float,
    ],
    cfg: tuple[str, float, float],
    line_batches: list[LineBatch],
    dot_batches: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]],
) -> None:
    """Appends shading marks and depth-occluded silhouettes for one lobe."""
    shade_key, span, alpha = cfg
    if shade_key not in ("solid", "outline"):
        ink_col = _ink_hex(ly.hex_col)
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
        occ = _is_occluded_mask(
            s_xy, s_z, idx, occ_ctx, cull_over_atoms=False
        )
        vis_sils.extend(
            _split_visible_runs(s_xy, occ, 0.025 * max(span, 1.0))
        )
    if vis_sils:
        _append_modulated_silhouettes(
            ly, vis_sils, cfg, occ_ctx, line_batches
        )


def build_isosurface_art(
    meshes: Sequence[tuple[isosurface.IsosurfaceMesh, str]],
    grid: electronic.VolumetricGrid,
    cam: geometry.Camera,
    alpha: float,
    shade: str = "solid",
) -> ShadeArtData:
    """Builds occluded 3D RGBA washes, light comic strokes, and silhouettes."""
    shade_key = normalize_iso_shade(shade)
    layers, span, atom_xy, atom_depth = project_lobe_layers(
        meshes, grid, cam, shade_key
    )
    if not layers:
        return ShadeArtData(None, None, None, [], [], None)

    all_xy = np.vstack([ly.xy_v for ly in layers])
    back_rgba, front_rgba, wash_extent, lobe_z_bufs, grid_spec = (
        _rasterize_wash_buffers(
            layers, all_xy, atom_xy, atom_depth, span, alpha, shade_key
        )
    )
    atom_radii = _atom_visual_radii(grid)
    line_batches: list[LineBatch] = []
    dot_batches: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
    for idx, ly in enumerate(layers):
        occ_ctx = (
            lobe_z_bufs,
            grid_spec,
            atom_xy,
            atom_depth,
            atom_radii,
            float(np.mean(ly.depths)),
        )
        _build_lobe_batches(
            ly,
            idx,
            occ_ctx,
            (shade_key, span, alpha),
            line_batches,
            dot_batches,
        )
    return ShadeArtData(
        back_rgba, front_rgba, wash_extent, line_batches, dot_batches, all_xy
    )
