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
from typing import Any
from typing import cast

from matplotlib import collections
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
    """Projected front-facing triangles and depth mask for one orbital lobe."""

    mesh: isosurface.IsosurfaceMesh
    xy_v: np.ndarray
    verts_cam: np.ndarray
    front_mask: np.ndarray
    tris_2d: np.ndarray
    depths: np.ndarray
    tones: np.ndarray
    face_rgbs: np.ndarray
    behind_mask: np.ndarray
    silhouettes: list[np.ndarray]
    hex_col: str


@dataclasses.dataclass(frozen=True)
class ShadeArtData:
    """Assembled fill collections, pen strokes, stipple dots, and bounds."""

    back_coll: collections.PolyCollection | None
    front_coll: collections.PolyCollection | None
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
    dot_batches: list[tuple[np.ndarray, np.ndarray, str, float, float]]
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


def _ink_hex(base_hex: str, factor: float = 0.68) -> str:
    """Darkens a lobe phase color slightly for colored comic pen ink."""
    r, g, b = mcolors.to_rgb(base_hex)
    return mcolors.to_hex(
        (
            float(np.clip(r * factor, 0.0, 1.0)),
            float(np.clip(g * factor, 0.0, 1.0)),
            float(np.clip(b * factor, 0.0, 1.0)),
        )
    )


def _wash_rgb(base_hex: str, lift: float = 0.36) -> np.ndarray:
    """Softens a base RGB triplet toward paper white for sheer washes."""
    rgb = np.asarray(mcolors.to_rgb(base_hex), dtype=np.float64)
    return np.clip(lift + (1.0 - lift) * rgb, 0.0, 1.0)


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
    return _chain_edges(sil_edges, xy_v, min_len=3)


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

    v_lo = float(np.min(v_sub)) + (0.45 + offset_frac) * spacing
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


def _chords_to_strokes(
    chords: Sequence[tuple[float, float, float, float, float]],
    angle_deg: float,
) -> list[np.ndarray]:
    """Converts chord intervals into multi-vertex 2D polylines for xkcd."""
    if not chords:
        return []
    rad = math.radians(angle_deg)
    e_u = np.array([math.cos(rad), math.sin(rad)], dtype=np.float64)
    e_v = np.array([-math.sin(rad), math.cos(rad)], dtype=np.float64)
    t_grid = np.linspace(0.0, 1.0, 7)[:, None]
    out: list[np.ndarray] = []
    for ca, cb, vk, _, _ in chords:
        p0 = ca * e_u + vk * e_v
        p1 = cb * e_u + vk * e_v
        out.append(p0[None, :] + t_grid * (p1 - p0)[None, :])
    return out


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
        n_loops = max(1.3, length / max(0.85 * spacing, 1e-4))
        n_pts = min(96, max(14, int(math.ceil(n_loops * 12))))
        ts = np.linspace(0.0, 1.0, n_pts)
        phi = float(rng.uniform(0.0, 2.0 * math.pi))
        psi = float(rng.uniform(0.0, 2.0 * math.pi))
        omega = 2.0 * math.pi * n_loops
        amp_u = 0.34 * spacing
        amp_v = 0.42 * spacing
        u_pts = ca + ts * length + amp_u * np.sin(omega * ts + phi)
        v_pts = vk + amp_v * np.cos(1.13 * omega * ts + psi)
        u_pts = np.clip(u_pts, ba, bb)
        xy_pts = u_pts[:, None] * e_u[None, :] + v_pts[:, None] * e_v[None, :]
        out.append(xy_pts)
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
    spacing: float,
    seed: int,
) -> list[np.ndarray]:
    """Slices 3D shadow mesh faces with planes to draw form-hugging arcs."""
    if len(sub_face_indices) == 0 or spacing <= 1e-9:
        return []
    n_vec = np.asarray(plane_normal, dtype=np.float64)
    n_vec /= max(float(np.linalg.norm(n_vec)), 1e-12)
    s_verts = verts_cam @ n_vec
    sub_faces = mesh.faces[sub_face_indices]
    s_tris = s_verts[sub_faces]
    s_lo = float(np.min(s_tris)) + 0.45 * spacing
    s_hi = float(np.max(s_tris)) - 0.25 * spacing
    if s_hi <= s_lo:
        return []

    rng = np.random.default_rng(seed)
    all_arcs: list[np.ndarray] = []
    for base_s in np.arange(s_lo, s_hi, spacing):
        sk = float(base_s + rng.uniform(-0.10, 0.10) * spacing)
        mask = (np.min(s_tris, axis=1) < sk) & (np.max(s_tris, axis=1) >= sk)
        if np.any(mask):
            all_arcs.extend(
                _slice_single_plane(sub_faces[mask], s_verts, xy_v, sk)
            )
    return all_arcs


def _point_in_tris_tones(
    cands: np.ndarray,
    sub_tris: np.ndarray,
    sub_tones: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluates barycentric hit mask and tonal intensity at candidate pts."""
    p0 = sub_tris[:, 0, :]
    v0 = sub_tris[:, 2, :] - p0
    v1 = sub_tris[:, 1, :] - p0
    d00 = np.sum(v0 * v0, axis=1)
    d01 = np.sum(v0 * v1, axis=1)
    d11 = np.sum(v1 * v1, axis=1)
    denom = d00 * d11 - d01 * d01
    inv_d = 1.0 / np.where(np.abs(denom) < 1e-18, 1e-18, denom)

    v2 = cands[:, None, :] - p0[None, :, :]
    d20 = np.sum(v2 * v0[None, :, :], axis=2)
    d21 = np.sum(v2 * v1[None, :, :], axis=2)
    u = (d11[None, :] * d20 - d01[None, :] * d21) * inv_d[None, :]
    v = (d00[None, :] * d21 - d01[None, :] * d20) * inv_d[None, :]
    inside = (u >= -1e-4) & (v >= -1e-4) & (u + v <= 1.0 + 1e-4)
    has_tri = np.any(inside, axis=1)
    if not np.any(has_tri):
        return (
            np.zeros((0, 2), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
        )
    tri_idx = np.argmax(inside[has_tri], axis=1)
    return cands[has_tri], sub_tones[tri_idx]


def _build_stipple_flecks(
    kept_pts: np.ndarray,
    span: float,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Builds short jittered marker micro-strokes for xkcd stippling."""
    n_flecks = max(0, len(kept_pts) // 4)
    if n_flecks == 0:
        return []
    ang = rng.uniform(0.0, math.pi, size=n_flecks)
    dxy = 0.0036 * max(span, 1.0) * np.column_stack([np.cos(ang), np.sin(ang)])
    sel = kept_pts[:n_flecks]
    return [
        np.vstack([sel[i] - dxy[i], sel[i] + dxy[i]]) for i in range(n_flecks)
    ]


def _sample_stipple_marks(
    sub_tris: np.ndarray,
    sub_tones: np.ndarray,
    span: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Generates tonal stipple dots and xkcd felt-tip micro-flecks."""
    empty: tuple[np.ndarray, np.ndarray, list[np.ndarray]] = (
        np.zeros((0, 2), dtype=np.float64),
        np.zeros((0,), dtype=np.float64),
        [],
    )
    if len(sub_tris) == 0:
        return empty

    step = 0.028 * max(span, 1.0)
    x0, y0 = np.min(sub_tris.reshape(-1, 2), axis=0)
    x1, y1 = np.max(sub_tris.reshape(-1, 2), axis=0)
    gx = np.arange(x0 + 0.25 * step, x1, step)
    gy = np.arange(y0 + 0.25 * step, y1, step)
    if len(gx) == 0 or len(gy) == 0:
        return empty

    rng = np.random.default_rng(seed)
    xx, yy = np.meshgrid(gx, gy)
    cands = np.column_stack([xx.ravel(), yy.ravel()])
    cands += rng.uniform(-0.38 * step, 0.38 * step, size=cands.shape)
    hit_pts, pt_tones = _point_in_tris_tones(cands, sub_tris, sub_tones)
    if len(hit_pts) == 0:
        return empty

    keep = rng.random(len(pt_tones)) < (
        np.clip((0.58 - pt_tones) / 0.58, 0.04, 1.0) ** 1.35
    )
    kept_pts = hit_pts[keep]
    if len(kept_pts) == 0:
        return empty

    sizes = 2.0 + 4.8 * np.clip(1.0 - pt_tones[keep], 0.0, 1.0)
    return kept_pts, sizes, _build_stipple_flecks(kept_pts, span, rng)


def _project_single_lobe(
    mesh: isosurface.IsosurfaceMesh,
    hex_col: str,
    cam: geometry.Camera,
    frame: tuple[np.ndarray, float, np.ndarray | None, np.ndarray | None],
    basis: np.ndarray,
    light_dir: np.ndarray,
    shade_key: str,
) -> MeshLayerInput | None:
    """Projects one lobe, computes shading tones, and splits atom depths."""
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
    illum = np.clip(n_rot[front_mask] @ light_dir, 0.0, 1.0)
    tones = 0.68 * illum + 0.32 * np.clip(n_rot[front_mask, 2], 0.0, 1.0)
    if shade_key == "solid":
        lambert = 0.68 + 0.32 * illum
        rgb_base = np.asarray(mcolors.to_rgb(hex_col), dtype=np.float64)
        rgbs = np.clip(
            0.26 + 0.74 * rgb_base[None, :] * lambert[:, None], 0.0, 1.0
        )
    else:
        rgbs = np.broadcast_to(
            _wash_rgb(hex_col)[None, :], (len(tris_2d), 3)
        ).copy()
    if atom_xy is not None and atom_depth is not None:
        tri_xy = np.mean(tris_2d, axis=1)
        nearest = np.argmin(
            np.sum((tri_xy[:, None, :] - atom_xy[None, :, :]) ** 2, axis=2),
            axis=1,
        )
        behind_mask = deps < (atom_depth[nearest] - 0.03)
    else:
        behind_mask = np.zeros(len(tris_2d), dtype=bool)
    return MeshLayerInput(
        mesh=mesh,
        xy_v=xy_v,
        verts_cam=(mesh.vertices - pivot) @ basis.T,
        front_mask=front_mask,
        tris_2d=tris_2d,
        depths=deps,
        tones=tones,
        face_rgbs=rgbs,
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
    """Projects lobes, computes comic tones, and splits front/back of atoms."""
    frame = iso_scene_frame(meshes, grid, cam)
    basis = geometry.rotation_matrix(cam.azimuth, cam.elevation, cam.roll)
    light_dir = np.array([-0.45, 0.55, 0.70], dtype=np.float64)
    light_dir /= float(np.linalg.norm(light_dir))
    layers: list[MeshLayerInput] = []
    for mesh, hex_col in meshes:
        layer = _project_single_lobe(
            mesh, hex_col, cam, frame, basis, light_dir, shade_key
        )
        if layer is not None:
            layers.append(layer)
    return layers, frame[1], frame[2], frame[3]


def _make_poly_coll(
    tris: np.ndarray,
    cols: np.ndarray,
    depths: np.ndarray,
    alpha: float,
    zorder: float,
) -> collections.PolyCollection | None:
    """Creates a seam-free un-sketched PolyCollection."""
    if len(tris) == 0 or alpha <= 0.0:
        return None
    order = np.argsort(depths)
    coll = collections.PolyCollection(
        cast(Any, tris[order]),
        facecolors=cols[order],
        edgecolors="none",
        linewidths=0.0,
        antialiaseds=False,
        alpha=alpha,
        zorder=zorder,
    )
    coll.set_sketch_params(0.0, 0.0, 0.0)
    return coll


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
    """Appends xkcd parallel hatch or cross-hatch shadow strokes."""
    xkcd_sketch = (1.05, 52.0, 2.0)
    sp = 0.056 * max(span, 1.0)
    shadow = sub_mask & (layer.tones < 0.58)
    c1 = _clip_xkcd_chords(
        layer.tris_2d[shadow],
        bound_tris,
        angle_deg=40.0,
        spacing=sp,
        span=span,
        seed=seed_base + 1,
    )
    s1 = _chords_to_strokes(c1, 40.0)
    if s1:
        line_batches.append((s1, ink_col, 1.05, s_alpha, z_stroke, xkcd_sketch))
    if shade_key == "hatch":
        cd = _clip_xkcd_chords(
            layer.tris_2d[sub_mask & (layer.tones < 0.34)],
            bound_tris,
            angle_deg=40.0,
            spacing=sp,
            span=span,
            offset_frac=0.5,
            seed=seed_base + 2,
        )
        sd = _chords_to_strokes(cd, 40.0)
        if sd:
            line_batches.append(
                (sd, ink_col, 1.0, s_alpha, z_stroke, xkcd_sketch)
            )
        return

    c2 = _clip_xkcd_chords(
        layer.tris_2d[sub_mask & (layer.tones < 0.38)],
        bound_tris,
        angle_deg=-48.0,
        spacing=sp,
        span=span,
        seed=seed_base + 3,
    )
    s2 = _chords_to_strokes(c2, -48.0)
    if s2:
        line_batches.append((s2, ink_col, 1.05, s_alpha, z_stroke, xkcd_sketch))
    c3 = _clip_xkcd_chords(
        layer.tris_2d[sub_mask & (layer.tones < 0.22)],
        bound_tris,
        angle_deg=-6.0,
        spacing=sp * 1.15,
        span=span,
        seed=seed_base + 4,
    )
    s3 = _chords_to_strokes(c3, -6.0)
    if s3:
        line_batches.append((s3, ink_col, 0.95, s_alpha, z_stroke, xkcd_sketch))


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
    """Appends 3D curvature-following contour and cross-contour arcs."""
    xkcd_sketch = (1.05, 52.0, 2.0)
    front_ids = np.flatnonzero(layer.front_mask)
    arcs1 = _slice_3d_contour_arcs(
        layer.mesh,
        layer.verts_cam,
        layer.xy_v,
        front_ids[sub_mask & (layer.tones < 0.58)],
        np.array([0.42, 0.91, 0.0]),
        spacing=0.058 * max(span, 1.0),
        seed=seed_base + 5,
    )
    if arcs1:
        line_batches.append(
            (arcs1, ink_col, 1.1, s_alpha, z_stroke, xkcd_sketch)
        )
    if shade_key == "cross_contour":
        arcs2 = _slice_3d_contour_arcs(
            layer.mesh,
            layer.verts_cam,
            layer.xy_v,
            front_ids[sub_mask & (layer.tones < 0.38)],
            np.array([0.91, -0.42, 0.15]),
            spacing=0.058 * max(span, 1.0),
            seed=seed_base + 6,
        )
        if arcs2:
            line_batches.append(
                (arcs2, ink_col, 1.05, s_alpha, z_stroke, xkcd_sketch)
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
    """Appends wandering circular xkcd pen-scribble loops in shadows."""
    sp_scum = 0.068 * max(span, 1.0)
    scum_sketch = (0.7, 45.0, 2.0)
    c1 = _clip_xkcd_chords(
        layer.tris_2d[sub_mask & (layer.tones < 0.56)],
        bound_tris,
        angle_deg=32.0,
        spacing=sp_scum,
        span=span,
        seed=seed_base + 8,
    )
    s1 = _chords_to_scumbles(c1, 32.0, sp_scum, seed=seed_base + 9)
    if s1:
        line_batches.append(
            (s1, ink_col, 0.92, s_alpha * 0.92, z_stroke, scum_sketch)
        )
    c2 = _clip_xkcd_chords(
        layer.tris_2d[sub_mask & (layer.tones < 0.32)],
        bound_tris,
        angle_deg=-52.0,
        spacing=sp_scum * 1.05,
        span=span,
        seed=seed_base + 10,
    )
    s2 = _chords_to_scumbles(c2, -52.0, sp_scum * 1.05, seed=seed_base + 11)
    if s2:
        line_batches.append(
            (s2, ink_col, 0.88, s_alpha * 0.88, z_stroke, scum_sketch)
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
    dot_batches: list[tuple[np.ndarray, np.ndarray, str, float, float]],
    seed_base: int,
) -> None:
    """Appends shading strokes/stipples for one depth side of one lobe."""
    if not np.any(sub_mask):
        return
    bound_tris = layer.tris_2d[sub_mask]
    s_alpha = min(1.0, alpha + 0.48)
    if shade_key in ("hatch", "crosshatch"):
        _emit_hatch_strokes(
            layer,
            sub_mask,
            bound_tris,
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
        shadow = sub_mask & (layer.tones < 0.58)
        pts, sz, flecks = _sample_stipple_marks(
            layer.tris_2d[shadow],
            layer.tones[shadow],
            span,
            seed=seed_base + 7,
        )
        if len(pts) > 0:
            dot_batches.append((pts, sz, ink_col, s_alpha, z_stroke))
        if flecks:
            line_batches.append(
                (flecks, ink_col, 1.05, s_alpha, z_stroke, None)
            )
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


def _silhouette_wash_collection(
    layers: Sequence[MeshLayerInput],
    z_list: Sequence[float],
    *,
    want_back: bool,
    alpha: float,
    zorder: float,
) -> collections.PolyCollection | None:
    """Builds seam-free flat lobe washes from closed outer silhouette loops."""
    polys: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    for ly, z_sil in zip(layers, z_list, strict=True):
        if (z_sil < 2.0) != want_back:
            continue
        wash_rgb = _wash_rgb(ly.hex_col)
        closed = [
            p for p in ly.silhouettes
            if len(p) >= 4 and np.allclose(p[0], p[-1], atol=1e-6)
        ]
        if closed:
            for loop in closed:
                polys.append(loop)
                cols.append(wash_rgb)
        else:
            for tri in ly.tris_2d:
                polys.append(tri)
                cols.append(wash_rgb)
    if not polys or alpha <= 0.0:
        return None
    coll = collections.PolyCollection(
        cast(Any, polys),
        facecolors=np.vstack(cols),
        edgecolors="none",
        linewidths=0.0,
        antialiaseds=False,
        alpha=alpha,
        zorder=zorder,
    )
    coll.set_sketch_params(0.0, 0.0, 0.0)
    return coll


def _assemble_wash_collections(
    layers: Sequence[MeshLayerInput],
    z_list: Sequence[float],
    alpha: float,
    shade_key: str,
) -> tuple[
    collections.PolyCollection | None,
    collections.PolyCollection | None,
]:
    """Builds behind-atom and front-of-atom polygon wash collections."""
    if shade_key != "solid":
        wash_alpha = alpha * (0.12 if shade_key == "outline" else 0.34)
        return (
            _silhouette_wash_collection(
                layers,
                z_list,
                want_back=True,
                alpha=wash_alpha * 0.95,
                zorder=1.4,
            ),
            _silhouette_wash_collection(
                layers,
                z_list,
                want_back=False,
                alpha=wash_alpha,
                zorder=40.0,
            ),
        )
    back_coll = _make_poly_coll(
        np.concatenate([ly.tris_2d[ly.behind_mask] for ly in layers], axis=0),
        np.concatenate(
            [ly.face_rgbs[ly.behind_mask] for ly in layers], axis=0
        ),
        np.concatenate([ly.depths[ly.behind_mask] for ly in layers], axis=0),
        alpha * 0.95,
        zorder=1.4,
    )
    front_coll = _make_poly_coll(
        np.concatenate([ly.tris_2d[~ly.behind_mask] for ly in layers], axis=0),
        np.concatenate(
            [ly.face_rgbs[~ly.behind_mask] for ly in layers], axis=0
        ),
        np.concatenate([ly.depths[~ly.behind_mask] for ly in layers], axis=0),
        alpha,
        zorder=40.0,
    )
    return back_coll, front_coll


def build_isosurface_art(
    meshes: Sequence[tuple[isosurface.IsosurfaceMesh, str]],
    grid: electronic.VolumetricGrid,
    cam: geometry.Camera,
    alpha: float,
    shade: str = "solid",
) -> ShadeArtData:
    """Builds depth-split fills, comic pen strokes, and silhouette rims."""
    shade_key = normalize_iso_shade(shade)
    layers, span, atom_xy, atom_depth = project_lobe_layers(
        meshes, grid, cam, shade_key
    )
    if not layers:
        return ShadeArtData(None, None, [], [], None)

    z_list: list[float] = []
    for ly in layers:
        z_sil = 41.0
        if atom_xy is not None and atom_depth is not None:
            c_xy = np.mean(ly.tris_2d.reshape(-1, 2), axis=0)
            near = int(np.argmin(np.sum((atom_xy - c_xy) ** 2, axis=1)))
            if float(np.mean(ly.depths)) < float(atom_depth[near]) - 0.03:
                z_sil = 1.5
        z_list.append(z_sil)

    back_coll, front_coll = _assemble_wash_collections(
        layers, z_list, alpha, shade_key
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
    dot_batches: list[tuple[np.ndarray, np.ndarray, str, float, float]] = []
    sil_lw = 1.6 if shade_key == "solid" else 1.85
    sil_alpha = min(1.0, alpha + (0.44 if shade_key == "solid" else 0.58))

    for idx, (ly, z_sil) in enumerate(zip(layers, z_list, strict=True)):
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
                (ly.silhouettes, ly.hex_col, sil_lw, sil_alpha, z_sil, None)
            )

    all_xy = np.vstack([ly.xy_v for ly in layers])
    return ShadeArtData(
        back_coll, front_coll, line_batches, dot_batches, all_xy
    )
