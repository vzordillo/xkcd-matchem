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
"""Vectorized 3D mesh rasterization, lighting, & occlusion for artistic3d."""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses

from matplotlib import colors as mcolors
import numpy as np

from xkcd_matchem.core import electronic
from xkcd_matchem.core import isosurface
from xkcd_matchem.data import elements


@dataclasses.dataclass(frozen=True)
class GridSpec:
    """Uniform 2D screen raster grid parameters for z-buffer washes."""

    gx: np.ndarray
    gy: np.ndarray
    x0: float
    y0: float
    dx: float
    dy: float
    res: int


@dataclasses.dataclass(frozen=True)
class OcclusionContext:
    """Shared per-lobe z-buffers and atomic sphere bounds for occlusion."""

    lobe_z_bufs: list[np.ndarray]
    grid_spec: GridSpec
    atom_xy: np.ndarray | None
    atom_depth: np.ndarray | None
    atom_radii: np.ndarray | None
    mean_z: float


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


def ink_hex(base_hex: str, charcoal_mix: float = 0.18) -> str:
    """Mixes comic charcoal into lobe color for pen-and-ink harmony."""
    rgb = np.asarray(mcolors.to_rgb(base_hex), dtype=np.float64)
    charcoal = np.array([0.13, 0.15, 0.17], dtype=np.float64)
    mixed = (1.0 - charcoal_mix) * rgb + charcoal_mix * charcoal
    clipped = np.clip(mixed, 0.0, 1.0)
    return mcolors.to_hex(
        (float(clipped[0]), float(clipped[1]), float(clipped[2]))
    )


def rim_hex(base_hex: str) -> str:
    """Produces charcoal-tinted comic ink for outer orbital rims."""
    return ink_hex(base_hex, charcoal_mix=0.26)


def light_vectors() -> tuple[np.ndarray, np.ndarray]:
    """Returns normalized key-light and Blinn-Phong half directions."""
    light_dir = np.array([-0.52, 0.58, 0.63], dtype=np.float64)
    light_dir /= float(np.linalg.norm(light_dir))
    half_dir = light_dir + np.array([0.0, 0.0, 1.0], dtype=np.float64)
    half_dir /= float(np.linalg.norm(half_dir))
    return light_dir, half_dir


def atom_visual_radii(grid: electronic.VolumetricGrid) -> np.ndarray | None:
    """Returns 2D visual radii matching ``structure3d.render_structure_3d``."""
    if grid.atoms is None or len(grid.atoms) == 0:
        return None
    radii = [
        max(0.24, min(0.58, 0.18 + 0.27 * elements.get_covalent_radius(sym)))
        * 1.08
        for sym in grid.atoms.symbols
    ]
    return np.asarray(radii, dtype=np.float64)


def smooth_vertex_normals(mesh: isosurface.IsosurfaceMesh) -> np.ndarray:
    """Averages outward triangle normals onto vertices for Phong shading."""
    v_norms = np.zeros_like(mesh.vertices, dtype=np.float64)
    if len(mesh.faces) == 0:
        return v_norms
    for col in range(3):
        np.add.at(v_norms, mesh.faces[:, col], mesh.normals)
    lens = np.linalg.norm(v_norms, axis=1, keepdims=True)
    return np.asarray(v_norms / np.maximum(lens, 1e-12), dtype=np.float64)


def compute_front_tones(
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


def outside_own_lobe_mask(
    xy_pts: np.ndarray,
    lobe_idx: int,
    occ_ctx: OcclusionContext,
) -> np.ndarray:
    """Returns True for points falling outside the lobe's projected zbuf."""
    z_buf = occ_ctx.lobe_z_bufs[lobe_idx]
    gs = occ_ctx.grid_spec
    if z_buf.size == 0 or gs.dx <= 0.0 or gs.dy <= 0.0:
        return np.ones(len(xy_pts), dtype=bool)
    out_box = (
        (xy_pts[:, 0] < gs.x0)
        | (xy_pts[:, 0] > gs.x0 + (gs.res - 1) * gs.dx)
        | (xy_pts[:, 1] < gs.y0)
        | (xy_pts[:, 1] > gs.y0 + (gs.res - 1) * gs.dy)
    )
    ix = np.clip(
        np.round((xy_pts[:, 0] - gs.x0) / gs.dx).astype(np.int64),
        0,
        gs.res - 1,
    )
    iy = np.clip(
        np.round((xy_pts[:, 1] - gs.y0) / gs.dy).astype(np.int64),
        0,
        gs.res - 1,
    )
    return np.asarray(out_box | (~np.isfinite(z_buf[iy, ix])), dtype=bool)


def is_occluded_mask(
    xy_pts: np.ndarray,
    z_pts: np.ndarray | None,
    lobe_idx: int,
    occ_ctx: OcclusionContext,
    *,
    cull_over_atoms: bool,
) -> np.ndarray:
    """Returns True for points occluded by foreground lobes or front atoms."""
    if len(xy_pts) == 0:
        return np.zeros((0,), dtype=bool)
    gs = occ_ctx.grid_spec
    ix = np.clip(
        np.round((xy_pts[:, 0] - gs.x0) / max(gs.dx, 1e-12)).astype(np.int64),
        0,
        gs.res - 1,
    )
    iy = np.clip(
        np.round((xy_pts[:, 1] - gs.y0) / max(gs.dy, 1e-12)).astype(np.int64),
        0,
        gs.res - 1,
    )
    if z_pts is None:
        z_eff = occ_ctx.lobe_z_bufs[lobe_idx][iy, ix].copy()
        z_eff = np.where(np.isfinite(z_eff), z_eff, occ_ctx.mean_z)
    else:
        z_eff = z_pts

    occ = np.zeros(len(xy_pts), dtype=bool)
    for o_idx, o_buf in enumerate(occ_ctx.lobe_z_bufs):
        if o_idx != lobe_idx and o_buf.size > 0:
            occ |= o_buf[iy, ix] > (z_eff + 0.04)

    a_xy = occ_ctx.atom_xy
    a_dep = occ_ctx.atom_depth
    a_rad = occ_ctx.atom_radii
    if a_xy is not None and a_dep is not None and a_rad is not None:
        d2 = (xy_pts[:, 0:1] - a_xy[None, :, 0]) ** 2 + (
            xy_pts[:, 1:2] - a_xy[None, :, 1]
        ) ** 2
        r_mul = 1.02 if cull_over_atoms else 1.01
        z_off = 0.58 if cull_over_atoms else 0.45
        in_disk = d2 < (a_rad[None, :] * r_mul) ** 2
        atom_in_front = z_eff[:, None] < (
            a_dep[None, :] + z_off * a_rad[None, :]
        )
        occ |= np.any(in_disk & atom_in_front, axis=1)
    return occ


def _expand_chunk_barycentrics(
    tris_2d: np.ndarray,
    box_meta: tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ],
    grid_spec: GridSpec,
) -> (
    tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]
    | None
):
    """Expands chunk bounding boxes and returns hit pixels and barycentrics."""
    c_idx, ix0, iy0, w_box, cnt_box, det = box_meta
    n_pix = int(np.sum(cnt_box))
    if n_pix <= 0:
        return None
    tri_rep = np.repeat(c_idx, cnt_box)
    local = np.arange(n_pix, dtype=np.int64) - np.repeat(
        np.cumsum(cnt_box) - cnt_box, cnt_box
    )
    w_rep = np.repeat(w_box, cnt_box)
    px = np.repeat(ix0[c_idx], cnt_box) + (local % w_rep)
    py = np.repeat(iy0[c_idx], cnt_box) + (local // w_rep)

    p0, p1, p2 = tris_2d[tri_rep, 0], tris_2d[tri_rep, 1], tris_2d[tri_rep, 2]
    gx_d = grid_spec.gx[px] - p2[:, 0]
    gy_d = grid_spec.gy[py] - p2[:, 1]
    inv_d = 1.0 / det[tri_rep]
    w0 = ((p1[:, 1] - p2[:, 1]) * gx_d + (p2[:, 0] - p1[:, 0]) * gy_d) * inv_d
    w1 = ((p2[:, 1] - p0[:, 1]) * gx_d + (p0[:, 0] - p2[:, 0]) * gy_d) * inv_d
    w2 = 1.0 - w0 - w1
    h_idx = np.flatnonzero((w0 >= -1e-5) & (w1 >= -1e-5) & (w2 >= -1e-5))
    if h_idx.size == 0:
        return None
    return (
        tri_rep[h_idx],
        px[h_idx],
        py[h_idx],
        w0[h_idx],
        w1[h_idx],
        w2[h_idx],
    )


def _rasterize_chunk(
    z_buf: np.ndarray,
    n_buf: np.ndarray,
    tris_2d: np.ndarray,
    tri_z: np.ndarray,
    tri_vn: np.ndarray,
    box_meta: tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ],
    grid_spec: GridSpec,
) -> None:
    """Vectorized barycentric z-buffer rasterizer for a chunk of triangles."""
    hits = _expand_chunk_barycentrics(tris_2d, box_meta, grid_spec)
    if hits is None:
        return
    tr_h, px_h, py_h, w0_h, w1_h, w2_h = hits
    z_h = (
        w0_h * tri_z[tr_h, 0]
        + w1_h * tri_z[tr_h, 1]
        + w2_h * tri_z[tr_h, 2]
    )
    upd = np.flatnonzero(z_h > z_buf[py_h, px_h])
    if upd.size == 0:
        return

    tr_u = tr_h[upd]
    order = np.lexsort((-tr_u, z_h[upd]))
    py_s, px_s = py_h[upd][order], px_h[upd][order]
    z_buf[py_s, px_s] = z_h[upd][order]
    n_u = (
        w0_h[upd, None] * tri_vn[tr_u, 0]
        + w1_h[upd, None] * tri_vn[tr_u, 1]
        + w2_h[upd, None] * tri_vn[tr_u, 2]
    )
    n_buf[py_s, px_s] = n_u[order]


def _rasterize_lobe_triangles(
    z_buf: np.ndarray,
    n_buf: np.ndarray,
    tris_2d: np.ndarray,
    tri_z: np.ndarray,
    tri_vn: np.ndarray,
    gs: GridSpec,
) -> None:
    """Rasterizes all front triangles into ``z_buf`` and ``n_buf``."""
    if len(tris_2d) == 0 or gs.dx <= 0.0 or gs.dy <= 0.0:
        return
    p0, p1, p2 = tris_2d[:, 0], tris_2d[:, 1], tris_2d[:, 2]
    det = (p1[:, 1] - p2[:, 1]) * (p0[:, 0] - p2[:, 0]) + (
        p2[:, 0] - p1[:, 0]
    ) * (p0[:, 1] - p2[:, 1])
    xs, ys = tris_2d[:, :, 0], tris_2d[:, :, 1]
    ix0 = np.maximum(
        0, np.floor((np.min(xs, axis=1) - gs.x0) / gs.dx).astype(np.int64)
    )
    ix1 = np.minimum(
        gs.res - 1,
        np.ceil((np.max(xs, axis=1) - gs.x0) / gs.dx).astype(np.int64),
    )
    iy0 = np.maximum(
        0, np.floor((np.min(ys, axis=1) - gs.y0) / gs.dy).astype(np.int64)
    )
    iy1 = np.minimum(
        gs.res - 1,
        np.ceil((np.max(ys, axis=1) - gs.y0) / gs.dy).astype(np.int64),
    )
    valid = np.flatnonzero(
        (np.abs(det) >= 1e-16) & (ix1 >= ix0) & (iy1 >= iy0)
    )
    if valid.size == 0:
        return

    w_v = ix1[valid] - ix0[valid] + 1
    cnt_v = w_v * (iy1[valid] - iy0[valid] + 1)
    cum = np.cumsum(cnt_v)
    budget = 160_000
    pos = 0
    while pos < len(valid):
        base = int(cum[pos - 1]) if pos > 0 else 0
        end = int(np.searchsorted(cum[pos:], base + budget, side="right")) + pos
        end = max(end, pos + 1)
        _rasterize_chunk(
            z_buf,
            n_buf,
            tris_2d,
            tri_z,
            tri_vn,
            (valid[pos:end], ix0, iy0, w_v[pos:end], cnt_v[pos:end], det),
            gs,
        )
        pos = end


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
    px_a = np.clip(base_a * form_a + spec_a, 0.0, min(0.80, alpha + 0.21))
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
    gs: GridSpec,
    atom_ref: tuple[np.ndarray | None, np.ndarray | None],
    shade_cfg: tuple[float, str, np.ndarray, np.ndarray],
) -> np.ndarray:
    """Rasterizes one lobe into PM buffers and returns its ``(H, W)`` zbuf."""
    z_buf = np.full((gs.res, gs.res), -np.inf, dtype=np.float64)
    n_buf = np.zeros((gs.res, gs.res, 3), dtype=np.float64)
    vis_faces = ly.mesh.faces[ly.front_mask]
    _rasterize_lobe_triangles(
        z_buf,
        n_buf,
        ly.tris_2d,
        ly.depth_v[vis_faces],
        ly.vn_rot[vis_faces],
        gs,
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
        px_xy = np.column_stack([gs.gx[ix_h], gs.gy[iy_h]])
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


def rasterize_wash_buffers(
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
    GridSpec,
]:
    """Rasterizes seam-free depth-split Phong 3D washes and lobe z-buffers."""
    res = 200
    pad = 0.02 * max(span, 1.0)
    x0 = float(np.min(all_xy[:, 0])) - pad
    x1 = float(np.max(all_xy[:, 0])) + pad
    y0 = float(np.min(all_xy[:, 1])) - pad
    y1 = float(np.max(all_xy[:, 1])) + pad
    grid_spec = GridSpec(
        gx=np.linspace(x0, x1, res),
        gy=np.linspace(y0, y1, res),
        x0=x0,
        y0=y0,
        dx=max((x1 - x0) / (res - 1), 1e-12),
        dy=max((y1 - y0) / (res - 1), 1e-12),
        res=res,
    )
    light_dir, half_dir = light_vectors()
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
