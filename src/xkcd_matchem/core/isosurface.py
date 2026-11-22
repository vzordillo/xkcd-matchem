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
"""Pure-NumPy Kuhn 6-tetrahedra Marching Tetrahedra isosurface extractor."""

from __future__ import annotations

import dataclasses

import numpy as np
from scipy import interpolate

from xkcd_matchem.core import electronic
from xkcd_matchem.core import exceptions


_CUBE_OFFSETS = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ],
    dtype=np.int64,
)

# Kuhn 6-tetrahedra subdivision along main cube diagonal (0 -> 6).
_KUHN_TETS: tuple[tuple[int, int, int, int], ...] = (
    (0, 5, 1, 6),
    (0, 1, 2, 6),
    (0, 2, 3, 6),
    (0, 3, 7, 6),
    (0, 7, 4, 6),
    (0, 4, 5, 6),
)


@dataclasses.dataclass(frozen=True)
class IsosurfaceMesh:
    """Closed-surface triangle mesh extracted from a 3D volumetric grid.

    Attributes:
      vertices: Unique vertex Cartesian coordinates in Angstrom ``(V, 3)``.
      faces: Zero-based integer triangle index array ``(F, 3)``.
      normals: Outward unit normal vectors per triangle ``(F, 3)``.
      level: Isovalue at which the surface was extracted.
    """

    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray
    level: float


def _empty_mesh(level: float) -> IsosurfaceMesh:
    """Returns an empty ``IsosurfaceMesh`` when no crossings exist."""
    return IsosurfaceMesh(
        vertices=np.zeros((0, 3), dtype=np.float64),
        faces=np.zeros((0, 3), dtype=np.int64),
        normals=np.zeros((0, 3), dtype=np.float64),
        level=float(level),
    )


def _active_tetrahedra(
    data: np.ndarray,
    level: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Enumerates crossing Kuhn tetrahedra grid coords, flat ids, & values."""
    nx, ny, nz = data.shape
    ii, jj, kk = np.meshgrid(
        np.arange(nx - 1, dtype=np.int64),
        np.arange(ny - 1, dtype=np.int64),
        np.arange(nz - 1, dtype=np.int64),
        indexing="ij",
    )
    origins = np.column_stack([ii.ravel(), jj.ravel(), kk.ravel()])
    cube_coords = origins[:, None, :] + _CUBE_OFFSETS[None, :, :]
    cube_vals = data[
        cube_coords[:, :, 0], cube_coords[:, :, 1], cube_coords[:, :, 2]
    ]
    cube_inside = cube_vals >= level
    active_cubes = np.any(cube_inside, axis=1) & (~np.all(cube_inside, axis=1))
    if not np.any(active_cubes):
        empty_i = np.zeros((0, 4), dtype=np.int64)
        return (
            np.zeros((0, 4, 3), dtype=np.int64),
            empty_i,
            np.zeros((0, 4), dtype=np.float64),
        )
    cube_coords = cube_coords[active_cubes]
    cube_vals = cube_vals[active_cubes]
    cube_flat = (
        cube_coords[:, :, 0] * (ny * nz)
        + cube_coords[:, :, 1] * nz
        + cube_coords[:, :, 2]
    )

    tet_idx = np.array(_KUHN_TETS, dtype=np.int64)
    tet_coords = cube_coords[:, tet_idx, :].reshape(-1, 4, 3)
    tet_flat = cube_flat[:, tet_idx].reshape(-1, 4)
    tet_vals = cube_vals[:, tet_idx].reshape(-1, 4)
    tet_in = tet_vals >= level
    keep = np.any(tet_in, axis=1) & (~np.all(tet_in, axis=1))
    return tet_coords[keep], tet_flat[keep], tet_vals[keep]


def _append_single_corner_tris(
    tri_pairs: list[np.ndarray],
    tri_dirs: list[np.ndarray],
    t_coords: np.ndarray,
    t_flat: np.ndarray,
    tet_in: np.ndarray,
    grid_vecs: np.ndarray,
    *,
    select_inside_lone: bool,
    out_sign: float,
) -> None:
    """Collects 1-in/3-out or 3-in/1-out tetrahedra triangles."""
    counts = np.sum(tet_in, axis=1)
    mask = (counts == 1) if select_inside_lone else (counts == 3)
    if not np.any(mask):
        return
    in_flag = tet_in[mask]
    coords_sub = t_coords[mask].astype(np.float64)
    flat_sub = t_flat[mask]
    lone_mask = in_flag if select_inside_lone else (~in_flag)
    lone_idx = np.argmax(lone_mask, axis=1)
    rest_mask = ~lone_mask
    rest_idx = np.nonzero(rest_mask)[1].reshape(-1, 3)

    rows = np.arange(len(flat_sub))
    u = flat_sub[rows, lone_idx][:, None]
    v = flat_sub[rows[:, None], rest_idx]
    pairs = np.stack([np.broadcast_to(u, v.shape), v], axis=-1)
    tri_pairs.append(pairs)

    lone_pt = coords_sub[rows, lone_idx] @ grid_vecs
    rest_pt = np.mean(coords_sub[rows[:, None], rest_idx], axis=1) @ grid_vecs
    vec = (rest_pt - lone_pt) if select_inside_lone else (lone_pt - rest_pt)
    tri_dirs.append(vec * out_sign)


def _append_two_in_tris(
    tri_pairs: list[np.ndarray],
    tri_dirs: list[np.ndarray],
    t_coords: np.ndarray,
    t_flat: np.ndarray,
    tet_in: np.ndarray,
    grid_vecs: np.ndarray,
    out_sign: float,
) -> None:
    """Collects 2-in/2-out saddle quad split into two triangles per tet."""
    mask = np.sum(tet_in, axis=1) == 2
    if not np.any(mask):
        return
    in_flag = tet_in[mask]
    coords_sub = t_coords[mask].astype(np.float64)
    flat_sub = t_flat[mask]
    in_idx = np.nonzero(in_flag)[1].reshape(-1, 2)
    out_idx = np.nonzero(~in_flag)[1].reshape(-1, 2)
    rows = np.arange(len(flat_sub))[:, None]
    ia = flat_sub[rows, in_idx[:, 0:1]][:, 0]
    ib = flat_sub[rows, in_idx[:, 1:2]][:, 0]
    oc = flat_sub[rows, out_idx[:, 0:1]][:, 0]
    od = flat_sub[rows, out_idx[:, 1:2]][:, 0]

    e_ac = np.stack([ia, oc], axis=1)
    e_ad = np.stack([ia, od], axis=1)
    e_bc = np.stack([ib, oc], axis=1)
    e_bd = np.stack([ib, od], axis=1)
    t1 = np.stack([e_ac, e_ad, e_bd], axis=1)
    t2 = np.stack([e_ac, e_bd, e_bc], axis=1)
    tri_pairs.append(np.concatenate([t1, t2], axis=0))

    in_center = np.mean(coords_sub[rows, in_idx], axis=1) @ grid_vecs
    out_center = np.mean(coords_sub[rows, out_idx], axis=1) @ grid_vecs
    d_vec = (out_center - in_center) * out_sign
    tri_dirs.append(np.concatenate([d_vec, d_vec], axis=0))


def _flat_to_ijk(f_idx: np.ndarray, ny: int, nz: int) -> np.ndarray:
    """Converts linearized 3D grid indices back to ``(i, j, k)`` triples."""
    ix = f_idx // (ny * nz)
    rem = f_idx % (ny * nz)
    iy = rem // nz
    iz = rem % nz
    return np.column_stack([ix, iy, iz]).astype(np.float64)


def _orient_triangle_faces(
    vertices: np.ndarray,
    faces: np.ndarray,
    all_dirs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Enforces outward triangle winding and filters degenerate triangles."""
    cross = np.cross(
        vertices[faces[:, 1]] - vertices[faces[:, 0]],
        vertices[faces[:, 2]] - vertices[faces[:, 0]],
    )
    flip = np.einsum("ij,ij->i", cross, all_dirs) < 0.0
    faces = faces.copy()
    tmp_col = faces[flip, 1].copy()
    faces[flip, 1] = faces[flip, 2]
    faces[flip, 2] = tmp_col
    cross[flip] = -cross[flip]
    norms = np.linalg.norm(cross, axis=1, keepdims=True)
    valid = norms[:, 0] > 1e-14
    return faces[valid], cross[valid] / norms[valid]


def _deduplicate_edges_and_orient(
    all_pairs: np.ndarray,
    all_dirs: np.ndarray,
    grid: electronic.VolumetricGrid,
    level: float,
) -> IsosurfaceMesh:
    """Builds indexed mesh from edge pairs and orients outward normals."""
    nx, ny, nz = grid.data.shape
    n_tot = nx * ny * nz
    edge_keys = (
        np.minimum(all_pairs[:, :, 0], all_pairs[:, :, 1]) * n_tot
        + np.maximum(all_pairs[:, :, 0], all_pairs[:, :, 1])
    )
    uniq_keys, inv = np.unique(edge_keys.ravel(), return_inverse=True)
    u_flat = uniq_keys // n_tot
    v_flat = uniq_keys % n_tot
    flat_vals = grid.data.ravel()
    dv = flat_vals[v_flat] - flat_vals[u_flat]
    alpha = np.clip(
        (level - flat_vals[u_flat]) / np.where(np.abs(dv) < 1e-15, 1e-15, dv),
        0.0,
        1.0,
    )
    g_u = _flat_to_ijk(u_flat, ny, nz)
    g_pos = g_u + alpha[:, None] * (_flat_to_ijk(v_flat, ny, nz) - g_u)
    vertices = grid.origin[None, :] + g_pos @ grid.voxel_vectors
    faces, normals = _orient_triangle_faces(
        vertices, inv.reshape(-1, 3).astype(np.int64), all_dirs
    )
    return IsosurfaceMesh(
        vertices=vertices,
        faces=faces,
        normals=normals,
        level=float(level),
    )


def extract_isosurface(
    grid: electronic.VolumetricGrid,
    level: float,
) -> IsosurfaceMesh:
    """Extracts a triangulated 3D isosurface from a ``VolumetricGrid``.

    Uses vectorized Kuhn 6-tetrahedra Marching Tetrahedra with exact shared
    edge vertex deduplication and consistent outward face normals.

    Args:
      grid: Input ``VolumetricGrid`` scalar field of shape ``(nx, ny, nz)``.
      level: Target isovalue.

    Returns:
      ``IsosurfaceMesh`` with Cartesian vertices in Angstrom, triangle indices,
      and unit outward normals.

    Raises:
      StructureError: If grid dimensions are below 2 or ``level`` is invalid.
    """
    if any(dim < 2 for dim in grid.data.shape):
        raise exceptions.StructureError(
            "Volumetric grid dimensions must all be >= 2 for isosurface."
        )
    if not np.isfinite(level):
        raise exceptions.StructureError("Isosurface level must be finite.")

    t_coords, t_flat, t_vals = _active_tetrahedra(grid.data, float(level))
    if len(t_flat) == 0:
        return _empty_mesh(level)

    tet_in = t_vals >= float(level)
    out_sign = 1.0 if float(level) >= 0.0 else -1.0
    tri_pairs: list[np.ndarray] = []
    tri_dirs: list[np.ndarray] = []
    _append_single_corner_tris(
        tri_pairs,
        tri_dirs,
        t_coords,
        t_flat,
        tet_in,
        grid.voxel_vectors,
        select_inside_lone=True,
        out_sign=out_sign,
    )
    _append_single_corner_tris(
        tri_pairs,
        tri_dirs,
        t_coords,
        t_flat,
        tet_in,
        grid.voxel_vectors,
        select_inside_lone=False,
        out_sign=out_sign,
    )
    _append_two_in_tris(
        tri_pairs,
        tri_dirs,
        t_coords,
        t_flat,
        tet_in,
        grid.voxel_vectors,
        out_sign,
    )
    if not tri_pairs:
        return _empty_mesh(level)

    all_pairs = np.concatenate(tri_pairs, axis=0)
    all_dirs = np.concatenate(tri_dirs, axis=0)
    return _deduplicate_edges_and_orient(
        all_pairs, all_dirs, grid, float(level)
    )


def upsample_coarse_grid(
    grid: electronic.VolumetricGrid,
    min_dim: int = 36,
) -> electronic.VolumetricGrid:
    """Smoothly cubic-upsamples coarse volumetric grids for round 3D lobes."""
    nx, ny, nz = grid.data.shape
    if min(nx, ny, nz) < 5 or min(nx, ny, nz) >= min_dim:
        return grid
    new_nx = max(nx, min_dim)
    new_ny = max(ny, min_dim)
    new_nz = max(nz, min_dim)
    interp = interpolate.RegularGridInterpolator(
        (
            np.arange(nx, dtype=np.float64),
            np.arange(ny, dtype=np.float64),
            np.arange(nz, dtype=np.float64),
        ),
        np.asarray(grid.data, dtype=np.float64),
        method="cubic",
    )
    gx, gy, gz = np.meshgrid(
        np.linspace(0.0, float(nx - 1), new_nx),
        np.linspace(0.0, float(ny - 1), new_ny),
        np.linspace(0.0, float(nz - 1), new_nz),
        indexing="ij",
    )
    pts = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    new_data = interp(pts).reshape(new_nx, new_ny, new_nz)
    scales = np.array(
        [
            float(nx - 1) / float(new_nx - 1),
            float(ny - 1) / float(new_ny - 1),
            float(nz - 1) / float(new_nz - 1),
        ],
        dtype=np.float64,
    )
    return electronic.VolumetricGrid(
        data=new_data,
        origin=grid.origin.copy(),
        voxel_vectors=grid.voxel_vectors * scales[:, None],
        atoms=grid.atoms,
    )
