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
"""Unified LLL-reduced periodic neighbor search for crystals and MD."""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING

import numpy as np
from scipy import spatial

from xkcd_matchem.core import exceptions


if TYPE_CHECKING:
    from xkcd_matchem.core import structure


@dataclasses.dataclass(frozen=True)
class NeighborHit:
    """Pairwise neighbor record between atom ``i`` and periodic image of ``j``.

    Attributes:
      i: Center atom index.
      j: Neighbor atom index.
      distance: Euclidean distance in Angstrom.
      vector: Displacement vector ``r_j + shift @ cell - r_i`` in Angstrom.
      shift: Integer lattice vector offset ``(sa, sb, sc)`` for atom ``j``.
    """

    i: int
    j: int
    distance: float
    vector: np.ndarray
    shift: tuple[int, int, int] = (0, 0, 0)


def _gram_schmidt(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Computes Gram-Schmidt orthogonalization for a small 1D-3D basis."""
    n_vec = mat.shape[0]
    ortho = np.zeros_like(mat)
    mu = np.eye(n_vec, dtype=np.float64)
    for row_i in range(n_vec):
        vec = mat[row_i].copy()
        for row_j in range(row_i):
            denom = float(np.dot(ortho[row_j], ortho[row_j]))
            if denom > 1e-18:
                mu[row_i, row_j] = float(np.dot(mat[row_i], ortho[row_j])) / (
                    denom
                )
                vec -= mu[row_i, row_j] * ortho[row_j]
        ortho[row_i] = vec
    return ortho, mu


def _lll_reduce_with_transform(
    basis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Applies LLL lattice reduction tracking the unimodular integer matrix."""
    red = np.array(basis, dtype=np.float64, copy=True)
    n_vec = red.shape[0]
    u_mat = np.eye(n_vec, dtype=np.int64)
    if n_vec <= 1:
        return red, u_mat

    k = 1
    iters = 0
    while k < n_vec and iters < 200:
        iters += 1
        ortho, mu = _gram_schmidt(red)
        for j in range(k - 1, -1, -1):
            q = round(mu[k, j])
            if q != 0:
                red[k] -= q * red[j]
                u_mat[k] -= q * u_mat[j]
                ortho, mu = _gram_schmidt(red)
        lhs = float(np.dot(ortho[k], ortho[k]))
        rhs = (0.75 - mu[k, k - 1] ** 2) * float(
            np.dot(ortho[k - 1], ortho[k - 1])
        )
        if lhs < rhs:
            red[[k, k - 1]] = red[[k - 1, k]]
            u_mat[[k, k - 1]] = u_mat[[k - 1, k]]
            k = max(k - 1, 1)
        else:
            k += 1
    return red, u_mat


def _canonicalize_undirected(
    i_idx: int,
    j_idx: int,
    shift: tuple[int, int, int],
    vec: np.ndarray,
) -> tuple[int, int, tuple[int, int, int], np.ndarray]:
    """Returns canonical undirected key/vector for deduplicating bond pairs."""
    if i_idx < j_idx:
        return i_idx, j_idx, shift, vec
    if i_idx > j_idx or shift < (0, 0, 0):
        return (
            j_idx,
            i_idx,
            (-shift[0], -shift[1], -shift[2]),
            -vec,
        )
    return i_idx, j_idx, shift, vec


def _aperiodic_neighbors(
    positions: np.ndarray,
    max_cutoff: float,
    min_cutoff: float,
    symmetric: bool,
) -> list[NeighborHit]:
    """Finds neighbor hits without periodic boundary conditions."""
    tree = spatial.cKDTree(positions)
    pairs = tree.query_pairs(r=max_cutoff, output_type="ndarray")
    if len(pairs) == 0:
        return []
    vecs = positions[pairs[:, 1]] - positions[pairs[:, 0]]
    dists = np.linalg.norm(vecs, axis=1)
    keep = dists >= min_cutoff
    pairs = pairs[keep]
    vecs = vecs[keep]
    dists = dists[keep]
    hits: list[NeighborHit] = []
    for k in range(len(pairs)):
        i_idx = int(pairs[k, 0])
        j_idx = int(pairs[k, 1])
        dist = float(dists[k])
        vec = vecs[k]
        hits.append(NeighborHit(i=i_idx, j=j_idx, distance=dist, vector=vec))
        if symmetric:
            hits.append(
                NeighborHit(i=j_idx, j=i_idx, distance=dist, vector=-vec)
            )
    if symmetric:
        hits.sort(key=lambda h: (h.i, h.j, h.shift))
    return hits


def _wrap_onto_reduced_basis(
    positions: np.ndarray,
    red_basis: np.ndarray,
    g_inv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Wraps Cartesian coordinates into ``[0, 1)`` of the LLL-reduced cell."""
    frac_r = (positions @ red_basis.T) @ g_inv
    f_wrap = np.mod(frac_r, 1.0)
    f_wrap[np.isclose(f_wrap, 1.0, rtol=0.0, atol=1e-12)] = 0.0
    f_wrap[np.isclose(f_wrap, 0.0, rtol=0.0, atol=1e-12)] = 0.0
    c_red = np.round(frac_r - f_wrap).astype(np.int64)
    wrapped = positions - c_red @ red_basis
    return wrapped, c_red


def _filter_shifts_by_aabb(
    w_pos: np.ndarray,
    shifts_red: np.ndarray,
    red_basis: np.ndarray,
    max_cutoff: float,
) -> np.ndarray:
    """Filters reduced lattice shifts whose AABB exceeds ``max_cutoff``."""
    w_min = np.min(w_pos, axis=0)
    w_max = np.max(w_pos, axis=0)
    shift_vecs = shifts_red.astype(np.float64) @ red_basis
    gap_lo = w_min[None, :] - (w_max[None, :] + shift_vecs)
    gap_hi = (w_min[None, :] + shift_vecs) - w_max[None, :]
    gap = np.maximum(0.0, np.maximum(gap_lo, gap_hi))
    dist_sq = np.einsum("ij,ij->i", gap, gap)
    return np.asarray(
        shifts_red[dist_sq <= max_cutoff * max_cutoff + 1e-7], dtype=np.int64
    )


def _record_periodic_hit(
    i_idx: int,
    j_idx: int,
    dist: float,
    vec: np.ndarray,
    shift: tuple[int, int, int],
    symmetric: bool,
    seen: set[tuple[int, int, tuple[int, int, int]]],
    hits: list[NeighborHit],
) -> None:
    """Appends one directed or canonicalized undirected NeighborHit."""
    if symmetric:
        key = (i_idx, j_idx, shift)
        if key not in seen:
            seen.add(key)
            hits.append(
                NeighborHit(
                    i=i_idx, j=j_idx, distance=dist, vector=vec, shift=shift
                )
            )
        return
    ci, cj, cshift, cvec = _canonicalize_undirected(i_idx, j_idx, shift, vec)
    key = (ci, cj, cshift)
    if key not in seen:
        seen.add(key)
        hits.append(
            NeighborHit(
                i=ci, j=cj, distance=dist, vector=cvec, shift=cshift
            )
        )


def _emit_chunk_hits(
    pair_indices: tuple[np.ndarray, np.ndarray],
    dists: np.ndarray,
    vecs: np.ndarray,
    full_shifts: np.ndarray,
    symmetric: bool,
    seen: set[tuple[int, int, tuple[int, int, int]]],
    hits: list[NeighborHit],
) -> None:
    """Appends filtered COO neighbor hits into ``hits``."""
    i_arr, j_arr = pair_indices
    for i_val, j_val, d_val, vec, sh_row in zip(
        i_arr, j_arr, dists, vecs, full_shifts, strict=True
    ):
        sh = (int(sh_row[0]), int(sh_row[1]), int(sh_row[2]))
        if int(i_val) == int(j_val) and sh == (0, 0, 0):
            continue
        _record_periodic_hit(
            int(i_val), int(j_val), float(d_val), vec, sh, symmetric, seen, hits
        )


def _collect_chunk_hits(
    w_pos: np.ndarray,
    tree_base: spatial.cKDTree,
    c_red: np.ndarray,
    basis_cfg: tuple[np.ndarray, np.ndarray, list[int]],
    chunk_shifts: np.ndarray,
    cutoffs: tuple[float, float, bool],
    seen: set[tuple[int, int, tuple[int, int, int]]],
    hits: list[NeighborHit],
) -> None:
    """Queries a batch of LLL-reduced shifts with a single cKDTree."""
    red_basis, u_mat, active = basis_cfg
    min_cutoff, max_cutoff, symmetric = cutoffs
    chunk_pts = (
        w_pos[None, :, :]
        + (chunk_shifts.astype(np.float64) @ red_basis)[:, None, :]
    ).reshape(-1, 3)
    coo = tree_base.sparse_distance_matrix(
        spatial.cKDTree(chunk_pts), max_cutoff, output_type="coo_matrix"
    )
    if coo.nnz == 0:
        return
    i_arr = np.asarray(coo.row, dtype=np.int64)
    col_arr = np.asarray(coo.col, dtype=np.int64)
    vecs = chunk_pts[col_arr] - w_pos[i_arr]
    dists = np.linalg.norm(vecs, axis=1)
    keep = (dists >= min_cutoff) & (dists <= max_cutoff)
    if not np.any(keep):
        return
    i_arr, col_arr = i_arr[keep], col_arr[keep]
    j_arr = col_arr % len(w_pos)
    net_active = (
        chunk_shifts[col_arr // len(w_pos)] + c_red[i_arr] - c_red[j_arr]
    ) @ u_mat
    full_shifts = np.zeros((len(i_arr), 3), dtype=np.int64)
    for pos_a, ax_idx in enumerate(active):
        full_shifts[:, ax_idx] = net_active[:, pos_a]
    _emit_chunk_hits(
        (i_arr, j_arr),
        dists[keep],
        vecs[keep],
        full_shifts,
        symmetric,
        seen,
        hits,
    )


def find_periodic_neighbors(
    atoms: structure.AtomsData,
    max_cutoff: float,
    min_cutoff: float = 1e-6,
    *,
    use_pbc: bool = True,
    symmetric: bool = False,
) -> list[NeighborHit]:
    """Finds all neighbor pairs within ``[min_cutoff, max_cutoff]`` Angstrom.

    Applies Gauss/LLL lattice reduction, wraps coordinates into the reduced
    unit cell so unwrapped or skewed triclinic geometries work identically to
    orthogonal cells, and queries metric-bounded integer lattice images with
    ``scipy.spatial.cKDTree``.

    Args:
      atoms: Input atomic or crystal structure.
      max_cutoff: Upper distance cutoff in Angstrom.
      min_cutoff: Lower distance cutoff in Angstrom.
      use_pbc: Whether to search across enabled periodic boundary axes.
      symmetric: When ``False``, returns unique undirected pairs ``i <= j``;
        when ``True``, returns all directed ``i -> j`` hits.

    Returns:
      List of ``NeighborHit`` records sorted by ``(i, j, shift)``.
    """
    if len(atoms) == 0 or max_cutoff <= 0.0 or max_cutoff < min_cutoff:
        return []
    active = [ax for ax, flag in enumerate(atoms.pbc) if flag]
    if not use_pbc or atoms.cell is None or not active:
        return _aperiodic_neighbors(
            atoms.positions, max_cutoff, min_cutoff, symmetric
        )

    red_basis, u_mat = _lll_reduce_with_transform(atoms.cell[active])
    gram = red_basis @ red_basis.T
    if abs(float(np.linalg.det(gram))) <= 1e-18:
        raise exceptions.StructureError(
            "Periodic lattice vectors are linearly dependent (singular cell)."
        )
    g_inv = np.linalg.inv(gram)
    w_pos, c_red = _wrap_onto_reduced_basis(atoms.positions, red_basis, g_inv)
    diag_g = np.maximum(np.diag(g_inv), 0.0)
    bounds = [
        int(math.ceil(max_cutoff * math.sqrt(float(dg)) + 1.0 + 1e-9))
        for dg in diag_g
    ]
    ranges = [np.arange(-b, b + 1, dtype=np.int64) for b in bounds]
    grids = np.meshgrid(*ranges, indexing="ij")
    shifts_red = _filter_shifts_by_aabb(
        w_pos,
        np.column_stack([g.ravel() for g in grids]),
        red_basis,
        max_cutoff,
    )
    if len(shifts_red) == 0:
        return []

    tree_base = spatial.cKDTree(w_pos)
    seen: set[tuple[int, int, tuple[int, int, int]]] = set()
    hits: list[NeighborHit] = []
    basis_cfg = (red_basis, u_mat, active)
    cutoffs = (min_cutoff, max_cutoff, symmetric)
    batch_size = max(1, 65536 // max(len(w_pos), 1))
    for start in range(0, len(shifts_red), batch_size):
        _collect_chunk_hits(
            w_pos,
            tree_base,
            c_red,
            basis_cfg,
            shifts_red[start : start + batch_size],
            cutoffs,
            seen,
            hits,
        )
    hits.sort(key=lambda h: (h.i, h.j, h.shift))
    return hits
