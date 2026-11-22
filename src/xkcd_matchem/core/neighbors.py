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
    hits: list[NeighborHit] = []
    for row in pairs:
        i_idx = int(row[0])
        j_idx = int(row[1])
        vec = positions[j_idx] - positions[i_idx]
        dist = float(np.linalg.norm(vec))
        if dist < min_cutoff:
            continue
        hits.append(NeighborHit(i=i_idx, j=j_idx, distance=dist, vector=vec))
        if symmetric:
            hits.append(
                NeighborHit(i=j_idx, j=i_idx, distance=dist, vector=-vec)
            )
    return hits


def _wrap_onto_reduced_basis(
    positions: np.ndarray,
    red_basis: np.ndarray,
    g_inv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Wraps Cartesian coordinates into ``[0, 1)`` of the LLL-reduced cell."""
    frac_r = (positions @ red_basis.T) @ g_inv
    f_wrap = np.mod(frac_r, 1.0)
    f_wrap[np.isclose(f_wrap, 1.0, atol=1e-9)] = 0.0
    f_wrap[np.isclose(f_wrap, 0.0, atol=1e-9)] = 0.0
    c_red = np.round(frac_r - f_wrap).astype(np.int64)
    wrapped = positions - c_red @ red_basis
    return wrapped, c_red


def _record_periodic_hit(
    i_idx: int,
    j_idx: int,
    vec: np.ndarray,
    shift: tuple[int, int, int],
    symmetric: bool,
    seen: set[tuple[int, int, tuple[int, int, int]]],
    hits: list[NeighborHit],
) -> None:
    """Appends one directed or canonicalized undirected NeighborHit."""
    dist = float(np.linalg.norm(vec))
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


def _append_periodic_shift_hits(
    w_pos: np.ndarray,
    tree_base: spatial.cKDTree,
    c_red: np.ndarray,
    basis_cfg: tuple[np.ndarray, np.ndarray, list[int]],
    s_red: np.ndarray,
    cutoffs: tuple[float, float, bool],
    seen: set[tuple[int, int, tuple[int, int, int]]],
    hits: list[NeighborHit],
) -> None:
    """Queries one LLL-reduced integer lattice shift and appends hits."""
    red_basis, u_mat, active = basis_cfg
    min_cutoff, max_cutoff, symmetric = cutoffs
    shifted = w_pos + s_red.astype(np.float64) @ red_basis
    tree_sh = spatial.cKDTree(shifted)
    for i_idx, nbrs in enumerate(
        tree_base.query_ball_tree(tree_sh, r=max_cutoff)
    ):
        for j_idx in nbrs:
            vec = shifted[j_idx] - w_pos[i_idx]
            dist = float(np.linalg.norm(vec))
            if dist < min_cutoff or dist > max_cutoff:
                continue
            net_active = (s_red + c_red[i_idx] - c_red[j_idx]) @ u_mat
            shift_arr = [0, 0, 0]
            for pos_a, ax_idx in enumerate(active):
                shift_arr[ax_idx] = int(net_active[pos_a])
            shift = (shift_arr[0], shift_arr[1], shift_arr[2])
            if i_idx == j_idx and shift == (0, 0, 0):
                continue
            _record_periodic_hit(
                i_idx, j_idx, vec, shift, symmetric, seen, hits
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
    if len(atoms) == 0 or max_cutoff <= 0.0:
        return []
    active = [ax for ax, flag in enumerate(atoms.pbc) if flag]
    if not use_pbc or atoms.cell is None or not active:
        return _aperiodic_neighbors(
            atoms.positions, max_cutoff, min_cutoff, symmetric
        )

    red_basis, u_mat = _lll_reduce_with_transform(atoms.cell[active])
    gram = red_basis @ red_basis.T
    g_inv = np.linalg.inv(gram)
    w_pos, c_red = _wrap_onto_reduced_basis(atoms.positions, red_basis, g_inv)
    diag_g = np.maximum(np.diag(g_inv), 0.0)
    bounds = [
        min(
            12,
            int(math.ceil(max_cutoff * math.sqrt(float(dg)) + 1.0 + 1e-9)),
        )
        for dg in diag_g
    ]
    ranges = [np.arange(-b, b + 1, dtype=np.int64) for b in bounds]
    grids = np.meshgrid(*ranges, indexing="ij")
    shifts_red = np.column_stack([g.ravel() for g in grids])

    tree_base = spatial.cKDTree(w_pos)
    seen: set[tuple[int, int, tuple[int, int, int]]] = set()
    hits: list[NeighborHit] = []
    basis_cfg = (red_basis, u_mat, active)
    cutoffs = (min_cutoff, max_cutoff, symmetric)
    for s_red in shifts_red:
        _append_periodic_shift_hits(
            w_pos, tree_base, c_red, basis_cfg, s_red, cutoffs, seen, hits
        )
    hits.sort(key=lambda h: (h.i, h.j, h.shift))
    return hits
