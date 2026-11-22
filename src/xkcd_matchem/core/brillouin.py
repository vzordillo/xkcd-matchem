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
"""Pure-NumPy/SciPy 3D first Brillouin zone Wigner-Seitz solver."""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import math

import numpy as np
from scipy import spatial

from xkcd_matchem.core import exceptions
from xkcd_matchem.core import structure


@dataclasses.dataclass(frozen=True)
class BrillouinFacet:
    """One planar Bragg-plane facet bounding the first Brillouin zone.

    Attributes:
      hkl: Reciprocal-lattice Miller vector ``(h, k, l)``.
      normal: Outward unit normal vector ``(3,)``.
      distance: Orthogonal distance ``||G|| / 2`` from Gamma in A^-1.
      vertices: Ordered CCW 3D corner coordinates ``(M, 3)`` in A^-1.
      area: Polygon facet area in A^-2.
    """

    hkl: tuple[int, int, int]
    normal: np.ndarray
    distance: float
    vertices: np.ndarray
    area: float


@dataclasses.dataclass(frozen=True)
class BrillouinZone:
    """Converged first Brillouin zone polyhedron and high-symmetry path.

    Attributes:
      vertices: Unique BZ polyhedron corners ``(V, 3)`` in A^-1.
      facets: Bounding Bragg-plane facets ordered by area.
      reciprocal_cell: Reciprocal basis matrix ``B = 2*pi*(A_p^-1)^T`` (3x3).
      lattice_system: Detected Bravais class (``"fcc"``, ``"bcc"``,
        ``"hexagonal"``, ``"cubic"``, or ``"general"``).
      high_symmetry_points: Mapping of point names to Cartesian ``k`` (A^-1).
      default_path: Canonical ordered k-point label tour.
      volume: First Brillouin zone volume in A^-3.
    """

    vertices: np.ndarray
    facets: tuple[BrillouinFacet, ...]
    reciprocal_cell: np.ndarray
    lattice_system: str
    high_symmetry_points: dict[str, np.ndarray]
    default_path: tuple[str, ...]
    volume: float


def _has_translation(
    frac: np.ndarray,
    symbols: Sequence[str],
    shift: tuple[float, float, float],
) -> bool:
    """Returns True if fractional coordinates are invariant under ``shift``."""
    if len(frac) < 2:
        return False
    sh = np.asarray(shift, dtype=np.float64)
    for idx, f_i in enumerate(frac):
        sym_i = symbols[idx]
        matched = False
        for jdx, f_j in enumerate(frac):
            if symbols[jdx] != sym_i:
                continue
            diff = np.mod((f_i + sh - f_j) + 0.5, 1.0) - 0.5
            if float(np.max(np.abs(diff))) < 1e-3:
                matched = True
                break
        if not matched:
            return False
    return True


def _reduce_conventional_cell(atoms: structure.AtomsData) -> np.ndarray:
    """Reduces F/I/C conventional cells to primitive lattice vectors."""
    if atoms.cell is None:
        raise exceptions.StructureError("Crystal cell is missing.")
    cell = np.asarray(atoms.cell, dtype=np.float64)
    if len(atoms) < 2 or not all(atoms.pbc):
        return cell
    frac = atoms.fractional_positions()
    f_shifts = ((0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0))
    if all(_has_translation(frac, atoms.symbols, s) for s in f_shifts):
        t_f = 0.5 * np.array(
            [[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.float64
        )
        return np.asarray(t_f @ cell, dtype=np.float64)
    if _has_translation(frac, atoms.symbols, (0.5, 0.5, 0.5)):
        t_i = 0.5 * np.array(
            [[-1, 1, 1], [1, -1, 1], [1, 1, -1]], dtype=np.float64
        )
        return np.asarray(t_i @ cell, dtype=np.float64)
    if _has_translation(frac, atoms.symbols, (0.5, 0.5, 0.0)):
        t_c = np.array(
            [[0.5, 0.5, 0.0], [-0.5, 0.5, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        return np.asarray(t_c @ cell, dtype=np.float64)
    return cell


def _lll_reduce_rows(basis: np.ndarray) -> np.ndarray:
    """Applies 3D LLL lattice basis reduction while preserving lattice span."""
    b_mat = np.array(basis, dtype=np.float64, copy=True)
    n_vec = b_mat.shape[0]
    for _ in range(40):
        changed = False
        q_mat = np.zeros_like(b_mat)
        mu = np.zeros((n_vec, n_vec), dtype=np.float64)
        for i in range(n_vec):
            vec = b_mat[i].copy()
            for j in range(i):
                denom = max(float(np.dot(q_mat[j], q_mat[j])), 1e-18)
                mu[i, j] = float(np.dot(b_mat[i], q_mat[j])) / denom
                vec -= mu[i, j] * q_mat[j]
            q_mat[i] = vec
        for i in range(1, n_vec):
            for j in range(i - 1, -1, -1):
                q_ij = round(float(mu[i, j]))
                if q_ij != 0:
                    b_mat[i] -= q_ij * b_mat[j]
                    mu[i, : j + 1] -= q_ij * mu[j, : j + 1]
                    changed = True
        swapped = False
        for k in range(n_vec - 1):
            lhs = float(np.dot(q_mat[k + 1], q_mat[k + 1]))
            rhs = (0.75 - mu[k + 1, k] ** 2) * float(
                np.dot(q_mat[k], q_mat[k])
            )
            if lhs < rhs - 1e-12:
                b_mat[[k, k + 1]] = b_mat[[k + 1, k]]
                swapped = True
                break
        if not changed and not swapped:
            break
    return b_mat


def _classify_bravais(
    prim_cell: np.ndarray,
    recip_cell: np.ndarray,
) -> tuple[str, dict[str, np.ndarray], tuple[str, ...]]:
    """Identifies standard Bravais type and high-symmetry k-points."""
    lens = np.linalg.norm(prim_cell, axis=1)
    a_l, b_l, c_l = float(lens[0]), float(lens[1]), float(lens[2])
    cos_ab = float(np.dot(prim_cell[0], prim_cell[1])) / max(a_l * b_l, 1e-12)
    cos_ac = float(np.dot(prim_cell[0], prim_cell[2])) / max(a_l * c_l, 1e-12)
    cos_bc = float(np.dot(prim_cell[1], prim_cell[2])) / max(b_l * c_l, 1e-12)
    ang = tuple(
        math.degrees(math.acos(max(-1.0, min(1.0, c_v))))
        for c_v in (cos_bc, cos_ac, cos_ab)
    )
    eq_abc = (
        abs(a_l - b_l) < 1e-2 * max(a_l, 1.0)
        and abs(a_l - c_l) < 1e-2 * max(a_l, 1.0)
    )
    eq_ab = abs(a_l - b_l) < 1e-2 * max(a_l, 1.0)
    path: tuple[str, ...]

    if eq_abc and all(abs(v - 60.0) < 1.0 for v in ang):
        frac_pts = {
            "G": (0.0, 0.0, 0.0),
            "X": (0.5, 0.0, 0.5),
            "W": (0.5, 0.25, 0.75),
            "K": (0.375, 0.375, 0.75),
            "L": (0.5, 0.5, 0.5),
            "U": (0.625, 0.25, 0.625),
        }
        sys_name, path = "fcc", ("G", "X", "W", "K", "G", "L")
    elif eq_abc and all(abs(v - 109.47122) < 1.0 for v in ang):
        frac_pts = {
            "G": (0.0, 0.0, 0.0),
            "H": (0.5, -0.5, 0.5),
            "P": (0.25, 0.25, 0.25),
            "N": (0.0, 0.0, 0.5),
        }
        sys_name, path = "bcc", ("G", "H", "N", "G", "P")
    elif (
        eq_ab
        and abs(ang[0] - 90.0) < 1.0
        and abs(ang[1] - 90.0) < 1.0
        and (abs(ang[2] - 120.0) < 1.0 or abs(ang[2] - 60.0) < 1.0)
    ):
        k_sgn = 1.0 / 3.0 if abs(ang[2] - 120.0) < 1.0 else -1.0 / 3.0
        frac_pts = {
            "G": (0.0, 0.0, 0.0),
            "M": (0.5, 0.0, 0.0),
            "K": (1.0 / 3.0, k_sgn, 0.0),
            "A": (0.0, 0.0, 0.5),
            "L": (0.5, 0.0, 0.5),
            "H": (1.0 / 3.0, k_sgn, 0.5),
        }
        sys_name, path = "hexagonal", ("G", "M", "K", "G", "A")
    elif eq_abc and all(abs(v - 90.0) < 1.0 for v in ang):
        frac_pts = {
            "G": (0.0, 0.0, 0.0),
            "X": (0.0, 0.5, 0.0),
            "M": (0.5, 0.5, 0.0),
            "R": (0.5, 0.5, 0.5),
        }
        sys_name, path = "cubic", ("G", "X", "M", "G", "R")
    else:
        frac_pts = {
            "G": (0.0, 0.0, 0.0),
            "X": (0.5, 0.0, 0.0),
            "Y": (0.0, 0.5, 0.0),
            "Z": (0.0, 0.0, 0.5),
            "R": (0.5, 0.5, 0.5),
        }
        sys_name, path = "general", ("G", "X", "R", "Z", "G")

    cart_pts = {
        lbl: np.asarray(
            np.asarray(f_q, dtype=np.float64) @ recip_cell, dtype=np.float64
        )
        for lbl, f_q in frac_pts.items()
    }
    return sys_name, cart_pts, path


def _order_ccw_and_area(
    pts: np.ndarray,
    normal: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Orders coplanar 3D facet vertices CCW and computes polygon area."""
    center = np.mean(pts, axis=0)
    trial = (
        np.array([1.0, 0.0, 0.0])
        if abs(float(normal[0])) < 0.85
        else np.array([0.0, 1.0, 0.0])
    )
    e_u = np.cross(normal, trial)
    e_u /= max(float(np.linalg.norm(e_u)), 1e-12)
    e_v = np.cross(normal, e_u)
    rel = pts - center[None, :]
    angles = np.arctan2(rel @ e_v, rel @ e_u)
    ordered = pts[np.argsort(angles)]
    p0 = ordered[0]
    total = np.zeros(3, dtype=np.float64)
    for idx in range(1, len(ordered) - 1):
        total += np.cross(ordered[idx] - p0, ordered[idx + 1] - p0)
    area = abs(0.5 * float(np.dot(total, normal)))
    return ordered, area


def _enumerate_brillouin_planes(
    prim: np.ndarray,
    recip: np.ndarray,
) -> list[tuple[tuple[int, int, int], np.ndarray, float]]:
    """Enumerates shortest unique Bragg bisectors from LLL-reduced basis."""
    reduced_b = _lll_reduce_rows(recip)
    shortest: dict[
        tuple[int, int, int], tuple[float, np.ndarray, np.ndarray]
    ] = {}
    for n1 in range(-3, 4):
        for n2 in range(-3, 4):
            for n3 in range(-3, 4):
                if n1 == 0 and n2 == 0 and n3 == 0:
                    continue
                g_vec = (
                    n1 * reduced_b[0] + n2 * reduced_b[1] + n3 * reduced_b[2]
                )
                g_norm = float(np.linalg.norm(g_vec))
                if g_norm < 1e-12:
                    continue
                unit_n = g_vec / g_norm
                key = (
                    int(round(float(unit_n[0]) * 1_000_000)),
                    int(round(float(unit_n[1]) * 1_000_000)),
                    int(round(float(unit_n[2]) * 1_000_000)),
                )
                prev = shortest.get(key)
                if prev is None or g_norm < prev[0]:
                    shortest[key] = (g_norm, unit_n, g_vec)

    plane_entries: list[tuple[tuple[int, int, int], np.ndarray, float]] = []
    for g_norm, unit_n, g_vec in shortest.values():
        hkl_f = (g_vec @ prim.T) / (2.0 * math.pi)
        hkl: tuple[int, int, int] = (
            int(round(float(hkl_f[0]))),
            int(round(float(hkl_f[1]))),
            int(round(float(hkl_f[2]))),
        )
        plane_entries.append((hkl, unit_n, 0.5 * g_norm))
    return plane_entries


def compute_brillouin_zone(
    source: structure.AtomsData | np.ndarray | Sequence[Sequence[float]],
    *,
    use_primitive: bool = True,
) -> BrillouinZone:
    """Computes the Wigner-Seitz first Brillouin zone in reciprocal space.

    Args:
      source: Periodic ``AtomsData`` crystal or ``(3, 3)`` real-space lattice
        matrix in Angstrom.
      use_primitive: Automatically reduce F/I/C-centered conventional cells to
        their standard primitive cells before constructing the BZ.

    Returns:
      ``BrillouinZone`` containing vertices, active facets, volume, and
      high-symmetry k-points in inverse Angstrom.

    Raises:
      StructureError: If input lacks a valid 3D periodic unit cell.
    """
    if isinstance(source, structure.AtomsData):
        if source.cell is None or not any(source.pbc):
            raise exceptions.StructureError(
                "Brillouin zone requires a periodic 3D crystal structure."
            )
        prim = (
            _reduce_conventional_cell(source)
            if use_primitive
            else np.asarray(source.cell, dtype=np.float64)
        )
    else:
        try:
            prim = np.asarray(source, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise exceptions.StructureError(
                "Lattice cell must be a non-singular (3, 3) matrix."
            ) from exc
    if prim.shape != (3, 3) or abs(float(np.linalg.det(prim))) < 1e-12:
        raise exceptions.StructureError(
            "Lattice cell must be a non-singular (3, 3) matrix."
        )

    recip = 2.0 * math.pi * np.linalg.inv(prim).T
    plane_entries = _enumerate_brillouin_planes(prim, recip)

    normals = np.vstack([entry[1] for entry in plane_entries])
    dists = np.array([entry[2] for entry in plane_entries], dtype=np.float64)
    halfspaces = np.column_stack([normals, -dists])
    hs = spatial.HalfspaceIntersection(
        halfspaces, np.zeros(3, dtype=np.float64)
    )
    verts = np.asarray(hs.intersections, dtype=np.float64)

    facets: list[BrillouinFacet] = []
    for hkl_v, n_v, d_v in plane_entries:
        on_plane = np.abs(verts @ n_v - d_v) <= 1e-6 * max(d_v, 1.0)
        if int(np.sum(on_plane)) < 3:
            continue
        ordered, area = _order_ccw_and_area(verts[on_plane], n_v)
        if area > 1e-8:
            facets.append(
                BrillouinFacet(
                    hkl=hkl_v,
                    normal=n_v,
                    distance=d_v,
                    vertices=ordered,
                    area=area,
                )
            )

    facets.sort(key=lambda f: -f.area)
    volume = sum(f.area * f.distance for f in facets) / 3.0
    sys_name, sym_pts, path = _classify_bravais(prim, recip)
    return BrillouinZone(
        vertices=verts,
        facets=tuple(facets),
        reciprocal_cell=recip,
        lattice_system=sys_name,
        high_symmetry_points=sym_pts,
        default_path=path,
        volume=volume,
    )
