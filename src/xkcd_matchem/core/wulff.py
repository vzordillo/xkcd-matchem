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
"""Thermodynamic equilibrium Wulff crystal shape solver."""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import functools
import itertools

import numpy as np
from scipy import spatial

from xkcd_matchem.core import exceptions
from xkcd_matchem.data import spacegroups


@dataclasses.dataclass(frozen=True)
class WulffFacet:
    """One active crystallographic facet on a converged Wulff polyhedron.

    Attributes:
      hkl: Miller index triple ``(h, k, l)`` of the facet.
      normal: Outward Cartesian unit normal vector ``(3,)``.
      surface_energy: Input surface energy for this facet family.
      vertices: Ordered CCW 3D polygon vertices ``(M, 3)``.
      area: Polygon facet area in squared-energy-scaled units.
      area_fraction: Fraction of total Wulff surface area contributed here.
    """

    hkl: tuple[int, int, int]
    normal: np.ndarray
    surface_energy: float
    vertices: np.ndarray
    area: float
    area_fraction: float


@dataclasses.dataclass(frozen=True)
class WulffShape:
    """Converged equilibrium Wulff polyhedron geometry and energetics.

    Attributes:
      vertices: All polyhedron corner coordinates ``(V, 3)``.
      facets: Active facets ordered by decreasing area fraction.
      volume: Polyhedron volume enclosed by the Wulff planes.
      total_area: Sum of all active facet surface areas.
    """

    vertices: np.ndarray
    facets: tuple[WulffFacet, ...]
    volume: float
    total_area: float


@functools.lru_cache(maxsize=1)
def _cubic_rotations() -> tuple[np.ndarray, ...]:
    """Generates the 48 octahedral O_h signed-permutation matrices."""
    mats: list[np.ndarray] = []
    for perm in itertools.permutations((0, 1, 2)):
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            mat = np.zeros((3, 3), dtype=np.float64)
            for row, (col, sgn) in enumerate(zip(perm, signs, strict=True)):
                mat[row, col] = sgn
            mats.append(mat)
    return tuple(mats)


@functools.lru_cache(maxsize=1)
def _hexagonal_rotations() -> tuple[np.ndarray, ...]:
    """Generates the 24 hexagonal D_6h crystallographic rotation matrices."""
    c6 = np.array([[0, -1, 0], [1, 1, 0], [0, 0, 1]], dtype=np.float64)
    mir = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
    inv = -np.eye(3, dtype=np.float64)
    mats: list[np.ndarray] = [np.eye(3, dtype=np.float64)]
    seen: set[tuple[int, ...]] = {tuple(int(round(v)) for v in mats[0].ravel())}

    def _add(mat: np.ndarray) -> None:
        key = tuple(int(round(v)) for v in mat.ravel())
        if key not in seen:
            seen.add(key)
            mats.append(mat)

    for gen in (c6, mir, inv):
        _add(gen)
    idx = 0
    while idx < len(mats) and len(mats) < 48:
        cur = mats[idx]
        for other in list(mats):
            _add(cur @ other)
        idx += 1
    return tuple(mats)


def _resolve_point_rotations(
    symops: Sequence[tuple[np.ndarray, np.ndarray]] | str | int | None,
) -> list[np.ndarray]:
    """Resolves reciprocal-space symmetry matrices for Miller expansion."""
    out: list[np.ndarray] | None = None
    if symops is None:
        out = [np.eye(3, dtype=np.float64)]
    elif isinstance(symops, str):
        key = symops.strip().lower()
        presets = {
            "none": [np.eye(3, dtype=np.float64)],
            "cubic": list(_cubic_rotations()),
            "hexagonal": list(_hexagonal_rotations()),
            "hex": list(_hexagonal_rotations()),
        }
        if key in presets:
            out = presets[key]
        else:
            ops = spacegroups.lookup_spacegroup_symops(sg_name=symops)
            if ops is None:
                raise exceptions.StructureError(
                    f"Unknown Wulff symmetry specification {symops!r}."
                )
            out = [np.linalg.inv(rot).T for rot, _ in ops]
    elif isinstance(symops, int):
        ops = spacegroups.lookup_spacegroup_symops(sg_number=symops)
        if ops is None:
            raise exceptions.StructureError(
                f"Space group number {symops} not available in table."
            )
        out = [np.linalg.inv(rot).T for rot, _ in ops]
    else:
        out = [np.linalg.inv(rot).T for rot, _ in symops]
    return out


def _expand_symmetry_planes(
    cell_arr: np.ndarray,
    surface_energies: Mapping[tuple[int, int, int], float],
    rotations: list[np.ndarray],
) -> tuple[list[tuple[int, int, int]], np.ndarray, np.ndarray]:
    """Expands Miller families into unique oriented halfspace planes."""
    recip = np.linalg.inv(cell_arr).T
    planes: dict[
        tuple[int, int, int],
        tuple[tuple[int, int, int], np.ndarray, float],
    ] = {}
    for raw_hkl, raw_gamma in surface_energies.items():
        if len(raw_hkl) != 3:
            raise exceptions.StructureError(
                f"Miller index must be a 3-tuple, got {raw_hkl!r}."
            )
        hkl_tuple = (int(raw_hkl[0]), int(raw_hkl[1]), int(raw_hkl[2]))
        if hkl_tuple == (0, 0, 0):
            raise exceptions.StructureError("Miller index (0, 0, 0) invalid.")
        gamma = float(raw_gamma)
        if not np.isfinite(gamma) or gamma <= 0.0:
            raise exceptions.StructureError(
                f"Surface energy for {hkl_tuple} must be > 0, got {gamma}."
            )
        hkl_vec = np.asarray(hkl_tuple, dtype=np.float64)
        for rot in rotations:
            sym_hkl = rot @ hkl_vec
            sym_int: tuple[int, int, int] = (
                int(round(float(sym_hkl[0]))),
                int(round(float(sym_hkl[1]))),
                int(round(float(sym_hkl[2]))),
            )
            if sym_int == (0, 0, 0):
                continue
            cart_n = np.asarray(sym_int, dtype=np.float64) @ recip
            n_norm = float(np.linalg.norm(cart_n))
            if n_norm < 1e-12:
                continue
            unit_n = cart_n / n_norm
            n_key = (
                int(round(float(unit_n[0]) * 1_000_000)),
                int(round(float(unit_n[1]) * 1_000_000)),
                int(round(float(unit_n[2]) * 1_000_000)),
            )
            prev = planes.get(n_key)
            if prev is None or gamma < prev[2]:
                planes[n_key] = (sym_int, unit_n, gamma)

    hkls = [entry[0] for entry in planes.values()]
    normals = np.vstack([entry[1] for entry in planes.values()])
    gammas = np.array([entry[2] for entry in planes.values()], dtype=np.float64)
    return hkls, normals, gammas


def _order_polygon_ccw(pts: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Orders coplanar 3D polygon vertices counter-clockwise around normal."""
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
    return pts[np.argsort(angles)]


def _polygon_area_3d(ordered_pts: np.ndarray, normal: np.ndarray) -> float:
    """Computes signed magnitude area of an ordered planar 3D polygon."""
    p0 = ordered_pts[0]
    total = np.zeros(3, dtype=np.float64)
    for k in range(1, len(ordered_pts) - 1):
        total += np.cross(ordered_pts[k] - p0, ordered_pts[k + 1] - p0)
    return abs(0.5 * float(np.dot(total, normal)))


def compute_wulff_shape(
    cell: np.ndarray,
    surface_energies: Mapping[tuple[int, int, int], float],
    *,
    symops: (
        Sequence[tuple[np.ndarray, np.ndarray]] | str | int | None
    ) = "cubic",
) -> WulffShape:
    """Computes the equilibrium Wulff construction via halfspace intersection.

    Args:
      cell: Lattice matrix of shape ``(3, 3)`` with row vectors ``a, b, c``.
      surface_energies: Mapping ``{(h, k, l): gamma}`` with ``gamma > 0``.
      symops: Symmetry expansion mode (``"cubic"``, ``"hexagonal"``, ``"none"``,
        space-group symbol/IT number, or affine ``(rot, trans)`` list).

    Returns:
      ``WulffShape`` containing vertices, ordered facets, volume, and area.

    Raises:
      StructureError: If inputs are malformed or do not bound a closed volume.
    """
    cell_arr = np.asarray(cell, dtype=np.float64)
    if cell_arr.shape != (3, 3) or abs(float(np.linalg.det(cell_arr))) < 1e-12:
        raise exceptions.StructureError(
            "cell must be a non-singular (3, 3) matrix."
        )
    if not surface_energies:
        raise exceptions.StructureError(
            "surface_energies must contain at least one (h, k, l) entry."
        )

    rotations = _resolve_point_rotations(symops)
    hkls, normals, gammas = _expand_symmetry_planes(
        cell_arr, surface_energies, rotations
    )
    if len(normals) < 4:
        raise exceptions.StructureError(
            "At least 4 non-coplanar planes are required for a closed Wulff."
        )

    halfspaces = np.column_stack([normals, -gammas])
    interior = np.zeros(3, dtype=np.float64)
    try:
        hs = spatial.HalfspaceIntersection(halfspaces, interior)
    except (spatial.QhullError, ValueError) as exc:
        raise exceptions.StructureError(
            "Surface planes do not bound a finite closed Wulff polyhedron."
        ) from exc

    verts = np.asarray(hs.intersections, dtype=np.float64)
    raw_facets: list[
        tuple[tuple[int, int, int], np.ndarray, float, np.ndarray, float]
    ] = []
    for p_idx, hkl_val in enumerate(hkls):
        n_vec = normals[p_idx]
        g_val = float(gammas[p_idx])
        on_plane = np.abs(verts @ n_vec - g_val) <= 1e-6 * max(g_val, 1.0)
        if int(np.sum(on_plane)) < 3:
            continue
        ordered = _order_polygon_ccw(verts[on_plane], n_vec)
        area = _polygon_area_3d(ordered, n_vec)
        if area > 1e-8:
            raw_facets.append((hkl_val, n_vec, g_val, ordered, area))

    if not raw_facets:
        raise exceptions.StructureError("Wulff construction yielded no facets.")

    total_area = sum(item[4] for item in raw_facets)
    volume = sum(item[4] * item[2] for item in raw_facets) / 3.0
    facets = tuple(
        WulffFacet(
            hkl=hkl_v,
            normal=n_v,
            surface_energy=g_v,
            vertices=pts_v,
            area=area_v,
            area_fraction=area_v / total_area,
        )
        for hkl_v, n_v, g_v, pts_v, area_v in sorted(
            raw_facets, key=lambda it: -it[4]
        )
    )
    return WulffShape(
        vertices=verts,
        facets=facets,
        volume=volume,
        total_area=total_area,
    )
