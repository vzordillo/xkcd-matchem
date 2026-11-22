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
import importlib
import math
from typing import Any

import numpy as np
from scipy import spatial

from xkcd_matchem.core import exceptions
from xkcd_matchem.core import geometry
from xkcd_matchem.core import structure
from xkcd_matchem.data import elements


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
      volume: Exact convex-hull Brillouin zone volume in A^-3.
      expected_volume: Analytical ``(2*pi)^3 / |det(A_p)|`` volume in A^-3.
      volume_error_rel: Relative error ``|V - V_expected| / V_expected``.
      classification_method: Path classifier (``"heuristic"`` or
        ``"seekpath"``).
    """

    vertices: np.ndarray
    facets: tuple[BrillouinFacet, ...]
    reciprocal_cell: np.ndarray
    lattice_system: str
    high_symmetry_points: dict[str, np.ndarray]
    default_path: tuple[str, ...]
    volume: float = 0.0
    expected_volume: float = 0.0
    volume_error_rel: float = 0.0
    classification_method: str = "heuristic"


def _has_translation(
    frac: np.ndarray,
    symbols: Sequence[str],
    shift: tuple[float, float, float],
) -> bool:
    """Returns True if fractional coordinates are invariant under ``shift``."""
    if len(frac) < 2:
        return False
    sh = np.asarray(shift, dtype=np.float64)
    by_sym: dict[str, list[int]] = {}
    for idx, sym in enumerate(symbols):
        by_sym.setdefault(sym, []).append(idx)
    for indices in by_sym.values():
        sub = frac[indices]
        shifted = sub[:, None, :] + sh[None, None, :] - sub[None, :, :]
        diffs = np.mod(shifted + 0.5, 1.0) - 0.5
        max_err = np.max(np.abs(diffs), axis=2)
        if not np.all(np.any(max_err < 1e-3, axis=1)):
            return False
    return True


def _reduce_conventional_cell(atoms: structure.AtomsData) -> np.ndarray:
    """Reduces F/I/C/A/B/R conventional cells to primitive lattice vectors."""
    if atoms.cell is None:
        raise exceptions.StructureError("Crystal cell is missing.")
    cell = np.asarray(atoms.cell, dtype=np.float64)
    if len(atoms) < 2 or not all(atoms.pbc):
        return cell
    frac = atoms.fractional_positions()
    f_shifts = ((0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0))
    r_obv = (
        (2.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
        (1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0),
    )
    r_rev = (
        (1.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0),
        (2.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0),
    )
    candidates: tuple[
        tuple[tuple[tuple[float, float, float], ...], np.ndarray], ...
    ] = (
        (
            f_shifts,
            0.5
            * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.float64),
        ),
        (
            ((0.5, 0.5, 0.5),),
            0.5
            * np.array(
                [[-1, 1, 1], [1, -1, 1], [1, 1, -1]], dtype=np.float64
            ),
        ),
        (
            ((0.5, 0.5, 0.0),),
            np.array([[0.5, 0.5, 0.0], [-0.5, 0.5, 0.0], [0.0, 0.0, 1.0]]),
        ),
        (
            ((0.0, 0.5, 0.5),),
            np.array([[1.0, 0.0, 0.0], [0.0, 0.5, -0.5], [0.0, 0.5, 0.5]]),
        ),
        (
            ((0.5, 0.0, 0.5),),
            np.array([[0.5, 0.0, -0.5], [0.0, 1.0, 0.0], [0.5, 0.0, 0.5]]),
        ),
        (
            r_obv,
            np.array([[2, 1, 1], [-1, 1, 1], [-1, -2, 1]], dtype=np.float64)
            / 3.0,
        ),
        (
            r_rev,
            np.array([[1, -1, 1], [1, 2, 1], [-2, -1, 1]], dtype=np.float64)
            / 3.0,
        ),
    )
    for shifts, t_mat in candidates:
        if all(_has_translation(frac, atoms.symbols, s) for s in shifts):
            return np.asarray(t_mat @ cell, dtype=np.float64)
    return cell


def _lll_reduce_rows(basis: np.ndarray) -> np.ndarray:
    """Applies 3D LLL lattice basis reduction while preserving lattice span."""
    return geometry.lll_reduce_cell(basis)


def _load_seekpath_module(backend_key: str) -> Any | None:
    """Resolves optional ``seekpath`` module according to ``backend_key``."""
    if backend_key not in ("heuristic", "seekpath", "auto"):
        raise exceptions.StructureError(
            f"Unsupported Brillouin zone backend {backend_key!r}; "
            "expected 'heuristic', 'seekpath', or 'auto'."
        )
    if backend_key == "heuristic":
        return None
    try:
        return importlib.import_module("seekpath")
    except ImportError as exc:
        if backend_key == "seekpath":
            raise exceptions.OptionalDependencyError(
                "Brillouin zone backend='seekpath' requires the optional "
                "'seekpath' package (pip install seekpath)."
            ) from exc
        return None


def _classify_with_seekpath(
    source: structure.AtomsData | np.ndarray | Sequence[Sequence[float]],
    raw_cell: np.ndarray,
    use_primitive: bool,
    sk_mod: Any,
) -> tuple[np.ndarray, np.ndarray, str, dict[str, np.ndarray], tuple[str, ...]]:
    """Standardizes primitive cell and k-path via SeeK-path."""
    if isinstance(source, structure.AtomsData) and len(source) > 0:
        frac_pos = source.fractional_positions().tolist()
        numbers = [
            elements.get_element(sym).atomic_number for sym in source.symbols
        ]
    else:
        frac_pos = [[0.0, 0.0, 0.0]]
        numbers = [1]
    sk_res = sk_mod.get_path((raw_cell.tolist(), frac_pos, numbers))
    prim = (
        np.asarray(sk_res["primitive_lattice"], dtype=np.float64)
        if use_primitive and "primitive_lattice" in sk_res
        else raw_cell
    )
    recip = 2.0 * math.pi * np.linalg.inv(prim).T
    bravais_map = {
        "cF": "fcc",
        "cI": "bcc",
        "cP": "cubic",
        "hP": "hexagonal",
    }
    raw_bravais = str(sk_res.get("bravais_lattice", "general"))
    sys_name = bravais_map.get(raw_bravais, raw_bravais.lower())
    sym_pts: dict[str, np.ndarray] = {}
    for raw_lbl, frac_k in sk_res.get("point_coords", {}).items():
        lbl = "G" if raw_lbl == "GAMMA" else str(raw_lbl)
        sym_pts[lbl] = np.asarray(
            np.asarray(frac_k, dtype=np.float64) @ recip, dtype=np.float64
        )
    if "G" not in sym_pts:
        sym_pts["G"] = np.zeros(3, dtype=np.float64)
    walk: list[str] = []
    for seg in sk_res.get("path", ()):
        u_lbl = "G" if seg[0] == "GAMMA" else str(seg[0])
        v_lbl = "G" if seg[1] == "GAMMA" else str(seg[1])
        if not walk:
            walk.extend([u_lbl, v_lbl])
        elif walk[-1] == u_lbl:
            walk.append(v_lbl)
        else:
            walk.extend([u_lbl, v_lbl])
    path = tuple(walk) if walk else ("G",)
    return prim, recip, sys_name, sym_pts, path


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


def _deduplicate_points_3d(pts: np.ndarray, tol: float) -> np.ndarray:
    """Merges coincident 3D polyhedron intersections within ``tol``."""
    if len(pts) <= 1:
        return pts
    uniq: list[np.ndarray] = [pts[0]]
    tol_sq = tol * tol
    for idx in range(1, len(pts)):
        cand = pts[idx]
        arr_u = np.vstack(uniq)
        diffs = arr_u - cand[None, :]
        if float(np.min(np.einsum("ij,ij->i", diffs, diffs))) > tol_sq:
            uniq.append(cand)
    return np.vstack(uniq)


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
    if len(ordered) < 3:
        return ordered, 0.0
    p0 = ordered[0]
    total = np.sum(
        np.cross(ordered[1:-1] - p0[None, :], ordered[2:] - p0[None, :]),
        axis=0,
    )
    area = abs(0.5 * float(np.dot(total, normal)))
    return ordered, area


def _enumerate_brillouin_planes(
    prim: np.ndarray,
    recip: np.ndarray,
) -> list[tuple[tuple[int, int, int], np.ndarray, float]]:
    """Enumerates shortest unique Bragg bisectors from LLL-reduced basis."""
    reduced_b = geometry.lll_reduce_cell(recip)
    b_norms = np.linalg.norm(reduced_b, axis=1)
    r_cover = 0.5 * float(np.sum(b_norms))
    g_cut = 2.0 * r_cover * (1.0 + 1e-6)
    gram_inv = np.linalg.inv(reduced_b @ reduced_b.T)
    diag_inv = np.maximum(np.diag(gram_inv), 0.0)
    bounds = [
        min(max(int(math.ceil(g_cut * math.sqrt(float(d)) + 1e-9)), 3), 64)
        for d in diag_inv
    ]
    ranges = [np.arange(-b, b + 1, dtype=np.int64) for b in bounds]
    g1, g2, g3 = np.meshgrid(ranges[0], ranges[1], ranges[2], indexing="ij")
    grid = np.column_stack([g1.ravel(), g2.ravel(), g3.ravel()])
    abs_grid = np.abs(grid)
    gcd_vals = np.gcd(np.gcd(abs_grid[:, 0], abs_grid[:, 1]), abs_grid[:, 2])
    prim_rays = grid[gcd_vals == 1]

    g_vecs = prim_rays.astype(np.float64) @ reduced_b
    g_norms = np.linalg.norm(g_vecs, axis=1)
    valid = (g_norms > 1e-12) & (g_norms <= g_cut)
    g_vecs = g_vecs[valid]
    g_norms = g_norms[valid]
    unit_normals = g_vecs / g_norms[:, None]
    hkl_arr = np.rint((g_vecs @ prim.T) / (2.0 * math.pi)).astype(np.int64)

    plane_entries: list[tuple[tuple[int, int, int], np.ndarray, float]] = []
    for idx, g_norm in enumerate(g_norms):
        hkl: tuple[int, int, int] = (
            int(hkl_arr[idx, 0]),
            int(hkl_arr[idx, 1]),
            int(hkl_arr[idx, 2]),
        )
        plane_entries.append((hkl, unit_normals[idx], 0.5 * float(g_norm)))
    return plane_entries


def _build_bz_polyhedron(
    plane_entries: list[tuple[tuple[int, int, int], np.ndarray, float]],
) -> tuple[np.ndarray, tuple[BrillouinFacet, ...]]:
    """Intersects Bragg half-spaces and extracts area-sorted BZ facets."""
    normals = np.vstack([entry[1] for entry in plane_entries])
    dists = np.array([entry[2] for entry in plane_entries], dtype=np.float64)
    halfspaces = np.column_stack([normals, -dists])
    hs = spatial.HalfspaceIntersection(
        halfspaces, np.zeros(3, dtype=np.float64)
    )
    raw_verts = np.asarray(hs.intersections, dtype=np.float64)
    v_scale = (
        max(float(np.max(np.abs(raw_verts))), 1.0) if raw_verts.size else 1.0
    )
    verts = _deduplicate_points_3d(raw_verts, tol=1e-7 * v_scale)
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
    return verts, tuple(facets)


def compute_brillouin_zone(
    source: structure.AtomsData | np.ndarray | Sequence[Sequence[float]],
    *,
    use_primitive: bool = True,
    backend: str = "heuristic",
) -> BrillouinZone:
    """Computes the Wigner-Seitz first Brillouin zone in reciprocal space.

    Args:
      source: Periodic ``AtomsData`` crystal or ``(3, 3)`` real-space lattice
        matrix in Angstrom.
      use_primitive: Automatically reduce F/I/C-centered conventional cells to
        their standard primitive cells before constructing the BZ.
      backend: Symmetry/path classifier: ``"heuristic"``, ``"seekpath"``, or
        ``"auto"`` (uses ``seekpath`` if installed, else ``"heuristic"``).

    Returns:
      ``BrillouinZone`` containing vertices, active facets, exact convex-hull
      volume, expected analytical volume, and high-symmetry k-points in A^-1.

    Raises:
      StructureError: If input lacks a valid 3D cell or ``backend`` is unknown.
      OptionalDependencyError: If ``backend="seekpath"`` and ``seekpath`` is
        not installed.
    """
    sk_mod = _load_seekpath_module(backend.strip().lower())
    if isinstance(source, structure.AtomsData):
        if source.cell is None or not any(source.pbc):
            raise exceptions.StructureError(
                "Brillouin zone requires a periodic 3D crystal structure."
            )
        raw_cell = np.asarray(source.cell, dtype=np.float64)
        heur_prim = (
            _reduce_conventional_cell(source) if use_primitive else raw_cell
        )
    else:
        try:
            raw_cell = np.asarray(source, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise exceptions.StructureError(
                "Lattice cell must be a non-singular (3, 3) matrix."
            ) from exc
        heur_prim = raw_cell
    if (
        raw_cell.shape != (3, 3)
        or abs(float(np.linalg.det(raw_cell))) < 1e-12
    ):
        raise exceptions.StructureError(
            "Lattice cell must be a non-singular (3, 3) matrix."
        )

    if sk_mod is not None:
        prim, recip, sys_name, sym_pts, path = _classify_with_seekpath(
            source, raw_cell, use_primitive, sk_mod
        )
        method_used = "seekpath"
    else:
        prim = heur_prim
        recip = 2.0 * math.pi * np.linalg.inv(prim).T
        sys_name, sym_pts, path = _classify_bravais(prim, recip)
        method_used = "heuristic"

    verts, facets = _build_bz_polyhedron(
        _enumerate_brillouin_planes(prim, recip)
    )
    volume = float(spatial.ConvexHull(verts).volume)
    expected_volume = ((2.0 * math.pi) ** 3) / abs(float(np.linalg.det(prim)))
    volume_error_rel = abs(volume - expected_volume) / max(
        expected_volume, 1e-30
    )
    return BrillouinZone(
        vertices=verts,
        facets=facets,
        reciprocal_cell=recip,
        lattice_system=sys_name,
        high_symmetry_points=sym_pts,
        default_path=path,
        volume=volume,
        expected_volume=expected_volume,
        volume_error_rel=volume_error_rel,
        classification_method=method_used,
    )
