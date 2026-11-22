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
"""Thermodynamic equilibrium Wulff crystal shapes and Birch-Murnaghan EOS."""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import functools
import itertools
import math
import warnings

import numpy as np
from scipy import optimize
from scipy import spatial

from xkcd_matchem.core import exceptions
from xkcd_matchem.data import elements
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
        out = []
        for rot, _ in symops:
            r_arr = np.asarray(rot, dtype=np.float64)
            if (
                r_arr.shape != (3, 3)
                or abs(float(np.linalg.det(r_arr))) < 1e-12
            ):
                raise exceptions.StructureError(
                    "Symmetry rotation matrix must be non-singular (3, 3)."
                )
            out.append(np.linalg.inv(r_arr).T)
    return out


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


def _expand_symmetry_planes(
    cell_arr: np.ndarray,
    surface_energies: Mapping[tuple[int, int, int], float],
    rotations: list[np.ndarray],
) -> tuple[list[tuple[int, int, int]], np.ndarray, np.ndarray]:
    """Expands Miller families into unique oriented halfspace planes."""
    recip = np.linalg.inv(cell_arr).T
    hkls: list[tuple[int, int, int]] = []
    normals_list: list[np.ndarray] = []
    gammas_list: list[float] = []
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
            matched_idx = -1
            for existing_idx, prev_n in enumerate(normals_list):
                if float(np.linalg.norm(unit_n - prev_n)) < 1e-5:
                    matched_idx = existing_idx
                    break
            if matched_idx < 0:
                hkls.append(sym_int)
                normals_list.append(unit_n)
                gammas_list.append(gamma)
            elif gamma < gammas_list[matched_idx]:
                hkls[matched_idx] = sym_int
                normals_list[matched_idx] = unit_n
                gammas_list[matched_idx] = gamma

    if not normals_list:
        return (
            [],
            np.zeros((0, 3), dtype=np.float64),
            np.zeros(0, dtype=np.float64),
        )
    normals = np.vstack(normals_list)
    gammas = np.array(gammas_list, dtype=np.float64)
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
    if len(ordered_pts) < 3:
        return 0.0
    p0 = ordered_pts[0]
    total = np.sum(
        np.cross(
            ordered_pts[1:-1] - p0[None, :], ordered_pts[2:] - p0[None, :]
        ),
        axis=0,
    )
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

    raw_verts = np.asarray(hs.intersections, dtype=np.float64)
    v_scale = (
        max(float(np.max(np.abs(raw_verts))), 1.0) if raw_verts.size else 1.0
    )
    verts = _deduplicate_points_3d(raw_verts, tol=1e-7 * v_scale)
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


_EOS_BASIS_ALIASES: dict[str, str] = {
    "cell": "cell",
    "formula_unit": "formula_unit",
    "f.u.": "formula_unit",
    "fu": "formula_unit",
    "atom": "atom",
}


def normalize_eos_basis(energy_basis: str) -> str:
    """Validates and normalizes EOS ``energy_basis`` names."""
    key = energy_basis.strip().lower()
    if key not in _EOS_BASIS_ALIASES:
        raise exceptions.StructureError(
            f"Unsupported energy_basis {energy_basis!r}; "
            "expected 'cell', 'formula_unit', or 'atom'."
        )
    return _EOS_BASIS_ALIASES[key]


@dataclasses.dataclass(frozen=True)
class EOSFitResult:
    """Fitted third-order Birch-Murnaghan equation-of-state parameters.

    Attributes:
      e0: Equilibrium ground-state energy ``E_0`` in eV.
      v0: Equilibrium volume ``V_0`` in cubic Angstrom.
      b0_ev_a3: Equilibrium isothermal bulk modulus ``B_0`` in eV/A^3.
      b0_gpa: Equilibrium isothermal bulk modulus ``B_0`` in GPa.
      b0_prime: Pressure derivative ``B_0' = (dB/dP)_0`` (dimensionless).
      rms_residual: Root-mean-square energy residual in eV.
      v0_err: 1-sigma covariance std error on ``V_0`` in A^3, if finite.
      e0_err: 1-sigma covariance std error on ``E_0`` in eV, if finite.
      b0_gpa_err: 1-sigma covariance std error on ``B_0`` in GPa, if finite.
      bp_err: 1-sigma covariance std error on ``B_0'``, if finite.
      rmse_ev: Root-mean-square energy fit error in eV.
      r_squared: Coefficient of determination ``R^2`` on input points.
      spans_minimum: True when ``min(volumes) < v0 < max(volumes)``.
      fit_status: ``"ok"`` when spanning minimum, else ``"extrapolated"``.
      energy_basis: Energy normalization basis (``"cell"``,
        ``"formula_unit"``, or ``"atom"``).
    """

    e0: float
    v0: float
    b0_ev_a3: float
    b0_gpa: float
    b0_prime: float
    rms_residual: float
    v0_err: float | None = None
    e0_err: float | None = None
    b0_gpa_err: float | None = None
    bp_err: float | None = None
    rmse_ev: float = 0.0
    r_squared: float = 1.0
    spans_minimum: bool = True
    fit_status: str = "ok"
    energy_basis: str = "cell"


def _validate_eos_inputs(
    volume: Sequence[float] | np.ndarray | float,
    v0: float,
    b0_ev_a3: float,
    b0_prime: float,
) -> np.ndarray:
    """Validates positive volumes, equilibrium volume, and bulk modulus."""
    vol = np.asarray(volume, dtype=np.float64)
    if vol.size == 0 or not np.all(np.isfinite(vol)) or np.any(vol <= 0.0):
        raise exceptions.StructureError(
            "EOS volume values must be finite and strictly positive."
        )
    if (
        not math.isfinite(v0)
        or v0 <= 0.0
        or not math.isfinite(b0_ev_a3)
        or b0_ev_a3 <= 0.0
        or not math.isfinite(b0_prime)
    ):
        raise exceptions.StructureError(
            "EOS parameters require finite v0 > 0, b0_ev_a3 > 0, and b0_prime."
        )
    return vol


def birch_murnaghan_energy(
    volume: Sequence[float] | np.ndarray | float,
    e0: float,
    v0: float,
    b0_ev_a3: float,
    b0_prime: float = 4.0,
) -> np.ndarray:
    """Evaluates the 3rd-order Birch-Murnaghan energy ``E(V)`` in eV."""
    vol = _validate_eos_inputs(volume, v0, b0_ev_a3, b0_prime)
    if not math.isfinite(e0):
        raise exceptions.StructureError("Equilibrium energy e0 must be finite.")
    eta = np.power(v0 / vol, 2.0 / 3.0)
    f2 = eta - 1.0
    energy = e0 + (9.0 * v0 * b0_ev_a3 / 16.0) * (
        (f2**3) * b0_prime + (f2**2) * (6.0 - 4.0 * eta)
    )
    return np.asarray(energy, dtype=np.float64)


def birch_murnaghan_pressure(
    volume: Sequence[float] | np.ndarray | float,
    v0: float,
    b0_ev_a3: float,
    b0_prime: float = 4.0,
    *,
    unit: str = "GPa",
) -> np.ndarray:
    """Evaluates the 3rd-order Birch-Murnaghan pressure ``P(V) = -dE/dV``."""
    vol = _validate_eos_inputs(volume, v0, b0_ev_a3, b0_prime)
    u_key = unit.strip().lower()
    if u_key == "gpa":
        scale = elements.CODATA2022.ev_per_a3_to_gpa
    elif u_key in ("ev/a^3", "ev/a3"):
        scale = 1.0
    else:
        raise exceptions.StructureError(
            f"Unsupported pressure unit {unit!r}; use 'GPa' or 'eV/A^3'."
        )
    ratio = v0 / vol
    eta = np.power(ratio, 2.0 / 3.0)
    p_ev = (
        1.5
        * b0_ev_a3
        * (np.power(ratio, 7.0 / 3.0) - np.power(ratio, 5.0 / 3.0))
        * (1.0 + 0.75 * (b0_prime - 4.0) * (eta - 1.0))
    )
    return np.asarray(p_ev * scale, dtype=np.float64)


def _validate_eos_fit_arrays(
    volumes: Sequence[float] | np.ndarray,
    energies: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validates matching 1D positive volume and finite energy arrays."""
    vs = np.asarray(volumes, dtype=np.float64)
    es = np.asarray(energies, dtype=np.float64)
    shape_bad = vs.ndim != 1 or es.ndim != 1 or vs.shape != es.shape
    if shape_bad or vs.size < 4:
        raise exceptions.StructureError(
            "Birch-Murnaghan fit requires >= 4 matching 1D E(V) points."
        )
    vals_bad = not (np.all(np.isfinite(vs)) and np.all(np.isfinite(es)))
    if vals_bad or np.any(vs <= 0.0) or float(np.ptp(vs)) < 1e-9:
        raise exceptions.StructureError(
            "Birch-Murnaghan fit requires distinct finite positive volumes."
        )
    return vs, es


def _extract_pcov_errors(
    pcov: np.ndarray,
) -> tuple[float | None, float | None, float | None, float | None]:
    """Extracts finite 1-sigma std errors ``(e0, v0, b0_gpa, bp)``."""
    cov_arr = np.asarray(pcov, dtype=np.float64)
    if cov_arr.shape != (4, 4):
        return None, None, None, None
    diag = np.diag(cov_arr)
    stds: list[float | None] = [
        math.sqrt(float(val))
        if np.isfinite(val) and float(val) >= 0.0
        else None
        for val in diag
    ]
    b0_gpa_err = (
        stds[2] * elements.CODATA2022.ev_per_a3_to_gpa
        if stds[2] is not None
        else None
    )
    return stds[0], stds[1], b0_gpa_err, stds[3]


def _initial_eos_guess(
    vs: np.ndarray,
    es: np.ndarray,
) -> list[float]:
    """Computes quadratic + Eulerian-strain OLS initial parameter estimate."""
    v_min, v_max = float(np.min(vs)), float(np.max(vs))
    min_idx = int(np.argmin(es))
    v0_g, e0_g, b0_g, bp_g = float(vs[min_idx]), float(es[min_idx]), 0.5, 4.0
    quad = np.polyfit(vs, es, 2)
    if quad[0] > 1e-12:
        v0_g = float(
            np.clip(-quad[1] / (2.0 * quad[0]), 0.65 * v_min, 1.45 * v_max)
        )
        b0_g = float(np.clip(2.0 * quad[0] * v0_g, 1e-4, 25.0))
        e0_g = float(np.polyval(quad, v0_g))
    fe = 0.5 * (np.power(v0_g / vs, 2.0 / 3.0) - 1.0)
    c_ols, _, _, _ = np.linalg.lstsq(
        np.column_stack([np.ones_like(fe), fe**2, fe**3]), es, rcond=None
    )
    if float(c_ols[1]) > 1e-9:
        e0_g = float(c_ols[0])
        b0_g = float(np.clip(2.0 * float(c_ols[1]) / (9.0 * v0_g), 1e-4, 25.0))
        bp_g = float(
            np.clip(4.0 + float(c_ols[2]) / float(c_ols[1]), 0.5, 12.0)
        )
    return [
        e0_g,
        float(np.clip(v0_g, 0.55 * v_min, 1.9 * v_max)),
        float(np.clip(b0_g, 1e-5, 25.0)),
        float(np.clip(bp_g, -1.5, 20.0)),
    ]


def fit_birch_murnaghan(
    volumes: Sequence[float] | np.ndarray,
    energies: Sequence[float] | np.ndarray,
    *,
    energy_basis: str = "cell",
    allow_extrapolation: bool = True,
) -> EOSFitResult:
    """Fits the 3rd-order Birch-Murnaghan EOS via Eulerian-strain OLS + NLS."""
    canon_basis = normalize_eos_basis(energy_basis)
    vs, es = _validate_eos_fit_arrays(volumes, energies)
    v_min, v_max = float(np.min(vs)), float(np.max(vs))
    p0 = _initial_eos_guess(vs, es)
    bounds = (
        [-np.inf, 0.4 * v_min, 1e-6, -2.0],
        [np.inf, 2.5 * v_max, 50.0, 25.0],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optimize.OptimizeWarning)
        popt, pcov = optimize.curve_fit(
            birch_murnaghan_energy,
            vs,
            es,
            p0=p0,
            bounds=bounds,
            maxfev=12000,
        )
    e0_f, v0_f, b0_f, bp_f = (float(v) for v in popt)
    spans_min = bool(v_min < v0_f < v_max)
    if not spans_min:
        if not allow_extrapolation:
            raise exceptions.StructureError(
                f"Fitted equilibrium volume v0={v0_f:.3f} A^3 lies outside "
                f"input volume interval [{v_min:.3f}, {v_max:.3f}] A^3."
            )
        fit_status = "extrapolated"
    else:
        fit_status = "ok"
    residuals = es - birch_murnaghan_energy(vs, e0_f, v0_f, b0_f, bp_f)
    rmse = math.sqrt(float(np.mean(np.square(residuals))))
    ss_res = float(np.sum(np.square(residuals)))
    ss_tot = float(np.sum(np.square(es - float(np.mean(es)))))
    r_sq = (
        1.0 - (ss_res / ss_tot)
        if ss_tot > 1e-30
        else (1.0 if ss_res <= 1e-30 else 0.0)
    )
    e0_err, v0_err, b0_gpa_err, bp_err = _extract_pcov_errors(pcov)
    return EOSFitResult(
        e0=e0_f,
        v0=v0_f,
        b0_ev_a3=b0_f,
        b0_gpa=b0_f * elements.CODATA2022.ev_per_a3_to_gpa,
        b0_prime=bp_f,
        rms_residual=rmse,
        v0_err=v0_err,
        e0_err=e0_err,
        b0_gpa_err=b0_gpa_err,
        bp_err=bp_err,
        rmse_ev=rmse,
        r_squared=r_sq,
        spans_minimum=spans_min,
        fit_status=fit_status,
        energy_basis=canon_basis,
    )
