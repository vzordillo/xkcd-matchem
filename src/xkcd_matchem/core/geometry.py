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
"""Projection camera, crystallographic cells, and Miller plane geometry."""

from __future__ import annotations

import dataclasses
import math

import numpy as np

from xkcd_matchem.core import exceptions


_DEFAULT_ISO_AZIMUTH = -118.0
_DEFAULT_ISO_ELEVATION = 24.0


@dataclasses.dataclass(frozen=True)
class Camera:
    """Orthographic or weak-perspective 3D viewing camera parameters.

    Attributes:
      azimuth: Rotation around the z-axis in degrees.
      elevation: Tilt angle above the xy-plane in degrees.
      roll: In-plane screen roll angle in degrees.
      perspective: Perspective foresight strength (``0.0`` is orthographic).
      zoom: Multiplicative scale applied after centering.
    """

    azimuth: float = _DEFAULT_ISO_AZIMUTH
    elevation: float = _DEFAULT_ISO_ELEVATION
    roll: float = 0.0
    perspective: float = 0.0
    zoom: float = 1.0

    @classmethod
    def from_preset(
        cls,
        name: str = "isometric",
        cell: np.ndarray | None = None,
        perspective: float = 0.0,
        zoom: float = 1.0,
    ) -> Camera:
        """Creates a ``Camera`` from a standard crystallographic view preset.

        Args:
          name: View name: ``"isometric"``, ``"perspective"``, ``"front"``,
            ``"side"``, ``"top"``, ``"a"``, ``"b"``, ``"c"``, ``"a*"``,
            ``"b*"``, or ``"c*"``.
          cell: Optional ``(3, 3)`` lattice matrix used to orient ``a/b/c``.
          perspective: Perspective strength override; ``"perspective"`` preset
            uses ``0.65`` when ``perspective == 0.0``.
          zoom: Camera zoom multiplier.

        Returns:
          Configured ``Camera`` instance.

        Raises:
          StructureError: If ``name`` is not a recognized camera preset.
        """
        key = name.strip().lower()
        if key == "isometric":
            return cls(
                azimuth=_DEFAULT_ISO_AZIMUTH,
                elevation=_DEFAULT_ISO_ELEVATION,
                perspective=perspective,
                zoom=zoom,
            )
        if key == "perspective":
            return cls(
                azimuth=_DEFAULT_ISO_AZIMUTH,
                elevation=_DEFAULT_ISO_ELEVATION,
                perspective=0.65 if perspective == 0.0 else perspective,
                zoom=zoom,
            )
        fixed = {
            "front": (-90.0, 0.0),
            "side": (0.0, 0.0),
            "top": (-90.0, 89.9),
        }
        if key in fixed:
            az, el = fixed[key]
            return cls(
                azimuth=az, elevation=el, perspective=perspective, zoom=zoom
            )
        axis_map = {"a": 0, "b": 1, "c": 2}
        if key.rstrip("*") in axis_map:
            idx = axis_map[key.rstrip("*")]
            if cell is not None:
                mat = np.asarray(cell, dtype=np.float64)
                vec = (
                    np.linalg.inv(mat).T[idx] if key.endswith("*") else mat[idx]
                )
            else:
                vec = np.eye(3, dtype=np.float64)[idx]
            norm = max(float(np.linalg.norm(vec)), 1e-12)
            v_hat = vec / norm
            el = math.degrees(math.asin(max(-1.0, min(1.0, float(v_hat[2])))))
            az = math.degrees(math.atan2(float(v_hat[1]), float(v_hat[0])))
            return cls(
                azimuth=az, elevation=el, perspective=perspective, zoom=zoom
            )
        raise exceptions.StructureError(
            f"Unknown camera preset {name!r}. Valid presets: isometric, "
            "perspective, front, side, top, a, b, c, a*, b*, c*."
        )


def resolve_camera(
    camera: Camera | str | None,
    cell: np.ndarray | None = None,
) -> Camera:
    """Resolves a ``Camera`` instance from ``None``, preset, or object."""
    if camera is None:
        return Camera()
    if isinstance(camera, str):
        return Camera.from_preset(camera, cell=cell)
    return camera


def rotation_matrix(
    azimuth: float = 35.0,
    elevation: float = 20.0,
    roll: float = 0.0,
) -> np.ndarray:
    """Builds a ``(3, 3)`` orthonormal view basis ``[right; up; forward]``.

    Args:
      azimuth: Azimuth angle in degrees.
      elevation: Elevation angle in degrees.
      roll: Screen roll angle in degrees.

    Returns:
      Row-basis ``(3, 3)`` matrix mapping Cartesian coords to projected screen
      axes ``(screen_x, screen_y, depth)``.
    """
    az = math.radians(azimuth)
    el = math.radians(elevation)
    rl = math.radians(roll)
    forward = np.array(
        [
            math.cos(el) * math.cos(az),
            math.cos(el) * math.sin(az),
            math.sin(el),
        ],
        dtype=np.float64,
    )
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(forward, world_up))) > 0.999:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.cross(world_up, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    up /= np.linalg.norm(up)
    if rl != 0.0:
        cos_r = math.cos(rl)
        sin_r = math.sin(rl)
        new_right = cos_r * right + sin_r * up
        new_up = -sin_r * right + cos_r * up
        right, up = new_right, new_up
    return np.vstack([right, up, forward])


def project_points(
    positions: np.ndarray,
    camera: Camera,
    center: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Projects 3D Cartesian coordinates into 2D screen space and depth.

    Args:
      positions: Array of shape ``(N, 3)`` in Angstrom.
      camera: Viewing ``Camera`` configuration.
      center: Optional rotation pivot in Angstrom; defaults to centroid.

    Returns:
      Tuple ``(xy, depth)`` where ``xy`` has shape ``(N, 2)`` and ``depth`` has
      shape ``(N,)`` with larger values closer to the viewer.
    """
    pts = np.asarray(positions, dtype=np.float64)
    if pts.size == 0:
        return np.zeros((0, 2), dtype=np.float64), np.zeros(
            (0,), dtype=np.float64
        )
    pivot = np.mean(pts, axis=0) if center is None else np.asarray(center)
    shifted = pts - pivot
    basis = rotation_matrix(camera.azimuth, camera.elevation, camera.roll)
    rotated = shifted @ basis.T
    depth = rotated[:, 2]
    xy = rotated[:, :2] * camera.zoom
    if camera.perspective > 0.0:
        span = max(float(np.max(np.abs(rotated))), 1.0)
        denom = np.maximum(
            1.0 - camera.perspective * (depth / span) * 0.15, 0.25
        )
        xy = xy / denom[:, None]
    return xy, depth


def cell_vectors_from_parameters(
    a: float,
    b: float,
    c: float,
    alpha: float,
    beta: float,
    gamma: float,
) -> np.ndarray:
    """Converts cell lengths and angles into ``(3, 3)`` row lattice vectors.

    Follows standard crystallographic convention: ``a`` lies along ``+x``,
    ``b`` lies in the ``xy`` plane, and ``c`` completes a right-handed frame.

    Args:
      a: Lattice parameter ``a`` in Angstrom.
      b: Lattice parameter ``b`` in Angstrom.
      c: Lattice parameter ``c`` in Angstrom.
      alpha: Interaxial angle ``alpha`` (between b and c) in degrees.
      beta: Interaxial angle ``beta`` (between a and c) in degrees.
      gamma: Interaxial angle ``gamma`` (between a and b) in degrees.

    Returns:
      Lattice matrix of shape ``(3, 3)`` whose rows are ``v_a, v_b, v_c``.
    """
    al = math.radians(alpha)
    be = math.radians(beta)
    ga = math.radians(gamma)
    cos_al = 0.0 if abs(math.cos(al)) < 1e-14 else math.cos(al)
    cos_be = 0.0 if abs(math.cos(be)) < 1e-14 else math.cos(be)
    cos_ga = 0.0 if abs(math.cos(ga)) < 1e-14 else math.cos(ga)
    sin_ga = math.sin(ga)
    va = np.array([a, 0.0, 0.0], dtype=np.float64)
    vb = np.array([b * cos_ga, b * sin_ga, 0.0], dtype=np.float64)
    cx = c * cos_be
    cy = c * (cos_al - cos_be * cos_ga) / max(sin_ga, 1e-12)
    if abs(cy) < 1e-14:
        cy = 0.0
    cz_sq = max(0.0, c * c - cx * cx - cy * cy)
    vc = np.array([cx, cy, math.sqrt(cz_sq)], dtype=np.float64)
    return np.vstack([va, vb, vc])


def _reduce_lattice_basis(basis: np.ndarray) -> np.ndarray:
    """Applies Gauss/LLL lattice reduction to a 1D-3D basis matrix."""
    red = np.array(basis, dtype=np.float64, copy=True)
    n_vec = red.shape[0]
    if n_vec <= 1:
        return red

    def _gram_schmidt(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ortho = np.zeros_like(mat)
        mu = np.eye(n_vec, dtype=np.float64)
        for i in range(n_vec):
            vec = mat[i].copy()
            for j in range(i):
                denom = float(np.dot(ortho[j], ortho[j]))
                if denom > 1e-18:
                    mu[i, j] = float(np.dot(mat[i], ortho[j])) / denom
                    vec -= mu[i, j] * ortho[j]
            ortho[i] = vec
        return ortho, mu

    k = 1
    iters = 0
    while k < n_vec and iters < 200:
        iters += 1
        ortho, mu = _gram_schmidt(red)
        for j in range(k - 1, -1, -1):
            q = round(mu[k, j])
            if q != 0:
                red[k] -= q * red[j]
                ortho, mu = _gram_schmidt(red)
        lhs = float(np.dot(ortho[k], ortho[k]))
        rhs = (0.75 - mu[k, k - 1] ** 2) * float(
            np.dot(ortho[k - 1], ortho[k - 1])
        )
        if lhs < rhs:
            red[[k, k - 1]] = red[[k - 1, k]]
            k = max(k - 1, 1)
        else:
            k += 1
    return red


def _integer_shift_grid(bounds: list[int]) -> np.ndarray:
    """Returns all integer shift tuples inside ``[-b, b]`` per periodic axis."""
    ranges = [np.arange(-b, b + 1, dtype=np.int64) for b in bounds]
    grids = np.meshgrid(*ranges, indexing="ij")
    return np.column_stack([g.ravel() for g in grids])


def minimum_image_vector(
    vec: np.ndarray,
    cell: np.ndarray,
    pbc: tuple[bool, bool, bool],
) -> np.ndarray:
    """Computes exact shortest-image displacement on general periodic cells.

    Uses LLL basis reduction followed by metric-bounded integer lattice
    enumeration so skewed and triclinic cells match exact Voronoi distances.

    Args:
      vec: Displacement vector ``(3,)`` or array ``(..., 3)``.
      cell: Row-vector lattice matrix ``(3, 3)``.
      pbc: Periodic boundary flags along each lattice vector.

    Returns:
      Exact shortest periodic displacement(s) with the same shape as ``vec``.
    """
    arr = np.asarray(vec, dtype=np.float64)
    active = [ax for ax, flag in enumerate(pbc) if flag]
    if not active or arr.size == 0:
        return arr.copy()

    basis_p = _reduce_lattice_basis(np.asarray(cell, dtype=np.float64)[active])
    gram = basis_p @ basis_p.T
    g_inv = np.linalg.inv(gram)

    orig_shape = arr.shape
    flat = arr.reshape(-1, 3)
    frac_p = (flat @ basis_p.T) @ g_inv
    r_frac = frac_p - np.round(frac_p)
    v_perp = flat - frac_p @ basis_p
    w0 = r_frac @ basis_p
    r_sq_max = float(np.max(np.sum(w0 * w0, axis=1)))
    diag_g = np.maximum(np.diag(g_inv), 0.0)
    bounds = [
        min(8, int(math.ceil(0.5 + math.sqrt(float(dg) * r_sq_max) + 1e-9)))
        for dg in diag_g
    ]
    shifts = _integer_shift_grid(bounds).astype(np.float64)
    shift_cart = shifts @ basis_p

    best_in_plane = np.empty_like(w0)
    for row_idx in range(flat.shape[0]):
        cands = w0[row_idx : row_idx + 1] + shift_cart
        dists_sq = np.einsum("ij,ij->i", cands, cands)
        best_in_plane[row_idx] = cands[int(np.argmin(dists_sq))]
    return (v_perp + best_in_plane).reshape(orig_shape)


def max_min_image_radius(
    cell: np.ndarray,
    pbc: tuple[bool, bool, bool] = (True, True, True),
) -> float:
    """Returns Wigner-Seitz inradius (half shortest periodic lattice norm)."""
    active = [ax for ax, flag in enumerate(pbc) if flag]
    if not active:
        return float("inf")
    basis_p = _reduce_lattice_basis(np.asarray(cell, dtype=np.float64)[active])
    gram = basis_p @ basis_p.T
    g_inv = np.linalg.inv(gram)
    r_guess = float(np.min(np.linalg.norm(basis_p, axis=1)))
    diag_g = np.maximum(np.diag(g_inv), 0.0)
    bounds = [
        min(8, int(math.ceil(math.sqrt(float(dg)) * r_guess + 1.0)))
        for dg in diag_g
    ]
    shifts = _integer_shift_grid(bounds)
    nonzero = np.any(shifts != 0, axis=1)
    vecs = shifts[nonzero].astype(np.float64) @ basis_p
    min_len = float(np.min(np.linalg.norm(vecs, axis=1)))
    return 0.5 * min_len


def cell_corners_projected(
    cell: np.ndarray,
    camera: Camera,
    center: np.ndarray | None = None,
    origin: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Projects the 8 parallelepiped corners and returns farthest corner index.

    Returns:
      Tuple ``(corners_2d, depth, farthest_idx)`` where corners follow bit order
      ``[000, 100, 010, 001, 110, 101, 011, 111]``.
    """
    orig = (
        np.zeros(3, dtype=np.float64)
        if origin is None
        else np.asarray(origin, dtype=np.float64)
    )
    frac_corners = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ],
        dtype=np.float64,
    )
    corners_3d = orig + frac_corners @ cell
    proj_center = np.mean(corners_3d, axis=0) if center is None else center
    corners_2d, depth = project_points(corners_3d, camera, center=proj_center)
    farthest_idx = int(np.argmin(depth))
    return corners_2d, depth, farthest_idx


def best_fractional_shift(
    frac_coords: np.ndarray,
    axes: tuple[int, ...] = (0, 1),
) -> np.ndarray:
    """Finds a fractional shift putting the widest void on cell faces.

    Args:
      frac_coords: Array of shape ``(N, 3)`` fractional coordinates.
      axes: Axis indices along which to optimize void placement.

    Returns:
      Length-3 float64 shift vector added before ``% 1.0`` wrapping.
    """
    shift = np.zeros(3, dtype=np.float64)
    if frac_coords.size == 0:
        return shift
    for axis in axes:
        fs = np.sort(np.mod(frac_coords[:, axis], 1.0))
        gaps = np.diff(np.append(fs, fs[0] + 1.0))
        widest_idx = int(np.argmax(gaps))
        cut = (fs[widest_idx] + 0.5 * gaps[widest_idx]) % 1.0
        shift[axis] = -cut
    return shift


def unit_cell_edges(
    cell: np.ndarray,
    camera: Camera,
    center: np.ndarray | None = None,
    origin: np.ndarray | None = None,
) -> list[tuple[np.ndarray, np.ndarray, bool]]:
    """Computes the 12 projected unit-cell edges with back-face flags.

    Args:
      cell: Lattice matrix ``(3, 3)`` with lattice vectors as rows.
      camera: Viewing ``Camera`` configuration.
      center: Optional 3D center used for projection alignment.
      origin: Optional origin shift for the unit cell (defaults to zero).

    Returns:
      List of 12 tuples ``(p0_2d, p1_2d, is_back)``, where ``is_back`` is True
      for edges touching the corner with smallest depth (drawn dashed behind).
    """
    corners_2d, _, farthest_idx = cell_corners_projected(
        cell, camera, center=center, origin=origin
    )
    edge_pairs = (
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 4),
        (1, 5),
        (2, 4),
        (2, 6),
        (3, 5),
        (3, 6),
        (4, 7),
        (5, 7),
        (6, 7),
    )
    edges: list[tuple[np.ndarray, np.ndarray, bool]] = []
    for i, j in edge_pairs:
        is_back = farthest_idx in (i, j)
        edges.append((corners_2d[i], corners_2d[j], is_back))
    return edges


_CUBE_EDGES: tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...] = (
    ((0, 0, 0), (1, 0, 0)),
    ((0, 0, 0), (0, 1, 0)),
    ((0, 0, 0), (0, 0, 1)),
    ((1, 0, 0), (1, 1, 0)),
    ((1, 0, 0), (1, 0, 1)),
    ((0, 1, 0), (1, 1, 0)),
    ((0, 1, 0), (0, 1, 1)),
    ((0, 0, 1), (1, 0, 1)),
    ((0, 0, 1), (0, 1, 1)),
    ((1, 1, 0), (1, 1, 1)),
    ((1, 0, 1), (1, 1, 1)),
    ((0, 1, 1), (1, 1, 1)),
)


def _intersect_box_plane(
    normal_frac: np.ndarray, rhs: float
) -> list[np.ndarray]:
    """Intersects a plane ``normal_frac . x = rhs`` with the 12 cube edges."""
    pts_frac: list[np.ndarray] = []
    for p0_t, p1_t in _CUBE_EDGES:
        p0 = np.array(p0_t, dtype=np.float64)
        p1 = np.array(p1_t, dtype=np.float64)
        v0 = float(np.dot(normal_frac, p0) - rhs)
        v1 = float(np.dot(normal_frac, p1) - rhs)
        denom = v0 - v1
        if abs(v0) < 1e-9:
            pts_frac.append(p0)
        elif abs(v1) < 1e-9:
            pts_frac.append(p1)
        elif v0 * v1 < 0.0 and abs(denom) > 1e-12:
            pts_frac.append(p0 + (v0 / denom) * (p1 - p0))
    return pts_frac


def miller_plane_polygon(
    cell: np.ndarray,
    hkl: tuple[int, int, int],
) -> np.ndarray:
    """Intersects a crystallographic ``(hkl)`` plane with the unit-cell box.

    Uses the standard crystallographic origin-shift rule inside ``[0, 1]^3``
    fractional coordinates: negative indices shift the origin to the opposite
    cell corner (``o_i = 1`` where ``m_i < 0``), yielding the exact first
    lattice plane ``h*x + k*y + l*z = sum(min(0, m_i)) + 1``.

    Args:
      cell: Lattice matrix ``(3, 3)`` with rows as lattice vectors.
      hkl: Integer Miller indices ``(h, k, l)``, not all zero.

    Returns:
      Ordered polygon vertices in Cartesian Angstrom of shape ``(M, 3)``.
    """
    h, k, l_idx = (int(v) for v in hkl)
    if h == 0 and k == 0 and l_idx == 0:
        return np.zeros((0, 3), dtype=np.float64)
    normal_frac = np.array([float(h), float(k), float(l_idx)], dtype=np.float64)
    v_min = sum(min(0.0, float(val)) for val in (h, k, l_idx))
    rhs = v_min + 1.0
    pts_frac = _intersect_box_plane(normal_frac, rhs)
    unique: list[np.ndarray] = []
    for pt in pts_frac:
        if not any(np.linalg.norm(pt - existing) < 1e-6 for existing in unique):
            unique.append(pt)
    if len(unique) < 3:
        return np.zeros((0, 3), dtype=np.float64)
    arr_frac = np.vstack(unique)
    centroid = np.mean(arr_frac, axis=0)
    n_hat = normal_frac / np.linalg.norm(normal_frac)
    ref = arr_frac[0] - centroid
    ref /= max(float(np.linalg.norm(ref)), 1e-12)
    ortho = np.cross(n_hat, ref)
    angles = [
        math.atan2(
            float(np.dot(pt - centroid, ortho)),
            float(np.dot(pt - centroid, ref)),
        )
        for pt in arr_frac
    ]
    order = np.argsort(angles)
    return arr_frac[order] @ cell
