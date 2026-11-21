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

    azimuth: float = 35.0
    elevation: float = 20.0
    roll: float = 0.0
    perspective: float = 0.0
    zoom: float = 1.0


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
    cos_al = math.cos(al)
    cos_be = math.cos(be)
    cos_ga = math.cos(ga)
    sin_ga = math.sin(ga)
    va = np.array([a, 0.0, 0.0], dtype=np.float64)
    vb = np.array([b * cos_ga, b * sin_ga, 0.0], dtype=np.float64)
    cx = c * cos_be
    cy = c * (cos_al - cos_be * cos_ga) / max(sin_ga, 1e-12)
    cz_sq = max(0.0, c * c - cx * cx - cy * cy)
    vc = np.array([cx, cy, math.sqrt(cz_sq)], dtype=np.float64)
    return np.vstack([va, vb, vc])


def minimum_image_vector(
    vec: np.ndarray,
    cell: np.ndarray,
    pbc: tuple[bool, bool, bool],
) -> np.ndarray:
    """Applies minimum-image wrapping to displacement vector(s) ``vec``.

    Args:
      vec: Displacement vector ``(3,)`` or array ``(M, 3)``.
      cell: Row-vector lattice matrix ``(3, 3)``.
      pbc: Periodic boundary flags along each lattice vector.

    Returns:
      Wrapped displacement vector(s) with the same shape as ``vec``.
    """
    arr = np.asarray(vec, dtype=np.float64)
    inv_cell = np.linalg.inv(cell)
    frac = arr @ inv_cell
    for axis, is_periodic in enumerate(pbc):
        if is_periodic:
            frac[..., axis] -= np.round(frac[..., axis])
    return frac @ cell


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

    Uses the normalized positive-intercept plane inside ``[0, 1]^3`` fractional
    coordinates so negative indices shift cleanly inside the unit box.

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
    shift = sum(-val for val in normal_frac if val < 0.0)
    if h + k + l_idx <= 0 and shift == 0.0:
        rhs = 0.5
    else:
        rhs = max(1.0, shift + 0.5) if shift > 0.0 else 1.0
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
