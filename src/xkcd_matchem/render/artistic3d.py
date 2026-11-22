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
"""Pure-NumPy pen-and-ink comic/xkcd 3D surface shading helpers."""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import math

import numpy as np

from xkcd_matchem.core import electronic
from xkcd_matchem.core import exceptions
from xkcd_matchem.core import geometry
from xkcd_matchem.core import isosurface
from xkcd_matchem.render import _artistic3d_raster as raster
from xkcd_matchem.render import _artistic3d_strokes as strokes


__all__ = [
    "LineBatch",
    "MeshLayerInput",
    "ShadeArtData",
    "VALID_ISO_SHADES",
    "auto_orbital_camera",
    "build_isosurface_art",
    "extract_silhouette_polylines",
    "iso_scene_frame",
    "normalize_iso_shade",
    "project_lobe_layers",
]

_SHADE_ALIASES: dict[str, str] = {
    "solid": "solid",
    "smooth": "solid",
    "blending": "solid",
    "hatch": "hatch",
    "hatching": "hatch",
    "diagonal": "hatch",
    "crosshatch": "crosshatch",
    "cross_hatching": "crosshatch",
    "contour": "contour",
    "contour_hatch": "contour",
    "cross_contour": "cross_contour",
    "stipple": "stipple",
    "stippling": "stipple",
    "dots": "stipple",
    "scumble": "scumble",
    "scribble": "scumble",
    "circulism": "scumble",
    "outline": "outline",
    "none": "outline",
}

VALID_ISO_SHADES: tuple[str, ...] = (
    "solid",
    "hatch",
    "crosshatch",
    "contour",
    "cross_contour",
    "stipple",
    "scumble",
    "outline",
)

LineBatch = strokes.LineBatch
MeshLayerInput = raster.MeshLayerInput


@dataclasses.dataclass(frozen=True)
class ShadeArtData:
    """Assembled seam-free RGBA washes, pen strokes, stipples, and bounds."""

    back_rgba: np.ndarray | None
    front_rgba: np.ndarray | None
    wash_extent: tuple[float, float, float, float] | None
    line_batches: list[LineBatch]
    dot_batches: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]]
    all_xy: np.ndarray | None


def normalize_iso_shade(shade: str) -> str:
    """Validates and normalizes a 3D isosurface shading mode name."""
    key = shade.strip().lower().replace("-", "_")
    if key not in _SHADE_ALIASES:
        valid = ", ".join(VALID_ISO_SHADES)
        raise exceptions.StyleError(
            f"Unknown isosurface shade {shade!r}. Valid options: {valid}."
        )
    return _SHADE_ALIASES[key]


def auto_orbital_camera(
    meshes: Sequence[tuple[isosurface.IsosurfaceMesh, str]],
    grid: electronic.VolumetricGrid,
) -> geometry.Camera:
    """Orients orbital dumbbells upright with atoms clearly separated."""
    if (
        grid.atoms is None
        or len(grid.atoms) < 2
        or grid.atoms.is_periodic
    ):
        return geometry.Camera()

    pivot = np.mean(grid.atoms.positions, axis=0)
    pts = [m.vertices for m, _ in meshes if len(m.vertices)]
    if pts:
        all_pts = np.vstack(pts)
        first_pts = pts[0]
    else:
        vals = np.abs(grid.data)
        max_abs = max(float(np.max(vals)), 1e-12)
        idx = np.argwhere(vals >= 0.35 * max_abs)
        if len(idx) < 4:
            return geometry.Camera()
        all_pts = grid.origin[None, :] + idx @ grid.voxel_vectors
        pos_mask = grid.data[idx[:, 0], idx[:, 1], idx[:, 2]] >= 0.0
        first_pts = all_pts[pos_mask] if np.any(pos_mask) else all_pts

    _, _, vt_orb = np.linalg.svd(all_pts - pivot, full_matrices=False)
    u_orb = vt_orb[0].copy()
    if float(np.dot(np.mean(first_pts, axis=0) - pivot, u_orb)) < 0.0:
        u_orb = -u_orb

    _, _, vt_mol = np.linalg.svd(
        grid.atoms.positions - pivot, full_matrices=False
    )
    u_mol = vt_mol[0]
    right_0 = u_mol - float(np.dot(u_mol, u_orb)) * u_orb
    if float(np.linalg.norm(right_0)) < 1e-4:
        right_0 = vt_orb[1]
    right_0 /= max(float(np.linalg.norm(right_0)), 1e-12)
    fwd_0 = np.cross(right_0, u_orb)
    fwd_0 /= max(float(np.linalg.norm(fwd_0)), 1e-12)

    fwd = 0.80 * fwd_0 - 0.30 * right_0 - 0.44 * u_orb
    fwd /= max(float(np.linalg.norm(fwd)), 1e-12)
    el = math.degrees(math.asin(float(np.clip(fwd[2], -1.0, 1.0))))
    az = math.degrees(math.atan2(float(fwd[1]), float(fwd[0])))
    b0 = geometry.rotation_matrix(az, el, 0.0)
    roll = math.degrees(
        math.atan2(-float(np.dot(b0[0], u_orb)), float(np.dot(b0[1], u_orb)))
    )
    return geometry.Camera(azimuth=az, elevation=el, roll=roll)


def extract_silhouette_polylines(
    mesh: isosurface.IsosurfaceMesh,
    xy_v: np.ndarray,
    front_mask: np.ndarray,
) -> list[np.ndarray]:
    """Finds smoothed 2D horizon silhouette loops separating front and back."""
    xy_paths, _ = strokes.extract_silhouette_3d(
        mesh, xy_v, np.zeros(len(xy_v), dtype=np.float64), front_mask
    )
    return xy_paths


def iso_scene_frame(
    meshes: Sequence[tuple[isosurface.IsosurfaceMesh, str]],
    grid: electronic.VolumetricGrid,
    cam: geometry.Camera,
) -> tuple[np.ndarray, float, np.ndarray | None, np.ndarray | None]:
    """Resolves common pivot, projection span, and projected atom depths."""
    if grid.atoms is not None and len(grid.atoms) > 0:
        pivot = (
            0.5 * (grid.atoms.cell[0] + grid.atoms.cell[1] + grid.atoms.cell[2])
            if grid.atoms.cell is not None
            else np.mean(grid.atoms.positions, axis=0)
        )
        span = geometry.scene_projection_span(
            grid.atoms.positions, cam, center=pivot, cell=grid.atoms.cell
        )
        atom_xy, atom_depth = geometry.project_points(
            grid.atoms.positions, cam, center=pivot, span=span
        )
        return pivot, span, atom_xy, atom_depth

    ext = np.asarray(grid.data.shape, dtype=np.float64)[:, None]
    pivot = grid.origin + 0.5 * np.sum(ext * grid.voxel_vectors, axis=0)
    pts = [mesh.vertices for mesh, _ in meshes if len(mesh.vertices)]
    box_pts = np.vstack(pts) if pts else np.zeros((0, 3), dtype=np.float64)
    span = (
        geometry.scene_projection_span(box_pts, cam, center=pivot)
        if len(box_pts)
        else 5.0
    )
    return pivot, span, None, None


def _compute_lobe_axes(
    mesh: isosurface.IsosurfaceMesh,
    grid: electronic.VolumetricGrid,
    pivot: np.ndarray,
    basis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Computes world-locked lobe axis and orthogonal meridian in cam space."""
    lobe_mean_w = np.mean(mesh.vertices, axis=0)
    u_ax_w: np.ndarray | None = None
    if grid.atoms is not None and len(grid.atoms) > 0:
        dists = np.linalg.norm(
            grid.atoms.positions - lobe_mean_w[None, :], axis=1
        )
        vec_w = lobe_mean_w - grid.atoms.positions[int(np.argmin(dists))]
        n_ax = float(np.linalg.norm(vec_w))
        if n_ax > 1e-3:
            u_ax_w = vec_w / n_ax
    if u_ax_w is None:
        _, _, vt = np.linalg.svd(
            mesh.vertices - lobe_mean_w, full_matrices=False
        )
        u_ax_w = vt[0] / max(float(np.linalg.norm(vt[0])), 1e-12)
        if float(np.dot(lobe_mean_w - pivot, u_ax_w)) < 0.0:
            u_ax_w = -u_ax_w

    ref_w = (
        np.linalg.svd(grid.atoms.positions - pivot, full_matrices=False)[2][0]
        if grid.atoms is not None and len(grid.atoms) >= 2
        else np.array([0.0, 0.0, 1.0], dtype=np.float64)
    )
    if abs(float(np.dot(ref_w, u_ax_w))) > 0.92:
        ref_w = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(ref_w, u_ax_w))) > 0.92:
        ref_w = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    mer_w = ref_w - float(np.dot(ref_w, u_ax_w)) * u_ax_w
    u_mer_w = mer_w / max(float(np.linalg.norm(mer_w)), 1e-12)
    return (
        np.asarray(basis @ u_ax_w, dtype=np.float64),
        np.asarray(basis @ u_mer_w, dtype=np.float64),
    )


def _project_single_lobe(
    mesh: isosurface.IsosurfaceMesh,
    hex_col: str,
    grid: electronic.VolumetricGrid,
    cam: geometry.Camera,
    frame: tuple[np.ndarray, float, np.ndarray | None, np.ndarray | None],
    basis: np.ndarray,
    light_pair: tuple[np.ndarray, np.ndarray],
) -> MeshLayerInput | None:
    """Projects one lobe, computes 3D tones & intrinsic axes, splits depth."""
    pivot, span, atom_xy, atom_depth = frame
    if len(mesh.faces) == 0:
        return None
    xy_v, depth_v = geometry.project_points(
        mesh.vertices, cam, center=pivot, span=span
    )
    n_rot = mesh.normals @ basis.T
    front_mask = n_rot[:, 2] >= -0.02
    sils, sil_z = strokes.extract_silhouette_3d(
        mesh, xy_v, depth_v, front_mask
    )
    if not np.any(front_mask):
        return None

    vis_faces = mesh.faces[front_mask]
    tris_2d = xy_v[vis_faces]
    deps = np.mean(depth_v[vis_faces], axis=1)
    tones = raster.compute_front_tones(
        n_rot[front_mask], light_pair[0], light_pair[1]
    )
    if atom_xy is not None and atom_depth is not None:
        tri_xy = np.mean(tris_2d, axis=1)
        nearest = np.argmin(
            np.sum((tri_xy[:, None, :] - atom_xy[None, :, :]) ** 2, axis=2),
            axis=1,
        )
        behind_mask = deps < (atom_depth[nearest] - 0.03)
    else:
        behind_mask = np.zeros(len(tris_2d), dtype=bool)
    verts_cam = (mesh.vertices - pivot) @ basis.T
    vn_rot = raster.smooth_vertex_normals(mesh) @ basis.T
    lobe_ax, mer_ax = _compute_lobe_axes(mesh, grid, pivot, basis)
    return MeshLayerInput(
        mesh=mesh,
        xy_v=xy_v,
        depth_v=depth_v,
        vn_rot=vn_rot,
        verts_cam=verts_cam,
        front_mask=front_mask,
        tris_2d=tris_2d,
        depths=deps,
        tones=tones,
        behind_mask=behind_mask,
        silhouettes=sils,
        sil_z=sil_z,
        lobe_axis=lobe_ax,
        meridian_axis=mer_ax,
        hex_col=hex_col,
    )


def project_lobe_layers(
    meshes: Sequence[tuple[isosurface.IsosurfaceMesh, str]],
    grid: electronic.VolumetricGrid,
    cam: geometry.Camera,
    shade_key: str,
) -> tuple[list[MeshLayerInput], float, np.ndarray | None, np.ndarray | None]:
    """Projects lobes, computes 3D tones, and splits front/back of atoms."""
    del shade_key
    frame = iso_scene_frame(meshes, grid, cam)
    basis = geometry.rotation_matrix(cam.azimuth, cam.elevation, cam.roll)
    light_pair = raster.light_vectors()
    layers: list[MeshLayerInput] = []
    for mesh, hex_col in meshes:
        layer = _project_single_lobe(
            mesh, hex_col, grid, cam, frame, basis, light_pair
        )
        if layer is not None:
            layers.append(layer)
    return layers, frame[1], frame[2], frame[3]


def build_isosurface_art(
    meshes: Sequence[tuple[isosurface.IsosurfaceMesh, str]],
    grid: electronic.VolumetricGrid,
    cam: geometry.Camera,
    alpha: float,
    shade: str = "solid",
) -> ShadeArtData:
    """Builds occluded 3D RGBA washes, light comic strokes, and silhouettes."""
    shade_key = normalize_iso_shade(shade)
    layers, span, atom_xy, atom_depth = project_lobe_layers(
        meshes, grid, cam, shade_key
    )
    if not layers:
        return ShadeArtData(None, None, None, [], [], None)

    all_xy = np.vstack([ly.xy_v for ly in layers])
    back_rgba, front_rgba, wash_extent, lobe_z_bufs, grid_spec = (
        raster.rasterize_wash_buffers(
            layers, all_xy, atom_xy, atom_depth, span, alpha, shade_key
        )
    )
    atom_radii = raster.atom_visual_radii(grid)
    line_batches: list[LineBatch] = []
    dot_batches: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
    for idx, ly in enumerate(layers):
        occ_ctx = raster.OcclusionContext(
            lobe_z_bufs=lobe_z_bufs,
            grid_spec=grid_spec,
            atom_xy=atom_xy,
            atom_depth=atom_depth,
            atom_radii=atom_radii,
            mean_z=float(np.mean(ly.depths)),
        )
        strokes.build_lobe_batches(
            ly,
            idx,
            occ_ctx,
            (shade_key, span, alpha),
            line_batches,
            dot_batches,
        )
    return ShadeArtData(
        back_rgba, front_rgba, wash_extent, line_batches, dot_batches, all_xy
    )
