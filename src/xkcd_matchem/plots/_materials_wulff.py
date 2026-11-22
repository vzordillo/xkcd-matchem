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
"""3D shaded equilibrium Wulff crystal shape projection and rendering."""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
import pathlib

from matplotlib import axes as maxes
from matplotlib import collections
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
import numpy as np

from xkcd_matchem.core import exceptions
from xkcd_matchem.core import geometry
from xkcd_matchem.core import structure
from xkcd_matchem.core import wulff
from xkcd_matchem.io import loader
from xkcd_matchem.plots import _materials_ternary as mat_tern
from xkcd_matchem.render import primitives
from xkcd_matchem.style import palettes
from xkcd_matchem.style import theme


def coerce_wulff_cell(
    source: (
        structure.AtomsData
        | np.ndarray
        | Sequence[Sequence[float]]
        | str
        | pathlib.Path
    ),
) -> np.ndarray:
    """Resolves a ``(3, 3)`` unit-cell matrix for Wulff construction."""
    if isinstance(source, (str, pathlib.Path)):
        loaded = loader.load(source)
        if not isinstance(loaded, structure.AtomsData) or loaded.cell is None:
            raise exceptions.StructureError(
                f"Structure {source!r} has no periodic unit cell."
            )
        return np.asarray(loaded.cell, dtype=np.float64)
    if isinstance(source, structure.AtomsData):
        if source.cell is None:
            raise exceptions.StructureError(
                "AtomsData has no periodic unit cell for Wulff construction."
            )
        return np.asarray(source.cell, dtype=np.float64)
    return np.asarray(source, dtype=np.float64)


def format_hkl_label(hkl: tuple[int, int, int]) -> str:
    """Formats crystallographic Miller indices cleanly onto facets."""
    parts = [str(int(v)) for v in hkl]
    if all(len(p) == 1 for p in parts):
        return f"({''.join(parts)})"
    return f"({','.join(parts)})"


def lambert_shade_rgb(
    hex_col: str,
    dot_light: float,
) -> tuple[float, float, float]:
    """Computes clamped RGB color modulated by directional Lambert shading."""
    r, g, b = mcolors.to_rgb(hex_col)
    factor = 0.62 + 0.38 * max(float(dot_light), 0.0)
    return (
        float(np.clip(r * factor, 0.0, 1.0)),
        float(np.clip(g * factor, 0.0, 1.0)),
        float(np.clip(b * factor, 0.0, 1.0)),
    )


def facet_clearance(xy_f: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Returns polygon 2D centroid, horizontal span, and min edge clearance."""
    ctr = np.mean(xy_f, axis=0)
    span_h = float(np.ptp(xy_f[:, 0]))
    if len(xy_f) < 2:
        return ctr, span_h, 0.0
    nxt = np.roll(xy_f, -1, axis=0)
    edges = nxt - xy_f
    e_len = np.linalg.norm(edges, axis=1)
    valid = e_len >= 1e-9
    if not np.any(valid):
        return ctr, span_h, 0.0
    cross = np.abs(
        edges[valid, 0] * (xy_f[valid, 1] - ctr[1])
        - (xy_f[valid, 0] - ctr[0]) * edges[valid, 1]
    )
    return ctr, span_h, float(np.min(cross / e_len[valid]))


def collect_wulff_facet_visuals(
    records: Sequence[tuple[float, wulff.WulffFacet, np.ndarray, np.ndarray]],
    series: Sequence[str],
    scene_w: float,
    light_dir: np.ndarray,
    show_labels: bool,
) -> tuple[
    list[np.ndarray],
    list[tuple[float, float, float]],
    list[tuple[np.ndarray, str, str, bool]],
]:
    """Shades sorted Wulff faces and filters collision-free facet labels."""
    family_colors: dict[tuple[int, int, int], str] = {}
    polys: list[np.ndarray] = []
    face_cols: list[tuple[float, float, float]] = []
    labels_meta: list[tuple[np.ndarray, str, str, bool]] = []
    labeled_fams: set[tuple[int, int, int]] = set()
    narrow_cands: dict[tuple[int, int, int], tuple[float, np.ndarray, str]] = (
        {}
    )
    for _, facet, xy_f, n_rot in records:
        s_vals = sorted(abs(x) for x in facet.hkl)
        fam_key = (s_vals[0], s_vals[1], s_vals[2])
        if fam_key not in family_colors:
            family_colors[fam_key] = series[
                (len(family_colors) + 1) % len(series)
            ]
        shaded = lambert_shade_rgb(
            family_colors[fam_key], float(np.dot(n_rot, light_dir))
        )
        polys.append(xy_f)
        face_cols.append(shaded)
        if not (show_labels and float(n_rot[2]) > 0.15):
            continue
        ctr, span_h, min_d = facet_clearance(xy_f)
        lbl = format_hkl_label(facet.hkl)
        fits = (
            span_h >= max(0.18 * scene_w, 0.044 * len(lbl) * scene_w)
            and min_d >= 0.085 * scene_w
        )
        if fits:
            labeled_fams.add(fam_key)
            labels_meta.append(
                (
                    ctr,
                    lbl,
                    primitives.contrast_text_color(mcolors.to_hex(shaded)),
                    False,
                )
            )
        elif fam_key not in narrow_cands or min_d > narrow_cands[fam_key][0]:
            narrow_cands[fam_key] = (min_d, ctr, lbl)
    for fam_key, (_, ctr, lbl) in narrow_cands.items():
        if fam_key not in labeled_fams:
            labels_meta.append((ctr, lbl, "", True))
    return polys, face_cols, labels_meta


def project_wulff_facets(
    shape: wulff.WulffShape,
    cam: geometry.Camera,
    series: Sequence[str],
    show_labels: bool,
) -> tuple[
    list[np.ndarray],
    list[tuple[float, float, float]],
    list[tuple[np.ndarray, str, str, bool]],
    tuple[float, float, float, float],
]:
    """Projects, depth-sorts, and shades Wulff facets back-to-front."""
    basis = geometry.rotation_matrix(cam.azimuth, cam.elevation, cam.roll)
    light_dir = np.array([-0.45, 0.55, 0.70], dtype=np.float64)
    light_dir /= float(np.linalg.norm(light_dir))
    zero_ctr = np.zeros(3, dtype=np.float64)
    span = geometry.scene_projection_span(
        shape.vertices, cam, center=zero_ctr
    )
    records = [
        (
            float(np.mean(dep_f)),
            facet,
            xy_f,
            facet.normal @ basis.T,
        )
        for facet in shape.facets
        for xy_f, dep_f in [
            geometry.project_points(
                facet.vertices, cam, center=zero_ctr, span=span
            )
        ]
    ]
    records.sort(key=lambda item: item[0])
    all_xy, _ = geometry.project_points(
        shape.vertices, cam, center=zero_ctr, span=span
    )
    x_min, y_min = np.min(all_xy, axis=0)
    x_max, y_max = np.max(all_xy, axis=0)
    scene_w = max(float(x_max - x_min), float(y_max - y_min), 1e-3)
    polys, face_cols, labels_meta = collect_wulff_facet_visuals(
        records, series, scene_w, light_dir, show_labels
    )
    pad = 0.24 * scene_w
    bounds = (
        float(x_min - pad),
        float(x_max + pad),
        float(y_min - pad),
        float(y_max + pad),
    )
    return polys, face_cols, labels_meta, bounds


def draw_wulff_labels(
    ax: maxes.Axes,
    labels_meta: Sequence[tuple[np.ndarray, str, str, bool]],
    ink_col: str,
    bg_color: str,
    sw: theme.StrokeHierarchy,
    bounds: tuple[float, float, float, float],
) -> None:
    """Places interior facet Miller indices and fallback leader tags."""
    step = 0.24 * max(bounds[1] - bounds[0], 1.0)
    for centroid, txt_lbl, txt_col, is_callout in labels_meta:
        if not is_callout:
            ax.text(
                float(centroid[0]),
                float(centroid[1]),
                txt_lbl,
                ha="center",
                va="center",
                fontsize=8.2,
                color=txt_col,
                zorder=5.0,
            )
            continue
        r_norm = max(float(np.linalg.norm(centroid)), 1e-6)
        target = centroid + (centroid / r_norm) * step
        ann = ax.annotate(
            txt_lbl,
            xy=(float(centroid[0]), float(centroid[1])),
            xytext=(float(target[0]), float(target[1])),
            ha="center",
            va="center",
            fontsize=8.0,
            color=ink_col,
            bbox=theme.label_halo_bbox(bg_color, pad=0.08, alpha=0.88),
            arrowprops={
                "arrowstyle": "->",
                "color": ink_col,
                "lw": sw.leader,
            },
            zorder=6.0,
        )
        patch = ann.get_bbox_patch()
        if patch is not None:
            patch.set_sketch_params(0.0, 0.0, 0.0)


def plot_wulff_shape(
    source: (
        structure.AtomsData
        | np.ndarray
        | Sequence[Sequence[float]]
        | str
        | pathlib.Path
    ),
    surface_energies: Mapping[tuple[int, int, int], float],
    *,
    symops: (
        Sequence[tuple[np.ndarray, np.ndarray]] | str | int | None
    ) = "cubic",
    camera: geometry.Camera | str | None = "isometric",
    ax: maxes.Axes | None = None,
    palette: str | Sequence[str] = "okabe_ito",
    show_labels: bool = True,
    alpha: float = 0.84,
    preset: str | theme.ThemePreset = "classic",
    title: str | None = None,
) -> maxes.Axes:
    """Renders an xkcd-style 3D shaded equilibrium Wulff crystal shape.

    Args:
      source: ``AtomsData``, ``(3, 3)`` cell matrix, file path, or sample.
      surface_energies: Mapping ``{(h, k, l): gamma}`` with ``gamma > 0``.
      symops: Crystal symmetry expansion mode passed to ``compute_wulff_shape``.
      camera: ``Camera`` object or preset string.
      ax: Optional target Matplotlib ``Axes``.
      palette: Series palette name or hex color list for Miller families.
      show_labels: Whether to label front-facing ``(hkl)`` facets.
      alpha: Facet polygon fill opacity.
      preset: Visual theme preset name or ``ThemePreset``.
      title: Optional heading above the Wulff crystal.

    Returns:
      Matplotlib ``Axes`` containing the projected Wulff polyhedron.
    """
    cell = coerce_wulff_cell(source)
    shape = wulff.compute_wulff_shape(cell, surface_energies, symops=symops)
    cam = geometry.resolve_camera(camera, cell=cell)
    preset_obj = theme.get_preset(preset)
    ink_col = preset_obj.ink_color
    bg_col = preset_obj.axes_facecolor
    sw = theme.stroke_weights(preset)
    raw_series = palettes.get_palette(palette)
    themed_series = [
        mat_tern.theme_series_color(col, bg_col, ink_col) for col in raw_series
    ]
    polys, face_cols, labels_meta, bounds = project_wulff_facets(
        shape, cam, themed_series, show_labels
    )

    with theme.xkcd_style(preset=preset):
        if ax is None:
            _, ax = plt.subplots(figsize=(5.6, 5.2))
        else:
            theme.style_existing_axes(ax, preset=preset)
        poly_coll = collections.PolyCollection(
            polys,
            facecolors=face_cols,
            edgecolors=ink_col,
            linewidths=sw.primary,
            alpha=alpha,
            joinstyle="round",
            zorder=3.0,
        )
        ax.add_collection(poly_coll)
        draw_wulff_labels(ax, labels_meta, ink_col, bg_col, sw, bounds)
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[2], bounds[3])
        ax.set_aspect("equal")
        ax.set_axis_off()
        if title:
            ax.set_title(title, color=ink_col)
    return ax
