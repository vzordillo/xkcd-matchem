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
"""Interactive 3D xkcd-style studio widget, browser launcher, and HTML."""

from __future__ import annotations

import base64
from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import html
import importlib.resources
import json
import pathlib
import re
import tempfile
from typing import Any
import webbrowser

from matplotlib import axes as maxes
import numpy as np

from xkcd_matchem.core import bonding
from xkcd_matchem.core import electronic
from xkcd_matchem.core import exceptions
from xkcd_matchem.core import geometry
from xkcd_matchem.core import isosurface as isosurf_mod
from xkcd_matchem.core import structure
from xkcd_matchem.core import trajectory
from xkcd_matchem.data import elements
from xkcd_matchem.io import loader
from xkcd_matchem.plots import electronic as elec_plots
from xkcd_matchem.plots import structure as struct_plots
from xkcd_matchem.render import artistic3d
from xkcd_matchem.style import fonts
from xkcd_matchem.style import palettes
from xkcd_matchem.style import theme


_LARGE_SYSTEM_ATOM_THRESHOLD = 400
_AUTO_LOD_ATOM_THRESHOLD = 500
_ORDER_INFERENCE_ATOM_CAP = 500
_MAX_ISO_GRID_DIM = 24
_DEFAULT_ISO_COLORS: tuple[str, str] = ("#0072B2", "#D55E00")
_DEFAULT_ISO_ALPHA = 0.28
_DEFAULT_BOND_SCALE = 1.20
_SNIPPET_EXTRA_KEYS = frozenset(
    {
        "animate",
        "boundary_atoms",
        "center_in_cell",
        "custom_colors",
        "fps",
        "polyhedra_spec",
        "show_atoms",
        "show_iso",
        "show_legend",
        "show_vectors",
        "supercell",
        "symmetric_iso",
        "vector_scale",
        "vectors",
    }
)


@dataclasses.dataclass(frozen=True)
class InteractiveView:
    """Self-contained interactive HTML view embeddable in Jupyter or browsers.

    Attributes:
      html_str: Standalone HTML document encoding the 3D xkcd viewer.
      title: Human-readable structure or studio title.
      height: Default iframe height in pixels inside notebooks.
      camera: Active ``Camera`` viewpoint associated with this view.
    """

    html_str: str
    title: str = "xkcd-matchem 3D Studio"
    height: int = 680
    camera: geometry.Camera = dataclasses.field(
        default_factory=geometry.Camera
    )

    def to_camera(self) -> geometry.Camera:
        """Returns the ``Camera`` configuration associated with this view."""
        return self.camera

    def _repr_html_(self) -> str:
        """Returns Jupyter-compatible sandboxed iframe HTML."""
        escaped = html.escape(self.html_str, quote=True)
        return (
            f'<iframe srcdoc="{escaped}" '
            f'title="{html.escape(self.title, quote=True)}" '
            f'style="width:100%;height:{int(self.height)}px;'
            'border:1px solid #d5dae4;border-radius:10px;background:#ffffff;" '
            'loading="lazy"></iframe>'
        )

    def save(self, path: str | pathlib.Path) -> pathlib.Path:
        """Writes the standalone interactive HTML viewer to disk."""
        out = pathlib.Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.html_str, encoding="utf-8")
        return out

    def open_in_browser(
        self, path: str | pathlib.Path | None = None
    ) -> pathlib.Path:
        """Writes the interactive viewer to HTML and opens it in a browser."""
        if path is not None:
            out_path = self.save(path)
        else:
            with tempfile.NamedTemporaryFile(
                suffix=".html",
                prefix="xkcd_matchem_",
                delete=False,
            ) as tmp:
                out_path = pathlib.Path(tmp.name)
            out_path.write_text(self.html_str, encoding="utf-8")
        webbrowser.open(out_path.resolve().as_uri())
        return out_path


def _coerce_interactive_input(
    source: (
        structure.AtomsData
        | trajectory.Trajectory
        | electronic.VolumetricGrid
        | str
        | pathlib.Path
    ),
) -> tuple[
    structure.AtomsData,
    list[structure.AtomsData] | None,
    electronic.VolumetricGrid | None,
    str | None,
]:
    """Coerces structure, trajectory, cube grid, or sample into render data."""
    if isinstance(source, (str, pathlib.Path)):
        loaded = loader.load(source)
        src_name: str | None = str(source)
    else:
        loaded = source
        src_name = None

    if isinstance(loaded, trajectory.Trajectory):
        if len(loaded) == 0:
            raise exceptions.StructureError("Trajectory contains zero frames.")
        if src_name is None:
            meta_src = loaded[0].metadata.get("source")
            if meta_src:
                src_name = pathlib.Path(str(meta_src)).name
        return loaded[0], list(loaded.frames), None, src_name
    if isinstance(loaded, electronic.VolumetricGrid):
        if loaded.atoms is None or len(loaded.atoms) == 0:
            raise exceptions.StructureError(
                "VolumetricGrid requires associated atoms for 3D Studio."
            )
        return loaded.atoms, None, loaded, src_name
    if isinstance(loaded, structure.AtomsData):
        if src_name is None:
            meta_src = loaded.metadata.get("source")
            if meta_src:
                src_name = pathlib.Path(str(meta_src)).name
        return loaded, None, None, src_name
    raise exceptions.StructureError(
        f"Expected AtomsData, Trajectory, or VolumetricGrid, got "
        f"{type(loaded).__name__}."
    )


def _resolve_js_vectors(
    atoms_obj: structure.AtomsData,
    vectors: str | np.ndarray | Sequence[Sequence[float]] | None,
    vector_scale: float,
) -> list[list[float]] | None:
    """Resolves per-atom 3D arrows into serializable float triplets."""
    if vectors is None:
        return None
    n_at = len(atoms_obj)
    raw: np.ndarray | None = None
    if isinstance(vectors, str):
        key = vectors.strip().lower()
        if key == "forces":
            raw = atoms_obj.forces
        elif key == "velocities":
            raw = atoms_obj.velocities
        elif key in ("magmom", "magnetic_moments", "spin"):
            mag = atoms_obj.magnetic_moments
            if mag is not None and mag.ndim == 1 and len(mag) == n_at:
                raw = np.column_stack(
                    [np.zeros(n_at, dtype=np.float64), np.zeros(n_at), mag]
                )
            else:
                raw = mag
        else:
            raise exceptions.StructureError(
                f"Unknown vector attribute {vectors!r}."
            )
        if raw is None:
            return None
    else:
        raw = np.asarray(vectors, dtype=np.float64)
    if raw.shape != (n_at, 3):
        raise exceptions.StructureError(
            f"Vectors shape {raw.shape} must match ({n_at}, 3)."
        )
    scaled = np.round(raw * float(vector_scale), 4)
    return scaled.tolist()  # type: ignore[no-any-return]


def _serialize_compact_iso_grid(
    grid: electronic.VolumetricGrid,
) -> dict[str, Any]:
    """Serializes a compact volumetric field for live browser isovalues."""
    nx, ny, nz = grid.data.shape
    sx = max(1, (nx + _MAX_ISO_GRID_DIM - 1) // _MAX_ISO_GRID_DIM)
    sy = max(1, (ny + _MAX_ISO_GRID_DIM - 1) // _MAX_ISO_GRID_DIM)
    sz = max(1, (nz + _MAX_ISO_GRID_DIM - 1) // _MAX_ISO_GRID_DIM)
    sub = grid.data[::sx, ::sy, ::sz]
    vvecs = grid.voxel_vectors.copy()
    vvecs[0] *= sx
    vvecs[1] *= sy
    vvecs[2] *= sz
    return {
        "origin": np.round(grid.origin, 5).tolist(),
        "voxel_vectors": np.round(vvecs, 5).tolist(),
        "shape": [int(sub.shape[0]), int(sub.shape[1]), int(sub.shape[2])],
        "data": np.round(sub.ravel(), 5).tolist(),
    }


def _serialize_isosurface_meshes(
    iso_source: (
        electronic.VolumetricGrid
        | isosurf_mod.IsosurfaceMesh
        | str
        | pathlib.Path
        | None
    ),
    isolevel: float | None,
    symmetric_iso: bool | None,
    colors: tuple[str, str] = _DEFAULT_ISO_COLORS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Extracts and serializes 3D isosurface triangles and metadata."""
    if iso_source is None:
        return [], {}
    pos_col, neg_col = str(colors[0]), str(colors[1])
    meshes_and_cols: list[tuple[isosurf_mod.IsosurfaceMesh, str, int]] = []
    meta: dict[str, Any] = {}
    if isinstance(iso_source, isosurf_mod.IsosurfaceMesh):
        meshes_and_cols.append((iso_source, pos_col, 1))
        meta["iso_level"] = round(float(abs(iso_source.level)), 5)
        meta["iso_max"] = round(max(float(abs(iso_source.level)) * 2.5, 0.1), 5)
        meta["iso_symmetric"] = False
    else:
        if isinstance(iso_source, (str, pathlib.Path)):
            meta["iso_source"] = str(iso_source)
        raw_grid = (
            iso_source
            if isinstance(iso_source, electronic.VolumetricGrid)
            else loader.load(iso_source)
        )
        if not isinstance(raw_grid, electronic.VolumetricGrid):
            raise exceptions.StructureError(
                f"Expected VolumetricGrid, got {type(raw_grid).__name__}."
            )
        grid = isosurf_mod.upsample_coarse_grid(raw_grid, min_dim=22)
        max_abs = max(float(np.max(np.abs(grid.data))), 1e-12)
        level = 0.35 * max_abs if isolevel is None else float(isolevel)
        sym_flag = (
            bool(float(np.min(grid.data)) < -0.05 * max_abs)
            if symmetric_iso is None
            else bool(symmetric_iso)
        )
        meshes_and_cols.append(
            (isosurf_mod.extract_isosurface(grid, abs(level)), pos_col, 1)
        )
        if sym_flag:
            meshes_and_cols.append(
                (isosurf_mod.extract_isosurface(grid, -abs(level)), neg_col, -1)
            )
        meta["iso_level"] = round(abs(level), 5)
        meta["iso_max"] = round(max_abs, 5)
        meta["iso_symmetric"] = sym_flag
        meta["iso_grid"] = _serialize_compact_iso_grid(grid)

    rows: list[dict[str, Any]] = []
    for mesh, hex_col, sign_val in meshes_and_cols:
        if len(mesh.faces) == 0:
            continue
        rows.append(
            {
                "color": hex_col,
                "sign": sign_val,
                "vertices": np.round(mesh.vertices, 4).tolist(),
                "faces": mesh.faces.tolist(),
                "normals": np.round(mesh.normals, 3).tolist(),
            }
        )
    if not rows:
        return [], {}
    meta["iso_colors"] = [pos_col, neg_col]
    meta["iso_alpha"] = _DEFAULT_ISO_ALPHA
    return rows, meta


def _serialize_bond_rows(
    atoms_obj: structure.AtomsData,
    bond_scale: float = _DEFAULT_BOND_SCALE,
) -> list[list[Any]]:
    """Extracts or infers bond index-order-shift quadruples cleanly."""
    if atoms_obj.bonds and abs(float(bond_scale) - _DEFAULT_BOND_SCALE) < 1e-6:
        bonds = list(atoms_obj.bonds)
    else:
        bonds = bonding.infer_bonds(
            atoms_obj,
            scale=float(bond_scale),
            use_pbc=atoms_obj.is_periodic,
            infer_orders=len(atoms_obj) < _ORDER_INFERENCE_ATOM_CAP,
        )
    return [
        [
            int(b.i),
            int(b.j),
            float(b.order),
            [int(b.shift[0]), int(b.shift[1]), int(b.shift[2])],
        ]
        for b in bonds
    ]


def _serialize_trajectory_frames(
    frames_list: Sequence[structure.AtomsData],
    vectors: str | None,
    vector_scale: float,
) -> list[dict[str, Any]]:
    """Serializes trajectory frame coordinates, cells, energies, and bonds."""
    out: list[dict[str, Any]] = []
    for fr in frames_list:
        entry: dict[str, Any] = {
            "positions": np.round(fr.positions, 5).tolist(),
            "cell": (
                np.round(fr.cell, 6).tolist() if fr.cell is not None else None
            ),
            "bonds": _serialize_bond_rows(fr),
        }
        if "total" in fr.energies:
            entry["energy"] = round(float(fr.energies["total"]), 5)
        vec_rows = _resolve_js_vectors(fr, vectors, vector_scale)
        if vec_rows is not None:
            entry["vectors"] = vec_rows
        out.append(entry)
    return out


def _serialize_polyhedra_rows(
    atoms_obj: structure.AtomsData,
    polyhedra: tuple[str, str | None]
    | Sequence[bonding.CoordinationPolyhedron]
    | None,
) -> list[dict[str, Any]]:
    """Extracts and serializes coordination polyhedra vertices and faces."""
    if polyhedra is None:
        return []
    if (
        isinstance(polyhedra, tuple)
        and len(polyhedra) == 2
        and isinstance(polyhedra[0], str)
    ):
        poly_list = bonding.find_coordination_polyhedra(
            atoms_obj,
            center_symbol=polyhedra[0],
            ligand_symbol=polyhedra[1],
        )
    else:
        poly_list = list(polyhedra)  # type: ignore[arg-type]
    return [
        {
            "center_symbol": poly.center_symbol,
            "vertices": np.round(poly.vertices_3d, 5).tolist(),
            "faces": [[int(a), int(b), int(c)] for a, b, c in poly.faces],
        }
        for poly in poly_list
    ]


def _resolve_custom_palette(
    atoms_obj: structure.AtomsData,
    palette: str | Mapping[str, str],
) -> tuple[str, dict[str, str]]:
    """Extracts palette name and explicit per-element hex color overrides."""
    custom_colors: dict[str, str] = {}
    if isinstance(palette, Mapping):
        for sym in dict.fromkeys(atoms_obj.symbols):
            custom_colors[sym] = palettes.get_element_color(
                sym, scheme=palette
            )
        return "xkcd", custom_colors
    palette_name = str(palette)
    known = palettes.list_element_palettes()
    for sym in dict.fromkeys(atoms_obj.symbols):
        col = palettes.get_element_color(sym, scheme=palette_name)
        base_col = palettes.get_element_color(sym, scheme="xkcd")
        if palette_name not in known and col != base_col:
            custom_colors[sym] = col
    return (palette_name if palette_name in known else "xkcd"), custom_colors


def _attach_studio_extras(
    out: dict[str, Any],
    atoms_obj: structure.AtomsData,
    frames_list: list[structure.AtomsData] | None,
    auto_iso: electronic.VolumetricGrid | None,
    *,
    vectors: str | np.ndarray | Sequence[Sequence[float]] | None,
    vector_scale: float,
    isosurface: (
        electronic.VolumetricGrid
        | isosurf_mod.IsosurfaceMesh
        | str
        | pathlib.Path
        | None
    ),
    isolevel: float | None,
    symmetric_iso: bool | None,
) -> None:
    """Attaches optional vector arrows, isosurfaces, and trajectory frames."""
    vec_rows = _resolve_js_vectors(atoms_obj, vectors, vector_scale)
    if vec_rows is not None:
        out["vectors"] = vec_rows
        if isinstance(vectors, str):
            out["vector_source"] = vectors.strip().lower()
        out["vector_scale"] = float(vector_scale)
    iso_rows, iso_meta = _serialize_isosurface_meshes(
        auto_iso if isosurface is None else isosurface, isolevel, symmetric_iso
    )
    if iso_rows:
        out["isosurfaces"] = iso_rows
        out.update(iso_meta)
    if frames_list is not None and len(frames_list) > 1:
        vec_attr = vectors if isinstance(vectors, str) else None
        out["frames"] = _serialize_trajectory_frames(
            frames_list, vec_attr, vector_scale
        )


def serialize_structure(
    atoms: (
        structure.AtomsData
        | trajectory.Trajectory
        | electronic.VolumetricGrid
        | str
        | pathlib.Path
    ),
    *,
    style: str = "ball_and_stick",
    palette: str | Mapping[str, str] = "xkcd",
    camera: geometry.Camera | str | None = None,
    show_cell: bool | str = True,
    background: str = palettes.PAPER,
    scale: float = 1.0,
    polyhedra: tuple[str, str | None]
    | Sequence[bonding.CoordinationPolyhedron]
    | None = None,
    name: str | None = None,
    vectors: str | np.ndarray | Sequence[Sequence[float]] | None = None,
    vector_scale: float = 1.0,
    isosurface: (
        electronic.VolumetricGrid
        | isosurf_mod.IsosurfaceMesh
        | str
        | pathlib.Path
        | None
    ) = None,
    isolevel: float | None = None,
    symmetric_iso: bool | None = None,
    iso_shade: str = "solid",
    category: str | None = None,
    description: str | None = None,
    preset: str = "classic",
    bond_scale: float = _DEFAULT_BOND_SCALE,
) -> dict[str, Any]:
    """Serializes an ``AtomsData``, ``Trajectory``, or cube grid for Studio."""
    atoms_obj, frames_list, auto_iso, source_sample = _coerce_interactive_input(
        atoms
    )
    if camera is None and auto_iso is not None:
        cam = artistic3d.auto_orbital_camera([], auto_iso)
    else:
        cam = geometry.resolve_camera(camera, cell=atoms_obj.cell)
    palette_name, custom_colors = _resolve_custom_palette(atoms_obj, palette)
    bond_rows = _serialize_bond_rows(atoms_obj, bond_scale=bond_scale)

    out: dict[str, Any] = {
        "name": name or atoms_obj.formula(reduce_formula=False),
        "source_sample": source_sample,
        "symbols": list(atoms_obj.symbols),
        "positions": np.round(atoms_obj.positions, 5).tolist(),
        "cell": (
            np.round(atoms_obj.cell, 6).tolist()
            if atoms_obj.cell is not None
            else None
        ),
        "pbc": [bool(flag) for flag in atoms_obj.pbc],
        "bonds": bond_rows,
        "polyhedra": _serialize_polyhedra_rows(atoms_obj, polyhedra),
        "style": style,
        "palette_name": palette_name,
        "custom_colors": custom_colors,
        "preset": theme.get_preset(preset).name,
        "background": background,
        "scale": float(scale),
        "bond_scale": float(bond_scale),
        "show_cell": show_cell,
        "iso_shade": artistic3d.normalize_iso_shade(iso_shade),
        "camera": {
            "azimuth": float(cam.azimuth),
            "elevation": float(cam.elevation),
            "roll": float(cam.roll),
            "perspective": float(cam.perspective),
            "zoom": float(cam.zoom),
        },
        "perf": {
            "atomCount": len(atoms_obj),
            "bondCount": len(bond_rows),
            "largeSystem": len(atoms_obj) >= _LARGE_SYSTEM_ATOM_THRESHOLD,
            "autoLod": len(atoms_obj) >= _AUTO_LOD_ATOM_THRESHOLD,
        },
    }
    if (
        isinstance(polyhedra, tuple)
        and len(polyhedra) == 2
        and isinstance(polyhedra[0], str)
    ):
        out["polyhedra_spec"] = [polyhedra[0], polyhedra[1]]
    if category is not None:
        out["category"] = category
    if description is not None:
        out["description"] = description
    _attach_studio_extras(
        out,
        atoms_obj,
        frames_list,
        auto_iso,
        vectors=vectors,
        vector_scale=vector_scale,
        isosurface=isosurface,
        isolevel=isolevel,
        symmetric_iso=symmetric_iso,
    )
    return out


def _py_str(value: str) -> str:
    """Formats a string as a double-quoted Python literal."""
    return json.dumps(str(value))


def _format_call(
    prefix: str,
    callee: str,
    args: Sequence[str],
    indent: str = "",
) -> str:
    """Formats a PEP-8 Python call wrapped to <= 80 cols when needed."""
    single = f"{indent}{prefix}{callee}({', '.join(args)})"
    if len(single) <= 78 and all("\n" not in arg for arg in args):
        return single
    inner_indent = f"{indent}    "
    joined = ",\n".join(f"{inner_indent}{arg}" for arg in args)
    return f"{indent}{prefix}{callee}(\n{joined},\n{indent})"


def _format_palette_arg(
    palette_name: str,
    custom_colors: Mapping[str, str],
    symbols: Sequence[str],
) -> str:
    """Formats either a named palette literal or an element color mapping."""
    if not custom_colors:
        return _py_str(palette_name)
    ordered: dict[str, str] = {}
    for sym in dict.fromkeys(symbols):
        ordered[sym] = str(
            custom_colors.get(
                sym, palettes.get_element_color(sym, scheme=palette_name)
            )
        )
    for sym, col in custom_colors.items():
        if sym not in ordered:
            ordered[sym] = str(col)
    items = [f"{_py_str(k)}: {_py_str(v)}" for k, v in ordered.items()]
    inline = "{" + ", ".join(items) + "}"
    if len(inline) <= 52:
        return inline
    body = ",\n        ".join(items)
    return "{\n        " + body + ",\n    }"


def _resolve_cell_arg(cell_val: bool | str) -> str | None:
    """Maps Studio cell mode onto ``show_cell`` keyword expression."""
    if cell_val is False:
        return "False"
    if cell_val is True:
        return None
    key = str(cell_val).strip().lower()
    if key in ("off", "none", "false"):
        return "False"
    if key == "box":
        return None
    return _py_str(key)


def _resolve_axes_arg(axes_val: bool | str | None) -> str | None:
    """Maps Studio crystal-axes mode onto ``crystal_axes`` expression."""
    if axes_val is None:
        return None
    if isinstance(axes_val, bool):
        return "True" if axes_val else "False"
    key = str(axes_val).strip().lower()
    if key == "edges":
        return None
    if key in ("off", "none", "false"):
        return "False"
    return _py_str(key)


def _extract_snippet_state(
    scene: Mapping[str, Any],
    overrides: Mapping[str, Any],
) -> dict[str, Any]:
    """Merges serialized scene defaults with browser UI state overrides."""
    cam_raw = scene.get("camera") or {}
    cam_ov = overrides.get("camera")
    if isinstance(cam_ov, geometry.Camera):
        cam = {
            "azimuth": cam_ov.azimuth,
            "elevation": cam_ov.elevation,
            "roll": cam_ov.roll,
            "perspective": cam_ov.perspective,
            "zoom": cam_ov.zoom,
        }
    elif isinstance(cam_ov, Mapping):
        cam = {**cam_raw, **cam_ov}
    else:
        cam = dict(cam_raw)

    preset_name = str(
        overrides.get("preset") or scene.get("preset") or "classic"
    )
    preset_obj = theme.get_preset(preset_name)
    pal_ov = overrides.get("palette")
    custom_ov = overrides.get("custom_colors")
    if isinstance(pal_ov, Mapping):
        palette_name = str(scene.get("palette_name") or "xkcd")
        custom_colors = dict(pal_ov)
    else:
        palette_name = str(pal_ov or scene.get("palette_name") or "xkcd")
        custom_colors = dict(
            custom_ov
            if custom_ov is not None
            else (scene.get("custom_colors") or {})
        )

    iso_colors_raw = overrides.get("iso_colors") or scene.get("iso_colors")
    iso_cols = (
        (str(iso_colors_raw[0]), str(iso_colors_raw[1]))
        if iso_colors_raw and len(iso_colors_raw) >= 2
        else _DEFAULT_ISO_COLORS
    )
    frames = scene.get("frames") or []
    has_iso_data = bool(scene.get("isosurfaces"))
    show_iso = bool(
        overrides["show_iso"]
        if overrides.get("show_iso") is not None
        else has_iso_data
    )
    raw_src = (
        overrides.get("source")
        or scene.get("source_sample")
        or scene.get("name")
        or "structure.cif"
    )
    return {
        "source": str(raw_src),
        "is_trajectory": len(frames) > 1
        or (bool(frames) and overrides.get("frame_index") is not None),
        "n_frames": max(len(frames), 1),
        "frame_index": int(overrides.get("frame_index") or 0),
        "animate": bool(overrides.get("animate", False)),
        "fps": int(overrides.get("fps") or 8),
        "has_iso": has_iso_data and show_iso,
        "iso_source": str(scene.get("iso_source") or raw_src),
        "isovalue": (
            overrides["isovalue"]
            if overrides.get("isovalue") is not None
            else scene.get("iso_level")
        ),
        "symmetric_iso": (
            overrides["symmetric_iso"]
            if overrides.get("symmetric_iso") is not None
            else scene.get("iso_symmetric")
        ),
        "iso_shade": artistic3d.normalize_iso_shade(
            str(overrides.get("iso_shade") or scene.get("iso_shade") or "solid")
        ),
        "iso_alpha": float(
            overrides["iso_alpha"]
            if overrides.get("iso_alpha") is not None
            else scene.get("iso_alpha", _DEFAULT_ISO_ALPHA)
        ),
        "iso_colors": iso_cols,
        "show_atoms": bool(overrides.get("show_atoms", True)),
        "is_periodic": scene.get("cell") is not None,
        "symbols": list(scene.get("symbols") or ()),
        "azimuth": float(cam.get("azimuth", -118.0)),
        "elevation": float(cam.get("elevation", 24.0)),
        "roll": float(cam.get("roll", 0.0)),
        "perspective": float(cam.get("perspective", 0.0)),
        "zoom": float(cam.get("zoom", 1.0)),
        "preset": preset_obj.name,
        "preset_bg": preset_obj.figure_facecolor.upper(),
        "preset_wiggle": float(preset_obj.sketch_scale),
        "background": str(
            overrides.get("background")
            or scene.get("background")
            or preset_obj.figure_facecolor
        ).upper(),
        "wiggle": float(
            overrides["wiggle"]
            if overrides.get("wiggle") is not None
            else preset_obj.sketch_scale
        ),
        "style": str(
            overrides.get("style") or scene.get("style") or "ball_and_stick"
        ),
        "palette_name": palette_name,
        "custom_colors": custom_colors,
        "atom_scale": float(
            overrides["atom_scale"]
            if overrides.get("atom_scale") is not None
            else scene.get("scale", 1.0)
        ),
        "bond_scale": float(
            overrides["bond_scale"]
            if overrides.get("bond_scale") is not None
            else scene.get("bond_scale", _DEFAULT_BOND_SCALE)
        ),
        "show_bonds": bool(overrides.get("show_bonds", True)),
        "show_hbonds": bool(overrides.get("show_hbonds", False)),
        "show_cell": (
            overrides["show_cell"]
            if overrides.get("show_cell") is not None
            else scene.get("show_cell", True)
        ),
        "crystal_axes": overrides.get(
            "crystal_axes", scene.get("crystal_axes")
        ),
        "show_labels": bool(overrides.get("show_labels", False)),
        "show_legend": bool(overrides.get("show_legend", True)),
        "boundary_atoms": bool(overrides.get("boundary_atoms", True)),
        "center_in_cell": bool(overrides.get("center_in_cell", False)),
        "supercell": tuple(overrides.get("supercell") or (1, 1, 1)),
        "show_vectors": bool(overrides.get("show_vectors", True)),
        "vector_source": (
            overrides.get("vectors")
            if overrides.get("vectors") is not None
            else scene.get("vector_source")
        ),
        "vector_scale": float(
            overrides["vector_scale"]
            if overrides.get("vector_scale") is not None
            else scene.get("vector_scale", 1.0)
        ),
        "polyhedra_spec": (
            overrides.get("polyhedra_spec") or scene.get("polyhedra_spec")
        ),
    }


def _build_isosurface_snippet_body(
    st: dict[str, Any],
    pal_expr: str,
    indent: str,
) -> tuple[list[str], str]:
    """Builds loading lines and ``xm.plot_isosurface`` call."""
    prep = [f"grid = xm.load({_py_str(st['iso_source'])})"]
    args = ["grid"]
    if st["isovalue"] is not None:
        args.append(f"level={float(st['isovalue']):.4g}")
    if st["symmetric_iso"] is not None:
        args.append(f"symmetric={'True' if st['symmetric_iso'] else 'False'}")
    args.append("camera=cam")
    c_pos, c_neg = st["iso_colors"]
    if (c_pos.upper(), c_neg.upper()) != _DEFAULT_ISO_COLORS:
        args.append(f"colors=({_py_str(c_pos)}, {_py_str(c_neg)})")
    if abs(st["iso_alpha"] - _DEFAULT_ISO_ALPHA) > 1e-3:
        args.append(f"alpha={st['iso_alpha']:.2f}")
    args.append(f"shade={_py_str(st['iso_shade'])}")
    if not st["show_atoms"]:
        args.append("show_atoms=False")
    if st["custom_colors"] or st["palette_name"] != "xkcd":
        args.append(f"palette={pal_expr}")
    if st["preset"] != "classic":
        args.append(f"preset={_py_str(st['preset'])}")
    return prep, _format_call("ax = ", "xm.plot_isosurface", args, indent)


def _append_periodic_cell_args(args: list[str], st: dict[str, Any]) -> None:
    """Appends non-default ``show_cell`` and ``crystal_axes`` arguments."""
    cell_arg = _resolve_cell_arg(st["show_cell"])
    if cell_arg is not None:
        args.append(f"show_cell={cell_arg}")
    axes_arg = _resolve_axes_arg(st["crystal_axes"])
    if axes_arg is not None:
        args.append(f"crystal_axes={axes_arg}")


def _append_vector_and_hbond_args(args: list[str], st: dict[str, Any]) -> None:
    """Appends vector arrow and hydrogen-bond keyword arguments."""
    if st["show_vectors"] and st["vector_source"]:
        args.append(f"vectors={_py_str(st['vector_source'])}")
        if abs(st["vector_scale"] - 1.0) > 1e-3:
            args.append(f"vector_scale={st['vector_scale']:.2f}")
    if st["show_hbonds"]:
        args.append("show_hbonds=True")


def _build_scaled_3d_snippet(
    prep: list[str],
    st: dict[str, Any],
    pal_expr: str,
    indent: str,
) -> tuple[list[str], str]:
    """Builds ``xm.render_structure_3d`` snippet when ``atom_scale != 1``."""
    sc = st["supercell"]
    has_sc = (int(sc[0]), int(sc[1]), int(sc[2])) != (1, 1, 1)
    if st["is_periodic"]:
        prep.append("atoms = atoms.wrap_positions()")
        if st["center_in_cell"]:
            prep.append("atoms = atoms.center_in_cell()")
        if has_sc:
            prep.append(f"atoms = atoms.repeat(({sc[0]}, {sc[1]}, {sc[2]}))")
        if st["boundary_atoms"] and st["style"] != "polyhedra":
            prep.append("atoms = atoms.with_boundary_atoms()")
    args = [
        "atoms",
        "camera=cam",
        f"style={_py_str(st['style'])}",
        f"palette={pal_expr}",
    ]
    if st["is_periodic"]:
        _append_periodic_cell_args(args, st)
    else:
        args.append("show_cell=False")
    if st["show_labels"]:
        args.append("show_labels=True")
    if not st["show_bonds"]:
        args.append("show_bonds=False")
    args.append(f"radius_scale={st['atom_scale']:.2f}")
    if abs(st["bond_scale"] - _DEFAULT_BOND_SCALE) > 1e-3:
        args.append(f"bond_scale={st['bond_scale']:.2f}")
    pspec = st["polyhedra_spec"]
    if st["style"] == "polyhedra" and pspec and len(pspec) == 2:
        lig = "None" if pspec[1] is None else _py_str(pspec[1])
        args.append(f"polyhedra=({_py_str(pspec[0])}, {lig})")
    _append_vector_and_hbond_args(args, st)
    return prep, _format_call("ax = ", "xm.render_structure_3d", args, indent)


def _build_coord_snippet(
    prep: list[str],
    st: dict[str, Any],
    pal_expr: str,
    indent: str,
) -> tuple[list[str], str]:
    """Builds ``xm.plot_coordination`` call for polyhedra crystal views."""
    sc = st["supercell"]
    has_sc = (int(sc[0]), int(sc[1]), int(sc[2])) != (1, 1, 1)
    pspec = st["polyhedra_spec"]
    args = ["atoms", _py_str(pspec[0])]
    if pspec[1] is not None:
        args.append(_py_str(pspec[1]))
    if has_sc:
        args.append(f"supercell=({sc[0]}, {sc[1]}, {sc[2]})")
    args.extend(["camera=cam", f"palette={pal_expr}"])
    if st["preset"] != "classic":
        args.append(f"preset={_py_str(st['preset'])}")
    _append_periodic_cell_args(args, st)
    if st["show_bonds"]:
        args.append("show_bonds=True")
    return prep, _format_call("ax = ", "xm.plot_coordination", args, indent)


def _can_use_plot_coordination(st: dict[str, Any]) -> bool:
    """Returns whether ``xm.plot_coordination`` matches the active state."""
    pspec = st["polyhedra_spec"]
    if not (st["is_periodic"] and st["style"] == "polyhedra"):
        return False
    if pspec is None or len(pspec) != 2:
        return False
    has_vec = bool(st["show_vectors"] and st["vector_source"])
    return not (
        st["show_labels"]
        or st["show_hbonds"]
        or st["center_in_cell"]
        or has_vec
    )


def _build_structure_snippet_body(
    st: dict[str, Any],
    pal_expr: str,
    indent: str,
) -> tuple[list[str], str]:
    """Builds loading/transform lines and molecule/crystal/3D plot call."""
    prep: list[str] = []
    if st["is_trajectory"]:
        f_idx = max(0, min(st["frame_index"], st["n_frames"] - 1))
        prep.append(f"traj = xm.load_trajectory({_py_str(st['source'])})")
        prep.append(f"atoms = traj[{f_idx}]")
    else:
        prep.append(f"atoms = xm.load({_py_str(st['source'])})")

    if abs(st["atom_scale"] - 1.0) > 1e-3:
        return _build_scaled_3d_snippet(prep, st, pal_expr, indent)

    if abs(st["bond_scale"] - _DEFAULT_BOND_SCALE) > 1e-3:
        pbc_lit = "True" if st["is_periodic"] else "False"
        prep.append(
            f"atoms.bonds = xm.infer_bonds("
            f"atoms, scale={st['bond_scale']:.2f}, use_pbc={pbc_lit})"
        )

    if _can_use_plot_coordination(st):
        return _build_coord_snippet(prep, st, pal_expr, indent)

    sc = st["supercell"]
    has_sc = (int(sc[0]), int(sc[1]), int(sc[2])) != (1, 1, 1)
    fn = "xm.plot_crystal" if st["is_periodic"] else "xm.plot_molecule"
    args = ["atoms"]
    if st["is_periodic"] and has_sc:
        args.append(f"supercell=({sc[0]}, {sc[1]}, {sc[2]})")
    args.extend(
        [
            f"style={_py_str(st['style'])}",
            "camera=cam",
            f"palette={pal_expr}",
        ]
    )
    if st["is_periodic"]:
        _append_periodic_cell_args(args, st)
    if st["show_labels"]:
        args.append("show_labels=True")
    if not st["show_legend"]:
        args.append("show_legend=False")
    if st["is_periodic"] and st["center_in_cell"]:
        args.append("center_in_cell=True")
    if st["preset"] != "classic":
        args.append(f"preset={_py_str(st['preset'])}")
    if not st["show_bonds"]:
        args.append("show_bonds=False")
    if st["is_periodic"] and not st["boundary_atoms"]:
        args.append("boundary_atoms=False")
    _append_vector_and_hbond_args(args, st)
    return prep, _format_call("ax = ", fn, args, indent)


def _assemble_snippet_from_state(st: dict[str, Any]) -> str:
    """Assembles complete Python snippet text from normalized Studio state."""
    pal_expr = _format_palette_arg(
        st["palette_name"], st["custom_colors"], st["symbols"]
    )
    needs_style_ctx = (
        st["background"] != st["preset_bg"]
        or abs(st["wiggle"] - st["preset_wiggle"]) > 0.01
        or (not st["has_iso"] and abs(st["atom_scale"] - 1.0) > 1e-3)
    )
    indent = "    " if needs_style_ctx else ""
    if st["has_iso"]:
        prep_lines, plot_block = _build_isosurface_snippet_body(
            st, pal_expr, indent
        )
    else:
        prep_lines, plot_block = _build_structure_snippet_body(
            st, pal_expr, indent
        )

    cam_call = _format_call(
        "cam = ",
        "xm.Camera",
        [
            f"azimuth={st['azimuth']:.1f}",
            f"elevation={st['elevation']:.1f}",
            f"roll={st['roll']:.1f}",
            f"perspective={st['perspective']:.2f}",
            f"zoom={st['zoom']:.2f}",
        ],
    )
    out_lines = ["import xkcd_matchem as xm", "", *prep_lines, cam_call]

    if needs_style_ctx:
        ctx_args = [_py_str(st["preset"])]
        if abs(st["wiggle"] - st["preset_wiggle"]) > 0.01:
            ctx_args.append(f"scale={st['wiggle']:.2f}")
        if st["background"] != st["preset_bg"]:
            ctx_args.append(f"background={_py_str(st['background'])}")
        out_lines.append(_format_call("with ", "xm.xkcd_style", ctx_args) + ":")
    out_lines.append(plot_block)

    if st["is_trajectory"] and st["animate"] and not st["has_iso"]:
        anim_args = [
            "traj",
            _py_str("trajectory.gif"),
            f"fps={st['fps']}",
            "camera=cam",
            f"palette={pal_expr}",
        ]
        if st["preset"] != "classic":
            anim_args.append(f"preset={_py_str(st['preset'])}")
        if st["is_periodic"]:
            cell_arg = _resolve_cell_arg(st["show_cell"])
            if cell_arg is not None:
                anim_args.append(f"show_cell={cell_arg}")
        if st["show_hbonds"]:
            anim_args.append("show_hbonds=True")
        out_lines.append(
            _format_call("gif_path = ", "xm.animate_trajectory", anim_args)
        )
    return "\n".join(out_lines) + "\n"


def generate_python_snippet(
    scene: (
        Mapping[str, Any]
        | structure.AtomsData
        | trajectory.Trajectory
        | electronic.VolumetricGrid
        | str
        | pathlib.Path
    ),
    *,
    source: str | None = None,
    camera: geometry.Camera | Mapping[str, float] | None = None,
    preset: str | None = None,
    background: str | None = None,
    wiggle: float | None = None,
    style: str | None = None,
    palette: str | Mapping[str, str] | None = None,
    atom_scale: float | None = None,
    bond_scale: float | None = None,
    show_bonds: bool | None = None,
    show_hbonds: bool | None = None,
    show_cell: bool | str | None = None,
    crystal_axes: bool | str | None = None,
    show_labels: bool | None = None,
    isovalue: float | None = None,
    iso_shade: str | None = None,
    iso_alpha: float | None = None,
    iso_colors: tuple[str, str] | None = None,
    frame_index: int | None = None,
    **extra_state: Any,
) -> str:
    """Builds runnable PEP-8 ``xkcd_matchem`` Python code for a Studio state.

    Mirrors ``buildPythonSnippet()`` in ``viewer.html`` 1:1 across molecules,
    periodic crystals, volumetric orbitals, and multi-frame trajectories using
    real supported parameters of ``xm.plot_molecule``, ``xm.plot_crystal``,
    ``xm.plot_coordination``, ``xm.render_structure_3d``,
    ``xm.plot_isosurface``, ``xm.animate_trajectory``, ``xm.Camera``, and
    ``xm.xkcd_style``.
    """
    unknown = sorted(set(extra_state) - _SNIPPET_EXTRA_KEYS)
    if unknown:
        raise TypeError(
            f"Unsupported generate_python_snippet state keys: {unknown!r}"
        )
    scene_dict = (
        dict(scene)
        if isinstance(scene, Mapping)
        else serialize_structure(scene)
    )
    return _assemble_snippet_from_state(
        _extract_snippet_state(
            scene_dict,
            {
                "source": source,
                "camera": camera,
                "preset": preset,
                "background": background,
                "wiggle": wiggle,
                "style": style,
                "palette": palette,
                "atom_scale": atom_scale,
                "bond_scale": bond_scale,
                "show_bonds": show_bonds,
                "show_hbonds": show_hbonds,
                "show_cell": show_cell,
                "crystal_axes": crystal_axes,
                "show_labels": show_labels,
                "isovalue": isovalue,
                "iso_shade": iso_shade,
                "iso_alpha": iso_alpha,
                "iso_colors": iso_colors,
                "frame_index": frame_index,
                **extra_state,
            },
        )
    )


def _safe_json_for_script(payload: Any) -> str:
    """Serializes ``payload`` to JSON safe for embedding in HTML ``<script>``.

    Escapes ``&``, ``<``, ``>``, U+2028, and U+2029 using JSON Unicode escapes
    so user-controlled titles, element labels, or metadata cannot break out of
    ``<script>`` (e.g. via ``</script>``) or inject HTML/XSS while remaining
    100% valid JSON for ``JSON.parse`` / ``json.loads``.
    """
    raw = json.dumps(payload, separators=(",", ":"))
    return (
        raw.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _safe_css_font_family(font_family: str) -> str:
    """Sanitizes a font family name for CSS, JS string, and SVG use."""
    cleaned = re.sub(r"[^A-Za-z0-9 _.-]", "", str(font_family)).strip()
    return cleaned or "Patrick Hand"


def _build_page_title(active_payload: Mapping[str, Any]) -> str:
    """Builds an HTML-escaped ``<title>`` string for the 3D Studio."""
    raw_name = str(active_payload.get("name") or "").strip()
    page_title = (
        f"{raw_name} - xkcd-matchem 3D Studio"
        if raw_name
        else "xkcd-matchem 3D Studio"
    )
    return html.escape(page_title, quote=True)


def _build_font_css(font_family: str = "Patrick Hand") -> str:
    """Embeds active bundled TTF font as base64 ``@font-face`` CSS."""
    safe_family = _safe_css_font_family(font_family)
    if safe_family not in fonts.list_bundled_fonts():
        return ""
    try:
        font_path = fonts.get_bundled_font_path(safe_family)
        encoded = base64.b64encode(font_path.read_bytes()).decode("ascii")
        return (
            "@font-face {\n"
            f"  font-family: '{safe_family}';\n"
            f"  src: url('data:font/ttf;base64,{encoded}') "
            "format('truetype');\n"
            "  font-display: swap;\n"
            "}\n"
        )
    except exceptions.StyleError:
        return ""


def _build_viewer_tables() -> (
    tuple[dict[str, Any], dict[str, dict[str, str]], dict[str, dict[str, Any]]]
):
    """Builds element, palette, and preset tables for offline JS switching."""
    elem_table: dict[str, dict[str, Any]] = {}
    pal_tables: dict[str, dict[str, str]] = {
        scheme: {} for scheme in palettes.list_element_palettes()
    }
    for rec in elements.ELEMENTS_BY_Z.values():
        sym = rec.symbol
        elem_table[sym] = {
            "cov": round(float(rec.covalent_radius), 4),
            "vdw": round(float(rec.vdw_radius), 4),
            "color": palettes.get_element_color(sym, scheme="xkcd"),
        }
        for scheme, table in pal_tables.items():
            col = palettes.get_element_color(sym, scheme=scheme)
            if col != elem_table[sym]["color"] or scheme == "xkcd":
                table[sym] = col
    presets_table: dict[str, dict[str, Any]] = {
        name: {
            "bg": p.figure_facecolor.upper(),
            "wiggle": round(float(p.sketch_scale), 2),
            "ink": p.ink_color,
        }
        for name, p in theme.PRESETS.items()
    }
    return elem_table, pal_tables, presets_table


def build_interactive_html(
    atoms: (
        structure.AtomsData
        | trajectory.Trajectory
        | electronic.VolumetricGrid
        | str
        | pathlib.Path
    ),
    *,
    samples: Sequence[dict[str, Any]] | None = None,
    style: str = "ball_and_stick",
    palette: str | Mapping[str, str] = "xkcd",
    camera: geometry.Camera | str | None = None,
    show_cell: bool | str = True,
    background: str = palettes.PAPER,
    scale: float = 1.0,
    polyhedra: tuple[str, str | None]
    | Sequence[bonding.CoordinationPolyhedron]
    | None = None,
    name: str | None = None,
    vectors: str | np.ndarray | Sequence[Sequence[float]] | None = None,
    vector_scale: float = 1.0,
    isosurface: (
        electronic.VolumetricGrid
        | isosurf_mod.IsosurfaceMesh
        | str
        | pathlib.Path
        | None
    ) = None,
    isolevel: float | None = None,
    symmetric_iso: bool | None = None,
    iso_shade: str = "solid",
    category: str | None = None,
    description: str | None = None,
    preset: str = "classic",
    bond_scale: float = _DEFAULT_BOND_SCALE,
) -> str:
    """Builds a standalone offline HTML document for the 3D xkcd Studio."""
    active_payload = serialize_structure(
        atoms,
        style=style,
        palette=palette,
        camera=camera,
        show_cell=show_cell,
        background=background,
        scale=scale,
        polyhedra=polyhedra,
        name=name,
        vectors=vectors,
        vector_scale=vector_scale,
        isosurface=isosurface,
        isolevel=isolevel,
        symmetric_iso=symmetric_iso,
        iso_shade=iso_shade,
        category=category,
        description=description,
        preset=preset,
        bond_scale=bond_scale,
    )
    elem_table, pal_tables, presets_table = _build_viewer_tables()
    payload = {
        "active": active_payload,
        "samples": list(samples) if samples is not None else [active_payload],
        "elements_table": elem_table,
        "palettes": pal_tables,
        "presets_table": presets_table,
    }
    template = (
        importlib.resources.files("xkcd_matchem.data")
        .joinpath("templates/viewer.html")
        .read_text(encoding="utf-8")
    )
    safe_font = _safe_css_font_family(
        fonts.get_default_font() or "Patrick Hand"
    )
    return (
        template.replace("__XM_TITLE__", _build_page_title(active_payload))
        .replace("__XM_FONT_CSS__", _build_font_css(safe_font))
        .replace("__XM_FONT_FAMILY__", html.escape(safe_font, quote=True))
        .replace("__XM_PAYLOAD__", _safe_json_for_script(payload))
    )


def _active_sample_filename(
    atoms: (
        structure.AtomsData
        | trajectory.Trajectory
        | electronic.VolumetricGrid
        | str
        | pathlib.Path
    ),
) -> str:
    """Returns sample filename if ``atoms`` originates from a known file."""
    if isinstance(atoms, (str, pathlib.Path)):
        return pathlib.Path(atoms).name
    src = atoms.metadata.get("source")
    return pathlib.Path(str(src)).name if src else ""


@dataclasses.dataclass(frozen=True)
class _CuratedSceneSpec:
    """Curated interactive 3D Studio sample preset entry."""

    filename: str
    title: str
    category: str
    description: str
    style: str = "ball_and_stick"
    polyhedra: tuple[str, str | None] | None = None
    camera: geometry.Camera | None = None
    iso_shade: str = "solid"
    is_trajectory: bool = False


_CURATED_PLAYGROUND_SPECS: tuple[_CuratedSceneSpec, ...] = (
    _CuratedSceneSpec(
        filename="water_homo.cube",
        title="Water HOMO 3D Orbital (water_homo.cube)",
        category="3D Orbitals",
        description=(
            "Cross-hatched 3D water HOMO orbital lobes from cube grid data."
        ),
        iso_shade="crosshatch",
    ),
    _CuratedSceneSpec(
        filename="tio2_slab.poscar",
        title="TiO2 (110) Octahedra (tio2_slab.poscar)",
        category="Crystals & Slabs",
        description=(
            "Rutile TiO2 (110) surface slab with TiO6 coordination octahedra."
        ),
        style="polyhedra",
        polyhedra=("Ti", "O"),
    ),
    _CuratedSceneSpec(
        filename="sn2_neb.xyz",
        title="SN2 Reaction Path (sn2_neb.xyz)",
        category="Reactions & MD",
        description=(
            "9-frame Cl- + CH3Cl SN2 minimum-energy trajectory with playback."
        ),
        is_trajectory=True,
    ),
    _CuratedSceneSpec(
        filename="aspirin.xyz",
        title="Aspirin Molecule (aspirin.xyz)",
        category="Molecules",
        description=(
            "Acetylsalicylic acid (C9H8O4) 3D ball-and-stick geometry."
        ),
        camera=geometry.Camera(azimuth=-125.0, elevation=42.0),
    ),
    _CuratedSceneSpec(
        filename="mos2_2h.poscar",
        title="2H-MoS2 Monolayer (mos2_2h.poscar)",
        category="Crystals & Slabs",
        description=(
            "Layered hexagonal 2H-MoS2 transition-metal dichalcogenide cell."
        ),
    ),
    _CuratedSceneSpec(
        filename="si_diamond.cif",
        title="Silicon Diamond Cell (si_diamond.cif)",
        category="Crystals & Slabs",
        description="Diamond-cubic silicon conventional unit cell (Fd-3m).",
    ),
    _CuratedSceneSpec(
        filename="nacl.cif",
        title="Rock-Salt NaCl (nacl.cif)",
        category="Crystals & Slabs",
        description="Ionic rock-salt NaCl face-centered cubic lattice (Fm-3m).",
    ),
    _CuratedSceneSpec(
        filename="benzene.sdf",
        title="Benzene Ring (benzene.sdf)",
        category="Molecules",
        description=(
            "Aromatic benzene (C6H6) with explicit alternating bond orders."
        ),
    ),
    _CuratedSceneSpec(
        filename="alaninedipeptide.pdb",
        title="Alanine Dipeptide (alaninedipeptide.pdb)",
        category="Biomolecules",
        description=(
            "Blocked alanine dipeptide backbone conformation with CONECT bonds."
        ),
    ),
)

_CURATED_SCENE_BY_FILE: dict[str, _CuratedSceneSpec] = {
    spec.filename: spec for spec in _CURATED_PLAYGROUND_SPECS
}


def build_playground_scenes(
    exclude_filenames: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Builds the curated multi-scene lineup for the interactive 3D Studio.

    Args:
      exclude_filenames: Optional sample filenames to omit (e.g., when
        deduplicating an explicit leading scene).

    Returns:
      Ordered list of serialized scene payloads with ``category`` and
      ``description`` metadata ready for ``viewer.html``.
    """
    excluded = {pathlib.Path(name).name for name in exclude_filenames if name}
    scenes: list[dict[str, Any]] = []
    for spec in _CURATED_PLAYGROUND_SPECS:
        if spec.filename in excluded:
            continue
        source: str | trajectory.Trajectory = (
            loader.load_trajectory(spec.filename)
            if spec.is_trajectory
            else spec.filename
        )
        scene = serialize_structure(
            source,
            style=spec.style,
            camera=spec.camera,
            polyhedra=spec.polyhedra,
            name=spec.title,
            iso_shade=spec.iso_shade,
            category=spec.category,
            description=spec.description,
        )
        scene["source_sample"] = spec.filename
        scenes.append(scene)
    return scenes


def export_interactive_html(
    atoms: (
        structure.AtomsData
        | trajectory.Trajectory
        | electronic.VolumetricGrid
        | str
        | pathlib.Path
    ),
    output_path: str | pathlib.Path,
    *,
    include_bundled_samples: bool = False,
    style: str = "ball_and_stick",
    palette: str | Mapping[str, str] = "xkcd",
    camera: geometry.Camera | str | None = None,
    show_cell: bool | str = True,
    background: str = palettes.PAPER,
    scale: float = 1.0,
    polyhedra: tuple[str, str | None]
    | Sequence[bonding.CoordinationPolyhedron]
    | None = None,
    name: str | None = None,
    vectors: str | np.ndarray | Sequence[Sequence[float]] | None = None,
    vector_scale: float = 1.0,
    isosurface: (
        electronic.VolumetricGrid
        | isosurf_mod.IsosurfaceMesh
        | str
        | pathlib.Path
        | None
    ) = None,
    isolevel: float | None = None,
    symmetric_iso: bool | None = None,
    iso_shade: str = "solid",
    preset: str = "classic",
    bond_scale: float = _DEFAULT_BOND_SCALE,
) -> pathlib.Path:
    """Exports a standalone offline interactive 3D studio HTML file."""
    active_fname = _active_sample_filename(atoms)
    curated = _CURATED_SCENE_BY_FILE.get(active_fname)
    lead_name = (
        name if name is not None else (curated.title if curated else None)
    )
    lead_cat = curated.category if curated else None
    lead_desc = curated.description if curated else None

    extra_samples: list[dict[str, Any]] | None = None
    if include_bundled_samples:
        first = serialize_structure(
            atoms,
            style=style,
            palette=palette,
            camera=camera,
            show_cell=show_cell,
            background=background,
            scale=scale,
            polyhedra=polyhedra,
            name=lead_name,
            vectors=vectors,
            vector_scale=vector_scale,
            isosurface=isosurface,
            isolevel=isolevel,
            symmetric_iso=symmetric_iso,
            iso_shade=iso_shade,
            category=lead_cat,
            description=lead_desc,
            preset=preset,
            bond_scale=bond_scale,
        )
        if active_fname and first.get("source_sample") is None:
            first["source_sample"] = active_fname
        extra_samples = [
            first,
            *build_playground_scenes(exclude_filenames=(active_fname,)),
        ]
    html_text = build_interactive_html(
        atoms,
        samples=extra_samples,
        style=style,
        palette=palette,
        camera=camera,
        show_cell=show_cell,
        background=background,
        scale=scale,
        polyhedra=polyhedra,
        name=lead_name,
        vectors=vectors,
        vector_scale=vector_scale,
        isosurface=isosurface,
        isolevel=isolevel,
        symmetric_iso=symmetric_iso,
        iso_shade=iso_shade,
        category=lead_cat,
        description=lead_desc,
        preset=preset,
        bond_scale=bond_scale,
    )
    out = pathlib.Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_text, encoding="utf-8")
    return out


def view_structure(
    atoms: (
        structure.AtomsData
        | trajectory.Trajectory
        | electronic.VolumetricGrid
        | str
        | pathlib.Path
    ),
    *,
    backend: str = "auto",
    output: str | pathlib.Path | None = None,
    style: str = "ball_and_stick",
    palette: str | Mapping[str, str] = "xkcd",
    preset: str = "classic",
    camera: geometry.Camera | str | None = None,
    supercell: tuple[int, int, int] | None = None,
    center: bool = False,
    show_cell: bool | str = True,
    background: str = palettes.PAPER,
    polyhedra: tuple[str, str | None] | None = None,
    vectors: str | np.ndarray | Sequence[Sequence[float]] | None = None,
    vector_scale: float = 1.0,
    isosurface: (
        electronic.VolumetricGrid
        | isosurf_mod.IsosurfaceMesh
        | str
        | pathlib.Path
        | None
    ) = None,
    isolevel: float | None = None,
    symmetric_iso: bool | None = None,
    iso_shade: str = "solid",
) -> InteractiveView | maxes.Axes:
    """Opens a molecule, crystal, trajectory, or cube grid in 3D Studio.

    Args:
      atoms: ``AtomsData``, ``Trajectory``, ``VolumetricGrid``, file path, or
        bundled sample filename.
      backend: One of ``"auto"``, ``"html"``, ``"browser"``, or
        ``"matplotlib"``.
      output: Optional path to write standalone ``.html`` output.
      style: Visual representation (``"ball_and_stick"``, ``"spacefill"``,
        ``"stick"``, ``"polyhedra"``).
      palette: Named element palette or custom ``{symbol: color}`` dictionary.
      preset: Theme preset name when rendering via ``"matplotlib"`` or Studio.
      camera: ``Camera`` instance or preset name (e.g. ``"isometric"``,
        ``"perspective"``, ``"top"``, ``"c*"``).
      supercell: Optional ``(na, nb, nc)`` replication applied before viewing.
      center: Whether to center periodic atoms inside the cell before viewing.
      show_cell: Unit-cell display flag/mode.
      background: Canvas background hex color string.
      polyhedra: Optional ``(center_symbol, ligand_symbol)`` pair for
        coordination polyhedra rendering.
      vectors: Optional per-atom vector attribute or ``(N, 3)`` array.
      vector_scale: Length multiplier for displayed vectors.
      isosurface: Optional ``VolumetricGrid``, ``IsosurfaceMesh``, or path.
      isolevel: Optional explicit isosurface threshold.
      symmetric_iso: Extract both ``+level`` and ``-level`` orbital lobes.
      iso_shade: 3D isosurface shading mode (``"solid"``, ``"crosshatch"``,
        ``"hatch"``, ``"contour"``, ``"cross_contour"``, ``"stipple"``,
        ``"scumble"``, or ``"outline"``).

    Returns:
      ``InteractiveView`` for HTML/browser backends, or Matplotlib ``Axes`` for
      ``backend="matplotlib"``.

    Raises:
      StructureError: If ``backend`` is invalid.
    """
    mode = backend.strip().lower()
    if mode not in ("auto", "html", "browser", "matplotlib"):
        raise exceptions.StructureError(
            f"Unsupported view_structure backend {backend!r}. "
            "Valid options: 'auto', 'html', 'browser', 'matplotlib'."
        )
    atoms_obj, frames_list, auto_iso, _ = _coerce_interactive_input(atoms)
    if center and atoms_obj.is_periodic:
        atoms_obj = atoms_obj.center_in_cell()
    if supercell is not None and supercell != (1, 1, 1):
        atoms_obj = atoms_obj.repeat(supercell)

    if mode == "matplotlib":
        iso_arg = isosurface if isosurface is not None else auto_iso
        if iso_arg is not None:
            if isinstance(iso_arg, isosurf_mod.IsosurfaceMesh):
                raise exceptions.StructureError(
                    "Matplotlib backend expects VolumetricGrid/path isosurface."
                )
            iso_pal = palette if isinstance(palette, str) else "okabe_ito"
            return elec_plots.plot_isosurface(
                iso_arg,
                level=isolevel,
                symmetric=symmetric_iso,
                camera=camera,
                palette=iso_pal,
                shade=iso_shade,
                preset=preset,
                title=atoms_obj.formula(),
            )
        if atoms_obj.is_periodic and polyhedra is not None:
            return struct_plots.plot_coordination(
                atoms_obj,
                polyhedra[0],
                polyhedra[1],
                camera=camera,
                palette=palette,
                preset=preset,
                title=atoms_obj.formula(),
            )
        if atoms_obj.is_periodic:
            return struct_plots.plot_crystal(
                atoms_obj,
                style=style,
                palette=palette,
                preset=preset,
                camera=camera,
                show_cell=show_cell,
                vectors=vectors,
                vector_scale=vector_scale,
                title=atoms_obj.formula(),
            )
        return struct_plots.plot_molecule(
            atoms_obj,
            style=style,
            palette=palette,
            preset=preset,
            camera=camera,
            vectors=vectors,
            vector_scale=vector_scale,
            title=atoms_obj.formula(),
        )

    cam = geometry.resolve_camera(camera, cell=atoms_obj.cell)
    target_src: (
        structure.AtomsData
        | trajectory.Trajectory
        | electronic.VolumetricGrid
    ) = (
        trajectory.Trajectory(frames=frames_list)
        if frames_list is not None
        else (auto_iso if auto_iso is not None else atoms_obj)
    )
    html_doc = build_interactive_html(
        target_src,
        style=style,
        palette=palette,
        camera=cam,
        show_cell=show_cell,
        background=background,
        polyhedra=polyhedra,
        vectors=vectors,
        vector_scale=vector_scale,
        isosurface=isosurface,
        isolevel=isolevel,
        symmetric_iso=symmetric_iso,
        iso_shade=iso_shade,
        preset=preset,
    )
    view = InteractiveView(
        html_str=html_doc,
        title=f"{atoms_obj.formula()} - xkcd-matchem 3D Studio",
        camera=cam,
    )
    if mode == "browser":
        view.open_in_browser(path=output)
    elif output is not None:
        view.save(output)
    return view
