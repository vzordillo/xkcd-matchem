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
from xkcd_matchem.data import samples_registry
from xkcd_matchem.io import loader
from xkcd_matchem.plots import electronic as elec_plots
from xkcd_matchem.plots import structure as struct_plots
from xkcd_matchem.style import fonts
from xkcd_matchem.style import palettes


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
            f'style="width:100%;height:{self.height}px;'
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
        return loaded[0], list(loaded.frames), None, src_name
    if isinstance(loaded, electronic.VolumetricGrid):
        if loaded.atoms is None or len(loaded.atoms) == 0:
            raise exceptions.StructureError(
                "VolumetricGrid requires associated atoms for 3D Studio."
            )
        return loaded.atoms, None, loaded, src_name
    if isinstance(loaded, structure.AtomsData):
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
    return [[float(r[0]), float(r[1]), float(r[2])] for r in scaled]


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
) -> list[dict[str, Any]]:
    """Extracts and serializes 3D isosurface triangles for the HTML viewer."""
    if iso_source is None:
        return []
    meshes_and_cols: list[tuple[isosurf_mod.IsosurfaceMesh, str]] = []
    if isinstance(iso_source, isosurf_mod.IsosurfaceMesh):
        meshes_and_cols.append((iso_source, "#0072B2"))
    else:
        grid = (
            iso_source
            if isinstance(iso_source, electronic.VolumetricGrid)
            else loader.load(iso_source)
        )
        if not isinstance(grid, electronic.VolumetricGrid):
            raise exceptions.StructureError(
                f"Expected VolumetricGrid, got {type(grid).__name__}."
            )
        max_abs = max(float(np.max(np.abs(grid.data))), 1e-12)
        level = 0.25 * max_abs if isolevel is None else float(isolevel)
        sym_flag = (
            bool(float(np.min(grid.data)) < -0.05 * max_abs)
            if symmetric_iso is None
            else bool(symmetric_iso)
        )
        meshes_and_cols.append(
            (isosurf_mod.extract_isosurface(grid, level), "#0072B2")
        )
        if sym_flag:
            meshes_and_cols.append(
                (isosurf_mod.extract_isosurface(grid, -abs(level)), "#D55E00")
            )

    rows: list[dict[str, Any]] = []
    for mesh, hex_col in meshes_and_cols:
        if len(mesh.faces) == 0:
            continue
        rows.append(
            {
                "color": hex_col,
                "vertices": np.round(mesh.vertices, 4).tolist(),
                "faces": mesh.faces.tolist(),
                "normals": np.round(mesh.normals, 3).tolist(),
            }
        )
    return rows


def _serialize_bond_rows(atoms_obj: structure.AtomsData) -> list[list[Any]]:
    """Extracts or infers bond index-order-shift quadruples."""
    bonds = (
        list(atoms_obj.bonds)
        if atoms_obj.bonds
        else bonding.infer_bonds(
            atoms_obj, scale=1.20, use_pbc=atoms_obj.is_periodic
        )
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
) -> dict[str, Any]:
    """Serializes an ``AtomsData``, ``Trajectory``, or cube grid for Studio."""
    atoms_obj, frames_list, auto_iso, source_sample = _coerce_interactive_input(
        atoms
    )
    cam = geometry.resolve_camera(camera, cell=atoms_obj.cell)
    bond_rows = _serialize_bond_rows(atoms_obj)
    poly_rows = _serialize_polyhedra_rows(atoms_obj, polyhedra)
    palette_name, custom_colors = _resolve_custom_palette(atoms_obj, palette)

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
        "polyhedra": poly_rows,
        "style": style,
        "palette_name": palette_name,
        "custom_colors": custom_colors,
        "background": background,
        "scale": float(scale),
        "show_cell": show_cell,
        "camera": {
            "azimuth": float(cam.azimuth),
            "elevation": float(cam.elevation),
            "roll": float(cam.roll),
            "perspective": float(cam.perspective),
            "zoom": float(cam.zoom),
        },
    }
    vec_rows = _resolve_js_vectors(atoms_obj, vectors, vector_scale)
    if vec_rows is not None:
        out["vectors"] = vec_rows
    iso_target = auto_iso if isosurface is None else isosurface
    iso_rows = _serialize_isosurface_meshes(
        iso_target, isolevel, symmetric_iso
    )
    if iso_rows:
        out["isosurfaces"] = iso_rows
    if frames_list is not None and len(frames_list) > 1:
        vec_attr = vectors if isinstance(vectors, str) else None
        out["frames"] = _serialize_trajectory_frames(
            frames_list, vec_attr, vector_scale
        )
    return out


def _build_font_css(font_family: str = "xkcd Script") -> str:
    """Embeds active bundled TTF font as base64 ``@font-face`` CSS."""
    if font_family not in fonts.list_bundled_fonts():
        return ""
    try:
        font_path = fonts.get_bundled_font_path(font_family)
        encoded = base64.b64encode(font_path.read_bytes()).decode("ascii")
        return (
            "@font-face {\n"
            f"  font-family: '{font_family}';\n"
            f"  src: url('data:font/ttf;base64,{encoded}') "
            "format('truetype');\n"
            "  font-display: swap;\n"
            "}\n"
        )
    except exceptions.StyleError:
        return ""


def _build_viewer_tables() -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    """Builds element radius/color tables for offline JS viewer switching."""
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
    return elem_table, pal_tables


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
    )
    elem_table, pal_tables = _build_viewer_tables()
    payload = {
        "active": active_payload,
        "samples": list(samples) if samples is not None else [active_payload],
        "elements_table": elem_table,
        "palettes": pal_tables,
    }
    template = (
        importlib.resources.files("xkcd_matchem.data")
        .joinpath("templates/viewer.html")
        .read_text(encoding="utf-8")
    )
    safe_json = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    active_font = fonts.get_default_font() or "xkcd Script"
    font_css = _build_font_css(active_font)
    return (
        template.replace("__XM_FONT_CSS__", font_css)
        .replace("__XM_FONT_FAMILY__", html.escape(active_font, quote=True))
        .replace("__XM_PAYLOAD__", safe_json)
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
) -> pathlib.Path:
    """Exports a standalone offline interactive 3D studio HTML file."""
    extra_samples: list[dict[str, Any]] | None = None
    if include_bundled_samples:
        extra_samples = []
        first = serialize_structure(
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
        )
        extra_samples.append(first)
        active_fname = _active_sample_filename(atoms)
        for sample_name in (
            "si_diamond.cif",
            "tio2_slab.poscar",
            "mos2_2h.poscar",
            "nacl.cif",
            "aspirin.xyz",
            "benzene.sdf",
            "alaninedipeptide.pdb",
            "water_homo.cube",
        ):
            if sample_name == active_fname:
                continue
            path_obj = samples_registry.get_sample_path(sample_name)
            loaded = loader.load(path_obj)
            if isinstance(loaded, structure.AtomsData):
                poly_opt: tuple[str, str | None] | None = (
                    ("Ti", "O") if sample_name == "tio2_slab.poscar" else None
                )
                style_opt = (
                    "polyhedra" if poly_opt is not None else "ball_and_stick"
                )
                extra_samples.append(
                    serialize_structure(
                        loaded,
                        style=style_opt,
                        name=f"{loaded.formula()} ({sample_name})",
                        polyhedra=poly_opt,
                    )
                )
            elif isinstance(loaded, electronic.VolumetricGrid):
                extra_samples.append(
                    serialize_structure(
                        loaded,
                        name="Water HOMO 3D Isosurface (water_homo.cube)",
                    )
                )
        if active_fname != "sn2_neb.xyz":
            sn2_traj = loader.load_trajectory("sn2_neb.xyz")
            extra_samples.append(
                serialize_structure(
                    sn2_traj,
                    name="SN2 Reaction Trajectory (sn2_neb.xyz)",
                )
            )
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
        name=name,
        vectors=vectors,
        vector_scale=vector_scale,
        isosurface=isosurface,
        isolevel=isolevel,
        symmetric_iso=symmetric_iso,
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
      preset: Theme preset name when rendering via ``"matplotlib"``.
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
