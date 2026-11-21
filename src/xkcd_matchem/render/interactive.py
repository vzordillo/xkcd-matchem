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
from xkcd_matchem.core import exceptions
from xkcd_matchem.core import geometry
from xkcd_matchem.core import structure
from xkcd_matchem.data import elements
from xkcd_matchem.data import samples_registry
from xkcd_matchem.io import loader
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
    """

    html_str: str
    title: str = "xkcd-matchem 3D Studio"
    height: int = 680

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


def _load_atoms_input(
    atoms: structure.AtomsData | str | pathlib.Path,
) -> tuple[structure.AtomsData, str | None]:
    """Coerces a file path or bundled sample name into ``AtomsData``."""
    if isinstance(atoms, (str, pathlib.Path)):
        loaded = loader.load(atoms)
        if not isinstance(loaded, structure.AtomsData):
            raise exceptions.StructureError(
                f"Expected an atomic structure from {atoms!r}."
            )
        return loaded, str(atoms)
    return atoms, None


def serialize_structure(
    atoms: structure.AtomsData | str | pathlib.Path,
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
) -> dict[str, Any]:
    """Serializes an ``AtomsData`` instance for the HTML5 3D viewer."""
    atoms_obj, source_sample = _load_atoms_input(atoms)
    cam = geometry.resolve_camera(camera, cell=atoms_obj.cell)
    bonds = (
        list(atoms_obj.bonds)
        if atoms_obj.bonds
        else bonding.infer_bonds(
            atoms_obj, scale=1.20, use_pbc=atoms_obj.is_periodic
        )
    )
    bond_rows = [
        [
            int(b.i),
            int(b.j),
            float(b.order),
            [int(b.shift[0]), int(b.shift[1]), int(b.shift[2])],
        ]
        for b in bonds
    ]

    poly_rows: list[dict[str, Any]] = []
    if polyhedra is not None:
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
        for poly in poly_list:
            poly_rows.append(
                {
                    "center_symbol": poly.center_symbol,
                    "vertices": np.round(poly.vertices_3d, 5).tolist(),
                    "faces": [
                        [int(a), int(b), int(c)] for a, b, c in poly.faces
                    ],
                }
            )

    custom_colors: dict[str, str] = {}
    palette_name = "xkcd"
    if isinstance(palette, Mapping):
        for sym in dict.fromkeys(atoms_obj.symbols):
            custom_colors[sym] = palettes.get_element_color(sym, scheme=palette)
    else:
        palette_name = str(palette)
        for sym in dict.fromkeys(atoms_obj.symbols):
            col = palettes.get_element_color(sym, scheme=palette_name)
            base_col = palettes.get_element_color(sym, scheme="xkcd")
            if (
                palette_name not in palettes.list_element_palettes()
                and col != base_col
            ):
                custom_colors[sym] = col

    return {
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
        "palette_name": (
            palette_name
            if palette_name in palettes.list_element_palettes()
            else "xkcd"
        ),
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


def _build_font_css() -> str:
    """Embeds bundled xkcd Script TTF as base64 ``@font-face`` CSS."""
    try:
        font_path = fonts.get_bundled_font_path("xkcd Script")
        encoded = base64.b64encode(font_path.read_bytes()).decode("ascii")
        return (
            "@font-face {\n"
            "  font-family: 'xkcd Script';\n"
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
    atoms: structure.AtomsData | str | pathlib.Path,
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
    return template.replace("__XM_FONT_CSS__", _build_font_css()).replace(
        "__XM_PAYLOAD__", safe_json
    )


def export_interactive_html(
    atoms: structure.AtomsData | str | pathlib.Path,
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
        )
        extra_samples.append(first)
        for sample_name in (
            "si_diamond.cif",
            "tio2_slab.poscar",
            "mos2_2h.poscar",
            "nacl.cif",
            "aspirin.xyz",
            "benzene.sdf",
            "alaninedipeptide.pdb",
        ):
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
    )
    out = pathlib.Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_text, encoding="utf-8")
    return out


def view_structure(
    atoms: structure.AtomsData | str | pathlib.Path,
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
) -> InteractiveView | maxes.Axes:
    """Opens a molecule or crystal in interactive HTML, browser, or Matplotlib.

    Args:
      atoms: ``AtomsData``, file path, or bundled sample filename.
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
    atoms_obj, _ = _load_atoms_input(atoms)
    if center and atoms_obj.is_periodic:
        atoms_obj = atoms_obj.center_in_cell()
    if supercell is not None and supercell != (1, 1, 1):
        atoms_obj = atoms_obj.repeat(supercell)

    if mode == "matplotlib":
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
                title=atoms_obj.formula(),
            )
        return struct_plots.plot_molecule(
            atoms_obj,
            style=style,
            palette=palette,
            preset=preset,
            camera=camera,
            title=atoms_obj.formula(),
        )

    html_doc = build_interactive_html(
        atoms_obj,
        style=style,
        palette=palette,
        camera=camera,
        show_cell=show_cell,
        background=background,
        polyhedra=polyhedra,
    )
    view = InteractiveView(
        html_str=html_doc,
        title=f"{atoms_obj.formula()} - xkcd-matchem 3D Studio",
    )
    if mode == "browser":
        view.open_in_browser(path=output)
    elif output is not None:
        view.save(output)
    return view
