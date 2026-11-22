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
"""xkcd-style visualization suite for chemistry and materials science."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from xkcd_matchem import _version


if TYPE_CHECKING:
    from xkcd_matchem import core
    from xkcd_matchem import data
    from xkcd_matchem import io
    from xkcd_matchem import layout
    from xkcd_matchem import plots
    from xkcd_matchem import render
    from xkcd_matchem import style
    from xkcd_matchem.io.native import cif
    from xkcd_matchem.io.native import lobster
    from xkcd_matchem.io.native import phonopy
    from xkcd_matchem.io.native import vasp

    AtomsData = core.AtomsData
    BACKGROUND_STYLES = style.BACKGROUND_STYLES
    Badge = layout.Badge
    BandStructure = core.BandStructure
    Board = layout.Board
    Bond = core.Bond
    BrillouinFacet = core.BrillouinFacet
    BrillouinZone = core.BrillouinZone
    CODATA2022 = data.CODATA2022
    CROMER_MANN_TABLE = data.CROMER_MANN_TABLE
    Camera = core.Camera
    Card = layout.Card
    CohpData = core.CohpData
    ComicPanel = layout.ComicPanel
    ComicStrip = layout.ComicStrip
    Container = layout.Container
    CoordinationPolyhedron = core.CoordinationPolyhedron
    DensityOfStates = core.DensityOfStates
    ELEMENTS_BY_SYMBOL = data.ELEMENTS_BY_SYMBOL
    ELEMENTS_BY_Z = data.ELEMENTS_BY_Z
    ElementRecord = data.ElementRecord
    InteractiveView = render.InteractiveView
    IsosurfaceMesh = core.IsosurfaceMesh
    MatchemError = core.MatchemError
    MolecularOrbitals = core.MolecularOrbitals
    NeighborHit = core.NeighborHit
    OKABE_ITO = style.OKABE_ITO
    OptionalDependencyError = core.OptionalDependencyError
    PRESETS = style.PRESETS
    ParseError = core.ParseError
    Peak = core.Peak
    PhononBandStructure = core.PhononBandStructure
    SHADE_STYLES = style.SHADE_STYLES
    ScatteringCoeffs = data.ScatteringCoeffs
    ScrollCard = layout.ScrollCard
    Spectrum = core.Spectrum
    StrokeHierarchy = style.StrokeHierarchy
    StructureError = core.StructureError
    StyleError = core.StyleError
    ThemePreset = style.ThemePreset
    Trajectory = core.Trajectory
    UnsupportedFormatError = core.UnsupportedFormatError
    VolumetricGrid = core.VolumetricGrid
    WulffFacet = core.WulffFacet
    WulffShape = core.WulffShape
    animate_trajectory = plots.animate_trajectory
    annotate_angle = render.annotate_angle
    annotate_distance = render.annotate_distance
    apply_background_grid = style.apply_background_grid
    atomic_form_factor = data.atomic_form_factor
    broaden_peaks = core.broaden_peaks
    clear_background_grid = style.clear_background_grid
    compute_background_grid_coords = style.compute_background_grid_coords
    compute_brillouin_zone = core.compute_brillouin_zone
    compute_wulff_shape = core.compute_wulff_shape
    compute_xrd = core.compute_xrd
    draw_callout = render.draw_callout
    draw_crystal_axes = render.draw_crystal_axes
    draw_curved_arrow = render.draw_curved_arrow
    draw_defect_site = render.draw_defect_site
    draw_element_badge = render.draw_element_badge
    draw_icon = render.draw_icon
    draw_speech_bubble = layout.draw_speech_bubble
    draw_stick_scientist = layout.draw_stick_scientist
    draw_vector_arrow_2d = render.draw_vector_arrow_2d
    ensure_readable_text_color = render.ensure_readable_text_color
    export_interactive_html = render.export_interactive_html
    extract_isosurface = core.extract_isosurface
    fetch_xkcd_script = style.fetch_xkcd_script
    find_coordination_polyhedra = core.find_coordination_polyhedra
    find_periodic_neighbors = core.find_periodic_neighbors
    from_ase = io.from_ase
    from_cclib = io.from_cclib
    from_mdanalysis = io.from_mdanalysis
    from_mdtraj = io.from_mdtraj
    from_pymatgen = io.from_pymatgen
    from_rdkit = io.from_rdkit
    from_smiles = io.from_smiles
    get_atomic_mass = data.get_atomic_mass
    get_covalent_radius = data.get_covalent_radius
    get_default_font = style.get_default_font
    get_element = data.get_element
    get_element_color = style.get_element_color
    get_jmol_color = data.get_jmol_color
    get_palette = style.get_palette
    get_sample_path = data.get_sample_path
    get_vdw_radius = data.get_vdw_radius
    infer_bond_orders = core.infer_bond_orders
    infer_bonds = core.infer_bonds
    is_dark_color = render.is_dark_color
    list_background_styles = style.list_background_styles
    list_element_palettes = style.list_element_palettes
    list_formats = io.list_formats
    list_icons = render.list_icons
    list_presets = style.list_presets
    list_samples = data.list_samples
    list_shade_styles = style.list_shade_styles
    load = io.load
    load_sample = io.load_sample
    load_trajectory = io.load_trajectory
    muted_surface_color = render.muted_surface_color
    plot_bands = plots.plot_bands
    plot_bands_dos = plots.plot_bands_dos
    plot_bar_comparison = plots.plot_bar_comparison
    plot_brillouin_zone = plots.plot_brillouin_zone
    plot_cohp = plots.plot_cohp
    plot_comic_strip = layout.plot_comic_strip
    plot_coordination = plots.plot_coordination
    plot_crystal = plots.plot_crystal
    plot_defect_formation = plots.plot_defect_formation
    plot_dos = plots.plot_dos
    plot_eos = plots.plot_eos
    plot_graph_network = plots.plot_graph_network
    plot_histogram = plots.plot_histogram
    plot_icon_sheet = render.plot_icon_sheet
    plot_isosurface = plots.plot_isosurface
    plot_md_timeseries = plots.plot_md_timeseries
    plot_mo_diagram = plots.plot_mo_diagram
    plot_molecule = plots.plot_molecule
    plot_neb = plots.plot_neb
    plot_parity = plots.plot_parity
    plot_periodic_table = plots.plot_periodic_table
    plot_phase_diagram_2d = plots.plot_phase_diagram_2d
    plot_phonon_bands = plots.plot_phonon_bands
    plot_pipeline_diagram = plots.plot_pipeline_diagram
    plot_pourbaix = plots.plot_pourbaix
    plot_rdf = plots.plot_rdf
    plot_reaction_profile = plots.plot_reaction_profile
    plot_reaction_scheme = plots.plot_reaction_scheme
    plot_rmsd = plots.plot_rmsd
    plot_scroll_card = layout.plot_scroll_card
    plot_slab = plots.plot_slab
    plot_spacegroup_distribution = plots.plot_spacegroup_distribution
    plot_spectrum = plots.plot_spectrum
    plot_structure_graph = plots.plot_structure_graph
    plot_ternary_phase_diagram = plots.plot_ternary_phase_diagram
    plot_volcano = plots.plot_volcano
    plot_volumetric_slice = plots.plot_volumetric_slice
    plot_wulff_shape = plots.plot_wulff_shape
    plot_xrd = plots.plot_xrd
    read_cif_all = cif.read_cif_all
    read_cohpcar = lobster.read_cohpcar
    read_phonopy_bands = phonopy.read_phonopy_bands
    read_procar = vasp.read_procar
    register_element_palette = style.register_element_palette
    register_font = style.register_font
    register_palette = style.register_palette
    register_preset = style.register_preset
    render_scroll_list = layout.render_scroll_list
    render_structure_2d = render.render_structure_2d
    render_structure_3d = render.render_structure_3d
    resolve_hatch = style.resolve_hatch
    resolve_ink = render.resolve_ink
    save = io.save
    save_figure = layout.save_figure
    set_default_font = style.set_default_font
    shade_between = style.shade_between
    spacegroup_to_crystal_system = plots.spacegroup_to_crystal_system
    stroke_weights = style.stroke_weights
    style_existing_axes = style.style_existing_axes
    to_ase = io.to_ase
    to_pymatgen = io.to_pymatgen
    use_xkcd_script = style.use_xkcd_script
    view_structure = render.view_structure
    xkcd_style = style.xkcd_style

__version__ = _version.__version__

__all__ = [
    "AtomsData",
    "BACKGROUND_STYLES",
    "Badge",
    "BandStructure",
    "Board",
    "Bond",
    "BrillouinFacet",
    "BrillouinZone",
    "CODATA2022",
    "CROMER_MANN_TABLE",
    "Camera",
    "Card",
    "CohpData",
    "ComicPanel",
    "ComicStrip",
    "Container",
    "CoordinationPolyhedron",
    "DensityOfStates",
    "ELEMENTS_BY_SYMBOL",
    "ELEMENTS_BY_Z",
    "ElementRecord",
    "InteractiveView",
    "IsosurfaceMesh",
    "MatchemError",
    "MolecularOrbitals",
    "NeighborHit",
    "OKABE_ITO",
    "OptionalDependencyError",
    "PRESETS",
    "ParseError",
    "Peak",
    "PhononBandStructure",
    "SHADE_STYLES",
    "ScatteringCoeffs",
    "ScrollCard",
    "Spectrum",
    "StrokeHierarchy",
    "StructureError",
    "StyleError",
    "ThemePreset",
    "Trajectory",
    "UnsupportedFormatError",
    "VolumetricGrid",
    "WulffFacet",
    "WulffShape",
    "__version__",
    "animate_trajectory",
    "annotate_angle",
    "annotate_distance",
    "apply_background_grid",
    "atomic_form_factor",
    "broaden_peaks",
    "clear_background_grid",
    "compute_background_grid_coords",
    "compute_brillouin_zone",
    "compute_wulff_shape",
    "compute_xrd",
    "core",
    "data",
    "draw_callout",
    "draw_crystal_axes",
    "draw_curved_arrow",
    "draw_defect_site",
    "draw_element_badge",
    "draw_icon",
    "draw_speech_bubble",
    "draw_stick_scientist",
    "draw_vector_arrow_2d",
    "ensure_readable_text_color",
    "export_interactive_html",
    "extract_isosurface",
    "fetch_xkcd_script",
    "find_coordination_polyhedra",
    "find_periodic_neighbors",
    "from_ase",
    "from_cclib",
    "from_mdanalysis",
    "from_mdtraj",
    "from_pymatgen",
    "from_rdkit",
    "from_smiles",
    "get_atomic_mass",
    "get_covalent_radius",
    "get_default_font",
    "get_element",
    "get_element_color",
    "get_jmol_color",
    "get_palette",
    "get_sample_path",
    "get_vdw_radius",
    "infer_bond_orders",
    "infer_bonds",
    "io",
    "is_dark_color",
    "layout",
    "list_background_styles",
    "list_element_palettes",
    "list_formats",
    "list_icons",
    "list_presets",
    "list_samples",
    "list_shade_styles",
    "load",
    "load_sample",
    "load_trajectory",
    "muted_surface_color",
    "plot_bands",
    "plot_bands_dos",
    "plot_bar_comparison",
    "plot_brillouin_zone",
    "plot_cohp",
    "plot_comic_strip",
    "plot_coordination",
    "plot_crystal",
    "plot_defect_formation",
    "plot_dos",
    "plot_eos",
    "plot_graph_network",
    "plot_histogram",
    "plot_icon_sheet",
    "plot_isosurface",
    "plot_md_timeseries",
    "plot_mo_diagram",
    "plot_molecule",
    "plot_neb",
    "plot_parity",
    "plot_periodic_table",
    "plot_phase_diagram_2d",
    "plot_phonon_bands",
    "plot_pipeline_diagram",
    "plot_pourbaix",
    "plot_rdf",
    "plot_reaction_profile",
    "plot_reaction_scheme",
    "plot_rmsd",
    "plot_scroll_card",
    "plot_slab",
    "plot_spacegroup_distribution",
    "plot_spectrum",
    "plot_structure_graph",
    "plot_ternary_phase_diagram",
    "plot_volcano",
    "plot_volumetric_slice",
    "plot_wulff_shape",
    "plot_xrd",
    "plots",
    "read_cif_all",
    "read_cohpcar",
    "read_phonopy_bands",
    "read_procar",
    "register_element_palette",
    "register_font",
    "register_palette",
    "register_preset",
    "render",
    "render_scroll_list",
    "render_structure_2d",
    "render_structure_3d",
    "resolve_hatch",
    "resolve_ink",
    "save",
    "save_figure",
    "set_default_font",
    "shade_between",
    "spacegroup_to_crystal_system",
    "stroke_weights",
    "style",
    "style_existing_axes",
    "to_ase",
    "to_pymatgen",
    "use_xkcd_script",
    "view_structure",
    "xkcd_style",
]

_SUBMODULES = frozenset(
    {"core", "data", "io", "layout", "plots", "render", "style"}
)

_CORE = "xkcd_matchem.core"
_DATA = "xkcd_matchem.data"
_IO = "xkcd_matchem.io"
_LAYOUT = "xkcd_matchem.layout"
_PLOTS = "xkcd_matchem.plots"
_RENDER = "xkcd_matchem.render"
_STYLE = "xkcd_matchem.style"
_NATIVE_CIF = "xkcd_matchem.io.native.cif"
_NATIVE_LOBSTER = "xkcd_matchem.io.native.lobster"
_NATIVE_PHONOPY = "xkcd_matchem.io.native.phonopy"
_NATIVE_VASP = "xkcd_matchem.io.native.vasp"

_LAZY_MAP: dict[str, tuple[str, str]] = {
    "AtomsData": (_CORE, "AtomsData"),
    "BACKGROUND_STYLES": (_STYLE, "BACKGROUND_STYLES"),
    "Badge": (_LAYOUT, "Badge"),
    "BandStructure": (_CORE, "BandStructure"),
    "Board": (_LAYOUT, "Board"),
    "Bond": (_CORE, "Bond"),
    "BrillouinFacet": (_CORE, "BrillouinFacet"),
    "BrillouinZone": (_CORE, "BrillouinZone"),
    "CODATA2022": (_DATA, "CODATA2022"),
    "CROMER_MANN_TABLE": (_DATA, "CROMER_MANN_TABLE"),
    "Camera": (_CORE, "Camera"),
    "Card": (_LAYOUT, "Card"),
    "CohpData": (_CORE, "CohpData"),
    "ComicPanel": (_LAYOUT, "ComicPanel"),
    "ComicStrip": (_LAYOUT, "ComicStrip"),
    "Container": (_LAYOUT, "Container"),
    "CoordinationPolyhedron": (_CORE, "CoordinationPolyhedron"),
    "DensityOfStates": (_CORE, "DensityOfStates"),
    "ELEMENTS_BY_SYMBOL": (_DATA, "ELEMENTS_BY_SYMBOL"),
    "ELEMENTS_BY_Z": (_DATA, "ELEMENTS_BY_Z"),
    "ElementRecord": (_DATA, "ElementRecord"),
    "InteractiveView": (_RENDER, "InteractiveView"),
    "IsosurfaceMesh": (_CORE, "IsosurfaceMesh"),
    "MatchemError": (_CORE, "MatchemError"),
    "MolecularOrbitals": (_CORE, "MolecularOrbitals"),
    "NeighborHit": (_CORE, "NeighborHit"),
    "OKABE_ITO": (_STYLE, "OKABE_ITO"),
    "OptionalDependencyError": (_CORE, "OptionalDependencyError"),
    "PRESETS": (_STYLE, "PRESETS"),
    "ParseError": (_CORE, "ParseError"),
    "Peak": (_CORE, "Peak"),
    "PhononBandStructure": (_CORE, "PhononBandStructure"),
    "SHADE_STYLES": (_STYLE, "SHADE_STYLES"),
    "ScatteringCoeffs": (_DATA, "ScatteringCoeffs"),
    "ScrollCard": (_LAYOUT, "ScrollCard"),
    "Spectrum": (_CORE, "Spectrum"),
    "StrokeHierarchy": (_STYLE, "StrokeHierarchy"),
    "StructureError": (_CORE, "StructureError"),
    "StyleError": (_CORE, "StyleError"),
    "ThemePreset": (_STYLE, "ThemePreset"),
    "Trajectory": (_CORE, "Trajectory"),
    "UnsupportedFormatError": (_CORE, "UnsupportedFormatError"),
    "VolumetricGrid": (_CORE, "VolumetricGrid"),
    "WulffFacet": (_CORE, "WulffFacet"),
    "WulffShape": (_CORE, "WulffShape"),
    "animate_trajectory": (_PLOTS, "animate_trajectory"),
    "annotate_angle": (_RENDER, "annotate_angle"),
    "annotate_distance": (_RENDER, "annotate_distance"),
    "apply_background_grid": (_STYLE, "apply_background_grid"),
    "atomic_form_factor": (_DATA, "atomic_form_factor"),
    "broaden_peaks": (_CORE, "broaden_peaks"),
    "clear_background_grid": (_STYLE, "clear_background_grid"),
    "compute_background_grid_coords": (
        _STYLE,
        "compute_background_grid_coords",
    ),
    "compute_brillouin_zone": (_CORE, "compute_brillouin_zone"),
    "compute_wulff_shape": (_CORE, "compute_wulff_shape"),
    "compute_xrd": (_CORE, "compute_xrd"),
    "draw_callout": (_RENDER, "draw_callout"),
    "draw_crystal_axes": (_RENDER, "draw_crystal_axes"),
    "draw_curved_arrow": (_RENDER, "draw_curved_arrow"),
    "draw_defect_site": (_RENDER, "draw_defect_site"),
    "draw_element_badge": (_RENDER, "draw_element_badge"),
    "draw_icon": (_RENDER, "draw_icon"),
    "draw_speech_bubble": (_LAYOUT, "draw_speech_bubble"),
    "draw_stick_scientist": (_LAYOUT, "draw_stick_scientist"),
    "draw_vector_arrow_2d": (_RENDER, "draw_vector_arrow_2d"),
    "ensure_readable_text_color": (_RENDER, "ensure_readable_text_color"),
    "export_interactive_html": (_RENDER, "export_interactive_html"),
    "extract_isosurface": (_CORE, "extract_isosurface"),
    "fetch_xkcd_script": (_STYLE, "fetch_xkcd_script"),
    "find_coordination_polyhedra": (_CORE, "find_coordination_polyhedra"),
    "find_periodic_neighbors": (_CORE, "find_periodic_neighbors"),
    "from_ase": (_IO, "from_ase"),
    "from_cclib": (_IO, "from_cclib"),
    "from_mdanalysis": (_IO, "from_mdanalysis"),
    "from_mdtraj": (_IO, "from_mdtraj"),
    "from_pymatgen": (_IO, "from_pymatgen"),
    "from_rdkit": (_IO, "from_rdkit"),
    "from_smiles": (_IO, "from_smiles"),
    "get_atomic_mass": (_DATA, "get_atomic_mass"),
    "get_covalent_radius": (_DATA, "get_covalent_radius"),
    "get_default_font": (_STYLE, "get_default_font"),
    "get_element": (_DATA, "get_element"),
    "get_element_color": (_STYLE, "get_element_color"),
    "get_jmol_color": (_DATA, "get_jmol_color"),
    "get_palette": (_STYLE, "get_palette"),
    "get_sample_path": (_DATA, "get_sample_path"),
    "get_vdw_radius": (_DATA, "get_vdw_radius"),
    "infer_bond_orders": (_CORE, "infer_bond_orders"),
    "infer_bonds": (_CORE, "infer_bonds"),
    "is_dark_color": (_RENDER, "is_dark_color"),
    "list_background_styles": (_STYLE, "list_background_styles"),
    "list_element_palettes": (_STYLE, "list_element_palettes"),
    "list_formats": (_IO, "list_formats"),
    "list_icons": (_RENDER, "list_icons"),
    "list_presets": (_STYLE, "list_presets"),
    "list_samples": (_DATA, "list_samples"),
    "list_shade_styles": (_STYLE, "list_shade_styles"),
    "load": (_IO, "load"),
    "load_sample": (_IO, "load_sample"),
    "load_trajectory": (_IO, "load_trajectory"),
    "muted_surface_color": (_RENDER, "muted_surface_color"),
    "plot_bands": (_PLOTS, "plot_bands"),
    "plot_bands_dos": (_PLOTS, "plot_bands_dos"),
    "plot_bar_comparison": (_PLOTS, "plot_bar_comparison"),
    "plot_brillouin_zone": (_PLOTS, "plot_brillouin_zone"),
    "plot_cohp": (_PLOTS, "plot_cohp"),
    "plot_comic_strip": (_LAYOUT, "plot_comic_strip"),
    "plot_coordination": (_PLOTS, "plot_coordination"),
    "plot_crystal": (_PLOTS, "plot_crystal"),
    "plot_defect_formation": (_PLOTS, "plot_defect_formation"),
    "plot_dos": (_PLOTS, "plot_dos"),
    "plot_eos": (_PLOTS, "plot_eos"),
    "plot_graph_network": (_PLOTS, "plot_graph_network"),
    "plot_histogram": (_PLOTS, "plot_histogram"),
    "plot_icon_sheet": (_RENDER, "plot_icon_sheet"),
    "plot_isosurface": (_PLOTS, "plot_isosurface"),
    "plot_md_timeseries": (_PLOTS, "plot_md_timeseries"),
    "plot_mo_diagram": (_PLOTS, "plot_mo_diagram"),
    "plot_molecule": (_PLOTS, "plot_molecule"),
    "plot_neb": (_PLOTS, "plot_neb"),
    "plot_parity": (_PLOTS, "plot_parity"),
    "plot_periodic_table": (_PLOTS, "plot_periodic_table"),
    "plot_phase_diagram_2d": (_PLOTS, "plot_phase_diagram_2d"),
    "plot_phonon_bands": (_PLOTS, "plot_phonon_bands"),
    "plot_pipeline_diagram": (_PLOTS, "plot_pipeline_diagram"),
    "plot_pourbaix": (_PLOTS, "plot_pourbaix"),
    "plot_rdf": (_PLOTS, "plot_rdf"),
    "plot_reaction_profile": (_PLOTS, "plot_reaction_profile"),
    "plot_reaction_scheme": (_PLOTS, "plot_reaction_scheme"),
    "plot_rmsd": (_PLOTS, "plot_rmsd"),
    "plot_scroll_card": (_LAYOUT, "plot_scroll_card"),
    "plot_slab": (_PLOTS, "plot_slab"),
    "plot_spacegroup_distribution": (_PLOTS, "plot_spacegroup_distribution"),
    "plot_spectrum": (_PLOTS, "plot_spectrum"),
    "plot_structure_graph": (_PLOTS, "plot_structure_graph"),
    "plot_ternary_phase_diagram": (_PLOTS, "plot_ternary_phase_diagram"),
    "plot_volcano": (_PLOTS, "plot_volcano"),
    "plot_volumetric_slice": (_PLOTS, "plot_volumetric_slice"),
    "plot_wulff_shape": (_PLOTS, "plot_wulff_shape"),
    "plot_xrd": (_PLOTS, "plot_xrd"),
    "read_cif_all": (_NATIVE_CIF, "read_cif_all"),
    "read_cohpcar": (_NATIVE_LOBSTER, "read_cohpcar"),
    "read_phonopy_bands": (_NATIVE_PHONOPY, "read_phonopy_bands"),
    "read_procar": (_NATIVE_VASP, "read_procar"),
    "register_element_palette": (_STYLE, "register_element_palette"),
    "register_font": (_STYLE, "register_font"),
    "register_palette": (_STYLE, "register_palette"),
    "register_preset": (_STYLE, "register_preset"),
    "render_scroll_list": (_LAYOUT, "render_scroll_list"),
    "render_structure_2d": (_RENDER, "render_structure_2d"),
    "render_structure_3d": (_RENDER, "render_structure_3d"),
    "resolve_hatch": (_STYLE, "resolve_hatch"),
    "resolve_ink": (_RENDER, "resolve_ink"),
    "save": (_IO, "save"),
    "save_figure": (_LAYOUT, "save_figure"),
    "set_default_font": (_STYLE, "set_default_font"),
    "shade_between": (_STYLE, "shade_between"),
    "spacegroup_to_crystal_system": (
        _PLOTS,
        "spacegroup_to_crystal_system",
    ),
    "stroke_weights": (_STYLE, "stroke_weights"),
    "style_existing_axes": (_STYLE, "style_existing_axes"),
    "to_ase": (_IO, "to_ase"),
    "to_pymatgen": (_IO, "to_pymatgen"),
    "use_xkcd_script": (_STYLE, "use_xkcd_script"),
    "view_structure": (_RENDER, "view_structure"),
    "xkcd_style": (_STYLE, "xkcd_style"),
}


def __getattr__(name: str) -> object:
    """Lazily resolves top-level symbols and subpackages (PEP 562)."""
    if name in _SUBMODULES:
        mod = importlib.import_module(f"xkcd_matchem.{name}")
        globals()[name] = mod
        return mod
    target = _LAZY_MAP.get(name)
    if target is not None:
        mod_path, attr_name = target
        val = getattr(importlib.import_module(mod_path), attr_name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Returns all public and imported attributes on ``xkcd_matchem``."""
    return sorted(set(globals()) | set(__all__))
