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
"""Hand-drawn xkcd-style visualization package for molecules and materials."""

from __future__ import annotations

from xkcd_matchem import _version
from xkcd_matchem import core
from xkcd_matchem import data
from xkcd_matchem import io
from xkcd_matchem import layout
from xkcd_matchem import plots
from xkcd_matchem import render
from xkcd_matchem import style


__version__ = _version.__version__

__all__ = [
    "AtomsData",
    "Badge",
    "BandStructure",
    "Board",
    "Bond",
    "CODATA2022",
    "CROMER_MANN_TABLE",
    "Camera",
    "Card",
    "Container",
    "CoordinationPolyhedron",
    "DensityOfStates",
    "ELEMENTS_BY_SYMBOL",
    "ELEMENTS_BY_Z",
    "ElementRecord",
    "InteractiveView",
    "MatchemError",
    "MolecularOrbitals",
    "OKABE_ITO",
    "OptionalDependencyError",
    "PRESETS",
    "ParseError",
    "Peak",
    "ScatteringCoeffs",
    "Spectrum",
    "StructureError",
    "StyleError",
    "ThemePreset",
    "Trajectory",
    "UnsupportedFormatError",
    "VolumetricGrid",
    "__version__",
    "animate_trajectory",
    "annotate_angle",
    "annotate_distance",
    "atomic_form_factor",
    "broaden_peaks",
    "compute_xrd",
    "core",
    "data",
    "draw_callout",
    "draw_curved_arrow",
    "export_interactive_html",
    "fetch_xkcd_script",
    "find_coordination_polyhedra",
    "from_ase",
    "from_cclib",
    "from_mdanalysis",
    "from_mdtraj",
    "from_pymatgen",
    "from_rdkit",
    "from_smiles",
    "get_atomic_mass",
    "get_covalent_radius",
    "get_element",
    "get_element_color",
    "get_jmol_color",
    "get_palette",
    "get_sample_path",
    "get_vdw_radius",
    "infer_bonds",
    "io",
    "layout",
    "list_element_palettes",
    "list_formats",
    "list_presets",
    "list_samples",
    "load",
    "load_sample",
    "load_trajectory",
    "plot_bands",
    "plot_bands_dos",
    "plot_bar_comparison",
    "plot_coordination",
    "plot_crystal",
    "plot_dos",
    "plot_eos",
    "plot_md_timeseries",
    "plot_mo_diagram",
    "plot_molecule",
    "plot_neb",
    "plot_parity",
    "plot_phase_diagram_2d",
    "plot_pipeline_diagram",
    "plot_pourbaix",
    "plot_rdf",
    "plot_reaction_profile",
    "plot_rmsd",
    "plot_slab",
    "plot_spectrum",
    "plot_volumetric_slice",
    "plot_xrd",
    "plots",
    "register_element_palette",
    "register_palette",
    "register_preset",
    "render",
    "render_structure_2d",
    "render_structure_3d",
    "save",
    "save_figure",
    "style",
    "style_existing_axes",
    "to_ase",
    "to_pymatgen",
    "view_structure",
    "xkcd_style",
]

AtomsData = core.AtomsData
Badge = layout.Badge
BandStructure = core.BandStructure
Board = layout.Board
Bond = core.Bond
CODATA2022 = data.CODATA2022
CROMER_MANN_TABLE = data.CROMER_MANN_TABLE
Camera = core.Camera
Card = layout.Card
Container = layout.Container
CoordinationPolyhedron = core.CoordinationPolyhedron
DensityOfStates = core.DensityOfStates
ELEMENTS_BY_SYMBOL = data.ELEMENTS_BY_SYMBOL
ELEMENTS_BY_Z = data.ELEMENTS_BY_Z
ElementRecord = data.ElementRecord
InteractiveView = render.InteractiveView
MatchemError = core.MatchemError
MolecularOrbitals = core.MolecularOrbitals
OKABE_ITO = style.OKABE_ITO
OptionalDependencyError = core.OptionalDependencyError
PRESETS = style.PRESETS
ParseError = core.ParseError
Peak = core.Peak
ScatteringCoeffs = data.ScatteringCoeffs
Spectrum = core.Spectrum
StructureError = core.StructureError
StyleError = core.StyleError
ThemePreset = style.ThemePreset
Trajectory = core.Trajectory
UnsupportedFormatError = core.UnsupportedFormatError
VolumetricGrid = core.VolumetricGrid
animate_trajectory = plots.animate_trajectory
annotate_angle = render.annotate_angle
annotate_distance = render.annotate_distance
atomic_form_factor = data.atomic_form_factor
broaden_peaks = core.broaden_peaks
compute_xrd = core.compute_xrd
draw_callout = render.draw_callout
draw_curved_arrow = render.draw_curved_arrow
export_interactive_html = render.export_interactive_html
fetch_xkcd_script = style.fetch_xkcd_script
find_coordination_polyhedra = core.find_coordination_polyhedra
from_ase = io.from_ase
from_cclib = io.from_cclib
from_mdanalysis = io.from_mdanalysis
from_mdtraj = io.from_mdtraj
from_pymatgen = io.from_pymatgen
from_rdkit = io.from_rdkit
from_smiles = io.from_smiles
get_atomic_mass = data.get_atomic_mass
get_covalent_radius = data.get_covalent_radius
get_element = data.get_element
get_element_color = style.get_element_color
get_jmol_color = data.get_jmol_color
get_palette = style.get_palette
get_sample_path = data.get_sample_path
get_vdw_radius = data.get_vdw_radius
infer_bonds = core.infer_bonds
list_element_palettes = style.list_element_palettes
list_formats = io.list_formats
list_presets = style.list_presets
list_samples = data.list_samples
load = io.load
load_sample = io.load_sample
load_trajectory = io.load_trajectory
plot_bands = plots.plot_bands
plot_bands_dos = plots.plot_bands_dos
plot_bar_comparison = plots.plot_bar_comparison
plot_coordination = plots.plot_coordination
plot_crystal = plots.plot_crystal
plot_dos = plots.plot_dos
plot_eos = plots.plot_eos
plot_md_timeseries = plots.plot_md_timeseries
plot_mo_diagram = plots.plot_mo_diagram
plot_molecule = plots.plot_molecule
plot_neb = plots.plot_neb
plot_parity = plots.plot_parity
plot_phase_diagram_2d = plots.plot_phase_diagram_2d
plot_pipeline_diagram = plots.plot_pipeline_diagram
plot_pourbaix = plots.plot_pourbaix
plot_rdf = plots.plot_rdf
plot_reaction_profile = plots.plot_reaction_profile
plot_rmsd = plots.plot_rmsd
plot_slab = plots.plot_slab
plot_spectrum = plots.plot_spectrum
plot_volumetric_slice = plots.plot_volumetric_slice
plot_xrd = plots.plot_xrd
register_element_palette = style.register_element_palette
register_palette = style.register_palette
register_preset = style.register_preset
render_structure_2d = render.render_structure_2d
render_structure_3d = render.render_structure_3d
save = io.save
save_figure = layout.save_figure
style_existing_axes = style.style_existing_axes
to_ase = io.to_ase
to_pymatgen = io.to_pymatgen
view_structure = render.view_structure
xkcd_style = style.xkcd_style
