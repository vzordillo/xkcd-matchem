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
"""Domain-specific plotting functions for molecules and materials."""

from __future__ import annotations

from xkcd_matchem.plots import animations
from xkcd_matchem.plots import charts
from xkcd_matchem.plots import crystalfield
from xkcd_matchem.plots import electrochem
from xkcd_matchem.plots import electronic
from xkcd_matchem.plots import graphs
from xkcd_matchem.plots import materials
from xkcd_matchem.plots import ptable
from xkcd_matchem.plots import reactions
from xkcd_matchem.plots import spectroscopy
from xkcd_matchem.plots import structure
from xkcd_matchem.plots import thermo
from xkcd_matchem.plots import trajectory


__all__ = [
    "animate_isosurface",
    "animate_orbit",
    "animate_trajectory",
    "plot_bands",
    "plot_bands_dos",
    "plot_bar_comparison",
    "plot_battery_profile",
    "plot_brillouin_zone",
    "plot_cohp",
    "plot_coordination",
    "plot_crystal",
    "plot_crystal_field",
    "plot_cyclic_voltammetry",
    "plot_defect_formation",
    "plot_dos",
    "plot_dqdv",
    "plot_eos",
    "plot_graph_network",
    "plot_histogram",
    "plot_isosurface",
    "plot_md_timeseries",
    "plot_mo_diagram",
    "plot_molecule",
    "plot_neb",
    "plot_nmr",
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
    "plot_slab",
    "plot_spacegroup_distribution",
    "plot_spectrum",
    "plot_structure_graph",
    "plot_tafel",
    "plot_ternary_phase_diagram",
    "plot_volcano",
    "plot_volumetric_slice",
    "plot_wulff_shape",
    "plot_xps",
    "plot_xrd",
    "spacegroup_to_crystal_system",
]

animate_isosurface = animations.animate_isosurface
animate_orbit = animations.animate_orbit
animate_trajectory = trajectory.animate_trajectory
plot_bands = electronic.plot_bands
plot_bands_dos = electronic.plot_bands_dos
plot_bar_comparison = charts.plot_bar_comparison
plot_battery_profile = electrochem.plot_battery_profile
plot_brillouin_zone = materials.plot_brillouin_zone
plot_cohp = electronic.plot_cohp
plot_coordination = structure.plot_coordination
plot_crystal = structure.plot_crystal
plot_crystal_field = crystalfield.plot_crystal_field
plot_cyclic_voltammetry = electrochem.plot_cyclic_voltammetry
plot_defect_formation = thermo.plot_defect_formation
plot_dos = electronic.plot_dos
plot_dqdv = electrochem.plot_dqdv
plot_eos = thermo.plot_eos
plot_graph_network = graphs.plot_graph_network
plot_histogram = materials.plot_histogram
plot_isosurface = electronic.plot_isosurface
plot_md_timeseries = trajectory.plot_md_timeseries
plot_mo_diagram = electronic.plot_mo_diagram
plot_molecule = structure.plot_molecule
plot_neb = reactions.plot_neb
plot_nmr = spectroscopy.plot_nmr
plot_parity = charts.plot_parity
plot_periodic_table = ptable.plot_periodic_table
plot_phase_diagram_2d = thermo.plot_phase_diagram_2d
plot_phonon_bands = electronic.plot_phonon_bands
plot_pipeline_diagram = charts.plot_pipeline_diagram
plot_pourbaix = thermo.plot_pourbaix
plot_rdf = trajectory.plot_rdf
plot_reaction_profile = reactions.plot_reaction_profile
plot_reaction_scheme = reactions.plot_reaction_scheme
plot_rmsd = trajectory.plot_rmsd
plot_slab = structure.plot_slab
plot_spacegroup_distribution = materials.plot_spacegroup_distribution
plot_spectrum = spectroscopy.plot_spectrum
plot_structure_graph = graphs.plot_structure_graph
plot_tafel = electrochem.plot_tafel
plot_ternary_phase_diagram = materials.plot_ternary_phase_diagram
plot_volcano = thermo.plot_volcano
plot_volumetric_slice = electronic.plot_volumetric_slice
plot_wulff_shape = materials.plot_wulff_shape
plot_xps = spectroscopy.plot_xps
plot_xrd = spectroscopy.plot_xrd
spacegroup_to_crystal_system = materials.spacegroup_to_crystal_system
