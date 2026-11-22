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
"""Core data models and scientific geometry/bonding algorithms."""

from __future__ import annotations

from xkcd_matchem.core import bonding
from xkcd_matchem.core import brillouin
from xkcd_matchem.core import electronic
from xkcd_matchem.core import exceptions
from xkcd_matchem.core import geometry
from xkcd_matchem.core import isosurface
from xkcd_matchem.core import neighbors
from xkcd_matchem.core import spectroscopy
from xkcd_matchem.core import structure
from xkcd_matchem.core import trajectory
from xkcd_matchem.core import wulff


__all__ = [
    "AtomsData",
    "BandStructure",
    "Bond",
    "BrillouinFacet",
    "BrillouinZone",
    "Camera",
    "CohpData",
    "CoordinationPolyhedron",
    "DensityOfStates",
    "EOSFitResult",
    "IsosurfaceMesh",
    "MatchemError",
    "MolecularOrbitals",
    "NeighborHit",
    "OptionalDependencyError",
    "ParseError",
    "Peak",
    "PhononBandStructure",
    "Spectrum",
    "StructureError",
    "StyleError",
    "Trajectory",
    "UnsupportedFormatError",
    "VolumetricGrid",
    "WulffFacet",
    "WulffShape",
    "birch_murnaghan_energy",
    "birch_murnaghan_pressure",
    "broaden_peaks",
    "cell_vectors_from_parameters",
    "compute_brillouin_zone",
    "compute_wulff_shape",
    "compute_xrd",
    "extract_isosurface",
    "find_coordination_polyhedra",
    "find_periodic_neighbors",
    "fit_birch_murnaghan",
    "infer_bond_orders",
    "infer_bonds",
    "miller_plane_polygon",
    "minimum_image_vector",
    "project_points",
    "rotation_matrix",
    "unit_cell_edges",
]

AtomsData = structure.AtomsData
BandStructure = electronic.BandStructure
Bond = structure.Bond
BrillouinFacet = brillouin.BrillouinFacet
BrillouinZone = brillouin.BrillouinZone
Camera = geometry.Camera
CohpData = electronic.CohpData
CoordinationPolyhedron = bonding.CoordinationPolyhedron
DensityOfStates = electronic.DensityOfStates
EOSFitResult = wulff.EOSFitResult
IsosurfaceMesh = isosurface.IsosurfaceMesh
MatchemError = exceptions.MatchemError
MolecularOrbitals = electronic.MolecularOrbitals
NeighborHit = neighbors.NeighborHit
OptionalDependencyError = exceptions.OptionalDependencyError
ParseError = exceptions.ParseError
Peak = spectroscopy.Peak
PhononBandStructure = electronic.PhononBandStructure
Spectrum = spectroscopy.Spectrum
StructureError = exceptions.StructureError
StyleError = exceptions.StyleError
Trajectory = trajectory.Trajectory
UnsupportedFormatError = exceptions.UnsupportedFormatError
VolumetricGrid = electronic.VolumetricGrid
WulffFacet = wulff.WulffFacet
WulffShape = wulff.WulffShape
birch_murnaghan_energy = wulff.birch_murnaghan_energy
birch_murnaghan_pressure = wulff.birch_murnaghan_pressure
broaden_peaks = spectroscopy.broaden_peaks
cell_vectors_from_parameters = geometry.cell_vectors_from_parameters
compute_brillouin_zone = brillouin.compute_brillouin_zone
compute_wulff_shape = wulff.compute_wulff_shape
compute_xrd = spectroscopy.compute_xrd
extract_isosurface = isosurface.extract_isosurface
find_coordination_polyhedra = bonding.find_coordination_polyhedra
find_periodic_neighbors = neighbors.find_periodic_neighbors
fit_birch_murnaghan = wulff.fit_birch_murnaghan
infer_bond_orders = bonding.infer_bond_orders
infer_bonds = bonding.infer_bonds
miller_plane_polygon = geometry.miller_plane_polygon
minimum_image_vector = geometry.minimum_image_vector
project_points = geometry.project_points
rotation_matrix = geometry.rotation_matrix
unit_cell_edges = geometry.unit_cell_edges
