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
from xkcd_matchem.core import electronic
from xkcd_matchem.core import exceptions
from xkcd_matchem.core import geometry
from xkcd_matchem.core import spectroscopy
from xkcd_matchem.core import structure
from xkcd_matchem.core import trajectory


__all__ = [
    "AtomsData",
    "BandStructure",
    "Bond",
    "Camera",
    "CoordinationPolyhedron",
    "DensityOfStates",
    "MatchemError",
    "MolecularOrbitals",
    "OptionalDependencyError",
    "ParseError",
    "Peak",
    "Spectrum",
    "StructureError",
    "StyleError",
    "Trajectory",
    "UnsupportedFormatError",
    "VolumetricGrid",
    "broaden_peaks",
    "cell_vectors_from_parameters",
    "compute_xrd",
    "find_coordination_polyhedra",
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
Camera = geometry.Camera
CoordinationPolyhedron = bonding.CoordinationPolyhedron
DensityOfStates = electronic.DensityOfStates
MatchemError = exceptions.MatchemError
MolecularOrbitals = electronic.MolecularOrbitals
OptionalDependencyError = exceptions.OptionalDependencyError
ParseError = exceptions.ParseError
Peak = spectroscopy.Peak
Spectrum = spectroscopy.Spectrum
StructureError = exceptions.StructureError
StyleError = exceptions.StyleError
Trajectory = trajectory.Trajectory
UnsupportedFormatError = exceptions.UnsupportedFormatError
VolumetricGrid = electronic.VolumetricGrid
broaden_peaks = spectroscopy.broaden_peaks
cell_vectors_from_parameters = geometry.cell_vectors_from_parameters
compute_xrd = spectroscopy.compute_xrd
find_coordination_polyhedra = bonding.find_coordination_polyhedra
infer_bonds = bonding.infer_bonds
miller_plane_polygon = geometry.miller_plane_polygon
minimum_image_vector = geometry.minimum_image_vector
project_points = geometry.project_points
rotation_matrix = geometry.rotation_matrix
unit_cell_edges = geometry.unit_cell_edges
