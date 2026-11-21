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
"""Scientific data tables, CODATA 2022 constants, and bundled samples."""

from __future__ import annotations

from xkcd_matchem.data import elements
from xkcd_matchem.data import samples_registry
from xkcd_matchem.data import scattering


__all__ = [
    "CODATA2022",
    "CROMER_MANN_TABLE",
    "ELEMENTS_BY_SYMBOL",
    "ELEMENTS_BY_Z",
    "ElementRecord",
    "ScatteringCoeffs",
    "atomic_form_factor",
    "describe_sample",
    "get_atomic_mass",
    "get_covalent_radius",
    "get_element",
    "get_jmol_color",
    "get_sample_path",
    "get_vdw_radius",
    "list_samples",
]

CODATA2022 = elements.CODATA2022
CROMER_MANN_TABLE = scattering.CROMER_MANN_TABLE
ELEMENTS_BY_SYMBOL = elements.ELEMENTS_BY_SYMBOL
ELEMENTS_BY_Z = elements.ELEMENTS_BY_Z
ElementRecord = elements.ElementRecord
ScatteringCoeffs = scattering.ScatteringCoeffs
atomic_form_factor = scattering.atomic_form_factor
describe_sample = samples_registry.describe_sample
get_atomic_mass = elements.get_atomic_mass
get_covalent_radius = elements.get_covalent_radius
get_element = elements.get_element
get_jmol_color = elements.get_jmol_color
get_sample_path = samples_registry.get_sample_path
get_vdw_radius = elements.get_vdw_radius
list_samples = samples_registry.list_samples
