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
"""Native chemical/materials file parsers and optional framework adapters."""

from __future__ import annotations

from xkcd_matchem.io import adapters
from xkcd_matchem.io import loader
from xkcd_matchem.io import native
from xkcd_matchem.io import registry


__all__ = [
    "FormatSpec",
    "adapters",
    "detect_format",
    "from_ase",
    "from_cclib",
    "from_mdanalysis",
    "from_mdtraj",
    "from_pymatgen",
    "from_rdkit",
    "from_smiles",
    "get_format",
    "list_formats",
    "load",
    "load_sample",
    "load_trajectory",
    "native",
    "read_via_ase",
    "read_via_cclib",
    "save",
    "to_ase",
    "to_pymatgen",
]

FormatSpec = registry.FormatSpec
detect_format = registry.detect_format
from_ase = adapters.from_ase
from_cclib = adapters.from_cclib
from_mdanalysis = adapters.from_mdanalysis
from_mdtraj = adapters.from_mdtraj
from_pymatgen = adapters.from_pymatgen
from_rdkit = adapters.from_rdkit
from_smiles = adapters.from_smiles
get_format = registry.get_format
list_formats = registry.list_formats
load = loader.load
load_sample = loader.load_sample
load_trajectory = loader.load_trajectory
read_via_ase = adapters.read_via_ase
read_via_cclib = adapters.read_via_cclib
save = loader.save
to_ase = adapters.to_ase
to_pymatgen = adapters.to_pymatgen
