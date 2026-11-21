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
"""Lazy optional adapters for ASE, pymatgen, cclib, RDKit, and MD tools."""

from __future__ import annotations

from xkcd_matchem.io.adapters import ase_adapter
from xkcd_matchem.io.adapters import cclib_adapter
from xkcd_matchem.io.adapters import md_adapter
from xkcd_matchem.io.adapters import pymatgen_adapter
from xkcd_matchem.io.adapters import rdkit_adapter


__all__ = [
    "from_ase",
    "from_cclib",
    "from_mdanalysis",
    "from_mdtraj",
    "from_pymatgen",
    "from_rdkit",
    "from_smiles",
    "read_via_ase",
    "read_via_cclib",
    "to_ase",
    "to_pymatgen",
]

from_ase = ase_adapter.from_ase
from_cclib = cclib_adapter.from_cclib
from_mdanalysis = md_adapter.from_mdanalysis
from_mdtraj = md_adapter.from_mdtraj
from_pymatgen = pymatgen_adapter.from_pymatgen
from_rdkit = rdkit_adapter.from_rdkit
from_smiles = rdkit_adapter.from_smiles
read_via_ase = ase_adapter.read_via_ase
read_via_cclib = cclib_adapter.read_via_cclib
to_ase = ase_adapter.to_ase
to_pymatgen = pymatgen_adapter.to_pymatgen
