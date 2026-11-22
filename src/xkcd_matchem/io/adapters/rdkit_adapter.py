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
"""Lazy adapter converting RDKit Mol objects and SMILES strings."""

from __future__ import annotations

from typing import Any

import numpy as np

from xkcd_matchem.core import exceptions
from xkcd_matchem.core import structure


def _require_rdkit() -> tuple[Any, Any]:
    """Lazily imports ``rdkit.Chem`` and ``rdkit.Chem.AllChem``."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as exc:
        raise exceptions.OptionalDependencyError(
            "RDKit adapter requires 'rdkit'. Install via: "
            "pip install 'xkcd-matchem[rdkit]'"
        ) from exc
    return Chem, AllChem


def from_rdkit(mol: Any, conf_id: int = -1) -> structure.AtomsData:
    """Converts an RDKit ``Mol`` (with conformer) into ``AtomsData``."""
    chem_mod, allchem_mod = _require_rdkit()
    working = mol
    if working.GetNumConformers() == 0:
        working = chem_mod.AddHs(working)
        status = allchem_mod.EmbedMolecule(working, allchem_mod.ETKDGv3())
        if status != 0:
            allchem_mod.Compute2DCoords(working)
    conf = working.GetConformer(conf_id)
    symbols = [atom.GetSymbol() for atom in working.GetAtoms()]
    positions = np.array(conf.GetPositions(), dtype=np.float64)
    dir_map = {
        "BEGINWEDGE": "wedge",
        "BEGINDASH": "dash",
        "UNKNOWN": "wavy",
    }
    bonds = [
        structure.Bond(
            i=bond.GetBeginAtomIdx(),
            j=bond.GetEndAtomIdx(),
            order=float(bond.GetBondTypeAsDouble()),
            stereo=dir_map.get(str(bond.GetBondDir()), "none"),
        )
        for bond in working.GetBonds()
    ]
    return structure.AtomsData(
        symbols=symbols, positions=positions, bonds=bonds
    )


def from_smiles(smiles: str, seed: int = 42) -> structure.AtomsData:
    """Generates a 3D ``AtomsData`` conformer directly from a SMILES string."""
    chem_mod, allchem_mod = _require_rdkit()
    mol = chem_mod.MolFromSmiles(smiles)
    if mol is None:
        raise exceptions.ParseError(f"Invalid SMILES string: {smiles!r}")
    mol_h = chem_mod.AddHs(mol)
    params = allchem_mod.ETKDGv3()
    params.randomSeed = int(seed)
    if allchem_mod.EmbedMolecule(mol_h, params) == 0:
        allchem_mod.MMFFOptimizeMolecule(mol_h)
    else:
        allchem_mod.Compute2DCoords(mol_h)
    atoms = from_rdkit(mol_h)
    atoms.metadata["smiles"] = smiles
    return atoms
