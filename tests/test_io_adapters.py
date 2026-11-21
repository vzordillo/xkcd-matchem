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
"""Tests for optional adapters (duck-typed mocks and import guards)."""

from __future__ import annotations

import types

import numpy as np
import pytest

from xkcd_matchem.core import exceptions
from xkcd_matchem.core import structure
from xkcd_matchem.io.adapters import ase_adapter
from xkcd_matchem.io.adapters import cclib_adapter
from xkcd_matchem.io.adapters import md_adapter
from xkcd_matchem.io.adapters import pymatgen_adapter
from xkcd_matchem.io.adapters import rdkit_adapter


def test_missing_optional_dependency_errors() -> None:
    atoms = structure.AtomsData(
        symbols=["H", "H"],
        positions=np.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]]),
    )
    for fn, arg in (
        (ase_adapter.to_ase, atoms),
        (pymatgen_adapter.to_pymatgen, atoms),
        (cclib_adapter.read_via_cclib, "dummy.log"),
        (rdkit_adapter.from_smiles, "C"),
    ):
        with pytest.raises(exceptions.OptionalDependencyError) as exc_info:
            fn(arg)
        assert "pip install" in str(exc_info.value)


def test_duck_typed_adapter_conversions() -> None:
    # Duck-typed ASE Atoms
    fake_ase = types.SimpleNamespace(
        get_chemical_symbols=lambda: ["C", "O"],
        get_positions=lambda: [[0.0, 0.0, 0.0], [1.13, 0.0, 0.0]],
        get_pbc=lambda: [False, False, False],
        get_cell=lambda: types.SimpleNamespace(array=np.zeros((3, 3))),
        calc=None,
        info={"tag": "co"},
    )
    co = ase_adapter.from_ase(fake_ase)
    assert co.formula() == "CO"

    # Duck-typed pymatgen Structure
    site1 = types.SimpleNamespace(specie=types.SimpleNamespace(symbol="Si"))
    site2 = types.SimpleNamespace(specie=types.SimpleNamespace(symbol="Si"))

    class _FakeStructure:
        cart_coords = np.array([[0.0, 0.0, 0.0], [1.35, 1.35, 1.35]])
        lattice = types.SimpleNamespace(matrix=np.diag([5.43, 5.43, 5.43]))

        def __iter__(self):
            return iter([site1, site2])

    si = pymatgen_adapter.from_pymatgen(_FakeStructure())
    assert si.formula() == "Si2"
    assert si.is_periodic

    # Duck-typed cclib data
    fake_cc = types.SimpleNamespace(
        atomnos=[8, 1, 1],
        atomcoords=[[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [0.0, 0.96, 0.0]]],
        scfenergies=[-2078.0],
        moenergies=[[-15.0, -10.0, 1.0]],
        homos=[1],
    )
    cc_atoms = cclib_adapter.from_cclib(fake_cc)
    assert cc_atoms.formula() == "H2O"
    assert "orbitals" in cc_atoms.metadata

    # Duck-typed MDAnalysis & mdtraj guard
    with pytest.raises(
        (exceptions.OptionalDependencyError, exceptions.ParseError)
    ):
        md_adapter.from_mdanalysis(object())
    with pytest.raises(
        (exceptions.OptionalDependencyError, exceptions.ParseError)
    ):
        md_adapter.from_mdtraj(object())
