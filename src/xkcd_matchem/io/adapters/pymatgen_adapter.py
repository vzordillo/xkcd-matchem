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
"""Lazy adapter converting pymatgen Structure and Molecule objects."""

from __future__ import annotations

from typing import Any

import numpy as np

from xkcd_matchem.core import exceptions
from xkcd_matchem.core import structure


def _require_pymatgen() -> Any:
    """Lazily imports ``pymatgen.core.structure``."""
    try:
        from pymatgen.core import structure as pmg_structure
    except ImportError as exc:
        raise exceptions.OptionalDependencyError(
            "Pymatgen adapter requires 'pymatgen'. Install via: "
            "pip install 'xkcd-matchem[pymatgen]'"
        ) from exc
    return pmg_structure


def _site_symbol(site: Any) -> str:
    """Extracts primary element symbol from a pymatgen site."""
    specie = getattr(site, "specie", None)
    if specie is not None:
        return str(specie.symbol)
    species_map = getattr(site, "species", None)
    if species_map:
        best = max(species_map, key=lambda sp: float(species_map[sp]))
        return str(getattr(best, "symbol", best))
    raise exceptions.ParseError(f"Cannot extract element from site {site!r}.")


def from_pymatgen(pmg_obj: Any) -> structure.AtomsData:
    """Converts a pymatgen ``Structure`` or ``Molecule`` into ``AtomsData``."""
    symbols = [_site_symbol(site) for site in pmg_obj]
    positions = np.array(pmg_obj.cart_coords, dtype=np.float64)
    lattice = getattr(pmg_obj, "lattice", None)
    if lattice is not None:
        cell = np.array(lattice.matrix, dtype=np.float64)
        return structure.AtomsData(
            symbols=symbols,
            positions=positions,
            cell=cell,
            pbc=(True, True, True),
        )
    return structure.AtomsData(symbols=symbols, positions=positions)


def to_pymatgen(atoms: structure.AtomsData) -> Any:
    """Converts ``AtomsData`` into pymatgen ``Structure`` or ``Molecule``."""
    pmg_structure = _require_pymatgen()
    if atoms.cell is not None:
        lattice = pmg_structure.Lattice(atoms.cell)
        return pmg_structure.Structure(
            lattice=lattice,
            species=atoms.symbols,
            coords=atoms.positions,
            coords_are_cartesian=True,
        )
    return pmg_structure.Molecule(species=atoms.symbols, coords=atoms.positions)
