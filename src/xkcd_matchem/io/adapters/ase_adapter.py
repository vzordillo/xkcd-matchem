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
"""Lazy adapter converting between ASE Atoms and xkcd-matchem AtomsData."""

from __future__ import annotations

import contextlib
import pathlib
from typing import Any

import numpy as np

from xkcd_matchem.core import exceptions
from xkcd_matchem.core import structure


def _require_ase() -> tuple[Any, Any]:
    """Lazily imports ``ase`` and ``ase.io``."""
    try:
        import ase
        from ase import io as ase_io
    except ImportError as exc:
        raise exceptions.OptionalDependencyError(
            "ASE adapter requires 'ase'. Install via: "
            "pip install 'xkcd-matchem[ase]'"
        ) from exc
    return ase, ase_io


def from_ase(ase_atoms: Any) -> structure.AtomsData:
    """Converts an ``ase.Atoms`` instance into ``AtomsData``."""
    symbols = list(ase_atoms.get_chemical_symbols())
    positions = np.array(ase_atoms.get_positions(), dtype=np.float64)
    pbc = tuple(bool(x) for x in ase_atoms.get_pbc())
    cell_arr = np.array(ase_atoms.get_cell().array, dtype=np.float64)
    has_cell = bool(np.any(np.abs(cell_arr) > 1e-8))
    energies: dict[str, float] = {}
    if getattr(ase_atoms, "calc", None) is not None:
        with contextlib.suppress(RuntimeError):
            energies["total"] = float(ase_atoms.get_potential_energy())
    return structure.AtomsData(
        symbols=symbols,
        positions=positions,
        cell=cell_arr if has_cell else None,
        pbc=(pbc[0], pbc[1], pbc[2]),
        energies=energies,
        metadata=dict(getattr(ase_atoms, "info", {})),
    )


def to_ase(atoms: structure.AtomsData) -> Any:
    """Converts ``AtomsData`` into an ``ase.Atoms`` instance."""
    ase, _ = _require_ase()
    cell = (
        atoms.cell
        if atoms.cell is not None
        else np.zeros((3, 3), dtype=np.float64)
    )
    return ase.Atoms(
        symbols=atoms.symbols,
        positions=atoms.positions,
        cell=cell,
        pbc=atoms.pbc,
    )


def read_via_ase(
    path: str | pathlib.Path,
    index: int | str = -1,
) -> structure.AtomsData:
    """Reads any ASE-supported file format using ``ase.io.read``."""
    _, ase_io = _require_ase()
    loaded = ase_io.read(str(path), index=index)
    return from_ase(loaded)
