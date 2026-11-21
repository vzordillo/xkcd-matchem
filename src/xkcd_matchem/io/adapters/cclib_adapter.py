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
"""Lazy adapter wrapping cclib quantum-chemistry output parsing."""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np

from xkcd_matchem.core import electronic
from xkcd_matchem.core import exceptions
from xkcd_matchem.core import structure
from xkcd_matchem.data import elements


def _require_cclib() -> Any:
    """Lazily imports ``cclib.io``."""
    try:
        from cclib import io as cclib_io
    except ImportError as exc:
        raise exceptions.OptionalDependencyError(
            "cclib adapter requires 'cclib'. Install via: "
            "pip install 'xkcd-matchem[qm]'"
        ) from exc
    return cclib_io


def from_cclib(ccdata: Any) -> structure.AtomsData:
    """Converts a ``cclib`` parsed ``ccData`` object into ``AtomsData``."""
    atomnos = np.asarray(ccdata.atomnos, dtype=int)
    symbols = [elements.get_element(int(z)).symbol for z in atomnos]
    coords = np.asarray(ccdata.atomcoords[-1], dtype=np.float64)
    energies: dict[str, float] = {}
    scf = getattr(ccdata, "scfenergies", None)
    if scf is not None and len(scf) > 0:
        energies["total"] = float(scf[-1])
    meta: dict[str, Any] = {}
    moenergies = getattr(ccdata, "moenergies", None)
    homos = getattr(ccdata, "homos", None)
    if moenergies and homos is not None:
        mo_vals = np.asarray(moenergies[0], dtype=np.float64)
        occ = np.zeros_like(mo_vals)
        occ[: int(homos[0]) + 1] = 2.0
        meta["orbitals"] = electronic.MolecularOrbitals(
            energies=mo_vals, occupations=occ
        )
    return structure.AtomsData(
        symbols=symbols,
        positions=coords,
        energies=energies,
        metadata=meta,
    )


def read_via_cclib(path: str | pathlib.Path) -> structure.AtomsData:
    """Reads a computational chemistry logfile via ``cclib.io.ccread``."""
    cclib_io = _require_cclib()
    ccdata = cclib_io.ccread(str(path))
    if ccdata is None:
        raise exceptions.ParseError(f"cclib could not parse file: {path}")
    return from_cclib(ccdata)
