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
    raw_atomnos = getattr(ccdata, "atomnos", None)
    raw_coords = getattr(ccdata, "atomcoords", None)
    if raw_atomnos is None or raw_coords is None or len(raw_coords) == 0:
        raise exceptions.ParseError("cclib data lacks atomic coordinates.")
    atomnos = np.asarray(raw_atomnos, dtype=int)
    symbols = [elements.get_element(int(z)).symbol for z in atomnos]
    coords = np.asarray(raw_coords[-1], dtype=np.float64)
    energies: dict[str, float] = {}
    scf = getattr(ccdata, "scfenergies", None)
    if scf is not None and len(scf) > 0:
        energies["total"] = float(scf[-1])
    meta: dict[str, Any] = {}
    moenergies = getattr(ccdata, "moenergies", None)
    homos = getattr(ccdata, "homos", None)
    if (
        moenergies is not None
        and len(moenergies) > 0
        and homos is not None
        and len(homos) > 0
    ):
        if len(moenergies) >= 2 and len(homos) >= 2:
            e_a = np.asarray(moenergies[0], dtype=np.float64)
            e_b = np.asarray(moenergies[1], dtype=np.float64)
            occ_a = np.zeros_like(e_a)
            occ_b = np.zeros_like(e_b)
            if int(homos[0]) >= 0:
                occ_a[: int(homos[0]) + 1] = 1.0
            if int(homos[1]) >= 0:
                occ_b[: int(homos[1]) + 1] = 1.0
            meta["orbitals"] = electronic.MolecularOrbitals(
                energies=np.concatenate([e_a, e_b]),
                occupations=np.concatenate([occ_a, occ_b]),
                spin=["alpha"] * len(e_a) + ["beta"] * len(e_b),
            )
        else:
            mo_vals = np.asarray(moenergies[0], dtype=np.float64)
            occ = np.zeros_like(mo_vals)
            if int(homos[0]) >= 0:
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
