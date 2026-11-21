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
"""Native parsers for V2000 SDF/MOL and Tripos MOL2 structure files."""

from __future__ import annotations

import pathlib

import numpy as np

from xkcd_matchem.core import exceptions
from xkcd_matchem.core import structure
from xkcd_matchem.data import elements
from xkcd_matchem.io import registry


def read_sdf(path: str | pathlib.Path) -> structure.AtomsData:
    """Reads an MDL V2000 ``.sdf`` or ``.mol`` file preserving bond orders."""
    p = pathlib.Path(path)
    lines = p.read_text(encoding="utf-8").splitlines()
    if len(lines) < 5:
        raise exceptions.ParseError(f"Truncated SDF/MOL file: {p}")
    title = lines[0].strip()
    counts = lines[3]
    try:
        n_atoms = int(counts[0:3])
        n_bonds = int(counts[3:6])
    except ValueError as exc:
        parts = counts.split()
        if len(parts) < 2:
            raise exceptions.ParseError(
                f"Invalid SDF counts line: {counts!r}"
            ) from exc
        n_atoms, n_bonds = int(parts[0]), int(parts[1])

    if len(lines) < 4 + n_atoms + n_bonds:
        raise exceptions.ParseError(f"Truncated SDF atom/bond tables in {p}.")
    symbols: list[str] = []
    coords = np.zeros((n_atoms, 3), dtype=np.float64)
    for k in range(n_atoms):
        parts = lines[4 + k].split()
        coords[k] = [float(parts[0]), float(parts[1]), float(parts[2])]
        symbols.append(elements.get_element(parts[3]).symbol)

    order_map = {1: 1.0, 2: 2.0, 3: 3.0, 4: 1.5}
    bonds: list[structure.Bond] = []
    for k in range(n_bonds):
        line = lines[4 + n_atoms + k]
        try:
            i_idx = int(line[0:3]) - 1
            j_idx = int(line[3:6]) - 1
            raw_ord = int(line[6:9])
        except ValueError:
            parts = line.split()
            i_idx = int(parts[0]) - 1
            j_idx = int(parts[1]) - 1
            raw_ord = int(parts[2])
        bonds.append(
            structure.Bond(
                i=i_idx,
                j=j_idx,
                order=order_map.get(raw_ord, float(raw_ord)),
            )
        )
    return structure.AtomsData(
        symbols=symbols,
        positions=coords,
        bonds=bonds,
        metadata={"title": title, "source": str(p)},
    )


def read_mol2(path: str | pathlib.Path) -> structure.AtomsData:
    """Reads a Tripos ``.mol2`` structure with explicit bonds and charges."""
    p = pathlib.Path(path)
    lines = p.read_text(encoding="utf-8").splitlines()
    section = ""
    symbols: list[str] = []
    coords: list[list[float]] = []
    charges: list[float] = []
    bonds: list[structure.Bond] = []
    order_map = {"1": 1.0, "2": 2.0, "3": 3.0, "ar": 1.5, "am": 1.0}

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("@<TRIPOS>"):
            section = line
            continue
        parts = line.split()
        if section == "@<TRIPOS>ATOM" and len(parts) >= 6:
            coords.append([float(parts[2]), float(parts[3]), float(parts[4])])
            atype = parts[5].split(".", maxsplit=1)[0]
            symbols.append(elements.get_element(atype).symbol)
            charges.append(float(parts[8]) if len(parts) >= 9 else 0.0)
        elif section == "@<TRIPOS>BOND" and len(parts) >= 4:
            bonds.append(
                structure.Bond(
                    i=int(parts[1]) - 1,
                    j=int(parts[2]) - 1,
                    order=order_map.get(parts[3].lower(), 1.0),
                )
            )
    if not symbols:
        raise exceptions.ParseError(f"No MOL2 atoms found in {p}.")
    return structure.AtomsData(
        symbols=symbols,
        positions=np.array(coords, dtype=np.float64),
        charges=np.array(charges, dtype=np.float64),
        bonds=bonds,
        metadata={"source": str(p)},
    )


registry.register_format(
    registry.FormatSpec(
        name="sdf",
        extensions=(".sdf", ".mol"),
        filenames=(),
        reader=read_sdf,
        description="MDL V2000 SDF/MOL file with explicit bond orders",
    )
)
registry.register_format(
    registry.FormatSpec(
        name="mol2",
        extensions=(".mol2",),
        filenames=(),
        reader=read_mol2,
        description="Tripos MOL2 structure and bond table",
    )
)
