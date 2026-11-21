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
"""Native parser for Protein Data Bank (PDB) coordinate and CONECT records."""

from __future__ import annotations

import pathlib
import re

import numpy as np

from xkcd_matchem.core import exceptions
from xkcd_matchem.core import geometry
from xkcd_matchem.core import structure
from xkcd_matchem.core import trajectory
from xkcd_matchem.data import elements
from xkcd_matchem.io import registry


def _extract_pdb_element(line: str) -> str:
    """Extracts the element symbol from columns 77-78 or fallback atom name."""
    raw_elem = line[76:78].strip() if len(line) >= 78 else ""
    if not raw_elem:
        atom_name = line[12:16].strip()
        raw_elem = re.sub(r"[^A-Za-z]", "", atom_name)
        if (
            len(raw_elem) >= 2
            and raw_elem[:2].capitalize() in elements.ELEMENTS_BY_SYMBOL
        ):
            raw_elem = raw_elem[:2]
        elif raw_elem:
            raw_elem = raw_elem[0]
    return elements.get_element(raw_elem).symbol


def _parse_conect_line(
    line: str,
    serial_to_idx: dict[int, int],
    conect_pairs: set[tuple[int, int]],
) -> None:
    """Adds unique atom index pairs from a PDB ``CONECT`` line."""
    fields = [
        int(line[k : k + 5])
        for k in range(6, min(len(line), 31), 5)
        if line[k : k + 5].strip()
    ]
    if not fields or fields[0] not in serial_to_idx:
        return
    i_idx = serial_to_idx[fields[0]]
    for other in fields[1:]:
        if other in serial_to_idx:
            j_idx = serial_to_idx[other]
            if i_idx != j_idx:
                conect_pairs.add((min(i_idx, j_idx), max(i_idx, j_idx)))


def read_pdb_trajectory(path: str | pathlib.Path) -> trajectory.Trajectory:
    """Reads single- or multi-MODEL PDB structures into a ``Trajectory``."""
    p = pathlib.Path(path)
    lines = p.read_text(encoding="utf-8").splitlines()
    cell: np.ndarray | None = None
    pbc = (False, False, False)
    frames: list[structure.AtomsData] = []
    cur_symbols: list[str] = []
    cur_coords: list[list[float]] = []
    cur_labels: list[str] = []
    serial_to_idx: dict[int, int] = {}
    conect_pairs: set[tuple[int, int]] = set()

    def flush_frame() -> None:
        if not cur_symbols:
            return
        bonds = [
            structure.Bond(i=i, j=j, order=1.0)
            for i, j in sorted(conect_pairs)
            if i < len(cur_symbols) and j < len(cur_symbols)
        ]
        frames.append(
            structure.AtomsData(
                symbols=list(cur_symbols),
                positions=np.array(cur_coords, dtype=np.float64),
                cell=cell,
                pbc=pbc,
                bonds=bonds,
                atom_labels=list(cur_labels),
                metadata={"source": str(p)},
            )
        )

    for line in lines:
        rec = line[:6].strip()
        if rec == "CRYST1" and len(line) >= 54:
            cell = geometry.cell_vectors_from_parameters(
                float(line[6:15]),
                float(line[15:24]),
                float(line[24:33]),
                float(line[33:40]),
                float(line[40:47]),
                float(line[47:54]),
            )
            pbc = (True, True, True)
        elif rec in ("ATOM", "HETATM"):
            serial_to_idx[int(line[6:11])] = len(cur_symbols)
            cur_symbols.append(_extract_pdb_element(line))
            cur_coords.append(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            )
            cur_labels.append(line[12:16].strip())
        elif rec == "CONECT":
            _parse_conect_line(line, serial_to_idx, conect_pairs)
        elif rec == "ENDMDL":
            flush_frame()
            cur_symbols.clear()
            cur_coords.clear()
            cur_labels.clear()
            serial_to_idx.clear()

    flush_frame()
    if not frames:
        raise exceptions.ParseError(f"No ATOM/HETATM records found in PDB: {p}")
    return trajectory.Trajectory(frames=frames, metadata={"source": str(p)})


def read_pdb(path: str | pathlib.Path) -> structure.AtomsData:
    """Reads the first structure frame from a PDB file."""
    return read_pdb_trajectory(path)[0]


registry.register_format(
    registry.FormatSpec(
        name="pdb",
        extensions=(".pdb", ".ent"),
        filenames=(),
        reader=read_pdb,
        description="Protein Data Bank (PDB) structure and CONECT bonds",
    )
)
