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

from xkcd_matchem.core import bonding
from xkcd_matchem.core import exceptions
from xkcd_matchem.core import geometry
from xkcd_matchem.core import structure
from xkcd_matchem.core import trajectory
from xkcd_matchem.data import elements
from xkcd_matchem.io import registry


def _extract_pdb_element(line: str) -> str:
    """Extracts the element symbol from columns 77-78 or fallback atom name."""
    raw_elem = line[76:78].strip() if len(line) >= 78 else ""
    if raw_elem:
        return elements.get_element(raw_elem).symbol
    atom_field = line[12:16] if len(line) >= 16 else line[12:]
    # Cols 13-14 hold left-justified 2-letter elements ("FE  ", "MG  ");
    # standard protein C/N/O/S/H atoms start at col 14 (" CA ", " CD1", " HG ").
    if (
        len(atom_field) >= 2
        and atom_field[0].isalpha()
        and atom_field[1].isalpha()
        and atom_field[0].upper() not in ("H", "D")
        and atom_field[:2].capitalize() in elements.ELEMENTS_BY_SYMBOL
    ):
        return elements.get_element(atom_field[:2].capitalize()).symbol
    cleaned = re.sub(r"[^A-Za-z]", "", atom_field.strip())
    if cleaned:
        return elements.get_element(cleaned[0].upper()).symbol
    raise exceptions.ParseError(
        f"Cannot infer PDB element symbol from {line!r}"
    )


def _parse_conect_serials(
    line: str,
    conect_serials: set[tuple[int, int]],
) -> None:
    """Collects unique atom serial pairs from a PDB ``CONECT`` line."""
    fields = [
        int(line[k : k + 5])
        for k in range(6, min(len(line), 31), 5)
        if line[k : k + 5].strip()
    ]
    if len(fields) < 2:
        return
    src = fields[0]
    for dst in fields[1:]:
        if src != dst:
            conect_serials.add((min(src, dst), max(src, dst)))


def _build_pdb_frame(
    raw_frame: tuple[list[str], list[list[float]], list[str], dict[int, int]],
    conect_serials: set[tuple[int, int]],
    cell: np.ndarray | None,
    pbc: tuple[bool, bool, bool],
    source: str,
) -> structure.AtomsData:
    """Constructs one ``AtomsData`` frame resolving CONECT bonds & orders."""
    syms, coords, labels, serial_to_idx = raw_frame
    pairs: set[tuple[int, int]] = set()
    for ser_a, ser_b in conect_serials:
        if ser_a in serial_to_idx and ser_b in serial_to_idx:
            i_idx = serial_to_idx[ser_a]
            j_idx = serial_to_idx[ser_b]
            if i_idx != j_idx:
                pairs.add((min(i_idx, j_idx), max(i_idx, j_idx)))
    bonds = [
        structure.Bond(i=i, j=j, order=1.0) for i, j in sorted(pairs)
    ]
    frame = structure.AtomsData(
        symbols=list(syms),
        positions=np.array(coords, dtype=np.float64),
        cell=cell,
        pbc=pbc,
        bonds=bonds,
        atom_labels=list(labels),
        metadata={"source": source},
    )
    if frame.bonds:
        frame.bonds = bonding.infer_bond_orders(frame, frame.bonds)
    return frame


def read_pdb_trajectory(
    path: str | pathlib.Path,
    *,
    max_frames: int | None = None,
) -> trajectory.Trajectory:
    """Reads single- or multi-MODEL PDB structures into a ``Trajectory``."""
    p = pathlib.Path(path)
    lines = p.read_text(encoding="utf-8").splitlines()
    cell: np.ndarray | None = None
    pbc = (False, False, False)
    raw_frames: list[
        tuple[list[str], list[list[float]], list[str], dict[int, int]]
    ] = []
    cur_symbols: list[str] = []
    cur_coords: list[list[float]] = []
    cur_labels: list[str] = []
    serial_to_idx: dict[int, int] = {}
    conect_serials: set[tuple[int, int]] = set()

    def stash_frame() -> None:
        if cur_symbols and (
            max_frames is None or len(raw_frames) < max_frames
        ):
            raw_frames.append(
                (
                    list(cur_symbols),
                    list(cur_coords),
                    list(cur_labels),
                    dict(serial_to_idx),
                )
            )
        cur_symbols.clear()
        cur_coords.clear()
        cur_labels.clear()
        serial_to_idx.clear()

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
            if max_frames is None or len(raw_frames) < max_frames:
                serial_to_idx[int(line[6:11])] = len(cur_symbols)
                cur_symbols.append(_extract_pdb_element(line))
                cur_coords.append(
                    [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                )
                cur_labels.append(line[12:16].strip())
        elif rec == "CONECT":
            _parse_conect_serials(line, conect_serials)
        elif rec == "ENDMDL":
            stash_frame()

    stash_frame()
    if not raw_frames:
        raise exceptions.ParseError(f"No ATOM/HETATM records found in PDB: {p}")
    frames = [
        _build_pdb_frame(rf, conect_serials, cell, pbc, str(p))
        for rf in raw_frames
    ]
    return trajectory.Trajectory(frames=frames, metadata={"source": str(p)})


def read_pdb(path: str | pathlib.Path) -> structure.AtomsData:
    """Reads the first structure frame from a PDB file."""
    return read_pdb_trajectory(path, max_frames=1)[0]


registry.register_format(
    registry.FormatSpec(
        name="pdb",
        extensions=(".pdb", ".ent"),
        filenames=(),
        reader=read_pdb,
        description="Protein Data Bank (PDB) structure and CONECT bonds",
    )
)
