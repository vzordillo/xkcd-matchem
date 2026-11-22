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
"""Native parser and writer for XYZ and extended XYZ files."""

from __future__ import annotations

import pathlib
import re
from typing import Any

import numpy as np

from xkcd_matchem.core import exceptions
from xkcd_matchem.core import structure
from xkcd_matchem.core import trajectory
from xkcd_matchem.data import elements
from xkcd_matchem.io import registry


_KEYVAL_PATTERN = re.compile(r'(\w+)=(?:"([^"]*)"|(\S+))')


def _parse_extxyz_comment(comment: str) -> dict[str, Any]:
    """Extracts Lattice, pbc, and scalar metadata from an extXYZ header line."""
    meta: dict[str, Any] = {"comment": comment}
    for match in _KEYVAL_PATTERN.finditer(comment):
        key = match.group(1)
        val = match.group(2) if match.group(2) is not None else match.group(3)
        meta[key] = val
    return meta


def _parse_single_frame(
    lines: list[str],
    start: int,
) -> tuple[structure.AtomsData, int]:
    """Parses one XYZ or extXYZ frame starting at ``lines[start]``."""
    while start < len(lines) and not lines[start].strip():
        start += 1
    if start >= len(lines):
        raise exceptions.ParseError("Unexpected end of XYZ stream.")
    try:
        n_atoms = int(lines[start].strip().split()[0])
    except (ValueError, IndexError) as exc:
        raise exceptions.ParseError(
            f"Invalid XYZ atom count line at line {start + 1}: {lines[start]!r}"
        ) from exc
    if start + 1 + n_atoms >= len(lines):
        raise exceptions.ParseError(
            "Truncated XYZ frame before all atoms read."
        )
    comment = lines[start + 1].strip()
    meta = _parse_extxyz_comment(comment)
    cell: np.ndarray | None = None
    pbc = (False, False, False)
    if "Lattice" in meta:
        nums = [float(x) for x in meta["Lattice"].split()]
        if len(nums) == 9:
            cell = np.array(nums, dtype=np.float64).reshape((3, 3))
            pbc = (True, True, True)
    if "pbc" in meta:
        flags = meta["pbc"].replace(",", " ").split()
        if len(flags) == 3:
            pbc = (
                flags[0].upper() in ("T", "TRUE", "1"),
                flags[1].upper() in ("T", "TRUE", "1"),
                flags[2].upper() in ("T", "TRUE", "1"),
            )

    symbols: list[str] = []
    coords = np.zeros((n_atoms, 3), dtype=np.float64)
    for idx in range(n_atoms):
        parts = lines[start + 2 + idx].split()
        if len(parts) < 4:
            raise exceptions.ParseError(
                f"Malformed XYZ atom line at {start + 3 + idx}: {parts!r}"
            )
        raw_sym = parts[0]
        try:
            sym = (
                elements.get_element(int(raw_sym)).symbol
                if raw_sym.strip().isdigit()
                else elements.get_element(raw_sym).symbol
            )
            coords[idx] = [float(parts[1]), float(parts[2]), float(parts[3])]
        except (exceptions.StructureError, ValueError) as exc:
            raise exceptions.ParseError(
                f"Invalid XYZ atom record at line {start + 3 + idx}: {parts!r}"
            ) from exc
        symbols.append(sym)

    energies: dict[str, float] = {}
    for ekey in ("energy", "Energy", "free_energy"):
        if ekey in meta:
            try:
                energies["total"] = float(meta[ekey])
                break
            except ValueError:
                pass
    if "total" not in energies:
        energy_match = re.search(
            r"energy\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", comment
        )
        if energy_match:
            energies["total"] = float(energy_match.group(1))

    atoms = structure.AtomsData(
        symbols=symbols,
        positions=coords,
        cell=cell,
        pbc=pbc,
        energies=energies,
        metadata=meta,
    )
    return atoms, start + 2 + n_atoms


def read_xyz_trajectory(path: str | pathlib.Path) -> trajectory.Trajectory:
    """Reads all frames from an XYZ or extended XYZ file into ``Trajectory``."""
    p = pathlib.Path(path)
    lines = p.read_text(encoding="utf-8").splitlines()
    frames: list[structure.AtomsData] = []
    cursor = 0
    while cursor < len(lines):
        while cursor < len(lines) and not lines[cursor].strip():
            cursor += 1
        if cursor >= len(lines):
            break
        frame, cursor = _parse_single_frame(lines, cursor)
        frames.append(frame)
    if not frames:
        raise exceptions.ParseError(f"No XYZ frames found in {p}.")
    times: np.ndarray | None = None
    temps: np.ndarray | None = None
    if all("time" in frame.metadata for frame in frames):
        try:
            times = np.array(
                [float(frame.metadata["time"]) for frame in frames],
                dtype=np.float64,
            )
        except ValueError:
            times = None
    if all("temperature" in frame.metadata for frame in frames):
        try:
            temps = np.array(
                [float(frame.metadata["temperature"]) for frame in frames],
                dtype=np.float64,
            )
        except ValueError:
            temps = None
    return trajectory.Trajectory(
        frames=frames,
        times=times,
        temperatures=temps,
        metadata={"source": str(p)},
    )


def read_xyz(path: str | pathlib.Path) -> structure.AtomsData:
    """Reads only the first structure frame from an XYZ/extXYZ file."""
    p = pathlib.Path(path)
    lines = p.read_text(encoding="utf-8").splitlines()
    cursor = 0
    while cursor < len(lines) and not lines[cursor].strip():
        cursor += 1
    if cursor >= len(lines):
        raise exceptions.ParseError(f"No XYZ frames found in {p}.")
    frame, _ = _parse_single_frame(lines, cursor)
    frame.metadata.setdefault("source", str(p))
    return frame


def write_xyz(atoms: structure.AtomsData, path: str | pathlib.Path) -> None:
    """Writes an ``AtomsData`` structure in extended XYZ format."""
    p = pathlib.Path(path)
    comment_parts: list[str] = []
    if atoms.cell is not None:
        flat = " ".join(f"{val:.8f}" for val in atoms.cell.ravel())
        comment_parts.append(f'Lattice="{flat}"')
        pbc_str = " ".join("T" if flag else "F" for flag in atoms.pbc)
        comment_parts.append(f'pbc="{pbc_str}"')
    if "total" in atoms.energies:
        comment_parts.append(f"energy={atoms.energies['total']:.8f}")
    comment_parts.append("Properties=species:S:1:pos:R:3")
    lines = [str(len(atoms)), " ".join(comment_parts)]
    for sym, pos in zip(atoms.symbols, atoms.positions, strict=True):
        lines.append(f"{sym:<4s} {pos[0]:15.8f} {pos[1]:15.8f} {pos[2]:15.8f}")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


registry.register_format(
    registry.FormatSpec(
        name="xyz",
        extensions=(".xyz", ".extxyz"),
        filenames=(),
        reader=read_xyz,
        writer=write_xyz,
        description="XYZ and extended XYZ molecular/periodic coordinates",
    )
)
