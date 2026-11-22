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
"""Native parsers for CP2K .ener files and xTB output summaries."""

from __future__ import annotations

import pathlib
import re

import numpy as np

from xkcd_matchem.core import exceptions
from xkcd_matchem.core import spectroscopy
from xkcd_matchem.core import structure
from xkcd_matchem.data import elements
from xkcd_matchem.io import registry


def read_cp2k_ener(path: str | pathlib.Path) -> spectroscopy.Spectrum:
    """Reads a CP2K ``.ener`` MD energy trace into ``Spectrum`` (time vs E)."""
    p = pathlib.Path(path)
    rows: list[list[float]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        rows.append([float(tok) for tok in stripped.split()])
    if not rows:
        raise exceptions.ParseError(
            f"No data rows found in CP2K .ener file: {p}"
        )
    arr = np.array(rows, dtype=np.float64)
    times_fs = arr[:, 1]
    pot_ev = arr[:, 4] * elements.CODATA2022.hartree_to_ev
    return spectroscopy.Spectrum(
        x=times_fs,
        y=pot_ev,
        kind="generic",
        x_label="Time (fs)",
        y_label="Potential Energy (eV)",
        metadata={"source": str(p), "raw": arr},
    )


def _parse_xtb_xyz_block(
    lines: list[str], start: int
) -> tuple[list[str], list[list[float]]]:
    """Extracts an inline XYZ coordinate block from an xTB output."""
    try:
        n_atoms = int(lines[start].strip().split()[0])
    except (ValueError, IndexError):
        return [], []
    if n_atoms <= 0 or start + 1 + n_atoms >= len(lines):
        return [], []
    syms: list[str] = []
    coords: list[list[float]] = []
    for k in range(n_atoms):
        parts = lines[start + 2 + k].split()
        if len(parts) < 4:
            return [], []
        try:
            sym = elements.get_element(parts[0]).symbol
            xyz = [float(parts[1]), float(parts[2]), float(parts[3])]
        except (exceptions.StructureError, ValueError):
            return [], []
        syms.append(sym)
        coords.append(xyz)
    return syms, coords


def _parse_xtb_coord_block(
    lines: list[str], start: int
) -> tuple[list[str], list[list[float]]]:
    """Extracts a Turbomole ``$coord`` block (Bohr) from an xTB output."""
    bohr = elements.CODATA2022.bohr_to_angstrom
    syms: list[str] = []
    coords: list[list[float]] = []
    idx = start + 1
    while idx < len(lines):
        row = lines[idx].strip()
        if row.startswith("$"):
            break
        parts = row.split()
        if len(parts) >= 4:
            try:
                xyz = [
                    float(parts[0]) * bohr,
                    float(parts[1]) * bohr,
                    float(parts[2]) * bohr,
                ]
                sym = elements.get_element(parts[3]).symbol
            except (exceptions.StructureError, ValueError):
                break
            syms.append(sym)
            coords.append(xyz)
        idx += 1
    return syms, coords


def read_xtb_out(path: str | pathlib.Path) -> structure.AtomsData:
    """Reads geometry, total energy, and HOMO-LUMO gap from xTB output."""
    p = pathlib.Path(path)
    text = p.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    energies: dict[str, float] = {}
    m_tot = re.search(r"TOTAL ENERGY\s+([-+]?\d+\.\d+)\s+Eh", text)
    if m_tot:
        energies["total"] = (
            float(m_tot.group(1)) * elements.CODATA2022.hartree_to_ev
        )
    m_gap = re.search(r"HOMO-LUMO GAP\s+([-+]?\d+\.\d+)\s+eV", text)
    meta: dict[str, object] = {"source": str(p)}
    if m_gap:
        meta["homo_lumo_gap_ev"] = float(m_gap.group(1))

    syms: list[str] = []
    coords: list[list[float]] = []
    for idx, line in enumerate(lines):
        low = line.strip().lower()
        if "final structure:" in low:
            for probe in range(idx + 1, min(idx + 5, len(lines))):
                if lines[probe].strip().startswith("$coord"):
                    syms, coords = _parse_xtb_coord_block(lines, probe)
                    break
                if lines[probe].strip().isdigit():
                    syms, coords = _parse_xtb_xyz_block(lines, probe)
                    break
        elif low.startswith("$coord") and not syms:
            syms, coords = _parse_xtb_coord_block(lines, idx)

    if not syms and not energies and "homo_lumo_gap_ev" not in meta:
        raise exceptions.ParseError(f"Could not extract xTB summary from {p}.")
    pos = (
        np.array(coords, dtype=np.float64)
        if coords
        else np.zeros((0, 3), dtype=np.float64)
    )
    return structure.AtomsData(
        symbols=syms,
        positions=pos,
        energies=energies,
        metadata=meta,
    )


registry.register_format(
    registry.FormatSpec(
        name="cp2k_ener",
        extensions=(".ener",),
        filenames=(),
        reader=read_cp2k_ener,
        description="CP2K .ener molecular dynamics energy trace",
    )
)
registry.register_format(
    registry.FormatSpec(
        name="xtb_out",
        extensions=(),
        filenames=("xtb.out",),
        reader=read_xtb_out,
        description="xTB calculation output energy and gap summary",
    )
)
