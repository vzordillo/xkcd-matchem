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
"""Native Quantum ESPRESSO parsers (pw.in, pw.out, bands.dat, dos.dat)."""

from __future__ import annotations

import pathlib
import re

import numpy as np

from xkcd_matchem.core import electronic
from xkcd_matchem.core import exceptions
from xkcd_matchem.core import structure
from xkcd_matchem.data import elements
from xkcd_matchem.io import registry


_RY_TO_EV = 0.5 * elements.CODATA2022.hartree_to_ev


def _clean_qe_symbol(label: str) -> str:
    """Strips numeric suffixes or modifiers from QE atomic species tags."""
    sym = re.sub(r"[^A-Za-z]", "", label.strip())
    return elements.get_element(
        sym[:2]
        if len(sym) >= 2 and sym[:2].capitalize() in elements.ELEMENTS_BY_SYMBOL
        else sym[:1]
    ).symbol


def read_qe_in(path: str | pathlib.Path) -> structure.AtomsData:
    """Reads a Quantum ESPRESSO ``pw.in`` input script into ``AtomsData``."""
    p = pathlib.Path(path)
    text = p.read_text(encoding="utf-8")
    lines = text.splitlines()
    bohr = elements.CODATA2022.bohr_to_angstrom
    celldm1 = 1.0
    m_cd = re.search(r"celldm\s*\(\s*1\s*\)\s*=\s*([0-9.eEdD+-]+)", text)
    if m_cd:
        celldm1 = float(m_cd.group(1).replace("d", "e").replace("D", "E"))

    cell: np.ndarray | None = None
    syms: list[str] = []
    coords: list[list[float]] = []
    pos_unit = "alat"
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        upper = line.upper()
        if upper.startswith("CELL_PARAMETERS"):
            scale = (
                bohr * celldm1
                if "ALAT" in upper or "(" not in upper
                else (bohr if "BOHR" in upper else 1.0)
            )
            cell = (
                np.array(
                    [
                        [float(x) for x in lines[idx + 1 + k].split()[:3]]
                        for k in range(3)
                    ],
                    dtype=np.float64,
                )
                * scale
            )
            idx += 4
            continue
        if upper.startswith("ATOMIC_POSITIONS"):
            if "CRYSTAL" in upper:
                pos_unit = "crystal"
            elif "ANGSTROM" in upper:
                pos_unit = "angstrom"
            elif "BOHR" in upper:
                pos_unit = "bohr"
            else:
                pos_unit = "alat"
            idx += 1
            while idx < len(lines):
                row = lines[idx].strip()
                if not row or row.upper().startswith(
                    ("K_POINTS", "CELL_", "ATOMIC_", "CONSTRAINTS")
                ):
                    break
                parts = row.split()
                if len(parts) >= 4:
                    syms.append(_clean_qe_symbol(parts[0]))
                    coords.append(
                        [float(parts[1]), float(parts[2]), float(parts[3])]
                    )
                idx += 1
            continue
        idx += 1

    if not syms:
        raise exceptions.ParseError(
            f"No ATOMIC_POSITIONS card in QE input: {p}"
        )
    arr = np.array(coords, dtype=np.float64)
    if pos_unit == "crystal":
        if cell is None:
            raise exceptions.ParseError(
                "QE crystal coordinates require CELL_PARAMETERS."
            )
        return structure.AtomsData.from_fractional(
            symbols=syms,
            frac_positions=arr,
            cell=cell,
            metadata={"source": str(p)},
        )
    if pos_unit == "bohr":
        arr *= bohr
    elif pos_unit == "alat":
        arr *= celldm1 * bohr
    return structure.AtomsData(
        symbols=syms,
        positions=arr,
        cell=cell,
        pbc=(cell is not None, cell is not None, cell is not None),
        metadata={"source": str(p)},
    )


def read_qe_out(path: str | pathlib.Path) -> structure.AtomsData:
    """Reads converged geometry, cell, and Fermi level from ``pw.out``."""
    p = pathlib.Path(path)
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    bohr = elements.CODATA2022.bohr_to_angstrom
    alat_bohr = 1.0
    cell: np.ndarray | None = None
    syms: list[str] = []
    coords: list[list[float]] = []
    last_energy: float | None = None
    fermi: float | None = None

    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if "lattice parameter (alat)" in line:
            alat_bohr = float(line.split("=")[1].split()[0])
        elif "a(1) =" in line and idx + 2 < len(lines):
            rows: list[list[float]] = []
            for k in range(3):
                vals = lines[idx + k].split("(")[2].split(")")[0].split()
                rows.append([float(x) for x in vals[:3]])
            cell = np.array(rows, dtype=np.float64) * alat_bohr * bohr
        elif "CELL_PARAMETERS" in line and idx + 3 < len(lines):
            scale = (
                bohr
                if "bohr" in line.lower()
                else (alat_bohr * bohr if "alat" in line.lower() else 1.0)
            )
            cell = (
                np.array(
                    [
                        [float(x) for x in lines[idx + 1 + k].split()[:3]]
                        for k in range(3)
                    ],
                    dtype=np.float64,
                )
                * scale
            )
        elif "Cartesian axes" in line and idx + 3 < len(lines):
            k = idx + 3
            tmp_s: list[str] = []
            tmp_c: list[list[float]] = []
            while k < len(lines) and "tau(" in lines[k]:
                parts = lines[k].split()
                tmp_s.append(_clean_qe_symbol(parts[1]))
                tau_part = lines[k].split("tau(")[1].split("=")[1]
                cvals = tau_part.replace("(", " ").replace(")", " ").split()
                tmp_c.append(
                    [float(cvals[0]), float(cvals[1]), float(cvals[2])]
                )
                k += 1
            if tmp_s:
                syms = tmp_s
                coords = (
                    np.array(tmp_c, dtype=np.float64) * alat_bohr * bohr
                ).tolist()
        elif "ATOMIC_POSITIONS" in line:
            scale = (
                bohr
                if "bohr" in line.lower()
                else (1.0 if "angstrom" in line.lower() else alat_bohr * bohr)
            )
            is_crys = "crystal" in line.lower()
            k = idx + 1
            tmp_s = []
            tmp_c = []
            while (
                k < len(lines)
                and lines[k].strip()
                and not lines[k].strip().startswith("End")
            ):
                parts = lines[k].split()
                if len(parts) < 4:
                    break
                tmp_s.append(_clean_qe_symbol(parts[0]))
                tmp_c.append(
                    [float(parts[1]), float(parts[2]), float(parts[3])]
                )
                k += 1
            if tmp_s:
                syms = tmp_s
                arr_tmp = np.array(tmp_c, dtype=np.float64)
                coords = (
                    (arr_tmp @ cell).tolist()
                    if (is_crys and cell is not None)
                    else (arr_tmp * scale).tolist()
                )
        elif "!" in line and "total energy" in line:
            last_energy = float(line.split("=")[1].split()[0]) * _RY_TO_EV
        elif "the Fermi energy is" in line:
            fermi = float(line.split("is")[1].split()[0])
        idx += 1

    if not syms:
        raise exceptions.ParseError(f"No structure found in QE pw.out: {p}")
    energies: dict[str, float] = {}
    if last_energy is not None:
        energies["total"] = last_energy
    meta: dict[str, object] = {"source": str(p)}
    if fermi is not None:
        meta["fermi_energy"] = fermi
    return structure.AtomsData(
        symbols=syms,
        positions=np.array(coords, dtype=np.float64),
        cell=cell,
        pbc=(cell is not None, cell is not None, cell is not None),
        energies=energies,
        metadata=meta,
    )


def read_qe_bands(path: str | pathlib.Path) -> electronic.BandStructure:
    """Reads a Quantum ESPRESSO ``bands.x`` plain-text output file."""
    p = pathlib.Path(path)
    lines = p.read_text(encoding="utf-8").splitlines()
    header = lines[0]
    m_nb = re.search(r"nbnd\s*=\s*(\d+)", header)
    m_nk = re.search(r"nks\s*=\s*(\d+)", header)
    if not m_nb or not m_nk:
        raise exceptions.ParseError(f"Invalid QE bands header in {p}.")
    nbnd = int(m_nb.group(1))
    nks = int(m_nk.group(1))
    tokens = " ".join(lines[1:]).replace("/", " ").split()
    if not m_nb or len(tokens) < nks * (3 + nbnd):
        # Parse k-point + eigenvalues sequentially cleanly per block.
        tokens = [tok for tok in " ".join(lines[1:]).split() if tok != "/"]
    k_coords = np.zeros((nks, 3), dtype=np.float64)
    eig = np.zeros((1, nks, nbnd), dtype=np.float64)
    cursor = 0
    for ik in range(nks):
        k_coords[ik] = [float(tokens[cursor + j]) for j in range(3)]
        cursor += 3
        eig[0, ik, :] = [float(tokens[cursor + j]) for j in range(nbnd)]
        cursor += nbnd
    diffs = np.diff(k_coords, axis=0)
    k_dist = np.concatenate([[0.0], np.cumsum(np.linalg.norm(diffs, axis=1))])
    return electronic.BandStructure(
        k_distances=k_dist, eigenvalues=eig, metadata={"source": str(p)}
    )


def read_qe_dos(path: str | pathlib.Path) -> electronic.DensityOfStates:
    """Reads a Quantum ESPRESSO ``dos.x`` output table."""
    p = pathlib.Path(path)
    lines = p.read_text(encoding="utf-8").splitlines()
    fermi = 0.0
    rows: list[list[float]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            m_ef = re.search(r"EFermi\s*=\s*([-+]?\d*\.?\d+)", stripped)
            if m_ef:
                fermi = float(m_ef.group(1))
            continue
        rows.append([float(tok) for tok in stripped.split()])
    arr = np.array(rows, dtype=np.float64)
    if arr.shape[1] >= 4:
        return electronic.DensityOfStates(
            energies=arr[:, 0],
            total_dos=arr[:, 1],
            spin_down_dos=arr[:, 2],
            fermi_energy=fermi,
            metadata={"source": str(p)},
        )
    return electronic.DensityOfStates(
        energies=arr[:, 0],
        total_dos=arr[:, 1],
        fermi_energy=fermi,
        metadata={"source": str(p)},
    )


registry.register_format(
    registry.FormatSpec(
        name="qe_in",
        extensions=(".pwi", ".in"),
        filenames=(),
        reader=read_qe_in,
        description="Quantum ESPRESSO pw.x input structure",
    )
)
registry.register_format(
    registry.FormatSpec(
        name="qe_out",
        extensions=(".pwo",),
        filenames=(),
        reader=read_qe_out,
        description="Quantum ESPRESSO pw.x output structure and energy",
    )
)
registry.register_format(
    registry.FormatSpec(
        name="qe_bands",
        extensions=(),
        filenames=("bands.dat",),
        reader=read_qe_bands,
        description="Quantum ESPRESSO bands.x electronic band dispersion",
    )
)
registry.register_format(
    registry.FormatSpec(
        name="qe_dos",
        extensions=(),
        filenames=("dos.dat",),
        reader=read_qe_dos,
        description="Quantum ESPRESSO dos.x electronic density of states",
    )
)
