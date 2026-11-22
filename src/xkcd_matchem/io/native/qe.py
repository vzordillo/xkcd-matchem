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


_RY_TO_EV = elements.CODATA2022.rydberg_to_ev


def _clean_qe_symbol(label: str) -> str:
    """Strips numeric suffixes or modifiers from QE atomic species tags."""
    sym = re.sub(r"[^A-Za-z]", "", label.strip())
    return elements.get_element(
        sym[:2]
        if len(sym) >= 2 and sym[:2].capitalize() in elements.ELEMENTS_BY_SYMBOL
        else sym[:1]
    ).symbol


def _parse_qe_celldm_and_ibrav(
    text: str, bohr: float
) -> tuple[float | None, int]:
    """Extracts ``celldm(1)`` (in Bohr) and ``ibrav`` from a QE input."""
    celldm1: float | None = None
    m_cd = re.search(r"celldm\s*\(\s*1\s*\)\s*=\s*([0-9.eEdD+-]+)", text)
    if m_cd:
        celldm1 = float(m_cd.group(1).replace("d", "e").replace("D", "E"))
    else:
        m_a = re.search(r"(?im)^\s*A\s*=\s*([0-9.eEdD+-]+)", text)
        if m_a:
            celldm1 = (
                float(m_a.group(1).replace("d", "e").replace("D", "E")) / bohr
            )
    m_ib = re.search(r"(?im)\bibrav\s*=\s*(-?\d+)", text)
    return celldm1, int(m_ib.group(1)) if m_ib else 0


def _parse_qe_cell_card(
    lines: list[str],
    idx: int,
    celldm1: float | None,
    bohr: float,
) -> tuple[np.ndarray, int]:
    """Parses a QE ``CELL_PARAMETERS`` card into an Angstrom matrix."""
    upper = lines[idx].strip().upper()
    if "ANGSTROM" in upper:
        scale = 1.0
    elif "BOHR" in upper:
        scale = bohr
    else:
        if celldm1 is None:
            raise exceptions.ParseError(
                "QE CELL_PARAMETERS in alat requires celldm(1) or A."
            )
        scale = bohr * celldm1
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
    return cell, idx + 4


def _parse_qe_positions_card(
    lines: list[str], idx: int
) -> tuple[list[str], list[list[float]], str, int]:
    """Parses a QE ``ATOMIC_POSITIONS`` block and its coordinate unit."""
    upper = lines[idx].strip().upper()
    pos_unit = "alat"
    for unit_key in ("crystal", "angstrom", "bohr"):
        if unit_key.upper() in upper:
            pos_unit = unit_key
            break
    syms: list[str] = []
    coords: list[list[float]] = []
    cur = idx + 1
    stops = ("K_POINTS", "CELL_", "ATOMIC_", "CONSTRAINTS")
    while cur < len(lines):
        row = lines[cur].strip()
        if not row or row.upper().startswith(stops):
            break
        parts = row.split()
        if len(parts) >= 4:
            syms.append(_clean_qe_symbol(parts[0]))
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
        cur += 1
    return syms, coords, pos_unit, cur


def read_qe_in(path: str | pathlib.Path) -> structure.AtomsData:
    """Reads a Quantum ESPRESSO ``pw.in`` input script into ``AtomsData``."""
    p = pathlib.Path(path)
    text = p.read_text(encoding="utf-8")
    lines = text.splitlines()
    bohr = elements.CODATA2022.bohr_to_angstrom
    celldm1, ibrav = _parse_qe_celldm_and_ibrav(text, bohr)

    cell: np.ndarray | None = None
    syms: list[str] = []
    coords: list[list[float]] = []
    pos_unit = "alat"
    idx = 0
    while idx < len(lines):
        upper = lines[idx].strip().upper()
        if upper.startswith("CELL_PARAMETERS"):
            cell, idx = _parse_qe_cell_card(lines, idx, celldm1, bohr)
        elif upper.startswith("ATOMIC_POSITIONS"):
            syms, coords, pos_unit, idx = _parse_qe_positions_card(lines, idx)
        else:
            idx += 1

    if cell is None and ibrav != 0:
        raise exceptions.ParseError(
            f"QE input specifies ibrav={ibrav} without CELL_PARAMETERS; use "
            "explicit CELL_PARAMETERS or xm.io.read_via_ase(...)."
        )
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
        if celldm1 is None:
            raise exceptions.ParseError(
                "QE ATOMIC_POSITIONS (alat) requires celldm(1) or A."
            )
        arr *= celldm1 * bohr
    return structure.AtomsData(
        symbols=syms,
        positions=arr,
        cell=cell,
        pbc=(cell is not None, cell is not None, cell is not None),
        metadata={"source": str(p)},
    )


def _parse_qe_atomic_positions(
    lines: list[str],
    idx: int,
    alat_bohr: float,
    cell: np.ndarray | None,
) -> tuple[list[str], list[list[float]]]:
    """Parses an ``ATOMIC_POSITIONS`` block in a QE ``pw.out`` file."""
    bohr = elements.CODATA2022.bohr_to_angstrom
    hdr = lines[idx].lower()
    scale = (
        bohr
        if "bohr" in hdr
        else (1.0 if "angstrom" in hdr else alat_bohr * bohr)
    )
    is_crys = "crystal" in hdr
    k = idx + 1
    tmp_s: list[str] = []
    tmp_c: list[list[float]] = []
    while (
        k < len(lines)
        and lines[k].strip()
        and not lines[k].strip().startswith("End")
    ):
        parts = lines[k].split()
        if len(parts) < 4:
            break
        tmp_s.append(_clean_qe_symbol(parts[0]))
        tmp_c.append([float(parts[1]), float(parts[2]), float(parts[3])])
        k += 1
    if not tmp_s:
        return [], []
    arr_tmp = np.array(tmp_c, dtype=np.float64)
    coords = (
        (arr_tmp @ cell).tolist()
        if (is_crys and cell is not None)
        else (arr_tmp * scale).tolist()
    )
    return tmp_s, coords


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
            tmp_s, tmp_c_ang = _parse_qe_atomic_positions(
                lines, idx, alat_bohr, cell
            )
            if tmp_s:
                syms, coords = tmp_s, tmp_c_ang
        elif "!" in line and "total energy" in line:
            last_energy = float(line.split("=")[1].split()[0]) * _RY_TO_EV
        elif "the Fermi energy is" in line:
            fermi = float(line.split("is")[1].split()[0])
        elif "highest occupied" in line and ":" in line and fermi is None:
            rhs_toks = line.split(":", 1)[1].split()
            if rhs_toks:
                fermi = float(rhs_toks[0])
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


def _split_touching_floats(text: str) -> str:
    """Separates concatenated negative Fortran floats like ``1.23-4.56``."""
    return re.sub(r"(?<=[0-9])-(?=[0-9.])", " -", text)


def read_qe_bands(path: str | pathlib.Path) -> electronic.BandStructure:
    """Reads a Quantum ESPRESSO ``bands.x`` plain-text output file."""
    p = pathlib.Path(path)
    lines = p.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise exceptions.ParseError(f"Empty QE bands file: {p}")
    header = lines[0]
    m_nb = re.search(r"nbnd\s*=\s*(\d+)", header)
    m_nk = re.search(r"nks\s*=\s*(\d+)", header)
    if not m_nb or not m_nk:
        raise exceptions.ParseError(f"Invalid QE bands header in {p}.")
    nbnd = int(m_nb.group(1))
    nks = int(m_nk.group(1))
    body = _split_touching_floats(" ".join(lines[1:]).replace("/", " "))
    n_needed = nks * (3 + nbnd)
    arr = np.fromstring(body, sep=" ", dtype=np.float64, count=n_needed)
    if arr.size < n_needed:
        raise exceptions.ParseError(
            f"Truncated QE bands data in {p}: expected {n_needed} values."
        )
    block = arr.reshape((nks, 3 + nbnd))
    k_coords = block[:, :3]
    eig = block[:, 3:][None, :, :]
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
        rows.append(
            [float(tok) for tok in _split_touching_floats(stripped).split()]
        )
    if not rows:
        raise exceptions.ParseError(f"No DOS rows found in QE dos.dat: {p}")
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
