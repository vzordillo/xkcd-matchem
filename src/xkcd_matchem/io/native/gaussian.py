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
"""Native parsers for Gaussian inputs (.gjf/.com), logs (.log), and .fchk."""

from __future__ import annotations

import pathlib
import re

import numpy as np

from xkcd_matchem.core import electronic
from xkcd_matchem.core import exceptions
from xkcd_matchem.core import spectroscopy
from xkcd_matchem.core import structure
from xkcd_matchem.data import elements
from xkcd_matchem.io import registry


def read_gaussian_gjf(path: str | pathlib.Path) -> structure.AtomsData:
    """Reads a Gaussian ``.gjf`` or ``.com`` Cartesian input file."""
    p = pathlib.Path(path)
    lines = p.read_text(encoding="utf-8").splitlines()
    idx = 0
    while idx < len(lines) and (
        lines[idx].startswith("%") or lines[idx].startswith("#")
    ):
        idx += 1
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    while idx < len(lines) and lines[idx].strip():
        idx += 1
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    if idx >= len(lines):
        raise exceptions.ParseError(f"Malformed Gaussian input file: {p}")
    idx += 1
    symbols: list[str] = []
    coords: list[list[float]] = []
    while idx < len(lines) and lines[idx].strip():
        parts = lines[idx].split()
        if len(parts) >= 4:
            symbols.append(elements.get_element(parts[0]).symbol)
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
        idx += 1
    if not symbols:
        raise exceptions.ParseError(f"No coordinates found in {p}.")
    return structure.AtomsData(
        symbols=symbols,
        positions=np.array(coords, dtype=np.float64),
        metadata={"source": str(p)},
    )


def read_gaussian_log_full(path: str | pathlib.Path) -> dict[str, object]:
    """Extracts structure, SCF energies, MOs, and IR spectrum from Gaussian log.

    Args:
      path: Path to a Gaussian ``.log`` output file.

    Returns:
      Dictionary with keys ``"atoms"``, ``"energies"``, ``"orbitals"``, and
      optionally ``"ir_spectrum"``.

    Raises:
      ParseError: If no atomic orientation block can be extracted.
    """
    p = pathlib.Path(path)
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    last_atoms: structure.AtomsData | None = None
    scf_energies: list[float] = []
    occ_ev: list[float] = []
    virt_ev: list[float] = []
    freqs: list[float] = []
    ir_ints: list[float] = []
    ha_to_ev = elements.CODATA2022.hartree_to_ev

    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if "Standard orientation:" in line or "Input orientation:" in line:
            idx += 5
            syms: list[str] = []
            pos: list[list[float]] = []
            while idx < len(lines) and not lines[idx].strip().startswith("---"):
                parts = lines[idx].split()
                if len(parts) >= 6:
                    z_num = int(parts[1])
                    syms.append(elements.get_element(z_num).symbol)
                    pos.append(
                        [float(parts[3]), float(parts[4]), float(parts[5])]
                    )
                idx += 1
            if syms:
                last_atoms = structure.AtomsData(
                    symbols=syms,
                    positions=np.array(pos, dtype=np.float64),
                    metadata={"source": str(p)},
                )
        elif "SCF Done:" in line:
            match = re.search(r"=\s*([-+]?\d+\.\d+)", line)
            if match:
                scf_energies.append(float(match.group(1)) * ha_to_ev)
        elif "Alpha  occ. eigenvalues --" in line:
            occ_ev.extend(
                float(x) * ha_to_ev for x in line.split("--", 1)[1].split()
            )
        elif "Alpha virt. eigenvalues --" in line:
            virt_ev.extend(
                float(x) * ha_to_ev for x in line.split("--", 1)[1].split()
            )
        elif line.strip().startswith("Frequencies --"):
            freqs.extend(float(x) for x in line.split("--", 1)[1].split())
        elif line.strip().startswith("IR Inten    --"):
            ir_ints.extend(float(x) for x in line.split("--", 1)[1].split())
        idx += 1

    if last_atoms is None:
        raise exceptions.ParseError(
            f"No geometry orientation found in Gaussian log: {p}"
        )
    if scf_energies:
        last_atoms.energies["total"] = scf_energies[-1]
        last_atoms.metadata["scf_trajectory_ev"] = scf_energies

    mo_energies = np.array(occ_ev + virt_ev, dtype=np.float64)
    mo_occs = np.array(
        [2.0] * len(occ_ev) + [0.0] * len(virt_ev), dtype=np.float64
    )
    orbitals = electronic.MolecularOrbitals(
        energies=mo_energies, occupations=mo_occs
    )

    result: dict[str, object] = {
        "atoms": last_atoms,
        "energies": scf_energies,
        "orbitals": orbitals,
    }
    if freqs and len(freqs) == len(ir_ints):
        x_grid = np.linspace(400.0, 4200.0, 1200)
        y_vals = spectroscopy.broaden_peaks(
            freqs, ir_ints, x_grid, fwhm=35.0, profile="lorentzian"
        )
        peaks = [
            spectroscopy.Peak(position=f, intensity=val, label=f"{f:.0f}")
            for f, val in zip(freqs, ir_ints, strict=True)
        ]
        result["ir_spectrum"] = spectroscopy.Spectrum(
            x=x_grid,
            y=y_vals,
            kind="ir",
            x_label=r"Wavenumber (cm$^{-1}$)",
            y_label="IR Intensity (km/mol)",
            peaks=peaks,
            metadata={"source": str(p)},
        )
    return result


def read_gaussian_log(path: str | pathlib.Path) -> structure.AtomsData:
    """Reads the final optimized structure and metadata from a Gaussian log."""
    data = read_gaussian_log_full(path)
    atoms = data["atoms"]
    if not isinstance(atoms, structure.AtomsData):
        raise exceptions.ParseError("Unexpected Gaussian structure type.")
    atoms.metadata["orbitals"] = data["orbitals"]
    if "ir_spectrum" in data:
        atoms.metadata["ir_spectrum"] = data["ir_spectrum"]
    return atoms


def _read_fchk_array(lines: list[str], start: int, count: int) -> list[float]:
    """Reads ``count`` floats following a Gaussian FCHK array section header."""
    vals: list[float] = []
    idx = start
    while idx < len(lines) and len(vals) < count:
        vals.extend(float(tok) for tok in lines[idx].split())
        idx += 1
    return vals[:count]


def read_gaussian_fchk(path: str | pathlib.Path) -> structure.AtomsData:
    """Reads atomic numbers and Cartesian coordinates from a ``.fchk`` file."""
    p = pathlib.Path(path)
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    bohr = elements.CODATA2022.bohr_to_angstrom
    z_nums: list[int] = []
    coords_bohr: list[float] = []
    total_e: float | None = None
    for idx, line in enumerate(lines):
        if line.startswith("Atomic numbers"):
            cnt = int(line.split()[-1])
            raw: list[int] = []
            k = idx + 1
            while k < len(lines) and len(raw) < cnt:
                raw.extend(int(t) for t in lines[k].split())
                k += 1
            z_nums = raw[:cnt]
        elif line.startswith("Current cartesian coordinates"):
            cnt = int(line.split()[-1])
            coords_bohr = _read_fchk_array(lines, idx + 1, cnt)
        elif line.startswith("Total Energy"):
            total_e = (
                float(line.split()[-1]) * elements.CODATA2022.hartree_to_ev
            )
    if not z_nums or len(coords_bohr) != 3 * len(z_nums):
        raise exceptions.ParseError(f"Missing geometry arrays in FCHK: {p}")
    syms = [elements.get_element(z).symbol for z in z_nums]
    pos = np.array(coords_bohr, dtype=np.float64).reshape((-1, 3)) * bohr
    energies = {"total": total_e} if total_e is not None else {}
    return structure.AtomsData(
        symbols=syms,
        positions=pos,
        energies=energies,
        metadata={"source": str(p)},
    )


registry.register_format(
    registry.FormatSpec(
        name="gaussian_gjf",
        extensions=(".gjf", ".com"),
        filenames=(),
        reader=read_gaussian_gjf,
        description="Gaussian input route and Cartesian geometry",
    )
)
registry.register_format(
    registry.FormatSpec(
        name="gaussian_log",
        extensions=(".log",),
        filenames=(),
        reader=read_gaussian_log,
        description="Gaussian log geometry, SCF, MOs, and IR spectrum",
    )
)
registry.register_format(
    registry.FormatSpec(
        name="gaussian_fchk",
        extensions=(".fchk",),
        filenames=(),
        reader=read_gaussian_fchk,
        description="Gaussian formatted checkpoint geometry and energy",
    )
)
