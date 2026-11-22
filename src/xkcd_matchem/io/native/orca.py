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
"""Native parsers for ORCA inputs (.inp) and outputs (.out)."""

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
from xkcd_matchem.io.native import xyz


def read_orca_inp(path: str | pathlib.Path) -> structure.AtomsData:
    """Reads Cartesian coordinates from an ORCA ``.inp`` ``* xyz`` block."""
    p = pathlib.Path(path)
    lines = p.read_text(encoding="utf-8").splitlines()
    syms: list[str] = []
    coords: list[list[float]] = []
    in_xyz = False
    for line in lines:
        stripped = line.strip()
        if re.match(r"^\*\s*xyzfile\b", stripped, flags=re.IGNORECASE):
            parts = stripped.split()
            if len(parts) >= 5:
                ext_path = p.parent / parts[-1]
                if ext_path.is_file():
                    return xyz.read_xyz(ext_path)
            raise exceptions.ParseError(
                f"ORCA * xyzfile target not found in {p}: {stripped!r}"
            )
        if re.match(r"^\*\s*xyz\b", stripped, flags=re.IGNORECASE):
            in_xyz = True
            continue
        if in_xyz:
            if stripped == "*":
                break
            parts = stripped.split()
            if len(parts) >= 4:
                syms.append(elements.get_element(parts[0]).symbol)
                coords.append(
                    [float(parts[1]), float(parts[2]), float(parts[3])]
                )
    if not syms:
        raise exceptions.ParseError(
            f"No '* xyz' block found in ORCA input: {p}"
        )
    return structure.AtomsData(
        symbols=syms,
        positions=np.array(coords, dtype=np.float64),
        metadata={"source": str(p)},
    )


def _parse_orca_coords(
    lines: list[str],
    start: int,
    source: str,
) -> tuple[structure.AtomsData | None, int]:
    """Parses one ORCA ``CARTESIAN COORDINATES (ANGSTROEM)`` block."""
    idx = start + 2
    syms: list[str] = []
    pos: list[list[float]] = []
    while idx < len(lines) and lines[idx].strip():
        parts = lines[idx].split()
        if len(parts) >= 4 and not parts[0].startswith("-"):
            syms.append(elements.get_element(parts[0]).symbol)
            pos.append([float(parts[1]), float(parts[2]), float(parts[3])])
        idx += 1
    if not syms:
        return None, idx
    return (
        structure.AtomsData(
            symbols=syms,
            positions=np.array(pos, dtype=np.float64),
            metadata={"source": source},
        ),
        idx,
    )


def _parse_orca_orbitals(
    lines: list[str],
    start: int,
) -> tuple[list[float], list[float], int]:
    """Parses ORCA ``ORBITAL ENERGIES`` returning ``(e_ev, occs, idx)``."""
    idx = start + 4
    mo_e: list[float] = []
    mo_occ: list[float] = []
    while (
        idx < len(lines)
        and lines[idx].strip()
        and not lines[idx].startswith("-")
    ):
        parts = lines[idx].split()
        if len(parts) >= 4 and parts[0].isdigit():
            mo_occ.append(float(parts[1]))
            mo_e.append(float(parts[3]))
        idx += 1
    return mo_e, mo_occ, idx


def read_orca_out_full(path: str | pathlib.Path) -> dict[str, object]:
    """Extracts structure, energy, MOs, IR, and UV-Vis from an ORCA output."""
    p = pathlib.Path(path)
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    ha_to_ev = elements.CODATA2022.hartree_to_ev
    last_atoms: structure.AtomsData | None = None
    total_ev: float | None = None
    mo_e: list[float] = []
    mo_occ: list[float] = []
    ir_freqs: list[float] = []
    ir_ints: list[float] = []
    uv_nm: list[float] = []
    uv_fosc: list[float] = []

    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if "CARTESIAN COORDINATES (ANGSTROEM)" in line:
            parsed, idx = _parse_orca_coords(lines, idx, str(p))
            if parsed is not None:
                last_atoms = parsed
        elif "FINAL SINGLE POINT ENERGY" in line:
            total_ev = float(line.split()[-1]) * ha_to_ev
        elif line.strip() == "ORBITAL ENERGIES":
            mo_e, mo_occ, idx = _parse_orca_orbitals(lines, idx)
        elif line.strip() == "VIBRATIONAL FREQUENCIES":
            idx += 2
            while idx < len(lines) and not lines[idx].strip().startswith("---"):
                match = re.search(
                    r":\s*([-+]?\d+\.\d+)\s+cm\*\*-1(?:\s+Int:\s*([\d.]+))?",
                    lines[idx],
                )
                if match and float(match.group(1)) > 10.0:
                    ir_freqs.append(float(match.group(1)))
                    ir_ints.append(
                        float(match.group(2)) if match.group(2) else 10.0
                    )
                idx += 1
        elif (
            "ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS" in line
        ):
            idx += 5
            while (
                idx < len(lines)
                and lines[idx].strip()
                and not lines[idx].strip().startswith("---")
            ):
                parts = lines[idx].split()
                if len(parts) >= 4 and parts[0].isdigit():
                    uv_nm.append(float(parts[2]))
                    uv_fosc.append(float(parts[3]))
                idx += 1
        idx += 1

    if last_atoms is None:
        raise exceptions.ParseError(f"No coordinates found in ORCA output: {p}")
    if total_ev is not None:
        last_atoms.energies["total"] = total_ev

    result: dict[str, object] = {"atoms": last_atoms}
    if mo_e:
        result["orbitals"] = electronic.MolecularOrbitals(
            energies=np.array(mo_e, dtype=np.float64),
            occupations=np.array(mo_occ, dtype=np.float64),
        )
    if ir_freqs:
        grid = np.linspace(400.0, 4000.0, 1200)
        result["ir_spectrum"] = spectroscopy.Spectrum(
            x=grid,
            y=spectroscopy.broaden_peaks(ir_freqs, ir_ints, grid, fwhm=35.0),
            kind="ir",
            x_label=r"Wavenumber (cm$^{-1}$)",
            y_label="IR Intensity",
            peaks=[
                spectroscopy.Peak(f, i, f"{f:.0f}")
                for f, i in zip(ir_freqs, ir_ints, strict=True)
            ],
        )
    if uv_nm:
        grid_uv = np.linspace(150.0, 500.0, 800)
        result["uvvis_spectrum"] = spectroscopy.Spectrum(
            x=grid_uv,
            y=spectroscopy.broaden_peaks(
                uv_nm, uv_fosc, grid_uv, fwhm=14.0, profile="gaussian"
            ),
            kind="uvvis",
            x_label="Wavelength (nm)",
            y_label="Oscillator Strength",
            peaks=[
                spectroscopy.Peak(w, f, f"{w:.1f} nm")
                for w, f in zip(uv_nm, uv_fosc, strict=True)
            ],
        )
    return result


def read_orca_out(path: str | pathlib.Path) -> structure.AtomsData:
    """Reads final geometry plus embedded electronic/spectral metadata."""
    data = read_orca_out_full(path)
    atoms = data["atoms"]
    if not isinstance(atoms, structure.AtomsData):
        raise exceptions.ParseError("Unexpected ORCA atoms object.")
    for key in ("orbitals", "ir_spectrum", "uvvis_spectrum"):
        if key in data:
            atoms.metadata[key] = data[key]
    return atoms


registry.register_format(
    registry.FormatSpec(
        name="orca_inp",
        extensions=(".inp",),
        filenames=(),
        reader=read_orca_inp,
        description="ORCA calculation input file",
    )
)
registry.register_format(
    registry.FormatSpec(
        name="orca_out",
        extensions=(".orca",),
        filenames=(),
        reader=read_orca_out,
        description="ORCA output structure, orbitals, IR, and UV-Vis",
    )
)
