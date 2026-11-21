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
"""Native VASP parsers (POSCAR, CONTCAR, XDATCAR, DOSCAR, EIGENVAL, CHGCAR)."""

from __future__ import annotations

import collections
import pathlib

import numpy as np

from xkcd_matchem.core import electronic
from xkcd_matchem.core import exceptions
from xkcd_matchem.core import structure
from xkcd_matchem.core import trajectory
from xkcd_matchem.io import registry


def _parse_poscar_header(
    lines: list[str],
) -> tuple[str, np.ndarray, list[str], bool, int]:
    """Parses the common VASP 4/5 header lines through coordinate mode."""
    if len(lines) < 8:
        raise exceptions.ParseError("VASP POSCAR header is too short.")
    comment = lines[0].strip()
    scale_val = float(lines[1].strip().split()[0])
    raw_cell = np.array(
        [[float(x) for x in lines[k].split()[:3]] for k in range(2, 5)],
        dtype=np.float64,
    )
    if scale_val < 0.0:
        raw_vol = abs(float(np.linalg.det(raw_cell)))
        scale = (-scale_val / raw_vol) ** (1.0 / 3.0)
    else:
        scale = scale_val
    cell = raw_cell * scale

    tokens5 = lines[5].split()
    if all(tok.lstrip("+-").isdigit() for tok in tokens5):
        counts = [int(tok) for tok in tokens5]
        species = comment.split()[: len(counts)]
        if len(species) != len(counts):
            raise exceptions.ParseError(
                "VASP 4 POSCAR lacks species symbols line; use VASP 5 format."
            )
        idx = 6
    else:
        species = tokens5
        counts = [int(tok) for tok in lines[6].split()]
        idx = 7

    if len(species) != len(counts):
        raise exceptions.ParseError("Mismatch between VASP species and counts.")
    symbols: list[str] = []
    for sym, cnt in zip(species, counts, strict=True):
        symbols.extend([sym] * cnt)

    if lines[idx].strip().lower().startswith("s"):
        idx += 1
    mode = lines[idx].strip().lower()
    is_cartesian = mode.startswith(("c", "k"))
    return comment, cell, symbols, is_cartesian, idx + 1


def read_poscar(path: str | pathlib.Path) -> structure.AtomsData:
    """Reads a VASP 4/5 ``POSCAR`` or ``CONTCAR`` structure file."""
    p = pathlib.Path(path)
    lines = p.read_text(encoding="utf-8").splitlines()
    comment, cell, symbols, is_cart, start_row = _parse_poscar_header(lines)
    n_atoms = len(symbols)
    if start_row + n_atoms > len(lines):
        raise exceptions.ParseError(
            f"Truncated POSCAR coordinate block in {p}."
        )
    coords = np.zeros((n_atoms, 3), dtype=np.float64)
    for k in range(n_atoms):
        parts = lines[start_row + k].split()
        coords[k] = [float(parts[0]), float(parts[1]), float(parts[2])]
    if is_cart:
        return structure.AtomsData(
            symbols=symbols,
            positions=coords,
            cell=cell,
            pbc=(True, True, True),
            metadata={"comment": comment, "source": str(p)},
        )
    return structure.AtomsData.from_fractional(
        symbols=symbols,
        frac_positions=coords,
        cell=cell,
        pbc=(True, True, True),
        metadata={"comment": comment, "source": str(p)},
    )


def write_poscar(
    atoms: structure.AtomsData,
    path: str | pathlib.Path,
) -> None:
    """Writes an ``AtomsData`` crystal structure in VASP 5 Direct POSCAR."""
    if atoms.cell is None:
        raise exceptions.StructureError("Writing POSCAR requires a unit cell.")
    p = pathlib.Path(path)
    grouped: collections.OrderedDict[str, list[int]] = collections.OrderedDict()
    for idx, sym in enumerate(atoms.symbols):
        grouped.setdefault(sym, []).append(idx)
    species = list(grouped.keys())
    counts = [str(len(grouped[s])) for s in species]
    frac = atoms.fractional_positions()
    lines = [
        str(atoms.metadata.get("comment", atoms.formula())),
        "1.0",
    ]
    for row in atoms.cell:
        lines.append(f"  {row[0]:18.12f} {row[1]:18.12f} {row[2]:18.12f}")
    lines.append("  " + "  ".join(species))
    lines.append("  " + "  ".join(counts))
    lines.append("Direct")
    for sym in species:
        for idx in grouped[sym]:
            f = frac[idx]
            lines.append(f"  {f[0]:18.12f} {f[1]:18.12f} {f[2]:18.12f}")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_xdatcar(path: str | pathlib.Path) -> trajectory.Trajectory:
    """Reads a multi-configuration VASP ``XDATCAR`` MD/relaxation trajectory."""
    p = pathlib.Path(path)
    lines = p.read_text(encoding="utf-8").splitlines()
    comment, cell, symbols, _, cursor = _parse_poscar_header(lines)
    n_atoms = len(symbols)
    frames: list[structure.AtomsData] = []
    idx = cursor - 1
    while idx < len(lines):
        line = lines[idx].strip().lower()
        if line.startswith("direct"):
            idx += 1
            if idx + n_atoms > len(lines):
                break
            frac = np.array(
                [
                    [float(x) for x in lines[idx + k].split()[:3]]
                    for k in range(n_atoms)
                ],
                dtype=np.float64,
            )
            frames.append(
                structure.AtomsData.from_fractional(
                    symbols=symbols,
                    frac_positions=frac,
                    cell=cell,
                    pbc=(True, True, True),
                    metadata={"comment": comment, "source": str(p)},
                )
            )
            idx += n_atoms
        else:
            idx += 1
    if not frames:
        raise exceptions.ParseError(f"No XDATCAR configurations found in {p}.")
    return trajectory.Trajectory(frames=frames, metadata={"source": str(p)})


def read_doscar(path: str | pathlib.Path) -> electronic.DensityOfStates:
    """Reads total density of states from a VASP ``DOSCAR`` file."""
    p = pathlib.Path(path)
    lines = p.read_text(encoding="utf-8").splitlines()
    if len(lines) < 7:
        raise exceptions.ParseError(f"Truncated VASP DOSCAR file: {p}")
    hdr = lines[5].split()
    nedos = int(hdr[2])
    fermi = float(hdr[3])
    data_rows = [
        [float(x) for x in line.split()] for line in lines[6 : 6 + nedos]
    ]
    arr = np.array(data_rows, dtype=np.float64)
    if arr.shape[1] >= 5:
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


def read_eigenval(path: str | pathlib.Path) -> electronic.BandStructure:
    """Reads band eigenvalues from a VASP ``EIGENVAL`` file."""
    p = pathlib.Path(path)
    lines = p.read_text(encoding="utf-8").splitlines()
    if len(lines) < 8:
        raise exceptions.ParseError(f"Truncated VASP EIGENVAL file: {p}")
    n_spin = int(lines[0].split()[3])
    hdr = lines[5].split()
    n_k = int(hdr[1])
    n_bands = int(hdr[2])
    cursor = 6
    k_coords = np.zeros((n_k, 3), dtype=np.float64)
    eig = np.zeros((n_spin, n_k, n_bands), dtype=np.float64)
    for ik in range(n_k):
        while cursor < len(lines) and not lines[cursor].strip():
            cursor += 1
        k_coords[ik] = [float(x) for x in lines[cursor].split()[:3]]
        cursor += 1
        for ib in range(n_bands):
            vals = [float(x) for x in lines[cursor].split()]
            eig[0, ik, ib] = vals[1]
            if n_spin == 2 and len(vals) >= 3:
                eig[1, ik, ib] = vals[2]
            cursor += 1
    diffs = np.diff(k_coords, axis=0)
    k_dist = np.concatenate([[0.0], np.cumsum(np.linalg.norm(diffs, axis=1))])
    return electronic.BandStructure(
        k_distances=k_dist,
        eigenvalues=eig,
        metadata={"source": str(p)},
    )


def read_chgcar(path: str | pathlib.Path) -> electronic.VolumetricGrid:
    """Reads a VASP ``CHGCAR`` or ``LOCPOT`` 3D volumetric grid file."""
    p = pathlib.Path(path)
    lines = p.read_text(encoding="utf-8").splitlines()
    comment, cell, symbols, is_cart, start_row = _parse_poscar_header(lines)
    n_atoms = len(symbols)
    coords = np.array(
        [
            [float(x) for x in lines[start_row + k].split()[:3]]
            for k in range(n_atoms)
        ],
        dtype=np.float64,
    )
    atoms = (
        structure.AtomsData(
            symbols=symbols,
            positions=coords,
            cell=cell,
            pbc=(True, True, True),
        )
        if is_cart
        else structure.AtomsData.from_fractional(
            symbols=symbols, frac_positions=coords, cell=cell
        )
    )
    cursor = start_row + n_atoms
    while cursor < len(lines) and not lines[cursor].strip():
        cursor += 1
    dims = [int(x) for x in lines[cursor].split()[:3]]
    nx, ny, nz = dims[0], dims[1], dims[2]
    n_vals = nx * ny * nz
    vals: list[float] = []
    cursor += 1
    while cursor < len(lines) and len(vals) < n_vals:
        row = lines[cursor].strip()
        if row:
            vals.extend(float(tok) for tok in row.split())
        cursor += 1
    if len(vals) < n_vals:
        raise exceptions.ParseError(f"Incomplete VASP grid data in {p}.")
    data = np.array(vals[:n_vals], dtype=np.float64).reshape(
        (nx, ny, nz), order="F"
    )
    vol = max(atoms.volume(), 1.0)
    if "CHGCAR" in p.name.upper():
        data = data / vol
    voxel = np.vstack([cell[0] / nx, cell[1] / ny, cell[2] / nz])
    return electronic.VolumetricGrid(
        origin=np.zeros(3, dtype=np.float64),
        voxel_vectors=voxel,
        data=data,
        atoms=atoms,
        units="e/A^3" if "CHGCAR" in p.name.upper() else "eV",
        metadata={"comment": comment, "source": str(p)},
    )


registry.register_format(
    registry.FormatSpec(
        name="vasp",
        extensions=(".poscar", ".contcar", ".vasp"),
        filenames=("POSCAR", "CONTCAR"),
        reader=read_poscar,
        writer=write_poscar,
        description="VASP 4/5 crystal POSCAR/CONTCAR structure",
    )
)
registry.register_format(
    registry.FormatSpec(
        name="xdatcar",
        extensions=(".xdatcar",),
        filenames=("XDATCAR",),
        reader=read_xdatcar,
        description="VASP XDATCAR multi-frame trajectory",
    )
)
registry.register_format(
    registry.FormatSpec(
        name="doscar",
        extensions=(".doscar",),
        filenames=("DOSCAR",),
        reader=read_doscar,
        description="VASP DOSCAR density of states",
    )
)
registry.register_format(
    registry.FormatSpec(
        name="eigenval",
        extensions=(".eigenval",),
        filenames=("EIGENVAL",),
        reader=read_eigenval,
        description="VASP EIGENVAL electronic band eigenvalues",
    )
)
registry.register_format(
    registry.FormatSpec(
        name="chgcar",
        extensions=(".chgcar", ".locpot"),
        filenames=("CHGCAR", "LOCPOT", "AECCAR0", "AECCAR2"),
        reader=read_chgcar,
        description="VASP volumetric CHGCAR or LOCPOT grid",
    )
)
