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
"""Native parsers for LAMMPS data files and .lammpstrj trajectories."""

from __future__ import annotations

import pathlib

import numpy as np

from xkcd_matchem.core import exceptions
from xkcd_matchem.core import structure
from xkcd_matchem.core import trajectory
from xkcd_matchem.data import elements
from xkcd_matchem.io import registry


def _symbol_from_mass(mass: float) -> str:
    """Infers closest chemical element symbol from an atomic mass in Daltons."""
    best_sym = "X"
    best_diff = 1e9
    for rec in elements.ELEMENTS_BY_Z.values():
        diff = abs(rec.atomic_mass - mass)
        if diff < best_diff:
            best_diff = diff
            best_sym = rec.symbol
    return best_sym


def _parse_lammps_atom_rows(
    atom_rows: list[list[str]],
    masses: dict[int, float],
) -> tuple[list[str], list[list[float]], dict[int, int]]:
    """Extracts symbols, coordinates, and ID mapping from LAMMPS data rows."""
    id_to_idx: dict[int, int] = {}
    symbols: list[str] = []
    coords: list[list[float]] = []
    for idx, row in enumerate(atom_rows):
        id_to_idx[int(row[0])] = idx
        if len(row) >= 7:
            atype = int(row[2])
            xyz = [float(row[4]), float(row[5]), float(row[6])]
        elif len(row) >= 5:
            atype = int(row[1])
            offset = 3 if len(row) >= 6 else 2
            xyz = [
                float(row[offset]),
                float(row[offset + 1]),
                float(row[offset + 2]),
            ]
        else:
            raise exceptions.ParseError(
                f"Unsupported LAMMPS Atoms row: {row!r}"
            )
        symbols.append(
            _symbol_from_mass(masses[atype]) if atype in masses else "C"
        )
        coords.append(xyz)
    return symbols, coords, id_to_idx


def read_lammps_data(path: str | pathlib.Path) -> structure.AtomsData:
    """Reads a LAMMPS ``read_data`` topology/coordinate file."""
    p = pathlib.Path(path)
    lines = p.read_text(encoding="utf-8").splitlines()
    bounds = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)}
    tilt = [0.0, 0.0, 0.0]
    masses: dict[int, float] = {}
    atom_rows: list[list[str]] = []
    bond_rows: list[list[str]] = []
    section = ""

    for raw in lines:
        line = raw.split("#", maxsplit=1)[0].strip()
        if not line:
            continue
        if line.endswith(("xlo xhi", "ylo yhi", "zlo zhi")):
            parts = line.split()
            bounds[parts[2][0]] = (float(parts[0]), float(parts[1]))
            continue
        if line.endswith("xy xz yz"):
            parts = line.split()
            tilt = [float(parts[0]), float(parts[1]), float(parts[2])]
            continue
        if line.split()[0] in (
            "Masses",
            "Atoms",
            "Bonds",
            "Velocities",
            "Angles",
        ):
            section = line.split()[0]
            continue
        if section == "Masses" and len(line.split()) >= 2:
            parts = line.split()
            masses[int(parts[0])] = float(parts[1])
        elif section == "Atoms":
            atom_rows.append(line.split())
        elif section == "Bonds":
            bond_rows.append(line.split())

    if not atom_rows:
        raise exceptions.ParseError(
            f"No Atoms section in LAMMPS data file: {p}"
        )

    cell = np.array(
        [
            [bounds["x"][1] - bounds["x"][0], 0.0, 0.0],
            [tilt[0], bounds["y"][1] - bounds["y"][0], 0.0],
            [tilt[1], tilt[2], bounds["z"][1] - bounds["z"][0]],
        ],
        dtype=np.float64,
    )
    symbols, coords, id_to_idx = _parse_lammps_atom_rows(atom_rows, masses)

    bonds: list[structure.Bond] = []
    for row in bond_rows:
        if len(row) >= 4:
            i_id, j_id = int(row[2]), int(row[3])
            if i_id in id_to_idx and j_id in id_to_idx:
                bonds.append(
                    structure.Bond(
                        i=id_to_idx[i_id], j=id_to_idx[j_id], order=1.0
                    )
                )
    return structure.AtomsData(
        symbols=symbols,
        positions=np.array(coords, dtype=np.float64),
        cell=cell,
        pbc=(True, True, True),
        bonds=bonds,
        metadata={"source": str(p)},
    )


def _parse_lammpstrj_frame(
    lines: list[str],
    idx: int,
    source: str,
) -> tuple[structure.AtomsData, float, int]:
    """Parses one ``ITEM: TIMESTEP`` block from a LAMMPS dump trajectory."""
    timestep = int(lines[idx + 1].strip())
    idx += 2
    if not lines[idx].startswith("ITEM: NUMBER OF ATOMS"):
        raise exceptions.ParseError(
            f"Expected NUMBER OF ATOMS at line {idx + 1}."
        )
    n_atoms = int(lines[idx + 1].strip())
    idx += 2
    bflags = lines[idx].replace("ITEM: BOX BOUNDS", "").split()
    pbc_flags = (
        tuple(flag.startswith("p") for flag in bflags[:3])
        if len(bflags) >= 3
        else (True, True, True)
    )
    b_rows = [[float(v) for v in lines[idx + 1 + k].split()] for k in range(3)]
    origin = np.array(
        [b_rows[0][0], b_rows[1][0], b_rows[2][0]], dtype=np.float64
    )
    cell = np.diag(
        [
            b_rows[0][1] - b_rows[0][0],
            b_rows[1][1] - b_rows[1][0],
            b_rows[2][1] - b_rows[2][0],
        ]
    ).astype(np.float64)
    idx += 4

    cols = lines[idx].replace("ITEM: ATOMS", "").split()
    idx += 1
    rows = [lines[idx + k].split() for k in range(n_atoms)]
    col_idx = {name: k for k, name in enumerate(cols)}
    is_scaled = "xs" in col_idx
    xk, yk, zk = (
        ("xs", "ys", "zs")
        if is_scaled
        else (("xu", "yu", "zu") if "xu" in col_idx else ("x", "y", "z"))
    )
    if "id" in col_idx:
        rows.sort(key=lambda r: int(r[col_idx["id"]]))

    symbols = [
        (
            elements.get_element(r[col_idx["element"]]).symbol
            if "element" in col_idx
            else "X"
        )
        for r in rows
    ]
    pos = np.array(
        [
            [
                float(r[col_idx[xk]]),
                float(r[col_idx[yk]]),
                float(r[col_idx[zk]]),
            ]
            for r in rows
        ],
        dtype=np.float64,
    )
    if is_scaled:
        pos = pos @ cell + origin
    atoms = structure.AtomsData(
        symbols=symbols,
        positions=pos,
        cell=cell,
        pbc=(pbc_flags[0], pbc_flags[1], pbc_flags[2]),
        metadata={"timestep": timestep, "source": source},
    )
    return atoms, float(timestep), idx + n_atoms


def read_lammpstrj(path: str | pathlib.Path) -> trajectory.Trajectory:
    """Reads a multi-frame ``.lammpstrj`` dump into a ``Trajectory``."""
    p = pathlib.Path(path)
    lines = p.read_text(encoding="utf-8").splitlines()
    frames: list[structure.AtomsData] = []
    times: list[float] = []
    idx = 0
    while idx < len(lines):
        if not lines[idx].startswith("ITEM: TIMESTEP"):
            idx += 1
            continue
        frame, t_val, idx = _parse_lammpstrj_frame(lines, idx, str(p))
        frames.append(frame)
        times.append(t_val)

    if not frames:
        raise exceptions.ParseError(f"No LAMMPS dump frames found in {p}.")
    return trajectory.Trajectory(
        frames=frames,
        times=np.array(times, dtype=np.float64),
        metadata={"source": str(p)},
    )


def read_lammpstrj_first(path: str | pathlib.Path) -> structure.AtomsData:
    """Reads the first snapshot from a ``.lammpstrj`` dump file."""
    return read_lammpstrj(path)[0]


registry.register_format(
    registry.FormatSpec(
        name="lammps_data",
        extensions=(".lmp", ".data"),
        filenames=("data.lammps",),
        reader=read_lammps_data,
        description="LAMMPS read_data atomic topology and box",
    )
)
registry.register_format(
    registry.FormatSpec(
        name="lammpstrj",
        extensions=(".lammpstrj",),
        filenames=(),
        reader=read_lammpstrj_first,
        description="LAMMPS dump trajectory",
    )
)
