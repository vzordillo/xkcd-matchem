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


_LAMMPS_ATOM_STYLES: dict[str, tuple[int, int, int]] = {
    # style -> (min_columns, type_column_index, x_start_column_index)
    "atomic": (5, 1, 2),
    "charge": (6, 1, 3),
    "bond": (6, 2, 3),
    "angle": (6, 2, 3),
    "molecular": (6, 2, 3),
    "full": (7, 2, 4),
}


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


def _resolve_lammps_style(
    row: list[str], atom_style: str | None
) -> tuple[int, int, int]:
    """Resolves column layout for a LAMMPS data Atoms row."""
    if atom_style is not None:
        key = atom_style.strip().lower()
        if key not in _LAMMPS_ATOM_STYLES:
            supported = ", ".join(sorted(_LAMMPS_ATOM_STYLES))
            raise exceptions.ParseError(
                f"Unsupported LAMMPS atom_style {atom_style!r}; "
                f"supported styles: {supported}."
            )
        spec = _LAMMPS_ATOM_STYLES[key]
        if len(row) < spec[0]:
            raise exceptions.ParseError(
                f"LAMMPS Atoms row has {len(row)} columns, "
                f"expected at least {spec[0]} for atom_style={key!r}."
            )
        return spec
    if len(row) == 5:
        return _LAMMPS_ATOM_STYLES["atomic"]
    if len(row) == 7:
        return _LAMMPS_ATOM_STYLES["full"]
    raise exceptions.ParseError(
        f"Ambiguous LAMMPS Atoms row with {len(row)} columns; "
        "specify '# <atom_style>' on the Atoms header or pass atom_style=..."
    )


def _parse_lammps_atom_rows(
    atom_rows: list[list[str]],
    masses: dict[int, float],
    atom_style: str | None = None,
    type_map: dict[int, str] | None = None,
) -> tuple[list[str], list[list[float]], dict[int, int]]:
    """Extracts symbols, coordinates, and ID mapping from LAMMPS data rows."""
    id_to_idx: dict[int, int] = {}
    symbols: list[str] = []
    coords: list[list[float]] = []
    for idx, row in enumerate(atom_rows):
        min_cols, type_col, xyz_col = _resolve_lammps_style(row, atom_style)
        if len(row) < min_cols:
            raise exceptions.ParseError(
                f"Unsupported LAMMPS Atoms row: {row!r}"
            )
        id_to_idx[int(row[0])] = idx
        atype = int(row[type_col])
        xyz = [
            float(row[xyz_col]),
            float(row[xyz_col + 1]),
            float(row[xyz_col + 2]),
        ]
        if type_map is not None:
            if atype not in type_map:
                raise exceptions.ParseError(
                    f"LAMMPS atom type {atype} missing from type_map."
                )
            sym = elements.get_element(type_map[atype]).symbol
        elif atype in masses:
            sym = _symbol_from_mass(masses[atype])
        else:
            raise exceptions.ParseError(
                f"Missing mass or type_map entry for LAMMPS atom type {atype}."
            )
        symbols.append(sym)
        coords.append(xyz)
    return symbols, coords, id_to_idx


def read_lammps_data(
    path: str | pathlib.Path,
    atom_style: str | None = None,
    type_map: dict[int, str] | None = None,
) -> structure.AtomsData:
    """Reads a LAMMPS ``read_data`` topology/coordinate file."""
    p = pathlib.Path(path)
    lines = p.read_text(encoding="utf-8").splitlines()
    bounds = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)}
    tilt = [0.0, 0.0, 0.0]
    masses: dict[int, float] = {}
    atom_rows: list[list[str]] = []
    bond_rows: list[list[str]] = []
    section = ""
    detected_style = atom_style

    for raw in lines:
        head, _, tail = raw.partition("#")
        line = head.strip()
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
        first_tok = line.split()[0]
        if first_tok in ("Masses", "Atoms", "Bonds", "Velocities", "Angles"):
            section = first_tok
            if first_tok == "Atoms" and detected_style is None:
                comment_tok = tail.strip().split()
                if comment_tok:
                    detected_style = comment_tok[0].lower()
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
    symbols, coords, id_to_idx = _parse_lammps_atom_rows(
        atom_rows,
        masses,
        atom_style=detected_style,
        type_map=type_map,
    )

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


def _lammpstrj_cell_and_origin(
    bflags: list[str], b_rows: list[list[float]]
) -> tuple[np.ndarray, np.ndarray, tuple[bool, bool, bool]]:
    """Extracts PBC flags, box origin, and cell matrix from BOX BOUNDS."""
    pbc_tokens = bflags[-3:] if len(bflags) >= 3 else []
    pbc_flags = (
        tuple(flag.startswith("p") for flag in pbc_tokens)
        if len(pbc_tokens) == 3
        else (True, True, True)
    )
    is_triclinic = "xy" in bflags or all(len(r) >= 3 for r in b_rows)
    xy = b_rows[0][2] if is_triclinic and len(b_rows[0]) >= 3 else 0.0
    xz = b_rows[1][2] if is_triclinic and len(b_rows[1]) >= 3 else 0.0
    yz = b_rows[2][2] if is_triclinic and len(b_rows[2]) >= 3 else 0.0
    xlo = b_rows[0][0] - min(0.0, xy, xz, xy + xz)
    xhi = b_rows[0][1] - max(0.0, xy, xz, xy + xz)
    ylo = b_rows[1][0] - min(0.0, yz)
    yhi = b_rows[1][1] - max(0.0, yz)
    zlo, zhi = b_rows[2][0], b_rows[2][1]
    origin = np.array([xlo, ylo, zlo], dtype=np.float64)
    cell = np.array(
        [
            [xhi - xlo, 0.0, 0.0],
            [xy, yhi - ylo, 0.0],
            [xz, yz, zhi - zlo],
        ],
        dtype=np.float64,
    )
    return (
        origin,
        cell,
        (bool(pbc_flags[0]), bool(pbc_flags[1]), bool(pbc_flags[2])),
    )


def _lammpstrj_symbols(
    rows: list[list[str]],
    col_idx: dict[str, int],
    type_map: dict[int, str] | None,
) -> list[str]:
    """Resolves element symbols from LAMMPS dump columns and optional map."""
    if "element" in col_idx:
        idx = col_idx["element"]
        return [elements.get_element(r[idx]).symbol for r in rows]
    if "type" in col_idx and type_map is not None:
        t_idx = col_idx["type"]
        syms: list[str] = []
        for row in rows:
            atype = int(row[t_idx])
            if atype not in type_map:
                raise exceptions.ParseError(
                    f"LAMMPS dump atom type {atype} missing from type_map."
                )
            syms.append(elements.get_element(type_map[atype]).symbol)
        return syms
    raise exceptions.ParseError(
        "LAMMPS dump lacks 'element' column; provide type_map={id: symbol}."
    )


def _parse_lammpstrj_frame(
    lines: list[str],
    idx: int,
    source: str,
    type_map: dict[int, str] | None = None,
) -> tuple[structure.AtomsData, float, int]:
    """Parses one ``ITEM: TIMESTEP`` block from a LAMMPS dump trajectory."""
    if idx + 9 >= len(lines):
        raise exceptions.ParseError(f"Truncated LAMMPS dump frame in {source}.")
    timestep = int(lines[idx + 1].strip())
    idx += 2
    if not lines[idx].startswith("ITEM: NUMBER OF ATOMS"):
        raise exceptions.ParseError(
            f"Expected NUMBER OF ATOMS at line {idx + 1}."
        )
    n_atoms = int(lines[idx + 1].strip())
    if n_atoms < 0:
        raise exceptions.ParseError(
            f"Negative atom count {n_atoms} in {source}."
        )
    idx += 2
    bflags = lines[idx].replace("ITEM: BOX BOUNDS", "").split()
    b_rows = [[float(v) for v in lines[idx + 1 + k].split()] for k in range(3)]
    origin, cell, pbc_tuple = _lammpstrj_cell_and_origin(bflags, b_rows)
    idx += 4

    cols = lines[idx].replace("ITEM: ATOMS", "").split()
    idx += 1
    if idx + n_atoms > len(lines):
        raise exceptions.ParseError(
            f"Truncated LAMMPS dump atom table at line {idx + 1}."
        )
    rows = [lines[idx + k].split() for k in range(n_atoms)]
    col_idx = {name: k for k, name in enumerate(cols)}
    is_scaled = "xs" in col_idx or "xsu" in col_idx
    if "xsu" in col_idx:
        xk, yk, zk = "xsu", "ysu", "zsu"
    elif "xs" in col_idx:
        xk, yk, zk = "xs", "ys", "zs"
    elif "xu" in col_idx:
        xk, yk, zk = "xu", "yu", "zu"
    else:
        xk, yk, zk = "x", "y", "z"
    if any(k not in col_idx for k in (xk, yk, zk)):
        raise exceptions.ParseError(
            f"Missing coordinate columns in LAMMPS dump {source}."
        )
    if "id" in col_idx:
        rows.sort(key=lambda r: int(r[col_idx["id"]]))

    symbols = _lammpstrj_symbols(rows, col_idx, type_map)
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
        pbc=pbc_tuple,
        metadata={"timestep": timestep, "source": source},
    )
    return atoms, float(timestep), idx + n_atoms


def read_lammpstrj(
    path: str | pathlib.Path,
    type_map: dict[int, str] | None = None,
) -> trajectory.Trajectory:
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
        frame, t_val, idx = _parse_lammpstrj_frame(
            lines, idx, str(p), type_map=type_map
        )
        frames.append(frame)
        times.append(t_val)

    if not frames:
        raise exceptions.ParseError(f"No LAMMPS dump frames found in {p}.")
    return trajectory.Trajectory(
        frames=frames,
        times=np.array(times, dtype=np.float64),
        metadata={"source": str(p)},
    )


def read_lammpstrj_first(
    path: str | pathlib.Path,
    type_map: dict[int, str] | None = None,
) -> structure.AtomsData:
    """Reads only the first snapshot from a ``.lammpstrj`` dump file."""
    p = pathlib.Path(path)
    lines = p.read_text(encoding="utf-8").splitlines()
    idx = 0
    while idx < len(lines) and not lines[idx].startswith("ITEM: TIMESTEP"):
        idx += 1
    if idx >= len(lines):
        raise exceptions.ParseError(f"No LAMMPS dump frames found in {p}.")
    frame, _, _ = _parse_lammpstrj_frame(lines, idx, str(p), type_map=type_map)
    return frame


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
