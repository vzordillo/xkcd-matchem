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

from collections.abc import Iterator
import math
import pathlib
from typing import Any

import numpy as np

from xkcd_matchem.core import exceptions
from xkcd_matchem.core import structure
from xkcd_matchem.core import trajectory
from xkcd_matchem.core import units
from xkcd_matchem.data import elements
from xkcd_matchem.io import registry
from xkcd_matchem.io.native import _limits


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
    if len(row) in (5, 8):
        return _LAMMPS_ATOM_STYLES["atomic"]
    if len(row) in (7, 10):
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
) -> tuple[list[str], list[list[float]], dict[int, int], np.ndarray | None]:
    """Extracts symbols, coordinates, IDs, & optional image flags."""
    id_to_idx: dict[int, int] = {}
    symbols: list[str] = []
    coords: list[list[float]] = []
    img_rows: list[list[int]] = []
    has_images = True
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
        if len(row) >= xyz_col + 6:
            img_rows.append(
                [
                    int(row[xyz_col + 3]),
                    int(row[xyz_col + 4]),
                    int(row[xyz_col + 5]),
                ]
            )
        else:
            has_images = False
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
    img_arr = (
        np.array(img_rows, dtype=np.int64) if has_images and img_rows else None
    )
    return symbols, coords, id_to_idx, img_arr


def _build_lammps_bonds(
    bond_rows: list[list[str]],
    id_to_idx: dict[int, int],
) -> list[structure.Bond]:
    """Maps LAMMPS Bonds table rows into indexed ``Bond`` instances."""
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
    return bonds


def _read_lammps_data_sections(
    p: pathlib.Path,
    atom_style: str | None,
) -> tuple[
    np.ndarray,
    dict[int, float],
    list[list[str]],
    list[list[str]],
    str | None,
]:
    """Streams header bounds, Masses, Atoms, and Bonds from a LAMMPS file."""
    atom_limit = _limits.max_atoms()
    bounds = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)}
    tilt = [0.0, 0.0, 0.0]
    masses: dict[int, float] = {}
    atom_rows: list[list[str]] = []
    bond_rows: list[list[str]] = []
    section = ""
    detected_style = atom_style

    with p.open("r", encoding="utf-8") as fh:
        for raw_idx, raw in enumerate(fh):
            head, _, tail = raw.partition("#")
            line = head.strip()
            if not line:
                continue
            parts = line.split()
            if section == "" and len(parts) == 2 and parts[1] == "atoms":
                try:
                    _limits.check_atom_count(int(parts[0]), source=p)
                except ValueError as exc:
                    if raw_idx > 0:
                        raise exceptions.ParseError(
                            f"Invalid LAMMPS atom count {line!r} in {p}."
                        ) from exc
                else:
                    continue
            if line.endswith(("xlo xhi", "ylo yhi", "zlo zhi")):
                bounds[parts[2][0]] = (float(parts[0]), float(parts[1]))
            elif line.endswith("xy xz yz"):
                tilt = [float(parts[0]), float(parts[1]), float(parts[2])]
            elif parts[0] in (
                "Masses",
                "Atoms",
                "Bonds",
                "Velocities",
                "Angles",
            ):
                section = parts[0]
                if section == "Atoms" and detected_style is None:
                    comment_tok = tail.strip().split()
                    if comment_tok:
                        detected_style = comment_tok[0].lower()
            elif section == "Masses" and len(parts) >= 2:
                masses[int(parts[0])] = float(parts[1])
            elif section == "Atoms":
                if len(atom_rows) + 1 > atom_limit:
                    _limits.check_atom_count(len(atom_rows) + 1, source=p)
                atom_rows.append(parts)
            elif section == "Bonds":
                bond_rows.append(parts)

    cell = np.array(
        [
            [bounds["x"][1] - bounds["x"][0], 0.0, 0.0],
            [tilt[0], bounds["y"][1] - bounds["y"][0], 0.0],
            [tilt[1], tilt[2], bounds["z"][1] - bounds["z"][0]],
        ],
        dtype=np.float64,
    )
    return cell, masses, atom_rows, bond_rows, detected_style


def read_lammps_data(
    path: str | pathlib.Path,
    atom_style: str | None = None,
    type_map: dict[int, str] | None = None,
) -> structure.AtomsData:
    """Reads a LAMMPS ``read_data`` topology/coordinate file."""
    p = pathlib.Path(path)
    _limits.check_file_size(p)
    cell, masses, atom_rows, bond_rows, detected_style = (
        _read_lammps_data_sections(p, atom_style)
    )
    if not atom_rows:
        raise exceptions.ParseError(
            f"No Atoms section in LAMMPS data file: {p}"
        )
    _limits.check_atom_count(len(atom_rows), source=p)
    symbols, coords, id_to_idx, img_arr = _parse_lammps_atom_rows(
        atom_rows,
        masses,
        atom_style=detected_style,
        type_map=type_map,
    )
    pos = np.array(coords, dtype=np.float64)
    meta: dict[str, Any] = {"source": str(p)}
    if img_arr is not None:
        meta["image_flags"] = img_arr
        meta["unwrapped_positions"] = pos + img_arr.astype(np.float64) @ cell
    return structure.AtomsData(
        symbols=symbols,
        positions=pos,
        cell=cell,
        pbc=(True, True, True),
        bonds=_build_lammps_bonds(bond_rows, id_to_idx),
        metadata=meta,
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


def _extract_lammpstrj_positions(
    rows: list[list[str]],
    col_idx: dict[str, int],
    origin: np.ndarray,
    cell: np.ndarray,
    source: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Extracts wrapped/unwrapped Cartesian positions and image flags."""
    has_wrap_cart = all(k in col_idx for k in ("x", "y", "z"))
    has_wrap_scal = all(k in col_idx for k in ("xs", "ys", "zs"))
    has_unwrap_cart = all(k in col_idx for k in ("xu", "yu", "zu"))
    has_unwrap_scal = all(k in col_idx for k in ("xsu", "ysu", "zsu"))
    has_images = all(k in col_idx for k in ("ix", "iy", "iz"))
    if not (
        has_wrap_cart or has_wrap_scal or has_unwrap_cart or has_unwrap_scal
    ):
        raise exceptions.ParseError(
            f"Missing coordinate columns in LAMMPS dump {source}."
        )

    def _read_xyz(kx: str, ky: str, kz: str, *, scaled: bool) -> np.ndarray:
        ix_c, iy_c, iz_c = col_idx[kx], col_idx[ky], col_idx[kz]
        arr = np.array(
            [[float(r[ix_c]), float(r[iy_c]), float(r[iz_c])] for r in rows],
            dtype=np.float64,
        )
        if scaled:
            return np.asarray(arr @ cell + origin, dtype=np.float64)
        return arr

    extra_meta: dict[str, Any] = {}
    if has_images:
        ix_i, iy_i, iz_i = col_idx["ix"], col_idx["iy"], col_idx["iz"]
        img = np.array(
            [[int(r[ix_i]), int(r[iy_i]), int(r[iz_i])] for r in rows],
            dtype=np.int64,
        )
        extra_meta["image_flags"] = img

    if has_wrap_scal:
        pos = _read_xyz("xs", "ys", "zs", scaled=True)
    elif has_wrap_cart:
        pos = _read_xyz("x", "y", "z", scaled=False)
    elif has_unwrap_scal:
        pos = _read_xyz("xsu", "ysu", "zsu", scaled=True)
        extra_meta["coords_unwrapped"] = True
    else:
        pos = _read_xyz("xu", "yu", "zu", scaled=False)
        extra_meta["coords_unwrapped"] = True

    if has_unwrap_cart:
        extra_meta["unwrapped_positions"] = _read_xyz(
            "xu", "yu", "zu", scaled=False
        )
    elif has_unwrap_scal:
        extra_meta["unwrapped_positions"] = _read_xyz(
            "xsu", "ysu", "zsu", scaled=True
        )
    elif has_images:
        extra_meta["unwrapped_positions"] = (
            pos + extra_meta["image_flags"].astype(np.float64) @ cell
        )
    return pos, extra_meta


def _next_stream_line(
    line_iter: Iterator[tuple[int, str]],
    err_msg: str,
) -> tuple[int, str]:
    """Reads the next ``(line_no, line)`` pair from a streaming dump file."""
    try:
        return next(line_iter)
    except StopIteration as exc:
        raise exceptions.ParseError(err_msg) from exc


def _stream_lammpstrj_box(
    line_iter: Iterator[tuple[int, str]],
    source: pathlib.Path,
    trunc_msg: str,
) -> tuple[np.ndarray, np.ndarray, tuple[bool, bool, bool]]:
    """Streams ``ITEM: BOX BOUNDS`` and its 3 row bounds from a dump file."""
    box_line_no, box_hdr = _next_stream_line(line_iter, trunc_msg)
    if not box_hdr.startswith("ITEM: BOX BOUNDS"):
        raise exceptions.ParseError(
            f"Expected BOX BOUNDS at line {box_line_no} in {source}."
        )
    bflags = box_hdr.replace("ITEM: BOX BOUNDS", "").split()
    b_rows = [
        [float(v) for v in _next_stream_line(line_iter, trunc_msg)[1].split()]
        for _ in range(3)
    ]
    return _lammpstrj_cell_and_origin(bflags, b_rows)


def _stream_lammpstrj_frame(
    line_iter: Iterator[tuple[int, str]],
    source: pathlib.Path,
    type_map: dict[int, str] | None = None,
    *,
    frame_index: int | None = None,
) -> tuple[structure.AtomsData, int] | None:
    """Streams one ``ITEM: TIMESTEP`` block from an open LAMMPS dump file."""
    for _, raw in line_iter:
        if raw.startswith("ITEM: TIMESTEP"):
            break
    else:
        return None

    if frame_index is not None:
        _limits.check_frame_count(frame_index + 1, source=source)

    trunc_msg = f"Truncated LAMMPS dump frame in {source}."
    _, ts_line = _next_stream_line(line_iter, trunc_msg)
    try:
        timestep = int(ts_line.strip())
    except ValueError as exc:
        raise exceptions.ParseError(
            f"Invalid LAMMPS timestep {ts_line.strip()!r} in {source}."
        ) from exc

    nat_line_no, nat_hdr = _next_stream_line(line_iter, trunc_msg)
    if not nat_hdr.startswith("ITEM: NUMBER OF ATOMS"):
        raise exceptions.ParseError(
            f"Expected NUMBER OF ATOMS at line {nat_line_no}."
        )
    _, nat_line = _next_stream_line(line_iter, trunc_msg)
    try:
        n_atoms = int(nat_line.strip())
    except ValueError as exc:
        raise exceptions.ParseError(
            f"Invalid atom count {nat_line.strip()!r} in {source}."
        ) from exc
    _limits.check_atom_count(n_atoms, source=source)

    origin, cell, pbc_tuple = _stream_lammpstrj_box(
        line_iter, source, trunc_msg
    )

    at_line_no, at_hdr = _next_stream_line(line_iter, trunc_msg)
    if not at_hdr.startswith("ITEM: ATOMS"):
        raise exceptions.ParseError(
            f"Expected ITEM: ATOMS at line {at_line_no} in {source}."
        )
    cols = at_hdr.replace("ITEM: ATOMS", "").split()
    table_err = f"Truncated LAMMPS dump atom table at line {at_line_no + 1}."
    rows = [
        _next_stream_line(line_iter, table_err)[1].split()
        for _ in range(n_atoms)
    ]
    col_idx = {name: k for k, name in enumerate(cols)}
    if "id" in col_idx:
        rows.sort(key=lambda r: int(r[col_idx["id"]]))

    src_str = str(source)
    pos, extra_meta = _extract_lammpstrj_positions(
        rows, col_idx, origin, cell, src_str
    )
    frame_meta: dict[str, Any] = {"timestep": timestep, "source": src_str}
    frame_meta.update(extra_meta)
    return (
        structure.AtomsData(
            symbols=_lammpstrj_symbols(rows, col_idx, type_map),
            positions=pos,
            cell=cell,
            pbc=pbc_tuple,
            metadata=frame_meta,
        ),
        timestep,
    )


def read_lammpstrj(
    path: str | pathlib.Path,
    type_map: dict[int, str] | None = None,
    *,
    timestep_size: float | None = None,
    time_unit: str = "fs",
) -> trajectory.Trajectory:
    """Streams a multi-frame ``.lammpstrj`` dump into a ``Trajectory``.

    Args:
      path: Path to a LAMMPS ``.lammpstrj`` dump file.
      type_map: Optional ``{type_id: element_symbol}`` mapping when the dump
        lacks an ``element`` column.
      timestep_size: Optional MD integration step size per LAMMPS step index.
        When ``None`` (default), ``Trajectory.times`` is ``None`` and integer
        step counters are stored in ``metadata["timesteps"]``.
      time_unit: Unit of ``timestep_size`` (for example ``"fs"`` or ``"ps"``),
        converted to canonical ``"fs"`` when ``timestep_size`` is supplied.

    Returns:
      Parsed ``Trajectory`` instance.
    """
    if timestep_size is not None and (
        isinstance(timestep_size, bool)
        or not (math.isfinite(timestep_size) and timestep_size > 0.0)
    ):
        raise exceptions.ParseError(
            f"Invalid timestep_size={timestep_size!r}; expected a finite "
            "positive number."
        )
    p = pathlib.Path(path)
    _limits.check_file_size(p)
    frames: list[structure.AtomsData] = []
    timesteps: list[int] = []
    with p.open("r", encoding="utf-8") as fh:
        line_iter = (
            (line_no, raw.rstrip("\r\n"))
            for line_no, raw in enumerate(fh, start=1)
        )
        while True:
            parsed = _stream_lammpstrj_frame(
                line_iter,
                p,
                type_map=type_map,
                frame_index=len(frames),
            )
            if parsed is None:
                break
            _limits.check_frame_count(len(frames) + 1, source=p)
            frame, step = parsed
            frames.append(frame)
            timesteps.append(step)

    if not frames:
        raise exceptions.ParseError(f"No LAMMPS dump frames found in {p}.")
    step_arr = np.array(timesteps, dtype=np.int64)
    meta: dict[str, Any] = {"source": str(p), "timesteps": step_arr}
    times: np.ndarray | None = None
    if timestep_size is None:
        meta["time_unit"] = None
    else:
        dt = float(timestep_size)
        times = units.convert_time(
            step_arr.astype(np.float64) * dt, time_unit, "fs"
        )
        meta["time_unit"] = "fs"
        meta["timestep_size"] = dt
        meta["timestep_unit"] = str(time_unit)
    return trajectory.Trajectory(
        frames=frames,
        times=times,
        metadata=meta,
    )


def read_lammpstrj_first(
    path: str | pathlib.Path,
    type_map: dict[int, str] | None = None,
) -> structure.AtomsData:
    """Streams only the first snapshot from a ``.lammpstrj`` dump file."""
    p = pathlib.Path(path)
    _limits.check_file_size(p)
    with p.open("r", encoding="utf-8") as fh:
        line_iter = (
            (line_no, raw.rstrip("\r\n"))
            for line_no, raw in enumerate(fh, start=1)
        )
        parsed = _stream_lammpstrj_frame(line_iter, p, type_map=type_map)
    if parsed is None:
        raise exceptions.ParseError(f"No LAMMPS dump frames found in {p}.")
    return parsed[0]


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
