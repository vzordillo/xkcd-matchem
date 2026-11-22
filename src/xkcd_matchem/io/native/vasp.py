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
import re

import numpy as np

from xkcd_matchem.core import electronic
from xkcd_matchem.core import exceptions
from xkcd_matchem.core import structure
from xkcd_matchem.core import trajectory
from xkcd_matchem.io import registry


def _split_touching_floats(text: str) -> str:
    """Separates concatenated negative floats such as ``1.23-4.56``."""
    return re.sub(r"(?<=[0-9])-(?=[0-9.])", " -", text)


def _parse_poscar_header(
    lines: list[str],
) -> tuple[str, np.ndarray, np.ndarray, list[str], bool, int]:
    """Parses the common VASP 4/5 header lines through coordinate mode."""
    if len(lines) < 8:
        raise exceptions.ParseError("VASP POSCAR header is too short.")
    comment = lines[0].strip()
    scale_tokens = [float(tok) for tok in lines[1].strip().split()]
    if not scale_tokens or any(abs(v) < 1e-12 for v in scale_tokens):
        raise exceptions.ParseError(
            "VASP POSCAR scaling factor cannot be zero."
        )
    raw_cell = np.array(
        [[float(x) for x in lines[k].split()[:3]] for k in range(2, 5)],
        dtype=np.float64,
    )
    raw_vol = abs(float(np.linalg.det(raw_cell)))
    if raw_vol < 1e-12:
        raise exceptions.ParseError(
            "VASP POSCAR lattice matrix has zero determinant."
        )
    if len(scale_tokens) >= 3:
        scale_vec = np.asarray(scale_tokens[:3], dtype=np.float64)
        cell = raw_cell * scale_vec[None, :]
    else:
        scale_val = scale_tokens[0]
        factor = (
            (-scale_val / raw_vol) ** (1.0 / 3.0)
            if scale_val < 0.0
            else scale_val
        )
        scale_vec = np.array([factor, factor, factor], dtype=np.float64)
        cell = raw_cell * factor

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
    return comment, cell, scale_vec, symbols, is_cartesian, idx + 1


def read_poscar(path: str | pathlib.Path) -> structure.AtomsData:
    """Reads a VASP 4/5 ``POSCAR`` or ``CONTCAR`` structure file."""
    p = pathlib.Path(path)
    lines = p.read_text(encoding="utf-8").splitlines()
    comment, cell, scale_vec, symbols, is_cart, start_row = (
        _parse_poscar_header(lines)
    )
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
            positions=coords * scale_vec[None, :],
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
    comment, cell, _, symbols, _, cursor = _parse_poscar_header(lines)
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
        [float(x) for x in _split_touching_floats(line).split()]
        for line in lines[6 : 6 + nedos]
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


def read_eigenval(
    path: str | pathlib.Path,
    cell: np.ndarray | None = None,
) -> electronic.BandStructure:
    """Reads band eigenvalues from a VASP ``EIGENVAL`` file.

    Args:
      path: Path to ``EIGENVAL``.
      cell: Optional ``(3, 3)`` real-space lattice matrix in Angstrom used to
        convert fractional k-points into physical reciprocal-space distances
        ``B = 2 * pi * inv(A).T``.

    Returns:
      Parsed ``BandStructure`` with explicit ``k_coordinate_mode`` metadata.
    """
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
    if cell is not None:
        b_mat = (
            2.0 * np.pi * np.linalg.inv(np.asarray(cell, dtype=np.float64)).T
        )
        diffs = diffs @ b_mat
        k_mode = "reciprocal_angstrom_inv"
    else:
        k_mode = "fractional"
    k_dist = np.concatenate([[0.0], np.cumsum(np.linalg.norm(diffs, axis=1))])
    return electronic.BandStructure(
        k_distances=k_dist,
        eigenvalues=eig,
        metadata={"source": str(p), "k_coordinate_mode": k_mode},
    )


def read_chgcar(path: str | pathlib.Path) -> electronic.VolumetricGrid:
    """Reads a VASP ``CHGCAR`` or ``LOCPOT`` 3D volumetric grid file."""
    p = pathlib.Path(path)
    lines = p.read_text(encoding="utf-8").splitlines()
    comment, cell, scale_vec, symbols, is_cart, start_row = (
        _parse_poscar_header(lines)
    )
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
            positions=coords * scale_vec[None, :],
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


def _orbital_group_map(headers: list[str]) -> dict[str, list[int]]:
    """Groups PROCAR orbital column headers into s, p, d, f, and tot indices."""
    groups: dict[str, list[int]] = {
        "s": [],
        "p": [],
        "d": [],
        "f": [],
        "tot": [],
    }
    for idx, col in enumerate(headers):
        key = col.strip().lower()
        if key == "ion":
            continue
        if key == "tot":
            groups["tot"].append(idx)
        elif key == "s":
            groups["s"].append(idx)
        elif key.startswith("p"):
            groups["p"].append(idx)
        elif key.startswith("d") or key == "x2-y2":
            groups["d"].append(idx)
        elif key.startswith("f"):
            groups["f"].append(idx)
    return {k: v for k, v in groups.items() if v}


def _parse_procar_band(
    lines: list[str],
    cursor: int,
    n_ions: int,
) -> tuple[float, float, dict[str, float], int]:
    """Parses one PROCAR band block -> ``(energy, occ, weights, next_idx)``."""
    band_hdr = lines[cursor]
    m_en = re.search(r"energy\s+([-+]?\d*\.?\d+)", band_hdr)
    m_occ = re.search(r"occ\.\s+([-+]?\d*\.?\d+)", band_hdr)
    energy = float(m_en.group(1)) if m_en else 0.0
    occ = float(m_occ.group(1)) if m_occ else 0.0
    cursor += 1
    while cursor < len(lines) and not lines[cursor].strip().startswith("ion"):
        cursor += 1
    if cursor >= len(lines):
        return energy, occ, {}, cursor
    col_hdrs = lines[cursor].split()
    orb_groups = _orbital_group_map(col_hdrs)
    cursor += 1

    tot_vals: list[float] | None = None
    ion_sum = np.zeros(len(col_hdrs), dtype=np.float64)
    seen_ions = 0
    while cursor < len(lines):
        row = lines[cursor].strip()
        if not row:
            cursor += 1
            if tot_vals is not None or seen_ions >= n_ions:
                break
            continue
        if row.startswith(("band ", "k-point ", "# of k-points:")):
            break
        parts = row.split()
        if parts[0].lower() == "tot":
            tot_vals = [float(x) for x in parts[1:]]
            cursor += 1
            break
        if parts[0].isdigit() and len(parts) >= len(col_hdrs):
            ion_sum[1 : len(col_hdrs)] += [
                float(x) for x in parts[1 : len(col_hdrs)]
            ]
            seen_ions += 1
        cursor += 1

    weights: dict[str, float] = {}
    vec = (
        np.array([0.0, *tot_vals], dtype=np.float64)
        if tot_vals is not None
        else ion_sum
    )
    for orb_name, cols in orb_groups.items():
        valid_cols = [c for c in cols if c < len(vec)]
        weights[orb_name] = float(np.sum(vec[valid_cols]))
    return energy, occ, weights, cursor


def _parse_procar_spin_block(
    lines: list[str],
    idx: int,
    source: pathlib.Path,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], int
]:
    """Parses one spin block from a VASP ``PROCAR`` file."""
    header = lines[idx]
    m_k = re.search(r"k-points:\s*(\d+)", header)
    m_b = re.search(r"bands:\s*(\d+)", header)
    m_i = re.search(r"ions:\s*(\d+)", header)
    if not (m_k and m_b and m_i):
        raise exceptions.ParseError(f"Malformed PROCAR header in {source}.")
    n_k, n_b, n_i = int(m_k.group(1)), int(m_b.group(1)), int(m_i.group(1))
    k_coords = np.zeros((n_k, 3), dtype=np.float64)
    eig = np.zeros((n_k, n_b), dtype=np.float64)
    occ = np.zeros((n_k, n_b), dtype=np.float64)
    proj: dict[str, np.ndarray] = {}
    idx += 1
    for ik in range(n_k):
        while idx < len(lines) and "k-point" not in lines[idx]:
            idx += 1
        if idx >= len(lines):
            raise exceptions.ParseError(
                f"Truncated PROCAR k-points in {source}."
            )
        k_rhs = _split_touching_floats(lines[idx].split(":", 1)[1])
        k_coords[ik] = [float(x) for x in k_rhs.split()[:3]]
        idx += 1
        for ib in range(n_b):
            while idx < len(lines) and not lines[idx].strip().startswith(
                "band "
            ):
                idx += 1
            en, oc, w_map, idx = _parse_procar_band(lines, idx, n_i)
            eig[ik, ib] = en
            occ[ik, ib] = oc
            for orb_key, val in w_map.items():
                if orb_key not in proj:
                    proj[orb_key] = np.zeros((n_k, n_b), dtype=np.float64)
                proj[orb_key][ik, ib] = val
    return k_coords, eig, occ, proj, idx


def read_procar(
    path: str | pathlib.Path,
    cell: np.ndarray | None = None,
) -> electronic.BandStructure:
    """Reads orbital-projected bands from a VASP ``PROCAR`` file."""
    p = pathlib.Path(path)
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    spin_blocks: list[
        tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]
    ] = []
    idx = 0
    while idx < len(lines):
        if "# of k-points:" not in lines[idx]:
            idx += 1
            continue
        k_c, eig_b, occ_b, proj_b, idx = _parse_procar_spin_block(
            lines, idx, p
        )
        spin_blocks.append((k_c, eig_b, occ_b, proj_b))

    if not spin_blocks:
        raise exceptions.ParseError(f"No PROCAR data blocks found in {p}.")
    k_coords = spin_blocks[0][0]
    diffs = np.diff(k_coords, axis=0)
    if cell is not None:
        b_mat = (
            2.0 * np.pi * np.linalg.inv(np.asarray(cell, dtype=np.float64)).T
        )
        diffs = diffs @ b_mat
    k_dist = np.concatenate([[0.0], np.cumsum(np.linalg.norm(diffs, axis=1))])
    eig_3d = np.stack([blk[1] for blk in spin_blocks], axis=0)
    occ_3d = np.stack([blk[2] for blk in spin_blocks], axis=0)
    proj_3d = {
        key: np.stack([blk[3][key] for blk in spin_blocks], axis=0)
        for key in spin_blocks[0][3]
    }
    return electronic.BandStructure(
        k_distances=k_dist,
        eigenvalues=eig_3d,
        occupations=occ_3d,
        projections=proj_3d,
        metadata={"source": str(p)},
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
        name="procar",
        extensions=(".procar",),
        filenames=("PROCAR",),
        reader=read_procar,
        description="VASP PROCAR orbital-projected fatband weights",
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
