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
"""Bond inference, bond-order perception, and coordination polyhedra."""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses

import numpy as np
from scipy import spatial

from xkcd_matchem.core import structure
from xkcd_matchem.data import elements


_ORGANIC_VALENCE: dict[str, int] = {
    "H": 1,
    "B": 3,
    "C": 4,
    "N": 3,
    "O": 2,
    "F": 1,
    "Si": 4,
    "P": 3,
    "S": 2,
    "Cl": 1,
    "Br": 1,
    "I": 1,
}

_TRIPLE_MAX_DIST: dict[tuple[str, str], float] = {
    ("C", "C"): 1.24,
    ("C", "N"): 1.20,
    ("N", "N"): 1.16,
    ("C", "O"): 1.17,
}

_DOUBLE_MAX_DIST: dict[tuple[str, str], float] = {
    ("C", "C"): 1.44,
    ("C", "O"): 1.30,
    ("C", "N"): 1.36,
    ("N", "N"): 1.32,
    ("N", "O"): 1.28,
    ("O", "O"): 1.28,
    ("C", "S"): 1.70,
    ("O", "S"): 1.54,
    ("O", "P"): 1.56,
}


@dataclasses.dataclass(frozen=True)
class CoordinationPolyhedron:
    """Convex coordination polyhedron around a central atom.

    Attributes:
      center_index: Atom index of the central cation/metal site.
      center_symbol: Element symbol of the central atom.
      ligand_indices: Original atom indices of coordinated ligands.
      vertices_3d: Unwrapped ligand coordinates of shape ``(K, 3)`` in Angstrom.
      faces: Triangular vertex index triples forming the convex hull surface.
    """

    center_index: int
    center_symbol: str
    ligand_indices: tuple[int, ...]
    vertices_3d: np.ndarray
    faces: tuple[tuple[int, int, int], ...]


def _periodic_shifts(
    pbc: tuple[bool, bool, bool],
) -> list[tuple[int, int, int]]:
    """Generates integer image offsets along enabled periodic axes."""
    rx = (-1, 0, 1) if pbc[0] else (0,)
    ry = (-1, 0, 1) if pbc[1] else (0,)
    rz = (-1, 0, 1) if pbc[2] else (0,)
    shifts: list[tuple[int, int, int]] = []
    for sx in rx:
        for sy in ry:
            for sz in rz:
                shifts.append((sx, sy, sz))
    return shifts


def _pair_key(s_a: str, s_b: str) -> tuple[str, str]:
    """Returns canonical alphabetically sorted element pair key."""
    return (s_a, s_b) if s_a <= s_b else (s_b, s_a)


def _match_deficit_double_bonds(
    candidates: list[tuple[float, int, int, int]],
    deficits: list[int],
    orders: list[float],
) -> None:
    """Assigns double bonds via min-degree-first matching for Kekule rings."""
    active_edges: list[tuple[float, int, int, int]] = [
        item
        for item in candidates
        if deficits[item[2]] >= 1 and deficits[item[3]] >= 1
    ]
    while active_edges:
        deg_map: dict[int, int] = {}
        for _, _, u, v in active_edges:
            deg_map[u] = deg_map.get(u, 0) + 1
            deg_map[v] = deg_map.get(v, 0) + 1
        chosen = min(
            active_edges,
            key=lambda item: (
                min(deg_map[item[2]], deg_map[item[3]]),
                deg_map[item[2]] + deg_map[item[3]],
                item[0],
            ),
        )
        _, b_idx, u, v = chosen
        orders[b_idx] = 2.0
        deficits[u] -= 1
        deficits[v] -= 1
        active_edges = [
            item
            for item in active_edges
            if item[1] != b_idx
            and deficits[item[2]] >= 1
            and deficits[item[3]] >= 1
        ]


def infer_bond_orders(
    atoms: structure.AtomsData,
    bonds: Sequence[structure.Bond],
) -> list[structure.Bond]:
    """Infers organic multiple bond orders from geometry and valence deficits.

    Preserves periodic bonds (``shift != (0, 0, 0)``) and non-organic metal or
    extended solid bonds at their existing orders while assigning triple,
    terminal carbonyl/heteroatom double, and conjugated/aromatic Kekule double
    bonds for ``C, N, O, S, P`` subgraphs.

    Args:
      atoms: Input atomic structure.
      bonds: Initial bond list (typically all order ``1.0``).

    Returns:
      Updated list of ``Bond`` objects with perceived bond orders.
    """
    if not bonds:
        return []
    n_atoms = len(atoms)
    degrees = [0] * n_atoms
    has_non_organic = [False] * n_atoms
    orders = [float(b.order) for b in bonds]
    dists: list[float] = []

    for b in bonds:
        degrees[b.i] += 1
        degrees[b.j] += 1
        if atoms.symbols[b.i] not in _ORGANIC_VALENCE:
            has_non_organic[b.j] = True
            has_non_organic[b.i] = True
        if atoms.symbols[b.j] not in _ORGANIC_VALENCE:
            has_non_organic[b.i] = True
            has_non_organic[b.j] = True
        dists.append(
            float(np.linalg.norm(atoms.positions[b.j] - atoms.positions[b.i]))
        )

    deficits = [
        max(0, _ORGANIC_VALENCE.get(atoms.symbols[i], 0) - degrees[i])
        if not has_non_organic[i]
        else 0
        for i in range(n_atoms)
    ]

    triple_cands: list[tuple[float, int, int, int]] = []
    terminal_doubles: list[tuple[float, int, int, int]] = []
    interior_doubles: list[tuple[float, int, int, int]] = []

    for b_idx, b in enumerate(bonds):
        if b.shift != (0, 0, 0) or orders[b_idx] > 1.0:
            continue
        if has_non_organic[b.i] or has_non_organic[b.j]:
            continue
        key = _pair_key(atoms.symbols[b.i], atoms.symbols[b.j])
        dist = dists[b_idx]
        t_lim = _TRIPLE_MAX_DIST.get(key)
        if t_lim is not None and dist <= t_lim:
            if key == ("C", "O") and not (
                degrees[b.i] == 1 and degrees[b.j] == 1
            ):
                pass
            elif degrees[b.i] <= 2 and degrees[b.j] <= 2:
                triple_cands.append((dist, b_idx, b.i, b.j))
        d_lim = _DOUBLE_MAX_DIST.get(key)
        if d_lim is not None and dist <= d_lim:
            if degrees[b.i] == 1 or degrees[b.j] == 1:
                terminal_doubles.append((dist, b_idx, b.i, b.j))
            else:
                interior_doubles.append((dist, b_idx, b.i, b.j))

    for _, b_idx, u, v in sorted(triple_cands):
        if deficits[u] >= 2 and deficits[v] >= 2:
            orders[b_idx] = 3.0
            deficits[u] -= 2
            deficits[v] -= 2

    for _, b_idx, u, v in sorted(terminal_doubles):
        if orders[b_idx] == 1.0 and deficits[u] >= 1 and deficits[v] >= 1:
            orders[b_idx] = 2.0
            deficits[u] -= 1
            deficits[v] -= 1

    _match_deficit_double_bonds(interior_doubles, deficits, orders)

    return [
        structure.Bond(i=b.i, j=b.j, order=orders[idx], shift=b.shift)
        for idx, b in enumerate(bonds)
    ]


def _collect_pair_bonds(
    atoms: structure.AtomsData,
    tree_base: spatial.cKDTree,
    radii: np.ndarray,
    shift_tuple: tuple[int, int, int],
    bounds: tuple[float, float, float],
    seen: set[tuple[int, int, tuple[int, int, int]]],
    bonds: list[structure.Bond],
) -> None:
    """Populates unique periodic bonds for a single lattice image offset."""
    scale, min_dist, max_dist = bounds
    sx, sy, sz = shift_tuple
    if shift_tuple == (0, 0, 0):
        shifted_pos = atoms.positions
    elif atoms.cell is not None:
        offset = sx * atoms.cell[0] + sy * atoms.cell[1] + sz * atoms.cell[2]
        shifted_pos = atoms.positions + offset
    else:
        return
    tree_shifted = spatial.cKDTree(shifted_pos)
    for i, nbrs in enumerate(
        tree_base.query_ball_tree(tree_shifted, r=max_dist)
    ):
        for j in nbrs:
            if shift_tuple == (0, 0, 0) and i >= j:
                continue
            dist = float(np.linalg.norm(shifted_pos[j] - atoms.positions[i]))
            cutoff = min(max_dist, (radii[i] + radii[j]) * scale)
            if not min_dist <= dist <= cutoff:
                continue
            key, bond = (
                (
                    (i, j, shift_tuple),
                    structure.Bond(i=i, j=j, shift=shift_tuple),
                )
                if i <= j
                else (
                    (j, i, (-sx, -sy, -sz)),
                    structure.Bond(i=j, j=i, shift=(-sx, -sy, -sz)),
                )
            )
            if key not in seen:
                seen.add(key)
                bonds.append(bond)


def infer_bonds(
    atoms: structure.AtomsData,
    scale: float = 1.20,
    min_dist: float = 0.4,
    max_dist: float = 3.6,
    use_pbc: bool = True,
    infer_orders: bool = True,
) -> list[structure.Bond]:
    """Infers chemical bonds using covalent-radius cutoffs and ``cKDTree``.

    Args:
      atoms: Input atomic or crystal structure.
      scale: Multiplicative tolerance applied to the sum of covalent radii.
      min_dist: Minimum allowed bond length in Angstrom.
      max_dist: Hard upper distance cutoff in Angstrom.
      use_pbc: Whether to search across periodic images when periodic.
      infer_orders: Whether to perceive organic double/triple bond orders from
        interatomic distances and valence deficits on non-periodic bonds.

    Returns:
      Sorted list of unique ``Bond`` objects.
    """
    if len(atoms) < 2:
        return []
    radii = np.array(
        [elements.get_covalent_radius(sym) for sym in atoms.symbols],
        dtype=np.float64,
    )
    shifts = _periodic_shifts(
        atoms.pbc
        if (use_pbc and atoms.cell is not None)
        else (False, False, False)
    )
    tree_base = spatial.cKDTree(atoms.positions)
    seen: set[tuple[int, int, tuple[int, int, int]]] = set()
    bonds: list[structure.Bond] = []
    for shift_tuple in shifts:
        _collect_pair_bonds(
            atoms,
            tree_base,
            radii,
            shift_tuple,
            (scale, min_dist, max_dist),
            seen,
            bonds,
        )
    bonds.sort(key=lambda b: (b.i, b.j, b.shift))
    if infer_orders:
        bonds = infer_bond_orders(atoms, bonds)
    return bonds


def _collect_ligand_vertices(
    atoms: structure.AtomsData,
    center_idx: int,
    lig_canon: str | None,
    shifts: list[tuple[int, int, int]],
    r_center: float,
    r_all: np.ndarray,
    cutoff: float | None,
    scale: float,
) -> tuple[list[np.ndarray], list[int]]:
    """Collects unwrapped periodic ligand positions around one center site."""
    origin = atoms.positions[center_idx]
    vert_coords: list[np.ndarray] = []
    vert_indices: list[int] = []
    for sx, sy, sz in shifts:
        shift_vec = np.zeros(3, dtype=np.float64)
        if atoms.cell is not None:
            shift_vec = (
                sx * atoms.cell[0] + sy * atoms.cell[1] + sz * atoms.cell[2]
            )
        for j, lsym in enumerate(atoms.symbols):
            if j == center_idx and (sx, sy, sz) == (0, 0, 0):
                continue
            if lig_canon is not None and lsym != lig_canon:
                continue
            lim = (r_center + r_all[j]) * scale if cutoff is None else cutoff
            pos_j = atoms.positions[j] + shift_vec
            dist = float(np.linalg.norm(pos_j - origin))
            if 0.4 <= dist <= lim:
                vert_coords.append(pos_j)
                vert_indices.append(j)
    return vert_coords, vert_indices


def find_coordination_polyhedra(
    atoms: structure.AtomsData,
    center_symbol: str,
    ligand_symbol: str | None = None,
    cutoff: float | None = None,
    scale: float = 1.28,
    min_coordination: int = 4,
) -> list[CoordinationPolyhedron]:
    """Finds 3D coordination polyhedra around selected central elements.

    Args:
      atoms: Input crystal or cluster structure.
      center_symbol: Central element symbol (e.g., ``"Ti"``, ``"Si"``).
      ligand_symbol: Optional ligand element filter (e.g., ``"O"``, ``"S"``).
      cutoff: Explicit distance cutoff in Angstrom; uses covalent radii times
        ``scale`` when ``None``.
      scale: Multiplier on covalent-radius sum when ``cutoff`` is ``None``.
      min_coordination: Minimum number of ligands needed to form a 3D hull.

    Returns:
      List of ``CoordinationPolyhedron`` objects with unwrapped ligand vertices
      and convex hull triangular faces.
    """
    center_canon = elements.get_element(center_symbol).symbol
    lig_canon = (
        None
        if ligand_symbol is None
        else elements.get_element(ligand_symbol).symbol
    )
    shifts = _periodic_shifts(
        atoms.pbc if atoms.cell is not None else (False, False, False)
    )
    r_center = elements.get_covalent_radius(center_canon)
    r_all = np.array(
        [elements.get_covalent_radius(s) for s in atoms.symbols],
        dtype=np.float64,
    )
    polyhedra: list[CoordinationPolyhedron] = []

    for idx, sym in enumerate(atoms.symbols):
        if sym != center_canon:
            continue
        vert_coords, vert_indices = _collect_ligand_vertices(
            atoms, idx, lig_canon, shifts, r_center, r_all, cutoff, scale
        )
        if len(vert_coords) < max(min_coordination, 4):
            continue
        verts = np.vstack(vert_coords)
        try:
            hull = spatial.ConvexHull(verts)
        except spatial.QhullError:
            continue
        faces = tuple(
            (int(tri[0]), int(tri[1]), int(tri[2])) for tri in hull.simplices
        )
        polyhedra.append(
            CoordinationPolyhedron(
                center_index=idx,
                center_symbol=center_canon,
                ligand_indices=tuple(vert_indices),
                vertices_3d=verts,
                faces=faces,
            )
        )
    return polyhedra
