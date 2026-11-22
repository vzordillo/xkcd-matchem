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

from xkcd_matchem.core import neighbors
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
    i_indices = np.array([b.i for b in bonds], dtype=np.int64)
    j_indices = np.array([b.j for b in bonds], dtype=np.int64)
    dists = np.linalg.norm(
        atoms.positions[j_indices] - atoms.positions[i_indices], axis=1
    )

    for b in bonds:
        degrees[b.i] += 1
        degrees[b.j] += 1
        if atoms.symbols[b.i] not in _ORGANIC_VALENCE:
            has_non_organic[b.j] = True
            has_non_organic[b.i] = True
        if atoms.symbols[b.j] not in _ORGANIC_VALENCE:
            has_non_organic[b.i] = True
            has_non_organic[b.j] = True

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
        dist = float(dists[b_idx])
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
        structure.Bond(
            i=b.i, j=b.j, order=orders[idx], shift=b.shift, stereo=b.stereo
        )
        for idx, b in enumerate(bonds)
    ]


def infer_bonds(
    atoms: structure.AtomsData,
    scale: float = 1.20,
    min_dist: float = 0.4,
    max_dist: float = 3.6,
    use_pbc: bool = True,
    infer_orders: bool = True,
) -> list[structure.Bond]:
    """Infers chemical bonds using covalent-radius cutoffs and LLL PBC search.

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
    hits = neighbors.find_periodic_neighbors(
        atoms,
        max_cutoff=max_dist,
        min_cutoff=min_dist,
        use_pbc=use_pbc,
        symmetric=False,
    )
    bonds: list[structure.Bond] = []
    for hit in hits:
        cutoff = min(max_dist, (radii[hit.i] + radii[hit.j]) * scale)
        if hit.distance <= cutoff:
            bonds.append(
                structure.Bond(i=hit.i, j=hit.j, shift=hit.shift)
            )
    bonds.sort(key=lambda b: (b.i, b.j, b.shift))
    if infer_orders:
        bonds = infer_bond_orders(atoms, bonds)
    return bonds


def _oriented_hull_faces(
    verts: np.ndarray,
    simplices: np.ndarray,
) -> tuple[tuple[int, int, int], ...]:
    """Orients convex-hull simplices with outward CCW vertex winding."""
    hull_center = np.mean(verts, axis=0)
    faces: list[tuple[int, int, int]] = []
    for tri in simplices:
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        v0 = verts[i0]
        cross = np.cross(verts[i1] - v0, verts[i2] - v0)
        if float(np.dot(cross, v0 - hull_center)) < 0.0:
            faces.append((i0, i2, i1))
        else:
            faces.append((i0, i1, i2))
    return tuple(faces)


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
    if len(atoms) < 4:
        return []
    center_canon = elements.get_element(center_symbol).symbol
    lig_canon = (
        None
        if ligand_symbol is None
        else elements.get_element(ligand_symbol).symbol
    )
    r_center = elements.get_covalent_radius(center_canon)
    r_all = np.array(
        [elements.get_covalent_radius(s) for s in atoms.symbols],
        dtype=np.float64,
    )
    max_search = (
        float(cutoff)
        if cutoff is not None
        else float((r_center + np.max(r_all)) * scale)
    )
    hits = neighbors.find_periodic_neighbors(
        atoms,
        max_cutoff=max_search,
        min_cutoff=0.4,
        use_pbc=atoms.cell is not None,
        symmetric=True,
    )
    by_center: dict[int, list[neighbors.NeighborHit]] = {}
    for hit in hits:
        if atoms.symbols[hit.i] != center_canon:
            continue
        if lig_canon is not None and atoms.symbols[hit.j] != lig_canon:
            continue
        lim = (r_center + r_all[hit.j]) * scale if cutoff is None else cutoff
        if hit.distance <= lim:
            by_center.setdefault(hit.i, []).append(hit)

    polyhedra: list[CoordinationPolyhedron] = []
    for idx, sym in enumerate(atoms.symbols):
        if sym != center_canon:
            continue
        c_hits = by_center.get(idx, [])
        if len(c_hits) < max(min_coordination, 4):
            continue
        origin = atoms.positions[idx]
        verts = np.vstack([origin + h.vector for h in c_hits])
        vert_indices = tuple(h.j for h in c_hits)
        try:
            hull = spatial.ConvexHull(verts)
        except spatial.QhullError:
            continue
        polyhedra.append(
            CoordinationPolyhedron(
                center_index=idx,
                center_symbol=center_canon,
                ligand_indices=vert_indices,
                vertices_3d=verts,
                faces=_oriented_hull_faces(verts, hull.simplices),
            )
        )
    return polyhedra
