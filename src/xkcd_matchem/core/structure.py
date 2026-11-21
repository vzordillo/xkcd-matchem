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
"""Core atomic structure and bond representations."""

from __future__ import annotations

import collections
import dataclasses
import math
from typing import Any

import numpy as np

from xkcd_matchem.core import exceptions
from xkcd_matchem.core import geometry
from xkcd_matchem.data import elements


@dataclasses.dataclass(frozen=True)
class Bond:
    """Chemical bond connecting two atoms across optional periodic images.

    Attributes:
      i: Index of the first atom in the parent ``AtomsData``.
      j: Index of the second atom in the parent ``AtomsData``.
      order: Bond order (``1.0`` single, ``1.5`` aromatic, ``2.0`` double,
        ``3.0`` triple).
      shift: Lattice translation ``(sa, sb, sc)`` added to atom ``j``.
    """

    i: int
    j: int
    order: float = 1.0
    shift: tuple[int, int, int] = (0, 0, 0)


@dataclasses.dataclass
class AtomsData:
    """Unified container for molecular and periodic crystal structures.

    Attributes:
      symbols: List of ``N`` element symbols.
      positions: Float64 Cartesian coordinates of shape ``(N, 3)`` in Angstrom.
      cell: Optional ``(3, 3)`` lattice matrix in Angstrom with vectors as rows.
      pbc: Periodic boundary flags along the three lattice vectors.
      bonds: Optional explicit chemical bonds.
      charges: Optional per-atom partial charges of shape ``(N,)``.
      velocities: Optional velocities of shape ``(N, 3)`` in A/fs.
      forces: Optional atomic forces of shape ``(N, 3)`` in eV/A.
      energies: Mapping of named scalar energies (typically in eV).
      atom_labels: Optional per-atom site or residue labels.
      metadata: Free-form parser or provenance dictionary.
    """

    symbols: list[str]
    positions: np.ndarray
    cell: np.ndarray | None = None
    pbc: tuple[bool, bool, bool] = (False, False, False)
    bonds: list[Bond] = dataclasses.field(default_factory=list)
    charges: np.ndarray | None = None
    velocities: np.ndarray | None = None
    forces: np.ndarray | None = None
    energies: dict[str, float] = dataclasses.field(default_factory=dict)
    atom_labels: list[str] | None = None
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalizes coordinate arrays and validates dimensions."""
        self.symbols = [
            elements.get_element(sym).symbol for sym in self.symbols
        ]
        n_atoms = len(self.symbols)
        pos = np.asarray(self.positions, dtype=np.float64)
        if pos.size == 0 and n_atoms == 0:
            pos = np.zeros((0, 3), dtype=np.float64)
        if pos.shape != (n_atoms, 3):
            raise exceptions.StructureError(
                f"Expected positions shape ({n_atoms}, 3), got {pos.shape}."
            )
        if not np.all(np.isfinite(pos)):
            raise exceptions.StructureError(
                "Atomic positions contain NaN or infinite values."
            )
        self.positions = pos
        if len(self.pbc) != 3:
            raise exceptions.StructureError(
                f"Expected 3 pbc flags, got {len(self.pbc)}."
            )
        self.pbc = (bool(self.pbc[0]), bool(self.pbc[1]), bool(self.pbc[2]))
        if self.cell is not None:
            cell_arr = np.asarray(self.cell, dtype=np.float64)
            if cell_arr.shape != (3, 3):
                raise exceptions.StructureError(
                    f"Expected cell shape (3, 3), got {cell_arr.shape}."
                )
            if not np.all(np.isfinite(cell_arr)):
                raise exceptions.StructureError(
                    "Cell matrix contains NaN or infinite values."
                )
            if any(self.pbc) and abs(float(np.linalg.det(cell_arr))) < 1e-12:
                raise exceptions.StructureError(
                    "Periodic unit cell matrix is singular (zero volume)."
                )
            self.cell = cell_arr
        for arr_name in ("velocities", "forces"):
            val = getattr(self, arr_name)
            if val is not None:
                arr_3d = np.asarray(val, dtype=np.float64)
                if arr_3d.shape != (n_atoms, 3) or not np.all(
                    np.isfinite(arr_3d)
                ):
                    raise exceptions.StructureError(
                        f"{arr_name} must be finite with shape ({n_atoms}, 3)."
                    )
                setattr(self, arr_name, arr_3d)
        if self.charges is not None:
            chg = np.asarray(self.charges, dtype=np.float64)
            if chg.shape != (n_atoms,) or not np.all(np.isfinite(chg)):
                raise exceptions.StructureError(
                    f"Expected finite charges ({n_atoms},), got {chg.shape}."
                )
            self.charges = chg
        if self.atom_labels is not None and len(self.atom_labels) != n_atoms:
            raise exceptions.StructureError(
                "atom_labels length does not match number of atoms."
            )

    def __len__(self) -> int:
        """Returns the number of atoms."""
        return len(self.symbols)

    @property
    def is_periodic(self) -> bool:
        """Returns True when a cell is present with any periodic axis."""
        return self.cell is not None and any(self.pbc)

    def formula(self, reduce_formula: bool = False) -> str:
        """Formats the chemical formula in Hill system order.

        Args:
          reduce_formula: Divides counts by their GCD when True.

        Returns:
          Chemical formula string, e.g. ``"C9H8O4"`` or ``"MoS2"``.
        """
        counts = collections.Counter(self.symbols)
        if not counts:
            return ""
        if reduce_formula:
            divisor = math.gcd(*counts.values())
            counts = collections.Counter(
                {k: v // divisor for k, v in counts.items()}
            )
        if "C" in counts:
            ordered = ["C"]
            if "H" in counts:
                ordered.append("H")
            ordered.extend(sorted(k for k in counts if k not in ("C", "H")))
        else:
            ordered = sorted(counts)
        parts = [
            f"{sym}{counts[sym]}" if counts[sym] > 1 else sym for sym in ordered
        ]
        return "".join(parts)

    def atomic_masses(self) -> np.ndarray:
        """Returns the per-atom standard atomic masses in Daltons."""
        return np.array(
            [elements.get_atomic_mass(sym) for sym in self.symbols],
            dtype=np.float64,
        )

    def center_of_mass(self) -> np.ndarray:
        """Computes the center of mass in Angstrom."""
        if len(self) == 0:
            return np.zeros(3, dtype=np.float64)
        masses = self.atomic_masses()
        total_mass = float(np.sum(masses))
        if total_mass <= 0.0:
            return np.mean(self.positions, axis=0)
        return (masses[:, None] * self.positions).sum(axis=0) / total_mass

    def volume(self) -> float:
        """Returns cell volume in cubic Angstrom, or ``0.0`` if aperiodic."""
        if self.cell is None:
            return 0.0
        return float(abs(np.linalg.det(self.cell)))

    def fractional_positions(self) -> np.ndarray:
        """Converts Cartesian positions to fractional cell coordinates."""
        if self.cell is None:
            raise exceptions.StructureError(
                "Cannot compute fractional positions without a unit cell."
            )
        return self.positions @ np.linalg.inv(self.cell)

    def wrap_positions(self) -> AtomsData:
        """Returns a copy with periodic coordinates wrapped into ``[0, 1)``."""
        if self.cell is None or not any(self.pbc):
            return self.copy()
        frac = self.fractional_positions()
        for axis, periodic in enumerate(self.pbc):
            if periodic:
                frac[:, axis] = np.mod(frac[:, axis], 1.0)
        wrapped = self.copy()
        wrapped.positions = frac @ self.cell
        return wrapped

    def center_in_cell(
        self,
        axes: tuple[int, ...] = (0, 1, 2),
    ) -> AtomsData:
        """Shifts fractional coords so the widest gap sits on cell faces.

        Keeping atoms away from box boundaries produces cleaner unit-cell
        renderings without clipped half-spheres along cell edges.

        Args:
          axes: Periodic lattice axis indices to recenter.

        Returns:
          Copy of ``AtomsData`` with wrapped and centered positions.
        """
        if self.cell is None or not any(self.pbc) or len(self) == 0:
            return self.copy()
        active_axes = tuple(ax for ax in axes if 0 <= ax < 3 and self.pbc[ax])
        if not active_axes:
            return self.wrap_positions()
        frac = np.mod(self.fractional_positions(), 1.0)
        shift = geometry.best_fractional_shift(frac, axes=active_axes)
        for ax in active_axes:
            frac[:, ax] = np.mod(frac[:, ax] + shift[ax], 1.0)
        centered = self.copy()
        centered.positions = frac @ self.cell
        return centered

    def repeat(self, rep: tuple[int, int, int] = (1, 1, 1)) -> AtomsData:
        """Builds a periodic ``(nx, ny, nz)`` supercell.

        Args:
          rep: Repetition counts ``(na, nb, nc)`` along each lattice vector.

        Returns:
          New ``AtomsData`` containing replicated sites and scaled unit cell.

        Raises:
          StructureError: If no cell exists or a repeat count is less than 1.
        """
        if self.cell is None:
            raise exceptions.StructureError(
                "Cannot build a supercell without a unit cell."
            )
        na, nb, nc = (int(v) for v in rep)
        if na < 1 or nb < 1 or nc < 1:
            raise exceptions.StructureError("Supercell counts must be >= 1.")
        if (na, nb, nc) == (1, 1, 1):
            return self.copy()
        n_base = len(self.symbols)
        n_cells = na * nb * nc
        pos_chunks: list[np.ndarray] = []
        new_bonds: list[Bond] = []

        for ia in range(na):
            for ib in range(nb):
                for ic in range(nc):
                    shift = (
                        ia * self.cell[0]
                        + ib * self.cell[1]
                        + ic * self.cell[2]
                    )
                    base_idx = ((ia * nb + ib) * nc + ic) * n_base
                    pos_chunks.append(self.positions + shift)
                    for b in self.bonds:
                        sa, rem_a = divmod(ia + b.shift[0], na)
                        sb, rem_b = divmod(ib + b.shift[1], nb)
                        sc, rem_c = divmod(ic + b.shift[2], nc)
                        dst_base = ((rem_a * nb + rem_b) * nc + rem_c) * n_base
                        new_bonds.append(
                            Bond(
                                i=base_idx + b.i,
                                j=dst_base + b.j,
                                order=b.order,
                                shift=(int(sa), int(sb), int(sc)),
                            )
                        )

        new_pos = (
            np.vstack(pos_chunks)
            if pos_chunks
            else np.zeros((0, 3), dtype=np.float64)
        )
        new_cell = np.vstack(
            [self.cell[0] * na, self.cell[1] * nb, self.cell[2] * nc]
        )
        new_meta = dict(self.metadata)
        new_meta["supercell_repeat"] = (na, nb, nc)
        return AtomsData(
            symbols=self.symbols * n_cells,
            positions=new_pos,
            cell=new_cell,
            pbc=self.pbc,
            bonds=new_bonds,
            charges=(
                np.tile(self.charges, n_cells)
                if self.charges is not None
                else None
            ),
            velocities=(
                np.tile(self.velocities, (n_cells, 1))
                if self.velocities is not None
                else None
            ),
            forces=(
                np.tile(self.forces, (n_cells, 1))
                if self.forces is not None
                else None
            ),
            energies={},
            atom_labels=(
                self.atom_labels * n_cells
                if self.atom_labels is not None
                else None
            ),
            metadata=new_meta,
        )

    def copy(self) -> AtomsData:
        """Creates a deep copy of the structure."""
        return AtomsData(
            symbols=list(self.symbols),
            positions=self.positions.copy(),
            cell=None if self.cell is None else self.cell.copy(),
            pbc=self.pbc,
            bonds=list(self.bonds),
            charges=None if self.charges is None else self.charges.copy(),
            velocities=None
            if self.velocities is None
            else self.velocities.copy(),
            forces=None if self.forces is None else self.forces.copy(),
            energies=dict(self.energies),
            atom_labels=(
                None if self.atom_labels is None else list(self.atom_labels)
            ),
            metadata=dict(self.metadata),
        )

    @classmethod
    def from_fractional(
        cls,
        symbols: list[str],
        frac_positions: np.ndarray,
        cell: np.ndarray,
        pbc: tuple[bool, bool, bool] = (True, True, True),
        **kwargs: Any,
    ) -> AtomsData:
        """Constructs ``AtomsData`` from fractional coords and a cell matrix."""
        cell_arr = np.asarray(cell, dtype=np.float64)
        frac_arr = np.asarray(frac_positions, dtype=np.float64)
        if frac_arr.size == 0 and not symbols:
            cart = np.zeros((0, 3), dtype=np.float64)
        else:
            cart = frac_arr @ cell_arr
        return cls(
            symbols=symbols, positions=cart, cell=cell_arr, pbc=pbc, **kwargs
        )

    @classmethod
    def from_cell_parameters(
        cls,
        symbols: list[str],
        frac_positions: np.ndarray,
        a: float,
        b: float,
        c: float,
        alpha: float = 90.0,
        beta: float = 90.0,
        gamma: float = 90.0,
        pbc: tuple[bool, bool, bool] = (True, True, True),
        **kwargs: Any,
    ) -> AtomsData:
        """Constructs ``AtomsData`` from cell parameters and frac coords."""
        cell = geometry.cell_vectors_from_parameters(
            a, b, c, alpha, beta, gamma
        )
        return cls.from_fractional(
            symbols=symbols,
            frac_positions=frac_positions,
            cell=cell,
            pbc=pbc,
            **kwargs,
        )
