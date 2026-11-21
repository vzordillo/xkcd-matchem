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
"""Electronic structure data containers (bands, DOS, MOs, volumetric grids)."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from xkcd_matchem.core import exceptions
from xkcd_matchem.core import structure


@dataclasses.dataclass
class BandStructure:
    """Electronic band dispersion along a reciprocal-space path.

    Attributes:
      k_distances: 1D cumulative path coordinate of shape ``(n_k,)``.
      eigenvalues: Energies in eV of shape ``(n_spin, n_k, n_bands)`` or
        ``(n_k, n_bands)`` (promoted automatically to 3D).
      fermi_energy: Reference Fermi/valence-band-maximum energy in eV.
      k_labels: High-symmetry point labels at ``k_ticks``.
      k_ticks: Cumulative path coordinates corresponding to ``k_labels``.
      metadata: Free-form source metadata.
    """

    k_distances: np.ndarray
    eigenvalues: np.ndarray
    fermi_energy: float = 0.0
    k_labels: list[str] = dataclasses.field(default_factory=list)
    k_ticks: list[float] = dataclasses.field(default_factory=list)
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalizes eigenvalues to shape ``(n_spin, n_k, n_bands)``."""
        self.k_distances = np.asarray(self.k_distances, dtype=np.float64)
        eig = np.asarray(self.eigenvalues, dtype=np.float64)
        if eig.ndim == 2:
            eig = eig[None, :, :]
        if eig.ndim != 3:
            raise exceptions.StructureError(
                f"Expected 2D or 3D eigenvalues, got shape {eig.shape}."
            )
        if eig.shape[1] != self.k_distances.shape[0]:
            raise exceptions.StructureError(
                "k_distances length must match eigenvalue k-point dimension."
            )
        self.eigenvalues = eig

    @property
    def spin_polarized(self) -> bool:
        """Returns True when two spin channels are stored."""
        return self.eigenvalues.shape[0] == 2

    def shifted_eigenvalues(self) -> np.ndarray:
        """Returns eigenvalues shifted so ``fermi_energy`` lies at zero eV."""
        return self.eigenvalues - self.fermi_energy

    def band_gap(self) -> float:
        """Estimates the electronic band gap across all bands around ``E_F``."""
        shifted = self.shifted_eigenvalues()
        occupied = shifted[shifted <= 1e-6]
        unoccupied = shifted[shifted > 1e-6]
        if occupied.size == 0 or unoccupied.size == 0:
            return 0.0
        vbm = float(np.max(occupied))
        cbm = float(np.min(unoccupied))
        return max(0.0, cbm - vbm)


@dataclasses.dataclass
class DensityOfStates:
    """Total and orbital/species-projected electronic density of states.

    Attributes:
      energies: Energy grid of shape ``(n_e,)`` in eV.
      total_dos: Total spin-up (or spin-summed) DOS of shape ``(n_e,)``.
      projected_dos: Mapping from orbital/atom label to DOS arrays.
      fermi_energy: Reference Fermi level in eV.
      spin_down_dos: Optional spin-down DOS array of shape ``(n_e,)``.
      metadata: Free-form source metadata.
    """

    energies: np.ndarray
    total_dos: np.ndarray
    projected_dos: dict[str, np.ndarray] = dataclasses.field(
        default_factory=dict
    )
    fermi_energy: float = 0.0
    spin_down_dos: np.ndarray | None = None
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        """Converts energy and DOS sequences to float64 arrays."""
        self.energies = np.asarray(self.energies, dtype=np.float64)
        self.total_dos = np.asarray(self.total_dos, dtype=np.float64)
        if self.energies.shape != self.total_dos.shape:
            raise exceptions.StructureError(
                "energies and total_dos must have matching shapes."
            )
        if self.spin_down_dos is not None:
            self.spin_down_dos = np.asarray(
                self.spin_down_dos, dtype=np.float64
            )
        self.projected_dos = {
            key: np.asarray(val, dtype=np.float64)
            for key, val in self.projected_dos.items()
        }


@dataclasses.dataclass
class MolecularOrbitals:
    """Discrete molecular orbital energies and occupations.

    Attributes:
      energies: Orbital energies in eV of shape ``(n_mo,)``.
      occupations: Electron occupation numbers of shape ``(n_mo,)``.
      labels: Optional symmetry or orbital labels.
      spin: Optional spin channel marker per orbital (``"alpha"`` / ``"beta"``).
      metadata: Free-form source metadata.
    """

    energies: np.ndarray
    occupations: np.ndarray
    labels: list[str] = dataclasses.field(default_factory=list)
    spin: list[str] = dataclasses.field(default_factory=list)
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validates MO array lengths."""
        self.energies = np.asarray(self.energies, dtype=np.float64)
        self.occupations = np.asarray(self.occupations, dtype=np.float64)
        if self.energies.shape != self.occupations.shape:
            raise exceptions.StructureError(
                "MO energies and occupations must have matching shapes."
            )

    @property
    def homo_index(self) -> int | None:
        """Returns the index of the highest occupied molecular orbital."""
        occ_idx = np.where(self.occupations > 0.5)[0]
        if occ_idx.size == 0:
            return None
        return int(occ_idx[np.argmax(self.energies[occ_idx])])

    @property
    def lumo_index(self) -> int | None:
        """Returns the index of the lowest unoccupied molecular orbital."""
        virt_idx = np.where(self.occupations <= 0.5)[0]
        if virt_idx.size == 0:
            return None
        return int(virt_idx[np.argmin(self.energies[virt_idx])])

    def homo_lumo_gap(self) -> float | None:
        """Returns the HOMO-LUMO gap in eV, or ``None`` if unavailable."""
        hi = self.homo_index
        lo = self.lumo_index
        if hi is None or lo is None:
            return None
        return float(self.energies[lo] - self.energies[hi])


@dataclasses.dataclass
class VolumetricGrid:
    """Regular 3D scalar field on an affine grid (CUBE, CHGCAR, LOCPOT).

    Attributes:
      origin: Grid origin in Angstrom of shape ``(3,)``.
      voxel_vectors: Step vectors per grid voxel ``(3, 3)`` in Angstrom.
      data: 3D scalar array of shape ``(nx, ny, nz)``.
      atoms: Optional associated ``AtomsData`` structure.
      units: Descriptive field quantity/unit string.
      metadata: Free-form source metadata.
    """

    origin: np.ndarray
    voxel_vectors: np.ndarray
    data: np.ndarray
    atoms: structure.AtomsData | None = None
    units: str = "e/A^3"
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        """Coerces grid arrays to float64 and validates shapes."""
        self.origin = np.asarray(self.origin, dtype=np.float64)
        self.voxel_vectors = np.asarray(self.voxel_vectors, dtype=np.float64)
        self.data = np.asarray(self.data, dtype=np.float64)
        if self.origin.shape != (3,):
            raise exceptions.StructureError("Grid origin must have shape (3,).")
        if self.voxel_vectors.shape != (3, 3):
            raise exceptions.StructureError(
                "voxel_vectors must have shape (3, 3)."
            )
        if self.data.ndim != 3:
            raise exceptions.StructureError("Volumetric data must be 3D.")

    def slice_plane(
        self,
        axis: str = "z",
        index: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extracts a 2D slice across an orthogonal or affine grid axis.

        Args:
          axis: Slice axis ``"x"``, ``"y"``, or ``"z"``.
          index: Plane index along ``axis``; defaults to midpoint.

        Returns:
          Tuple ``(u_coords, v_coords, plane_values)`` with 1D spatial axes
          in Angstrom and a 2D slice of shape ``(nu, nv)``.

        Raises:
          StructureError: If ``axis`` is invalid or ``index`` out of bounds.
        """
        axis_map = {"x": 0, "y": 1, "z": 2}
        key = axis.lower()
        if key not in axis_map:
            raise exceptions.StructureError(
                f"Invalid slice axis {axis!r}. Expected 'x', 'y', or 'z'."
            )
        ax_idx = axis_map[key]
        n_along = self.data.shape[ax_idx]
        sl_idx = n_along // 2 if index is None else int(index)
        if not 0 <= sl_idx < n_along:
            raise exceptions.StructureError(
                f"Slice index {sl_idx} out of range [0, {n_along})."
            )
        lengths = np.linalg.norm(self.voxel_vectors, axis=1)
        if ax_idx == 0:
            u = self.origin[1] + np.arange(self.data.shape[1]) * lengths[1]
            v = self.origin[2] + np.arange(self.data.shape[2]) * lengths[2]
            return u, v, self.data[sl_idx, :, :]
        if ax_idx == 1:
            u = self.origin[0] + np.arange(self.data.shape[0]) * lengths[0]
            v = self.origin[2] + np.arange(self.data.shape[2]) * lengths[2]
            return u, v, self.data[:, sl_idx, :]
        u = self.origin[0] + np.arange(self.data.shape[0]) * lengths[0]
        v = self.origin[1] + np.arange(self.data.shape[1]) * lengths[1]
        return u, v, self.data[:, :, sl_idx]
