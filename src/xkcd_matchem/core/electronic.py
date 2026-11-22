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
      occupations: Optional band occupations matching ``eigenvalues`` shape.
      projections: Optional orbital/atom projection weights keyed by channel
        (e.g. ``"s"``, ``"p"``, ``"d"``, ``"tot"``) with shape
        ``(n_spin, n_k, n_bands)`` or ``(n_k, n_bands)``.
      metadata: Free-form source metadata.
    """

    k_distances: np.ndarray
    eigenvalues: np.ndarray
    fermi_energy: float = 0.0
    k_labels: list[str] = dataclasses.field(default_factory=list)
    k_ticks: list[float] = dataclasses.field(default_factory=list)
    occupations: np.ndarray | None = None
    projections: dict[str, np.ndarray] = dataclasses.field(default_factory=dict)
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalizes eigenvalues, occupations, and projections to 3D."""
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
        if self.occupations is not None:
            occ = np.asarray(self.occupations, dtype=np.float64)
            if occ.ndim == 2:
                occ = occ[None, :, :]
            if occ.shape != eig.shape:
                raise exceptions.StructureError(
                    "occupations shape must match eigenvalues shape."
                )
            self.occupations = occ
        norm_proj: dict[str, np.ndarray] = {}
        for key, raw_w in self.projections.items():
            w_arr = np.asarray(raw_w, dtype=np.float64)
            if w_arr.ndim == 2:
                w_arr = w_arr[None, :, :]
            if w_arr.shape != eig.shape:
                raise exceptions.StructureError(
                    f"Projection {key!r} shape {w_arr.shape} must match "
                    f"eigenvalues shape {eig.shape}."
                )
            norm_proj[str(key)] = w_arr
        self.projections = norm_proj

    @property
    def spin_polarized(self) -> bool:
        """Returns True when two spin channels are stored."""
        return bool(self.eigenvalues.shape[0] == 2)

    def shifted_eigenvalues(self) -> np.ndarray:
        """Returns eigenvalues shifted so ``fermi_energy`` lies at zero eV."""
        return self.eigenvalues - self.fermi_energy

    def band_gap(self) -> float:
        """Estimates the band gap from occupations or Fermi-level crossing."""
        if self.occupations is not None:
            occ_mask = self.occupations > 0.5
            unocc_mask = ~occ_mask
            if np.any((self.occupations > 1e-3) & (self.occupations < 0.999)):
                return 0.0
            if not np.any(occ_mask) or not np.any(unocc_mask):
                return 0.0
            vbm = float(np.max(self.eigenvalues[occ_mask]))
            cbm = float(np.min(self.eigenvalues[unocc_mask]))
            return max(0.0, cbm - vbm)

        shifted = self.shifted_eigenvalues()
        for spin_idx in range(shifted.shape[0]):
            for band_idx in range(shifted.shape[2]):
                col = shifted[spin_idx, :, band_idx]
                if np.min(col) < -1e-4 and np.max(col) > 1e-4:
                    return 0.0
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
        """Extracts 1D voxel-length axes and 2D slice array along ``axis``."""
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

    def slice_mesh(
        self,
        axis: str = "z",
        index: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extracts an affine-accurate 2D slice mesh and orthonormal basis.

        Projects each grid point's true 3D Cartesian coordinate onto a
        2D orthonormal basis spanning the slice plane so non-orthogonal voxel
        vectors render without geometric skew distortion.

        Returns:
          Tuple ``(uu, vv, plane_values, basis_2x3)`` where ``uu``, ``vv``, and
          ``plane_values`` all have shape ``(n1, n2)``.
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
        pair_axes = [i for i in (0, 1, 2) if i != ax_idx]
        a1, a2 = pair_axes[0], pair_axes[1]
        v1 = self.voxel_vectors[a1]
        v2 = self.voxel_vectors[a2]
        e_u = v1 / max(float(np.linalg.norm(v1)), 1e-12)
        v2_orth = v2 - float(np.dot(v2, e_u)) * e_u
        e_v = v2_orth / max(float(np.linalg.norm(v2_orth)), 1e-12)
        basis = np.vstack([e_u, e_v])

        n1, n2 = self.data.shape[a1], self.data.shape[a2]
        g1, g2 = np.meshgrid(
            np.arange(n1, dtype=np.float64),
            np.arange(n2, dtype=np.float64),
            indexing="ij",
        )
        base = self.origin + sl_idx * self.voxel_vectors[ax_idx]
        pts = (
            base[None, None, :]
            + g1[:, :, None] * v1[None, None, :]
            + g2[:, :, None] * v2[None, None, :]
        )
        uu = pts @ e_u
        vv = pts @ e_v
        plane = np.take(self.data, sl_idx, axis=ax_idx)
        return uu, vv, plane, basis


_PHONON_UNIT_FACTORS: dict[str, float] = {
    "thz": 1.0,
    "cm-1": 33.3564095198152,
    "cm^-1": 33.3564095198152,
    "mev": 4.135667696,
}


@dataclasses.dataclass
class PhononBandStructure:
    """Phonon dispersion branches along a reciprocal-space q-point path.

    Attributes:
      q_distances: 1D cumulative path coordinate of shape ``(n_q,)``.
      frequencies: Phonon frequencies in THz of shape ``(n_q, n_modes)``.
      q_labels: High-symmetry q-point labels at ``q_ticks``.
      q_ticks: Cumulative path coordinates corresponding to ``q_labels``.
      metadata: Free-form source metadata.
    """

    q_distances: np.ndarray
    frequencies: np.ndarray
    q_labels: list[str] = dataclasses.field(default_factory=list)
    q_ticks: list[float] = dataclasses.field(default_factory=list)
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        """Coerces q_distances and frequencies and validates shapes."""
        self.q_distances = np.asarray(self.q_distances, dtype=np.float64)
        freq = np.asarray(self.frequencies, dtype=np.float64)
        if freq.ndim != 2 or freq.shape[0] != self.q_distances.shape[0]:
            raise exceptions.StructureError(
                "frequencies must be 2D (n_q, n_modes) matching q_distances."
            )
        self.frequencies = freq

    @property
    def has_imaginary_modes(self) -> bool:
        """Returns True if any branch has imaginary modes (< -0.03 THz)."""
        return bool(np.any(self.frequencies < -0.03))

    def frequencies_in(self, unit: str = "THz") -> np.ndarray:
        """Converts phonon frequencies from THz to THz, cm-1, or meV."""
        key = unit.strip().lower()
        if key not in _PHONON_UNIT_FACTORS:
            raise exceptions.StructureError(
                f"Unsupported phonon frequency unit {unit!r}. "
                "Use 'THz', 'cm-1', or 'meV'."
            )
        return self.frequencies * _PHONON_UNIT_FACTORS[key]


@dataclasses.dataclass
class CohpData:
    """Crystal Orbital Hamilton/Overlap Population (pCOHP / pCOOP) data.

    Attributes:
      energies: 1D energy grid in eV (relative to ``fermi_energy`` when zero).
      cohp: Mapping from bond label (e.g. ``"No.1:Si1->Si2"``) to pCOHP.
      icohp: Mapping from bond label to integrated ICOHP values in eV.
      fermi_energy: Reference Fermi energy in eV (``0.0`` if already shifted).
      spin_down_cohp: Optional spin-down pCOHP mapping by bond label.
      spin_down_icohp: Optional spin-down ICOHP mapping by bond label.
      is_coop: True when representing COOP (overlap) rather than COHP.
      metadata: Free-form parser metadata.
    """

    energies: np.ndarray
    cohp: dict[str, np.ndarray]
    icohp: dict[str, np.ndarray] = dataclasses.field(default_factory=dict)
    fermi_energy: float = 0.0
    spin_down_cohp: dict[str, np.ndarray] = dataclasses.field(
        default_factory=dict
    )
    spin_down_icohp: dict[str, np.ndarray] = dataclasses.field(
        default_factory=dict
    )
    is_coop: bool = False
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        """Coerces COHP arrays to float64 and validates grid lengths."""
        self.energies = np.asarray(self.energies, dtype=np.float64)
        if self.energies.ndim != 1 or self.energies.size == 0:
            raise exceptions.StructureError(
                "COHP energies must be a non-empty 1D array."
            )
        if not self.cohp:
            raise exceptions.StructureError(
                "CohpData requires at least one interaction entry."
            )
        n_pts = self.energies.shape[0]
        for mapping in (
            self.cohp,
            self.icohp,
            self.spin_down_cohp,
            self.spin_down_icohp,
        ):
            for key, vals in list(mapping.items()):
                arr = np.asarray(vals, dtype=np.float64)
                if arr.shape != (n_pts,):
                    raise exceptions.StructureError(
                        f"COHP series {key!r} shape {arr.shape} must match "
                        f"energies shape ({n_pts},)."
                    )
                mapping[key] = arr
