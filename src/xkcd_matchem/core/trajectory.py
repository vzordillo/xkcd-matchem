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
"""Molecular dynamics and reaction-path trajectory containers and analysis."""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import math
from typing import Any

import numpy as np

from xkcd_matchem.core import exceptions
from xkcd_matchem.core import geometry
from xkcd_matchem.core import neighbors
from xkcd_matchem.core import structure
from xkcd_matchem.data import elements


@dataclasses.dataclass
class Trajectory:
    """Multi-frame atomic trajectory for MD simulations or NEB reaction paths.

    Attributes:
      frames: Sequence of ``AtomsData`` snapshots.
      times: Optional simulation time stamps in femtoseconds or picoseconds.
      energies: Optional total energies per frame in eV.
      temperatures: Optional instantaneous temperatures in Kelvin.
      pressures: Optional pressures in bar or GPa.
      metadata: Free-form parser or simulation metadata.
    """

    frames: list[structure.AtomsData]
    times: np.ndarray | None = None
    energies: np.ndarray | None = None
    temperatures: np.ndarray | None = None
    pressures: np.ndarray | None = None
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalizes 1D scalar series, checks lengths, and pulls energies."""
        n_frames = len(self.frames)
        if self.energies is None and n_frames > 0:
            extracted = [frame.energies.get("total") for frame in self.frames]
            if all(val is not None for val in extracted):
                self.energies = np.array(extracted, dtype=np.float64)
        for field_name in ("times", "energies", "temperatures", "pressures"):
            val = getattr(self, field_name)
            if val is None:
                continue
            arr = np.asarray(val, dtype=np.float64)
            if arr.shape != (n_frames,):
                raise exceptions.StructureError(
                    f"{field_name} length {arr.shape} must match "
                    f"frames count ({n_frames},)."
                )
            if not np.all(np.isfinite(arr)):
                raise exceptions.StructureError(
                    f"{field_name} contains non-finite values."
                )
            setattr(self, field_name, arr)

    def __len__(self) -> int:
        """Returns the number of frames."""
        return len(self.frames)

    def __getitem__(self, index: int) -> structure.AtomsData:
        """Retrieves a single frame by index."""
        return self.frames[index]

    def _unwrapped_positions(self, idx_arr: np.ndarray) -> list[np.ndarray]:
        """Unwraps periodic coordinates cumulatively across frames."""
        unwrapped = [self.frames[0].positions[idx_arr].copy()]
        for k in range(1, len(self.frames)):
            prev_raw = self.frames[k - 1].positions[idx_arr]
            cur_frame = self.frames[k]
            cur_raw = cur_frame.positions[idx_arr]
            delta = cur_raw - prev_raw
            if cur_frame.cell is not None and any(cur_frame.pbc):
                delta = geometry.minimum_image_vector(
                    delta, cur_frame.cell, cur_frame.pbc
                )
            unwrapped.append(unwrapped[-1] + delta)
        return unwrapped

    def rmsd(
        self,
        ref_index: int = 0,
        indices: Sequence[int] | None = None,
        unwrap: bool = True,
    ) -> np.ndarray:
        """Computes Kabsch-aligned RMSD relative to a reference frame.

        Args:
          ref_index: Reference frame index.
          indices: Optional atom indices to select for alignment and RMSD.
          unwrap: Whether to unwrap periodic displacements across frames first.

        Returns:
          1D float64 RMSD array in Angstrom of shape ``(n_frames,)``.
        """
        if not self.frames:
            return np.zeros((0,), dtype=np.float64)
        n_atoms = len(self.frames[ref_index])
        if indices is None:
            idx_arr = np.arange(n_atoms, dtype=np.int64)
        else:
            idx_arr = np.asarray(list(indices), dtype=np.int64)
            if (
                idx_arr.size == 0
                or np.any(idx_arr < 0)
                or np.any(idx_arr >= n_atoms)
            ):
                raise exceptions.StructureError(
                    "Invalid atom selection indices for RMSD."
                )

        if unwrap and all(len(f) == n_atoms for f in self.frames):
            coords_list = self._unwrapped_positions(idx_arr)
        else:
            coords_list = [f.positions[idx_arr] for f in self.frames]

        ref = coords_list[ref_index]
        ref_centered = ref - np.mean(ref, axis=0)
        out = np.zeros(len(self.frames), dtype=np.float64)
        for idx, cur in enumerate(coords_list):
            cur_centered = cur - np.mean(cur, axis=0)
            cov = cur_centered.T @ ref_centered
            u_mat, _, vt_mat = np.linalg.svd(cov)
            det_sign = 1.0 if np.linalg.det(u_mat @ vt_mat) >= 0.0 else -1.0
            diag = np.diag([1.0, 1.0, det_sign])
            rot = u_mat @ diag @ vt_mat
            aligned = cur_centered @ rot
            diff = aligned - ref_centered
            out[idx] = math.sqrt(float(np.mean(np.sum(diff * diff, axis=1))))
        return out

    def energy_profile(
        self,
        relative: bool = True,
        unit: str = "eV",
    ) -> np.ndarray:
        """Returns frame energies in ``eV``, ``kJ/mol``, or ``kcal/mol``."""
        if self.energies is None:
            raise exceptions.StructureError(
                "Trajectory has no stored energies."
            )
        vals = self.energies.copy()
        if relative and vals.size > 0:
            vals -= vals[0]
        u_key = unit.lower()
        if u_key == "ev":
            return np.asarray(vals, dtype=np.float64)
        if u_key == "kj/mol":
            return np.asarray(
                vals * elements.CODATA2022.ev_to_kj_mol, dtype=np.float64
            )
        if u_key == "kcal/mol":
            return np.asarray(
                vals * elements.CODATA2022.ev_to_kcal_mol, dtype=np.float64
            )
        raise exceptions.StructureError(f"Unsupported energy unit {unit!r}.")

    def _frame_pair_distances(
        self,
        frame: structure.AtomsData,
        sym_a: str,
        sym_b: str,
        r_max: float,
    ) -> tuple[list[float], int, float]:
        """Collects periodic pair distances up to ``r_max`` using cKDTree."""
        n_a = sum(1 for s in frame.symbols if s == sym_a)
        n_b = sum(1 for s in frame.symbols if s == sym_b)
        if n_a == 0 or n_b == 0:
            return [], 0, 0.0
        pair_count = float(n_a * (n_b - 1 if sym_a == sym_b else n_b))
        if pair_count <= 0.0:
            pair_count = float(n_a)

        hits = neighbors.find_periodic_neighbors(
            frame,
            max_cutoff=r_max,
            min_cutoff=1e-6,
            use_pbc=True,
            symmetric=True,
        )
        dists = [
            hit.distance
            for hit in hits
            if hit.distance < r_max
            and frame.symbols[hit.i] == sym_a
            and frame.symbols[hit.j] == sym_b
        ]
        return dists, n_a, pair_count

    def compute_rdf(
        self,
        pair: tuple[str, str],
        r_max: float = 8.0,
        n_bins: int = 80,
        start: int = 0,
        stop: int | None = None,
        strict: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Computes the radial distribution function ``g(r)`` for a 3D cell.

        Uses exact spherical-shell volumes ``(4*pi/3)*(r_out^3 - r_in^3)`` and
        evaluates periodic image shells up to ``r_max``.

        Args:
          pair: Tuple of element symbols ``(sym_a, sym_b)``.
          r_max: Maximum pair separation in Angstrom.
          n_bins: Number of radial histogram bins.
          start: First frame index to include.
          stop: Exclusive last frame index to include.
          strict: If True, reject ``r_max`` exceeding the Wigner-Seitz
            minimum-image inradius.

        Returns:
          Tuple ``(r_centers, g_r)`` of 1D float64 arrays.

        Raises:
          StructureError: If frames lack a 3D periodic cell or violate bounds.
        """
        if r_max <= 0.0 or n_bins < 2:
            raise exceptions.StructureError("Invalid RDF binning parameters.")
        sym_a = elements.get_element(pair[0]).symbol
        sym_b = elements.get_element(pair[1]).symbol
        selected = self.frames[start:stop]
        if not selected:
            raise exceptions.StructureError(
                "No trajectory frames selected for RDF."
            )
        edges = np.linspace(0.0, r_max, n_bins + 1)
        r_centers = 0.5 * (edges[:-1] + edges[1:])
        shell_vols = (4.0 * math.pi / 3.0) * (edges[1:] ** 3 - edges[:-1] ** 3)
        g_sum = np.zeros(n_bins, dtype=np.float64)
        norm_accum = 0.0

        for frame in selected:
            if (
                frame.cell is None
                or not all(frame.pbc)
                or frame.volume() <= 0.0
            ):
                raise exceptions.StructureError(
                    "RDF requires 3D periodic frames and positive volume."
                )
            if strict:
                r_ws = geometry.max_min_image_radius(frame.cell, frame.pbc)
                if r_max > r_ws + 1e-9:
                    raise exceptions.StructureError(
                        f"r_max={r_max:.2f} A exceeds minimum-image inradius "
                        f"{r_ws:.2f} A; pass strict=False for multi-image RDF."
                    )
            dists, n_a, pair_count = self._frame_pair_distances(
                frame, sym_a, sym_b, r_max
            )
            if n_a == 0:
                continue
            rho = pair_count / frame.volume()
            norm_accum += 1.0
            hist_frame, _ = np.histogram(dists, bins=edges)
            g_sum += hist_frame / (rho * shell_vols)

        if norm_accum == 0.0:
            raise exceptions.StructureError(
                f"No matching atoms for RDF pair {pair!r}."
            )
        return r_centers, g_sum / norm_accum
