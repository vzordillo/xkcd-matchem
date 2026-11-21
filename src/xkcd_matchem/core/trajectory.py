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

import dataclasses
import math
from typing import Any

import numpy as np

from xkcd_matchem.core import exceptions
from xkcd_matchem.core import geometry
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
        """Normalizes 1D scalar series and pulls energies if stored."""
        n_frames = len(self.frames)
        if self.times is not None:
            self.times = np.asarray(self.times, dtype=np.float64)
            if self.times.shape != (n_frames,):
                raise exceptions.StructureError(
                    "times length must match frames count."
                )
        if self.energies is None and n_frames > 0:
            extracted = [frame.energies.get("total") for frame in self.frames]
            if all(val is not None for val in extracted):
                self.energies = np.array(extracted, dtype=np.float64)
        elif self.energies is not None:
            self.energies = np.asarray(self.energies, dtype=np.float64)
            if self.energies.shape != (n_frames,):
                raise exceptions.StructureError(
                    "energies length must match frames count."
                )
        if self.temperatures is not None:
            self.temperatures = np.asarray(self.temperatures, dtype=np.float64)
        if self.pressures is not None:
            self.pressures = np.asarray(self.pressures, dtype=np.float64)

    def __len__(self) -> int:
        """Returns the number of frames."""
        return len(self.frames)

    def __getitem__(self, index: int) -> structure.AtomsData:
        """Retrieves a single frame by index."""
        return self.frames[index]

    def rmsd(self, ref_index: int = 0) -> np.ndarray:
        """Computes Kabsch-aligned heavy-atom/all-atom RMSD relative to a frame.

        Args:
          ref_index: Reference frame index.

        Returns:
          1D float64 RMSD array in Angstrom of shape ``(n_frames,)``.
        """
        if not self.frames:
            return np.zeros((0,), dtype=np.float64)
        ref = self.frames[ref_index].positions
        ref_centered = ref - np.mean(ref, axis=0)
        out = np.zeros(len(self.frames), dtype=np.float64)
        for idx, frame in enumerate(self.frames):
            cur = frame.positions
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
            return vals
        if u_key == "kj/mol":
            return vals * elements.CODATA2022.ev_to_kj_mol
        if u_key == "kcal/mol":
            return vals * elements.CODATA2022.ev_to_kcal_mol
        raise exceptions.StructureError(f"Unsupported energy unit {unit!r}.")

    def _frame_pair_distances(
        self,
        frame: structure.AtomsData,
        sym_a: str,
        sym_b: str,
        r_max: float,
    ) -> tuple[list[float], int, int]:
        """Collects valid A-B pair distances within ``r_max`` for one frame."""
        idx_a = [i for i, s in enumerate(frame.symbols) if s == sym_a]
        idx_b = [i for i, s in enumerate(frame.symbols) if s == sym_b]
        if not idx_a or not idx_b:
            return [], 0, 0
        dists: list[float] = []
        for ia in idx_a:
            for ib in idx_b:
                if ia == ib:
                    continue
                vec = frame.positions[ib] - frame.positions[ia]
                if frame.cell is not None and any(frame.pbc):
                    vec = geometry.minimum_image_vector(
                        vec, frame.cell, frame.pbc
                    )
                dist = float(np.linalg.norm(vec))
                if 1e-6 < dist < r_max:
                    dists.append(dist)
        return dists, len(idx_a), len(idx_b) - (1 if sym_a == sym_b else 0)

    def compute_rdf(
        self,
        pair: tuple[str, str],
        r_max: float = 8.0,
        n_bins: int = 80,
        start: int = 0,
        stop: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Computes the radial distribution function ``g(r)`` for a pair.

        Args:
          pair: Tuple of element symbols ``(sym_a, sym_b)``.
          r_max: Maximum pair separation in Angstrom.
          n_bins: Number of radial histogram bins.
          start: First frame index to include.
          stop: Exclusive last frame index to include.

        Returns:
          Tuple ``(r_centers, g_r)`` of 1D float64 arrays.

        Raises:
          StructureError: If no frames match or ``r_max <= 0``.
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
        shell_vols = 4.0 * math.pi * (r_centers**2) * (edges[1] - edges[0])
        g_sum = np.zeros(n_bins, dtype=np.float64)
        norm_accum = 0.0

        for frame in selected:
            dists, n_a, n_b = self._frame_pair_distances(
                frame, sym_a, sym_b, r_max
            )
            if n_a == 0:
                continue
            vol = frame.volume()
            if vol <= 0.0:
                span = np.ptp(frame.positions, axis=0) + 4.0
                vol = float(np.prod(np.maximum(span, 4.0)))
            rho = max(float(n_a * n_b), 1.0) / vol
            norm_accum += 1.0
            if np.all(shell_vols > 0.0):
                hist_frame, _ = np.histogram(dists, bins=edges)
                g_sum += hist_frame / (rho * shell_vols)

        if norm_accum == 0.0:
            raise exceptions.StructureError(
                f"No matching atoms for RDF pair {pair!r}."
            )
        return r_centers, g_sum / norm_accum
