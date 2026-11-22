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
import warnings

import numpy as np

from xkcd_matchem.core import exceptions
from xkcd_matchem.core import geometry
from xkcd_matchem.core import neighbors
from xkcd_matchem.core import structure
from xkcd_matchem.data import elements


def compute_tetrahedral_order(
    atoms: structure.AtomsData,
    center_symbol: str = "O",
    max_cutoff: float = 5.5,
) -> float:
    """Computes the Errington-Debenedetti tetrahedral order parameter ``q``.

    For each atom of ``center_symbol``, evaluates::

        q_i = 1 - (3/8) * sum_{j=1..3} sum_{k=j+1..4} (cos(psi_jk) + 1/3)^2

    over its 4 nearest periodic neighbors of the same species. Ideal
    tetrahedral networks (Ice Ih, diamond Si) yield ``q = 1.0``, while
    disordered liquid water typically sits around ``q ~ 0.45-0.65``.

    Args:
      atoms: Input molecular or periodic crystal structure.
      center_symbol: Species whose local tetrahedral network is evaluated.
      max_cutoff: Initial neighbor search radius in Angstrom (expanded if
        needed to locate at least 4 neighbors per center).

    Returns:
      Mean tetrahedral order parameter ``q`` in ``[-3, 1]``.

    Raises:
      StructureError: If fewer than 5 matching atoms exist or 4 neighbors
        cannot be located within search range.
    """
    sym = elements.get_element(center_symbol).symbol
    sel = [i for i, s in enumerate(atoms.symbols) if s == sym]
    if len(sel) < 5:
        raise exceptions.StructureError(
            f"Need at least 5 {sym!r} atoms to compute tetrahedral order q."
        )
    sub = structure.AtomsData(
        symbols=[sym] * len(sel),
        positions=atoms.positions[sel],
        cell=atoms.cell,
        pbc=atoms.pbc,
    )
    hits = neighbors.find_periodic_neighbors(
        sub,
        max_cutoff=max_cutoff,
        min_cutoff=1e-4,
        use_pbc=sub.cell is not None and any(sub.pbc),
        symmetric=True,
    )
    by_i: dict[int, list[neighbors.NeighborHit]] = {}
    for hit in hits:
        by_i.setdefault(hit.i, []).append(hit)
    q_vals: list[float] = []
    for i in range(len(sub)):
        nbrs = sorted(by_i.get(i, []), key=lambda h: h.distance)[:4]
        if len(nbrs) < 4:
            raise exceptions.StructureError(
                f"Atom {i} has fewer than 4 {sym!r} neighbors within "
                f"{max_cutoff:.2f} A."
            )
        vecs = np.array(
            [h.vector / max(h.distance, 1e-12) for h in nbrs],
            dtype=np.float64,
        )
        acc = 0.0
        for j in range(3):
            for k in range(j + 1, 4):
                cos_t = float(np.clip(np.dot(vecs[j], vecs[k]), -1.0, 1.0))
                acc += (cos_t + 1.0 / 3.0) ** 2
        q_vals.append(1.0 - (3.0 / 8.0) * acc)
    return float(np.mean(q_vals))


def _validate_variable_cell_mode(mode: str) -> str:
    """Validates trajectory periodic unwrapping coordinate mode."""
    key = mode.strip().lower()
    if key not in ("cartesian", "fractional"):
        raise exceptions.StructureError(
            f"Unsupported variable_cell_mode {mode!r}; "
            "expected 'cartesian' or 'fractional'."
        )
    return key


def _frame_lossless_unwrapped(
    frame: structure.AtomsData,
) -> np.ndarray | None:
    """Returns explicit unwrapped Cartesian coordinates if stored on frame."""
    raw_u = frame.metadata.get("unwrapped_positions")
    if raw_u is not None:
        arr = np.asarray(raw_u, dtype=np.float64)
        if arr.shape == frame.positions.shape and np.all(np.isfinite(arr)):
            return arr
    if bool(frame.metadata.get("coords_unwrapped", False)):
        return frame.positions
    return None


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

    def _unwrapped_positions(
        self,
        idx_arr: np.ndarray,
        *,
        variable_cell_mode: str = "cartesian",
    ) -> list[np.ndarray]:
        """Unwraps periodic coordinates across frames.

        Prefers lossless ``frame.metadata["unwrapped_positions"]`` or direct
        ``coords_unwrapped=True`` coordinates when present on all frames.
        Otherwise reconstructs cumulative trajectories from frame-to-frame
        minimum-image displacements under the Nyquist sampling assumption
        (inter-frame displacement strictly below the Wigner-Seitz inradius),
        emitting a ``UserWarning`` if any jump exceeds 45% of that inradius.
        """
        mode = _validate_variable_cell_mode(variable_cell_mode)
        lossless = [_frame_lossless_unwrapped(f) for f in self.frames]
        if lossless and all(arr is not None for arr in lossless):
            return [
                arr[idx_arr].copy() for arr in lossless if arr is not None
            ]

        first_f = self.frames[0]
        unwrapped = [first_f.positions[idx_arr].copy()]
        s_unwrapped = (
            first_f.positions[idx_arr] @ np.linalg.inv(first_f.cell)
            if mode == "fractional" and first_f.cell is not None
            else None
        )
        worst_frame = -1
        worst_r_ws = 0.0
        worst_step = 0.0
        worst_ratio = 0.45

        for k in range(1, len(self.frames)):
            prev_f = self.frames[k - 1]
            cur_f = self.frames[k]
            prev_raw = prev_f.positions[idx_arr]
            cur_raw = cur_f.positions[idx_arr]
            has_pbc = cur_f.cell is not None and any(cur_f.pbc)
            if (
                mode == "fractional"
                and has_pbc
                and cur_f.cell is not None
                and prev_f.cell is not None
                and s_unwrapped is not None
            ):
                s_prev = prev_raw @ np.linalg.inv(prev_f.cell)
                s_cur = cur_raw @ np.linalg.inv(cur_f.cell)
                mic_disp = geometry.minimum_image_vector(
                    (s_cur - s_prev) @ cur_f.cell, cur_f.cell, cur_f.pbc
                )
                s_unwrapped = s_unwrapped + mic_disp @ np.linalg.inv(
                    cur_f.cell
                )
                unwrapped.append(
                    np.asarray(s_unwrapped @ cur_f.cell, dtype=np.float64)
                )
            else:
                delta = cur_raw - prev_raw
                if has_pbc and cur_f.cell is not None:
                    delta = geometry.minimum_image_vector(
                        delta, cur_f.cell, cur_f.pbc
                    )
                mic_disp = delta
                unwrapped.append(unwrapped[-1] + delta)

            if has_pbc and cur_f.cell is not None and mic_disp.size:
                r_ws = geometry.max_min_image_radius(cur_f.cell, cur_f.pbc)
                max_step = float(np.max(np.linalg.norm(mic_disp, axis=1)))
                if math.isfinite(r_ws) and r_ws > 0.0:
                    ratio = max_step / r_ws
                    if ratio > worst_ratio:
                        worst_ratio = ratio
                        worst_frame = k
                        worst_r_ws = r_ws
                        worst_step = max_step

        if worst_frame >= 0:
            warnings.warn(
                f"Frame {worst_frame} minimum-image jump ({worst_step:.3f} A) "
                f"exceeds 45% of Wigner-Seitz inradius ({worst_r_ws:.3f} A); "
                "sampling may violate the Nyquist displacement bound. Supply "
                "LAMMPS ix/iy/iz image flags or xu/yu/zu unwrapped positions.",
                UserWarning,
                stacklevel=3,
            )
        return unwrapped

    def _resolve_selection(
        self,
        ref_index: int,
        indices: Sequence[int] | None,
    ) -> tuple[int, np.ndarray]:
        """Validates ``ref_index`` and atom selection ``indices``."""
        n_frames = len(self.frames)
        ref_idx = ref_index if ref_index >= 0 else n_frames + ref_index
        if not 0 <= ref_idx < n_frames:
            raise exceptions.StructureError(
                f"Reference frame index {ref_index} out of range."
            )
        n_atoms = len(self.frames[ref_idx])
        idx_arr: np.ndarray
        if indices is None:
            if n_atoms == 0:
                raise exceptions.StructureError(
                    "Reference frame contains zero atoms."
                )
            idx_arr = np.arange(n_atoms, dtype=np.int64)
        else:
            idx_arr = np.asarray(list(indices), dtype=np.int64)
            if (
                idx_arr.size == 0
                or np.any(idx_arr < 0)
                or np.any(idx_arr >= n_atoms)
            ):
                raise exceptions.StructureError(
                    "Invalid atom selection indices for trajectory analysis."
                )
        max_sel = int(np.max(idx_arr))
        if any(len(frame) <= max_sel for frame in self.frames):
            raise exceptions.StructureError(
                "Atom count changes across trajectory frames for selection."
            )
        return ref_idx, idx_arr

    def rmsd(
        self,
        ref_index: int = 0,
        indices: Sequence[int] | None = None,
        unwrap: bool = True,
        *,
        variable_cell_mode: str = "cartesian",
    ) -> np.ndarray:
        """Computes Kabsch-aligned RMSD relative to a reference frame.

        Args:
          ref_index: Reference frame index.
          indices: Optional atom indices to select for alignment and RMSD.
          unwrap: Whether to unwrap periodic displacements across frames first
            (under the Nyquist displacement bound ``|dr| < r_WS`` unless
            lossless unwrapped positions or image flags are present).
          variable_cell_mode: Unwrapping coordinate frame for variable-cell
            trajectories (``"cartesian"`` or ``"fractional"``).

        Returns:
          1D float64 RMSD array in Angstrom of shape ``(n_frames,)``.
        """
        _validate_variable_cell_mode(variable_cell_mode)
        if not self.frames:
            return np.zeros((0,), dtype=np.float64)
        ref_idx, idx_arr = self._resolve_selection(ref_index, indices)
        n_ref = len(self.frames[ref_idx])
        if unwrap and all(len(f) == n_ref for f in self.frames):
            coords = np.stack(
                self._unwrapped_positions(
                    idx_arr, variable_cell_mode=variable_cell_mode
                ),
                axis=0,
            )
        else:
            coords = np.stack(
                [f.positions[idx_arr] for f in self.frames], axis=0
            )

        centered = coords - np.mean(coords, axis=1, keepdims=True)
        ref_centered = centered[ref_idx]
        covs = np.einsum("fni,nj->fij", centered, ref_centered)
        u_all, _, vt_all = np.linalg.svd(covs)
        det_signs = np.where(np.linalg.det(u_all @ vt_all) >= 0.0, 1.0, -1.0)
        u_scaled = u_all.copy()
        u_scaled[:, :, 2] *= det_signs[:, None]
        rots = u_scaled @ vt_all
        aligned = np.einsum("fni,fij->fnj", centered, rots)
        diffs = aligned - ref_centered[None, :, :]
        return np.asarray(
            np.sqrt(np.mean(np.sum(diffs * diffs, axis=2), axis=1)),
            dtype=np.float64,
        )

    def msd(
        self,
        ref_index: int = 0,
        indices: Sequence[int] | None = None,
        unwrap: bool = True,
        remove_com: bool = False,
        *,
        variable_cell_mode: str = "cartesian",
    ) -> np.ndarray:
        """Computes periodic-unwrapped mean-squared displacement (MSD).

        Args:
          ref_index: Reference frame index ``t_0``.
          indices: Optional atom indices to include in the MSD average.
          unwrap: Whether to unwrap periodic boundary jumps cumulatively first
            (subject to the Nyquist displacement bound ``|dr| < r_WS`` unless
            explicit unwrapped positions or image flags are stored).
          remove_com: Subtract overall center-of-mass drift before MSD.
          variable_cell_mode: Unwrapping coordinate frame for variable-cell
            trajectories (``"cartesian"`` or ``"fractional"``).

        Returns:
          1D float64 MSD array in squared Angstrom of shape ``(n_frames,)``.
        """
        _validate_variable_cell_mode(variable_cell_mode)
        if not self.frames:
            return np.zeros((0,), dtype=np.float64)
        ref_idx, idx_arr = self._resolve_selection(ref_index, indices)
        n_ref = len(self.frames[ref_idx])
        if remove_com and all(len(f) == n_ref for f in self.frames):
            all_idx = np.arange(n_ref, dtype=np.int64)
            full_coords = (
                np.stack(
                    self._unwrapped_positions(
                        all_idx, variable_cell_mode=variable_cell_mode
                    ),
                    axis=0,
                )
                if unwrap
                else np.stack([f.positions for f in self.frames], axis=0)
            )
            masses = self.frames[ref_idx].atomic_masses()
            tot_m = max(float(np.sum(masses)), 1e-12)
            com = np.einsum("fni,n->fi", full_coords, masses) / tot_m
            coords = (full_coords - com[:, None, :])[:, idx_arr, :]
        elif unwrap and all(len(f) == n_ref for f in self.frames):
            coords = np.stack(
                self._unwrapped_positions(
                    idx_arr, variable_cell_mode=variable_cell_mode
                ),
                axis=0,
            )
        else:
            coords = np.stack(
                [f.positions[idx_arr] for f in self.frames], axis=0
            )
        disp = coords - coords[ref_idx : ref_idx + 1]
        return np.asarray(
            np.mean(np.sum(disp * disp, axis=2), axis=1), dtype=np.float64
        )

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
        *,
        same_species_correction: bool = True,
    ) -> tuple[np.ndarray, int, float]:
        """Collects periodic pair distances up to ``r_max`` using cKDTree."""
        idx_ab = [
            i for i, s in enumerate(frame.symbols) if s in (sym_a, sym_b)
        ]
        n_a = sum(1 for i in idx_ab if frame.symbols[i] == sym_a)
        n_b = sum(1 for i in idx_ab if frame.symbols[i] == sym_b)
        if n_a == 0 or n_b == 0:
            return np.zeros((0,), dtype=np.float64), 0, 0.0
        if sym_a == sym_b and n_a > 1 and same_species_correction:
            pair_count = float(n_a * (n_a - 1))
        else:
            pair_count = float(n_a * n_b)
        sub_frame = (
            frame
            if len(idx_ab) == len(frame)
            else structure.AtomsData(
                symbols=[frame.symbols[i] for i in idx_ab],
                positions=frame.positions[idx_ab],
                cell=frame.cell,
                pbc=frame.pbc,
            )
        )
        hits = neighbors.find_periodic_neighbors(
            sub_frame,
            max_cutoff=r_max,
            min_cutoff=1e-6,
            use_pbc=True,
            symmetric=True,
        )
        dists = np.array(
            [
                hit.distance
                for hit in hits
                if hit.distance < r_max
                and sub_frame.symbols[hit.i] == sym_a
                and sub_frame.symbols[hit.j] == sym_b
            ],
            dtype=np.float64,
        )
        return dists, n_a, pair_count

    def compute_rdf(
        self,
        pair: tuple[str, str],
        r_max: float = 8.0,
        n_bins: int = 80,
        start: int = 0,
        stop: int | None = None,
        strict: bool = False,
        *,
        same_species_correction: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Computes the radial distribution function ``g(r)`` for a 3D cell.

        Evaluates the exact finite-size shell estimator across frames::

            g_AB(r_k) = < V * dn_AB(r_k) / (N_pairs * dV_shell(r_k)) >

        with ``dV_shell(r_k) = (4*pi/3) * (r_k1^3 - r_k0^3)`` and
        ``N_pairs = N_A * (N_A - 1)`` when ``sym_a == sym_b``, ``N_A > 1``,
        and ``same_species_correction=True``, or ``N_A * N_B`` otherwise.

        Args:
          pair: Tuple of element symbols ``(sym_a, sym_b)``.
          r_max: Maximum pair separation in Angstrom.
          n_bins: Number of radial histogram bins.
          start: First frame index to include.
          stop: Exclusive last frame index to include.
          strict: If True, reject ``r_max`` exceeding the Wigner-Seitz
            minimum-image inradius.
          same_species_correction: Use exact ``N_A * (N_A - 1)`` distinct pair
            count for same-species ``g_AA(r)`` when ``N_A > 1``.

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
                frame,
                sym_a,
                sym_b,
                r_max,
                same_species_correction=same_species_correction,
            )
            if n_a == 0 or pair_count <= 0.0:
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

    def tetrahedral_order(
        self,
        center_symbol: str = "O",
        max_cutoff: float = 5.5,
    ) -> np.ndarray:
        """Evaluates Errington-Debenedetti tetrahedral order ``q`` per frame."""
        return np.array(
            [
                compute_tetrahedral_order(
                    frame,
                    center_symbol=center_symbol,
                    max_cutoff=max_cutoff,
                )
                for frame in self.frames
            ],
            dtype=np.float64,
        )
