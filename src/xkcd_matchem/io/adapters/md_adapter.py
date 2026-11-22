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
"""Lazy adapters converting MDAnalysis Universes and MDTraj Trajectories."""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from xkcd_matchem.core import exceptions
from xkcd_matchem.core import geometry
from xkcd_matchem.core import structure
from xkcd_matchem.core import trajectory
from xkcd_matchem.data import elements


def _require_mdanalysis() -> Any:
    """Lazily checks ``MDAnalysis`` availability."""
    try:
        import MDAnalysis
    except ImportError as exc:
        raise exceptions.OptionalDependencyError(
            "MDAnalysis adapter requires 'MDAnalysis'. Install via: "
            "pip install 'xkcd-matchem[md]'"
        ) from exc
    return MDAnalysis


def _require_mdtraj() -> Any:
    """Lazily checks ``mdtraj`` availability."""
    try:
        import mdtraj
    except ImportError as exc:
        raise exceptions.OptionalDependencyError(
            "MDTraj adapter requires 'mdtraj'. Install via: "
            "pip install 'xkcd-matchem[md]'"
        ) from exc
    return mdtraj


def _infer_mda_symbol(raw_name: str) -> str:
    """Infers an element symbol from an MDAnalysis atom name."""
    cleaned = re.sub(r"[^A-Za-z]", "", str(raw_name).strip())
    if len(cleaned) >= 2:
        cand2 = cleaned[:2].capitalize()
        if cand2 in ("Na", "Cl", "Mg", "Fe", "Zn", "Cu", "Br", "Si", "Li", "K"):
            return cand2
    if cleaned:
        return elements.get_element(cleaned[0].upper()).symbol
    raise exceptions.ParseError(
        f"Cannot infer element symbol from MDAnalysis atom name {raw_name!r}."
    )


def from_mdanalysis(
    universe: Any,
    stride: int = 1,
    max_frames: int = 200,
) -> trajectory.Trajectory:
    """Converts an ``MDAnalysis.Universe`` trajectory into ``Trajectory``."""
    if not hasattr(universe, "atoms"):
        _require_mdanalysis()
        raise exceptions.ParseError("Expected an MDAnalysis.Universe instance.")
    elements_attr = getattr(universe.atoms, "elements", None)
    symbols = (
        [elements.get_element(str(el)).symbol for el in elements_attr]
        if elements_attr is not None
        and len(elements_attr) == len(universe.atoms)
        and all(str(el).strip() for el in elements_attr)
        else [_infer_mda_symbol(name) for name in universe.atoms.names]
    )
    frames: list[structure.AtomsData] = []
    times: list[float] = []
    for step_idx, ts in enumerate(universe.trajectory[::stride]):
        if step_idx >= max_frames:
            break
        pos = np.array(universe.atoms.positions, dtype=np.float64)
        dims = getattr(ts, "dimensions", None)
        cell = None
        if dims is not None and len(dims) >= 6 and dims[0] > 0.0:
            cell = geometry.cell_vectors_from_parameters(
                *[float(x) for x in dims[:6]]
            )
        elif dims is not None and len(dims) >= 3 and dims[0] > 0.0:
            cell = np.diag(dims[:3]).astype(np.float64)
        frames.append(
            structure.AtomsData(
                symbols=list(symbols),
                positions=pos,
                cell=cell,
                pbc=(cell is not None, cell is not None, cell is not None),
            )
        )
        times.append(float(getattr(ts, "time", step_idx)))
    return trajectory.Trajectory(
        frames=frames, times=np.array(times, dtype=np.float64)
    )


def from_mdtraj(
    traj: Any,
    stride: int = 1,
    max_frames: int = 200,
) -> trajectory.Trajectory:
    """Converts ``mdtraj.Trajectory`` (nm -> Angstrom) to ``Trajectory``."""
    if not hasattr(traj, "xyz") or not hasattr(traj, "topology"):
        _require_mdtraj()
        raise exceptions.ParseError("Expected an mdtraj.Trajectory instance.")
    symbols = [atom.element.symbol for atom in traj.topology.atoms]
    frames: list[structure.AtomsData] = []
    times: list[float] = []
    unitcell_vectors = getattr(traj, "unitcell_vectors", None)
    for idx in range(0, len(traj), stride):
        if len(frames) >= max_frames:
            break
        pos_a = np.asarray(traj.xyz[idx], dtype=np.float64) * 10.0
        cell = (
            np.asarray(unitcell_vectors[idx], dtype=np.float64) * 10.0
            if unitcell_vectors is not None
            else None
        )
        frames.append(
            structure.AtomsData(
                symbols=list(symbols),
                positions=pos_a,
                cell=cell,
                pbc=(cell is not None, cell is not None, cell is not None),
            )
        )
        times.append(float(traj.time[idx]))
    return trajectory.Trajectory(
        frames=frames, times=np.array(times, dtype=np.float64)
    )
