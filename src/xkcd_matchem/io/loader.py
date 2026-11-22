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
"""Unified loading and saving entrypoints dispatching native & adapter I/O."""

from __future__ import annotations

import pathlib
from typing import Any

from xkcd_matchem.core import exceptions
from xkcd_matchem.core import structure
from xkcd_matchem.core import trajectory
from xkcd_matchem.data import samples_registry
from xkcd_matchem.io import native as _native  # noqa: F401
from xkcd_matchem.io import registry
from xkcd_matchem.io.adapters import ase_adapter
from xkcd_matchem.io.native import lammps
from xkcd_matchem.io.native import pdb
from xkcd_matchem.io.native import vasp
from xkcd_matchem.io.native import xyz


def _resolve_input_path(source: str | pathlib.Path) -> pathlib.Path:
    """Resolves a filesystem path or falls back to a bundled sample name."""
    p = pathlib.Path(source)
    if p.is_file():
        return p
    if isinstance(source, str) and source in samples_registry.list_samples():
        return samples_registry.get_sample_path(source)
    raise FileNotFoundError(f"Structure or data file not found: {source}")


def load(
    source: str | pathlib.Path | structure.AtomsData,
    fmt: str | None = None,
) -> Any:
    """Loads a molecule, crystal, grid, band structure, DOS, or spectrum.

    Args:
      source: Path to a file, name of a bundled sample, or an existing object.
      fmt: Optional format override (e.g., ``"vasp"``, ``"cif"``, ``"xyz"``).

    Returns:
      Parsed ``AtomsData``, ``VolumetricGrid``, ``BandStructure``,
      ``DensityOfStates``, ``Spectrum``, or ``Trajectory``.
    """
    if isinstance(source, structure.AtomsData):
        return source
    path = _resolve_input_path(source)
    if fmt is not None:
        spec = registry.get_format(fmt)
        return spec.reader(path)
    try:
        spec = registry.detect_format(path)
        return spec.reader(path)
    except exceptions.UnsupportedFormatError:
        try:
            return ase_adapter.read_via_ase(path)
        except exceptions.OptionalDependencyError as exc:
            raise exceptions.UnsupportedFormatError(
                f"Cannot detect format for {path} and 'ase' is not installed."
            ) from exc


def load_sample(name: str) -> Any:
    """Loads one of the bundled sample data files by filename."""
    return load(samples_registry.get_sample_path(name))


def load_trajectory(
    source: str | pathlib.Path,
    fmt: str | None = None,
) -> trajectory.Trajectory:
    """Loads a multi-frame trajectory (XYZ/extXYZ, XDATCAR, LAMMPS, PDB)."""
    path = _resolve_input_path(source)
    key = fmt.lower() if fmt is not None else registry.detect_format(path).name
    result: trajectory.Trajectory | None = None
    if key == "xyz":
        result = xyz.read_xyz_trajectory(path)
    elif key == "xdatcar":
        result = vasp.read_xdatcar(path)
    elif key == "vasp":
        result = (
            vasp.read_xdatcar(path)
            if "XDATCAR" in path.name.upper()
            else trajectory.Trajectory(frames=[vasp.read_poscar(path)])
        )
    elif key == "lammpstrj":
        result = lammps.read_lammpstrj(path)
    elif key == "pdb":
        result = pdb.read_pdb_trajectory(path)
    else:
        obj = load(path, fmt=fmt)
        if isinstance(obj, trajectory.Trajectory):
            result = obj
        elif isinstance(obj, structure.AtomsData):
            result = trajectory.Trajectory(frames=[obj])
    if result is None:
        raise exceptions.UnsupportedFormatError(
            f"Format {key!r} does not produce a trajectory."
        )
    return result


def save(
    atoms: structure.AtomsData,
    path: str | pathlib.Path,
    fmt: str | None = None,
) -> pathlib.Path:
    """Saves an ``AtomsData`` structure using a registered native writer."""
    out = pathlib.Path(path)
    spec = (
        registry.get_format(fmt)
        if fmt is not None
        else registry.detect_format(out)
    )
    if spec.writer is None:
        raise exceptions.UnsupportedFormatError(
            f"Format {spec.name!r} does not implement a native writer."
        )
    spec.writer(atoms, out)
    return out
