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
"""Native parser for Gaussian CUBE volumetric density and orbital files."""

from __future__ import annotations

import pathlib

import numpy as np

from xkcd_matchem.core import electronic
from xkcd_matchem.core import exceptions
from xkcd_matchem.core import structure
from xkcd_matchem.data import elements
from xkcd_matchem.io import registry


def read_cube(path: str | pathlib.Path) -> electronic.VolumetricGrid:
    """Reads a Gaussian ``.cube`` file, converting Bohr coordinates to Angstrom.

    Args:
      path: Filesystem path to the ``.cube`` file.

    Returns:
      ``VolumetricGrid`` containing the 3D scalar field and embedded
      ``AtomsData``.

    Raises:
      ParseError: If the header or volumetric grid is truncated or invalid.
    """
    p = pathlib.Path(path)
    lines = p.read_text(encoding="utf-8").splitlines()
    if len(lines) < 7:
        raise exceptions.ParseError(f"Truncated Gaussian CUBE file: {p}")
    bohr = elements.CODATA2022.bohr_to_angstrom
    line3 = lines[2].split()
    n_raw = int(line3[0])
    has_mo_line = n_raw < 0
    n_atoms = abs(n_raw)
    origin = np.array([float(x) for x in line3[1:4]], dtype=np.float64) * bohr

    dims: list[int] = []
    voxels = np.zeros((3, 3), dtype=np.float64)
    for axis_idx in range(3):
        parts = lines[3 + axis_idx].split()
        n_pts = int(parts[0])
        step_sign = -1.0 if n_pts < 0 else 1.0
        dims.append(abs(n_pts))
        scale = 1.0 if step_sign < 0.0 else bohr
        voxels[axis_idx] = (
            np.array([float(x) for x in parts[1:4]], dtype=np.float64) * scale
        )

    symbols: list[str] = []
    coords = np.zeros((n_atoms, 3), dtype=np.float64)
    for k in range(n_atoms):
        parts = lines[6 + k].split()
        atomic_num = int(parts[0])
        symbols.append(elements.get_element(atomic_num).symbol)
        coords[k] = (
            np.array([float(x) for x in parts[2:5]], dtype=np.float64) * bohr
        )

    cursor = 6 + n_atoms + (1 if has_mo_line else 0)
    values: list[float] = []
    n_total = dims[0] * dims[1] * dims[2]
    while cursor < len(lines) and len(values) < n_total:
        row = lines[cursor].strip()
        if row:
            values.extend(float(tok) for tok in row.split())
        cursor += 1
    if len(values) < n_total:
        raise exceptions.ParseError(
            f"Truncated volumetric data in CUBE file: {p}"
        )

    grid_data = np.array(values[:n_total], dtype=np.float64).reshape(
        tuple(dims)
    )
    atoms = structure.AtomsData(
        symbols=symbols,
        positions=coords,
        metadata={"comment": lines[0].strip(), "source": str(p)},
    )
    return electronic.VolumetricGrid(
        origin=origin,
        voxel_vectors=voxels,
        data=grid_data,
        atoms=atoms,
        units="a.u.",
        metadata={"title": lines[0].strip(), "source": str(p)},
    )


registry.register_format(
    registry.FormatSpec(
        name="cube",
        extensions=(".cube", ".cub"),
        filenames=(),
        reader=read_cube,
        description="Gaussian CUBE 3D volumetric grid and molecular geometry",
    )
)
