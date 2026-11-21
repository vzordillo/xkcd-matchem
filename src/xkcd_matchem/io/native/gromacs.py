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
"""Native parsers for GROMACS coordinate (.gro) and plot (.xvg) files."""

from __future__ import annotations

import pathlib
import re

import numpy as np

from xkcd_matchem.core import exceptions
from xkcd_matchem.core import spectroscopy
from xkcd_matchem.core import structure
from xkcd_matchem.data import elements
from xkcd_matchem.io import registry


def _guess_gro_element(atom_name: str) -> str:
    """Infers an element symbol from a GROMACS atom label."""
    cleaned = re.sub(r"[^A-Za-z]", "", atom_name.strip())
    if (
        len(cleaned) >= 2
        and cleaned[:2].capitalize() in elements.ELEMENTS_BY_SYMBOL
    ):
        two = cleaned[:2].capitalize()
        # Distinguish calcium/chlorine/sodium from common C-alpha/C-beta labels.
        if two in ("Na", "Cl", "Mg", "Fe", "Zn", "Cu", "Br", "Si"):
            return two
    if cleaned:
        return elements.get_element(cleaned[0].upper()).symbol
    raise exceptions.ParseError(
        f"Cannot infer element from GRO atom {atom_name!r}"
    )


def read_gro(path: str | pathlib.Path) -> structure.AtomsData:
    """Reads a GROMACS ``.gro`` file converting coordinates from nm to A."""
    p = pathlib.Path(path)
    lines = p.read_text(encoding="utf-8").splitlines()
    if len(lines) < 4:
        raise exceptions.ParseError(f"Truncated GROMACS GRO file: {p}")
    title = lines[0].strip()
    n_atoms = int(lines[1].strip())
    if len(lines) < 3 + n_atoms:
        raise exceptions.ParseError(f"Truncated GRO atom block in {p}.")
    syms: list[str] = []
    labels: list[str] = []
    pos = np.zeros((n_atoms, 3), dtype=np.float64)
    for k in range(n_atoms):
        line = lines[2 + k]
        atom_name = line[10:15].strip()
        x_nm = float(line[20:28])
        y_nm = float(line[28:36])
        z_nm = float(line[36:44])
        pos[k] = [x_nm * 10.0, y_nm * 10.0, z_nm * 10.0]
        labels.append(atom_name)
        syms.append(_guess_gro_element(atom_name))

    box_vals = [float(x) * 10.0 for x in lines[2 + n_atoms].split()]
    if len(box_vals) == 3:
        cell = np.diag(box_vals).astype(np.float64)
    elif len(box_vals) >= 9:
        cell = np.array(
            [
                [box_vals[0], box_vals[3], box_vals[4]],
                [box_vals[5], box_vals[1], box_vals[6]],
                [box_vals[7], box_vals[8], box_vals[2]],
            ],
            dtype=np.float64,
        )
    else:
        cell = None
    return structure.AtomsData(
        symbols=syms,
        positions=pos,
        cell=cell,
        pbc=(cell is not None, cell is not None, cell is not None),
        atom_labels=labels,
        metadata={"title": title, "source": str(p)},
    )


def read_xvg(path: str | pathlib.Path) -> spectroscopy.Spectrum:
    """Reads a GROMACS ``.xvg`` plot (RDF, energy, RMSD) into ``Spectrum``."""
    p = pathlib.Path(path)
    lines = p.read_text(encoding="utf-8").splitlines()
    title = ""
    x_label = "x"
    y_label = "y"
    rows: list[list[float]] = []

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("@"):
            m_title = re.search(r'@\s*title\s+"([^"]*)"', stripped)
            if m_title:
                title = m_title.group(1)
            m_xl = re.search(r'@\s*xaxis\s+label\s+"([^"]*)"', stripped)
            if m_xl:
                x_label = m_xl.group(1)
            m_yl = re.search(r'@\s*yaxis\s+label\s+"([^"]*)"', stripped)
            if m_yl:
                y_label = m_yl.group(1)
            continue
        rows.append([float(tok) for tok in stripped.split()])

    if not rows:
        raise exceptions.ParseError(f"No numeric series rows found in XVG: {p}")
    arr = np.array(rows, dtype=np.float64)
    kind = "rdf" if "g(r)" in y_label or "radial" in title.lower() else "xvg"
    return spectroscopy.Spectrum(
        x=arr[:, 0],
        y=arr[:, 1],
        kind=kind,
        x_label=x_label,
        y_label=y_label,
        metadata={"title": title, "source": str(p), "all_columns": arr},
    )


registry.register_format(
    registry.FormatSpec(
        name="gro",
        extensions=(".gro",),
        filenames=(),
        reader=read_gro,
        description="GROMACS GRO structure in nm converted to Angstrom",
    )
)
registry.register_format(
    registry.FormatSpec(
        name="xvg",
        extensions=(".xvg",),
        filenames=(),
        reader=read_xvg,
        description="GROMACS XVG curve (RDF, RMSD, thermodynamic time series)",
    )
)
