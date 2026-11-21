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
"""Native parsers for CP2K .ener files and xTB output summaries."""

from __future__ import annotations

import pathlib
import re

import numpy as np

from xkcd_matchem.core import exceptions
from xkcd_matchem.core import spectroscopy
from xkcd_matchem.core import structure
from xkcd_matchem.data import elements
from xkcd_matchem.io import registry


def read_cp2k_ener(path: str | pathlib.Path) -> spectroscopy.Spectrum:
    """Reads a CP2K ``.ener`` MD energy trace into ``Spectrum`` (time vs E)."""
    p = pathlib.Path(path)
    rows: list[list[float]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        rows.append([float(tok) for tok in stripped.split()])
    if not rows:
        raise exceptions.ParseError(
            f"No data rows found in CP2K .ener file: {p}"
        )
    arr = np.array(rows, dtype=np.float64)
    times_fs = arr[:, 1]
    pot_ev = arr[:, 4] * elements.CODATA2022.hartree_to_ev
    return spectroscopy.Spectrum(
        x=times_fs,
        y=pot_ev,
        kind="generic",
        x_label="Time (fs)",
        y_label="Potential Energy (eV)",
        metadata={"source": str(p), "raw": arr},
    )


def read_xtb_out(path: str | pathlib.Path) -> structure.AtomsData:
    """Reads total energy and HOMO-LUMO gap from an xTB summary output."""
    p = pathlib.Path(path)
    text = p.read_text(encoding="utf-8", errors="replace")
    energies: dict[str, float] = {}
    m_tot = re.search(r"TOTAL ENERGY\s+([-+]?\d+\.\d+)\s+Eh", text)
    if m_tot:
        energies["total"] = (
            float(m_tot.group(1)) * elements.CODATA2022.hartree_to_ev
        )
    m_gap = re.search(r"HOMO-LUMO GAP\s+([-+]?\d+\.\d+)\s+eV", text)
    meta: dict[str, object] = {"source": str(p)}
    if m_gap:
        meta["homo_lumo_gap_ev"] = float(m_gap.group(1))
    if not energies and "homo_lumo_gap_ev" not in meta:
        raise exceptions.ParseError(f"Could not extract xTB summary from {p}.")
    return structure.AtomsData(
        symbols=[],
        positions=np.zeros((0, 3), dtype=np.float64),
        energies=energies,
        metadata=meta,
    )


registry.register_format(
    registry.FormatSpec(
        name="cp2k_ener",
        extensions=(".ener",),
        filenames=(),
        reader=read_cp2k_ener,
        description="CP2K .ener molecular dynamics energy trace",
    )
)
registry.register_format(
    registry.FormatSpec(
        name="xtb_out",
        extensions=(),
        filenames=("xtb.out",),
        reader=read_xtb_out,
        description="xTB calculation output energy and gap summary",
    )
)
