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
"""Tests for band structure, DOS, MO diagram, and volumetric slice plots."""

from __future__ import annotations

import matplotlib
from matplotlib import pyplot as plt

from xkcd_matchem.io import loader
from xkcd_matchem.plots import electronic as elec_plots


def test_electronic_plots_end_to_end() -> None:
    matplotlib.use("Agg")
    bands = loader.load("si_qe_bands.dat")
    dos = loader.load("si_qe_dos.dat")
    fig, axes_pair = elec_plots.plot_bands_dos(
        bands, dos, ylim=(-10.0, 10.0), title="Si Bands + DOS"
    )
    assert len(axes_pair) == 2
    plt.close(fig)

    orca_atoms = loader.load("benzene_orca.out")
    ax_mo = elec_plots.plot_mo_diagram(orca_atoms.metadata["orbitals"])
    assert len(ax_mo.lines) > 0
    plt.close(ax_mo.figure)

    cube = loader.load("water_homo.cube")
    ax_cube = elec_plots.plot_volumetric_slice(cube, axis="z")
    assert ax_cube.collections
    plt.close(ax_cube.figure)
