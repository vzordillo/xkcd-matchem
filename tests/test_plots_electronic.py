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
import numpy as np
import pytest

from xkcd_matchem.core import electronic
from xkcd_matchem.core import structure
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
    mo_labels = [t.get_text() for t in ax_mo.texts]
    assert any(lbl.startswith("HOMO") for lbl in mo_labels)
    assert any(lbl.startswith("LUMO") for lbl in mo_labels)
    plt.close(ax_mo.figure)

    cube = loader.load("water_homo.cube")
    ax_cube = elec_plots.plot_volumetric_slice(cube, axis="z")
    assert ax_cube.collections
    plt.close(ax_cube.figure)


def test_electronic_accuracy_edge_cases() -> None:
    matplotlib.use("Agg")
    discont = electronic.BandStructure(
        k_distances=np.array([0.0, 1.0, 1.0, 2.0]),
        eigenvalues=np.array(
            [[[-2.0, 2.0], [-1.0, 3.0], [-4.0, 1.0], [-3.0, 2.0]]]
        ),
        fermi_energy=0.0,
    )
    ax_b = elec_plots.plot_bands(discont)
    assert any(np.any(np.isnan(ln.get_ydata())) for ln in ax_b.lines[:2])
    plt.close(ax_b.figure)

    energies = np.linspace(-10.0, 10.0, 41)
    tot = np.where(np.abs(energies) <= 2.0, 2.0, 50.0)
    dn = np.where(np.abs(energies) <= 2.0, 1.5, 40.0)
    dos_nonspin = electronic.DensityOfStates(energies=energies, total_dos=tot)
    ax_ns = elec_plots.plot_dos(dos_nonspin, xlim=(-2.0, 2.0))
    assert ax_ns.get_ylim()[0] == pytest.approx(0.0)
    assert ax_ns.get_ylim()[1] < 10.0
    plt.close(ax_ns.figure)

    dos_spin = electronic.DensityOfStates(
        energies=energies, total_dos=tot, spin_down_dos=dn
    )
    ax_sp = elec_plots.plot_dos(
        dos_spin, orientation="vertical", xlim=(-2.0, 2.0)
    )
    legend_texts = [t.get_text() for t in ax_sp.get_legend().get_texts()]
    assert "Spin Up" in legend_texts
    assert "Spin Down" in legend_texts
    assert len(ax_sp.collections) + len(ax_sp.patches) >= 2
    plt.close(ax_sp.figure)

    pos_data = np.linspace(0.0, 2.0, 27).reshape((3, 3, 3))
    mol = structure.AtomsData(
        symbols=["O", "H"],
        positions=np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 1.9]]),
    )
    rho_grid = electronic.VolumetricGrid(
        origin=np.zeros(3),
        voxel_vectors=np.diag([0.5, 0.5, 0.5]),
        data=pos_data,
        atoms=mol,
    )
    ax_rho = elec_plots.plot_volumetric_slice(rho_grid, axis="z", slice_index=1)
    assert ax_rho.collections[0].norm.vmin == pytest.approx(0.0)
    plt.close(ax_rho.figure)
