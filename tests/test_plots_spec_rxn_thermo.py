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
"""Tests for spectroscopy, reaction/NEB, MD, thermo, and chart plots."""

from __future__ import annotations

import pathlib

import matplotlib
from matplotlib import pyplot as plt
import numpy as np

from xkcd_matchem.io import loader
from xkcd_matchem.plots import charts
from xkcd_matchem.plots import reactions
from xkcd_matchem.plots import spectroscopy as spec_plots
from xkcd_matchem.plots import thermo
from xkcd_matchem.plots import trajectory as traj_plots


def test_spectroscopy_and_reaction_plots() -> None:
    matplotlib.use("Agg")
    ax_xrd = spec_plots.plot_xrd("si_diamond.cif", two_theta_range=(20.0, 70.0))
    assert len(ax_xrd.lines) >= 1
    plt.close(ax_xrd.figure)

    h2o = loader.load("h2o_gaussian.log")
    ax_ir = spec_plots.plot_spectrum(h2o.metadata["ir_spectrum"])
    assert ax_ir.xaxis_inverted()
    plt.close(ax_ir.figure)

    ax_rxn = reactions.plot_reaction_profile(
        [0.0, 0.85, -0.45], ["Reactants", "TS", "Products"]
    )
    assert len(ax_rxn.lines) >= 3
    plt.close(ax_rxn.figure)

    neb_traj = loader.load_trajectory("sn2_neb.xyz")
    ax_neb = reactions.plot_neb(neb_traj)
    assert len(ax_neb.lines) >= 1
    plt.close(ax_neb.figure)


def test_trajectory_thermo_and_chart_plots(tmp_path: pathlib.Path) -> None:
    matplotlib.use("Agg")
    md = loader.load_trajectory("cu_md.lammpstrj")
    ax_ts = traj_plots.plot_md_timeseries(md, quantity="rmsd", running_window=3)
    assert len(ax_ts.lines) == 2
    plt.close(ax_ts.figure)

    r_vals, g_vals = md.compute_rdf(("Cu", "Cu"), r_max=4.0, n_bins=30)
    ax_rdf = traj_plots.plot_rdf(r_vals, g_vals)
    assert len(ax_rdf.lines) >= 1
    plt.close(ax_rdf.figure)

    gif_path = tmp_path / "cu_traj.gif"
    traj_plots.animate_trajectory(md, gif_path, fps=4, stride=2)
    assert gif_path.is_file()
    assert gif_path.stat().st_size > 100

    ax_hull = thermo.plot_phase_diagram_2d(
        [0.0, 0.25, 0.5, 0.75, 1.0],
        [0.0, -0.35, -0.40, 0.05, 0.0],
        ["A", "A3B", "AB", "AB3", "B"],
    )
    assert len(ax_hull.lines) >= 1
    plt.close(ax_hull.figure)

    vols = np.linspace(36.0, 44.0, 7)
    e_eos = -10.5 + 0.06 * (vols - 40.0) ** 2
    ax_eos = thermo.plot_eos(vols, e_eos)
    assert len(ax_eos.lines) >= 1
    plt.close(ax_eos.figure)

    ax_pb = thermo.plot_pourbaix(
        [
            ("M2+", [(0, -0.5), (5, -0.5), (5, 1.2), (0, 1.2)]),
            ("MO", [(5, -0.5), (14, -0.5), (14, 1.2), (5, 1.2)]),
        ]
    )
    assert len(ax_pb.patches) >= 2
    plt.close(ax_pb.figure)

    rng = np.random.default_rng(7)
    y_true = np.linspace(-2.0, 2.0, 20)
    y_pred = y_true + rng.normal(0.0, 0.08, size=20)
    ax_par = charts.plot_parity(y_true, y_pred)
    assert len(ax_par.texts) >= 1
    plt.close(ax_par.figure)

    ax_bar = charts.plot_bar_comparison(["PBE", "HSE06", "GW"], [0.6, 1.1, 1.2])
    assert len(ax_bar.patches) == 3
    plt.close(ax_bar.figure)

    ax_pipe = charts.plot_pipeline_diagram(
        [
            ("Parse", "CIF / POSCAR"),
            ("Render", "xkcd 3D"),
            ("Export", "PDF / SVG"),
        ]
    )
    assert len(ax_pipe.patches) == 3
    plt.close(ax_pipe.figure)
