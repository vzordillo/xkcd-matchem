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
import pytest

from xkcd_matchem.core import spectroscopy as spec_core
from xkcd_matchem.core import structure
from xkcd_matchem.core import trajectory as traj_core
from xkcd_matchem.data import elements
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


def test_spectroscopy_reaction_thermo_md_accuracy() -> None:
    matplotlib.use("Agg")
    x_sp = np.linspace(0.0, 10.0, 101)
    y_sp = 100.0 * np.exp(-0.5 * ((x_sp - 5.0) / 0.35) ** 2)
    spec = spec_core.Spectrum(
        x=x_sp,
        y=y_sp,
        kind="generic",
        peaks=[spec_core.Peak(position=5.0, intensity=100.0, label="P1")],
    )
    ax_sp = spec_plots.plot_spectrum(spec)
    stick_cols = [c for c in ax_sp.collections if hasattr(c, "get_segments")]
    assert len(stick_cols) == 1
    stick_segs = stick_cols[0].get_segments()
    assert float(stick_segs[0][1][1]) == pytest.approx(100.0)
    plt.close(ax_sp.figure)

    ax_rxn = reactions.plot_reaction_profile(
        [0.0, 0.42, -0.35, 0.78, -0.85], ["R", "TS1", "Int", "TS2", "P"]
    )
    assert any("Ea = 1.13 eV" in t.get_text() for t in ax_rxn.texts)
    plt.close(ax_rxn.figure)

    offsets = [0.0, 0.5, 1.0, 3.0, 4.0]
    energies = [0.0, 0.2, 0.8, 0.3, -0.1]
    frames = [
        structure.AtomsData(
            symbols=["H"],
            positions=np.array([[dx, 0.0, 0.0]]),
            energies={"total": en},
        )
        for dx, en in zip(offsets, energies, strict=True)
    ]
    neb_traj = traj_core.Trajectory(frames=frames)
    ax_neb = reactions.plot_neb(neb_traj)
    pts_x = ax_neb.collections[0].get_offsets()[:, 0]
    assert np.allclose(pts_x, [0.0, 0.125, 0.25, 0.75, 1.0])
    y_curve = ax_neb.lines[0].get_ydata()
    assert float(np.max(y_curve)) <= 0.8 + 1e-9
    assert float(np.min(y_curve)) >= -0.1 - 1e-9
    plt.close(ax_neb.figure)

    ax_neg_neb = reactions.plot_neb([0.0, -0.2, -0.1, -0.4])
    assert all("+-" not in t.get_text() for t in ax_neg_neb.texts)
    plt.close(ax_neg_neb.figure)

    for xs_h, ys_h in (
        ([0.0, 1.0], [0.0, 0.0]),
        ([0.0, 0.5, 1.0], [0.0, 0.0, 0.0]),
        ([0.0, 0.5, 0.5, 1.0], [0.0, -0.5, -0.1, 0.0]),
    ):
        ax_h = thermo.plot_phase_diagram_2d(xs_h, ys_h)
        assert len(ax_h.lines) >= 1
        plt.close(ax_h.figure)

    ev_per_a3_to_gpa = elements.CODATA2022.ev_to_joule * 1e21
    v0_t, b0_gpa_t, bp_t = 45.0, 440.0, 3.8
    b0_t = b0_gpa_t / ev_per_a3_to_gpa
    vols = np.linspace(38.0, 52.0, 9)
    eta = (v0_t / vols) ** (2.0 / 3.0)
    f_bm = eta - 1.0
    es_bm = -80.0 + (9.0 * v0_t * b0_t / 16.0) * (
        (f_bm**3) * bp_t + (f_bm**2) * (6.0 - 4.0 * eta)
    )
    ax_bm = thermo.plot_eos(vols, es_bm)
    box_txt = ax_bm.texts[0].get_text()
    assert "V0 = 45.00 A^3" in box_txt
    assert "B0 = 440.0 GPa" in box_txt
    assert "B0' = 3.80" in box_txt
    plt.close(ax_bm.figure)

    poly_pts = [
        (0.0, 0.0),
        (10.0, 0.0),
        (10.0, 0.2),
        (10.0, 0.4),
        (10.0, 0.6),
        (10.0, 0.8),
        (10.0, 1.0),
        (0.0, 1.0),
    ]
    ax_pb = thermo.plot_pourbaix([("Rect", poly_pts)], show_water_window=False)
    assert ax_pb.texts[0].get_position()[0] == pytest.approx(5.0)
    plt.close(ax_pb.figure)

    ts_frames = [
        structure.AtomsData(
            symbols=["H"], positions=np.array([[k * 1.0, 0.0, 0.0]])
        )
        for k in range(5)
    ]
    ts_traj = traj_core.Trajectory(
        frames=ts_frames, times=np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    )
    ax_ts = traj_plots.plot_md_timeseries(ts_traj, running_window=3)
    assert ax_ts.lines[1].get_xdata()[0] == pytest.approx(1.0)
    plt.close(ax_ts.figure)

    ax_rdf = traj_plots.plot_rdf(
        np.array([1.0, 2.0, 3.0]), np.array([0.0, 2.5, 1.0])
    )
    assert ax_rdf.get_ylim()[0] == pytest.approx(0.0)
    plt.close(ax_rdf.figure)
