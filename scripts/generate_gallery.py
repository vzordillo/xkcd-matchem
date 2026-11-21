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
"""Generates gallery showcase figures for README and Jupyter Book."""

from __future__ import annotations

import pathlib

import matplotlib
from matplotlib import pyplot as plt
import numpy as np

import xkcd_matchem as xm


_OUT_DIR = (
    pathlib.Path(__file__).resolve().parent.parent
    / "book"
    / "_static"
    / "gallery"
)


def _make_hero(out_dir: pathlib.Path) -> None:
    """Generates the 6-panel hero showcase poster."""
    board = xm.Board(
        300.0,
        195.0,
        dpi=180,
        title=(
            "xkcd-matchem: Hand-Drawn Visualization for Molecules & Materials"
        ),
        subtitle=(
            "Zero-dependency QM/MD/Crystal I/O  |  Depth-sorted 3D & 2D  |  "
            "Electronic, XRD, NEB, MD & Thermo"
        ),
    )
    c1 = board.add_card(
        8.0,
        26.0,
        90.0,
        76.0,
        title="Aspirin (3D)",
        inner_pad_mm=(4.0, 4.0, 11.5, 4.0),
    )
    c2 = board.add_card(
        105.0,
        26.0,
        90.0,
        76.0,
        title="TiO2 (110) Polyhedra",
        inner_pad_mm=(4.0, 4.0, 11.5, 4.0),
    )
    c3 = board.add_card(202.0, 26.0, 90.0, 76.0, title="NaCl Powder XRD")
    c4 = board.add_card(8.0, 110.0, 90.0, 76.0, title="Si Bands (QE)")
    c5 = board.add_card(105.0, 110.0, 90.0, 76.0, title="SN2 NEB Barrier")
    c6 = board.add_card(202.0, 110.0, 90.0, 76.0, title="Binary Convex Hull")

    xm.plot_molecule("aspirin.xyz", ax=c1.axes, show_legend=False)
    xm.plot_coordination(
        "tio2_slab.poscar",
        "Ti",
        "O",
        supercell=(3, 2, 1),
        ax=c2.axes,
    )
    xm.plot_xrd("nacl.cif", ax=c3.axes, label_peaks=4)
    xm.plot_bands(xm.load("si_qe_bands.dat"), ax=c4.axes, ylim=(-6.0, 8.0))
    xm.plot_neb(xm.load_trajectory("sn2_neb.xyz"), ax=c5.axes)
    xm.plot_phase_diagram_2d(
        [0.0, 0.25, 0.5, 0.75, 1.0],
        [0.0, -0.34, -0.45, -0.08, 0.0],
        labels=["A", "A3B", "AB", "AB3", "B"],
        ax=c6.axes,
    )
    board.save(out_dir / "hero_showcase.png")
    plt.close(board.figure)


def _make_molecules(out_dir: pathlib.Path) -> None:
    """Generates 3D ball-and-stick, spacefill, and 2D skeletal comparison."""
    board = xm.Board(270.0, 115.0, dpi=170, title="Molecular Renderers")
    c1 = board.add_card(
        8.0,
        24.0,
        80.0,
        82.0,
        title="Aspirin 3D",
        inner_pad_mm=(4.0, 4.0, 11.5, 4.0),
    )
    c2 = board.add_card(
        95.0,
        24.0,
        80.0,
        82.0,
        title="Aspirin Spacefill",
        inner_pad_mm=(4.0, 4.0, 11.5, 4.0),
    )
    c3 = board.add_card(
        182.0,
        24.0,
        80.0,
        82.0,
        title="Aspirin 2D Skeletal",
        inner_pad_mm=(4.0, 4.0, 11.5, 4.0),
    )
    aspirin = xm.load("aspirin.xyz")
    xm.plot_molecule(
        aspirin, ax=c1.axes, style="ball_and_stick", show_legend=False
    )
    xm.plot_molecule(aspirin, ax=c2.axes, style="spacefill", show_legend=False)
    xm.plot_molecule(aspirin, ax=c3.axes, mode="2d")
    board.save(out_dir / "molecules_3d_2d.png")
    plt.close(board.figure)


def _make_crystals(out_dir: pathlib.Path) -> None:
    """Generates crystal supercell, Miller plane, polyhedra, and XRD panel."""
    board = xm.Board(
        275.0, 120.0, dpi=170, title="Crystals, Miller Planes & Polyhedra"
    )
    c1 = board.add_card(
        8.0,
        24.0,
        82.0,
        88.0,
        title="Si Diamond + (111)",
        inner_pad_mm=(4.0, 4.0, 11.5, 4.0),
    )
    c2 = board.add_card(
        96.0,
        24.0,
        82.0,
        88.0,
        title="2H-MoS2 2x2x1",
        inner_pad_mm=(4.0, 4.0, 11.5, 4.0),
    )
    c3 = board.add_card(184.0, 24.0, 83.0, 88.0, title="Si Cu Ka1 XRD")
    xm.plot_crystal("si_diamond.cif", miller_planes=[(1, 1, 1)], ax=c1.axes)
    xm.plot_crystal("mos2_2h.poscar", supercell=(2, 2, 1), ax=c2.axes)
    xm.plot_xrd("si_diamond.cif", two_theta_range=(20.0, 85.0), ax=c3.axes)
    board.save(out_dir / "crystal_polyhedra_xrd.png")
    plt.close(board.figure)


def _make_electronic(out_dir: pathlib.Path) -> None:
    """Generates synchronized bands + DOS and MO + CUBE gallery figures."""
    bands = xm.load("si_qe_bands.dat")
    dos = xm.load("si_qe_dos.dat")
    fig, _ = xm.plot_bands_dos(
        bands, dos, ylim=(-8.0, 10.0), title="Silicon Band Structure & DOS"
    )
    xm.save_figure(fig, out_dir / "bands_dos.png", dpi=170)
    plt.close(fig)

    board = xm.Board(220.0, 120.0, dpi=170, title="Orbitals & Volumetric Grid")
    c1 = board.add_card(8.0, 24.0, 96.0, 88.0, title="Benzene MOs (ORCA)")
    c2 = board.add_card(
        114.0, 24.0, 98.0, 88.0, title="Water HOMO Slice (CUBE)"
    )
    orca_atoms = xm.load("benzene_orca.out")
    xm.plot_mo_diagram(orca_atoms.metadata["orbitals"], ax=c1.axes)
    xm.plot_volumetric_slice(xm.load("water_homo.cube"), axis="z", ax=c2.axes)
    board.save(out_dir / "mo_and_cube.png")
    plt.close(board.figure)


def _make_spectroscopy_reactions(out_dir: pathlib.Path) -> None:
    """Generates IR/UV-Vis spectroscopy and reaction/NEB figures."""
    board = xm.Board(
        230.0, 118.0, dpi=170, title="Vibrational & UV-Vis Spectra"
    )
    c1 = board.add_card(8.0, 24.0, 103.0, 86.0, title="H2O IR (Gaussian)")
    c2 = board.add_card(119.0, 24.0, 103.0, 86.0, title="Benzene UV-Vis (ORCA)")
    h2o = xm.load("h2o_gaussian.log")
    benz = xm.load("benzene_orca.out")
    xm.plot_spectrum(h2o.metadata["ir_spectrum"], ax=c1.axes)
    xm.plot_spectrum(benz.metadata["uvvis_spectrum"], ax=c2.axes)
    board.save(out_dir / "spectroscopy.png")
    plt.close(board.figure)

    board_r = xm.Board(230.0, 118.0, dpi=170, title="Reaction Pathways & NEB")
    r1 = board_r.add_card(8.0, 24.0, 103.0, 86.0, title="Catalytic Profile")
    r2 = board_r.add_card(119.0, 24.0, 103.0, 86.0, title="SN2 NEB MEP")
    xm.plot_reaction_profile(
        [0.0, 0.42, -0.35, 0.78, -0.85],
        ["R", "TS1", "Int", "TS2", "P"],
        ax=r1.axes,
    )
    xm.plot_neb(xm.load_trajectory("sn2_neb.xyz"), ax=r2.axes)
    board_r.save(out_dir / "reaction_neb.png")
    plt.close(board_r.figure)


def _make_md_thermo_poster(out_dir: pathlib.Path) -> None:
    """Generates MD, thermo/ML parity, and workflow poster figures."""
    board_m = xm.Board(
        230.0, 118.0, dpi=170, title="MD Trajectories & Structure"
    )
    m1 = board_m.add_card(8.0, 24.0, 103.0, 86.0, title="Cu MD RMSD Trace")
    m2 = board_m.add_card(119.0, 24.0, 103.0, 86.0, title="Water O-O RDF (XVG)")
    md = xm.load_trajectory("cu_md.lammpstrj")
    xm.plot_md_timeseries(md, quantity="rmsd", running_window=3, ax=m1.axes)
    rdf = xm.load("water_rdf.xvg")
    xm.plot_rdf(rdf.x, rdf.y, unit="nm", ax=m2.axes)
    board_m.save(out_dir / "md_trajectory_rdf.png")
    plt.close(board_m.figure)

    board_t = xm.Board(
        275.0, 120.0, dpi=170, title="Thermodynamics & ML Benchmarking"
    )
    t1 = board_t.add_card(8.0, 24.0, 82.0, 88.0, title="Binary Convex Hull")
    t2 = board_t.add_card(96.0, 24.0, 82.0, 88.0, title="Birch-Murnaghan EOS")
    t3 = board_t.add_card(184.0, 24.0, 83.0, 88.0, title="ML Energy Parity")
    xm.plot_phase_diagram_2d(
        [0.0, 0.25, 0.5, 0.75, 1.0],
        [0.0, -0.34, -0.45, -0.05, 0.0],
        ["A", "A3B", "AB", "AB3", "B"],
        ax=t1.axes,
    )
    vols = np.linspace(36.0, 44.0, 8)
    energies = -12.4 + 0.065 * (vols - 40.0) ** 2
    xm.plot_eos(vols, energies, ax=t2.axes)
    rng = np.random.default_rng(12)
    yt = np.linspace(-3.0, 1.0, 24)
    yp = yt + rng.normal(0.0, 0.11, size=24)
    xm.plot_parity(yt, yp, ax=t3.axes)
    board_t.save(out_dir / "thermo_hull_eos.png")
    plt.close(board_t.figure)

    board_p = xm.Board(
        260.0,
        176.0,
        dpi=170,
        title="High-Throughput Screening Poster",
        subtitle="Generated with Board / Container / Card / Badge in mm units",
        preset="poster",
    )
    board_p.add_container(6.0, 25.0, 248.0, 144.0, title="Workflow & Results")
    p1 = board_p.add_card(
        10.0,
        36.5,
        118.0,
        59.0,
        title="Pipeline Architecture",
        inner_pad_mm=(4.5, 4.5, 11.5, 4.5),
    )
    p2 = board_p.add_card(
        134.0,
        36.5,
        116.0,
        59.0,
        title="Functional Benchmark",
        inner_pad_mm=(19.5, 6.0, 13.5, 14.0),
    )
    p3 = board_p.add_card(
        10.0,
        102.0,
        118.0,
        61.0,
        title="TiO2 (110) Slab",
        inner_pad_mm=(4.5, 4.5, 11.5, 4.5),
    )
    p4 = board_p.add_card(
        134.0,
        102.0,
        116.0,
        61.0,
        title="Pourbaix Regions",
        inner_pad_mm=(19.5, 6.0, 13.5, 15.0),
    )
    xm.plot_pipeline_diagram(
        [
            ("Parse", "CIF / POSCAR"),
            ("Relax", "PBE / HSE"),
            ("Plot", "xkcd-matchem"),
        ],
        ax=p1.axes,
        preset="poster",
    )
    xm.plot_bar_comparison(
        ["PBE", "r2SCAN", "HSE06"],
        [0.62, 1.04, 1.15],
        ylabel="Gap (eV)",
        ax=p2.axes,
        preset="poster",
    )
    xm.plot_slab(
        "tio2_slab.poscar",
        supercell=(3, 2, 1),
        ax=p3.axes,
        preset="poster",
    )
    xm.plot_pourbaix(
        [
            ("M2+", [(0, -0.5), (4.5, -0.5), (4.5, 1.5), (0, 1.5)]),
            ("MO2", [(4.5, -0.5), (14, -0.5), (14, 1.5), (4.5, 1.5)]),
        ],
        ax=p4.axes,
        preset="poster",
    )
    board_p.add_badge(234.0, 15.0, "Poster Preset")
    board_p.save(out_dir / "poster_board.png")
    plt.close(board_p.figure)


def main() -> None:
    """Generates all gallery figures."""
    matplotlib.use("Agg")
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    _make_hero(_OUT_DIR)
    _make_molecules(_OUT_DIR)
    _make_crystals(_OUT_DIR)
    _make_electronic(_OUT_DIR)
    _make_spectroscopy_reactions(_OUT_DIR)
    _make_md_thermo_poster(_OUT_DIR)
    print(f"Generated gallery images in {_OUT_DIR}")


if __name__ == "__main__":
    main()
