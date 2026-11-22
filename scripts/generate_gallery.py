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
"""Generates borderless gallery figures, fonts, and interactive 3D studio."""

from __future__ import annotations

import io
import pathlib
import shutil

import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

import xkcd_matchem as xm


_BOOK_STATIC = (
    pathlib.Path(__file__).resolve().parent.parent / "book" / "_static"
)
_OUT_DIR = _BOOK_STATIC / "gallery"


def _make_hero(out_dir: pathlib.Path) -> None:
    """Generates a clean 2x3 open multi-panel hero showcase on white paper."""
    with xm.xkcd_style("classic"):
        fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.8), dpi=180)
        fig.patch.set_facecolor(xm.style.PAPER)

        xm.plot_molecule(
            "aspirin.xyz",
            ax=axes[0, 0],
            camera=xm.Camera(azimuth=-112.0, elevation=22.0, perspective=0.35),
            title="Aspirin (3D)",
        )
        xm.plot_coordination(
            "tio2_slab.poscar",
            "Ti",
            "O",
            supercell=(2, 2, 1),
            ax=axes[0, 1],
            title="Rutile TiO2 (110) Polyhedra",
        )
        xm.plot_xrd(
            "nacl.cif",
            ax=axes[0, 2],
            label_peaks=4,
            title="NaCl Powder XRD (Cu Ka1)",
        )
        xm.plot_bands(
            xm.load("si_qe_bands.dat"),
            ax=axes[1, 0],
            ylim=(-6.0, 8.0),
            title="Si Band Structure (QE)",
        )
        xm.plot_neb(
            xm.load_trajectory("sn2_neb.xyz"),
            ax=axes[1, 1],
            title="SN2 NEB Barrier",
        )
        xm.plot_phase_diagram_2d(
            [0.0, 0.25, 0.5, 0.75, 1.0],
            [0.0, -0.34, -0.45, -0.08, 0.0],
            labels=["A", "A3B", "AB", "AB3", "B"],
            ax=axes[1, 2],
            title="Binary Convex Hull",
        )
        fig.suptitle(
            "xkcd-matchem: Playful Exploration for Molecules & Materials",
            fontsize=17,
            y=0.98,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94), w_pad=2.2, h_pad=2.4)
        xm.save_figure(fig, out_dir / "hero_showcase.png", dpi=180)
        plt.close(fig)


def _make_molecules(out_dir: pathlib.Path) -> None:
    """Generates 3D ball-and-stick, spacefill, and 2D skeletal comparison."""
    aspirin = xm.load("aspirin.xyz")
    with xm.xkcd_style("classic"):
        fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.2), dpi=170)
        fig.patch.set_facecolor(xm.style.PAPER)
        xm.plot_molecule(
            aspirin,
            ax=axes[0],
            style="ball_and_stick",
            title="Aspirin 3D (Ball & Stick)",
        )
        xm.plot_molecule(
            aspirin,
            ax=axes[1],
            style="spacefill",
            title="Aspirin 3D (Spacefill)",
        )
        xm.plot_molecule(
            aspirin,
            ax=axes[2],
            mode="2d",
            title="Aspirin 2D Skeletal",
        )
        fig.suptitle(
            "Molecular Renderers: Default xkcd Presets",
            fontsize=15,
            y=0.98,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92), w_pad=2.0)
        xm.save_figure(fig, out_dir / "molecules_3d_2d.png", dpi=170)
        plt.close(fig)


def _make_crystals(out_dir: pathlib.Path) -> None:
    """Generates crystal supercell, slab view, Miller plane, and XRD panel."""
    with xm.xkcd_style("classic"):
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4), dpi=170)
        fig.patch.set_facecolor(xm.style.PAPER)
        xm.plot_crystal(
            "si_diamond.cif",
            miller_planes=[(1, 1, 1)],
            crystal_axes="corner",
            ax=axes[0],
            title="Si Diamond + (111) Plane",
        )
        xm.plot_slab(
            "tio2_slab.poscar",
            supercell=(3, 2, 1),
            view="side",
            ax=axes[1],
            title="TiO2 (110) Slab Profile",
        )
        xm.plot_xrd(
            "si_diamond.cif",
            two_theta_range=(20.0, 85.0),
            ax=axes[2],
            title="Si Cu Ka1 Powder XRD",
        )
        fig.suptitle(
            "Crystals, Slabs, Miller Planes & Powder Diffraction",
            fontsize=15,
            y=0.98,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92), w_pad=2.0)
        xm.save_figure(fig, out_dir / "crystal_polyhedra_xrd.png", dpi=170)
        plt.close(fig)


def _make_electronic(out_dir: pathlib.Path) -> None:
    """Generates synchronized bands + DOS and MO + CUBE gallery figures."""
    bands = xm.load("si_qe_bands.dat")
    dos = xm.load("si_qe_dos.dat")
    fig, _ = xm.plot_bands_dos(
        bands, dos, ylim=(-8.0, 10.0), title="Silicon Band Structure & DOS"
    )
    xm.save_figure(fig, out_dir / "bands_dos.png", dpi=170)
    plt.close(fig)

    with xm.xkcd_style("classic"):
        fig_mo, axes = plt.subplots(1, 2, figsize=(12.2, 5.4), dpi=170)
        fig_mo.patch.set_facecolor(xm.style.PAPER)
        orca_atoms = xm.load("benzene_orca.out")
        xm.plot_mo_diagram(
            orca_atoms.metadata["orbitals"],
            ax=axes[0],
            title="Benzene Molecular Orbitals (ORCA)",
        )
        xm.plot_volumetric_slice(
            xm.load("water_homo.cube"),
            axis="z",
            ax=axes[1],
            title="Water HOMO Electron Density Slice",
        )
        fig_mo.suptitle(
            "Molecular Orbitals & Real-Space Volumetric Cubes",
            fontsize=15,
            y=0.98,
        )
        fig_mo.tight_layout(rect=(0.0, 0.0, 1.0, 0.92), w_pad=2.4)
        xm.save_figure(fig_mo, out_dir / "mo_and_cube.png", dpi=170)
        plt.close(fig_mo)


def _make_spectroscopy_reactions(out_dir: pathlib.Path) -> None:
    """Generates IR/UV-Vis spectroscopy and reaction/NEB figures."""
    h2o = xm.load("h2o_gaussian.log")
    benz = xm.load("benzene_orca.out")
    with xm.xkcd_style("classic"):
        fig_s, axes = plt.subplots(1, 2, figsize=(12.4, 5.2), dpi=170)
        fig_s.patch.set_facecolor(xm.style.PAPER)
        xm.plot_spectrum(
            h2o.metadata["ir_spectrum"],
            ax=axes[0],
            title="H2O IR Spectrum (Gaussian)",
        )
        xm.plot_spectrum(
            benz.metadata["uvvis_spectrum"],
            ax=axes[1],
            title="Benzene UV-Vis Spectrum (ORCA)",
        )
        fig_s.suptitle(
            "Vibrational & Optical Spectra (Exploratory xkcd Sketches)",
            fontsize=15,
            y=0.98,
        )
        fig_s.tight_layout(rect=(0.0, 0.0, 1.0, 0.92), w_pad=2.2)
        xm.save_figure(fig_s, out_dir / "spectroscopy.png", dpi=170)
        plt.close(fig_s)

    with xm.xkcd_style("classic"):
        fig_r, axes_r = plt.subplots(1, 2, figsize=(12.4, 5.2), dpi=170)
        fig_r.patch.set_facecolor(xm.style.PAPER)
        xm.plot_reaction_profile(
            [0.0, 0.42, -0.35, 0.78, -0.85],
            ["R", "TS1", "Int", "TS2", "P"],
            ax=axes_r[0],
            title="Two-Step Catalytic Free Energy Profile",
        )
        xm.plot_neb(
            xm.load_trajectory("sn2_neb.xyz"),
            ax=axes_r[1],
            title="SN2 Minimum Energy Path (NEB)",
        )
        fig_r.suptitle(
            "Reaction Pathways & Climbing-Image NEB",
            fontsize=15,
            y=0.98,
        )
        fig_r.tight_layout(rect=(0.0, 0.0, 1.0, 0.92), w_pad=2.2)
        xm.save_figure(fig_r, out_dir / "reaction_neb.png", dpi=170)
        plt.close(fig_r)


def _make_md_thermo(out_dir: pathlib.Path) -> None:
    """Generates borderless MD trajectory and thermodynamics/parity panels."""
    md = xm.load_trajectory("cu_md.lammpstrj")
    rdf = xm.load("water_rdf.xvg")
    with xm.xkcd_style("classic"):
        fig_m, axes_m = plt.subplots(1, 2, figsize=(12.4, 5.2), dpi=170)
        fig_m.patch.set_facecolor(xm.style.PAPER)
        xm.plot_md_timeseries(
            md,
            quantity="rmsd",
            running_window=3,
            ax=axes_m[0],
            title="Cu Nanoparticle MD RMSD Trace",
        )
        xm.plot_rdf(
            rdf.x,
            rdf.y,
            unit="nm",
            ax=axes_m[1],
            title="Liquid Water O-O RDF (GROMACS)",
        )
        fig_m.suptitle(
            "Molecular Dynamics RMSD & Radial Distribution",
            fontsize=15,
            y=0.98,
        )
        fig_m.tight_layout(rect=(0.0, 0.0, 1.0, 0.92), w_pad=2.2)
        xm.save_figure(fig_m, out_dir / "md_trajectory_rdf.png", dpi=170)
        plt.close(fig_m)

    with xm.xkcd_style("classic"):
        fig_t, axes_t = plt.subplots(1, 3, figsize=(15.2, 5.2), dpi=170)
        fig_t.patch.set_facecolor(xm.style.PAPER)
        xm.plot_phase_diagram_2d(
            [0.0, 0.25, 0.5, 0.75, 1.0],
            [0.0, -0.34, -0.45, -0.05, 0.0],
            ["A", "A3B", "AB", "AB3", "B"],
            ax=axes_t[0],
            title="Binary Convex Hull",
        )
        vols = np.linspace(35.5, 45.0, 8)
        ev_per_a3_to_gpa = 160.2176634
        v0_si = 40.00
        b0_si = 98.5 / ev_per_a3_to_gpa
        bp_si = 4.15
        eta = (v0_si / vols) ** (2.0 / 3.0)
        f_bm = eta - 1.0
        energies = -10.84 + (9.0 * v0_si * b0_si / 16.0) * (
            (f_bm**3) * bp_si + (f_bm**2) * (6.0 - 4.0 * eta)
        )
        xm.plot_eos(
            vols,
            energies,
            ax=axes_t[1],
            title="Birch-Murnaghan EOS Fit",
        )
        rng = np.random.default_rng(12)
        yt = np.linspace(-3.0, 1.0, 24)
        yp = yt + rng.normal(0.0, 0.11, size=24)
        xm.plot_parity(
            yt,
            yp,
            ax=axes_t[2],
            title="MLIP Energy Parity",
        )
        fig_t.suptitle(
            "Thermodynamics, Equations of State & ML Benchmarking",
            fontsize=15,
            y=0.98,
        )
        fig_t.tight_layout(rect=(0.0, 0.0, 1.0, 0.92), w_pad=2.0)
        xm.save_figure(fig_t, out_dir / "thermo_hull_eos.png", dpi=170)
        plt.close(fig_t)


def _make_opt_in_poster(out_dir: pathlib.Path) -> None:
    """Generates opt-in Board with stacked Cards grouped inside Containers."""
    board_p = xm.Board(
        264.0,
        180.0,
        dpi=170,
        title="Opt-In Board Layout: Stacking Cards Inside Container Boxes",
        subtitle=(
            "Group and stack Cards vertically or side by side "
            "inside Containers (mm units)"
        ),
        preset="poster",
    )
    board_p.add_container(
        6.0,
        25.0,
        123.0,
        148.0,
        title="1. Workflow & Structure (Stacked Cards)",
        solid_header=True,
    )
    board_p.add_container(
        135.0,
        25.0,
        123.0,
        148.0,
        title="2. Benchmarks & Stability (Stacked Cards)",
        solid_header=True,
        header_color="#2A9D8F",
    )
    p1 = board_p.add_card(
        10.5,
        36.5,
        114.0,
        61.0,
        title="Pipeline Architecture",
        header_band=True,
        inner_pad_mm=(4.5, 4.5, 11.5, 4.5),
    )
    p3 = board_p.add_card(
        10.5,
        103.5,
        114.0,
        64.5,
        title="TiO2 (110) Slab",
        header_band=True,
        inner_pad_mm=(4.5, 4.5, 11.5, 4.5),
    )
    p2 = board_p.add_card(
        139.5,
        36.5,
        114.0,
        61.0,
        title="Functional Benchmark",
        header_band=True,
        header_color="#2A9D8F",
        inner_pad_mm=(19.5, 6.0, 13.5, 15.0),
    )
    p4 = board_p.add_card(
        139.5,
        103.5,
        114.0,
        64.5,
        title="Pourbaix Stability",
        header_band=True,
        header_color="#2A9D8F",
        inner_pad_mm=(19.5, 6.0, 13.5, 18.0),
    )
    xm.plot_pipeline_diagram(
        [
            ("Parse", "CIF / POSCAR"),
            ("Relax", "PBE / HSE"),
            ("Sketch", "xkcd-matchem"),
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
        view="side",
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
    board_p.add_badge(238.0, 15.0, "Opt-in Board")
    board_p.save(out_dir / "poster_board.png")
    plt.close(board_p.figure)


def _export_docs_assets() -> None:
    """Exports offline interactive studio HTML and xkcd Script font file."""
    fonts_dir = _BOOK_STATIC / "fonts"
    fonts_dir.mkdir(parents=True, exist_ok=True)
    ttf_src = xm.style.get_bundled_font_path("xkcd Script")
    shutil.copy2(ttf_src, fonts_dir / "xkcd-script.ttf")

    xm.export_interactive_html(
        "tio2_slab.poscar",
        _BOOK_STATIC / "interactive_studio.html",
        include_bundled_samples=True,
        style="polyhedra",
        polyhedra=("Ti", "O"),
        name="TiO2 (110) Polyhedra (tio2_slab.poscar)",
    )


def _fig_to_pil(fig: matplotlib.figure.Figure) -> Image.Image:
    """Converts a Matplotlib Figure into an adaptive-palette Pillow image."""
    buf = io.BytesIO()
    fig.patch.set_sketch_params(0.0, 0.0, 0.0)
    for ax in fig.axes:
        ax.patch.set_sketch_params(0.0, 0.0, 0.0)
    fig.savefig(
        buf,
        format="png",
        dpi=fig.dpi,
        facecolor=fig.get_facecolor(),
        edgecolor=fig.get_facecolor(),
    )
    buf.seek(0)
    with Image.open(buf) as raw:
        rgb = raw.convert("RGB")
        return rgb.quantize(colors=128, method=Image.Quantize.MEDIANCUT)


def _save_gif(
    frames: list[Image.Image],
    out_path: pathlib.Path,
    duration_ms: int,
) -> None:
    """Saves an optimized infinite-loop GIF from Pillow frames."""
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )


def _make_crystal_orbit_gif(out_dir: pathlib.Path) -> None:
    """Generates a 24-frame 360-degree orbit GIF of TiO2 polyhedra."""
    tio2 = xm.load("tio2_slab.poscar").center_in_cell().repeat((2, 2, 1))
    frames: list[Image.Image] = []
    for deg in np.linspace(0.0, 360.0, 24, endpoint=False):
        cam = xm.Camera(
            azimuth=-118.0 + float(deg),
            elevation=24.0,
            perspective=0.35,
        )
        with xm.xkcd_style("classic"):
            fig, ax = plt.subplots(figsize=(5.2, 4.4), dpi=115)
            fig.patch.set_facecolor(xm.style.PAPER)
            xm.plot_coordination(
                tio2,
                "Ti",
                "O",
                camera=cam,
                ax=ax,
                title="TiO2 (110) Polyhedra Orbit",
            )
            fig.tight_layout()
            frames.append(_fig_to_pil(fig))
            plt.close(fig)
    _save_gif(frames, out_dir / "crystal_orbit.gif", duration_ms=95)


def _make_neb_gif(out_dir: pathlib.Path) -> None:
    """Generates a 16-frame synchronized SN2 NEB structure + MEP GIF."""
    neb = xm.load_trajectory("sn2_neb.xyz")
    rel_e = neb.energy_profile(relative=True, unit="eV")
    all_pts = np.vstack([f.positions for f in neb.frames])
    pivot = np.mean(all_pts, axis=0)
    cam = xm.Camera(azimuth=-115.0, elevation=20.0, perspective=0.25)
    all_xy, _ = xm.core.geometry.project_points(all_pts, cam, center=pivot)
    x_min, x_max = float(np.min(all_xy[:, 0])), float(np.max(all_xy[:, 0]))
    y_min, y_max = float(np.min(all_xy[:, 1])), float(np.max(all_xy[:, 1]))
    pad = 1.65

    seq = list(range(len(neb))) + list(range(len(neb) - 2, 0, -1))
    frames: list[Image.Image] = []
    n_tot = len(neb) - 1
    for img_idx in seq:
        with xm.xkcd_style("classic"):
            fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(9.2, 4.0), dpi=115)
            fig.patch.set_facecolor(xm.style.PAPER)
            xm.render_structure_3d(
                neb[img_idx],
                ax=ax_l,
                camera=cam,
                show_cell=False,
            )
            half_x = max(0.5 * (x_max - x_min) + pad, 3.6)
            half_y = max(0.5 * (y_max - y_min) + pad, 3.2)
            ax_l.set_xlim(-half_x, half_x)
            ax_l.set_ylim(-half_y, half_y)
            ax_l.set_title(f"SN2 Image {img_idx + 1}/{len(neb)}")
            xm.plot_neb(
                neb, ax=ax_r, title="SN2 Minimum Energy Path (Cl- + CH3Cl)"
            )
            ax_r.scatter(
                [img_idx / max(n_tot, 1)],
                [float(rel_e[img_idx])],
                s=130,
                color="#D55E00",
                edgecolor=xm.style.INK,
                linewidth=1.6,
                zorder=10,
            )
            fig.tight_layout(w_pad=2.0)
            frames.append(_fig_to_pil(fig))
            plt.close(fig)
    _save_gif(frames, out_dir / "neb_reaction.gif", duration_ms=140)


def _make_theme_switcher_gif(out_dir: pathlib.Path) -> None:
    """Generates a 4-frame GIF cycling default and custom themes/colors."""
    mos2 = xm.load("mos2_2h.poscar").center_in_cell().repeat((2, 2, 1))
    bands = xm.load("si_qe_bands.dat")
    xm.register_preset(
        "warm_pastel_gif",
        base="classic",
        figure_facecolor=xm.style.WARM_PAPER,
        axes_facecolor=xm.style.WARM_PAPER,
    )
    configs: list[tuple[str, str | dict[str, str], str | list[str], str]] = [
        (
            "classic",
            "xkcd",
            "okabe_ito",
            "Default Preset: classic + xkcd",
        ),
        (
            "warm_pastel_gif",
            "pastel",
            "pastel",
            "Custom Warm Paper + pastel",
        ),
        (
            "classic",
            {"Mo": "#2A9D8F", "S": "#E9C46A"},
            ["#264653", "#E76F51", "#2A9D8F"],
            "Custom Inline Element Dict",
        ),
        (
            "dark_chalk",
            "xkcd",
            ["#56B4E9", "#F0E442", "#009E73"],
            "Built-In dark_chalk Theme",
        ),
    ]
    frames: list[Image.Image] = []
    for preset_name, elem_pal, band_pal, label in configs:
        preset_obj = xm.style.get_preset(preset_name)
        with xm.xkcd_style(preset_name):
            fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(9.2, 4.0), dpi=120)
            fig.patch.set_facecolor(preset_obj.figure_facecolor)
            xm.plot_crystal(
                mos2,
                palette=elem_pal,
                preset=preset_name,
                ax=ax_l,
                title="2H-MoS2 (2x2x1)",
            )
            xm.plot_bands(
                bands,
                palette=band_pal,
                preset=preset_name,
                ax=ax_r,
                ylim=(-6.0, 8.0),
                title="Si Bands (QE)",
            )
            fig.suptitle(
                label,
                fontsize=14,
                color=preset_obj.ink_color,
                y=0.98,
            )
            fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92), w_pad=2.0)
            frames.append(_fig_to_pil(fig))
            plt.close(fig)
    _save_gif(frames, out_dir / "theme_switcher.gif", duration_ms=1300)


def _make_readme_gifs(out_dir: pathlib.Path) -> None:
    """Generates lightweight animated GIFs showcasing 3D, NEB, and themes."""
    _make_crystal_orbit_gif(out_dir)
    _make_neb_gif(out_dir)
    _make_theme_switcher_gif(out_dir)


def _make_periodic_table_gallery(out_dir: pathlib.Path) -> None:
    """Generates a 2-row xkcd Periodic Table family + heatmap showcase."""
    with xm.xkcd_style("classic"):
        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=(15.0, 14.2), dpi=165
        )
        fig.patch.set_facecolor(xm.style.PAPER)
        xm.plot_periodic_table(
            ax=ax_top,
            title="Periodic Table of Elements (xkcd Chemical Families)",
        )
        xm.plot_periodic_table(
            element_property="electronegativity",
            cmap="YlOrRd",
            missing_shade="hatch",
            ax=ax_bot,
            title="Pauling Electronegativity Heatmap (Hatched Missing Values)",
        )
        fig.tight_layout(h_pad=1.6)
        xm.save_figure(fig, out_dir / "periodic_table.png", dpi=165)
        plt.close(fig)


def _make_materials_informatics_gallery(out_dir: pathlib.Path) -> None:
    """Generates ternary hull, spacegroup sunburst, and hatched histogram."""
    comps = [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.5, 0.0, 0.5),
        (0.0, 0.4, 0.6),
        (0.0, 0.5, 0.5),
        (0.25, 0.25, 0.50),
        (0.50, 0.17, 0.33),
        (0.36, 0.36, 0.28),
    ]
    energies = [0.0, 0.0, 0.0, -1.95, -1.65, -1.35, -1.88, -2.02, -1.42]
    labels = [
        "",
        "",
        "",
        "Li2O",
        "Fe2O3",
        "FeO",
        "LiFeO2",
        "Li3FeO3",
        "meta",
    ]
    sg_counts = {
        225: 32,
        227: 18,
        221: 24,
        194: 26,
        166: 14,
        139: 19,
        129: 12,
        62: 28,
        63: 15,
        14: 22,
        12: 11,
        2: 9,
    }
    rng = np.random.default_rng(42)
    gaps = np.concatenate(
        [rng.normal(1.35, 0.42, 160), rng.normal(3.15, 0.55, 95)]
    )
    gaps = gaps[gaps > 0.05]
    with xm.xkcd_style("classic"):
        fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.4), dpi=165)
        fig.patch.set_facecolor(xm.style.PAPER)
        xm.plot_ternary_phase_diagram(
            comps,
            energies,
            corners=("Li", "Fe", "O"),
            labels=labels,
            ax=axes[0],
            title="Ternary Li-Fe-O Convex Hull",
        )
        xm.plot_spacegroup_distribution(
            sg_counts,
            mode="sunburst",
            shade="cycle",
            ax=axes[1],
            title="Space Group Sunburst (1-230)",
        )
        xm.plot_histogram(
            gaps,
            bins=16,
            xlabel="DFT Band Gap (eV)",
            shade="hatch",
            background_grid="dotgrid",
            ax=axes[2],
            title="Hatched Band-Gap Distribution",
        )
        fig.suptitle(
            "Materials Informatics in xkcd Style (Inspired by pymatviz)",
            fontsize=15,
            y=0.98,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92), w_pad=2.0)
        xm.save_figure(fig, out_dir / "materials_informatics.png", dpi=165)
        plt.close(fig)


def _make_scrolls_and_grids_gallery(out_dir: pathlib.Path) -> None:
    """Generates portrait/landscape scroll cards, grid papers, and hatching."""
    board = xm.Board(
        268.0,
        190.0,
        dpi=165,
        title="Background Grids, Hatched Shading & Vintage Scroll Cards",
        subtitle=(
            "Graph/dot/blueprint backgrounds, hatched fills, "
            "plus portrait & landscape scroll templates"
        ),
        preset="classic",
        background_style="dotgrid",
    )
    c_tl = board.add_card(
        8.0,
        26.0,
        122.0,
        74.0,
        title="Hatched Bars on Graph Paper",
        inner_pad_mm=(16.5, 5.5, 13.0, 13.0),
    )
    c_tr = board.add_card(
        138.0,
        26.0,
        122.0,
        74.0,
        title="Blueprint Grid + Crosshatch Pourbaix",
        preset="blueprint",
        inner_pad_mm=(17.5, 5.5, 13.0, 14.5),
    )
    xm.plot_bar_comparison(
        ["PBE", "r2SCAN", "HSE06", "GW"],
        [0.62, 1.04, 1.15, 1.24],
        ylabel="Si Gap (eV)",
        shade="cycle",
        background_grid="graph",
        ax=c_tl.axes,
    )
    xm.plot_pourbaix(
        [
            ("Ti3+", [(0, -0.8), (3.8, -0.8), (3.8, 1.4), (0, 1.4)]),
            ("TiO2", [(3.8, -0.8), (14, -0.8), (14, 1.4), (3.8, 1.4)]),
        ],
        shade=("crosshatch", "dots"),
        background_grid="blueprint",
        preset="blueprint",
        ax=c_tr.axes,
    )

    board.add_scroll_card(
        8.0,
        104.0,
        106.0,
        80.0,
        orientation="portrait",
        title="Portrait Scroll: DFT Checklist",
        items=[
            ("1. Cell", "Check POSCAR & CIF symmetry"),
            ("2. Mesh", "Converge k-points & ENCUT"),
            ("3. Relax", "Forces < 0.01 eV/A"),
            ("4. Bands", "Align DOS & Fermi level"),
            ("5. SciComm", "Sketch with xkcd-matchem!"),
        ],
        bullet="checked",
        background_grid="ruled",
        align_to_grid=True,
    )
    sc_land = board.add_scroll_card(
        120.0,
        104.0,
        140.0,
        80.0,
        orientation="landscape",
        title="Landscape Scroll: Synthesis Roadmap",
        background_grid=None,
        inner_pad_mm=(4.0, 4.0, 13.5, 3.5),
    )
    xm.plot_pipeline_diagram(
        [
            ("Precursors", "TiO2 + Li2CO3"),
            ("Calcination", "850 C / 12 h"),
            ("XRD Check", "Cu Ka1"),
        ],
        ax=sc_land.axes,
        palette="pastel",
    )
    sc_land.axes.set_facecolor(xm.layout.scroll.PARCHMENT_LIGHT)
    board.save(out_dir / "scrolls_grids_shading.png")
    plt.close(board.figure)


def _make_icons_and_graphs_gallery(out_dir: pathlib.Path) -> None:
    """Generates MatChem icons, Crystal-to-Graph view, and synthesis graph."""
    with xm.xkcd_style("classic"):
        fig = plt.figure(figsize=(16.0, 9.2), dpi=165)
        fig.patch.set_facecolor(xm.style.PAPER)
        gs = fig.add_gridspec(
            2, 2, width_ratios=[1.25, 1.35], height_ratios=[1.05, 1.0]
        )
        ax_icons = fig.add_subplot(gs[:, 0])
        sub_top = gs[0, 1].subgridspec(1, 2, wspace=0.06)
        ax_crys = fig.add_subplot(sub_top[0, 0])
        ax_graph = fig.add_subplot(sub_top[0, 1])
        ax_net = fig.add_subplot(gs[1, 1])

        xm.plot_icon_sheet(
            elements_to_show=("H", "Li", "C", "O", "Si", "Ti", "Fe", "Au"),
            ncols=6,
            ax=ax_icons,
            title="xkcd MatChem Icons & Element Badges",
        )
        cam = xm.Camera(azimuth=25.0, elevation=14.0)
        xm.plot_crystal(
            "si_diamond.cif",
            camera=cam,
            boundary_atoms=False,
            crystal_axes="corner",
            show_labels="index",
            ax=ax_crys,
            title="1. Actual Crystal (8-Atom Unit Cell)",
        )
        xm.plot_structure_graph(
            "si_diamond.cif",
            layout="camera",
            camera=cam,
            show_cell=True,
            ax=ax_graph,
            title="2. Matching GNN Graph (Same View)",
        )
        xlim_c, ylim_c = ax_crys.get_xlim(), ax_crys.get_ylim()
        ax_crys.set_aspect("equal", adjustable="box")
        ax_crys.set_xlim(xlim_c)
        ax_crys.set_ylim(ylim_c)
        ax_graph.set_xlim(xlim_c)
        ax_graph.set_ylim(ylim_c)

        synth_nodes = [
            "Anatase\nTiO2",
            "Rutile\nTiO2",
            "Li2CO3",
            "Li4Ti5O12\n(Spinel)",
            "BaCO3",
            "BaTiO3\n(Perovskite)",
        ]
        synth_pos = [
            (-1.25, 0.08),
            (-0.15, 1.05),
            (1.25, 1.05),
            (1.25, 0.08),
            (-1.25, -0.95),
            (1.25, -0.95),
        ]
        synth_edges = [
            (0, 1, "650 C"),
            (0, 3, "800 C"),
            (2, 3, "+ Li"),
            (0, 5, "950 C"),
            (4, 5, "+ Ba"),
        ]
        xm.plot_graph_network(
            synth_nodes,
            synth_edges,
            pos=synth_pos,
            node_shape="pill",
            directed=True,
            palette="pastel",
            ax=ax_net,
            title="Materials Synthesis & Phase-Transformation Network",
        )
        fig.tight_layout(w_pad=1.8, h_pad=2.0)
        xm.save_figure(fig, out_dir / "icons_and_graphs.png", dpi=165)
        plt.close(fig)


def _make_crystal_graphs_2d_gallery(out_dir: pathlib.Path) -> None:
    """Generates bond-free crystal sites and planar flat 2D GNN graphs."""
    a_si = 5.4309
    prim_cell = 0.5 * np.array(
        [[0.0, a_si, a_si], [a_si, 0.0, a_si], [a_si, a_si, 0.0]],
        dtype=np.float64,
    )
    si_prim = xm.AtomsData(
        symbols=["Si", "Si"],
        positions=np.vstack(
            [
                0.25 * (prim_cell[0] + prim_cell[1] + prim_cell[2]),
                np.zeros(3),
            ]
        ),
        cell=prim_cell,
        pbc=(True, True, True),
    )
    with xm.xkcd_style("classic"):
        fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.4), dpi=165)
        fig.patch.set_facecolor(xm.style.PAPER)
        xm.plot_crystal(
            "nacl.cif",
            show_bonds=False,
            crystal_axes="corner",
            ax=axes[0],
            title="NaCl Unit Cell (show_bonds=False)",
        )
        xm.plot_structure_graph(
            "si_diamond.cif",
            layout="planar",
            ax=axes[1],
            title="Planar 2D GNN Graph (Si Diamond)",
        )
        xm.plot_structure_graph(
            si_prim,
            layout="planar",
            multi_edges=True,
            show_shift_labels=True,
            directed=True,
            ax=axes[2],
            title="2D Quotient Multigraph (Si Primitive)",
        )
        fig.suptitle(
            "Bond-Free Crystals & Planar Flat 2D Crystal GNN Graphs",
            fontsize=15,
            y=0.98,
        )
        fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.92), w_pad=2.0)
        xm.save_figure(fig, out_dir / "crystal_graphs_2d.png", dpi=165)
        plt.close(fig)


def _make_flagship_features_gallery(out_dir: pathlib.Path) -> None:
    """Generates flagship 3D isosurface, phonon, Wulff & scheme showcase."""
    with xm.xkcd_style("classic"):
        fig, axes = plt.subplots(2, 2, figsize=(12.2, 9.4), dpi=165)
        fig.patch.set_facecolor(xm.style.PAPER)
        xm.plot_isosurface(
            "water_homo.cube",
            shade="crosshatch",
            ax=axes[0, 0],
            title="Water HOMO (xkcd Cross-Hatch)",
        )
        xm.plot_phonon_bands(
            "si_phonon_bands.yaml",
            ax=axes[0, 1],
            title="Silicon Phonon Dispersion (THz)",
        )
        xm.plot_wulff_shape(
            "si_diamond.cif",
            {(1, 0, 0): 1.25, (1, 1, 0): 1.15, (1, 1, 1): 0.88},
            ax=axes[1, 0],
            title="Equilibrium Si Wulff Shape",
        )
        xm.plot_reaction_scheme(
            ["benzene.sdf"],
            ["aspirin.xyz"],
            conditions_above="Carboxylation",
            conditions_below=r"$\mathrm{AlCl_3},\ 85\,^\circ\mathrm{C}$",
            ax=axes[1, 1],
            title="2D Skeletal Reaction Scheme",
        )
        fig.suptitle(
            "3D Isosurfaces, Phonons, Wulff Shapes & Reaction Schemes",
            fontsize=15,
            y=0.98,
        )
        fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.93), w_pad=1.8, h_pad=2.0)
        xm.save_figure(fig, out_dir / "flagship_features.png", dpi=165)
        plt.close(fig)


def _make_isosurface_shading_gallery(out_dir: pathlib.Path) -> None:
    """Generates a 2x4 reference gallery of all 8 xkcd 3D shading modes."""
    modes = (
        ("solid", "1. Smooth Blending (solid)"),
        ("hatch", "2. Parallel Hatching (hatch)"),
        ("crosshatch", "3. Comic Cross-Hatch (crosshatch)"),
        ("contour", "4. 3D Axis Contour (contour)"),
        ("cross_contour", "5. 3D Cross-Contour (cross_contour)"),
        ("stipple", "6. Fine-Liner Stippling (stipple)"),
        ("scumble", "7. Wandering Scumble (scumble)"),
        ("outline", "8. Clean Silhouette (outline)"),
    )
    with xm.xkcd_style("classic"):
        fig, axes = plt.subplots(2, 4, figsize=(16.8, 8.8), dpi=165)
        fig.patch.set_facecolor(xm.style.PAPER)
        for ax, (mode, heading) in zip(axes.ravel(), modes, strict=True):
            xm.plot_isosurface(
                "water_homo.cube",
                shade=mode,
                ax=ax,
                title=heading,
            )
        axes[0, 0].annotate(
            "Light",
            xy=(0.24, 0.78),
            xytext=(0.06, 0.92),
            xycoords="axes fraction",
            fontsize=10.5,
            arrowprops={"arrowstyle": "->", "lw": 1.3, "color": xm.style.INK},
        )
        fig.suptitle(
            "xkcd Pen-and-Ink 3D Isosurface Shading Techniques",
            fontsize=15,
            y=0.98,
        )
        fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.93), w_pad=1.4, h_pad=2.0)
        xm.save_figure(
            fig, out_dir / "isosurface_shading_styles.png", dpi=165
        )
        plt.close(fig)


def _make_brillouin_cohp_volcano_gallery(out_dir: pathlib.Path) -> None:
    """Generates Brillouin zone, LOBSTER COHP, volcano, and defect gallery."""
    with xm.xkcd_style("classic"):
        fig, axes = plt.subplots(2, 2, figsize=(12.4, 9.6), dpi=165)
        fig.patch.set_facecolor(xm.style.PAPER)
        xm.plot_brillouin_zone(
            "si_diamond.cif",
            ax=axes[0, 0],
            title="Silicon FCC First Brillouin Zone",
        )
        xm.plot_cohp(
            "si_cohp.lobster",
            ax=axes[0, 1],
            title="Si-Si Bonding (-pCOHP, LOBSTER)",
        )
        xm.plot_volcano(
            [-0.75, -0.35, -0.05, 0.25, 0.65],
            [-2.9, -1.35, -0.35, -1.25, -3.1],
            labels=["Ru", "Rh", "Pt", "Pd", "Au"],
            ax=axes[1, 0],
            title="Sabatier Catalysis Volcano",
        )
        xm.plot_defect_formation(
            {
                "V_Si": [(2, 2.0), (1, 2.35), (0, 3.05)],
                "Si_i": [(1, 2.5), (0, 2.95), (-1, 3.85)],
                "O_i": [(0, 2.4), (-1, 2.95), (-2, 3.85)],
            },
            band_gap=1.12,
            ax=axes[1, 1],
            title="Si Defect Formation Energies",
        )
        fig.suptitle(
            "Brillouin Zones, LOBSTER -pCOHP, Volcanos & Defect Levels",
            fontsize=15,
            y=0.98,
        )
        fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.93), w_pad=2.0, h_pad=2.2)
        xm.save_figure(fig, out_dir / "brillouin_cohp_volcano.png", dpi=165)
        plt.close(fig)


def _make_xkcd_comics_gallery(out_dir: pathlib.Path) -> None:
    """Generates xkcd.com stick-scientist comic panels & inline examples."""
    q_pts = np.linspace(0.0, 1.0, 41)
    soft_branch = 3.2 * np.sin(np.pi * q_pts) ** 2 - 1.6
    opt_branch = 6.2 + 1.6 * np.cos(np.pi * q_pts)
    ph_soft = xm.PhononBandStructure(
        q_distances=q_pts,
        frequencies=np.column_stack([soft_branch, opt_branch]),
        q_labels=["G", "X", "M"],
        q_ticks=[0.0, 0.5, 1.0],
    )

    with xm.xkcd_style("classic"):
        fig = plt.figure(figsize=(15.6, 8.8), dpi=165)
        fig.patch.set_facecolor(xm.style.PAPER)
        gs = fig.add_gridspec(2, 1, height_ratios=[1.05, 1.0], hspace=0.34)
        top_gs = gs[0].subgridspec(1, 3, wspace=0.14)
        bot_gs = gs[1].subgridspec(1, 2, width_ratios=[1.1, 1.2], wspace=0.16)

        ax0 = fig.add_subplot(top_gs[0, 0])
        ax1 = fig.add_subplot(top_gs[0, 1])
        ax2 = fig.add_subplot(top_gs[0, 2])
        for p_ax, p_title in (
            (ax0, "1. DFT Converged!"),
            (ax1, "2. Check Phonons?!"),
            (ax2, "3. Barrier Resolved"),
        ):
            p_ax.set_xticks([])
            p_ax.set_yticks([])
            for sp in p_ax.spines.values():
                sp.set_visible(True)
                sp.set_linewidth(2.0)
                sp.set_color(xm.style.INK)
            p_ax.set_title(p_title, fontsize=11.5, pad=6.0)

        ax0_sub = ax0.inset_axes((0.32, 0.45, 0.36, 0.50))
        xm.plot_crystal(
            "si_diamond.cif",
            show_legend=False,
            show_cell="clean",
            ax=ax0_sub,
        )
        xm.draw_stick_scientist(
            ax0,
            xy=(0.14, 0.05),
            scale=0.28,
            character="cueball",
            pose="stand",
            prop="clipboard",
            speech="Forces < 1e-4.\nShip it!",
            speech_offset=(0.01, 0.12),
        )
        xm.draw_stick_scientist(
            ax0,
            xy=(0.86, 0.05),
            scale=0.28,
            character="ponytail",
            facing="left",
            pose="think",
            speech="Wait... did you\ncheck gamma/X?",
            speech_offset=(-0.01, 0.12),
        )

        ax1_sub = ax1.inset_axes((0.36, 0.38, 0.59, 0.54))
        xm.plot_phonon_bands(ph_soft, ax=ax1_sub)
        ax1_sub.tick_params(labelsize=7.5)
        ax1_sub.set_xlabel("")
        ax1_sub.set_ylabel("THz", fontsize=8.0)
        xm.draw_stick_scientist(
            ax1,
            xy=(0.11, 0.05),
            scale=0.28,
            character="ponytail",
            pose="point",
            prop="pointer",
            speech="Imaginary\nmode!",
            speech_offset=(0.01, 0.12),
        )
        xm.draw_stick_scientist(
            ax1,
            xy=(0.88, 0.05),
            scale=0.26,
            character="cueball",
            facing="left",
            pose="shrug",
        )

        ax2_sub = ax2.inset_axes((0.36, 0.38, 0.59, 0.54))
        xm.plot_neb(
            [0.0, 0.38, 0.96, 0.42, 1.08, 0.35, -0.22], ax=ax2_sub
        )
        ax2_sub.tick_params(labelsize=7.5)
        ax2_sub.set_xlabel("")
        ax2_sub.set_ylabel("eV", fontsize=8.0)
        xm.draw_stick_scientist(
            ax2,
            xy=(0.13, 0.05),
            scale=0.28,
            character="megan",
            pose="cheer",
            speech="Double-well\nfound!",
            speech_offset=(0.01, 0.12),
        )

        ax_inline = fig.add_subplot(bot_gs[0, 0])
        xm.plot_defect_formation(
            {
                "V_Si": [(2, 2.0), (1, 2.35), (0, 3.05)],
                "Si_i": [(1, 2.5), (0, 2.95), (-1, 3.85)],
                "O_i": [(0, 2.4), (-1, 2.95), (-2, 3.85)],
            },
            band_gap=1.12,
            ax=ax_inline,
            title="Inline Plot Commentary (draw_stick_scientist)",
        )
        xm.draw_stick_scientist(
            ax_inline,
            xy=(0.17, 0.04),
            scale=0.26,
            character="ponytail",
            facing="right",
            pose="point",
            speech="Low E_F drives\np-type V_Si!",
            speech_offset=(0.05, 0.06),
        )

        ax_cast = fig.add_subplot(bot_gs[0, 1])
        ax_cast.set_xlim(0.0, 1.0)
        ax_cast.set_ylim(0.0, 1.0)
        ax_cast.set_axis_off()
        ax_cast.set_title(
            "xkcd.com Scientist Cast, Poses & Lab Props", fontsize=12.5
        )
        cast = (
            ("cueball", "stand", "clipboard", "cueball\n(stand)", "right"),
            ("ponytail", "think", None, "ponytail\n(think)", "right"),
            ("megan", "point", "pointer", "megan\n(point)", "right"),
            ("beret", "shrug", None, "beret\n(shrug)", "right"),
            ("goggles", "cheer", "flask", "goggles\n(cheer)", "right"),
        )
        for idx, (char_k, pose_k, prop_k, tag, face_k) in enumerate(cast):
            x_c = 0.11 + idx * 0.195
            xm.draw_stick_scientist(
                ax_cast,
                xy=(x_c, 0.28),
                scale=0.38,
                character=char_k,
                pose=pose_k,
                prop=prop_k,
                facing=face_k,
                speech="Hi!" if idx == 3 else None,
                speech_offset=(0.0, 0.14),
            )
            ax_cast.text(
                x_c,
                0.14,
                tag,
                ha="center",
                va="top",
                fontsize=9.5,
                color=xm.style.INK,
            )

        fig.suptitle(
            "xkcd.com Stick Scientists, Inline Plot Commentary & Comic Strips",
            fontsize=15,
            y=0.98,
        )
        fig.subplots_adjust(
            left=0.05, right=0.97, top=0.90, bottom=0.06
        )
        xm.save_figure(
            fig, out_dir / "xkcd_comics_and_stickmen.png", dpi=165
        )
        plt.close(fig)


def main() -> None:
    """Generates all gallery PNGs, GIFs, and static interactive assets."""
    matplotlib.use("Agg")
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    _make_hero(_OUT_DIR)
    _make_molecules(_OUT_DIR)
    _make_crystals(_OUT_DIR)
    _make_electronic(_OUT_DIR)
    _make_spectroscopy_reactions(_OUT_DIR)
    _make_md_thermo(_OUT_DIR)
    _make_opt_in_poster(_OUT_DIR)
    _make_periodic_table_gallery(_OUT_DIR)
    _make_materials_informatics_gallery(_OUT_DIR)
    _make_scrolls_and_grids_gallery(_OUT_DIR)
    _make_icons_and_graphs_gallery(_OUT_DIR)
    _make_crystal_graphs_2d_gallery(_OUT_DIR)
    _make_flagship_features_gallery(_OUT_DIR)
    _make_isosurface_shading_gallery(_OUT_DIR)
    _make_brillouin_cohp_volcano_gallery(_OUT_DIR)
    _make_xkcd_comics_gallery(_OUT_DIR)
    _make_readme_gifs(_OUT_DIR)
    _export_docs_assets()
    print(f"Generated gallery images, GIFs, and studio in {_BOOK_STATIC}")


if __name__ == "__main__":
    main()
