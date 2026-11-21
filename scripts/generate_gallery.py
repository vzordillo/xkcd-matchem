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
        vols = np.linspace(36.0, 44.0, 8)
        energies = -12.4 + 0.065 * (vols - 40.0) ** 2
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
    """Generates opt-in Board/Container/Card/Badge millimeter poster."""
    board_p = xm.Board(
        260.0,
        176.0,
        dpi=170,
        title="Opt-In Board Layout: High-Throughput Catalyst Screening",
        subtitle="Composed with Board / Container / Card / Badge in mm units",
        preset="poster",
    )
    board_p.add_container(
        6.0,
        25.0,
        248.0,
        144.0,
        title="Workflow & Screening Summary",
        solid_header=True,
    )
    p1 = board_p.add_card(
        10.0,
        36.5,
        118.0,
        59.0,
        title="Pipeline Architecture",
        header_band=True,
        inner_pad_mm=(4.5, 4.5, 11.5, 4.5),
    )
    p2 = board_p.add_card(
        134.0,
        36.5,
        116.0,
        59.0,
        title="Functional Benchmark",
        header_band=True,
        inner_pad_mm=(19.5, 6.0, 13.5, 14.0),
    )
    p3 = board_p.add_card(
        10.0,
        102.0,
        118.0,
        61.0,
        title="TiO2 (110) Slab",
        header_band=True,
        inner_pad_mm=(4.5, 4.5, 11.5, 4.5),
    )
    p4 = board_p.add_card(
        134.0,
        102.0,
        116.0,
        61.0,
        title="Pourbaix Stability",
        header_band=True,
        inner_pad_mm=(19.5, 6.0, 13.5, 15.0),
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
    board_p.add_badge(234.0, 15.0, "Opt-in Board")
    board_p.save(out_dir / "poster_board.png")
    plt.close(board_p.figure)


def _export_docs_assets() -> None:
    """Exports offline interactive studio HTML and Patrick Hand font file."""
    fonts_dir = _BOOK_STATIC / "fonts"
    fonts_dir.mkdir(parents=True, exist_ok=True)
    ttf_src = xm.style.get_bundled_font_path("Patrick Hand")
    shutil.copy2(ttf_src, fonts_dir / "PatrickHand-Regular.ttf")

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
    _make_readme_gifs(_OUT_DIR)
    _export_docs_assets()
    print(f"Generated gallery images, GIFs, and studio in {_BOOK_STATIC}")


if __name__ == "__main__":
    main()
