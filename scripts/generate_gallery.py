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

import dataclasses
import io
import pathlib
import shutil

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import text as mtext
import numpy as np
from PIL import Image

import xkcd_matchem as xm


_BOOK_STATIC = (
    pathlib.Path(__file__).resolve().parent.parent / "book" / "_static"
)
_OUT_DIR = _BOOK_STATIC / "gallery"

_ASPIRIN_CAM = xm.Camera(azimuth=-125.0, elevation=42.0, perspective=0.28)

_SILICON_DEFECTS: dict[str, list[tuple[int, float]]] = {
    "V_Si": [(2, 1.85), (1, 2.20), (0, 2.90)],
    "Si_i": [(1, 2.85), (0, 3.30), (-1, 4.20)],
    "O_i": [(0, 2.15), (-1, 2.65), (-2, 3.55)],
}


def _load_aligned_si_bands_dos() -> tuple[
    xm.BandStructure, xm.DensityOfStates
]:
    """Loads QE Si bands and DOS aligned to the shared DOS Fermi energy."""
    bands = xm.load("si_qe_bands.dat")
    dos = xm.load("si_qe_dos.dat")
    aligned = dataclasses.replace(bands, fermi_energy=dos.fermi_energy)
    return aligned, dos


def _nudge_hull_labels(ax: matplotlib.axes.Axes) -> None:
    """Adds subtle vertical clearance below convex-hull node labels."""
    for txt in ax.texts:
        if isinstance(txt, mtext.Annotation):
            dx, dy = txt.xyann
            if dy < -5:
                txt.xyann = (dx, dy - 3.5)
    y_lo, y_hi = ax.get_ylim()
    span = max(y_hi - y_lo, 0.2)
    ax.set_ylim(y_lo - 0.05 * span, y_hi)


def _lift_slab_callout(ax: matplotlib.axes.Axes, lift: float = 0.72) -> None:
    """Lifts the slab vacuum callout box cleanly above the unit-cell border."""
    lifted = False
    for txt in ax.texts:
        if isinstance(txt, mtext.Annotation) and "Vacuum" in txt.get_text():
            tx, ty = txt.xyann
            txt.xyann = (tx, ty + lift)
            lifted = True
    if lifted:
        y_lo, y_hi = ax.get_ylim()
        ax.set_ylim(y_lo, y_hi + lift * 1.15)


def _make_hydroxybenzoic_precursor() -> xm.AtomsData:
    """Derives the exact hydroxybenzoic-acid precursor from aspirin.xyz."""
    asp = xm.load("aspirin.xyz")
    drop = {7, 8, 9, 17, 18, 19}
    keep = [i for i in range(len(asp)) if i not in drop]
    o_pos = asp.positions[6]
    c_acetyl = asp.positions[7]
    oh_dir = c_acetyl - o_pos
    oh_dir = oh_dir / max(float(np.linalg.norm(oh_dir)), 1e-12)
    h_pos = o_pos + 0.97 * oh_dir
    symbols = [asp.symbols[i] for i in keep] + ["H"]
    positions = np.vstack([asp.positions[keep], h_pos[None, :]])
    return xm.AtomsData(symbols=symbols, positions=positions)


def _make_hero(out_dir: pathlib.Path) -> None:
    """Generates a clean 2x3 open multi-panel hero showcase on white paper."""
    bands, _ = _load_aligned_si_bands_dos()
    with xm.xkcd_style("classic"):
        fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.8), dpi=180)
        fig.patch.set_facecolor(xm.style.PAPER)

        xm.plot_molecule(
            "aspirin.xyz",
            ax=axes[0, 0],
            camera=_ASPIRIN_CAM,
            title="Aspirin (3D)",
        )
        xm.plot_coordination(
            "tio2_slab.poscar",
            "Ti",
            "O",
            supercell=(2, 2, 1),
            crystal_axes="corner",
            ax=axes[0, 1],
            title="Rutile TiO2 (110) Polyhedra",
        )
        xm.plot_xrd(
            "nacl.cif",
            ax=axes[0, 2],
            label_peaks=4,
            title=r"NaCl Powder XRD (Cu $K\alpha_1$)",
        )
        xm.plot_bands(
            bands,
            ax=axes[1, 0],
            ylim=(-6.5, 5.5),
            title="Silicon Band Structure (QE)",
        )
        xm.plot_neb(
            xm.load_trajectory("sn2_neb.xyz"),
            ax=axes[1, 1],
            title="SN2 Reaction Barrier (NEB)",
        )
        xm.plot_phase_diagram_2d(
            [0.0, 0.25, 0.5, 0.75, 1.0],
            [0.0, -0.34, -0.45, -0.08, 0.0],
            labels=["A", "A3B", "AB", "AB3", "B"],
            ax=axes[1, 2],
            title="Binary Convex Hull",
        )
        _nudge_hull_labels(axes[1, 2])
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
            camera=_ASPIRIN_CAM,
            title="Aspirin 3D (Ball & Stick)",
        )
        xm.plot_molecule(
            aspirin,
            ax=axes[1],
            style="spacefill",
            camera=_ASPIRIN_CAM,
            title="Aspirin 3D (Spacefill)",
        )
        xm.plot_molecule(
            aspirin,
            ax=axes[2],
            mode="2d",
            title="Aspirin 2D Skeletal",
        )
        axes[2].set_aspect("equal", adjustable="box")
        x0, x1 = axes[2].get_xlim()
        y0, y1 = axes[2].get_ylim()
        ymid = 0.5 * (y0 + y1)
        half_h = max(0.5 * (y1 - y0), 0.44 * (x1 - x0))
        axes[2].set_ylim(ymid - half_h, ymid + half_h)

        fig.suptitle(
            "Molecular Representations: 3D Ball-and-Stick, Spacefill "
            "& 2D Skeletal",
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
            camera=xm.Camera(azimuth=26.0, elevation=18.0),
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
        _lift_slab_callout(axes[1], lift=0.72)
        xm.plot_xrd(
            "si_diamond.cif",
            two_theta_range=(20.0, 85.0),
            ax=axes[2],
            title=r"Silicon Powder XRD (Cu $K\alpha_1$)",
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
    bands, dos = _load_aligned_si_bands_dos()
    fig, (_, ax_d) = xm.plot_bands_dos(
        bands, dos, ylim=(-8.0, 5.6), title="Silicon Band Structure & DOS"
    )
    fig.patch.set_facecolor(xm.style.PAPER)
    with xm.xkcd_style("classic"):
        ax_d.set_xlim(0.0, 2.75)
        ax_d.legend(loc="upper right", frameon=True, fontsize=8.8)
    fig.subplots_adjust(
        left=0.10, right=0.97, top=0.88, bottom=0.13, wspace=0.08
    )
    xm.save_figure(fig, out_dir / "bands_dos.png", dpi=170)
    plt.close(fig)

    with xm.xkcd_style("classic"):
        fig_mo, axes = plt.subplots(1, 2, figsize=(12.2, 5.4), dpi=170)
        fig_mo.patch.set_facecolor(xm.style.PAPER)
        orca_atoms = xm.load("benzene_orca.out")
        xm.plot_mo_diagram(
            orca_atoms.metadata["orbitals"],
            window=(-12.5, 4.5),
            ax=axes[0],
            title="Benzene Molecular Orbitals (ORCA)",
        )
        xm.plot_volumetric_slice(
            xm.load("water_homo.cube"),
            axis="z",
            ax=axes[1],
            title="Water HOMO Wavefunction Slice (z = 0)",
        )
        fig_mo.suptitle(
            "Molecular Orbitals & Real-Space Volumetric Slices",
            fontsize=15,
            y=0.98,
        )
        fig_mo.tight_layout(rect=(0.0, 0.0, 1.0, 0.92), w_pad=2.4)
        xm.save_figure(fig_mo, out_dir / "mo_and_cube.png", dpi=170)
        plt.close(fig_mo)


def _polish_spectrum_labels(
    ax: matplotlib.axes.Axes,
    nudges: dict[str, tuple[float, float, str]],
) -> None:
    """Offsets crowded spectral peak tags with un-sketched halos and leaders."""
    sw = xm.stroke_weights("classic")
    halo = xm.style.theme.label_halo_bbox(xm.style.PAPER, pad=0.06, alpha=0.88)
    existing = [t for t in ax.texts if isinstance(t, mtext.Annotation)]
    for txt in existing:
        label = txt.get_text().strip()
        if label in nudges:
            dx, dy, ha = nudges[label]
            xy_pt = txt.xy
            txt.remove()
            ann = ax.annotate(
                label,
                xy=xy_pt,
                xytext=(dx, dy),
                textcoords="offset points",
                ha=ha,
                va="center",
                fontsize=8.5,
                color=xm.style.INK,
                bbox=halo,
                arrowprops={
                    "arrowstyle": "-",
                    "color": xm.style.INK,
                    "lw": sw.leader * 0.85,
                    "shrinkA": 2.0,
                    "shrinkB": 3.0,
                },
                zorder=6,
            )
            if ann.arrow_patch is not None:
                ann.arrow_patch.set_sketch_params(0.0, 0.0, 0.0)
            patch = ann.get_bbox_patch()
            if patch is not None:
                patch.set_sketch_params(0.0, 0.0, 0.0)
        else:
            txt.set_bbox(halo)
            patch = txt.get_bbox_patch()
            if patch is not None:
                patch.set_sketch_params(0.0, 0.0, 0.0)


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
        _polish_spectrum_labels(axes[0], {"3812": (24.0, 14.0, "left")})
        xm.plot_spectrum(
            benz.metadata["uvvis_spectrum"],
            ax=axes[1],
            title="Benzene UV-Vis Spectrum (ORCA)",
        )
        axes[1].set_xlim(145.0, 310.0)
        _polish_spectrum_labels(axes[1], {"205.7 nm": (14.0, 12.0, "left")})
        fig_s.suptitle(
            "Vibrational (IR) & Optical (UV-Vis) Molecular Spectra",
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
        _nudge_hull_labels(axes_t[0])
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
            title="ML Potential Energy Parity",
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
        title="Poster & Figure Board Layout: Composing Cards in Containers",
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
        inner_pad_mm=(4.5, 4.5, 12.0, 5.0),
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
    _lift_slab_callout(p3.axes, lift=0.82)
    xm.plot_pourbaix(
        [
            ("M(s)", [(0, -1.2), (14, -1.2), (14, -0.58), (0, -0.58)]),
            ("M2+", [(0, -0.58), (5.0, -0.58), (5.0, 1.65), (0, 1.65)]),
            ("MO2", [(5.0, -0.58), (14, -0.58), (14, 1.05), (5.0, 1.05)]),
            ("MO4-", [(5.0, 1.05), (14, 1.05), (14, 1.65), (5.0, 1.65)]),
        ],
        potential_range=(-1.2, 1.8),
        ax=p4.axes,
        preset="poster",
    )
    board_p.add_badge(238.0, 15.0, "Opt-in Board")
    board_p.save(out_dir / "poster_board.png")
    plt.close(board_p.figure)


def _export_docs_assets() -> None:
    """Exports offline interactive studio HTML and SIL OFL default font."""
    fonts_dir = _BOOK_STATIC / "fonts"
    fonts_dir.mkdir(parents=True, exist_ok=True)
    ttf_src = xm.style.get_bundled_font_path("Patrick Hand")
    shutil.copy2(ttf_src, fonts_dir / "PatrickHand-Regular.ttf")

    xm.export_interactive_html(
        "water_homo.cube",
        _BOOK_STATIC / "interactive_studio.html",
        include_bundled_samples=True,
        iso_shade="crosshatch",
    )


def _fig_to_rgb_array(
    fig: matplotlib.figure.Figure,
    dpi: int = 220,
) -> np.ndarray:
    """Renders a Matplotlib Figure into an exact full-color RGB NumPy array."""
    buf = io.BytesIO()
    fig.patch.set_sketch_params(0.0, 0.0, 0.0)
    for ax in fig.axes:
        ax.patch.set_sketch_params(0.0, 0.0, 0.0)
    fig.savefig(
        buf,
        format="png",
        dpi=dpi,
        facecolor=fig.get_facecolor(),
        edgecolor=fig.get_facecolor(),
    )
    buf.seek(0)
    with Image.open(buf) as raw:
        return np.asarray(raw.convert("RGB"))


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
                crystal_axes="corner",
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
    bands, _ = _load_aligned_si_bands_dos()
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
                ylim=(-6.5, 5.5),
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


def _draw_orbit_triad(
    ax: matplotlib.axes.Axes,
    cam: xm.Camera,
    lim: float,
) -> None:
    """Draws a compact rotating (x, y, z) triad in corner of orbit panel."""
    basis = xm.core.geometry.rotation_matrix(
        cam.azimuth, cam.elevation, cam.roll
    )
    sw = xm.stroke_weights("classic")
    ox, oy = -0.78 * lim, -0.76 * lim
    axis_len = 0.24 * lim
    specs = [(0, "x", "#D55E00"), (1, "y", "#009E73"), (2, "z", "#0072B2")]
    rows = [
        (float(basis[2, k]), float(basis[0, k]), float(basis[1, k]), lbl, col)
        for k, lbl, col in specs
    ]
    rows.sort(key=lambda item: item[0])
    for _, dx, dy, lbl, col in rows:
        tx, ty = ox + dx * axis_len, oy + dy * axis_len
        arr = ax.annotate(
            "",
            xy=(tx, ty),
            xytext=(ox, oy),
            arrowprops={
                "arrowstyle": "-|>,head_width=0.16,head_length=0.25",
                "lw": sw.leader,
                "color": col,
                "shrinkA": 0.0,
                "shrinkB": 0.0,
            },
            zorder=52,
        )
        if arr.arrow_patch is not None:
            arr.arrow_patch.set_sketch_params(0.0, 0.0, 0.0)
        n2 = max(float(np.hypot(dx, dy)), 1e-6)
        txt = ax.text(
            tx + 0.11 * lim * (dx / n2),
            ty + 0.11 * lim * (dy / n2),
            lbl,
            color=col,
            fontsize=8.0,
            ha="center",
            va="center",
            zorder=53,
        )
        txt.set_sketch_params(0.0, 0.0, 0.0)
    dot = ax.scatter([ox], [oy], s=14, color=xm.style.INK, zorder=54)
    dot.set_sketch_params(0.0, 0.0, 0.0)


def _make_isosurface_orbit_gif(out_dir: pathlib.Path) -> None:
    """Generates a 2x4 orthographic isometric 360-deg orbit comparison GIF."""
    cube_raw = xm.load("water_homo.cube")
    cube_up = xm.core.isosurface.upsample_coarse_grid(cube_raw, min_dim=36)
    iso_elev = float(np.degrees(np.arcsin(1.0 / np.sqrt(3.0))))
    max_abs = max(float(np.max(np.abs(cube_up.data))), 1e-12)
    level = 0.35 * max_abs
    m_pos = xm.core.isosurface.extract_isosurface(cube_up, level)
    m_neg = xm.core.isosurface.extract_isosurface(cube_up, -level)
    pivot = np.mean(cube_up.atoms.positions, axis=0)
    pts_all = np.vstack(
        [cube_up.atoms.positions, m_pos.vertices, m_neg.vertices]
    )
    lim = float(np.max(np.linalg.norm(pts_all - pivot, axis=1))) + 0.48
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
    sw = xm.stroke_weights("classic")
    frames: list[Image.Image] = []
    for az in 45.0 + np.linspace(0.0, 360.0, 18, endpoint=False):
        cam = xm.Camera(
            azimuth=float(az),
            elevation=iso_elev,
            roll=0.0,
            perspective=0.0,
            zoom=1.0,
        )
        with xm.xkcd_style("classic"):
            fig, axes = plt.subplots(2, 4, figsize=(13.6, 7.2), dpi=100)
            fig.patch.set_facecolor(xm.style.PAPER)
            for ax, (mode, heading) in zip(axes.ravel(), modes, strict=True):
                xm.plot_isosurface(
                    cube_up, camera=cam, shade=mode, ax=ax, title=heading
                )
                ax.set_adjustable("box")
                ax.set_xlim(-lim, lim)
                ax.set_ylim(-lim, lim)
            _draw_orbit_triad(axes[0, 0], cam, lim)
            l_ann = axes[0, 0].annotate(
                "Light",
                xy=(0.20, 0.80),
                xytext=(0.04, 0.94),
                xycoords="axes fraction",
                fontsize=9.0,
                color=xm.style.INK,
                arrowprops={
                    "arrowstyle": "->",
                    "lw": sw.leader,
                    "color": xm.style.INK,
                },
            )
            if l_ann.arrow_patch is not None:
                l_ann.arrow_patch.set_sketch_params(0.0, 0.0, 0.0)
            fig.suptitle(
                "3D Isosurface Shading — Orthographic Isometric Orbit "
                f"(el = 35.3°, az = {float(az) % 360.0:05.1f}°)",
                fontsize=14.0,
                y=0.975,
            )
            fig.tight_layout(
                rect=(0.0, 0.01, 1.0, 0.935), w_pad=1.1, h_pad=1.5
            )
            frames.append(_fig_to_pil(fig))
            plt.close(fig)
    _save_gif(frames, out_dir / "isosurface_shading_orbit.gif", 130)


def _make_water_to_ice_gif(out_dir: pathlib.Path) -> None:
    """Generates a synchronized MACE-MPA-0 water-to-ice Ih MD freezing GIF."""
    traj = xm.load_trajectory("water_to_ice.xyz")
    times = (
        traj.times
        if traj.times is not None
        else np.arange(len(traj), dtype=np.float64)
    )
    temps = traj.temperatures
    rel_e = traj.energy_profile(relative=True, unit="eV")
    q_vals = traj.tetrahedral_order("O")
    hb_counts = [len(xm.infer_hbonds(frame)) for frame in traj.frames]

    cam = xm.Camera(azimuth=-72.0, elevation=14.0, perspective=0.25)
    chunks: list[np.ndarray] = []
    for frame in traj.frames:
        if frame.cell is None:
            continue
        pivot = 0.5 * (frame.cell[0] + frame.cell[1] + frame.cell[2])
        span = xm.core.geometry.scene_projection_span(
            frame.positions, cam, center=pivot, cell=frame.cell
        )
        xy, _ = xm.core.geometry.project_points(
            frame.positions, cam, center=pivot, span=span
        )
        c2d, _, _ = xm.core.geometry.cell_corners_projected(
            frame.cell, cam, center=pivot, span=span
        )
        chunks.extend([xy, c2d])
    all_pts = np.vstack(chunks)
    pad = 1.15
    x_lo = float(all_pts[:, 0].min()) - pad
    x_hi = float(all_pts[:, 0].max()) + pad
    y_lo = float(all_pts[:, 1].min()) - pad
    y_hi = float(all_pts[:, 1].max()) + pad

    sw = xm.stroke_weights("classic")
    t_end = max(float(times[-1]), 1.0)
    q_lo, q_hi = 0.70, 1.04
    e_lo = float(np.min(rel_e)) - 0.55
    e_hi = max(float(np.max(rel_e)) + 0.45, 0.5)

    frames: list[Image.Image] = []
    # Brief pause on final frozen crystal frame so viewers absorb Ice Ih lattice
    seq = list(range(len(traj))) + [len(traj) - 1, len(traj) - 1]
    for idx in seq:
        with xm.xkcd_style("classic"):
            fig = plt.figure(figsize=(10.2, 5.2), dpi=95)
            fig.patch.set_facecolor(xm.style.PAPER)
            fig.patch.set_edgecolor(xm.style.PAPER)
            fig.patch.set_sketch_params(0.0, 0.0, 0.0)
            gs = fig.add_gridspec(
                2, 2, width_ratios=[1.02, 1.18], hspace=0.42, wspace=0.18
            )
            ax_3d = fig.add_subplot(gs[:, 0])
            ax_q = fig.add_subplot(gs[0, 1])
            ax_e = fig.add_subplot(gs[1, 1])

            xm.render_structure_3d(
                traj[idx],
                ax=ax_3d,
                camera=cam,
                show_cell="clean",
                show_hbonds=True,
                radius_scale=0.88,
            )
            ax_3d.set_xlim(x_lo, x_hi)
            ax_3d.set_ylim(y_lo, y_hi)
            ax_3d.set_aspect("equal", adjustable="box")
            t_tag = (
                f" | {float(temps[idx]):.0f} K"
                if temps is not None
                else ""
            )
            ax_3d.set_title(
                f"Water -> Ice Ih ({hb_counts[idx]} H-bonds{t_tag})",
                fontsize=12.0,
            )

            # Top-right: Errington-Debenedetti tetrahedral order q(t)
            ax_q.plot(
                times,
                q_vals,
                color="#0072B2",
                lw=sw.overlay,
                alpha=0.25,
                zorder=2.0,
            )
            ax_q.plot(
                times[: idx + 1],
                q_vals[: idx + 1],
                color="#0072B2",
                lw=sw.primary,
                zorder=3.5,
            )
            ax_q.fill_between(
                times[: idx + 1],
                q_lo,
                q_vals[: idx + 1],
                color="#56B4E9",
                alpha=0.18,
                zorder=1.5,
            )
            q_ref = ax_q.axhline(
                1.0, color="#64748B", linestyle="--", lw=sw.guide, zorder=1.2
            )
            q_ref.set_sketch_params(0.0, 0.0, 0.0)
            ax_q.text(
                0.03 * t_end,
                0.985,
                "Ideal Ice Ih (q = 1.0)",
                fontsize=8.5,
                va="top",
                color=xm.style.INK,
            )
            ax_q.scatter(
                [float(times[idx])],
                [float(q_vals[idx])],
                s=95,
                color="#D55E00",
                edgecolor=xm.style.INK,
                linewidth=1.4,
                clip_on=False,
                zorder=6.0,
            )
            ax_q.set_xlim(-0.25, t_end + 0.35)
            ax_q.set_ylim(q_lo, q_hi)
            ax_q.set_ylabel("Tetrahedral q")
            ax_q.set_title(
                f"MACE-MPA-0 Crystallization | t = {float(times[idx]):.1f} ps"
                f" (q = {float(q_vals[idx]):.2f})",
                fontsize=11.0,
            )
            xm.style.apply_xkcd_spines(ax_q, preset="classic")

            # Bottom-right: Exothermic potential energy drop
            ax_e.plot(
                times,
                rel_e,
                color="#009E73",
                lw=sw.overlay,
                alpha=0.25,
                zorder=2.0,
            )
            ax_e.plot(
                times[: idx + 1],
                rel_e[: idx + 1],
                color="#009E73",
                lw=sw.primary,
                zorder=3.5,
            )
            e_zero = ax_e.axhline(
                0.0, color="#64748B", linestyle="--", lw=sw.guide, zorder=1.2
            )
            e_zero.set_sketch_params(0.0, 0.0, 0.0)
            ax_e.scatter(
                [float(times[idx])],
                [float(rel_e[idx])],
                s=95,
                color="#D55E00",
                edgecolor=xm.style.INK,
                linewidth=1.4,
                clip_on=False,
                zorder=6.0,
            )
            ax_e.set_xlim(-0.25, t_end + 0.35)
            ax_e.set_ylim(e_lo, e_hi)
            ax_e.set_xlabel("Simulation Time (ps)")
            ax_e.set_ylabel("Delta E (eV)")
            xm.style.apply_xkcd_spines(ax_e, preset="classic")

            fig.subplots_adjust(
                left=0.03, right=0.97, top=0.91, bottom=0.12
            )
            frames.append(_fig_to_pil(fig))
            plt.close(fig)
    _save_gif(frames, out_dir / "water_to_ice_md.gif", duration_ms=175)


def _make_readme_gifs(out_dir: pathlib.Path) -> None:
    """Generates lightweight animated GIFs showcasing 3D, NEB, MD, & themes."""
    _make_crystal_orbit_gif(out_dir)
    _make_neb_gif(out_dir)
    _make_water_to_ice_gif(out_dir)
    _make_theme_switcher_gif(out_dir)
    _make_isosurface_orbit_gif(out_dir)


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
        (2.0 / 3.0, 0.0, 1.0 / 3.0),
        (0.0, 0.4, 0.6),
        (0.0, 0.5, 0.5),
        (0.25, 0.25, 0.50),
        (0.50, 0.10, 0.40),
        (0.12, 0.36, 0.52),
    ]
    energies = [0.0, 0.0, 0.0, -1.95, -1.68, -1.32, -1.86, -1.98, -1.38]
    labels = [
        "",
        "",
        "",
        "Li2O",
        "Fe2O3",
        "FeO",
        "LiFeO2",
        "Li5FeO4",
        "meta",
    ]
    sg_counts = {
        225: 36,
        221: 24,
        227: 22,
        194: 26,
        186: 16,
        139: 22,
        129: 18,
        62: 30,
        63: 20,
        14: 24,
        12: 18,
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
        for txt in axes[1].texts:
            if txt.get_text() == "Orthorhombic":
                txt.set_fontsize(7.1)
        xm.plot_histogram(
            gaps,
            bins=16,
            xlabel="DFT Band Gap (eV)",
            shade="hatch",
            background_grid="dotgrid",
            ax=axes[2],
            title="Hatched Band-Gap Distribution",
        )
        for ln in axes[2].lines:
            if ln.get_linestyle() == "-" and len(ln.get_xdata()) > 20:
                ln.set_color("#0072B2")
                ln.set_linewidth(xm.stroke_weights("classic").overlay + 0.35)
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
            ("5. Share", "Sketch with xkcd-matchem!"),
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
            ("Calcination", "850 °C / 12 h"),
            ("XRD Check", r"Cu $K\alpha_1$"),
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
            2, 2, width_ratios=[1.22, 1.38], height_ratios=[1.05, 1.02]
        )
        ax_icons = fig.add_subplot(gs[:, 0])
        sub_top = gs[0, 1].subgridspec(1, 2, wspace=0.08)
        ax_crys = fig.add_subplot(sub_top[0, 0])
        ax_graph = fig.add_subplot(sub_top[0, 1])
        ax_net = fig.add_subplot(gs[1, 1])

        xm.plot_icon_sheet(
            elements_to_show=("H", "Li", "C", "O", "Si", "Ti", "Fe", "Au"),
            ncols=6,
            ax=ax_icons,
            title="xkcd MatChem Icons & Element Badges",
        )
        cam = xm.Camera(azimuth=-65.0, elevation=18.0)
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
            (-1.85, 0.05),
            (-0.20, 1.25),
            (1.75, 1.25),
            (1.75, 0.05),
            (-1.85, -1.15),
            (1.75, -1.15),
        ]
        synth_edges = [
            (0, 1, "650 °C"),
            (0, 3, "800 °C"),
            (2, 3, "+ Li"),
            (0, 5, "950 °C"),
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
        ax_net.set_aspect("equal", adjustable="box")
        ax_net.set_xlim(-2.65, 2.75)
        ax_net.set_ylim(-1.75, 1.80)
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
            title="NaCl Unit Cell (Bond-Free Sites)",
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
    precursor = _make_hydroxybenzoic_precursor()
    with xm.xkcd_style("classic"):
        fig, axes = plt.subplots(2, 2, figsize=(12.4, 9.6), dpi=165)
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
            [precursor],
            ["aspirin.xyz"],
            conditions_above=r"Acetylation ($\mathrm{Ac_2O}$)",
            conditions_below=r"$\mathrm{H_3PO_4},\ 85\,^\circ\mathrm{C}$",
            ax=axes[1, 1],
            title="2D Skeletal Reaction Scheme",
        )
        axes[1, 1].set_aspect("equal", adjustable="box")
        rx0, rx1 = axes[1, 1].get_xlim()
        ry0, ry1 = axes[1, 1].get_ylim()
        ry_mid = 0.5 * (ry0 + ry1)
        half_ry = max(0.5 * (ry1 - ry0), 0.36 * (rx1 - rx0))
        axes[1, 1].set_ylim(ry_mid - half_ry, ry_mid + half_ry)

        fig.suptitle(
            "3D Isosurfaces, Phonons, Wulff Shapes & Reaction Schemes",
            fontsize=15,
            y=0.98,
        )
        fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.93), w_pad=2.0, h_pad=2.4)
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
    sw = xm.stroke_weights("classic")
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
        l_ann = axes[0, 0].annotate(
            "Light",
            xy=(0.24, 0.78),
            xytext=(0.06, 0.92),
            xycoords="axes fraction",
            fontsize=10.5,
            color=xm.style.INK,
            arrowprops={
                "arrowstyle": "->",
                "lw": sw.leader,
                "color": xm.style.INK,
            },
        )
        if l_ann.arrow_patch is not None:
            l_ann.arrow_patch.set_sketch_params(0.0, 0.0, 0.0)
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
            camera=xm.Camera(azimuth=-128.0, elevation=26.0),
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
            _SILICON_DEFECTS,
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
            speech="Wait... did you\ncheck phonons?",
            speech_offset=(-0.01, 0.12),
        )

        ax1_sub = ax1.inset_axes((0.36, 0.38, 0.59, 0.54))
        xm.plot_phonon_bands(ph_soft, ax=ax1_sub)
        ax1_sub.tick_params(labelsize=7.5)
        ax1_sub.set_xlabel("")
        ax1_sub.set_ylabel("THz", fontsize=8.0)
        leg1 = ax1_sub.get_legend()
        if leg1 is not None:
            for t_leg in leg1.get_texts():
                t_leg.set_fontsize(6.8)
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
        ax2_sub.set_ylim(-0.38, 1.48)
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
            _SILICON_DEFECTS,
            band_gap=1.12,
            ax=ax_inline,
            title="Inline Plot Commentary with Stick Scientists",
        )
        xm.draw_stick_scientist(
            ax_inline,
            xy=(0.17, 0.04),
            scale=0.25,
            character="ponytail",
            facing="right",
            pose="point",
            speech=r"Low $E_F$ drives" + "\n" + r"p-type $\mathrm{V_{Si}}$!",
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


def _make_scicomm_templates_gallery(out_dir: pathlib.Path) -> None:
    """Generates showcase of Graphical Abstract, Talk Slide & Comic recipes."""
    bands, _ = _load_aligned_si_bands_dos()
    neb = xm.load_trajectory("sn2_neb.xyz")
    cam_sn2 = xm.Camera(azimuth=-115.0, elevation=20.0, perspective=0.25)

    ga = xm.create_graphical_abstract(
        title="Rutile TiO2 (110) -> Sabatier Activity",
        takeaway="Optimal *H binding energy sits right at the volcano apex",
        panels=2,
        size_mm=(136.0, 76.5),
        arrow_label="DFT",
        dpi=240,
    )
    xm.plot_coordination(
        "tio2_slab.poscar",
        "Ti",
        "O",
        supercell=(2, 1, 1),
        show_cell=False,
        ax=ga.left_ax,
        preset="graphical_abstract",
        title="TiO2 Octahedra",
    )
    ga.left_ax.title.set_fontsize(8.8)
    if ga.right_ax is not None:
        xm.plot_volcano(
            [-0.75, -0.35, -0.05, 0.25, 0.65],
            [-2.9, -1.35, -0.35, -1.25, -3.1],
            labels=["Ru", "Rh", "Pt", "Pd", "Au"],
            ylabel="Limiting Potential (V)",
            ax=ga.right_ax,
            preset="graphical_abstract",
            title="HER Volcano",
        )
        leg_ga = ga.right_ax.get_legend()
        if leg_ga is not None:
            leg_ga.remove()
        ga.right_ax.set_xlim(-1.05, 0.88)
        ga.right_ax.set_ylim(-3.65, 0.35)
        ga.right_ax.tick_params(labelsize=6.8)
        ga.right_ax.xaxis.label.set_fontsize(7.4)
        ga.right_ax.yaxis.label.set_fontsize(7.4)
        ga.right_ax.title.set_fontsize(8.8)
    img_ga = _fig_to_rgb_array(ga.figure, dpi=240)
    plt.close(ga.figure)

    slide = xm.create_talk_slide(
        title="Silicon Bands & Charged-Defect Levels",
        subtitle="Self-consistent DFT bands paired with defect thermodynamics",
        takeaway="Indirect 1.12 eV gap sets the p-type V_Si stability window",
        layout="two_column",
        preset="talk_16x9",
        character="ponytail",
        speech="Look at V_Si (+2/0)!",
        dpi=200,
    )
    for ax_s in (slide.axes["left"], slide.axes["right"]):
        pos_s = ax_s.get_position()
        ax_s.set_position(
            [pos_s.x0, pos_s.y0 + 0.032, pos_s.width, pos_s.height - 0.032]
        )
    xm.plot_bands(
        bands,
        ax=slide.axes["left"],
        ylim=(-5.5, 4.5),
        preset="talk_16x9",
        title="Si Band Structure",
    )
    slide.axes["left"].set_xlabel("Wavevector Path", fontsize=11.0, labelpad=3)
    slide.axes["left"].set_ylabel(r"$E - E_F$ (eV)", fontsize=11.0)
    xm.plot_defect_formation(
        _SILICON_DEFECTS,
        band_gap=1.12,
        ax=slide.axes["right"],
        preset="talk_16x9",
        title="Si Defect Levels",
    )
    slide.axes["right"].set_xlabel(
        r"$E_F - E_{\mathrm{VBM}}$ (eV)", fontsize=11.0, labelpad=3
    )
    slide.axes["right"].set_ylabel(r"Formation Energy $E^f$ (eV)", fontsize=11)
    leg_def = slide.axes["right"].get_legend()
    if leg_def is not None:
        for txt_l in leg_def.get_texts():
            txt_l.set_fontsize(9.5)
    img_slide = _fig_to_rgb_array(slide.figure, dpi=200)
    plt.close(slide.figure)

    comic = xm.create_mechanism_comic(
        captions=(
            "1. Backside Approach",
            "2. Pentacoordinate TS!",
            "3. Walden Inversion",
        ),
        speeches=(
            "Cl- lines up...",
            "Look at that barrier!",
            "Inversion complete!",
        ),
        characters=("cueball", "ponytail", "goggles"),
        poses=("stand", "point", "cheer"),
        title=None,
        figsize=(14.6, 4.2),
    )
    xm.render_structure_3d(
        neb[0],
        ax=comic.inset_axes[0],
        camera=cam_sn2,
        show_cell=False,
    )
    xm.plot_neb(neb, ax=comic.inset_axes[1])
    comic.inset_axes[1].set_ylim(-0.08, 1.16)
    comic.inset_axes[1].tick_params(labelsize=7.5)
    comic.inset_axes[1].set_xlabel("")
    comic.inset_axes[1].set_ylabel("eV", fontsize=8.0)
    xm.render_structure_3d(
        neb[-1],
        ax=comic.inset_axes[2],
        camera=cam_sn2,
        show_cell=False,
    )
    img_comic = _fig_to_rgb_array(comic.figure, dpi=200)
    plt.close(comic.figure)

    with xm.xkcd_style("classic"):
        fig = plt.figure(figsize=(16.0, 9.8), dpi=175)
        fig.patch.set_facecolor(xm.style.PAPER)
        gs = fig.add_gridspec(
            2,
            2,
            height_ratios=[1.04, 0.96],
            width_ratios=[1.0, 1.0],
            hspace=0.22,
            wspace=0.06,
        )
        ax_ga = fig.add_subplot(gs[0, 0])
        ax_sl = fig.add_subplot(gs[0, 1])
        ax_cm = fig.add_subplot(gs[1, :])

        for ax_p, img_p, heading, framed in (
            (
                ax_ga,
                img_ga,
                "A. Journal TOC Abstract — create_graphical_abstract()",
                True,
            ),
            (
                ax_sl,
                img_slide,
                "B. 16:9 Talk Slide + Presenter — create_talk_slide()",
                True,
            ),
            (
                ax_cm,
                img_comic,
                "C. 3-Panel SN2 Mechanism Strip — create_mechanism_comic()",
                False,
            ),
        ):
            ax_p.imshow(img_p, interpolation="lanczos")
            ax_p.set_xticks([])
            ax_p.set_yticks([])
            ax_p.set_title(heading, fontsize=11.8, pad=7.0)
            for sp in ax_p.spines.values():
                sp.set_visible(framed)
                if framed:
                    sp.set_color("#94A3B8")
                    sp.set_linewidth(1.1)
                    sp.set_sketch_params(0.0, 0.0, 0.0)

        fig.suptitle(
            "Turnkey SciComm Templates: Graphical Abstracts, 16:9 Talk Slides "
            "& Mechanism Comics",
            fontsize=15.0,
            y=0.975,
        )
        fig.subplots_adjust(
            left=0.025, right=0.975, top=0.905, bottom=0.02
        )
        xm.save_figure(
            fig, out_dir / "scicomm_layout_templates.png", dpi=175
        )
        plt.close(fig)


def _synthesize_reversible_cv() -> tuple[np.ndarray, np.ndarray]:
    """Synthesizes a classic reversible Fe(CN)6 3-/4- cyclic voltammogram."""
    e_fwd = np.linspace(0.00, 0.72, 160)
    e_rev = np.linspace(0.72, 0.00, 160)
    # Forward anodic oxidation peak near E_pa ~ 0.39 V with diffusion tail.
    u_a = (e_fwd - 0.382) / 0.036
    j_fwd = (
        0.10
        + 2.45 * np.exp(-0.5 * u_a**2) * (1.0 + 0.30 * np.tanh(u_a))
        + 0.78
        / (1.0 + np.exp(-18.0 * (e_fwd - 0.405)))
        * (1.0 / np.sqrt(1.0 + np.maximum(e_fwd - 0.39, 0.0) * 6.5))
    )
    # Reverse cathodic reduction peak near E_pc ~ 0.33 V with smooth join.
    u_c = (e_rev - 0.338) / 0.036
    j_end = float(j_fwd[-1])
    join_w = np.exp(-((0.72 - e_rev) / 0.055) ** 2)
    j_rev_core = (
        -0.12
        - 2.40 * np.exp(-0.5 * u_c**2) * (1.0 - 0.30 * np.tanh(u_c))
        - 0.72
        / (1.0 + np.exp(18.0 * (e_rev - 0.315)))
        * (1.0 / np.sqrt(1.0 + np.maximum(0.33 - e_rev, 0.0) * 6.5))
    )
    j_rev = (1.0 - join_w) * j_rev_core + join_w * j_end
    return np.concatenate([e_fwd, e_rev]), np.concatenate([j_fwd, j_rev])


def _synthesize_lfp_battery() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Synthesizes LiFePO4 galvanostatic discharge/charge hysteresis curves."""
    v_grid = np.linspace(2.65, 3.95, 220)
    q_chg = 158.0 / (1.0 + np.exp(-(v_grid - 3.46) / 0.022)) + 3.2 * (
        v_grid - 2.65
    )
    q_chg = q_chg - float(q_chg[0])
    q_chg = q_chg * (162.0 / float(q_chg[-1]))

    q_dis_asc = 158.0 / (1.0 + np.exp(-(v_grid - 3.39) / 0.022)) + 3.2 * (
        v_grid - 2.65
    )
    q_dis_asc = q_dis_asc - float(q_dis_asc[0])
    q_dis_asc = q_dis_asc * (162.0 / float(q_dis_asc[-1]))
    # Keep Discharge monotonically below Charge on the shared capacity axis.
    q_dis = q_dis_asc[::-1]
    v_dis = v_grid[::-1]
    return q_dis, v_dis, q_chg, v_grid


def _make_electrochem_crystalfield_nmr_gallery(out_dir: pathlib.Path) -> None:
    """Generates 2x3 crystal-field, CV, battery/dQ/dV, 1H NMR & XPS sheet."""
    e_cv, j_cv = _synthesize_reversible_cv()
    q_dis, v_dis, q_chg, v_chg = _synthesize_lfp_battery()

    et_oac_nmr = [
        {
            "shift": 4.12,
            "integral": 2.0,
            "multiplicity": "q",
            "j_hz": 7.1,
            "label": r"$\mathrm{-OCH_2-}$",
        },
        {
            "shift": 2.05,
            "integral": 3.0,
            "multiplicity": "s",
            "j_hz": 0.0,
            "label": r"$\mathrm{CH_3CO-}$",
        },
        {
            "shift": 1.26,
            "integral": 3.0,
            "multiplicity": "t",
            "j_hz": 7.1,
            "label": r"$\mathrm{-CH_3}$",
        },
    ]

    be_grid = np.linspace(281.0, 292.5, 360)
    xps_comps = [
        {
            "label": "C-C / C-H",
            "center": 284.8,
            "amplitude": 1.00,
            "fwhm": 1.15,
        },
        {"label": "C-O", "center": 286.4, "amplitude": 0.46, "fwhm": 1.25},
        {"label": "O-C=O", "center": 288.9, "amplitude": 0.34, "fwhm": 1.25},
    ]
    net_xps = np.zeros_like(be_grid)
    for comp in xps_comps:
        net_xps += xm.broaden_peaks(
            [float(comp["center"])],
            [float(comp["amplitude"])],
            be_grid,
            fwhm=float(comp["fwhm"]),
            profile="pseudo_voigt",
            normalize="height",
        )
    cum = np.cumsum(net_xps)
    cum = (cum - cum[0]) / max(float(cum[-1] - cum[0]), 1e-12)
    rng = np.random.default_rng(7)
    raw_xps = 0.14 + 0.24 * cum + net_xps + rng.normal(0.0, 0.015, be_grid.size)

    with xm.xkcd_style("classic"):
        fig, axes = plt.subplots(2, 3, figsize=(16.5, 10.2), dpi=170)
        fig.patch.set_facecolor(xm.style.PAPER)

        xm.plot_crystal_field(
            geometry="octahedral",
            d_electrons=6,
            spin_state="high",
            delta=1.25,
            unit="eV",
            ax=axes[0, 0],
            title=r"Octahedral High-Spin $\mathrm{Fe^{2+}}\ (d^6)$",
        )
        xm.plot_cyclic_voltammetry(
            e_cv,
            j_cv,
            convention="iupac",
            potential_label="Potential E (V vs. SHE)",
            ax=axes[0, 1],
            title=r"$\mathrm{[Fe(CN)_6]^{3-/4-}}$ Cyclic Voltammetry",
        )
        xm.plot_battery_profile(
            q_dis,
            v_dis,
            charge_voltage=v_chg,
            charge_capacity=q_chg,
            plateau_shade=True,
            ax=axes[0, 2],
            title=r"$\mathrm{LiFePO_4}$ Charge / Discharge",
        )
        xm.plot_dqdv(
            v_dis,
            q_dis,
            charge_voltage=v_chg,
            charge_capacity=q_chg,
            smooth_window=5,
            annotate_peaks=1,
            ax=axes[1, 0],
            title=r"$\mathrm{LiFePO_4}$ Differential Capacity (dQ/dV)",
        )
        xm.plot_nmr(
            et_oac_nmr,
            spectrometer_mhz=400.0,
            ppm_range=(-0.3, 4.9),
            fwhm_ppm=0.016,
            show_integrals=True,
            show_trees=True,
            show_tms=True,
            nucleus="1H",
            ax=axes[1, 1],
            title=r"Ethyl Acetate $^1\mathrm{H}$ NMR (400 MHz)",
        )
        xm.plot_xps(
            be_grid,
            raw_xps,
            components=xps_comps,
            background="shirley",
            invert_x=True,
            ax=axes[1, 2],
            title="C 1s XPS (Shirley Deconvolution)",
        )
        fig.suptitle(
            "Crystal-Field Theory, Electrochemistry, Battery dQ/dV, "
            "NMR & XPS Spectroscopy",
            fontsize=15.5,
            y=0.98,
        )
        fig.tight_layout(rect=(0.0, 0.01, 1.0, 0.94), w_pad=2.1, h_pad=2.5)
        xm.save_figure(
            fig, out_dir / "electrochem_crystalfield_nmr.png", dpi=170
        )
        plt.close(fig)


def main() -> None:
    """Generates all gallery PNGs, GIFs, and static interactive assets."""
    matplotlib.use("Agg")
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Render front-facing gallery showcase figures with optional xkcd Script,
    # while keeping commercial-safe SIL OFL Patrick Hand as the library &
    # interactive 3D Studio default.
    xm.use_xkcd_script()
    try:
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
        _make_electrochem_crystalfield_nmr_gallery(_OUT_DIR)
        _make_scicomm_templates_gallery(_OUT_DIR)
        _make_readme_gifs(_OUT_DIR)
    finally:
        xm.set_default_font(None)
    _export_docs_assets()
    print(f"Generated gallery images, GIFs, and studio in {_BOOK_STATIC}")


if __name__ == "__main__":
    main()
