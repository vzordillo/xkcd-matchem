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
"""Comprehensive tests for scientific fixes, new engines, UI, CLI, & API."""

from __future__ import annotations

import json
import pathlib
import struct
import subprocess
import sys
import types
import warnings

import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pytest

import xkcd_matchem as xm
from xkcd_matchem import cli
from xkcd_matchem.core import bonding
from xkcd_matchem.core import electronic
from xkcd_matchem.core import exceptions
from xkcd_matchem.core import geometry
from xkcd_matchem.core import isosurface
from xkcd_matchem.core import neighbors
from xkcd_matchem.core import spectroscopy
from xkcd_matchem.core import structure
from xkcd_matchem.core import trajectory
from xkcd_matchem.core import wulff
from xkcd_matchem.data import spacegroups
from xkcd_matchem.io import loader
from xkcd_matchem.io import registry
from xkcd_matchem.io.adapters import md_adapter
from xkcd_matchem.io.adapters import rdkit_adapter
from xkcd_matchem.io.native import cif
from xkcd_matchem.io.native import cp2k
from xkcd_matchem.io.native import gaussian
from xkcd_matchem.io.native import orca
from xkcd_matchem.io.native import pdb
from xkcd_matchem.io.native import phonopy
from xkcd_matchem.io.native import qe
from xkcd_matchem.io.native import sdf
from xkcd_matchem.io.native import vasp
from xkcd_matchem.io.native import xyz
from xkcd_matchem.layout import canvas
from xkcd_matchem.layout import comic
from xkcd_matchem.plots import electronic as elec_plots
from xkcd_matchem.plots import materials as mat_plots
from xkcd_matchem.plots import ptable as ptable_plots
from xkcd_matchem.plots import reactions as rxn_plots
from xkcd_matchem.plots import structure as struct_plots
from xkcd_matchem.plots import trajectory as traj_plots
from xkcd_matchem.render import callouts
from xkcd_matchem.render import interactive
from xkcd_matchem.render import primitives
from xkcd_matchem.render import structure2d
from xkcd_matchem.render import structure3d
from xkcd_matchem.style import fonts
from xkcd_matchem.style import patterns
from xkcd_matchem.style import theme


matplotlib.use("Agg")


def test_triclinic_neighbors_atoms_copy_occ_and_voigt_xrd() -> None:
    cell = geometry.cell_vectors_from_parameters(
        4.2, 4.5, 5.0, 62.0, 74.0, 68.0
    )
    frac = np.array([[0.95, 0.95, 0.95], [0.05, 0.05, 0.05]])
    atoms = structure.AtomsData.from_fractional(
        symbols=["Fe", "O"],
        frac_positions=frac,
        cell=cell,
        pbc=(True, True, True),
        occupancies=np.array([0.5, 1.0]),
        magnetic_moments=np.array([[0.0, 0.0, 2.2], [0.0, 0.0, 0.0]]),
    )
    assert atoms.formula(use_occupancies=False) == "FeO"
    assert atoms.formula(use_occupancies=True) == "Fe0.5O"

    cloned = atoms.copy()
    cloned.positions[0, 0] += 99.0
    cloned.occupancies[0] = 0.1  # type: ignore[index]
    assert atoms.occupancies is not None
    assert atoms.occupancies[0] == pytest.approx(0.5)

    rep = atoms.repeat((2, 1, 1))
    assert rep.occupancies is not None
    assert len(rep.occupancies) == 4
    assert rep.magnetic_moments is not None
    assert rep.magnetic_moments.shape == (4, 3)

    hits = neighbors.find_periodic_neighbors(
        atoms, max_cutoff=3.0, symmetric=False
    )
    assert hits
    d_exact = float(
        np.linalg.norm(
            geometry.minimum_image_vector(
                atoms.positions[1] - atoms.positions[0], cell, atoms.pbc
            )
        )
    )
    assert any(
        h.i == 0 and h.j == 1 and abs(h.distance - d_exact) < 1e-8 for h in hits
    )
    assert len(bonding.infer_bonds(atoms, scale=1.35)) >= 1

    # Voigt broadening and area normalization
    xs = np.linspace(200.0, 1300.0, 1500)
    ys = spectroscopy.broaden_peaks(
        np.array([500.0, 1000.0]),
        np.array([1.0, 2.0]),
        xs,
        fwhm=25.0,
        profile="voigt",
        normalize="area",
    )
    area = float(np.trapezoid(ys, xs))
    assert area == pytest.approx(3.0, rel=0.05)

    # Occupancy-weighted XRD alters relative peak ratios on partial sublattices
    nacl = loader.load("nacl.cif")
    nacl_half = nacl.copy()
    nacl_half.occupancies = np.array(
        [0.2 if sym == "Na" else 1.0 for sym in nacl_half.symbols]
    )
    xrd_full = spectroscopy.compute_xrd(nacl, two_theta_range=(20.0, 65.0))
    xrd_half = spectroscopy.compute_xrd(nacl_half, two_theta_range=(20.0, 65.0))
    r_full = [p.intensity for p in xrd_full.peaks[:2]]
    r_half = [p.intensity for p in xrd_half.peaks[:2]]
    assert r_full != pytest.approx(r_half, abs=1.0)


def test_parsers_spacegroups_cif_pdb_gaussian_procar_phonopy(
    tmp_path: pathlib.Path,
) -> None:
    # Registry extension before prefix + new formats
    fake_cif = tmp_path / "POSCAR.cif"
    fake_cif.write_text("data_test\n", encoding="utf-8")
    assert registry.detect_format(fake_cif).name == "cif"
    procar_hdr = tmp_path / "PROCAR_hdr"
    procar_hdr.write_text("# PROCAR lm decomposed\n", encoding="utf-8")
    assert registry.detect_format(procar_hdr).name == "procar"
    ph_hdr = tmp_path / "custom_ph.yaml"
    ph_hdr.write_text(
        "nqpoint: 10\nphonon:\n- q-position: [0, 0, 0]\n", encoding="utf-8"
    )
    assert registry.detect_format(ph_hdr).name == "phonopy_bands"

    # Spacegroup closure + multi-block CIF
    for sg_num in (1, 14, 62, 194, 225, 227, 230):
        ops = spacegroups.lookup_spacegroup_symops(sg_number=sg_num)
        assert ops is not None
        assert len(ops) == spacegroups._SPACE_GROUP_SPECS[sg_num][0]
    assert len(spacegroups.lookup_spacegroup_candidates(sg_number=227)) == 2

    multi_cif = tmp_path / "multi.cif"
    multi_cif.write_text(
        "data_alpha\n_cell_length_a 4.0\n_cell_length_b 4.0\n"
        "_cell_length_c 4.0\n_space_group_IT_number 225\n"
        "loop_\n_atom_site_label\n_atom_site_type_symbol\n"
        "_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n"
        "Cu1 Cu 0.0 0.0 0.0\n"
        "data_beta\n_cell_length_a 3.0\n_cell_length_b 3.0\n"
        "_cell_length_c 3.0\n_space_group_IT_number 1\n"
        "loop_\n_atom_site_label\n_atom_site_type_symbol\n"
        "_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n"
        "Fe1 Fe 0.0 0.0 0.0\n",
        encoding="utf-8",
    )
    blocks = cif.read_cif_all(multi_cif)
    assert len(blocks) == 2
    assert len(blocks[0]) == 4
    assert len(cif.read_cif(multi_cif, block="beta")) == 1

    # Multi-step Gaussian resets earlier orbitals & freqs
    glog = tmp_path / "opt_multi.log"
    glog.write_text(
        " Standard orientation:\n"
        " ---------------------------------------------------------------\n"
        " Center     Atomic      Atomic             Coordinates (Angstroms)\n"
        " Number     Number       Type             X           Y           Z\n"
        " ---------------------------------------------------------------\n"
        " 1 8 0 0.0 0.0 0.0\n"
        " 2 1 0 0.0 0.0 0.96\n"
        " ---------------------------------------------------------------\n"
        " SCF Done: E(RHF) = -75.00\n"
        " Alpha  occ. eigenvalues -- -10.0 -1.0\n"
        " Alpha virt. eigenvalues --   2.0\n"
        " Harmonic frequencies (cm**-1)\n"
        " Frequencies -- 999.0\n"
        " IR Inten    -- 11.0\n"
        " Alpha  occ. eigenvalues -- -0.45\n"
        " Alpha virt. eigenvalues --  0.25\n"
        " Harmonic frequencies (cm**-1)\n"
        " Frequencies -- 1650.0 3750.0\n"
        " IR Inten    -- 50.0 80.0\n",
        encoding="utf-8",
    )
    g_res = gaussian.read_gaussian_log(glog)
    assert len(g_res.metadata["orbitals"].energies) == 2
    assert len(g_res.metadata["ir_spectrum"].peaks) == 2

    # Multi-MODEL PDB with FE vs CA + CONECT
    mpdb = tmp_path / "multi.pdb"
    mpdb.write_text(
        "MODEL        1\n"
        "HETATM    1 FE   HEM A   1       0.000   0.000   0.000  1.00  0.00\n"
        "ATOM      2  CA  ALA A   2       2.000   0.000   0.000  1.00  0.00\n"
        "ENDMDL\n"
        "MODEL        2\n"
        "HETATM    1 FE   HEM A   1       0.100   0.000   0.000  1.00  0.00\n"
        "ATOM      2  CA  ALA A   2       2.100   0.000   0.000  1.00  0.00\n"
        "ENDMDL\n"
        "CONECT    1    2\n",
        encoding="utf-8",
    )
    first_pdb = pdb.read_pdb(mpdb)
    assert first_pdb.symbols == ["Fe", "C"]
    assert len(first_pdb.bonds) == 1
    assert len(pdb.read_pdb_trajectory(mpdb, max_frames=1)) == 1
    assert len(pdb.read_pdb_trajectory(mpdb)) == 2

    # xtb $coord, ORCA * xyzfile, QE/DOSCAR touching floats
    xtb_f = tmp_path / "xtb.out"
    xtb_f.write_text(
        "$coord\n  0.0 0.0 0.0 o\n  0.0 0.0 1.8 h\n$end\n"
        " TOTAL ENERGY   -5.05 Eh\n",
        encoding="utf-8",
    )
    assert cp2k.read_xtb_out(xtb_f).symbols == ["O", "H"]

    ref_xyz = tmp_path / "geom.xyz"
    ref_xyz.write_text("2\n\nH 0 0 0\nH 0 0 0.74\n", encoding="utf-8")
    oinp = tmp_path / "job.inp"
    oinp.write_text("* xyzfile 0 1 geom.xyz\n", encoding="utf-8")
    assert orca.read_orca_inp(oinp).formula() == "H2"

    qe_b = tmp_path / "touch.bands.dat"
    qe_b.write_text(
        " &plot nbnd=   1, nks=   2 /\n"
        " 0.0 0.0 0.0\n  -12.345-100.123\n"
        " 0.5 0.0 0.0\n  -11.000  -99.000\n",
        encoding="utf-8",
    )
    bs_qe = qe.read_qe_bands(qe_b)
    assert bs_qe.eigenvalues.shape == (1, 2, 1)

    dos_f = tmp_path / "DOSCAR"
    dos_f.write_text(
        " 1 1 1 0\n 0 0 0 0 0\n 1e-4\n CAR\n X\n"
        " 10.0  -10.0   2   1.500\n"
        " -5.000-100.000   1.0\n  5.000-100.000   1.0\n",
        encoding="utf-8",
    )
    assert len(vasp.read_doscar(dos_f).energies) == 2

    # VASP 3-component scaling + PROCAR + Phonopy + SDF stereo + XYZ streaming
    pos = tmp_path / "scale.poscar"
    pos.write_text(
        "Scale\n 2.0 3.0 4.0\n 1 0 0\n 0 1 0\n 0 0 1\n"
        "Si\n 1\nDirect\n 0.5 0.5 0.5\n",
        encoding="utf-8",
    )
    assert np.allclose(vasp.read_poscar(pos).positions[0], [1.0, 1.5, 2.0])

    procar = tmp_path / "PROCAR"
    procar.write_text(
        "PROCAR lm decomposed\n"
        "# of k-points:  2         # of bands:  1         # of ions:  1\n"
        " k-point    1 :    0.0 0.0 0.0     weight = 0.5\n"
        "band   1 # energy   -1.0 # occ.  2.0\n"
        "ion      s     p     d    tot\n"
        "    1  0.10  0.20  0.70  1.00\n"
        " k-point    2 :    0.5 0.0 0.0     weight = 0.5\n"
        "band   1 # energy    1.2 # occ.  0.0\n"
        "ion      s     p     d    tot\n"
        "    1  0.30  0.50  0.20  1.00\n",
        encoding="utf-8",
    )
    bs_pro = vasp.read_procar(procar)
    assert "d" in bs_pro.projections
    assert bs_pro.projections["d"][0, 0, 0] == pytest.approx(0.70)

    ph_si = phonopy.read_phonopy_bands(
        xm.get_sample_path("si_phonon_bands.yaml")
    )
    assert isinstance(ph_si, electronic.PhononBandStructure)
    assert not ph_si.has_imaginary_modes
    assert ph_si.frequencies_in("meV")[0, -1] > 40.0

    st_sdf = tmp_path / "stereo.sdf"
    st_sdf.write_text(
        "chiral\n\n\n  3  2  0  0  0  0            999 V2000\n"
        "    0.0 0.0 0.0 C   0  0\n    1.0 0.0 0.0 F   0  0\n"
        "    0.0 1.0 0.0 Cl  0  0\n"
        "  1  2  1  1\n  1  3  1  6\nM  END\n$$$$\n",
        encoding="utf-8",
    )
    s_atoms = sdf.read_sdf(st_sdf)
    assert [b.stereo for b in s_atoms.bonds] == ["wedge", "dash"]
    assert len(xyz.read_xyz(xm.get_sample_path("sn2_neb.xyz"))) == 6
    assert len(loader.load_trajectory("sn2_neb.xyz")) == 9


def test_stage3_stage4_renderers_plotters_and_engines(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    # Themes & XM_COMMERCIAL_SAFE + deprecation warning
    for p_name in ("talk_16x9", "talk_dark", "graphical_abstract"):
        assert p_name in theme.list_presets()
        assert theme.get_preset(p_name).font == "xkcd Script"

    monkeypatch.setenv("XM_COMMERCIAL_SAFE", "1")
    fonts.set_default_font(None)
    assert fonts.get_default_font() == "Patrick Hand"
    monkeypatch.delenv("XM_COMMERCIAL_SAFE")
    bundled_ttf = fonts.get_bundled_font_path("xkcd Script")
    (tmp_path / "xkcd-script.ttf").write_bytes(bundled_ttf.read_bytes())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fonts.use_xkcd_script(cache_dir=tmp_path)
        assert any(issubclass(w.category, UserWarning) for w in caught)
    fonts.set_default_font(None)

    assert primitives.is_dark_color("#181C24")
    assert primitives.muted_surface_color("#181C24") == "#252C3A"
    assert (
        primitives.ensure_readable_text_color("#111827", "#181C24") != "#111827"
    )

    # 2D stereobonds on dark background
    chiral = structure.AtomsData(
        symbols=["C", "O", "N", "H"],
        positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.4, 0.2, 0.0],
                [-1.3, 0.4, 0.0],
                [0.0, -1.1, 0.0],
            ]
        ),
        bonds=[
            structure.Bond(0, 1, 1.0, stereo="wedge"),
            structure.Bond(0, 2, 1.0, stereo="dash"),
            structure.Bond(0, 3, 1.0, stereo="wavy"),
        ],
    )
    chiral.forces = np.ones((4, 3), dtype=np.float64) * 0.4
    ax3d = structure3d.render_structure_3d(chiral, vectors="forces")
    plt.close(ax3d.figure)
    with theme.xkcd_style("talk_dark"):
        ax2d = structure2d.render_structure_2d(chiral)
        assert len(ax2d.patches) >= 1
        plt.close(ax2d.figure)

    # 3D force/spin vectors + defect callout + slab grid clipping
    si = loader.load("si_diamond.cif")
    spins = np.zeros((len(si), 3))
    spins[:, 2] = 1.0
    ax_v = struct_plots.plot_crystal(
        si, vectors=spins, vector_color="spin", boundary_atoms=True
    )
    callouts.draw_defect_site(
        ax_v, si, (0.0, 0.0, 0.0), kind="vacancy", label="V_Si"
    )
    callouts.draw_defect_site(
        ax_v, si, (0.25, 0.25, 0.25), fractional=True, kind="interstitial"
    )
    callouts.draw_defect_site(
        ax_v, si, (0.5, 0.5, 0.5), fractional=True, kind="dopant"
    )
    plt.close(ax_v.figure)

    fig_s, ax_s = plt.subplots()
    patterns.apply_background_grid(ax_s, style="grid")
    n_grid_before = len(ax_s.lines)
    struct_plots.plot_slab("tio2_slab.poscar", ax=ax_s, view="side")
    for gl in ax_s.lines[:n_grid_before]:
        assert gl.get_clip_path() is not None or gl.get_clip_on()
    plt.close(fig_s)

    # Core Isosurface engine + plot_isosurface
    nx = 10
    grid_vals = np.linspace(-2.0, 2.0, nx)
    gx, gy, gz = np.meshgrid(grid_vals, grid_vals, grid_vals, indexing="ij")
    rho = np.exp(-(gx**2 + gy**2 + gz**2))
    vgrid = electronic.VolumetricGrid(
        origin=np.array([-2.0, -2.0, -2.0]),
        voxel_vectors=np.eye(3) * (4.0 / (nx - 1)),
        data=rho,
    )
    mesh = isosurface.extract_isosurface(vgrid, level=np.exp(-1.0))
    assert len(mesh.faces) > 20
    ax_iso = elec_plots.plot_isosurface("water_homo.cube", symmetric=True)
    plt.close(ax_iso.figure)
    with pytest.raises(exceptions.StructureError):
        isosurface.extract_isosurface(
            electronic.VolumetricGrid(
                origin=np.zeros(3),
                voxel_vectors=np.eye(3),
                data=np.zeros((1, 4, 4)),
            ),
            0.5,
        )

    # Core Wulff + plot_wulff_shape
    ws = wulff.compute_wulff_shape(
        np.eye(3) * 4.0, {(1, 0, 0): 1.0}, symops="cubic"
    )
    assert len(ws.facets) == 6
    assert ws.volume == pytest.approx(8.0, rel=1e-4)
    assert ws.total_area == pytest.approx(24.0, rel=1e-4)
    ax_w = mat_plots.plot_wulff_shape(
        "si_diamond.cif", {(1, 0, 0): 1.2, (1, 1, 1): 0.9}
    )
    plt.close(ax_w.figure)
    with pytest.raises(exceptions.StructureError):
        wulff.compute_wulff_shape(np.eye(3), {(1, 0, 0): -1.0})

    # Fatbands, phonons, pressure timeseries, dark ptable/materials, rxn
    bs = electronic.BandStructure(
        k_distances=np.linspace(0.0, 1.0, 5),
        eigenvalues=np.linspace(-2.0, 2.0, 5)[:, None],
        projections={"d": np.ones((1, 5, 1))},
    )
    ax_fb = elec_plots.plot_bands(bs, fatband="d")
    assert len(ax_fb.collections) >= 1
    plt.close(ax_fb.figure)

    ph = electronic.PhononBandStructure(
        q_distances=np.linspace(0, 1, 10),
        frequencies=np.column_stack(
            [np.linspace(-1.5, 2.0, 10), np.linspace(3.0, 6.0, 10)]
        ),
        q_labels=["G", "X"],
        q_ticks=[0.0, 1.0],
    )
    ax_ph = elec_plots.plot_phonon_bands(ph, unit="cm-1")
    plt.close(ax_ph.figure)

    md = loader.load_trajectory("cu_md.lammpstrj")
    with pytest.raises(exceptions.StructureError):
        traj_plots.plot_md_timeseries(md, quantity="pressure")
    md_p = trajectory.Trajectory(
        frames=md.frames, pressures=np.linspace(1.0, 2.5, len(md))
    )
    ax_pr = traj_plots.plot_md_timeseries(md_p, quantity="pressure")
    plt.close(ax_pr.figure)

    ax_pt = ptable_plots.plot_periodic_table(
        highlight="Fe2O3", preset="talk_dark"
    )
    plt.close(ax_pt.figure)
    ax_sg = mat_plots.plot_spacegroup_distribution(
        [225, 225, 194, 62], preset="dark_chalk"
    )
    plt.close(ax_sg.figure)

    ax_rx = rxn_plots.plot_reaction_profile(
        [0.0, 0.8, 0.2, 1.1, -0.3], barrier_mode="elementary"
    )
    assert sum("Ea =" in t.get_text() for t in ax_rx.texts) == 2
    plt.close(ax_rx.figure)

    ax_sc = rxn_plots.plot_reaction_scheme(
        ["benzene.sdf", "benzene.sdf"],
        ["aspirin.xyz"],
        arrow="<=>",
        conditions_above="Ac2O",
        conditions_below="85 C",
    )
    plt.close(ax_sc.figure)


def test_stage5_exact_board_comic_studio_cli_and_lazy_api(
    tmp_path: pathlib.Path,
) -> None:
    # Exact-mm Board PNG pixel dimensions (101.6 x 50.8 mm == 4.0 x 2.0 in)
    board = canvas.Board(width_mm=101.6, height_mm=50.8, dpi=100)
    out_png = board.save(tmp_path / "exact_board.png", exact_mm=True)
    raw = out_png.read_bytes()
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    w_px, h_px = struct.unpack(">II", raw[16:24])
    assert w_px == 400
    assert h_px == 200
    plt.close(board.figure)

    # ComicStrip
    strip = comic.plot_comic_strip(
        [
            comic.ComicPanel(title="A", speech="Aha!", bubble_kind="thought"),
            comic.ComicPanel(title="B", caption="Done!"),
        ],
        title="Mini Comic",
    )
    strip.add_stick_scientist(0, pose="cheer", prop="pointer")
    strip.add_speech_bubble(1, "Nice!", kind="speech")
    strip_out = strip.save(tmp_path / "comic.png")
    assert strip_out.is_file()
    plt.close(strip.figure)

    # Studio commercial-safe font sync + sample dedup + to_camera
    fonts.set_default_font("Patrick Hand")
    try:
        html_doc = interactive.build_interactive_html("aspirin.xyz")
        assert "Patrick Hand" in html_doc
        assert "xkcd Script" not in html_doc
        studio_path = interactive.export_interactive_html(
            "si_diamond.cif",
            tmp_path / "studio.html",
            include_bundled_samples=True,
        )
        studio_text = studio_path.read_text(encoding="utf-8")
        marker = "const PAYLOAD = "
        start = studio_text.index(marker) + len(marker)
        end = studio_text.index(";\n  const ELEM_TABLE", start)
        payload = json.loads(studio_text[start:end])
        names = [s.get("source_sample") for s in payload["samples"]]
        assert names.count("si_diamond.cif") == 1
        iv = interactive.view_structure("aspirin.xyz", camera="top")
        assert isinstance(iv, interactive.InteractiveView)
        assert iv.to_camera().elevation >= 89.9
    finally:
        fonts.set_default_font(None)

    # CLI palette preservation & new subcommands
    merged = cli._parse_color_overrides(["Ti=#112233"], base_palette="jmol")
    assert isinstance(merged, dict)
    assert merged["Ti"] == "#112233"
    assert merged["O"] == "#FF0D0D"

    for argv in (
        ["bands", "si_qe_bands.dat", "-o", str(tmp_path / "b.png")],
        ["dos", "si_qe_dos.dat", "-o", str(tmp_path / "d.png")],
        ["neb", "sn2_neb.xyz", "-o", str(tmp_path / "n.png")],
        ["ptable", "-o", str(tmp_path / "p.png"), "--property", "atomic_mass"],
        ["phonon", "si_phonon_bands.yaml", "-o", str(tmp_path / "ph.png")],
    ):
        assert cli.main(argv) == 0

    # Lazy top-level __all__ completeness + __dir__ + AttributeError
    for sym in xm.__all__:
        assert hasattr(xm, sym)
    assert set(xm.__all__).issubset(set(dir(xm)))
    with pytest.raises(AttributeError):
        _ = xm.nonexistent_symbol_xyz

    for snippet in (
        "import xkcd_matchem as xm; assert len(xm.list_samples()) >= 16",
        (
            "import xkcd_matchem.data as d; "
            "assert d.CODATA2022.bohr_to_angstrom > 0"
        ),
    ):
        proc = subprocess.run(  # noqa: S603
            [sys.executable, "-c", snippet],
            check=False,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr


def test_additional_parsers_adapters_and_branches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    # VASP write_poscar, XDATCAR, EIGENVAL, CHGCAR, spin DOS/PROCAR
    si = loader.load("si_diamond.cif")
    pos_out = tmp_path / "POSCAR"
    vasp.write_poscar(si, pos_out)
    assert len(vasp.read_poscar(pos_out)) == 8

    neg_pos = tmp_path / "NEG.poscar"
    neg_pos.write_text(
        "Neg\n -64.0\n 1 0 0\n 0 1 0\n 0 0 1\n Si\n 1\n"
        "Selective dynamics\nCartesian\n 0.0 0.0 0.0 T T T\n",
        encoding="utf-8",
    )
    neg_atoms = vasp.read_poscar(neg_pos)
    assert neg_atoms.volume() == pytest.approx(64.0)

    xdat = tmp_path / "XDATCAR"
    xdat.write_text(
        "Si MD\n 1.0\n 5.4 0 0\n 0 5.4 0\n 0 0 5.4\n Si\n 2\n"
        "Direct configuration= 1\n 0.0 0.0 0.0\n 0.25 0.25 0.25\n"
        "Direct configuration= 2\n 0.01 0.0 0.0\n 0.26 0.25 0.25\n",
        encoding="utf-8",
    )
    assert len(vasp.read_xdatcar(xdat)) == 2

    eig = tmp_path / "EIGENVAL"
    eig.write_text(
        " 2 2 1 1\n 1.0 1.0 1.0 1.0 1.0\n 1.0e-4\n CAR\n Si\n"
        " 1 2 2\n\n"
        " 0.0 0.0 0.0 0.5\n 1 -2.0 1.0\n 2 2.5 0.0\n\n"
        " 0.5 0.0 0.0 0.5\n 1 -1.0 1.0\n 2 3.5 0.0\n",
        encoding="utf-8",
    )
    assert vasp.read_eigenval(eig).eigenvalues.shape == (1, 2, 2)

    chg = tmp_path / "CHGCAR"
    chg.write_text(
        "H2 Chg\n 1.0\n 4.0 0 0\n 0 4.0 0\n 0 0 4.0\n H\n 1\nDirect\n"
        " 0.0 0.0 0.0\n\n 2 2 2\n 1.0 2.0 3.0 4.0 5.0\n 6.0 7.0 8.0\n",
        encoding="utf-8",
    )
    assert vasp.read_chgcar(chg).data.shape == (2, 2, 2)

    sp_dos = tmp_path / "SPINDOS"
    sp_dos.write_text(
        " 1 1 1 0\n 0 0 0 0 0\n 1e-4\n CAR\n Fe\n"
        " 5.0 -5.0 2 0.5\n -1.0 1.2 0.8 1.0 1.0\n 1.0 0.5 0.3 1.5 1.3\n",
        encoding="utf-8",
    )
    assert vasp.read_doscar(sp_dos).spin_down_dos is not None

    sp_pro = tmp_path / "SPINPROCAR"
    sp_pro.write_text(
        "PROCAR lm decomposed\n"
        "# of k-points: 1 # of bands: 1 # of ions: 1\n"
        " k-point 1 : 0 0 0 weight = 1.0\n"
        "band 1 # energy -0.5 # occ. 1.0\nion s p d tot\n1 0.2 0.3 0.5 1.0\n"
        "# of k-points: 1 # of bands: 1 # of ions: 1\n"
        " k-point 1 : 0 0 0 weight = 1.0\n"
        "band 1 # energy 0.5 # occ. 0.0\nion s p d tot\n1 0.4 0.4 0.2 1.0\n",
        encoding="utf-8",
    )
    assert vasp.read_procar(sp_pro).eigenvalues.shape[0] == 2

    # QE out/in/dos, CP2K, Gaussian gjf/fchk, SDF/MOL2
    qe_pwo = tmp_path / "si_pw.out"
    qe_pwo.write_text(
        "     lattice parameter (alat)   =      10.2623  a.u.\n"
        "     crystal axes: (cart. coord. in units of alat)\n"
        "       a(1) = (  0.000000  0.500000  0.500000 )\n"
        "       a(2) = (  0.500000  0.000000  0.500000 )\n"
        "       a(3) = (  0.500000  0.500000  0.000000 )\n"
        "     Cartesian axes\n\n"
        "     site n. atom  positions (alat units)\n"
        "       1  Si  tau(  1) = ( 0.0000  0.0000  0.0000 )\n"
        "CELL_PARAMETERS (angstrom)\n"
        " 5.43 0.0 0.0\n 0.0 5.43 0.0\n 0.0 0.0 5.43\n"
        "ATOMIC_POSITIONS (crystal)\n"
        " Si 0.0 0.0 0.0\n Si 0.25 0.25 0.25\n\n"
        "!    total energy              =     -15.845678 Ry\n"
        "     the Fermi energy is     6.2500 ev\n",
        encoding="utf-8",
    )
    qe_out_at = qe.read_qe_out(qe_pwo)
    assert qe_out_at.is_periodic
    assert qe_out_at.metadata["fermi_energy"] == pytest.approx(6.25)

    qe_pwi = tmp_path / "si_pw.in"
    qe_pwi.write_text(
        "&system\n ibrav=0, celldm(1)=10.26\n/\n"
        "CELL_PARAMETERS (alat)\n 0 0.5 0.5\n 0.5 0 0.5\n 0.5 0.5 0\n"
        "ATOMIC_POSITIONS (crystal)\n Si 0 0 0\n",
        encoding="utf-8",
    )
    assert qe.read_qe_in(qe_pwi).is_periodic
    qe_sp = tmp_path / "sp.dos.dat"
    qe_sp.write_text(
        "# E (eV) dosup dosdw idos EFermi = 2.0 eV\n0.0 1.0 0.5 1.5\n",
        encoding="utf-8",
    )
    assert qe.read_qe_dos(qe_sp).spin_down_dos is not None

    ener = tmp_path / "md.ener"
    ener.write_text(
        "# Step Time[fs] Kin Temp Pot Cons\n"
        " 0 0.0 0.1 300.0 -10.0 -9.9\n"
        " 1 0.5 0.1 302.0 -10.1 -10.0\n",
        encoding="utf-8",
    )
    assert len(cp2k.read_cp2k_ener(ener).x) == 2

    gjf = tmp_path / "h2.gjf"
    gjf.write_text(
        "# HF/STO-3G\n\nh2\n\n0 1\nH 0 0 0\nH 0 0 0.74\n", encoding="utf-8"
    )
    assert gaussian.read_gaussian_gjf(gjf).formula() == "H2"

    fchk = tmp_path / "h2.fchk"
    fchk.write_text(
        "Title\nSP HF STO-3G\n"
        "Atomic numbers                             I   N=           2\n"
        "           1           1\n"
        "Current cartesian coordinates              R   N=           6\n"
        "  0.0 0.0 0.0 0.0 0.0 1.4\n"
        "Alpha Orbital Energies                     R   N=           2\n"
        " -0.5 0.5\n"
        "Total Energy                               R    -1.11\n",
        encoding="utf-8",
    )
    assert len(gaussian.read_gaussian_fchk(fchk)) == 2

    mol2 = tmp_path / "h2.mol2"
    mol2.write_text(
        "@<TRIPOS>MOLECULE\nh2\n 2 1\n\n@<TRIPOS>ATOM\n"
        " 1 H1 0.0 0.0 0.0 H\n 2 H2 0.74 0.0 0.0 H\n"
        "@<TRIPOS>BOND\n 1 1 2 1\n",
        encoding="utf-8",
    )
    assert len(sdf.read_mol2(mol2).bonds) == 1

    # Mocked RDKit, MDAnalysis, and MDTraj adapters
    b0 = types.SimpleNamespace(
        GetBeginAtomIdx=lambda: 0,
        GetEndAtomIdx=lambda: 1,
        GetBondTypeAsDouble=lambda: 1.0,
        GetBondDir=lambda: "BEGINWEDGE",
    )
    a0 = types.SimpleNamespace(GetSymbol=lambda: "C")
    a1 = types.SimpleNamespace(GetSymbol=lambda: "F")
    conf = types.SimpleNamespace(
        GetPositions=lambda: [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]]
    )
    fake_rdkit = types.SimpleNamespace(
        GetAtoms=lambda: [a0, a1],
        GetBonds=lambda: [b0],
        GetNumConformers=lambda: 1,
        GetConformer=lambda _cid: conf,
    )
    fake_chem = types.SimpleNamespace(
        MolFromSmiles=lambda _s: fake_rdkit,
        AddHs=lambda m: m,
    )
    fake_allchem = types.SimpleNamespace(
        ETKDGv3=lambda: types.SimpleNamespace(randomSeed=0),
        EmbedMolecule=lambda _m, _p: 0,
        MMFFOptimizeMolecule=lambda _m: None,
        Compute2DCoords=lambda _m: None,
    )
    fake_pkg = types.SimpleNamespace(Chem=fake_chem)
    fake_chem.AllChem = fake_allchem
    monkeypatch.setitem(sys.modules, "rdkit", fake_pkg)
    monkeypatch.setitem(sys.modules, "rdkit.Chem", fake_chem)
    monkeypatch.setitem(sys.modules, "rdkit.Chem.AllChem", fake_allchem)
    assert rdkit_adapter.from_rdkit(fake_rdkit).bonds[0].stereo == "wedge"
    assert rdkit_adapter.from_smiles("CF").formula() == "CF"

    class _FakeAG:
        positions = np.array([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]])
        elements = ["C", "O"]
        names = ["C1", "O1"]

        def __len__(self) -> int:
            return 2

    fake_ts = types.SimpleNamespace(
        dimensions=np.array([10.0, 10.0, 10.0, 90.0, 90.0, 90.0]), time=0.5
    )
    fake_u = types.SimpleNamespace(atoms=_FakeAG(), trajectory=[fake_ts])
    mda_traj = md_adapter.from_mdanalysis(fake_u)
    assert len(mda_traj) == 1
    assert mda_traj[0].is_periodic

    class _FakeMDT:
        xyz = np.array([[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]])
        unitcell_vectors = np.array([np.eye(3)])
        time = np.array([0.0])
        topology = types.SimpleNamespace(
            atoms=[
                types.SimpleNamespace(
                    element=types.SimpleNamespace(symbol="H")
                ),
                types.SimpleNamespace(
                    element=types.SimpleNamespace(symbol="H")
                ),
            ]
        )

        def __len__(self) -> int:
            return 1

    assert len(md_adapter.from_mdtraj(_FakeMDT())) == 1

    # Wulff symops + Comic shrug/flask + 3D modes/vectors + animation/graphs
    hex_cell = geometry.cell_vectors_from_parameters(
        3.2, 3.2, 5.2, 90.0, 90.0, 120.0
    )
    assert len(
        wulff.compute_wulff_shape(
            hex_cell, {(0, 0, 1): 1.0, (1, 0, 0): 1.2}, symops="hexagonal"
        ).facets
    ) >= 8
    assert len(
        wulff.compute_wulff_shape(
            np.eye(3) * 4.0, {(1, 0, 0): 1.0}, symops=225
        ).facets
    ) == 6
    assert len(
        wulff.compute_wulff_shape(
            np.eye(3) * 4.0,
            {
                (1, 0, 0): 1.0,
                (-1, 0, 0): 1.0,
                (0, 1, 0): 1.0,
                (0, -1, 0): 1.0,
                (0, 0, 1): 1.0,
                (0, 0, -1): 1.0,
            },
            symops="none",
        ).facets
    ) == 6

    c_strip = comic.plot_comic_strip(2)
    c_strip.add_stick_scientist(0, pose="shrug", prop="flask")
    c_strip.add_stick_scientist(1, pose="point", prop="pointer")
    plt.close(c_strip.figure)

    md_t = loader.load_trajectory("cu_md.lammpstrj")
    gif_out = traj_plots.animate_trajectory(
        trajectory.Trajectory(frames=md_t.frames[:2]),
        tmp_path / "md.gif",
        fps=4,
    )
    assert gif_out.is_file()
    ax_rm = traj_plots.plot_rmsd(
        np.arange(len(md_t), dtype=np.float64),
        md_t.rmsd(),
        title="RMSD",
    )
    plt.close(ax_rm.figure)

    md_full = trajectory.Trajectory(
        frames=md_t.frames,
        energies=np.linspace(-10.0, -9.5, len(md_t)),
        temperatures=np.linspace(300.0, 310.0, len(md_t)),
    )
    for qty in ("energy", "temperature"):
        ax_m = traj_plots.plot_md_timeseries(md_full, quantity=qty)
        plt.close(ax_m.figure)

    si_vel = si.copy()
    si_vel.velocities = np.ones((len(si_vel), 3)) * 0.5
    si_vel.magnetic_moments = np.ones(len(si_vel))
    for axes_mode in ("origin", "edges", "top_right"):
        ax_c = struct_plots.plot_crystal(
            si_vel,
            style="stick",
            crystal_axes=axes_mode,
            vectors="velocities",
            boundary_atoms=False,
        )
        plt.close(ax_c.figure)
    ax_mg = struct_plots.plot_crystal(
        si_vel,
        style="spacefill",
        vectors="magmom",
        vector_color="spin",
        boundary_atoms=False,
    )
    plt.close(ax_mg.figure)

    ax_g = xm.plot_structure_graph(
        si, layout="camera", show_cell=True, boundary_atoms=True
    )
    plt.close(ax_g.figure)
    for g_lay in ("planar", "circular", "grid"):
        ax_net = xm.plot_graph_network(
            ["A", "B", "C", "D"],
            [(0, 1, "step"), (1, 2), (2, 3)],
            layout=g_lay,
            node_shape="pill",
            directed=True,
            dashed_edges=[(2, 3)],
            title="Net",
        )
        plt.close(ax_net.figure)

    # Extra xTB/SDF/VolumetricGrid/AtomsData/primitives coverage
    xtb_xyz = tmp_path / "xtb_fin.out"
    xtb_xyz.write_text(
        " TOTAL ENERGY   -5.05 Eh\n"
        " HOMO-LUMO GAP   4.20 eV\n"
        " final structure:\n"
        "2\ncomment\nO 0.0 0.0 0.0\nH 0.0 0.0 0.96\n",
        encoding="utf-8",
    )
    xtb_at = cp2k.read_xtb_out(xtb_xyz)
    assert xtb_at.metadata["homo_lumo_gap_ev"] == pytest.approx(4.2)

    empty_ener = tmp_path / "empty.ener"
    empty_ener.write_text("# header only\n", encoding="utf-8")
    with pytest.raises(exceptions.ParseError):
        cp2k.read_cp2k_ener(empty_ener)
    bad_xtb = tmp_path / "bad.xtb"
    bad_xtb.write_text("nothing relevant\n", encoding="utf-8")
    with pytest.raises(exceptions.ParseError):
        cp2k.read_xtb_out(bad_xtb)

    space_sdf = tmp_path / "space.sdf"
    space_sdf.write_text(
        "t\n\n\n2 1 0 0 V2000\n"
        " 0.0 0.0 0.0 C\n 1.4 0.0 0.0 F\n 1 2 1 1\nM END\n",
        encoding="utf-8",
    )
    assert sdf.read_sdf(space_sdf).bonds[0].stereo == "wedge"
    trunc_sdf = tmp_path / "trunc.sdf"
    trunc_sdf.write_text("only\n", encoding="utf-8")
    with pytest.raises(exceptions.ParseError):
        sdf.read_sdf(trunc_sdf)

    vg = electronic.VolumetricGrid(
        origin=np.zeros(3),
        voxel_vectors=np.eye(3),
        data=np.arange(27.0).reshape((3, 3, 3)),
    )
    for ax_ch in ("x", "y", "z"):
        u, v, sl = vg.slice_plane(axis=ax_ch)
        assert u.shape == (3,)
        assert v.shape == (3,)
        assert sl.shape == (3, 3)
        uu, vv, pm, b2 = vg.slice_mesh(axis=ax_ch)
        assert pm.shape == (3, 3)
        assert uu.shape == vv.shape
        assert b2.shape == (2, 3)

    occ_at = structure.AtomsData(
        symbols=["C", "H", "Fe"],
        positions=np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
        ),
        occupancies=np.array([1.0, 1.0, 0.5]),
    )
    assert "Fe0.5" in occ_at.formula(use_occupancies=True)
    assert occ_at.center_of_mass().shape == (3,)
    xyz_round = tmp_path / "si_out.xyz"
    loader.save(si, xyz_round)
    assert len(loader.load_trajectory(xyz_round)) == 1
    assert len(loader.load_trajectory(xdat)) == 2
    assert len(loader.load_trajectory(pos_out)) == 1

    fig_pr, ax_pr2 = plt.subplots()
    p0 = np.array([0.0, 0.0])
    p1 = np.array([1.0, 0.0])
    for ord_v in (1.5, 2.0, 3.0):
        primitives.draw_half_bond(ax_pr2, p0, p1, "#888888", order=ord_v)
        primitives.draw_two_tone_bond(
            ax_pr2, p0, p1, "#112233", "#445566", order=ord_v
        )
    primitives.draw_polyhedron_faces(
        ax_pr2,
        np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]),
        ((0, 1, 2),),
        np.array([0.0, 0.1, 0.2]),
        facecolor="#56B4E9",
    )
    plt.close(fig_pr)

    ax_p = xm.plot_molecule("aspirin.xyz", style="spacefill")
    primitives.draw_vector_arrow_2d(
        ax_p, np.array([0.0, 0.0]), np.array([1.0, 0.5])
    )
    plt.close(ax_p.figure)

    assert cli.main(["info", "si_diamond.cif"]) == 0
    assert (
        cli.main(["xrd", "nacl.cif", "-o", str(tmp_path / "cli_xrd.png")]) == 0
    )
    assert (
        cli.main(
            [
                "render",
                "tio2_slab.poscar",
                "--palette",
                "jmol",
                "--colors",
                "Ti=#E07A5F",
                "-o",
                str(tmp_path / "cli_render.png"),
            ]
        )
        == 0
    )
