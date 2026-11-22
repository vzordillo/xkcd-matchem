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
"""Tests for v0.1.0 features: Brillouin zones, volcano/defects, COHP, Studio."""

from __future__ import annotations

import math
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pytest

import xkcd_matchem as xm
from xkcd_matchem import cli
from xkcd_matchem.core import brillouin
from xkcd_matchem.core import electronic
from xkcd_matchem.core import exceptions
from xkcd_matchem.core import geometry
from xkcd_matchem.core import isosurface
from xkcd_matchem.core import structure
from xkcd_matchem.io.native import lobster
from xkcd_matchem.render import artistic3d
from xkcd_matchem.render import interactive
from xkcd_matchem.render import structure2d
from xkcd_matchem.style import patterns


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_version_010_consistent_across_repo() -> None:
    """Ensures package metadata, changelog, citation, and docs stay 0.1.0."""
    assert xm.__version__ == "0.1.0"
    stale = "0." + "2.0"

    pyproject = (_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    citation = (_REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    changelog = (_REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    readme = (_REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert "version: 0.1.0" in citation
    assert "## [0.1.0]" in changelog
    assert "0.1.0" in readme
    assert "0.1.0" in pyproject or 'dynamic = ["version"]' in pyproject

    for target_dir in ("src", "book", "tests", ".github"):
        base = _REPO_ROOT / target_dir
        for path in base.rglob("*"):
            if path.is_file() and path.suffix in {
                ".py",
                ".md",
                ".yml",
                ".yaml",
                ".ipynb",
            }:
                text = path.read_text(encoding="utf-8", errors="ignore")
                assert stale not in text, f"Stale version found in {path}"

    for root_doc in (changelog, citation, readme, pyproject):
        assert stale not in root_doc


def test_brillouin_zone_lattices_and_plot() -> None:
    """Verifies Wigner-Seitz BZs across FCC, BCC, Hex, C-centered & generic."""
    si = xm.load("si_diamond.cif")
    bz_si = brillouin.compute_brillouin_zone(si)
    assert bz_si.lattice_system == "fcc"
    assert len(bz_si.facets) == 14
    prim_vol_si = abs(float(np.linalg.det(si.cell))) / 4.0
    assert bz_si.volume == pytest.approx(
        (2.0 * math.pi) ** 3 / prim_vol_si, rel=1e-4
    )

    bz_conv = brillouin.compute_brillouin_zone(si, use_primitive=False)
    assert bz_conv.lattice_system == "cubic"
    assert len(bz_conv.facets) == 6

    mos2 = xm.load("mos2_2h.poscar")
    bz_mos2 = brillouin.compute_brillouin_zone(mos2)
    assert bz_mos2.lattice_system == "hexagonal"
    assert len(bz_mos2.facets) == 8

    bcc_fe = structure.AtomsData(
        symbols=["Fe", "Fe"],
        positions=[[0.0, 0.0, 0.0], [1.435, 1.435, 1.435]],
        cell=2.87 * np.eye(3),
        pbc=(True, True, True),
    )
    bz_bcc = brillouin.compute_brillouin_zone(bcc_fe)
    assert bz_bcc.lattice_system == "bcc"
    assert len(bz_bcc.facets) == 12

    c_centered = structure.AtomsData(
        symbols=["Al", "Al"],
        positions=[[0.0, 0.0, 0.0], [2.0, 2.0, 0.0]],
        cell=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 6.0]],
        pbc=(True, True, True),
    )
    bz_c = brillouin.compute_brillouin_zone(c_centered)
    assert len(bz_c.facets) >= 6

    skewed = np.array([[3.1, 0.4, 0.0], [0.2, 3.6, 0.3], [0.1, 0.2, 4.2]])
    bz_gen = brillouin.compute_brillouin_zone(skewed)
    assert bz_gen.lattice_system == "general"
    assert bz_gen.volume == pytest.approx(
        (2.0 * math.pi) ** 3 / abs(float(np.linalg.det(skewed))), rel=1e-4
    )

    with pytest.raises(exceptions.StructureError):
        brillouin.compute_brillouin_zone("aspirin.xyz")  # type: ignore[arg-type]
    with pytest.raises(exceptions.StructureError):
        brillouin.compute_brillouin_zone(xm.load("aspirin.xyz"))
    with pytest.raises(exceptions.StructureError):
        brillouin.compute_brillouin_zone(np.zeros((3, 3)))

    ax1 = xm.plot_brillouin_zone(si, title="Si FCC BZ")
    ax2 = xm.plot_brillouin_zone(
        mos2,
        kpath=["G", "M", "K", "G"],
        show_axes=False,
        color="#009E73",
    )
    assert ax1 is not None
    assert ax2 is not None
    plt.close("all")


def test_volcano_and_defect_formation_plots() -> None:
    """Verifies Sabatier volcano fitting and defect envelope transitions."""
    ax_v = xm.plot_volcano(
        [-0.8, -0.4, -0.05, 0.3, 0.7],
        [-3.0, -1.4, -0.4, -1.3, -3.2],
        labels=["Ru", "Rh", "Pt", "Pd", "Au"],
        title="HER Volcano",
    )
    fig_v2, ax_v2 = plt.subplots()
    xm.plot_volcano(
        [-0.6, 0.0, 0.6],
        [1.4, 0.4, 1.5],
        left_leg=(-1.6, 0.4),
        right_leg=(1.8, 0.4),
        fit_legs=False,
        invert_y=True,
        ax=ax_v2,
    )
    assert ax_v is not None
    assert ax_v2 is not None
    assert ax_v2.yaxis_inverted()

    with pytest.raises(exceptions.StructureError):
        xm.plot_volcano([0.1], [0.2])
    with pytest.raises(exceptions.StructureError):
        xm.plot_volcano([0.1, 0.2], [0.2, 0.3], labels=["OnlyOne"])

    ax_d = xm.plot_defect_formation(
        {
            "V_Si": [(2, 2.0), (1, 2.35), (0, 3.05)],
            "NegU": [(2, 2.1), (1, 3.5), (0, 3.1)],
            "O_i": [(0, 2.4), (-1, 2.95), (-2, 3.85)],
        },
        band_gap=1.12,
        fermi_range=(-0.1, 1.2),
        title="Si Defects",
    )
    assert ax_d is not None
    with pytest.raises(exceptions.StructureError):
        xm.plot_defect_formation({})
    with pytest.raises(exceptions.StructureError):
        xm.plot_defect_formation({"V": [(0, 1.0)]}, band_gap=0.0)
    plt.close("all")


def test_lobster_cohp_reader_plotter_and_cli(tmp_path: pathlib.Path) -> None:
    """Verifies LOBSTER COHPCAR/COOPCAR parsing, plotting, and cohp CLI."""
    si_cohp = xm.load("si_cohp.lobster")
    assert isinstance(si_cohp, electronic.CohpData)
    assert len(si_cohp.energies) == 45
    assert "No.1:Si1->Si2(2.3516)" in si_cohp.cohp

    spin_path = tmp_path / "COHPCAR.lobster"
    spin_path.write_text(
        "LOBSTER spin test\n"
        "1 2\n"
        "Average\n"
        "-2.0  -0.5  -1.0  -0.3  -0.6\n"
        " 0.0  -0.2  -1.2  -0.1  -0.7\n"
        " 2.0   0.4  -0.8   0.2  -0.5\n",
        encoding="utf-8",
    )
    spin_data = lobster.read_cohpcar(spin_path)
    assert "Average" in spin_data.spin_down_cohp
    assert "Average" in spin_data.spin_down_icohp

    coop_path = tmp_path / "COOPCAR.lobster"
    coop_path.write_text(
        "COOPCAR synthetic overlap\n"
        "1 1\n"
        "No.1:C1->C2\n"
        "-1.0  0.4  0.8\n"
        " 1.0 -0.2  0.6\n",
        encoding="utf-8",
    )
    coop_data = xm.load(coop_path)
    assert isinstance(coop_data, electronic.CohpData)
    assert coop_data.is_coop is True

    ax1 = xm.plot_cohp(si_cohp, ylim=(-8.0, 4.0), title="Si -pCOHP")
    ax2 = xm.plot_cohp(
        spin_data,
        orientation="horizontal",
        xlim=(-1.5, 1.5),
        ylim=(-3.0, 3.0),
    )
    ax3 = xm.plot_cohp(coop_data, integrated=True)
    assert ax1 is not None
    assert ax2 is not None
    assert ax3 is not None

    with pytest.raises(exceptions.StructureError):
        xm.plot_cohp(si_cohp, interactions=["Missing"])

    bad_path = tmp_path / "BAD.lobster"
    bad_path.write_text("short\n1 3\nAvg\n", encoding="utf-8")
    with pytest.raises(exceptions.ParseError):
        lobster.read_cohpcar(bad_path)

    out_png = tmp_path / "si_cohp.png"
    code = cli.main(
        ["cohp", "si_cohp.lobster", "-o", str(out_png), "--ylim", "-6", "4"]
    )
    assert code == 0
    assert out_png.is_file()
    plt.close("all")


def test_studio_20_trajectory_isosurface_and_vectors(
    tmp_path: pathlib.Path,
) -> None:
    """Verifies Studio 2.0 frames, CUBE isosurfaces, and force vectors."""
    cube_payload = interactive.serialize_structure("water_homo.cube")
    assert len(cube_payload.get("isosurfaces", [])) == 2

    neb = xm.load_trajectory("sn2_neb.xyz")
    neb_payload = interactive.serialize_structure(neb)
    assert len(neb_payload.get("frames", [])) == 9

    aspirin = xm.load("aspirin.xyz")
    aspirin.forces = np.ones_like(aspirin.positions) * 0.25
    vec_payload = interactive.serialize_structure(aspirin, vectors="forces")
    assert len(vec_payload.get("vectors", [])) == len(aspirin)

    neb_html = interactive.build_interactive_html(neb)
    assert "frames" in neb_html

    html_path = tmp_path / "studio20.html"
    exported = interactive.export_interactive_html(
        "tio2_slab.poscar",
        html_path,
        include_bundled_samples=True,
        isosurface="water_homo.cube",
    )
    assert exported.is_file()
    html_str = exported.read_text(encoding="utf-8")
    assert "xm-traj-group" in html_str
    assert "xm-show-iso" in html_str
    assert "water_homo.cube" in html_str
    assert "sn2_neb.xyz" in html_str

    ax_mpl = xm.view_structure("water_homo.cube", backend="matplotlib")
    assert ax_mpl is not None
    plt.close("all")


def test_isosurface_clarity_and_mathtext_subscripts() -> None:
    """Verifies translucent depth-split isosurfaces and mathtext subscripts."""
    ax_iso = xm.plot_isosurface("water_homo.cube")
    imgs = ax_iso.images
    assert len(imgs) >= 2
    for im in imgs:
        arr = np.asarray(im.get_array())
        assert float(np.max(arr[..., 3])) <= 0.52
    assert any(im.get_zorder() < 2.0 for im in imgs)
    assert any(im.get_zorder() >= 20.0 for im in imgs)

    ax_cohp = xm.plot_cohp("si_cohp.lobster")
    assert "$E - E_F$" in ax_cohp.get_ylabel()

    ax_vol = xm.plot_volcano([-0.5, 0.0, 0.5], [-1.0, -0.2, -1.1])
    assert "$\\Delta G_{\\mathrm{ads}}$" in ax_vol.get_xlabel()

    ax_def = xm.plot_defect_formation({"V_Si": [(0, 2.0), (1, 1.5)]})
    assert "$E_F - E_{\\mathrm{VBM}}$" in ax_def.get_xlabel()
    assert "$E^f$" in ax_def.get_ylabel()
    leg = ax_def.get_legend()
    assert leg is not None
    assert "$\\mathrm{V}_{\\mathrm{Si}}$" in [
        t.get_text() for t in leg.get_texts()
    ]
    plt.close("all")


def test_artistic_3d_isosurface_shading_modes(tmp_path: pathlib.Path) -> None:
    """Verifies pen-and-ink 3D shading modes, aliases, and Studio iso_shade."""
    for mode in artistic3d.VALID_ISO_SHADES:
        ax_m = xm.plot_isosurface("water_homo.cube", shade=mode)
        assert ax_m is not None

    for alias, canonical in (
        ("hatching", "hatch"),
        ("cross_hatching", "crosshatch"),
        ("stippling", "stipple"),
        ("scribble", "scumble"),
        ("circulism", "scumble"),
        ("blending", "solid"),
    ):
        assert artistic3d.normalize_iso_shade(alias) == canonical
        assert patterns.resolve_hatch(alias) is not None or canonical == "solid"

    with pytest.raises(exceptions.StyleError):
        xm.plot_isosurface("water_homo.cube", shade="watercolor_unknown")

    payload = interactive.serialize_structure(
        "water_homo.cube", iso_shade="crosshatch"
    )
    assert payload["iso_shade"] == "crosshatch"

    ax_st = xm.view_structure(
        "water_homo.cube", backend="matplotlib", iso_shade="contour"
    )
    assert ax_st is not None

    out_html = tmp_path / "studio_shade.html"
    interactive.export_interactive_html(
        "tio2_slab.poscar", out_html, include_bundled_samples=True
    )
    html_text = out_html.read_text(encoding="utf-8")
    assert "xm-iso-shade" in html_text

    # Verify light strokes, wandering scumbles, atom occlusion & auto cam.
    raw_grid = xm.load("water_homo.cube")
    assert isinstance(raw_grid, electronic.VolumetricGrid)
    smooth = isosurface.upsample_coarse_grid(raw_grid, min_dim=36)
    max_v = float(np.max(np.abs(smooth.data)))
    meshes = [
        (isosurface.extract_isosurface(smooth, 0.35 * max_v), "#0072B2"),
        (isosurface.extract_isosurface(smooth, -0.35 * max_v), "#D55E00"),
    ]
    cam = artistic3d.auto_orbital_camera(meshes, smooth)
    assert isinstance(cam, geometry.Camera)
    ax_auto = xm.plot_isosurface("water_homo.cube", camera="auto")
    assert ax_auto is not None

    hatch_art = artistic3d.build_isosurface_art(
        meshes, smooth, cam, alpha=0.28, shade="crosshatch"
    )
    interior_lines = [b for b in hatch_art.line_batches if b[2] < 1.0]
    assert len(interior_lines) > 0
    for _, _, lw_b, al_b, _, _ in interior_lines:
        assert lw_b <= 0.85
        assert al_b <= 0.50

    scum_art = artistic3d.build_isosurface_art(
        meshes, smooth, cam, alpha=0.28, shade="scumble"
    )
    assert any(b[2] < 1.0 for b in scum_art.line_batches)

    stipple_art = artistic3d.build_isosurface_art(
        meshes, smooth, cam, alpha=0.28, shade="stipple"
    )
    assert len(stipple_art.dot_batches) > 0
    all_sz = np.concatenate([b[1] for b in stipple_art.dot_batches])
    assert float(np.max(all_sz)) <= 6.0
    assert float(np.min(all_sz)) < 1.0
    assert float(np.std(all_sz)) > 0.45
    assert all(b[2] >= 1.2 for b in stipple_art.line_batches)

    # Confirm stipple points stay outside the Oxygen atom disk.
    pivot, span, atom_xy, _ = artistic3d.iso_scene_frame(meshes, smooth, cam)
    del pivot, span
    assert atom_xy is not None
    all_pts = np.vstack([b[0] for b in stipple_art.dot_batches])
    r_o = max(0.24, min(0.58, 0.18 + 0.27 * 0.66)) * 1.08
    d_o = np.linalg.norm(all_pts - atom_xy[0][None, :], axis=1)
    assert float(np.min(d_o)) >= 0.90 * r_o

    # Verify coaxial squiggly bonds in ball-and-stick 3D plots & 2D CH3 gap.
    ax_mol = xm.plot_molecule("aspirin.xyz", style="ball_and_stick")
    assert any(len(ln.get_xdata()) > 2 for ln in ax_mol.lines)

    ax_2d = structure2d.render_structure_2d(xm.load("aspirin.xyz"))
    ch3_txt = next(t for t in ax_2d.texts if t.get_text() == "CH3")
    tx, ty = ch3_txt.get_position()
    for ln in ax_2d.lines:
        xs = np.asarray(ln.get_xdata(), dtype=np.float64)
        ys = np.asarray(ln.get_ydata(), dtype=np.float64)
        hit = (np.abs(xs - tx) < 0.20) & (np.abs(ys - ty) < 0.10)
        assert not np.any(hit)
    plt.close("all")


def test_v010_top_level_lazy_exports() -> None:
    """Verifies all new public APIs resolve via lazy top-level getattr."""
    expected = (
        "BrillouinFacet",
        "BrillouinZone",
        "CohpData",
        "compute_brillouin_zone",
        "plot_brillouin_zone",
        "plot_cohp",
        "plot_defect_formation",
        "plot_volcano",
        "read_cohpcar",
    )
    for sym in expected:
        assert sym in xm.__all__
        assert getattr(xm, sym) is not None


def test_sciviz_stroke_hierarchy_and_ink_labels() -> None:
    """Verifies 4-tier stroke hierarchy, ink labels, scheme scale & sticks."""
    sw = xm.stroke_weights("classic")
    assert sw.guide < sw.leader < sw.overlay < sw.primary <= sw.frame
    assert "stroke_weights" in xm.__all__

    # BZ interior k-path must stay thinner than host polyhedron edges.
    ax_bz = xm.plot_brillouin_zone("si_diamond.cif")
    front_poly_lw = float(ax_bz.collections[1].get_linewidth()[0])
    kpath_lw = float(ax_bz.lines[0].get_linewidth())
    assert kpath_lw < front_poly_lw
    plt.close(ax_bz.figure)

    # Defect kink labels use dark theme ink, not bright curve colors.
    ax_def = xm.plot_defect_formation(
        {
            "V_Si": [(2, 2.0), (1, 2.35), (0, 3.05)],
            "Si_i": [(1, 2.5), (0, 2.95), (-1, 3.85)],
        },
        band_gap=1.12,
    )
    kink_texts = [t for t in ax_def.texts if "/" in t.get_text()]
    assert len(kink_texts) >= 2
    for t_art in kink_texts:
        assert t_art.get_color() == xm.style.INK
    plt.close(ax_def.figure)

    # Reaction scheme uses unified coordinate system with 1:1 bond length.
    ax_rxn = xm.plot_reaction_scheme(
        ["benzene.sdf"],
        ["aspirin.xyz"],
        conditions_above="Acylation",
        conditions_below=r"$\mathrm{AlCl_3}$",
    )
    assert len(ax_rxn.child_axes) == 0
    bond_lens = [
        float(
            np.hypot(
                ln.get_xdata()[-1] - ln.get_xdata()[0],
                ln.get_ydata()[-1] - ln.get_ydata()[0],
            )
        )
        for ln in ax_rxn.lines
        if len(ln.get_xdata()) == 2
    ]
    assert any(1.25 <= length <= 1.52 for length in bond_lens)
    plt.close(ax_rxn.figure)

    # Aspect-invariant stick scientist characters & circular heads on wide axes.
    fig_s, ax_s = plt.subplots(figsize=(8.0, 3.2), dpi=100)
    for idx, (char_k, pose_k, prop_k) in enumerate(
        (
            ("cueball", "stand", "clipboard"),
            ("ponytail", "think", None),
            ("megan", "point", "pointer"),
            ("beret", "shrug", None),
            ("goggles", "cheer", "flask"),
        )
    ):
        xm.draw_stick_scientist(
            ax_s,
            xy=(0.12 + 0.18 * idx, 0.15),
            scale=0.35,
            character=char_k,
            pose=pose_k,
            facing="left" if idx % 2 else "right",
            prop=prop_k,
            speech="Check!" if idx == 1 else None,
        )
    fig_s.canvas.draw()
    ell = ax_s.patches[0]
    w_px = ell.width * ax_s.bbox.width
    h_px = ell.height * ax_s.bbox.height
    assert w_px == pytest.approx(h_px, rel=0.08)
    with pytest.raises(exceptions.StyleError):
        xm.draw_stick_scientist(ax_s, character="unknown_char")
    with pytest.raises(exceptions.StyleError):
        xm.draw_stick_scientist(ax_s, facing="upward")
    with pytest.raises(exceptions.StyleError):
        xm.draw_speech_bubble(ax_s, "hi", kind="unknown_kind")
    plt.close(fig_s)


def test_isosurface_orthographic_orbit_bounds_and_studio_shades() -> None:
    """Ensures stipple/scumble/contour stay bounded across orbit angles."""
    cube = xm.load("water_homo.cube")
    grid_up = isosurface.upsample_coarse_grid(cube, min_dim=36)
    max_abs = float(np.max(np.abs(grid_up.data)))
    level = 0.35 * max_abs
    meshes = [
        (isosurface.extract_isosurface(grid_up, level), "#0072B2"),
        (isosurface.extract_isosurface(grid_up, -level), "#D55E00"),
    ]
    cam_90 = geometry.Camera(
        azimuth=90.0, elevation=35.2643897, perspective=0.0
    )
    pivot, _, atom_xy, _ = artistic3d.iso_scene_frame(
        meshes, grid_up, cam_90
    )
    assert atom_xy is not None
    proj_pts = np.vstack(
        [
            geometry.project_points(m.vertices, cam_90, center=pivot)[0]
            for m, _ in meshes
        ]
    )
    lx0, ly0 = np.min(proj_pts, axis=0)
    lx1, ly1 = np.max(proj_pts, axis=0)

    for mode in ("stipple", "scumble", "contour", "cross_contour"):
        ax = xm.plot_isosurface(cube, camera=cam_90, shade=mode)
        for col in ax.collections:
            offsets = col.get_offsets()
            if len(offsets) > 0:
                xs, ys = offsets[:, 0], offsets[:, 1]
                assert float(np.min(xs)) >= float(lx0) - 0.05
                assert float(np.max(xs)) <= float(lx1) + 0.05
                assert float(np.min(ys)) >= float(ly0) - 0.05
                assert float(np.max(ys)) <= float(ly1) + 0.05
        for ln in ax.lines:
            if float(ln.get_linewidth()) < 1.0:
                xs = np.asarray(ln.get_xdata(), dtype=np.float64)
                ys = np.asarray(ln.get_ydata(), dtype=np.float64)
                assert float(np.min(xs)) >= float(lx0) - 0.05
                assert float(np.max(xs)) <= float(lx1) + 0.05
                assert float(np.min(ys)) >= float(ly0) - 0.05
                assert float(np.max(ys)) <= float(ly1) + 0.05
        plt.close(ax.figure)

    html_out = interactive.build_interactive_html(
        cube, iso_shade="cross_contour"
    )
    assert 'value="cross_contour"' in html_out
    assert "cross_contour" in html_out
    assert "isoShade" in html_out
