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
from xkcd_matchem.core import structure
from xkcd_matchem.io.native import lobster
from xkcd_matchem.render import interactive


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
