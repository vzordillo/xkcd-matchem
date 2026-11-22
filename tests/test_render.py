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
"""Tests for 3D, 2D, crystal, polyhedra, and annotation renderers."""

from __future__ import annotations

import json
import pathlib

import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pytest

from xkcd_matchem.core import exceptions
from xkcd_matchem.core import geometry
from xkcd_matchem.core import structure
from xkcd_matchem.io import loader
from xkcd_matchem.plots import structure as struct_plots
from xkcd_matchem.render import callouts
from xkcd_matchem.render import interactive
from xkcd_matchem.render import structure2d
from xkcd_matchem.render import structure3d


def test_render_3d_styles_and_annotations() -> None:
    matplotlib.use("Agg")
    aspirin = loader.load("aspirin.xyz")
    for mode in ("ball_and_stick", "spacefill", "stick"):
        ax = structure3d.render_structure_3d(aspirin, style=mode)
        assert len(ax.patches) > 0
        plt.close(ax.figure)

    d = callouts.annotate_distance(
        structure3d.render_structure_3d(aspirin), aspirin, 0, 1
    )
    assert d > 1.0
    ang = callouts.annotate_angle(plt.gca(), aspirin, 0, 1, 2)
    assert 60.0 < ang < 150.0
    callouts.draw_curved_arrow(plt.gca(), (0.0, 0.0), (1.0, 1.0), label="step")
    callouts.draw_callout(plt.gca(), "ring", (0.0, 0.0), (1.0, 1.5))
    plt.close("all")

    with pytest.raises(exceptions.StructureError):
        structure3d.render_structure_3d(aspirin, style="invalid_style")


def test_render_2d_crystal_slab_and_polyhedra() -> None:
    matplotlib.use("Agg")
    benzene = loader.load("benzene.sdf")
    ax_2d = structure2d.render_structure_2d(benzene)
    assert len(ax_2d.lines) >= 9
    plt.close(ax_2d.figure)

    aspirin = loader.load("aspirin.xyz")
    ax_asp_2d = structure2d.render_structure_2d(aspirin)
    # 14 visible heavy+polar bonds + 5 double-bond second strokes = 19 lines.
    assert len(ax_asp_2d.lines) == 19
    plt.close(ax_asp_2d.figure)

    ax_crys = struct_plots.plot_crystal(
        "si_diamond.cif",
        supercell=(1, 1, 1),
        miller_planes=[(1, 1, 1)],
        crystal_axes="corner",
    )
    assert len(ax_crys.patches) > 0
    plt.close(ax_crys.figure)

    ax_axes_only = struct_plots.plot_crystal(
        "si_diamond.cif",
        show_cell=False,
        crystal_axes="bottom_right",
    )
    assert len(ax_axes_only.texts) >= 3
    plt.close(ax_axes_only.figure)

    ax_no_bonds = struct_plots.plot_crystal("nacl.cif", show_bonds=False)
    # Only the 12 unit-cell box edges are drawn as Line2D (zero bonds),
    # and all 27 closed-cell rock-salt sites render (27 spheres + highlights).
    assert len(ax_no_bonds.lines) == 12
    assert len(ax_no_bonds.patches) >= 55
    plt.close(ax_no_bonds.figure)

    ax_slab_nb = struct_plots.plot_slab(
        "tio2_slab.poscar", view="side", show_bonds=False
    )
    assert len(ax_slab_nb.lines) == 0
    plt.close(ax_slab_nb.figure)

    ax_origin = struct_plots.plot_coordination(
        "tio2_slab.poscar",
        "Ti",
        "O",
        show_cell=False,
        crystal_axes="origin",
    )
    assert len(ax_origin.lines) == 0
    plt.close(ax_origin.figure)

    ax_slab = struct_plots.plot_slab("tio2_slab.poscar", view="side")
    assert len(ax_slab.patches) > 0
    plt.close(ax_slab.figure)

    ax_slab_iso = struct_plots.plot_slab("tio2_slab.poscar", view="isometric")
    assert len(ax_slab_iso.patches) > 0
    plt.close(ax_slab_iso.figure)

    ax_custom = struct_plots.plot_crystal(
        "si_diamond.cif",
        camera=geometry.Camera.from_preset("perspective"),
        palette={"Si": "#E07A5F"},
    )
    assert len(ax_custom.patches) > 0
    plt.close(ax_custom.figure)

    ax_poly = struct_plots.plot_coordination("tio2_slab.poscar", "Ti", "O")
    assert len(ax_poly.patches) > 0
    # Only the 12 unit-cell edges should be drawn as Line2D (no stick bonds).
    assert len(ax_poly.lines) == 12
    plt.close(ax_poly.figure)


def test_interactive_studio_and_exports(tmp_path: pathlib.Path) -> None:
    matplotlib.use("Agg")
    payload = interactive.serialize_structure(
        "tio2_slab.poscar",
        style="polyhedra",
        polyhedra=("Ti", "O"),
        palette={"Ti": "#264653"},
    )
    assert payload["custom_colors"]["Ti"] == "#264653"
    assert payload["polyhedra"]

    view = interactive.view_structure("aspirin.xyz", backend="html")
    assert isinstance(view, interactive.InteractiveView)
    assert "<iframe srcdoc=" in view._repr_html_()

    html_out = tmp_path / "studio.html"
    exported = interactive.export_interactive_html(
        "tio2_slab.poscar",
        html_out,
        include_bundled_samples=True,
    )
    assert exported.is_file()
    assert "__XM_PAYLOAD__" not in exported.read_text(encoding="utf-8")

    ax_mpl = interactive.view_structure("aspirin.xyz", backend="matplotlib")
    assert len(ax_mpl.patches) > 0
    plt.close(ax_mpl.figure)

    with pytest.raises(exceptions.StructureError):
        interactive.view_structure("aspirin.xyz", backend="invalid_backend")

    # Perspective projection aligns atom and unit-cell corner coordinates.
    cell = np.diag([5.0, 5.0, 5.0])
    box_atoms = structure.AtomsData(
        symbols=["Si", "Si"],
        positions=np.array([[0.0, 0.0, 0.0], [2.5, 2.5, 2.5]]),
        cell=cell,
        pbc=(True, True, True),
    )
    ax_persp = structure3d.render_structure_3d(box_atoms, camera="perspective")
    corner_2d, _, _ = geometry.cell_corners_projected(
        cell,
        geometry.Camera.from_preset("perspective", cell=cell),
        center=np.array([2.5, 2.5, 2.5]),
        span=geometry.scene_projection_span(
            box_atoms.positions,
            geometry.Camera.from_preset("perspective", cell=cell),
            center=np.array([2.5, 2.5, 2.5]),
            cell=cell,
        ),
    )
    atom_centers = [
        p.center
        for p in ax_persp.patches
        if hasattr(p, "center") and p.radius > 0.2
    ]
    assert any(np.allclose(c, corner_2d[0], atol=1e-9) for c in atom_centers)
    plt.close(ax_persp.figure)

    # 2D skeletal terminal carbon CH2 / CH labels (double / triple bonds).
    ethene = structure.AtomsData(
        symbols=["C", "C", "H", "H", "H", "H"],
        positions=np.array(
            [
                [-0.67, 0.0, 0.0],
                [0.67, 0.0, 0.0],
                [-1.23, 0.92, 0.0],
                [-1.23, -0.92, 0.0],
                [1.23, 0.92, 0.0],
                [1.23, -0.92, 0.0],
            ]
        ),
        bonds=[
            structure.Bond(0, 1, order=2.0),
            structure.Bond(0, 2, order=1.0),
            structure.Bond(0, 3, order=1.0),
            structure.Bond(1, 4, order=1.0),
            structure.Bond(1, 5, order=1.0),
        ],
    )
    ax_eth = structure2d.render_structure_2d(ethene)
    assert [t.get_text() for t in ax_eth.texts] == ["CH2", "CH2"]
    plt.close(ax_eth.figure)

    ethyne = structure.AtomsData(
        symbols=["C", "C"],
        positions=np.array([[-0.6, 0.0, 0.0], [0.6, 0.0, 0.0]]),
        bonds=[structure.Bond(0, 1, order=3.0)],
    )
    ax_aky = structure2d.render_structure_2d(ethyne)
    assert [t.get_text() for t in ax_aky.texts] == ["CH", "CH"]
    plt.close(ax_aky.figure)

    # Callout camera presets, vertical arrow label side, and center=False.
    fig_c, ax_c = plt.subplots()
    d_pbc = callouts.annotate_distance(ax_c, box_atoms, 0, 1, camera="top")
    assert d_pbc > 3.0
    callouts.draw_curved_arrow(
        ax_c, (0.0, 0.0), (0.0, 2.0), rad=0.25, label="up"
    )
    assert abs(ax_c.texts[-1].get_position()[0]) > 0.1
    plt.close(fig_c)

    nacl_view = interactive.view_structure("nacl.cif", backend="html")
    html_txt = nacl_view.html_str
    assert "computeSceneSpan" in html_txt
    assert "cell_edge" in html_txt
    payload_line = next(
        line for line in html_txt.splitlines() if "const PAYLOAD = " in line
    )
    embedded = json.loads(
        payload_line.split("const PAYLOAD = ", 1)[1].rstrip(";")
    )
    assert embedded["active"]["positions"][0] == pytest.approx([0.0, 0.0, 0.0])


def test_render_3d_cell_modes_vectors_polyhedra_and_edge_cases() -> None:
    """Covers cell/axes modes, polyhedra bond filtering, vectors, & guards."""
    matplotlib.use("Agg")
    empty = structure.AtomsData(symbols=[], positions=np.zeros((0, 3)))
    ax_empty = structure3d.render_structure_3d(empty)
    assert len(ax_empty.patches) == 0
    plt.close(ax_empty.figure)

    si = loader.load("si_diamond.cif")
    for cell_mode in ("none", "wireframe", "clean", "tripod"):
        ax_m = structure3d.render_structure_3d(
            si, show_cell=cell_mode, crystal_axes=True
        )
        plt.close(ax_m.figure)
    ax_f = structure3d.render_structure_3d(
        si, show_cell=True, crystal_axes=False, show_labels="none"
    )
    plt.close(ax_f.figure)

    with pytest.raises(exceptions.StructureError):
        structure3d.render_structure_3d(si, crystal_axes="invalid_axes")
    with pytest.raises(exceptions.StructureError):
        structure3d.render_structure_3d(si, vectors="unknown_vec")
    with pytest.raises(exceptions.StructureError):
        structure3d.render_structure_3d(si, vectors=np.zeros((2, 2)))

    # Missing forces vector attribute returns None cleanly; 0-norm & -z spin.
    ax_nf = structure3d.render_structure_3d(si, vectors="forces")
    plt.close(ax_nf.figure)

    spins = np.zeros((len(si), 3), dtype=np.float64)
    spins[0, 2] = -1.5
    spins[1, 2] = 1.5
    labeled_si = si.copy()
    labeled_si.atom_labels = [f"Site{i}" for i in range(len(si))]
    labeled_si.bonds = [structure.Bond(0, 1, 1.0), structure.Bond(0, 99, 1.0)]
    ax_sp = structure3d.render_structure_3d(
        labeled_si,
        vectors=spins,
        vector_color="spin",
        show_labels=True,
        miller_planes=[(0, 0, 0), (10, 1, 0)],
    )
    assert any("Site0" in t.get_text() for t in ax_sp.texts)
    assert any("(10 1 0)" in t.get_text() for t in ax_sp.texts)
    plt.close(ax_sp.figure)

    # Polyhedra style with explicit CoordinationPolyhedron list & bond skip.
    tio2 = loader.load("tio2_slab.poscar")
    polys = struct_plots.bonding.find_coordination_polyhedra(
        tio2, "Ti", "O", min_coordination=4
    )
    ax_p1 = structure3d.render_structure_3d(
        tio2, style="polyhedra", polyhedra=("Ti", "O"), show_bonds=True
    )
    ax_p2 = structure3d.render_structure_3d(
        tio2, style="polyhedra", polyhedra=polys, show_bonds=True
    )
    plt.close(ax_p1.figure)
    plt.close(ax_p2.figure)

    # Large structure (> 2000 atoms) emits UserWarning when autobonding skipped.
    big = structure.AtomsData(
        symbols=["C"] * 2005,
        positions=np.zeros((2005, 3), dtype=np.float64),
    )
    with pytest.warns(UserWarning, match="skipping automatic bond inference"):
        ax_big = structure3d.render_structure_3d(big)
    plt.close(ax_big.figure)


def test_interactive_view_vectors_isosurface_and_backends(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    """Covers Studio browser opening, vector/mesh serialization, & errors."""
    matplotlib.use("Agg")
    opened_uris: list[str] = []
    monkeypatch.setattr(
        "webbrowser.open", lambda uri: opened_uris.append(uri) or True
    )

    view = interactive.view_structure(
        "si_diamond.cif",
        backend="browser",
        output=tmp_path / "browser_out.html",
        center=True,
        supercell=(2, 1, 1),
    )
    assert isinstance(view, interactive.InteractiveView)
    assert (tmp_path / "browser_out.html").is_file()
    assert len(opened_uris) == 1

    tmp_open = view.open_in_browser(path=None)
    assert tmp_open.is_file()
    tmp_open.unlink(missing_ok=True)

    saved_html = tmp_path / "saved_view.html"
    interactive.view_structure(
        "aspirin.xyz", backend="html", output=saved_html
    )
    assert saved_html.is_file()

    # Matplotlib backend branches for periodic crystal & coordination polyhedra.
    ax_c = interactive.view_structure("si_diamond.cif", backend="matplotlib")
    plt.close(ax_c.figure)
    ax_p = interactive.view_structure(
        "tio2_slab.poscar", backend="matplotlib", polyhedra=("Ti", "O")
    )
    plt.close(ax_p.figure)

    # Vector resolution: velocities, 1D/2D magmom, None, explicit array, errors.
    mol = loader.load("aspirin.xyz")
    mol.velocities = np.ones((len(mol), 3)) * 0.2
    mol.magnetic_moments = np.ones(len(mol)) * 0.5
    p_vel = interactive.serialize_structure(mol, vectors="velocities")
    p_mag1 = interactive.serialize_structure(mol, vectors="magmom")
    mol.magnetic_moments = np.ones((len(mol), 3)) * 0.5
    p_mag2 = interactive.serialize_structure(mol, vectors="spin")
    p_none = interactive.serialize_structure(mol, vectors="forces")
    p_arr = interactive.serialize_structure(
        mol, vectors=np.ones((len(mol), 3))
    )
    assert len(p_vel["vectors"]) == len(mol)
    assert len(p_mag1["vectors"]) == len(mol)
    assert len(p_mag2["vectors"]) == len(mol)
    assert "vectors" not in p_none
    assert len(p_arr["vectors"]) == len(mol)

    with pytest.raises(exceptions.StructureError):
        interactive.serialize_structure(mol, vectors="invalid_vec")
    with pytest.raises(exceptions.StructureError):
        interactive.serialize_structure(mol, vectors=np.ones((2, 3)))

    # Trajectory frames with forces + explicit IsosurfaceMesh + empty mesh.
    f0 = mol.copy()
    f0.forces = np.ones((len(f0), 3)) * 0.1
    f1 = f0.copy()
    traj = interactive.trajectory.Trajectory(frames=[f0, f1])
    p_traj = interactive.serialize_structure(traj, vectors="forces")
    assert "vectors" in p_traj["frames"][0]

    empty_mesh = interactive.isosurf_mod.IsosurfaceMesh(
        vertices=np.zeros((0, 3)),
        faces=np.zeros((0, 3), dtype=np.int64),
        normals=np.zeros((0, 3)),
        level=0.1,
    )
    p_empty_iso = interactive.serialize_structure(mol, isosurface=empty_mesh)
    assert "isosurfaces" not in p_empty_iso

    nonempty_mesh = interactive.isosurf_mod.IsosurfaceMesh(
        vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
        normals=np.array([[0.0, 0.0, 1.0]]),
        level=0.1,
    )
    p_mesh = interactive.serialize_structure(mol, isosurface=nonempty_mesh)
    assert len(p_mesh["isosurfaces"]) == 1
    with pytest.raises(exceptions.StructureError):
        interactive.view_structure(
            mol, backend="matplotlib", isosurface=nonempty_mesh
        )
    with pytest.raises(exceptions.StructureError):
        interactive.serialize_structure(mol, isosurface="si_qe_bands.dat")

    # Input coercion errors & custom element palette resolution.
    with pytest.raises(exceptions.StructureError):
        interactive.serialize_structure(
            interactive.trajectory.Trajectory(frames=[])
        )
    with pytest.raises(exceptions.StructureError):
        interactive.serialize_structure(
            interactive.electronic.VolumetricGrid(
                origin=np.zeros(3),
                voxel_vectors=np.eye(3),
                data=np.zeros((2, 2, 2)),
                atoms=None,
            )
        )
    with pytest.raises(exceptions.StructureError):
        interactive.serialize_structure("si_qe_bands.dat")

    assert interactive._build_font_css("NonBundledFont") == ""
    mol_src = mol.copy()
    mol_src.metadata["source"] = str(tmp_path / "aspirin.xyz")
    assert interactive._active_sample_filename(mol_src) == "aspirin.xyz"

