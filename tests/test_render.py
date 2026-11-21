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

import pathlib

import matplotlib
from matplotlib import pyplot as plt
import pytest

from xkcd_matchem.core import exceptions
from xkcd_matchem.core import geometry
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
    # Only the 12 unit-cell box edges are drawn as Line2D (zero bonds).
    assert len(ax_no_bonds.lines) == 12
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
