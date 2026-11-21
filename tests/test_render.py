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

import matplotlib
from matplotlib import pyplot as plt
import pytest

from xkcd_matchem.core import exceptions
from xkcd_matchem.io import loader
from xkcd_matchem.plots import structure as struct_plots
from xkcd_matchem.render import callouts
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
    assert len(ax_2d.lines) >= 6
    plt.close(ax_2d.figure)

    ax_crys = struct_plots.plot_crystal(
        "si_diamond.cif", supercell=(1, 1, 1), miller_planes=[(1, 1, 1)]
    )
    assert len(ax_crys.patches) > 0
    plt.close(ax_crys.figure)

    ax_slab = struct_plots.plot_slab("tio2_slab.poscar")
    assert len(ax_slab.patches) > 0
    plt.close(ax_slab.figure)

    ax_poly = struct_plots.plot_coordination("tio2_slab.poscar", "Ti", "O")
    assert len(ax_poly.patches) > 0
    plt.close(ax_poly.figure)
