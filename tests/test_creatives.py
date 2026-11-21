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
"""Tests for grids, hatched shading, scrolls, periodic table, icons, graphs."""

from __future__ import annotations

import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pytest

import xkcd_matchem as xm


def test_background_grids_and_hatched_shading() -> None:
    matplotlib.use("Agg")
    assert "blueprint" in xm.list_background_styles()
    assert "crosshatch" in xm.list_shade_styles()
    assert xm.resolve_hatch("hatch") == "///"
    assert xm.resolve_hatch("cycle", index=1) == "..."
    assert xm.resolve_hatch("solid") is None

    for grid_style in ("ruled", "graph", "dotgrid", "isometric", "blueprint"):
        fig, ax = plt.subplots()
        xm.apply_background_grid(ax, style=grid_style)
        plt.close(fig)

    fig, ax = plt.subplots()
    try:
        with pytest.raises(xm.StyleError):
            xm.apply_background_grid(ax, style="invalid_grid")
    finally:
        plt.close(fig)

    ax_bar = xm.plot_bar_comparison(
        ["A", "B", "C"], [1.0, 2.0, 1.5], shade="cycle", background_grid="graph"
    )
    assert len(ax_bar.patches) >= 3
    plt.close(ax_bar.figure)

    ax_pbx = xm.plot_pourbaix(
        [("M+", [(0, 0), (7, 0), (7, 1), (0, 1)])],
        shade="crosshatch",
        background_grid="blueprint",
        preset="blueprint",
    )
    assert len(ax_pbx.patches) >= 1
    plt.close(ax_pbx.figure)


def test_scroll_cards_portrait_and_landscape() -> None:
    matplotlib.use("Agg")
    fig_p, sc_p = xm.plot_scroll_card(
        title="Portrait Notes",
        items=[("Step 1", "Parse CIF"), ("Step 2", "Render")],
        orientation="portrait",
        background_grid="ruled",
    )
    assert sc_p.orientation == "portrait"
    plt.close(fig_p)

    fig_l, sc_l = xm.plot_scroll_card(
        title="Landscape Map",
        orientation="landscape",
        background_grid="dotgrid",
    )
    assert sc_l.orientation == "landscape"
    xm.plot_bar_comparison(["X", "Y"], [2.0, 3.0], ax=sc_l.axes)
    plt.close(fig_l)

    board = xm.Board(240.0, 160.0, background_style="dotgrid")
    sc_b = board.add_scroll_card(
        10.0,
        24.0,
        100.0,
        120.0,
        orientation="portrait",
        title="Board Scroll",
        items=["Item A", "Item B"],
    )
    assert sc_b in board.scroll_cards
    plt.close(board.figure)

    with pytest.raises(xm.StyleError):
        xm.plot_scroll_card(orientation="diagonal")


def test_periodic_table_modes() -> None:
    matplotlib.use("Agg")
    ax_fam = xm.plot_periodic_table(title="Families")
    assert len(ax_fam.patches) >= 118
    plt.close(ax_fam.figure)

    ax_hm = xm.plot_periodic_table(
        element_property="electronegativity",
        missing_shade="hatch",
        title="EN Heatmap",
    )
    assert len(ax_hm.patches) >= 118
    plt.close(ax_hm.figure)

    ax_hi = xm.plot_periodic_table(
        highlight="LiFePO4",
        missing_shade="dots",
        title="LFP Spotlight",
    )
    assert len(ax_hi.patches) >= 118
    plt.close(ax_hi.figure)


def test_pymatviz_inspired_materials_plots() -> None:
    matplotlib.use("Agg")
    comps = [
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (0.5, 0.5, 0.0),
        (0.25, 0.25, 0.5),
    ]
    energies = [0.0, 0.0, 0.0, -0.5, -0.8]
    ax_t = xm.plot_ternary_phase_diagram(
        comps, energies, corners=("A", "B", "C"), labels=["", "", "", "AB", "X"]
    )
    assert len(ax_t.lines) > 0
    plt.close(ax_t.figure)

    assert xm.spacegroup_to_crystal_system(225) == "Cubic"
    assert xm.spacegroup_to_crystal_system(194) == "Hexagonal"
    assert xm.spacegroup_to_crystal_system(2) == "Triclinic"
    with pytest.raises(xm.StructureError):
        xm.spacegroup_to_crystal_system(300)

    sgs = {225: 10, 194: 8, 62: 6, 14: 4, 2: 2}
    ax_sb = xm.plot_spacegroup_distribution(sgs, mode="sunburst", shade="cycle")
    assert len(ax_sb.patches) >= 5
    plt.close(ax_sb.figure)

    ax_bar = xm.plot_spacegroup_distribution(
        [225, 225, 194, 62], mode="bar", shade="hatch", background_grid="graph"
    )
    assert len(ax_bar.patches) >= 3
    plt.close(ax_bar.figure)

    rng = np.random.default_rng(0)
    ax_h = xm.plot_histogram(
        {"PBE": rng.normal(1.2, 0.3, 40), "HSE": rng.normal(2.1, 0.4, 40)},
        shade="cycle",
        background_grid="dotgrid",
    )
    assert len(ax_h.patches) > 0
    plt.close(ax_h.figure)


def test_icons_and_graph_networks() -> None:
    matplotlib.use("Agg")
    assert len(xm.list_icons()) >= 14
    ax_sheet = xm.plot_icon_sheet()
    assert len(ax_sheet.patches) > 20
    plt.close(ax_sheet.figure)

    fig, ax = plt.subplots()
    try:
        with pytest.raises(xm.StyleError):
            xm.draw_icon(ax, "unknown_icon", (0.0, 0.0))
    finally:
        plt.close(fig)

    ax_net = xm.plot_graph_network(
        ["TiO2", "Li4Ti5O12", "BaTiO3"],
        [(0, 1, "800 C"), (0, 2, "950 C")],
        node_shape="pill",
        directed=True,
    )
    assert len(ax_net.texts) >= 3
    plt.close(ax_net.figure)

    ax_sg = xm.plot_structure_graph(
        "si_diamond.cif", layout="camera", show_cell=True
    )
    assert len(ax_sg.patches) >= 8
    plt.close(ax_sg.figure)
