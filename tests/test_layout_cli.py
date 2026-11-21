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
"""Tests for physical-mm Board/Card layout, exports, and CLI subcommands."""

from __future__ import annotations

import pathlib

import matplotlib
from matplotlib import pyplot as plt

from xkcd_matchem import cli
from xkcd_matchem.layout import canvas
from xkcd_matchem.plots import spectroscopy as spec_plots
from xkcd_matchem.plots import structure as struct_plots


def test_board_layout_and_exports(tmp_path: pathlib.Path) -> None:
    matplotlib.use("Agg")
    board = canvas.Board(240.0, 140.0, title="Board Test", subtitle="Sub")
    board.add_container(6.0, 24.0, 228.0, 110.0, title="Panels")
    board.add_container(
        8.0, 26.0, 224.0, 106.0, title="Solid", solid_header=True
    )
    n_after_container = len(board.overlay.patches)
    c1 = board.add_card(10.0, 32.0, 105.0, 96.0, title="Crystal")
    assert len(board.overlay.patches) == n_after_container + 1
    c2 = board.add_card(125.0, 32.0, 105.0, 96.0, title="XRD", header_band=True)
    assert len(board.overlay.patches) == n_after_container + 3
    struct_plots.plot_crystal("nacl.cif", ax=c1.axes)
    spec_plots.plot_xrd("nacl.cif", ax=c2.axes)
    board.add_badge(205.0, 15.0, "v0.1.0")

    for ext in ("png", "svg", "pdf"):
        out = board.save(tmp_path / f"board.{ext}", dpi=120)
        assert out.is_file()
        assert out.stat().st_size > 200
    plt.close(board.figure)


def test_cli_commands(tmp_path: pathlib.Path) -> None:
    matplotlib.use("Agg")
    assert cli.main(["samples"]) == 0
    assert cli.main(["formats"]) == 0
    assert cli.main(["info", "aspirin.xyz"]) == 0
    out_mol = tmp_path / "cli_mol.png"
    assert (
        cli.main(
            [
                "render",
                "tio2_slab.poscar",
                "-o",
                str(out_mol),
                "--supercell",
                "2",
                "1",
                "1",
                "--colors",
                "Ti=#264653",
            ]
        )
        == 0
    )
    assert out_mol.is_file()
    out_view = tmp_path / "cli_view.html"
    assert (
        cli.main(
            [
                "view",
                "mos2_2h.poscar",
                "-o",
                str(out_view),
                "--supercell",
                "2",
                "1",
                "1",
                "--view",
                "perspective",
                "--colors",
                "Mo=#2A9D8F",
                "S=#E9C46A",
            ]
        )
        == 0
    )
    assert out_view.is_file()
    text_out = out_view.read_text(encoding="utf-8")
    assert "Mo4S8" in text_out
    assert "#2A9D8F" in text_out
    out_xrd = tmp_path / "cli_xrd.png"
    assert cli.main(["xrd", "nacl.cif", "-o", str(out_xrd)]) == 0
    assert out_xrd.is_file()
