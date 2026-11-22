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
"""Tests for vector/raster export fidelity and dark-theme visual parity."""

from __future__ import annotations

from collections.abc import Callable
import pathlib

import matplotlib
from matplotlib import axes as maxes
from matplotlib import colors as mcolors
from matplotlib import figure as mfigure
from matplotlib import patches
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import pytest

import xkcd_matchem as xm
from xkcd_matchem.style import theme


_ALL_PRESETS: tuple[str, ...] = (
    "classic",
    "dark_chalk",
    "talk_dark",
    "blueprint",
)
_DARK_PRESETS: tuple[str, ...] = ("dark_chalk", "talk_dark", "blueprint")


def _composite_rgb(
    fg_color: object,
    bg_rgb: tuple[float, float, float],
    alpha_override: float | None = None,
) -> tuple[float, float, float]:
    """Alpha-composites ``fg_color`` over an opaque ``bg_rgb`` triple."""
    rgba = mcolors.to_rgba(fg_color)  # type: ignore[arg-type]
    a = rgba[3] if alpha_override is None else float(alpha_override)
    return (
        a * rgba[0] + (1.0 - a) * bg_rgb[0],
        a * rgba[1] + (1.0 - a) * bg_rgb[1],
        a * rgba[2] + (1.0 - a) * bg_rgb[2],
    )


def _outer_border_pixels(rgb_arr: np.ndarray, width: int = 2) -> np.ndarray:
    """Extracts the outer ``width``-pixel frame of an ``(H, W, 3)`` array."""
    return np.concatenate(
        [
            rgb_arr[:width, :, :].reshape(-1, 3),
            rgb_arr[-width:, :, :].reshape(-1, 3),
            rgb_arr[:, :width, :].reshape(-1, 3),
            rgb_arr[:, -width:, :].reshape(-1, 3),
        ],
        axis=0,
    )


def _assert_png_border_matches(
    fig: mfigure.Figure,
    preset_name: str,
    out_path: pathlib.Path,
    *,
    tight: bool = True,
    tol: float = 3.0,
) -> None:
    """Saves PNG via ``save_figure`` and asserts clean outer 2px borders."""
    preset_obj = theme.get_preset(preset_name)
    xm.save_figure(fig, out_path, dpi=90, tight=tight)
    assert out_path.is_file()
    with Image.open(out_path) as img:
        arr = np.asarray(img.convert("RGB"), dtype=np.float64)
    exp_rgb = np.asarray(mcolors.to_rgb(preset_obj.figure_facecolor)) * 255.0
    border = _outer_border_pixels(arr, width=2)
    max_diff = float(np.max(np.abs(border - exp_rgb[None, :])))
    assert max_diff <= tol, (
        f"Outer 2px border leaked on {preset_name}: max RGB diff = {max_diff}"
    )


def _effective_text_bg(
    ax: maxes.Axes,
    txt: matplotlib.text.Text,
) -> tuple[float, float, float]:
    """Resolves the visible background RGB directly behind a Text artist."""
    ax_rgb = mcolors.to_rgb(ax.get_facecolor())
    bp = txt.get_bbox_patch()
    if bp is not None and bp.get_visible():
        fc = bp.get_facecolor()
        if fc[3] > 0.15:
            return _composite_rgb(fc, ax_rgb)

    if txt.get_transform() == ax.transData:
        x_pt, y_pt = txt.get_position()
        best_z = -1e9
        best_rgb = ax_rgb
        for patch in ax.patches:
            if not patch.get_visible():
                continue
            fc = patch.get_facecolor()
            if fc[3] <= 0.15 or patch.get_zorder() >= txt.get_zorder():
                continue
            hit = False
            if isinstance(patch, patches.Circle):
                cx, cy = patch.center
                r = patch.radius
                hit = (x_pt - cx) ** 2 + (y_pt - cy) ** 2 <= r * r
            elif isinstance(patch, (patches.Rectangle, patches.FancyBboxPatch)):
                px, py = patch.get_x(), patch.get_y()
                pw, ph = patch.get_width(), patch.get_height()
                hit = (
                    min(px, px + pw) <= x_pt <= max(px, px + pw)
                    and min(py, py + ph) <= y_pt <= max(py, py + ph)
                )
            if hit and patch.get_zorder() >= best_z:
                best_z = float(patch.get_zorder())
                best_rgb = _composite_rgb(fc, ax_rgb)
        return best_rgb
    return ax_rgb


def _assert_texts_wcag_contrast(
    fig: mfigure.Figure,
    preset_name: str,
    *,
    min_ratio: float = 4.5,
) -> None:
    """Asserts every visible Text artist meets WCAG contrast on its canvas."""
    fig.canvas.draw()
    fig_rgb = mcolors.to_rgb(fig.get_facecolor())
    for attr in ("_suptitle", "_supxlabel", "_supylabel"):
        st = getattr(fig, attr, None)
        if st is not None and st.get_visible() and st.get_text().strip():
            fg = _composite_rgb(st.get_color(), fig_rgb, st.get_alpha())
            ratio = theme.wcag_contrast_ratio(fg, fig_rgb)
            assert ratio >= min_ratio, (
                f"[{preset_name}] figure text {st.get_text()!r} contrast "
                f"{ratio:.2f} < {min_ratio}"
            )

    for ax in fig.get_axes():
        ax_rgb = mcolors.to_rgb(ax.get_facecolor())
        candidates: list[matplotlib.text.Text] = [
            *ax.texts,
            ax.title,
            ax.xaxis.label,
            ax.yaxis.label,
        ]
        if ax.axison:
            candidates.extend(ax.get_xticklabels())
            candidates.extend(ax.get_yticklabels())
        for txt in candidates:
            if not txt.get_visible() or not txt.get_text().strip():
                continue
            bg_rgb = _effective_text_bg(ax, txt)
            fg_rgb = _composite_rgb(txt.get_color(), bg_rgb, txt.get_alpha())
            ratio = theme.wcag_contrast_ratio(fg_rgb, bg_rgb)
            assert ratio >= min_ratio, (
                f"[{preset_name}] text {txt.get_text()!r} fg={txt.get_color()} "
                f"bg={mcolors.to_hex(bg_rgb)} contrast={ratio:.2f} < "
                f"{min_ratio}"
            )

        leg = ax.get_legend()
        if leg is not None and leg.get_visible():
            frame = leg.get_frame()
            if frame is not None and frame.get_visible():
                lbg = _composite_rgb(frame.get_facecolor(), ax_rgb)
            else:
                lbg = ax_rgb
            leg_texts = [*leg.get_texts(), leg.get_title()]
            for ltxt in leg_texts:
                if not ltxt.get_visible() or not ltxt.get_text().strip():
                    continue
                fg_rgb = _composite_rgb(ltxt.get_color(), lbg, ltxt.get_alpha())
                ratio = theme.wcag_contrast_ratio(fg_rgb, lbg)
                assert ratio >= min_ratio, (
                    f"[{preset_name}] legend text {ltxt.get_text()!r} "
                    f"contrast={ratio:.2f} < {min_ratio}"
                )


def test_save_figure_png_svg_pdf_and_fonttypes(
    tmp_path: pathlib.Path,
) -> None:
    """Verifies PNG, SVG (path vs none), PDF (Type 42), and transparency."""
    matplotlib.use("Agg")
    before_pdf = matplotlib.rcParams["pdf.fonttype"]
    before_ps = matplotlib.rcParams["ps.fonttype"]
    before_svg = matplotlib.rcParams["svg.fonttype"]

    for preset_name in ("dark_chalk", "blueprint"):
        preset_obj = theme.get_preset(preset_name)
        with theme.xkcd_style(preset=preset_name):
            fig, ax = plt.subplots(figsize=(4.4, 3.2))
            ax.plot([0.0, 1.0, 2.0], [0.2, 1.1, 0.8], label="Trace")
            ax.set_title("Vector Export Title")
            ax.set_xlabel("X Axis")
            ax.set_ylabel("Y Axis")
            ax.legend(frameon=True)

        # 1. Opaque PNG (tight=True and tight=False) -> zero white border leak
        _assert_png_border_matches(
            fig, preset_name, tmp_path / f"{preset_name}_tight.png", tight=True
        )
        _assert_png_border_matches(
            fig, preset_name, tmp_path / f"{preset_name}_exact.png", tight=False
        )

        # 2. Transparent PNG -> corner pixels actually transparent (alpha == 0)
        png_tr = xm.save_figure(
            fig, tmp_path / f"{preset_name}_tr.png", dpi=90, transparent=True
        )
        with Image.open(png_tr) as img_tr:
            rgba_arr = np.asarray(img_tr.convert("RGBA"))
            assert int(rgba_arr[0, 0, 3]) == 0
            assert int(rgba_arr[-1, -1, 3]) == 0

        # 3. SVG with svg_fonttype="path" -> vector paths + embedded dark bg
        svg_path = xm.save_figure(
            fig,
            tmp_path / f"{preset_name}_path.svg",
            svg_fonttype="path",
        )
        svg_text_path = svg_path.read_text(encoding="utf-8")
        assert "<svg" in svg_text_path
        assert "<path" in svg_text_path
        assert "<text" not in svg_text_path
        assert preset_obj.figure_facecolor.lower() in svg_text_path.lower()

        # 4. SVG with svg_fonttype="none" -> editable <text> preserved
        svg_none = xm.save_figure(
            fig,
            tmp_path / f"{preset_name}_none.svg",
            svg_fonttype="none",
        )
        svg_text_none = svg_none.read_text(encoding="utf-8")
        assert "<svg" in svg_text_none
        assert "<text" in svg_text_none
        assert "Vector Export Title" in svg_text_none
        assert preset_obj.figure_facecolor.lower() in svg_text_none.lower()

        # 5. PDF with pdf_fonttype=42 and custom metadata
        pdf_out = xm.save_figure(
            fig,
            tmp_path / f"{preset_name}.pdf",
            pdf_fonttype=42,
            metadata={"Title": f"Export {preset_name}"},
        )
        pdf_bytes = pdf_out.read_bytes()
        assert pdf_bytes.startswith(b"%PDF-")
        assert len(pdf_bytes) > 1000
        assert f"Export {preset_name}".encode() in pdf_bytes
        plt.close(fig)

    assert matplotlib.rcParams["pdf.fonttype"] == before_pdf
    assert matplotlib.rcParams["ps.fonttype"] == before_ps
    assert matplotlib.rcParams["svg.fonttype"] == before_svg


def test_xkcd_style_rcparams_and_style_existing_axes_dark_parity(
    tmp_path: pathlib.Path,
) -> None:
    """Checks xkcd_style rcParams and style_existing_axes on prebuilt figs."""
    matplotlib.use("Agg")
    before_leg_fc = matplotlib.rcParams["legend.facecolor"]
    before_leg_ec = matplotlib.rcParams["legend.edgecolor"]
    before_leg_lc = matplotlib.rcParams["legend.labelcolor"]

    for preset_name in _ALL_PRESETS:
        resolved = theme.get_preset(preset_name)
        with theme.xkcd_style(preset=preset_name):
            assert matplotlib.rcParams["axes.facecolor"] == (
                resolved.axes_facecolor
            )
            assert matplotlib.rcParams["figure.facecolor"] == (
                resolved.figure_facecolor
            )
            assert matplotlib.rcParams["figure.edgecolor"] == (
                resolved.figure_facecolor
            )
            assert matplotlib.rcParams["savefig.facecolor"] == (
                resolved.figure_facecolor
            )
            assert matplotlib.rcParams["savefig.edgecolor"] == (
                resolved.figure_facecolor
            )
            assert matplotlib.rcParams["legend.facecolor"] == (
                resolved.axes_facecolor
            )
            assert matplotlib.rcParams["legend.edgecolor"] == resolved.ink_color
            assert matplotlib.rcParams["legend.labelcolor"] == (
                resolved.ink_color
            )
            assert matplotlib.rcParams["text.color"] == resolved.ink_color
            assert matplotlib.rcParams["axes.labelcolor"] == resolved.ink_color
            assert matplotlib.rcParams["xtick.color"] == resolved.ink_color
            assert matplotlib.rcParams["ytick.color"] == resolved.ink_color

    assert matplotlib.rcParams["legend.facecolor"] == before_leg_fc
    assert matplotlib.rcParams["legend.edgecolor"] == before_leg_ec
    assert matplotlib.rcParams["legend.labelcolor"] == before_leg_lc

    # Pre-created figure with legend, colorbar, and suptitle styled afterward
    for preset_name in _DARK_PRESETS:
        resolved = theme.get_preset(preset_name)
        fig, ax = plt.subplots(figsize=(5.2, 3.8))
        grid = np.linspace(0.0, 1.0, 16).reshape(4, 4)
        im = ax.imshow(grid, cmap="viridis")
        ax.plot([0.0, 3.0], [0.5, 2.5], color="#F0E442", label="Path")
        ax.legend(loc="upper right", frameon=True, title="Series")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Charge Density")
        ax.set_title("Local Panel")
        ax.set_xlabel("X (A)")
        ax.set_ylabel("Y (A)")
        fig.suptitle("Global Figure Suptitle")

        theme.style_existing_axes(ax, preset=preset_name)

        leg = ax.get_legend()
        assert leg is not None
        assert mcolors.to_hex(leg.get_frame().get_facecolor()) == (
            resolved.axes_facecolor.lower()
        )
        assert mcolors.to_hex(leg.get_frame().get_edgecolor()) == (
            resolved.ink_color.lower()
        )
        assert mcolors.to_hex(cbar.ax.yaxis.label.get_color()) == (
            resolved.ink_color.lower()
        )
        _assert_texts_wcag_contrast(fig, preset_name)
        _assert_png_border_matches(
            fig, preset_name, tmp_path / f"existing_{preset_name}.png"
        )
        plt.close(fig)


@pytest.mark.parametrize("preset_name", _ALL_PRESETS)
def test_dark_theme_parity_electrochem_ptable_graphs_traj_structure(
    preset_name: str,
    tmp_path: pathlib.Path,
) -> None:
    """Verifies border purity and text WCAG contrast across 5 plot modules."""
    matplotlib.use("Agg")
    before_fignums = set(plt.get_fignums())
    is_dark = preset_name in _DARK_PRESETS
    pot = np.linspace(-0.2, 0.8, 120)
    cur = np.sin(pot * 4.0) * 2.0
    cap = np.linspace(0.0, 180.0, 60)
    vol = 3.7 - 0.45 * (cap / 180.0) + 0.04 * np.sin(cap / 20.0)
    traj = xm.load_trajectory("sn2_neb.xyz")
    r_grid = np.linspace(0.5, 6.0, 50)
    g_vals = 1.0 + 2.2 * np.exp(-(((r_grid - 2.4) / 0.3) ** 2))

    builders: list[tuple[str, Callable[[], maxes.Axes]]] = [
        (
            "cv",
            lambda: xm.plot_cyclic_voltammetry(
                [(pot, cur), (pot, 0.75 * cur)],
                scan_labels=["Scan 1", "Scan 2"],
                preset=preset_name,
                title="CV Dark Parity",
            ),
        ),
        (
            "tafel",
            lambda: xm.plot_tafel(
                np.linspace(20.0, 140.0, 10),
                np.logspace(-2.0, 0.0, 10),
                preset=preset_name,
                title="Tafel Parity",
            ),
        ),
        (
            "battery",
            lambda: xm.plot_battery_profile(
                cap,
                vol,
                charge_voltage=vol + 0.14,
                plateau_shade=True,
                preset=preset_name,
                title="Battery Profile",
            ),
        ),
        (
            "dqdv",
            lambda: xm.plot_dqdv(
                vol,
                cap,
                charge_voltage=vol + 0.14,
                preset=preset_name,
                title="dQ/dV Parity",
            ),
        ),
        (
            "ptable_family",
            lambda: xm.plot_periodic_table(
                preset=preset_name, title="Periodic Table Families"
            ),
        ),
        (
            "ptable_heatmap",
            lambda: xm.plot_periodic_table(
                element_property="electronegativity",
                preset=preset_name,
                title="Electronegativity Map",
            ),
        ),
        (
            "ptable_highlight",
            lambda: xm.plot_periodic_table(
                highlight="LiFePO4",
                preset=preset_name,
                title="LiFePO4 Spotlight",
            ),
        ),
        (
            "graph_pill",
            lambda: xm.plot_graph_network(
                ["Anatase TiO2", "Rutile TiO2", "Brookite TiO2"],
                [(0, 1, "600 C"), (1, 2, "800 C")],
                node_shape="pill",
                directed=True,
                preset=preset_name,
                title="Polymorph Graph",
            ),
        ),
        (
            "graph_struct",
            lambda: xm.plot_structure_graph(
                "nacl.cif",
                preset=preset_name,
                title="NaCl Graph",
            ),
        ),
        (
            "graph_closed",
            lambda: xm.plot_structure_graph(
                "nacl.cif",
                boundary_atoms=True,
                preset=preset_name,
                title="NaCl Closed Graph",
            ),
        ),
        (
            "md_ts",
            lambda: xm.plot_md_timeseries(
                traj,
                quantity="rmsd",
                running_window=3,
                preset=preset_name,
                title="MD RMSD",
            ),
        ),
        (
            "rdf",
            lambda: xm.plot_rdf(
                r_grid,
                g_vals,
                preset=preset_name,
                title="Liquid RDF",
            ),
        ),
        (
            "rmsd",
            lambda: xm.plot_rmsd(
                np.linspace(0.0, 10.0, 20),
                np.linspace(0.0, 1.2, 20),
                preset=preset_name,
                title="RMSD Curve",
            ),
        ),
        (
            "mol_3d",
            lambda: xm.plot_molecule(
                "aspirin.xyz", preset=preset_name, title="Aspirin 3D"
            ),
        ),
        (
            "mol_2d",
            lambda: xm.plot_molecule(
                "aspirin.xyz",
                mode="2d",
                preset=preset_name,
                title="Aspirin 2D",
            ),
        ),
        (
            "crystal",
            lambda: xm.plot_crystal(
                "nacl.cif", preset=preset_name, title="NaCl Cell"
            ),
        ),
        (
            "slab_side",
            lambda: xm.plot_slab(
                "tio2_slab.poscar",
                view="side",
                preset=preset_name,
                title="TiO2 Slab Side",
            ),
        ),
        (
            "slab_iso",
            lambda: xm.plot_slab(
                "tio2_slab.poscar",
                view="isometric",
                preset=preset_name,
                title="TiO2 Slab Iso",
            ),
        ),
        (
            "coordination",
            lambda: xm.plot_coordination(
                "tio2_slab.poscar",
                center="Ti",
                ligand="O",
                preset=preset_name,
                title="TiO6 Polyhedra",
            ),
        ),
    ]

    for name, build_ax in builders:
        ax = build_ax()
        fig = ax.figure
        assert isinstance(fig, mfigure.Figure)
        try:
            _assert_png_border_matches(
                fig, preset_name, tmp_path / f"{preset_name}_{name}.png"
            )
            if is_dark:
                _assert_texts_wcag_contrast(fig, preset_name, min_ratio=4.5)
        finally:
            plt.close(fig)
    assert set(plt.get_fignums()) == before_fignums


@pytest.mark.parametrize("preset_name", _ALL_PRESETS)
def test_gif_animations_dark_theme_border_fidelity(
    preset_name: str,
    tmp_path: pathlib.Path,
) -> None:
    """Verifies every GIF frame keeps clean theme borders with zero leaks."""
    matplotlib.use("Agg")
    before_fignums = set(plt.get_fignums())
    preset_obj = theme.get_preset(preset_name)
    exp_rgb = np.asarray(mcolors.to_rgb(preset_obj.figure_facecolor)) * 255.0

    gif_paths = [
        xm.animate_orbit(
            "aspirin.xyz",
            tmp_path / f"orbit_{preset_name}.gif",
            n_frames=2,
            fps=6,
            preset=preset_name,
            dpi=64,
            title="Orbit",
        ),
        xm.animate_isosurface(
            "water_homo.cube",
            tmp_path / f"iso_{preset_name}.gif",
            n_frames=2,
            fps=6,
            preset=preset_name,
            dpi=64,
            title="Orbital",
        ),
        xm.animate_trajectory(
            "sn2_neb.xyz",
            tmp_path / f"traj_{preset_name}.gif",
            fps=6,
            stride=4,
            show_cell=False,
            preset=preset_name,
            dpi=64,
            title="SN2",
        ),
    ]

    for gif_path in gif_paths:
        assert gif_path.is_file()
        with Image.open(gif_path) as img:
            n_frames = getattr(img, "n_frames", 1)
            assert n_frames >= 2
            for frame_idx in range(n_frames):
                img.seek(frame_idx)
                arr = np.asarray(img.convert("RGB"), dtype=np.float64)
                border = _outer_border_pixels(arr, width=2)
                max_diff = float(np.max(np.abs(border - exp_rgb[None, :])))
                assert max_diff <= 12.0, (
                    f"{gif_path.name} frame {frame_idx} border leak: "
                    f"max_diff={max_diff:.1f}"
                )
    assert set(plt.get_fignums()) == before_fignums
