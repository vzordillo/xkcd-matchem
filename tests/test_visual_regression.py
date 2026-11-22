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
"""Visual layout auditing, anti-collision, and perceptual regression tests."""

from __future__ import annotations

from collections.abc import Callable
import functools
import importlib.util
import pathlib
import sys
from typing import Any

import matplotlib
from matplotlib import figure as mfigure
from matplotlib import image as mimage
from matplotlib import pyplot as plt
import numpy as np
import pytest

import xkcd_matchem as xm
from xkcd_matchem.core import spectroscopy as spec_core
from xkcd_matchem.style import layout


matplotlib.use("Agg")


def _make_xps_figure(preset: str) -> mfigure.Figure:
    be = np.linspace(282.0, 292.0, 140)
    y_xps = (
        110.0
        + 420.0 * np.exp(-0.5 * ((be - 284.8) / 0.55) ** 2)
        + 210.0 * np.exp(-0.5 * ((be - 286.6) / 0.60) ** 2)
    )
    comps = [
        {
            "center": 284.8,
            "amplitude": 420.0,
            "fwhm": 1.2,
            "label": "C-C",
        },
        {
            "center": 286.6,
            "amplitude": 210.0,
            "fwhm": 1.3,
            "label": "C-O",
        },
    ]
    return xm.plot_xps(be, y_xps, components=comps, preset=preset).figure


def _make_molecule_2d_figure(preset: str) -> mfigure.Figure:
    bg_hex = xm.style.get_preset(preset).axes_facecolor
    pal = {
        "O": layout.ensure_wcag_contrast("#E74C3C", bg_hex, min_contrast=4.0),
    }
    with xm.xkcd_style(preset=preset):
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        xm.plot_molecule(
            "aspirin.xyz", mode="2d", ax=ax, palette=pal, preset=preset
        )
        xm.style_existing_axes(ax, preset=preset)
        ax.set_axis_off()
    return fig


def _make_charts_figure(preset: str) -> mfigure.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    xm.plot_parity(
        [0.0, 0.8, 1.6, 2.4],
        [0.1, 0.75, 1.7, 2.3],
        ax=axes[0],
        preset=preset,
    )
    xm.plot_bar_comparison(
        ["PBE", "SCAN", "HSE06"],
        [1.12, 1.38, 1.65],
        ax=axes[1],
        preset=preset,
    )
    xm.plot_pipeline_diagram(
        [("Relax", "DFT"), ("Screen", "Hull")],
        ax=axes[2],
        preset=preset,
    )
    fig.tight_layout()
    return fig


def _make_cv_figure(preset: str) -> mfigure.Figure:
    fwd = np.linspace(0.0, 0.8, 90)
    rev = fwd[::-1]
    e_cv = np.concatenate([fwd, rev])
    j_cv = np.concatenate(
        [
            3.2 * np.exp(-((fwd - 0.43) / 0.055) ** 2),
            -3.1 * np.exp(-((rev - 0.36) / 0.055) ** 2),
        ]
    )
    return xm.plot_cyclic_voltammetry(e_cv, j_cv, preset=preset).figure


def _make_struct_3d_figure(
    sample: str, is_crystal: bool, preset: str
) -> mfigure.Figure:
    with xm.xkcd_style(preset=preset):
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        if is_crystal:
            xm.plot_crystal(sample, ax=ax, preset=preset)
        else:
            xm.plot_molecule(sample, ax=ax, preset=preset)
    return fig


def _build_plotter_figure(plotter_key: str, preset: str) -> mfigure.Figure:
    """Constructs one representative core plotter Figure under ``preset``."""
    vols = np.linspace(36.0, 44.0, 7)
    dispatch: dict[str, Callable[[], mfigure.Figure]] = {
        "molecule_3d": lambda: _make_struct_3d_figure(
            "aspirin.xyz", False, preset
        ),
        "molecule_2d": lambda: _make_molecule_2d_figure(preset),
        "crystal": lambda: _make_struct_3d_figure(
            "si_diamond.cif", True, preset
        ),
        "bands_dos": lambda: xm.plot_bands_dos(
            xm.load("si_qe_bands.dat"),
            xm.load("si_qe_dos.dat"),
            preset=preset,
        )[0],
        "xrd": lambda: xm.plot_xrd("si_diamond.cif", preset=preset).figure,
        "nmr": lambda: xm.plot_nmr(
            [1.26, 2.04, 4.12],
            couplings_hz=[7.1, 0.0, 7.1],
            multiplicities=["t", "s", "q"],
            integrals=[3.0, 3.0, 2.0],
            labels=["CH3", "COCH3", "OCH2"],
            preset=preset,
        ).figure,
        "xps": lambda: _make_xps_figure(preset),
        "phase_diagram": lambda: xm.plot_phase_diagram_2d(
            [0.0, 0.25, 0.5, 0.75, 1.0],
            [0.0, -0.32, -0.45, -0.10, 0.0],
            ["A", "A3B", "AB", "AB3", "B"],
            preset=preset,
        ).figure,
        "volcano": lambda: xm.plot_volcano(
            [-0.85, -0.42, -0.05, 0.32, 0.75],
            [-2.6, -1.2, -0.35, -1.15, -2.5],
            labels=["Ru", "Rh", "Pt", "Pd", "Au"],
            preset=preset,
        ).figure,
        "eos": lambda: xm.plot_eos(
            vols, -10.5 + 0.06 * (vols - 40.0) ** 2, preset=preset
        ).figure,
        "pourbaix": lambda: xm.plot_pourbaix(
            [
                ("Fe2+", [(0.0, -0.6), (6.5, -0.6), (6.5, 0.7), (0.0, 0.7)]),
                ("Fe2O3", [(6.5, -0.6), (14.0, -0.6), (14.0, 0.7), (6.5, 0.7)]),
            ],
            preset=preset,
        ).figure,
        "defects": lambda: xm.plot_defect_formation(
            {
                "V_Si": [(2, 2.0), (1, 2.35), (0, 3.05)],
                "O_i": [(0, 2.4), (-1, 2.95)],
            },
            band_gap=1.12,
            preset=preset,
        ).figure,
        "reaction_profile": lambda: xm.plot_reaction_profile(
            [0.0, 0.78, -0.32, 0.54, -0.85],
            ["R", "TS1", "Int", "TS2", "P"],
            preset=preset,
        ).figure,
        "neb": lambda: xm.plot_neb(
            xm.load_trajectory("sn2_neb.xyz"), preset=preset
        ).figure,
        "reaction_scheme": lambda: xm.plot_reaction_scheme(
            "benzene.sdf",
            "aspirin.xyz",
            conditions_above="Ac2O / H+",
            conditions_below="80 C",
            preset=preset,
        ).figure,
        "crystal_field": lambda: xm.plot_crystal_field(
            "octahedral",
            d_electrons=6,
            spin_state="high",
            delta=2.1,
            preset=preset,
        ).figure,
        "electrochem_cv": lambda: _make_cv_figure(preset),
        "charts_bar_parity_pipe": lambda: _make_charts_figure(preset),
        "comic": lambda: xm.create_mechanism_comic(
            title="SN2 Walden Inversion", preset=preset
        ).figure,
    }
    return dispatch[plotter_key]()


_CORE_PLOTTER_KEYS: tuple[str, ...] = (
    "molecule_3d",
    "molecule_2d",
    "crystal",
    "bands_dos",
    "xrd",
    "nmr",
    "xps",
    "phase_diagram",
    "volcano",
    "eos",
    "pourbaix",
    "defects",
    "reaction_profile",
    "neb",
    "reaction_scheme",
    "crystal_field",
    "electrochem_cv",
    "charts_bar_parity_pipe",
    "comic",
)


def test_style_layout_exports_and_1d_anti_collision() -> None:
    expected_exports = {
        "LayoutAuditReport",
        "VisualFingerprint",
        "audit_figure_layout",
        "avoid_text_overlaps",
        "ensure_wcag_contrast",
        "figure_visual_fingerprint",
        "relative_luminance",
        "resolve_1d_label_positions",
        "visual_similarity",
        "wcag_contrast_ratio",
    }
    assert expected_exports.issubset(set(xm.style.__all__))
    for name in expected_exports:
        assert getattr(xm.style, name) is getattr(layout, name)

    empty_x, empty_y = layout.resolve_1d_label_positions([], [], 0.2, 1.0)
    assert empty_x.size == 0
    assert empty_y.size == 0

    with pytest.raises(xm.StyleError):
        layout.resolve_1d_label_positions([1.0, 2.0], [1.0], 0.2, 1.0)

    xs_in = np.array([0.50, 0.51, 0.52, 0.53, 0.95])
    ys_in = np.zeros_like(xs_in)
    rx, ry = layout.resolve_1d_label_positions(
        xs_in, ys_in, min_dx=0.15, y_step=10.0, max_tiers=3, x_bounds=(0.0, 1.0)
    )
    assert rx.shape == xs_in.shape
    assert np.all((rx >= 0.0) & (rx <= 1.0))
    assert len(set(np.round(ry[:4], 4))) == 3
    assert ry[4] == pytest.approx(0.0)


def test_avoid_text_overlaps_and_audit_detectors() -> None:
    with xm.xkcd_style("classic"):
        fig, ax = plt.subplots(figsize=(5.5, 4.0))
        ax.plot([0.0, 1.0], [0.0, 1.0])
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        for idx in range(5):
            ax.text(0.5, 0.5, f"Overlapping Label {idx}", ha="center")
        xm.style.apply_xkcd_spines(ax, preset="classic")
        before = layout.audit_figure_layout(fig)
        assert len(before.text_overlaps) >= 4
        assert not before.passed

        moved = layout.avoid_text_overlaps(ax, clamp_to_axes=True)
        assert moved >= 4
        after = layout.audit_figure_layout(fig)
        assert not after.text_overlaps
        assert after.passed
        plt.close(fig)

    # Annotation coordinate modes + thin leader attachment.
    with xm.xkcd_style("classic"):
        fig_a, ax_a = plt.subplots(figsize=(5.5, 4.0))
        ax_a.set_xlim(0.0, 1.0)
        ax_a.set_ylim(0.0, 1.0)
        ax_a.annotate(
            "OffsetPt A",
            xy=(0.4, 0.4),
            xytext=(0.0, 4.0),
            textcoords="offset points",
        )
        ax_a.annotate(
            "OffsetPt B",
            xy=(0.4, 0.4),
            xytext=(0.0, 4.5),
            textcoords="offset points",
        )
        ax_a.annotate(
            "OffsetPx C",
            xy=(0.4, 0.4),
            xytext=(0.0, 5.0),
            textcoords="offset pixels",
        )
        ax_a.annotate(
            "DataCoord D",
            xy=(0.7, 0.7),
            xytext=(0.7, 0.72),
            textcoords="data",
        )
        ax_a.annotate(
            "DataCoord E",
            xy=(0.7, 0.7),
            xytext=(0.7, 0.721),
            textcoords="data",
        )
        ax_a.annotate(
            "AxesFrac F",
            xy=(0.2, 0.8),
            xytext=(0.2, 0.8),
            textcoords="axes fraction",
        )
        ax_a.annotate(
            "AxesFrac G",
            xy=(0.2, 0.8),
            xytext=(0.2, 0.802),
            textcoords="axes fraction",
        )
        xm.style.apply_xkcd_spines(ax_a, preset="classic")
        layout.avoid_text_overlaps(
            ax_a, clamp_to_axes=True, draw_leaders=True
        )
        rep_a = layout.audit_figure_layout(fig_a)
        assert not rep_a.text_overlaps
        plt.close(fig_a)

    # Verify audit catches clipping, dark border leaks, and low contrast.
    fig_bad, ax_bad = plt.subplots(figsize=(4.8, 3.6))
    fig_bad.patch.set_facecolor("#181C24")
    fig_bad.patch.set_edgecolor("#FFFFFF")
    fig_bad.patch.set_linewidth(6.0)
    ax_bad.set_facecolor("#181C24")
    ax_bad.set_axis_off()
    # Deliberate white border rectangle along figure edges + low-contrast text.
    border_rect = matplotlib.patches.Rectangle(
        (0.0, 0.0),
        1.0,
        1.0,
        transform=fig_bad.transFigure,
        fill=False,
        edgecolor="#FFFFFF",
        linewidth=8.0,
    )
    fig_bad.patches.append(border_rect)
    ax_bad.text(0.5, 0.5, "Murky Dark Gray", color="#232834", ha="center")
    ax_bad.text(
        0.5,
        0.25,
        "Haloed Ink",
        color="#111111",
        ha="center",
        bbox={"facecolor": "#FFFFFF", "edgecolor": "none", "alpha": 0.94},
    )
    ax_bad.text(3.2, 3.2, "Way Off Canvas", color="#F2EFE9")
    rep_bad = layout.audit_figure_layout(fig_bad)
    assert not rep_bad.passed
    assert rep_bad.border_leak_ratio > 0.05
    assert "Way Off Canvas" in rep_bad.clipped_texts
    low_names = {name for name, _ in rep_bad.low_contrast_texts}
    assert "Murky Dark Gray" in low_names
    assert "Haloed Ink" not in low_names

    rep_noborder = layout.audit_figure_layout(
        fig_bad, check_dark_border=False
    )
    assert rep_noborder.border_leak_ratio == 0.0
    plt.close(fig_bad)


@pytest.mark.parametrize("preset", ["classic", "dark_chalk", "blueprint"])
@pytest.mark.parametrize("plotter_key", _CORE_PLOTTER_KEYS)
def test_core_plotters_pass_layout_audit_light_and_dark(
    plotter_key: str,
    preset: str,
) -> None:
    fig = _build_plotter_figure(plotter_key, preset)
    try:
        report = layout.audit_figure_layout(fig)
        assert not report.text_overlaps, (
            f"{plotter_key} [{preset}] overlaps: {report.text_overlaps}"
        )
        assert not report.clipped_texts, (
            f"{plotter_key} [{preset}] clipped: {report.clipped_texts}"
        )
        assert report.border_leak_ratio <= 0.01, (
            f"{plotter_key} [{preset}] border leak: {report.border_leak_ratio}"
        )
        assert not report.low_contrast_texts, (
            f"{plotter_key} [{preset}] low contrast: "
            f"{report.low_contrast_texts}"
        )
        assert report.ink_ratio >= 0.005
        assert report.passed
    finally:
        plt.close(fig)


def test_deterministic_visual_fingerprint_and_similarity() -> None:
    xs = [-0.8, -0.4, 0.0, 0.4, 0.8]
    ys = [-2.4, -1.1, -0.3, -1.2, -2.5]
    lbls = ["Ru", "Rh", "Pt", "Pd", "Au"]

    fig_c1 = xm.plot_volcano(xs, ys, labels=lbls, preset="classic").figure
    fig_c2 = xm.plot_volcano(xs, ys, labels=lbls, preset="classic").figure
    fig_dc = xm.plot_volcano(xs, ys, labels=lbls, preset="dark_chalk").figure
    fig_bp = xm.plot_volcano(xs, ys, labels=lbls, preset="blueprint").figure
    fig_blank, ax_blank = plt.subplots(figsize=(6.4, 4.4))
    ax_blank.set_axis_off()
    fig_blank.patch.set_sketch_params(0.0, 0.0, 0.0)

    try:
        fp_c1 = layout.figure_visual_fingerprint(fig_c1)
        fp_c2 = layout.figure_visual_fingerprint(fig_c2)
        fp_dc = layout.figure_visual_fingerprint(fig_dc)
        fp_bp = layout.figure_visual_fingerprint(fig_bp)
        fp_32 = layout.figure_visual_fingerprint(fig_c1, size=(32, 32))
        fp_blank = layout.figure_visual_fingerprint(fig_blank)

        assert fp_c1.ink_ratio >= 0.015
        assert fp_dc.ink_ratio >= 0.015
        assert fp_bp.ink_ratio >= 0.015
        assert fp_blank.ink_ratio == 0.0

        assert layout.visual_similarity(fp_c1, fp_c2) >= 0.99
        assert layout.visual_similarity(fp_c1, fp_32) >= 0.90
        assert layout.visual_similarity(fp_c1, fp_dc) < 0.45
        assert layout.visual_similarity(fp_c1, fp_bp) < 0.45
        assert layout.visual_similarity(fp_dc, fp_bp) < 0.85
        assert layout.visual_similarity(fp_c1, fp_blank) < 0.82

        with pytest.raises(xm.StyleError):
            layout.figure_visual_fingerprint(fig_c1, size=(2, 2))
    finally:
        for f_obj in (fig_c1, fig_c2, fig_dc, fig_bp, fig_blank):
            plt.close(f_obj)


def test_stress_20_clustered_peaks_and_dense_points() -> None:
    # 1. 20 tightly spaced catalysis volcano points near the apex.
    xs_dense = np.linspace(-0.32, 0.32, 20)
    ys_dense = -0.35 - 2.2 * np.abs(xs_dense)
    lbls_dense = [f"M{k}" for k in range(20)]
    fig_v = xm.plot_volcano(
        xs_dense, ys_dense, labels=lbls_dense, preset="classic"
    ).figure
    try:
        rep_v = layout.audit_figure_layout(fig_v)
        assert not rep_v.text_overlaps
        assert rep_v.border_leak_ratio <= 0.01
    finally:
        plt.close(fig_v)

    # 2. 20 tightly clustered binary hull compositions.
    comp_20 = np.concatenate(
        [[0.0], np.linspace(0.38, 0.62, 18), [1.0]]
    )
    eng_20 = -0.45 * (1.0 - (2.0 * comp_20 - 1.0) ** 2) + np.where(
        np.arange(20) % 2 == 0, 0.0, 0.08
    )
    eng_20[0] = 0.0
    eng_20[-1] = 0.0
    fig_pd = xm.plot_phase_diagram_2d(
        comp_20,
        eng_20,
        labels=[f"P{k}" for k in range(20)],
        preset="dark_chalk",
    ).figure
    try:
        rep_pd = layout.audit_figure_layout(fig_pd)
        assert not rep_pd.text_overlaps
        assert not rep_pd.low_contrast_texts
        assert rep_pd.border_leak_ratio <= 0.01
    finally:
        plt.close(fig_pd)

    # 3. 20 tightly spaced spectral peaks resolved cleanly.
    x_grid = np.linspace(950.0, 1250.0, 600)
    peak_xs = np.linspace(1040.0, 1155.0, 20)
    y_grid = np.zeros_like(x_grid)
    peaks = []
    for idx, px in enumerate(peak_xs):
        amp = 100.0 - 2.2 * float(idx % 5)
        y_grid += amp * np.exp(-0.5 * ((x_grid - float(px)) / 3.2) ** 2)
        peaks.append(
            spec_core.Peak(position=float(px), intensity=amp, label=f"v{idx}")
        )
    spec = spec_core.Spectrum(
        x=x_grid, y=y_grid, kind="raman", peaks=peaks, x_label="Shift (cm^-1)"
    )
    ax_sp = xm.plot_spectrum(
        spec, annotate_peaks=12, show_sticks=True, preset="classic"
    )
    layout.avoid_text_overlaps(ax_sp, clamp_to_axes=True, draw_leaders=True)
    try:
        rep_sp = layout.audit_figure_layout(ax_sp.figure)
        assert not rep_sp.text_overlaps
        assert rep_sp.passed
    finally:
        plt.close(ax_sp.figure)


def test_stress_very_long_chemical_formulas() -> None:
    formulas = [
        "Ba0.5Sr0.5Co0.8Fe0.2O3-delta",
        "La0.6Sr0.4Co0.2Fe0.8O3-delta",
        "Li7La3Zr1.75Nb0.25O12",
        "Cs0.05FA0.80MA0.15PbI2.55Br0.45",
    ]
    builders: list[Callable[[], Any]] = [
        lambda: xm.plot_volcano(
            [-0.6, -0.15, 0.2, 0.65],
            [-1.8, -0.4, -0.55, -1.9],
            labels=formulas,
            preset="classic",
        ).figure,
        lambda: xm.plot_bar_comparison(
            formulas, [0.42, 0.51, 0.31, 0.64], preset="dark_chalk"
        ).figure,
        lambda: xm.plot_reaction_profile(
            [0.0, 0.82, -0.35, -0.90], formulas, preset="classic"
        ).figure,
        lambda: xm.plot_pipeline_diagram(
            [(formulas[0], formulas[1]), (formulas[2], formulas[3])],
            preset="blueprint",
        ).figure,
        lambda: xm.plot_crystal_field(
            levels=[
                (formulas[0], -0.45, 2),
                (formulas[1], -0.40, 1),
                (formulas[2], 0.62, 2),
            ],
            d_electrons=5,
            preset="classic",
        ).figure,
    ]
    for build_fn in builders:
        fig = build_fn()
        try:
            rep = layout.audit_figure_layout(fig)
            assert not rep.text_overlaps
            assert not rep.clipped_texts
            assert not rep.low_contrast_texts
            assert rep.border_leak_ratio <= 0.01
        finally:
            plt.close(fig)


@pytest.mark.parametrize("figsize", [(14.0, 2.2), (3.2, 10.0)])
def test_stress_extreme_aspect_ratios(figsize: tuple[float, float]) -> None:
    for preset in ("classic", "dark_chalk"):
        with xm.xkcd_style(preset=preset):
            fig_v, ax_v = plt.subplots(figsize=figsize)
            xm.plot_volcano(
                [-0.7, -0.2, 0.1, 0.5],
                [-1.9, -0.5, -0.4, -1.6],
                labels=["Ru", "Rh", "Pt", "Au"],
                ax=ax_v,
                preset=preset,
            )
            fig_v.tight_layout()
        try:
            rep_v = layout.audit_figure_layout(fig_v)
            assert not rep_v.text_overlaps
            assert not rep_v.low_contrast_texts
            assert rep_v.border_leak_ratio <= 0.01
        finally:
            plt.close(fig_v)

        with xm.xkcd_style(preset=preset):
            fig_e, ax_e = plt.subplots(figsize=figsize)
            vs = np.linspace(36.0, 44.0, 7)
            xm.plot_eos(
                vs, -12.0 + 0.05 * (vs - 40.0) ** 2, ax=ax_e, preset=preset
            )
            fig_e.tight_layout()
        try:
            rep_e = layout.audit_figure_layout(fig_e)
            assert not rep_e.text_overlaps
            assert not rep_e.low_contrast_texts
            assert rep_e.border_leak_ratio <= 0.01
        finally:
            plt.close(fig_e)


@pytest.mark.parametrize("grid_shape", [(2, 3), (3, 3)])
@pytest.mark.parametrize("preset", ["classic", "dark_chalk"])
def test_stress_crowded_multi_subplot_grids(
    grid_shape: tuple[int, int],
    preset: str,
) -> None:
    nrows, ncols = grid_shape
    with xm.xkcd_style(preset=preset):
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(4.4 * ncols, 3.6 * nrows)
        )
        flat = list(np.ravel(axes))
        xm.plot_volcano(
            [-0.7, -0.2, 0.1, 0.6],
            [-2.0, -0.6, -0.4, -1.9],
            labels=["Ru", "Rh", "Pt", "Au"],
            ax=flat[0],
            preset=preset,
        )
        vs = np.linspace(36.0, 44.0, 6)
        xm.plot_eos(
            vs, -10.0 + 0.05 * (vs - 40.0) ** 2, ax=flat[1], preset=preset
        )
        xm.plot_phase_diagram_2d(
            [0.0, 0.33, 0.5, 1.0],
            [0.0, -0.28, -0.35, 0.0],
            ["A", "A2B", "AB", "B"],
            ax=flat[2],
            preset=preset,
        )
        xm.plot_reaction_profile(
            [0.0, 0.65, -0.40], ["R", "TS", "P"], ax=flat[3], preset=preset
        )
        xm.plot_crystal_field(
            "octahedral", d_electrons=5, ax=flat[4], preset=preset
        )
        xm.plot_bar_comparison(
            ["PBE", "SCAN", "HSE"], [1.1, 1.4, 1.7], ax=flat[5], preset=preset
        )
        for extra_idx in range(6, len(flat)):
            if extra_idx % 3 == 0:
                xm.plot_parity(
                    [0.0, 1.0, 2.0],
                    [0.1, 0.9, 2.05],
                    ax=flat[extra_idx],
                    preset=preset,
                )
            elif extra_idx % 3 == 1:
                xm.plot_defect_formation(
                    {"V_O": [(2, 1.2), (0, 2.6)]},
                    band_gap=2.0,
                    ax=flat[extra_idx],
                    preset=preset,
                )
            else:
                xm.plot_pipeline_diagram(
                    [("DFT", "VASP"), ("Fit", "EOS")],
                    ax=flat[extra_idx],
                    preset=preset,
                )
        fig.tight_layout()
    try:
        rep = layout.audit_figure_layout(fig)
        assert not rep.text_overlaps
        assert not rep.clipped_texts
        assert not rep.low_contrast_texts
        assert rep.border_leak_ratio <= 0.01
        assert rep.ink_ratio >= 0.01
    finally:
        plt.close(fig)


_BASELINES_DIR = pathlib.Path(__file__).resolve().parent / "baselines"
_BASELINE_SCRIPT = (
    pathlib.Path(__file__).resolve().parents[1]
    / "scripts"
    / "update_visual_baselines.py"
)
_GOLDEN_BASELINE_NAMES: tuple[str, ...] = (
    "molecule_aspirin",
    "crystal_si_111",
    "bands_si_qe",
    "xrd_nacl",
    "dark_chalk_volcano",
)
_GOLDEN_SIMILARITY_THRESHOLD = 0.94
_GOLDEN_ASPECT_TOLERANCE = 0.04
_GOLDEN_INK_TOLERANCE = 0.04


@functools.lru_cache(maxsize=1)
def _load_baseline_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "update_visual_baselines", _BASELINE_SCRIPT
    )
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_png_rgb(path: pathlib.Path) -> np.ndarray:
    arr = np.asarray(mimage.imread(path, format="png"), dtype=np.float64)
    if arr.ndim == 3 and arr.shape[2] >= 4:
        alpha = arr[..., 3:4]
        return np.asarray(
            arr[..., :3] * alpha + (1.0 - alpha), dtype=np.float64
        )
    return np.asarray(arr[..., :3], dtype=np.float64)


def _area_downsample_rgb(
    rgb: np.ndarray,
    size: tuple[int, int] = (64, 64),
) -> np.ndarray:
    target_h, target_w = size
    src_h, src_w, _ = rgb.shape
    row_edges = np.linspace(0, src_h, target_h + 1, dtype=np.int64)
    col_edges = np.linspace(0, src_w, target_w + 1, dtype=np.int64)
    out = np.zeros((target_h, target_w, 3), dtype=np.float64)
    for r_idx in range(target_h):
        r0 = int(row_edges[r_idx])
        r1 = min(max(r0 + 1, int(row_edges[r_idx + 1])), src_h)
        for c_idx in range(target_w):
            c0 = int(col_edges[c_idx])
            c1 = min(max(c0 + 1, int(col_edges[c_idx + 1])), src_w)
            patch = rgb[r0:r1, c0:c1, :]
            out[r_idx, c_idx, 0] = float(np.mean(patch[..., 0]))
            out[r_idx, c_idx, 1] = float(np.mean(patch[..., 1]))
            out[r_idx, c_idx, 2] = float(np.mean(patch[..., 2]))
    return out


def _rgb_to_visual_fingerprint(
    rgb: np.ndarray,
    size: tuple[int, int] = (64, 64),
) -> layout.VisualFingerprint:
    rim = np.concatenate(
        [
            rgb[:2, :, :].reshape(-1, 3),
            rgb[-2:, :, :].reshape(-1, 3),
            rgb[2:-2, :2, :].reshape(-1, 3),
            rgb[2:-2, -2:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    bg_arr = np.median(rim, axis=0)
    diff = np.max(np.abs(rgb - bg_arr[None, None, :]), axis=-1)
    ink_ratio = float(np.mean(diff > 0.06))
    rgb_small = _area_downsample_rgb(rgb, size=size)
    lum_small = (
        0.2126 * rgb_small[..., 0]
        + 0.7152 * rgb_small[..., 1]
        + 0.0722 * rgb_small[..., 2]
    )
    return layout.VisualFingerprint(
        luminance_grid=np.asarray(lum_small, dtype=np.float64),
        rgb_grid=rgb_small,
        bg_rgb=(float(bg_arr[0]), float(bg_arr[1]), float(bg_arr[2])),
        ink_ratio=round(ink_ratio, 6),
        size=size,
    )


@pytest.mark.parametrize("name", _GOLDEN_BASELINE_NAMES)
def test_fixed_baseline_visual_regression(
    name: str,
    tmp_path: pathlib.Path,
) -> None:
    baseline_mod = _load_baseline_module()
    assert tuple(baseline_mod.BASELINE_NAMES) == _GOLDEN_BASELINE_NAMES
    baseline_path = _BASELINES_DIR / f"{name}.png"
    assert baseline_path.is_file()
    assert baseline_path.stat().st_size > 0

    prev_font = xm.get_default_font()
    xm.set_default_font(None)
    fig = baseline_mod.render_baseline_figure(name)
    try:
        spec = baseline_mod.BASELINE_SPECS[name]
        report = layout.audit_figure_layout(
            fig, min_contrast=spec.min_contrast
        )
        assert not report.text_overlaps, (
            f"{name} text overlaps: {report.text_overlaps}"
        )
        assert not report.clipped_texts, (
            f"{name} clipped texts: {report.clipped_texts}"
        )
        assert not report.low_contrast_texts, (
            f"{name} low contrast: {report.low_contrast_texts}"
        )
        assert report.border_leak_ratio <= 0.01, (
            f"{name} border leak: {report.border_leak_ratio}"
        )
        assert report.passed

        fresh_path = tmp_path / f"{name}.png"
        xm.save_figure(
            fig, fresh_path, dpi=baseline_mod.BASELINE_DPI, tight=True
        )
    finally:
        plt.close(fig)
        xm.set_default_font(prev_font)

    base_rgb = _load_png_rgb(baseline_path)
    fresh_rgb = _load_png_rgb(fresh_path)
    base_aspect = float(base_rgb.shape[1]) / max(float(base_rgb.shape[0]), 1.0)
    fresh_aspect = float(fresh_rgb.shape[1]) / max(
        float(fresh_rgb.shape[0]), 1.0
    )
    aspect_rel_diff = abs(fresh_aspect - base_aspect) / base_aspect
    assert aspect_rel_diff <= _GOLDEN_ASPECT_TOLERANCE, (
        f"{name} aspect drift {aspect_rel_diff:.4f} "
        f"(fresh={fresh_aspect:.3f}, base={base_aspect:.3f})"
    )

    fp_base = _rgb_to_visual_fingerprint(base_rgb)
    fp_fresh = _rgb_to_visual_fingerprint(fresh_rgb)
    sim = layout.visual_similarity(fp_fresh, fp_base)
    assert sim >= _GOLDEN_SIMILARITY_THRESHOLD, (
        f"{name} visual similarity {sim:.4f} < {_GOLDEN_SIMILARITY_THRESHOLD}"
    )
    ink_diff = abs(fp_fresh.ink_ratio - fp_base.ink_ratio)
    assert ink_diff <= _GOLDEN_INK_TOLERANCE, (
        f"{name} ink-ratio diff {ink_diff:.4f} > {_GOLDEN_INK_TOLERANCE}"
    )


def test_fixed_baseline_detects_visual_drift(tmp_path: pathlib.Path) -> None:
    baseline_mod = _load_baseline_module()
    dpi = int(baseline_mod.BASELINE_DPI)

    def _compare(name: str, perturbed_fig: mfigure.Figure, tag: str) -> float:
        out_path = tmp_path / f"{name}_{tag}.png"
        try:
            xm.save_figure(perturbed_fig, out_path, dpi=dpi, tight=True)
        finally:
            plt.close(perturbed_fig)
        fp_base = _rgb_to_visual_fingerprint(
            _load_png_rgb(_BASELINES_DIR / f"{name}.png")
        )
        fp_pert = _rgb_to_visual_fingerprint(_load_png_rgb(out_path))
        return layout.visual_similarity(fp_pert, fp_base)

    prev_font = xm.get_default_font()
    xm.set_default_font(None)
    try:
        # 1. Theme drift: dark_chalk -> classic drops similarity sharply.
        with xm.xkcd_style("classic"):
            fig_theme, ax_theme = plt.subplots(figsize=(5.4, 3.8))
            xm.plot_volcano(
                [-0.75, -0.35, -0.05, 0.25, 0.65],
                [-2.9, -1.35, -0.35, -1.25, -3.1],
                labels=["Ru", "Rh", "Pt", "Pd", "Au"],
                preset="classic",
                ax=ax_theme,
                title="HER Volcano",
            )
            fig_theme.tight_layout()
        sim_theme = _compare("dark_chalk_volcano", fig_theme, "classic")
        assert sim_theme < _GOLDEN_SIMILARITY_THRESHOLD - 0.30

        # 2. 3D camera rotation drift (+90 deg azimuth on aspirin).
        with xm.xkcd_style("classic"):
            fig_cam, ax_cam = plt.subplots(figsize=(4.8, 3.8))
            xm.plot_molecule(
                "aspirin.xyz",
                ax=ax_cam,
                camera=xm.Camera(
                    azimuth=-35.0,
                    elevation=42.0,
                    perspective=0.28,
                ),
                title="Aspirin (3D)",
            )
            fig_cam.tight_layout()
        sim_cam = _compare("molecule_aspirin", fig_cam, "rot90")
        assert sim_cam < _GOLDEN_SIMILARITY_THRESHOLD - 0.10

        # 3. Dropped data curves on electronic band structure.
        fig_drop = baseline_mod.render_baseline_figure("bands_si_qe")
        for line in list(fig_drop.axes[0].lines):
            line.remove()
        sim_drop = _compare("bands_si_qe", fig_drop, "dropped_curves")
        assert sim_drop < _GOLDEN_SIMILARITY_THRESHOLD - 0.10

        # 4. Missing Miller plane overlay on Si (111) crystal.
        with xm.xkcd_style("classic"):
            fig_nomiller, ax_nomiller = plt.subplots(figsize=(4.8, 4.2))
            xm.plot_crystal(
                "si_diamond.cif",
                ax=ax_nomiller,
                crystal_axes="corner",
                title="Si (111)",
            )
            fig_nomiller.tight_layout()
        sim_nomiller = _compare("crystal_si_111", fig_nomiller, "no_miller")
        assert sim_nomiller < _GOLDEN_SIMILARITY_THRESHOLD - 0.10
    finally:
        xm.set_default_font(prev_font)
