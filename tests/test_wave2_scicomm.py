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
"""Analytical and regression tests for Wave 2 plots, layouts, and 3D Studio."""

from __future__ import annotations

import pathlib

import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pytest

import xkcd_matchem as xm
from xkcd_matchem.core import exceptions
from xkcd_matchem.core import spectroscopy as spec_core
from xkcd_matchem.layout import templates
from xkcd_matchem.plots import crystalfield
from xkcd_matchem.plots import electrochem
from xkcd_matchem.plots import spectroscopy as spec_plots
from xkcd_matchem.render import interactive


matplotlib.use("Agg")


def _unpaired_count(
    occ: list[list[tuple[int, int]]],
) -> int:
    """Computes total unpaired electron count across all orbital levels."""
    return sum(up - down for level in occ for up, down in level)


def _total_electrons(
    occ: list[list[tuple[int, int]]],
) -> int:
    """Computes total electron population across all orbital levels."""
    return sum(up + down for level in occ for up, down in level)


def test_crystal_field_barycenter_and_spin_invariants() -> None:
    geometries = (
        "octahedral",
        "oh",
        "tetrahedral",
        "td",
        "square_planar",
        "d4h",
        "trigonal_bipyramidal",
        "tbp",
        "d3h",
    )
    for geom in geometries:
        for delta in (1.25, 2.40, 3.10):
            lvls = crystalfield.crystal_field_levels(geom, delta=delta)
            assert sum(deg for _, _, deg in lvls) == 5
            barycenter = sum(float(deg) * eng for _, eng, deg in lvls)
            assert barycenter == pytest.approx(0.0, abs=1e-12)

    oct_lvls = crystalfield.crystal_field_levels("octahedral", delta=2.0)
    expected_unpaired = {
        4: (4, 2),
        5: (5, 1),
        6: (4, 0),
        7: (3, 1),
    }
    for n_elec, (hi_exp, lo_exp) in expected_unpaired.items():
        occ_hi = crystalfield.fill_crystal_field_orbitals(
            oct_lvls, d_electrons=n_elec, spin_state="high"
        )
        occ_lo = crystalfield.fill_crystal_field_orbitals(
            oct_lvls, d_electrons=n_elec, spin_state="low"
        )
        assert _total_electrons(occ_hi) == n_elec
        assert _total_electrons(occ_lo) == n_elec
        assert _unpaired_count(occ_hi) == hi_exp
        assert _unpaired_count(occ_lo) == lo_exp

    occ_0 = crystalfield.fill_crystal_field_orbitals(oct_lvls, 0, "high")
    occ_10 = crystalfield.fill_crystal_field_orbitals(oct_lvls, 10, "low")
    assert _total_electrons(occ_0) == 0
    assert _total_electrons(occ_10) == 10
    assert _unpaired_count(occ_10) == 0

    for geom, d_cnt, spin in (
        ("octahedral", 6, "high"),
        ("tetrahedral", 5, "high"),
        ("square_planar", 8, "low"),
        ("trigonal_bipyramidal", 3, "high"),
    ):
        ax = crystalfield.plot_crystal_field(
            geometry=geom,
            d_electrons=d_cnt,
            spin_state=spin,
            delta=2.2,
            title=f"{geom} d{d_cnt}",
        )
        assert len(ax.lines) >= 6
        plt.close(ax.figure)

    custom_lvls = [
        (r"$a_{1g}$", -0.60, 1),
        (r"$e_g^{(1)}$", 0.14, 2),
        (r"$e_g^{(2)}$", 0.16, 2),
    ]
    _, ax_custom = plt.subplots(figsize=(5.2, 4.0))
    crystalfield.plot_crystal_field(
        levels=custom_lvls,
        d_electrons=4,
        spin_state="low",
        show_barycenter=False,
        delta_label=r"$\Delta_{cf}$",
        ax=ax_custom,
    )
    plt.close(ax_custom.figure)

    ax_single = crystalfield.plot_crystal_field(
        levels=[("s", 0.0, 1)], d_electrons=0, show_barycenter=False
    )
    plt.close(ax_single.figure)

    with pytest.raises(exceptions.StructureError):
        crystalfield.crystal_field_levels("hexagonal_prism")
    with pytest.raises(exceptions.StructureError):
        crystalfield.crystal_field_levels("octahedral", delta=0.0)
    with pytest.raises(exceptions.StructureError):
        crystalfield.crystal_field_levels(levels=[])
    with pytest.raises(exceptions.StructureError):
        crystalfield.crystal_field_levels(levels=[("bad", 0.0, 0)])
    with pytest.raises(exceptions.StructureError):
        crystalfield.fill_crystal_field_orbitals(oct_lvls, 5, "intermediate")
    with pytest.raises(exceptions.StructureError):
        crystalfield.fill_crystal_field_orbitals(oct_lvls, 11, "high")


def test_electrochem_cv_and_tafel_invariants() -> None:
    fwd_e = np.linspace(0.0, 0.8, 161)
    rev_e = np.linspace(0.8, 0.0, 161)
    e_cv = np.concatenate([fwd_e, rev_e])
    j_fwd = 3.5 * np.exp(-((fwd_e - 0.42) / 0.055) ** 2)
    j_rev = -3.4 * np.exp(-((rev_e - 0.36) / 0.055) ** 2)
    j_cv = np.concatenate([j_fwd, j_rev])

    e_pa, j_pa, e_pc, j_pc, e_half = electrochem._detect_cv_peaks(e_cv, j_cv)
    assert e_pa == pytest.approx(0.42, abs=1e-6)
    assert j_pa == pytest.approx(3.5, abs=1e-6)
    assert e_pc == pytest.approx(0.36, abs=1e-6)
    assert j_pc == pytest.approx(-3.4, abs=1e-6)
    assert e_half == pytest.approx(0.39, abs=1e-6)

    ax_iupac = electrochem.plot_cyclic_voltammetry(
        e_cv, j_cv, convention="iupac", title="IUPAC CV"
    )
    assert not ax_iupac.xaxis_inverted()
    texts_iupac = "\n".join(t.get_text() for t in ax_iupac.texts)
    assert "0.390 V" in texts_iupac
    assert "60 mV" in texts_iupac
    plt.close(ax_iupac.figure)

    ax_us = electrochem.plot_cyclic_voltammetry(
        [(e_cv, j_cv), (e_cv, 0.8 * j_cv)],
        convention="us",
        scan_labels=["Scan 1", "Scan 2"],
    )
    assert ax_us.xaxis_inverted()
    plt.close(ax_us.figure)

    e_2d = np.vstack([e_cv, e_cv])
    j_2d = np.vstack([j_cv, 0.6 * j_cv])
    ax_2d = electrochem.plot_cyclic_voltammetry(
        e_2d, j_2d, annotate_peaks=False, show_e_half=False
    )
    plt.close(ax_2d.figure)

    ax_list = electrochem.plot_cyclic_voltammetry(
        [e_cv * 30.0, e_cv[:-4] * 30.0], [j_cv, 0.9 * j_cv[:-4]]
    )
    plt.close(ax_list.figure)

    with pytest.raises(exceptions.StructureError):
        electrochem.plot_cyclic_voltammetry(e_cv, j_cv, convention="invalid")
    with pytest.raises(exceptions.StructureError):
        electrochem.plot_cyclic_voltammetry([])
    with pytest.raises(exceptions.StructureError):
        electrochem.plot_cyclic_voltammetry(
            e_cv, j_cv, scan_labels=["A", "B"]
        )

    b_true = 118.4
    log_j0_true = -2.25
    log_j = np.linspace(-1.2, 1.4, 30)
    j_tafel = 10.0**log_j
    eta_mv = b_true * (log_j - log_j0_true)

    _, _, _, fit_mv = electrochem._prepare_tafel_fit(
        eta_mv, j_tafel, "mV", None
    )
    assert fit_mv[2] == pytest.approx(b_true, rel=1e-9)
    assert fit_mv[3] == pytest.approx(log_j0_true, rel=1e-9)

    _, _, _, fit_v = electrochem._prepare_tafel_fit(
        eta_mv / 1000.0, j_tafel, "V", (-0.8, 1.1)
    )
    assert fit_v[2] == pytest.approx(b_true, rel=1e-9)

    ax_tafel = electrochem.plot_tafel(
        eta_mv,
        j_tafel,
        unit="mV",
        fit_range=(150.0, 400.0),
        title="Tafel Kinetics",
    )
    tafel_txt = "\n".join(t.get_text() for t in ax_tafel.texts)
    assert "118.4 mV/dec" in tafel_txt
    assert "-2.25" in tafel_txt
    plt.close(ax_tafel.figure)

    with pytest.raises(exceptions.StructureError):
        electrochem.plot_tafel(eta_mv, j_tafel, unit="meV")
    with pytest.raises(exceptions.StructureError):
        electrochem.plot_tafel([10.0, 20.0], [0.0, 0.0])
    with pytest.raises(exceptions.StructureError):
        electrochem.plot_tafel(eta_mv, j_tafel, fit_range=(9000.0, 9500.0))


def test_battery_profile_and_dqdv_invariants() -> None:
    q_arr = np.linspace(0.0, 160.0, 120)
    v_dis = 3.45 - 0.08 * (q_arr / 160.0) - 0.25 * (q_arr / 160.0) ** 8
    v_chg = 3.48 + 0.08 * (q_arr / 160.0) + 0.22 * (q_arr / 160.0) ** 8

    ax_pair = electrochem.plot_battery_profile(
        q_arr,
        v_dis,
        charge_voltage=v_chg,
        charge_capacity=q_arr,
        cycle_labels=("Discharge", "Charge"),
        plateau_shade="hatch",
        title="LiFePO4 Profile",
    )
    assert len(ax_pair.lines) >= 2
    plt.close(ax_pair.figure)

    ax_band = electrochem.plot_battery_profile(
        q_arr, v_dis, plateau_shade=(3.38, 3.48)
    )
    plt.close(ax_band.figure)

    ax_auto_band = electrochem.plot_battery_profile(
        q_arr, v_dis, plateau_shade=True
    )
    plt.close(ax_auto_band.figure)

    ax_multi = electrochem.plot_battery_profile(
        [(q_arr, v_dis), (0.95 * q_arr, v_dis - 0.02)]
    )
    plt.close(ax_multi.figure)

    ax_multi_seq = electrochem.plot_battery_profile(
        [q_arr, 0.92 * q_arr[:-4]],
        [v_dis, v_dis[:-4] - 0.03],
        cycle_labels=["Cycle 1", "Cycle 50"],
    )
    plt.close(ax_multi_seq.figure)

    with pytest.raises(exceptions.StructureError):
        electrochem.plot_battery_profile([])
    with pytest.raises(exceptions.StructureError):
        electrochem.plot_battery_profile([q_arr, q_arr[:-4]], [v_dis])

    v_grid = np.linspace(3.0, 4.0, 501)
    true_p1, true_p2 = 3.34, 3.72
    dqdv_exact = 220.0 * np.exp(
        -0.5 * ((v_grid - true_p1) / 0.022) ** 2
    ) + 165.0 * np.exp(-0.5 * ((v_grid - true_p2) / 0.025) ** 2)
    dv = float(v_grid[1] - v_grid[0])
    q_grid = np.cumsum(dqdv_exact) * dv

    v_dup = np.concatenate([v_grid, [v_grid[100]]])
    q_dup = np.concatenate([q_grid, [q_grid[100]]])
    v_asc, dqdv_rec = electrochem._compute_dqdv_trace(
        v_dup, q_dup, smooth_window=5
    )
    pk_idx = electrochem._find_local_peak_indices(v_asc, np.abs(dqdv_rec), 2)
    detected_v = sorted(float(v_asc[idx]) for idx in pk_idx)
    assert len(detected_v) == 2
    assert detected_v[0] == pytest.approx(true_p1, abs=5e-3)
    assert detected_v[1] == pytest.approx(true_p2, abs=5e-3)

    ax_dqdv = electrochem.plot_dqdv(
        v_grid,
        q_grid,
        charge_voltage=v_grid + 0.02,
        charge_capacity=q_grid,
        smooth_window=5,
        annotate_peaks=2,
        title="Differential Capacity",
    )
    ann_texts = [t.get_text() for t in ax_dqdv.texts]
    assert any("3.34 V" in txt for txt in ann_texts)
    assert any("3.72 V" in txt for txt in ann_texts)
    plt.close(ax_dqdv.figure)

    ax_raw_dqdv = electrochem.plot_dqdv(
        v_grid, q_grid, smooth_window=1, annotate_peaks=0
    )
    plt.close(ax_raw_dqdv.figure)

    with pytest.raises(exceptions.StructureError):
        electrochem.plot_dqdv([3.4, 3.4, 3.4], [10.0, 20.0, 30.0])


def test_nmr_multiplet_and_integral_invariants() -> None:
    mhz = 400.0
    j_hz = 7.2
    d_ppm = j_hz / mhz

    pos_t, wt_t = spec_plots.expand_nmr_multiplet(
        1.25, integral=3.0, multiplicity="t", j_hz=j_hz, spectrometer_mhz=mhz
    )
    np.testing.assert_allclose(
        pos_t, [1.25 - d_ppm, 1.25, 1.25 + d_ppm], rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(wt_t / wt_t[0], [1.0, 2.0, 1.0], atol=1e-12)
    assert float(np.sum(wt_t)) == pytest.approx(3.0, rel=1e-12)

    pos_q, wt_q = spec_plots.expand_nmr_multiplet(
        4.12, integral=2.0, multiplicity="q", j_hz=j_hz, spectrometer_mhz=mhz
    )
    np.testing.assert_allclose(
        pos_q,
        [
            4.12 - 1.5 * d_ppm,
            4.12 - 0.5 * d_ppm,
            4.12 + 0.5 * d_ppm,
            4.12 + 1.5 * d_ppm,
        ],
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        wt_q / wt_q[0], [1.0, 3.0, 3.0, 1.0], atol=1e-12
    )
    assert float(np.sum(wt_q)) == pytest.approx(2.0, rel=1e-12)

    pos_dd, wt_dd = spec_plots.expand_nmr_multiplet(
        5.80,
        integral=1.0,
        multiplicity="dd",
        j_hz=(16.0, 10.0),
        spectrometer_mhz=mhz,
    )
    assert pos_dd.size == 4
    np.testing.assert_allclose(wt_dd, [0.25, 0.25, 0.25, 0.25], atol=1e-12)
    np.testing.assert_allclose(
        pos_dd,
        [
            5.80 - 13.0 / mhz,
            5.80 - 3.0 / mhz,
            5.80 + 3.0 / mhz,
            5.80 + 13.0 / mhz,
        ],
        atol=1e-12,
    )

    for code, n_lines in (("quint", 5), ("sext", 6), ("sept", 7), ("m", 5)):
        pos_m, wt_m = spec_plots.expand_nmr_multiplet(
            2.5, integral=2.0, multiplicity=code, j_hz=7.0
        )
        assert pos_m.size == n_lines
        assert float(np.sum(wt_m)) == pytest.approx(2.0, rel=1e-12)

    resonances = [
        {
            "shift": 4.12,
            "integral": 2.0,
            "multiplicity": "q",
            "j_hz": 7.1,
            "label": "OCH2",
        },
        {
            "shift": 2.04,
            "integral": 3.0,
            "multiplicity": "s",
            "j_hz": 0.0,
            "label": "COCH3",
        },
        (1.26, 3.0, "t", 7.1, "CH3"),
    ]
    ax_nmr = spec_plots.plot_nmr(
        resonances,
        spectrometer_mhz=400.0,
        show_integrals=True,
        show_trees=True,
        show_tms=True,
        nucleus="1H",
        title="Ethyl Acetate 1H NMR",
    )
    assert ax_nmr.xaxis_inverted()
    int_lines = [
        ln for ln in ax_nmr.lines if ln.get_gid() == "nmr_step_integral"
    ]
    assert len(int_lines) == 1
    y_int = np.asarray(int_lines[0].get_ydata(), dtype=np.float64)
    assert bool(np.all(np.diff(y_int) >= -1e-12))

    bundle = spec_plots._prepare_nmr_data(
        resonances, None, None, None, None, (400.0, None, 0.018, True)
    )
    ppm_grid, y_curve = bundle[0], bundle[1]
    cum_raw = spec_plots._cumulative_trapezoid(ppm_grid[::-1], y_curve[::-1])
    tot_h = sum(float(r[1]) for r in bundle[4])
    cum_scaled = cum_raw * (tot_h / float(cum_raw[-1]))
    assert float(cum_scaled[-1]) == pytest.approx(8.0, rel=1e-9)
    plt.close(ax_nmr.figure)

    ax_par = spec_plots.plot_nmr(
        [1.26, 2.04, 4.12],
        couplings_hz=[7.1, 0.0, 7.1],
        multiplicities=["t", "s", "q"],
        integrals=[3.0, 3.0, 2.0],
        labels=["a", "b", "c"],
        ppm_range=(-0.4, 5.0),
    )
    plt.close(ax_par.figure)

    spec_obj = spec_core.Spectrum(
        x=ppm_grid,
        y=y_curve,
        kind="nmr",
        x_label="Chemical Shift (ppm)",
        peaks=[spec_core.Peak(position=2.04, intensity=3.0)],
    )
    ax_spec = spec_plots.plot_nmr(spec_obj, ppm_range=(0.0, 5.0))
    plt.close(ax_spec.figure)

    with pytest.raises(exceptions.StructureError):
        spec_plots.expand_nmr_multiplet(1.0, spectrometer_mhz=0.0)
    with pytest.raises(exceptions.StructureError):
        spec_plots.expand_nmr_multiplet(1.0, integral=-1.0)
    with pytest.raises(exceptions.StructureError):
        spec_plots.expand_nmr_multiplet(1.0, multiplicity="invalid_code")
    with pytest.raises(exceptions.StructureError):
        spec_plots.plot_nmr([])


def test_xps_shirley_background_and_envelope_invariants() -> None:
    be_asc = np.linspace(282.0, 292.0, 260)
    true_bg = 120.0 + 300.0 * (
        0.5 * (1.0 + np.tanh((be_asc - 285.5) / 1.4))
    )
    p1 = 850.0 * np.exp(-0.5 * ((be_asc - 284.8) / 0.55) ** 2)
    p2 = 340.0 * np.exp(-0.5 * ((be_asc - 286.4) / 0.60) ** 2)
    p3 = 210.0 * np.exp(-0.5 * ((be_asc - 288.6) / 0.65) ** 2)
    y_asc = true_bg + p1 + p2 + p3

    bg_asc = spec_plots.compute_shirley_background(be_asc, y_asc)
    assert bg_asc[0] == pytest.approx(float(y_asc[0]), rel=1e-9)
    assert bg_asc[-1] == pytest.approx(float(y_asc[-1]), rel=1e-9)
    assert bool(np.all(np.diff(bg_asc) >= -1e-9))
    assert float(np.min(bg_asc)) >= float(y_asc[0]) - 1e-6
    assert float(np.max(bg_asc)) <= float(y_asc[-1]) + 1e-6

    be_desc = be_asc[::-1]
    y_desc = y_asc[::-1]
    bg_desc = spec_plots.compute_shirley_background(be_desc, y_desc)
    np.testing.assert_allclose(bg_desc, bg_asc[::-1], rtol=1e-9, atol=1e-9)

    components = [
        {"center": 284.8, "amplitude": 850.0, "fwhm": 1.2, "label": "C-C"},
        {"center": 286.4, "amplitude": 340.0, "fwhm": 1.3, "label": "C-O"},
        {"center": 288.6, "area": 360.0, "fwhm": 1.4, "label": "O-C=O"},
    ]
    ax_xps = spec_plots.plot_xps(
        be_asc,
        y_asc,
        components=components,
        background="shirley",
        invert_x=True,
        show_residuals=True,
        title="C 1s Core-Level XPS",
    )
    assert ax_xps.xaxis_inverted()
    labels = [ln.get_label() for ln in ax_xps.lines]
    assert "Shirley BG" in labels
    assert "Total Envelope" in labels
    assert "Residual" in labels
    plt.close(ax_xps.figure)

    ax_so = spec_plots.plot_xps(
        be_asc,
        components=[
            {
                "center": 284.8,
                "amplitude": 700.0,
                "splitting": 0.4,
                "branching_ratio": 0.5,
                "label": "Ti 2p3/2",
            },
            ("C-O", 286.4, 280.0, 1.2),
            (288.5, 190.0, "C=O"),
            ("RawComp", p1),
        ],
        background="linear",
        invert_x=False,
    )
    assert not ax_so.xaxis_inverted()
    plt.close(ax_so.figure)

    xps_spec = spec_core.Spectrum(
        x=be_asc,
        y=y_asc,
        kind="xps",
        x_label="Binding Energy (eV)",
    )
    ax_spec_xps = spec_plots.plot_xps(xps_spec, background=None)
    plt.close(ax_spec_xps.figure)

    with pytest.raises(exceptions.StructureError):
        spec_plots.compute_shirley_background([284.0], [100.0])
    with pytest.raises(exceptions.StructureError):
        spec_plots.plot_xps(be_asc)
    with pytest.raises(exceptions.StructureError):
        spec_plots.plot_xps(be_asc, y_asc, background="tougaard_unknown")


def test_scicomm_layout_templates(tmp_path: pathlib.Path) -> None:
    ga = templates.create_graphical_abstract(
        title="Solar-to-Hydrogen Graphical Abstract",
        takeaway="Single-atom Pt sites lower the Volmer barrier by 0.35 eV",
        panels=2,
        size_mm=(130.0, 55.0),
        arrow_label="hν",
    )
    assert isinstance(ga, templates.GraphicalAbstractLayout)
    assert set(ga.axes) == {"left", "right"}
    assert ga.arrow_ax is not None
    ga.left_ax.plot([0.0, 1.0], [0.0, 1.0])
    ga.right_ax.plot([0.0, 1.0], [1.0, 0.0])
    ga_out = ga.save(tmp_path / "ga.png", dpi=100)
    assert ga_out.is_file()
    plt.close(ga.figure)

    ga3 = templates.create_graphical_abstract(
        panels=3, arrow_label="→", preset="talk_dark"
    )
    assert ga3.center_ax is not None
    plt.close(ga3.figure)

    ga1 = templates.create_graphical_abstract(panels=1, arrow_label=None)
    assert set(ga1.axes) == {"main"}
    plt.close(ga1.figure)

    with pytest.raises(exceptions.StyleError):
        templates.create_graphical_abstract(panels=0)
    with pytest.raises(exceptions.StyleError):
        templates.create_graphical_abstract(size_mm=(-10.0, 50.0))

    slide = templates.create_talk_slide(
        title="Operando Electrochemistry & Crystal-Field Splitting",
        subtitle="Wave 2 Science Communication Showcase",
        takeaway="Direct correlation between eg occupancy and OER activity",
        layout="two_column",
        character="ponytail",
        speech="Look at that d-orbital splitting!",
    )
    assert isinstance(slide, templates.TalkSlideLayout)
    assert set(slide.axes) == {"left", "right"}
    assert slide.presenter_ax is not None
    assert slide.footer_ax is not None
    slide_out = slide.save(tmp_path / "slide.png", dpi=90)
    assert slide_out.is_file()
    plt.close(slide.figure)

    for lay_mode, pre in (
        ("hero_right_notes", "talk_dark"),
        ("three_panel", "talk_16x9"),
    ):
        sl_i = templates.create_talk_slide(
            title=lay_mode, layout=lay_mode, preset=pre
        )
        plt.close(sl_i.figure)

    with pytest.raises(exceptions.StyleError):
        templates.create_talk_slide(title="Bad", layout="four_quadrant")

    poster = templates.create_conference_poster(
        title="xkcd-matchem: Hand-Sketched Materials Science Communication",
        authors="V. Z. Ordillo et al.",
        affiliation="Institute for Computational Materials",
        size="compact_board",
        dpi=90,
    )
    assert isinstance(poster, templates.ConferencePosterLayout)
    assert len(poster.cards) == 6
    assert poster.board.width_mm == pytest.approx(360.0)
    assert poster.board.height_mm == pytest.approx(240.0)
    poster_out = poster.save(tmp_path / "poster.png", dpi=80)
    assert poster_out.is_file()
    plt.close(poster.board.figure)

    poster_a0 = templates.create_conference_poster(
        title="A0 Poster",
        authors="Author Only",
        section_titles=("1. Intro", "2. Method", "3. Results", "4. Summary"),
        size="A0_portrait",
        dpi=72,
    )
    assert len(poster_a0.cards) == 4
    plt.close(poster_a0.board.figure)

    with pytest.raises(exceptions.StyleError):
        templates.create_conference_poster("Bad", size="letter_size")
    with pytest.raises(exceptions.StyleError):
        templates.create_conference_poster("Bad", section_titles=())

    comic_layout = templates.create_mechanism_comic(
        title="SN2 Bimolecular Nucleophilic Substitution",
    )
    assert isinstance(comic_layout, templates.MechanismComicLayout)
    assert len(comic_layout.panel_axes) == 3
    assert len(comic_layout.inset_axes) == 3
    comic_layout.inset_axes[0].plot([0, 1], [0, 1])
    comic_out = comic_layout.save(tmp_path / "comic.png", dpi=90)
    assert comic_out.is_file()
    plt.close(comic_layout.figure)

    with pytest.raises(exceptions.StyleError):
        templates.create_mechanism_comic(captions=())
    with pytest.raises(exceptions.StyleError):
        templates.create_mechanism_comic(characters=())


def test_playground_scenes_and_top_level_lazy_exports(
    tmp_path: pathlib.Path,
) -> None:
    scenes = interactive.build_playground_scenes()
    assert len(scenes) == 9
    expected_samples = [
        "water_homo.cube",
        "tio2_slab.poscar",
        "sn2_neb.xyz",
        "aspirin.xyz",
        "mos2_2h.poscar",
        "si_diamond.cif",
        "nacl.cif",
        "benzene.sdf",
        "alaninedipeptide.pdb",
    ]
    assert [s["source_sample"] for s in scenes] == expected_samples
    expected_categories = {
        "3D Orbitals",
        "Crystals & Slabs",
        "Reactions & MD",
        "Molecules",
        "Biomolecules",
    }
    assert {s["category"] for s in scenes} == expected_categories
    assert all(bool(s.get("description")) for s in scenes)
    assert scenes[0]["iso_shade"] == "crosshatch"
    assert len(scenes[0].get("isosurfaces", [])) >= 1
    assert scenes[1]["style"] == "polyhedra"
    assert len(scenes[1].get("polyhedra", [])) >= 1
    assert len(scenes[2].get("frames", [])) == 9
    assert scenes[3]["camera"]["azimuth"] == pytest.approx(-125.0)
    assert scenes[3]["camera"]["elevation"] == pytest.approx(42.0)

    out_html = tmp_path / "playground_studio.html"
    interactive.export_interactive_html(
        "water_homo.cube",
        out_html,
        include_bundled_samples=True,
        iso_shade="crosshatch",
    )
    html_text = out_html.read_text(encoding="utf-8")
    for sample_name in expected_samples:
        assert sample_name in html_text
    for cat in expected_categories:
        assert cat in html_text
    for dom_hook in (
        'id="xm-scene-bar"',
        'id="xm-scene-chips"',
        'id="xm-iso-bar"',
        'id="xm-iso-chips"',
        'id="xm-struct-category"',
        'id="xm-struct-desc"',
    ):
        assert dom_hook in html_text

    wave2_symbols = {
        "plot_battery_profile": electrochem.plot_battery_profile,
        "plot_crystal_field": crystalfield.plot_crystal_field,
        "plot_cyclic_voltammetry": electrochem.plot_cyclic_voltammetry,
        "plot_dqdv": electrochem.plot_dqdv,
        "plot_nmr": spec_plots.plot_nmr,
        "plot_tafel": electrochem.plot_tafel,
        "plot_xps": spec_plots.plot_xps,
        "ConferencePosterLayout": templates.ConferencePosterLayout,
        "GraphicalAbstractLayout": templates.GraphicalAbstractLayout,
        "MechanismComicLayout": templates.MechanismComicLayout,
        "TalkSlideLayout": templates.TalkSlideLayout,
        "create_conference_poster": templates.create_conference_poster,
        "create_graphical_abstract": templates.create_graphical_abstract,
        "create_mechanism_comic": templates.create_mechanism_comic,
        "create_talk_slide": templates.create_talk_slide,
    }
    assert len(wave2_symbols) == 15
    xm_dir = set(dir(xm))
    for sym_name, target_obj in wave2_symbols.items():
        assert sym_name in xm.__all__
        assert sym_name in xm_dir
        assert getattr(xm, sym_name) is target_obj
