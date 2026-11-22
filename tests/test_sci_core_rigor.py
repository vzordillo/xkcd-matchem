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
"""Focused scientific validation tests for Stage A core rigor upgrades."""

from __future__ import annotations

import json
import pathlib
import sys
import types
import warnings

import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pytest

from xkcd_matchem import core
from xkcd_matchem.core import brillouin
from xkcd_matchem.core import electronic
from xkcd_matchem.core import exceptions
from xkcd_matchem.core import geometry
from xkcd_matchem.core import structure
from xkcd_matchem.core import trajectory
from xkcd_matchem.core import units
from xkcd_matchem.core import wulff
from xkcd_matchem.data import elements
from xkcd_matchem.io import loader
from xkcd_matchem.io.native import lammps
from xkcd_matchem.io.native import xyz
from xkcd_matchem.plots import thermo
from xkcd_matchem.plots import trajectory as traj_plots
from xkcd_matchem.style import theme


matplotlib.use("Agg")


def test_units_contract_converters_and_provenance() -> None:
    expected_units = {
        "position": "A",
        "cell": "A",
        "energy": "eV",
        "force": "eV/A",
        "pressure": "GPa",
        "time": "fs",
        "reciprocal_length": "A^-1",
        "temperature": "K",
        "potential": "V",
    }
    assert dict(core.INTERNAL_UNITS) == expected_units
    for name in (
        "BandGapResult",
        "INTERNAL_UNITS",
        "ScientificProvenance",
        "convert_energy",
        "convert_pressure",
        "convert_time",
    ):
        assert name in core.__all__
        assert getattr(core, name) is not None

    e_ev = core.convert_energy(1.0, "Ha", "eV")
    assert isinstance(e_ev, float)
    assert e_ev == pytest.approx(elements.CODATA2022.hartree_to_ev, rel=1e-12)
    assert core.convert_energy(2.0, "Ry") == pytest.approx(
        elements.CODATA2022.hartree_to_ev, rel=1e-12
    )
    arr_e = core.convert_energy([1.0, 2.0], "eV", "kJ/mol")
    assert np.allclose(
        core.convert_energy(arr_e, "kJ/mol", "eV"), [1.0, 2.0], rtol=1e-12
    )
    assert core.convert_energy(1.0, "kcal/mol", "kJ/mol") == pytest.approx(
        4.184, rel=1e-6
    )

    assert core.convert_time(2.5, "ps") == pytest.approx(2500.0)
    assert np.allclose(
        core.convert_time([1000.0, 2000.0], "fs", "ps"), [1.0, 2.0]
    )
    assert core.convert_time(1.0, "ns", "ps") == pytest.approx(1000.0)

    assert core.convert_pressure(1.0, "eV/A^3") == pytest.approx(
        elements.CODATA2022.ev_per_a3_to_gpa, rel=1e-12
    )
    assert core.convert_pressure(10000.0, "bar", "GPa") == pytest.approx(1.0)
    assert np.allclose(
        core.convert_pressure([1.0, 2.0], "GPa", "kbar"), [10.0, 20.0]
    )

    for fn, bad_src, bad_dst in (
        (units.convert_energy, "erg", "eV"),
        (units.convert_energy, "eV", "erg"),
        (units.convert_time, "fortnight", "fs"),
        (units.convert_pressure, "torr", "GPa"),
    ):
        with pytest.raises(exceptions.StructureError):
            fn(1.0, bad_src, bad_dst)

    prov = core.ScientificProvenance(
        algorithm="birch_murnaghan_3rd",
        units={"volume": "A^3", "energy": "eV"},
        source="si_eos.dat",
        parser="native",
        parameters={"order": 3, "basis": "cell"},
        assumptions=["isothermal 0 K DFT"],
    )
    as_dict = prov.to_dict()
    assert as_dict["package_version"] == "0.1.0"
    assert as_dict["parameters"] == {"order": 3, "basis": "cell"}
    assert as_dict["assumptions"] == ["isothermal 0 K DFT"]
    assert json.loads(json.dumps(as_dict)) == as_dict
    with pytest.raises(exceptions.StructureError):
        core.ScientificProvenance(algorithm="   ")


def test_band_gap_occupation_semantics_smearing_and_indeterminate() -> None:
    k_d = np.array([0.0, 0.5, 1.0])
    # Indirect gap: VBM = 0.0 at k=0, CBM = 1.12 at k=2; direct_gap = 1.80.
    eigs_indirect = np.array(
        [[0.0, 2.20], [-0.40, 1.40], [-0.80, 1.12]], dtype=np.float64
    )
    occ_deg = np.array(
        [[2.0, 0.0], [2.0, 0.0], [2.0, 0.0]], dtype=np.float64
    )
    bs_deg = electronic.BandStructure(
        k_distances=k_d, eigenvalues=eigs_indirect, occupations=occ_deg
    )
    res_deg = bs_deg.analyze_band_gap()
    assert res_deg.convention == "spin_degenerate"
    assert res_deg.occupation_threshold == pytest.approx(1.0)
    assert not res_deg.is_metallic
    assert not res_deg.is_indeterminate
    assert res_deg.vbm == pytest.approx(0.0)
    assert res_deg.cbm == pytest.approx(1.12)
    assert res_deg.indirect_gap == pytest.approx(1.12)
    assert res_deg.fundamental_gap == pytest.approx(1.12)
    assert res_deg.direct_gap == pytest.approx(1.80)
    assert bs_deg.band_gap() == pytest.approx(1.12)

    # Smeared insulator: small thermal tails within smearing_tol=0.05 stay
    # insulating instead of being misclassified as metallic!
    occ_smeared = np.array(
        [[1.96, 0.03], [1.99, 0.01], [1.98, 0.02]], dtype=np.float64
    )
    bs_smear = electronic.BandStructure(
        k_distances=k_d, eigenvalues=eigs_indirect, occupations=occ_smeared
    )
    res_smear = bs_smear.analyze_band_gap(smearing_tol=0.05)
    assert not res_smear.is_metallic
    assert not res_smear.is_indeterminate
    assert res_smear.fundamental_gap == pytest.approx(1.12)
    assert "smearing" in res_smear.note

    # Spin-degenerate metal with occ = 1.2 (60% of 2.0) MUST be metallic!
    occ_metal_deg = np.array(
        [[2.0, 1.2], [2.0, 0.0], [2.0, 0.0]], dtype=np.float64
    )
    bs_metal_deg = electronic.BandStructure(
        k_distances=k_d, eigenvalues=eigs_indirect, occupations=occ_metal_deg
    )
    res_metal = bs_metal_deg.analyze_band_gap()
    assert res_metal.convention == "spin_degenerate"
    assert res_metal.is_metallic
    assert res_metal.fundamental_gap == 0.0
    assert bs_metal_deg.band_gap() == 0.0

    # Two-channel spin-polarized direct gap insulator.
    eigs_sp = np.array(
        [
            [[-0.2, 1.3], [-0.5, 1.6], [-0.6, 1.8]],
            [[-0.4, 1.5], [-0.7, 1.7], [-0.8, 2.0]],
        ],
        dtype=np.float64,
    )
    occ_sp = np.array(
        [
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
        ],
        dtype=np.float64,
    )
    bs_sp = electronic.BandStructure(
        k_distances=k_d, eigenvalues=eigs_sp, occupations=occ_sp
    )
    res_sp = bs_sp.analyze_band_gap()
    assert res_sp.convention == "spin_polarized"
    assert res_sp.occupation_threshold == pytest.approx(0.5)
    assert res_sp.direct_gap == pytest.approx(1.5)
    assert res_sp.fundamental_gap == pytest.approx(1.5)

    # Semimetal indirect overlap (CBM < VBM despite integer occupations).
    bs_semimetal = electronic.BandStructure(
        k_distances=np.array([0.0, 1.0]),
        eigenvalues=np.array([[0.2, 1.5], [-1.2, -0.1]]),
        occupations=np.array([[2.0, 0.0], [2.0, 0.0]]),
    )
    res_semi = bs_semimetal.analyze_band_gap()
    assert res_semi.is_metallic
    assert res_semi.fundamental_gap == 0.0
    assert bs_semimetal.band_gap() == 0.0

    # Indeterminate cases: all occupied, all empty, out-of-range occupations.
    bs_all_occ = electronic.BandStructure(
        k_distances=np.array([0.0, 1.0]),
        eigenvalues=np.array([[-2.0, -1.0], [-1.8, -0.8]]),
        occupations=np.full((2, 2), 2.0),
    )
    res_all_occ = bs_all_occ.analyze_band_gap()
    assert res_all_occ.is_indeterminate
    assert not res_all_occ.is_metallic
    assert res_all_occ.fundamental_gap is None
    assert bs_all_occ.band_gap() == 0.0

    bs_all_empty = electronic.BandStructure(
        k_distances=np.array([0.0, 1.0]),
        eigenvalues=np.array([[1.0, 2.0], [1.2, 2.3]]),
        fermi_energy=0.0,
    )
    res_empty = bs_all_empty.analyze_band_gap()
    assert res_empty.convention == "fermi_level"
    assert res_empty.is_indeterminate
    assert res_empty.fundamental_gap is None
    assert bs_all_empty.band_gap() == 0.0

    bs_bad_occ = electronic.BandStructure(
        k_distances=np.array([0.0, 1.0]),
        eigenvalues=np.array([[-1.0, 1.0], [-0.8, 1.2]]),
        occupations=np.array([[3.5, 0.0], [3.5, 0.0]]),
    )
    assert bs_bad_occ.analyze_band_gap().is_indeterminate

    with pytest.raises(exceptions.StructureError):
        bs_deg.analyze_band_gap(occupation_scale="bogus")
    with pytest.raises(exceptions.StructureError):
        bs_deg.analyze_band_gap(smearing_tol=0.6)
    with pytest.raises(exceptions.StructureError):
        bs_deg.analyze_band_gap(occ_threshold=-0.1)


def test_brillouin_zone_exact_volume_lll_skew_and_seekpath(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    si = loader.load("si_diamond.cif")
    bz_si = brillouin.compute_brillouin_zone(si)
    assert bz_si.classification_method == "heuristic"
    assert bz_si.expected_volume > 0.0
    assert bz_si.volume_error_rel < 1e-10
    assert bz_si.volume == pytest.approx(bz_si.expected_volume, rel=1e-10)

    # Strongly unimodular-sheared lattice (det(U) == 1) and high-aspect cell.
    unimod = np.array([[1, 5, 17], [0, 1, 4], [0, 0, 1]], dtype=np.float64)
    base_tri = np.array([[3.2, 0.4, 0.1], [0.2, 3.7, 0.3], [0.1, 0.2, 4.5]])
    skewed = unimod @ base_tri
    red = geometry.lll_reduce_cell(skewed)
    det_red = abs(float(np.linalg.det(red)))
    det_skew = abs(float(np.linalg.det(skewed)))
    assert abs(det_red - det_skew) < 1e-9
    assert float(np.max(np.linalg.norm(red, axis=1))) < 5.0

    bz_skew = brillouin.compute_brillouin_zone(skewed)
    assert bz_skew.volume_error_rel < 1e-10
    assert bz_skew.volume == pytest.approx(bz_skew.expected_volume, rel=1e-10)

    aniso = np.diag([2.8, 3.1, 24.0])
    bz_aniso = brillouin.compute_brillouin_zone(aniso)
    assert bz_aniso.volume_error_rel < 1e-10

    with pytest.raises(exceptions.StructureError):
        geometry.lll_reduce_cell(skewed, delta=1.2)
    with pytest.raises(exceptions.StructureError):
        brillouin.compute_brillouin_zone(si, backend="invalid_backend")

    # Missing seekpath: backend="seekpath" raises OptionalDependencyError,
    # while backend="auto" falls back cleanly to "heuristic".
    monkeypatch.setitem(sys.modules, "seekpath", None)
    with pytest.raises(exceptions.OptionalDependencyError):
        brillouin.compute_brillouin_zone(si, backend="seekpath")
    bz_auto_fallback = brillouin.compute_brillouin_zone(si, backend="auto")
    assert bz_auto_fallback.classification_method == "heuristic"

    # Mocked seekpath present: both "seekpath" and "auto" use it cleanly.
    a0 = 5.4309
    fake_prim = (
        0.5 * a0 * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
    )
    fake_sk = types.SimpleNamespace(
        get_path=lambda _s: {
            "primitive_lattice": fake_prim.tolist(),
            "bravais_lattice": "cF",
            "point_coords": {
                "GAMMA": [0.0, 0.0, 0.0],
                "X": [0.5, 0.0, 0.5],
                "L": [0.5, 0.5, 0.5],
            },
            "path": [("GAMMA", "X"), ("X", "L"), ("L", "GAMMA")],
        }
    )
    monkeypatch.setitem(sys.modules, "seekpath", fake_sk)
    bz_sk = brillouin.compute_brillouin_zone(si, backend="seekpath")
    assert bz_sk.classification_method == "seekpath"
    assert bz_sk.lattice_system == "fcc"
    assert bz_sk.default_path == ("G", "X", "L", "G")
    assert bz_sk.volume_error_rel < 1e-10
    assert brillouin.compute_brillouin_zone(
        si, backend="auto"
    ).classification_method == "seekpath"


def test_eos_uncertainties_extrapolation_and_plot_basis() -> None:
    v0_t = 40.0
    e0_t = -10.8
    b0_gpa_t = 105.0
    b0_ev_t = b0_gpa_t / elements.CODATA2022.ev_per_a3_to_gpa
    bp_t = 4.2
    vols = np.linspace(35.0, 46.0, 9)
    rng = np.random.default_rng(42)
    es_clean = wulff.birch_murnaghan_energy(vols, e0_t, v0_t, b0_ev_t, bp_t)
    es_noisy = es_clean + rng.normal(0.0, 5e-4, size=vols.shape)

    fit = wulff.fit_birch_murnaghan(vols, es_noisy, energy_basis="atom")
    assert fit.fit_status == "ok"
    assert fit.spans_minimum is True
    assert fit.energy_basis == "atom"
    assert fit.v0_err is not None
    assert fit.v0_err > 0.0
    assert fit.e0_err is not None
    assert fit.e0_err > 0.0
    assert fit.b0_gpa_err is not None
    assert fit.b0_gpa_err > 0.0
    assert fit.bp_err is not None
    assert fit.bp_err > 0.0
    assert fit.rmse_ev == pytest.approx(fit.rms_residual)
    assert 0.0 < fit.rmse_ev < 2e-3
    assert 0.99 < fit.r_squared <= 1.0

    # 4-point exact fit (zero degrees of freedom -> pcov inf handled cleanly).
    v4 = np.linspace(36.0, 44.0, 4)
    e4 = wulff.birch_murnaghan_energy(v4, e0_t, v0_t, b0_ev_t, bp_t)
    fit4 = wulff.fit_birch_murnaghan(v4, e4)
    assert fit4.v0 == pytest.approx(v0_t, rel=1e-4)
    assert fit4.b0_gpa_err is None

    # Extrapolated volumes strictly above v0=40.0.
    v_ext = np.linspace(43.0, 50.0, 7)
    e_ext = wulff.birch_murnaghan_energy(v_ext, e0_t, v0_t, b0_ev_t, bp_t)
    fit_ext = wulff.fit_birch_murnaghan(v_ext, e_ext, allow_extrapolation=True)
    assert fit_ext.spans_minimum is False
    assert fit_ext.fit_status == "extrapolated"
    with pytest.raises(exceptions.StructureError):
        wulff.fit_birch_murnaghan(v_ext, e_ext, allow_extrapolation=False)
    with pytest.raises(exceptions.StructureError):
        wulff.fit_birch_murnaghan(vols, es_clean, energy_basis="mol")

    # plot_eos energy_basis y-labels & ink-colored uncertainty/RMSE callout.
    ink = theme.get_preset("classic").ink_color
    for basis, expected_ylabel in (
        ("cell", "Energy (eV/cell)"),
        ("formula_unit", "Energy (eV/f.u.)"),
        ("atom", "Energy (eV/atom)"),
    ):
        ax = thermo.plot_eos(vols, es_noisy, energy_basis=basis)
        assert ax.get_ylabel() == expected_ylabel
        txt_obj = ax.texts[0]
        assert txt_obj.get_color() == ink
        txt = txt_obj.get_text()
        assert "\u00b1" in txt
        assert "RMSE =" in txt
        plt.close("all")


def test_lammps_image_flags_unwrapping_variable_cell_and_rdf(
    tmp_path: pathlib.Path,
) -> None:
    # 1. LAMMPS dump with wrapped x,y,z + ix,iy,iz across a >0.5*L step (+7.0 A
    # in a 10.0 A box): minimum-image alone aliases to -3.0 A, whereas ix=1
    # recovers exact +7.0 A (MSD = 49.0 A^2) with wrapped positions preserved!
    dump_ix = tmp_path / "jump_ix.lammpstrj"
    dump_ix.write_text(
        "ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n1\n"
        "ITEM: BOX BOUNDS pp pp pp\n0.0 10.0\n0.0 10.0\n0.0 10.0\n"
        "ITEM: ATOMS id type element x y z ix iy iz\n"
        "1 1 Cu 1.0 5.0 5.0 0 0 0\n"
        "ITEM: TIMESTEP\n10\nITEM: NUMBER OF ATOMS\n1\n"
        "ITEM: BOX BOUNDS pp pp pp\n0.0 10.0\n0.0 10.0\n0.0 10.0\n"
        "ITEM: ATOMS id type element x y z ix iy iz\n"
        "1 1 Cu 8.0 5.0 5.0 0 0 0\n"
        "ITEM: TIMESTEP\n20\nITEM: NUMBER OF ATOMS\n1\n"
        "ITEM: BOX BOUNDS pp pp pp\n0.0 10.0\n0.0 10.0\n0.0 10.0\n"
        "ITEM: ATOMS id type element x y z ix iy iz\n"
        "1 1 Cu 5.0 5.0 5.0 1 0 0\n",
        encoding="utf-8",
    )
    traj_ix = lammps.read_lammpstrj(dump_ix)
    assert traj_ix[2].positions[0, 0] == pytest.approx(5.0)
    assert traj_ix[2].metadata["image_flags"][0, 0] == 1
    assert traj_ix[2].metadata["unwrapped_positions"][0, 0] == pytest.approx(
        15.0
    )
    with warnings.catch_warnings(record=True) as caught_ix:
        warnings.simplefilter("always")
        msd_ix = traj_ix.msd()
    assert len(caught_ix) == 0
    assert np.allclose(msd_ix, [0.0, 49.0, 196.0])

    # 2. LAMMPS dump with explicit xu,yu,zu sets coords_unwrapped=True.
    dump_xu = tmp_path / "jump_xu.lammpstrj"
    dump_xu.write_text(
        "ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n1\n"
        "ITEM: BOX BOUNDS pp pp pp\n0.0 10.0\n0.0 10.0\n0.0 10.0\n"
        "ITEM: ATOMS id type element xu yu zu\n"
        "1 1 Cu 1.0 5.0 5.0\n"
        "ITEM: TIMESTEP\n10\nITEM: NUMBER OF ATOMS\n1\n"
        "ITEM: BOX BOUNDS pp pp pp\n0.0 10.0\n0.0 10.0\n0.0 10.0\n"
        "ITEM: ATOMS id type element xu yu zu\n"
        "1 1 Cu 15.0 5.0 5.0\n",
        encoding="utf-8",
    )
    traj_xu = lammps.read_lammpstrj(dump_xu)
    assert traj_xu[0].metadata["coords_unwrapped"] is True
    assert np.allclose(traj_xu.msd(), [0.0, 196.0])

    # 3. LAMMPS data file with trailing ix, iy, iz image flags.
    lmp_data = tmp_path / "img.data"
    lmp_data.write_text(
        "Image flags\n1 atoms\n1 atom types\n"
        "0.0 10.0 xlo xhi\n0.0 10.0 ylo yhi\n0.0 10.0 zlo zhi\n\n"
        "Masses\n\n1 63.546\n\nAtoms\n\n"
        "1 1 2.0 3.0 4.0 2 -1 0\n",
        encoding="utf-8",
    )
    at_lmp = lammps.read_lammps_data(lmp_data)
    assert np.allclose(at_lmp.positions[0], [2.0, 3.0, 4.0])
    assert np.allclose(
        at_lmp.metadata["unwrapped_positions"][0], [22.0, -7.0, 4.0]
    )

    # 4. Nyquist jump warning (>45% of Wigner-Seitz inradius) on wrapped frames.
    box10 = np.diag([10.0, 10.0, 10.0])  # r_ws == 5.0 A; 0.45 * r_ws == 2.25 A.
    w0 = structure.AtomsData(
        ["Cu"], np.array([[1.0, 5.0, 5.0]]), cell=box10, pbc=(True, True, True)
    )
    w1 = structure.AtomsData(
        ["Cu"], np.array([[3.6, 5.0, 5.0]]), cell=box10, pbc=(True, True, True)
    )
    traj_warn = trajectory.Trajectory([w0, w1])
    with pytest.warns(UserWarning, match="45%"):
        _ = traj_warn.msd()

    # 5. Variable-cell NPT expansion: fractional vs cartesian unwrapping.
    c0 = np.diag([10.0, 10.0, 10.0])
    c1 = np.diag([22.0, 22.0, 22.0])
    vf0 = structure.AtomsData(
        ["Cu", "Cu"],
        np.array([[0.0, 0.0, 0.0], [9.8, 0.0, 0.0]]),
        cell=c0,
        pbc=(True, True, True),
    )
    vf1 = structure.AtomsData(
        ["Cu", "Cu"],
        np.array([[0.0, 0.0, 0.0], [0.98 * 22.0, 0.0, 0.0]]),
        cell=c1,
        pbc=(True, True, True),
    )
    vtraj = trajectory.Trajectory([vf0, vf1])
    with warnings.catch_warnings(record=True) as caught_frac:
        warnings.simplefilter("always")
        msd_frac = vtraj.msd(variable_cell_mode="fractional")
    assert len(caught_frac) == 0
    expected_dx = 0.98 * 22.0 - 9.8
    assert msd_frac[1] == pytest.approx(0.5 * expected_dx**2)
    assert vtraj.rmsd(variable_cell_mode="fractional").shape == (2,)
    with pytest.raises(exceptions.StructureError):
        vtraj.msd(variable_cell_mode="lagrangian")

    # 6. RDF exact finite-size N_A*(N_A - 1) same-species pair normalization.
    box20 = np.diag([20.0, 20.0, 20.0])
    cu4 = structure.AtomsData(
        symbols=["Cu", "Cu", "Cu", "Cu", "Au"],
        positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [2.5, 0.0, 0.0],
                [0.0, 2.5, 0.0],
                [0.0, 0.0, 2.5],
                [5.0, 5.0, 5.0],
            ]
        ),
        cell=box20,
        pbc=(True, True, True),
    )
    rdf_traj = trajectory.Trajectory([cu4])
    _, g_corr = rdf_traj.compute_rdf(
        ("Cu", "Cu"), r_max=6.0, n_bins=12, same_species_correction=True
    )
    _, g_uncorr = rdf_traj.compute_rdf(
        ("Cu", "Cu"), r_max=6.0, n_bins=12, same_species_correction=False
    )
    mask = g_uncorr > 0.0
    assert np.any(mask)
    # N_A = 4 => N_A^2 / (N_A * (N_A - 1)) = 16 / 12 = 4 / 3.
    assert np.allclose(g_corr[mask] / g_uncorr[mask], 4.0 / 3.0, rtol=1e-12)

    _, g_ab_1 = rdf_traj.compute_rdf(
        ("Cu", "Au"), r_max=9.5, n_bins=12, same_species_correction=True
    )
    _, g_ab_2 = rdf_traj.compute_rdf(
        ("Cu", "Au"), r_max=9.5, n_bins=12, same_species_correction=False
    )
    assert np.allclose(g_ab_1, g_ab_2, rtol=1e-12)


def test_lammps_and_extxyz_time_semantics_and_timeseries_labels(
    tmp_path: pathlib.Path,
) -> None:
    """Verifies D02 LAMMPS/extXYZ time semantics and plot_md_timeseries axes."""
    dump_path = tmp_path / "steps.lammpstrj"
    dump_path.write_text(
        "ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n1\n"
        "ITEM: BOX BOUNDS pp pp pp\n0.0 10.0\n0.0 10.0\n0.0 10.0\n"
        "ITEM: ATOMS id type element x y z\n1 1 Cu 1.0 1.0 1.0\n"
        "ITEM: TIMESTEP\n100\nITEM: NUMBER OF ATOMS\n1\n"
        "ITEM: BOX BOUNDS pp pp pp\n0.0 10.0\n0.0 10.0\n0.0 10.0\n"
        "ITEM: ATOMS id type element x y z\n1 1 Cu 1.5 1.0 1.0\n"
        "ITEM: TIMESTEP\n200\nITEM: NUMBER OF ATOMS\n1\n"
        "ITEM: BOX BOUNDS pp pp pp\n0.0 10.0\n0.0 10.0\n0.0 10.0\n"
        "ITEM: ATOMS id type element x y z\n1 1 Cu 2.0 1.0 1.0\n",
        encoding="utf-8",
    )

    # 1. Default LAMMPS trajectory: times is None, raw steps in metadata.
    trj_raw = lammps.read_lammpstrj(dump_path)
    assert trj_raw.times is None
    assert trj_raw.metadata["time_unit"] is None
    assert np.array_equal(trj_raw.metadata["timesteps"], [0, 100, 200])
    assert trj_raw.metadata["timesteps"].dtype == np.int64
    assert [f.metadata["timestep"] for f in trj_raw.frames] == [0, 100, 200]

    # 2. Explicit timestep_size in fs and ps converts cleanly to canonical fs.
    trj_fs = lammps.read_lammpstrj(
        dump_path, timestep_size=0.5, time_unit="fs"
    )
    assert trj_fs.times is not None
    assert np.allclose(trj_fs.times, [0.0, 50.0, 100.0])
    assert trj_fs.metadata["time_unit"] == "fs"
    assert trj_fs.metadata["timestep_size"] == pytest.approx(0.5)
    assert trj_fs.metadata["timestep_unit"] == "fs"

    trj_ps = loader.load_trajectory(
        dump_path, timestep_size=0.002, time_unit="ps"
    )
    assert trj_ps.times is not None
    assert np.allclose(trj_ps.times, [0.0, 200.0, 400.0])
    assert trj_ps.metadata["time_unit"] == "fs"
    assert trj_ps.metadata["timestep_unit"] == "ps"

    for bad_dt in (0.0, -0.5, float("nan"), float("inf"), True):
        with pytest.raises(exceptions.ParseError):
            lammps.read_lammpstrj(dump_path, timestep_size=bad_dt)  # type: ignore[arg-type]

    # 3. extXYZ time_unit: unspecified vs caller-supplied vs header-supplied.
    water_trj = loader.load_trajectory("water_to_ice.xyz")
    assert water_trj.times is not None
    assert water_trj.metadata["time_unit"] == "unspecified"
    water_ps = xyz.read_xyz_trajectory(
        loader._resolve_input_path("water_to_ice.xyz"), time_unit="ps"
    )
    assert water_ps.times is not None
    assert water_ps.metadata["time_unit"] == "fs"
    assert np.allclose(water_ps.times, water_trj.times * 1000.0)

    ext_ps = tmp_path / "tagged_ps.xyz"
    ext_ps.write_text(
        "1\ntime=0.25 time_unit=ps energy=-1.0\nH 0.0 0.0 0.0\n"
        "1\ntime=0.75 time_unit=ps energy=-0.9\nH 0.2 0.0 0.0\n",
        encoding="utf-8",
    )
    trj_ext_ps = xyz.read_xyz_trajectory(ext_ps)
    assert trj_ext_ps.times is not None
    assert np.allclose(trj_ext_ps.times, [250.0, 750.0])
    assert trj_ext_ps.metadata["time_unit"] == "fs"

    ext_fs = tmp_path / "tagged_fs.xyz"
    ext_fs.write_text(
        "1\ntime=10.0 time_unit=fs\nH 0.0 0.0 0.0\n"
        "1\ntime=25.0 time_unit=fs\nH 0.3 0.0 0.0\n",
        encoding="utf-8",
    )
    trj_ext_fs = xyz.read_xyz_trajectory(ext_fs)
    assert trj_ext_fs.times is not None
    assert np.allclose(trj_ext_fs.times, [10.0, 25.0])
    assert trj_ext_fs.metadata["time_unit"] == "fs"

    # 4. plot_md_timeseries x-axis label & data accuracy across all 4 regimes.
    ax_step = traj_plots.plot_md_timeseries(trj_raw, quantity="rmsd")
    assert ax_step.get_xlabel() == "Timestep"
    assert np.allclose(ax_step.lines[0].get_xdata(), [0.0, 100.0, 200.0])
    plt.close(ax_step.figure)

    ax_fs = traj_plots.plot_md_timeseries(trj_fs, quantity="rmsd")
    assert ax_fs.get_xlabel() == "Time (fs)"
    assert np.allclose(ax_fs.lines[0].get_xdata(), [0.0, 50.0, 100.0])
    plt.close(ax_fs.figure)

    ax_unspec = traj_plots.plot_md_timeseries(water_trj, quantity="energy")
    assert ax_unspec.get_xlabel() == "Time"
    plt.close(ax_unspec.figure)

    neb_trj = loader.load_trajectory("sn2_neb.xyz")
    ax_idx = traj_plots.plot_md_timeseries(neb_trj, quantity="energy")
    assert ax_idx.get_xlabel() == "Frame Index"
    assert np.allclose(ax_idx.lines[0].get_xdata(), np.arange(len(neb_trj)))
    plt.close(ax_idx.figure)


def test_fermi_level_band_gap_tolerance_and_occ_threshold_rejection() -> None:
    """Verifies D04 Fermi-level band-gap tolerance and occ_threshold guard."""
    k_d = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    # Valence top sits at +4e-5 eV above E_F=0.0; CBM sits at +1.12 eV.
    eigs_edge = np.array(
        [[-0.50, 1.40], [4.0e-5, 1.25], [-0.25, 1.12]], dtype=np.float64
    )
    bs_no_occ = electronic.BandStructure(
        k_distances=k_d, eigenvalues=eigs_edge, fermi_energy=0.0
    )
    bs_with_occ = electronic.BandStructure(
        k_distances=k_d,
        eigenvalues=eigs_edge,
        occupations=np.array([[2.0, 0.0], [2.0, 0.0], [2.0, 0.0]]),
        fermi_energy=0.0,
    )

    # Reject occ_threshold whenever Fermi-level mode is active.
    with pytest.raises(exceptions.StructureError, match="fermi_tolerance_ev"):
        bs_no_occ.analyze_band_gap(occ_threshold=0.5)
    with pytest.raises(exceptions.StructureError, match="fermi_tolerance_ev"):
        bs_with_occ.analyze_band_gap(
            occupation_scale="fermi_level", occ_threshold=1.0
        )

    for bad_tol in (-1e-5, float("nan"), float("inf"), True):
        with pytest.raises(exceptions.StructureError):
            bs_no_occ.analyze_band_gap(fermi_tolerance_ev=bad_tol)  # type: ignore[arg-type]

    # Default fermi_tolerance_ev=1e-4 accommodates +4e-5 eV VBM rounding.
    res_loose = bs_no_occ.analyze_band_gap(fermi_tolerance_ev=1e-4)
    assert res_loose.convention == "fermi_level"
    assert not res_loose.is_metallic
    assert res_loose.occupation_threshold == 0.0
    assert res_loose.fermi_tolerance_ev == pytest.approx(1e-4)
    assert res_loose.vbm == pytest.approx(4.0e-5)
    assert res_loose.cbm == pytest.approx(1.12)
    assert res_loose.fundamental_gap == pytest.approx(1.12 - 4.0e-5)

    # Tightening fermi_tolerance_ev=1e-5 flags the +4e-5 eV crossing as metal.
    res_tight = bs_no_occ.analyze_band_gap(fermi_tolerance_ev=1e-5)
    assert res_tight.is_metallic
    assert res_tight.fundamental_gap == 0.0
    assert res_tight.fermi_tolerance_ev == pytest.approx(1e-5)

    # Flat valence band at +0.005 eV above E_F=0.0 partitions via tolerance.
    bs_flat = electronic.BandStructure(
        k_distances=np.array([0.0, 1.0]),
        eigenvalues=np.array([[0.005, 1.50], [0.005, 1.70]]),
        fermi_energy=0.0,
    )
    assert bs_flat.analyze_band_gap(fermi_tolerance_ev=1e-4).is_indeterminate
    res_flat = bs_flat.analyze_band_gap(fermi_tolerance_ev=0.01)
    assert not res_flat.is_metallic
    assert not res_flat.is_indeterminate
    assert res_flat.fundamental_gap == pytest.approx(1.495)


def test_scientific_provenance_deep_immutability_and_json_roundtrip(
    tmp_path: pathlib.Path,
) -> None:
    """Verifies D05 deep freezing and JSON serialization of provenance."""
    with pytest.raises(TypeError):
        core.INTERNAL_UNITS["energy"] = "Ha"  # type: ignore[index]

    raw_nested = {"smearing": np.float64(0.05), "tags": ["pbe", "paw"]}
    raw_list = [1, 2, 3]
    raw_arr = np.array([[1.0, 2.5], [3.5, 4.0]], dtype=np.float64)
    src_path = tmp_path / "OUTCAR"
    cfg_path = tmp_path / "INCAR"

    prov = core.ScientificProvenance(
        algorithm="band_gap_analysis",
        units={"energy": "eV", "length": "A"},
        source=src_path,
        parser="vasp",
        parameters={
            "n_kpts": np.int64(41),
            "converged": np.bool_(True),
            "kmesh": raw_list,
            "species": {"Si", "O"},
            "matrix": raw_arr,
            "nested": raw_nested,
            "incar_path": cfg_path,
        },
        assumptions=["collinear spin"],
    )

    # Mutating original caller containers after construction has zero effect.
    raw_nested["smearing"] = np.float64(9.9)
    raw_list[0] = 99
    raw_arr[0, 0] = -99.0

    with pytest.raises(TypeError):
        prov.units["energy"] = "Ha"  # type: ignore[index]

    param_map = dict(prov.parameters)
    with pytest.raises(TypeError):
        param_map["nested"]["smearing"] = 0.2
    with pytest.raises(TypeError):
        param_map["nested"]["tags"][0] = "lda"
    with pytest.raises(TypeError):
        param_map["kmesh"][0] = 8
    with pytest.raises(ValueError, match="read-only"):
        param_map["matrix"][0, 0] = 42.0

    as_dict = prov.to_dict()
    assert isinstance(as_dict["units"], dict)
    assert as_dict["source"] == str(src_path)
    params_out = as_dict["parameters"]
    assert type(params_out["n_kpts"]) is int
    assert type(params_out["converged"]) is bool
    assert params_out["kmesh"] == [1, 2, 3]
    assert params_out["species"] == ["O", "Si"]
    assert params_out["matrix"] == [[1.0, 2.5], [3.5, 4.0]]
    assert params_out["nested"] == {"smearing": 0.05, "tags": ["pbe", "paw"]}
    assert params_out["incar_path"] == str(cfg_path)

    encoded = json.dumps(as_dict)
    assert json.loads(encoded) == as_dict
