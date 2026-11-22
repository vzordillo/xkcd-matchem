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
"""Tests for verified scientific constants, element tables, and samples."""

from __future__ import annotations

import math

import numpy as np
import pytest

from xkcd_matchem.core import exceptions
from xkcd_matchem.data import elements
from xkcd_matchem.data import samples_registry
from xkcd_matchem.data import scattering


def test_codata2022_constants() -> None:
    c = elements.CODATA2022
    assert math.isclose(c.bohr_to_angstrom, 0.529177210544, rel_tol=1e-12)
    assert math.isclose(c.hartree_to_ev, 27.211386245981, rel_tol=1e-12)
    assert math.isclose(c.cu_kalpha1_angstrom, 1.54059313, rel_tol=1e-8)


def test_elements_table_complete_and_verified() -> None:
    assert len(elements.ELEMENTS_BY_Z) == 118
    assert len(elements.ELEMENTS_BY_SYMBOL) == 118
    h = elements.get_element(1)
    assert h.symbol == "H"
    assert h.vdw_radius == pytest.approx(1.20)
    assert elements.get_element("c").vdw_radius == pytest.approx(1.77)
    assert elements.get_element("O").atomic_mass == pytest.approx(15.999)
    assert elements.get_element("Zr").atomic_mass == pytest.approx(91.222)
    assert elements.get_element("Gd").atomic_mass == pytest.approx(157.249)
    assert elements.get_element("Lu").atomic_mass == pytest.approx(174.96669)
    assert elements.get_element("Og").jmol_color == "#FF1493"
    with pytest.raises(exceptions.StructureError):
        elements.get_element("Xx")


def test_cromer_mann_scattering_factors() -> None:
    assert len(scattering.CROMER_MANN_TABLE) == 102
    # Neutral H uses exact 1s analytical expression: f(0) == 1.0 exactly.
    assert float(scattering.atomic_form_factor("H", 0.0)) == pytest.approx(1.0)
    assert float(scattering.atomic_form_factor("H", 0.5)) < 0.25
    for sym, coeffs in scattering.CROMER_MANN_TABLE.items():
        f0 = float(scattering.atomic_form_factor(sym, 0.0))
        assert abs(f0 - coeffs.atomic_number) < 0.03
        f_high = float(scattering.atomic_form_factor(sym, np.array([1.2]))[0])
        assert 0.0 < f_high < f0


def test_bundled_samples_present() -> None:
    names = samples_registry.list_samples()
    assert len(names) >= 14
    for name in names:
        path = samples_registry.get_sample_path(name)
        assert path.is_file()
        assert path.stat().st_size > 20
        assert samples_registry.describe_sample(name)
    with pytest.raises(FileNotFoundError):
        samples_registry.get_sample_path("nonexistent_sample.xyz")
    with pytest.raises(FileNotFoundError):
        samples_registry.describe_sample("nonexistent_sample.xyz")


def test_codata_unit_conversion_round_trips_and_element_helpers() -> None:
    c = elements.CODATA2022
    # Exact round-trip conversions across length, energy, and wavenumber units.
    for val in (0.529177210544, 1.0, 3.75, 12.4):
        in_ang = val * c.bohr_to_angstrom
        assert in_ang / c.bohr_to_angstrom == pytest.approx(val, rel=1e-15)

        in_ev = val * c.hartree_to_ev
        assert in_ev / c.hartree_to_ev == pytest.approx(val, rel=1e-15)

        in_kj = in_ev * c.ev_to_kj_mol
        assert in_kj / c.ev_to_kj_mol == pytest.approx(in_ev, rel=1e-15)

        in_kcal = in_ev * c.ev_to_kcal_mol
        assert in_kcal / c.ev_to_kcal_mol == pytest.approx(in_ev, rel=1e-15)

        in_cm = in_ev * c.ev_to_cm_inv
        assert in_cm / c.ev_to_cm_inv == pytest.approx(in_ev, rel=1e-15)

    # Physical cross-consistency among CODATA 2022 constants.
    assert c.ev_to_kj_mol == pytest.approx(
        c.ev_to_joule * c.avogadro / 1000.0, rel=1e-12
    )
    assert c.ev_to_kcal_mol == pytest.approx(c.ev_to_kj_mol / 4.184, rel=1e-12)
    assert c.hartree_to_cm_inv == pytest.approx(
        c.hartree_to_ev * c.ev_to_cm_inv, rel=1e-9
    )

    # Helper functions and Z>103 Lr-scaled form factor fallback.
    assert elements.get_atomic_mass("Si") == pytest.approx(28.085)
    assert elements.get_covalent_radius("Si") == pytest.approx(1.16)
    assert elements.get_vdw_radius("Si") == pytest.approx(2.19)
    assert elements.get_jmol_color("Si") == "#F0C8A0"
    assert float(scattering.atomic_form_factor("Og", 0.0)) == pytest.approx(
        float(scattering.atomic_form_factor("Lr", 0.0)) * 118.0 / 103.0
    )
    with pytest.raises(exceptions.StructureError):
        elements.get_element(999)

