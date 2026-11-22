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
"""Independent quantitative scientific validation benchmarks across 9 domains.

Every test in this module validates numerical invariants, analytical closed-form
solutions, or independent brute-force physical reference calculations with
explicit tolerances (zero visual-only smoke tests):

- Domain 1: Skewed triclinic LLL minimum-image & unimodular basis invariance.
- Domain 2: Exact 3D Brillouin zone volume ``(2*pi)^3 / V_prim`` & GL(3, Z).
- Domain 3: Analytical Bragg powder XRD positions, absences, & occupancy^2.
- Domain 4: Electronic band-gap occupation semantics, smearing, & metals.
- Domain 5: Ballistic PBC MD unwrapping, Kabsch RMSD, & analytical FCC RDF.
- Domain 6: 3rd-order Birch-Murnaghan EOS covariance & binary/ternary hulls.
- Domain 7: Spherical 3D Gaussian Marching Tetrahedra & triclinic slice mesh.
- Domain 8: Multi-metric MACE-MPA-0 water-to-ice Ih crystallization benchmark.
- Domain 9: Reversible Nernst/Randles-Sevcik CV, Tafel, NMR, & Shirley XPS.
"""

from __future__ import annotations

import math
import pathlib
import warnings

import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pytest
from scipy import sparse

import xkcd_matchem as xm
from xkcd_matchem.core import brillouin
from xkcd_matchem.core import electronic
from xkcd_matchem.core import exceptions
from xkcd_matchem.core import geometry
from xkcd_matchem.core import isosurface
from xkcd_matchem.core import neighbors
from xkcd_matchem.core import spectroscopy
from xkcd_matchem.core import structure
from xkcd_matchem.core import trajectory
from xkcd_matchem.core import wulff
from xkcd_matchem.data import elements
from xkcd_matchem.data import scattering
from xkcd_matchem.io.native import lammps
from xkcd_matchem.plots import electrochem
from xkcd_matchem.plots import materials
from xkcd_matchem.plots import spectroscopy as spec_plots
from xkcd_matchem.plots import thermo


matplotlib.use("Agg")


def _trapezoid_1d(y_vals: np.ndarray, x_vals: np.ndarray) -> float:
    """Computes 1D composite trapezoidal integral on NumPy 1.24+."""
    return float(
        np.sum(0.5 * (y_vals[:-1] + y_vals[1:]) * np.diff(x_vals))
    )


def _brute_force_supercell_mic(
    disp: np.ndarray,
    cell: np.ndarray,
    half_shell: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluates exact shortest periodic image via (2*half_shell+1)^3 search."""
    frac = disp @ np.linalg.inv(cell)
    wrapped = (frac - np.round(frac)) @ cell
    rng = np.arange(-half_shell, half_shell + 1, dtype=np.int64)
    ga, gb, gc = np.meshgrid(rng, rng, rng, indexing="ij")
    shifts = np.column_stack([ga.ravel(), gb.ravel(), gc.ravel()])
    lattice_vecs = shifts.astype(np.float64) @ cell
    cand = wrapped[:, None, :] + lattice_vecs[None, :, :]
    d_sq = np.sum(cand * cand, axis=2)
    best = np.argmin(d_sq, axis=1)
    best_vecs = cand[np.arange(len(disp)), best]
    return best_vecs, np.sqrt(d_sq[np.arange(len(disp)), best])


def _brute_force_supercell_neighbors(
    atoms: structure.AtomsData,
    max_cutoff: float,
    min_cutoff: float = 1e-6,
    half_shell: int = 2,
) -> list[float]:
    """Enumerates sorted undirected periodic pair distances on a supercell."""
    assert atoms.cell is not None
    wrapped = atoms.wrap_positions()
    pos = wrapped.positions
    cell = atoms.cell
    rng = np.arange(-half_shell, half_shell + 1, dtype=np.int64)
    ga, gb, gc = np.meshgrid(rng, rng, rng, indexing="ij")
    shifts = np.column_stack([ga.ravel(), gb.ravel(), gc.ravel()])
    dists: list[float] = []
    n_at = len(pos)
    for sh_row in shifts:
        sh = (int(sh_row[0]), int(sh_row[1]), int(sh_row[2]))
        offset = sh_row.astype(np.float64) @ cell
        for i_idx in range(n_at):
            for j_idx in range(n_at):
                if i_idx > j_idx:
                    continue
                if i_idx == j_idx and sh <= (0, 0, 0):
                    continue
                dist = float(np.linalg.norm(pos[j_idx] + offset - pos[i_idx]))
                if min_cutoff <= dist <= max_cutoff:
                    dists.append(dist)
    dists.sort()
    return dists


# ============================================================================
# Domain 1 — Triclinic Geometry & LLL Minimum-Image Stress Tests
# ============================================================================


def test_domain1_triclinic_minimum_image_vs_brute_force_supercell() -> None:
    """Verifies LLL minimum-image vectors against exact 5x5x5 supercell search.

    Governing equation:
      d_MIC(dr) = min_{n in Z^3} ||dr + n @ A||_2.
    Tolerance:
      atol = 1e-12 A across acute (20 deg), obtuse (160 deg), 10:1 aspect,
      and unimodular-sheared triclinic cells.
    """
    base_tri = geometry.cell_vectors_from_parameters(
        3.8, 4.3, 4.9, 74.0, 81.0, 66.0
    )
    shear_u = np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1]], dtype=np.float64)
    cells = [
        geometry.cell_vectors_from_parameters(4.3, 4.7, 5.1, 20.0, 24.0, 20.0),
        geometry.cell_vectors_from_parameters(
            4.2, 5.1, 6.3, 82.0, 85.0, 160.0
        ),
        geometry.cell_vectors_from_parameters(
            2.6, 2.9, 26.0, 78.0, 83.0, 68.0
        ),
        shear_u @ base_tri,
    ]
    rng = np.random.default_rng(20261123)
    for cell in cells:
        raw_disp = rng.uniform(-3.5, 3.5, size=(48, 3)) @ cell
        # Confirm 5x5x5 (half_shell=2) matches 7x7x7 (half_shell=3) reference.
        ref_vecs, ref_dists = _brute_force_supercell_mic(
            raw_disp, cell, half_shell=2
        )
        _, check_7 = _brute_force_supercell_mic(raw_disp, cell, half_shell=3)
        np.testing.assert_allclose(ref_dists, check_7, rtol=0.0, atol=1e-13)

        got_vecs = xm.core.minimum_image_vector(
            raw_disp, cell, (True, True, True)
        )
        got_dists = np.linalg.norm(got_vecs, axis=1)
        np.testing.assert_allclose(got_dists, ref_dists, rtol=0.0, atol=1e-12)
        # Non-degenerate displacements match component-wise to 1e-12 A.
        non_tie = np.abs(got_dists - ref_dists) < 1e-13
        assert np.all(non_tie)


def test_domain1_periodic_neighbors_skewed_cells_vs_supercell() -> None:
    """Verifies LLL periodic neighbor search against brute-force 5x5x5 search.

    Governing equation:
      {(i, j, s) : r_min <= ||r_j + s @ A - r_i||_2 <= r_max}.
    Tolerance:
      Exact pair count and sorted distances within atol = 1e-12 A.
    """
    frac_sites = np.array(
        [
            [0.05, 0.12, 0.18],
            [0.37, 0.64, 0.42],
            [0.71, 0.29, 0.83],
            [0.88, 0.91, 0.55],
        ],
        dtype=np.float64,
    )
    cells_and_cutoffs = [
        (
            geometry.cell_vectors_from_parameters(
                4.3, 4.7, 5.1, 20.0, 24.0, 20.0
            ),
            2.4,
        ),
        (
            geometry.cell_vectors_from_parameters(
                4.2, 5.0, 5.8, 83.0, 86.0, 160.0
            ),
            2.5,
        ),
        (
            geometry.cell_vectors_from_parameters(
                2.7, 3.0, 27.0, 79.0, 84.0, 71.0
            ),
            3.8,
        ),
    ]
    for cell, r_cut in cells_and_cutoffs:
        atoms = structure.AtomsData.from_fractional(
            ["Si", "O", "Si", "O"], frac_sites, cell
        )
        ref_d5 = _brute_force_supercell_neighbors(
            atoms, max_cutoff=r_cut, half_shell=2
        )
        ref_d7 = _brute_force_supercell_neighbors(
            atoms, max_cutoff=r_cut, half_shell=3
        )
        assert len(ref_d5) == len(ref_d7)
        np.testing.assert_allclose(ref_d5, ref_d7, rtol=0.0, atol=1e-13)

        hits = xm.find_periodic_neighbors(
            atoms, max_cutoff=r_cut, symmetric=False
        )
        got_d = sorted(h.distance for h in hits)
        assert len(got_d) == len(ref_d5)
        np.testing.assert_allclose(got_d, ref_d5, rtol=0.0, atol=1e-12)

        for h in hits:
            sh_vec = np.asarray(h.shift, dtype=np.float64) @ cell
            exact_vec = atoms.positions[h.j] + sh_vec - atoms.positions[h.i]
            np.testing.assert_allclose(
                h.vector, exact_vec, rtol=0.0, atol=1e-12
            )

        hits_sym = xm.find_periodic_neighbors(
            atoms, max_cutoff=r_cut, symmetric=True
        )
        assert len(hits_sym) == 2 * len(hits)


def test_domain1_unimodular_basis_invariance() -> None:
    """Verifies neighbor distances and counts are invariant under U in GL(3, Z).

    Governing equation:
      L(U @ A) = L(A) for U in GL(3, Z) with |det(U)| = 1 (both +1 and -1).
    Tolerance:
      atol = 1e-12 A on sorted pairwise distances and Wigner-Seitz inradius.
    """
    base_cell = geometry.cell_vectors_from_parameters(
        3.9, 4.6, 5.4, 71.0, 83.0, 108.0
    )
    cart_pos = (
        np.array(
            [
                [0.10, 0.20, 0.30],
                [0.45, 0.55, 0.15],
                [0.75, 0.35, 0.80],
                [0.25, 0.85, 0.65],
                [0.60, 0.15, 0.50],
            ],
            dtype=np.float64,
        )
        @ base_cell
    )
    atoms_ref = structure.AtomsData(
        symbols=["Cu", "Zr", "Cu", "Zr", "Al"],
        positions=cart_pos,
        cell=base_cell,
        pbc=(True, True, True),
    )
    r_cut = 4.45
    hits_ref = neighbors.find_periodic_neighbors(
        atoms_ref, max_cutoff=r_cut, symmetric=True
    )
    dists_ref = np.array(sorted(h.distance for h in hits_ref))
    counts_ref = [sum(1 for h in hits_ref if h.i == idx) for idx in range(5)]
    r_ws_ref = geometry.max_min_image_radius(base_cell)
    vol_ref = atoms_ref.volume()

    unimodular_mats = [
        np.array([[1, 2, 3], [0, 1, 2], [0, 0, 1]], dtype=np.float64),
        np.array([[2, 1, 1], [1, 1, 1], [1, 0, 1]], dtype=np.float64),
        np.array([[1, 2, -1], [0, -1, 2], [0, 0, 1]], dtype=np.float64),
    ]
    expected_dets = [1.0, 1.0, -1.0]
    for u_mat, det_exp in zip(unimodular_mats, expected_dets, strict=True):
        assert float(np.linalg.det(u_mat)) == pytest.approx(det_exp, abs=1e-12)
        cell_u = u_mat @ base_cell
        atoms_u = structure.AtomsData(
            symbols=atoms_ref.symbols,
            positions=cart_pos.copy(),
            cell=cell_u,
            pbc=(True, True, True),
        )
        assert atoms_u.volume() == pytest.approx(vol_ref, rel=1e-12)
        assert geometry.max_min_image_radius(cell_u) == pytest.approx(
            r_ws_ref, abs=1e-12
        )
        hits_u = neighbors.find_periodic_neighbors(
            atoms_u, max_cutoff=r_cut, symmetric=True
        )
        assert len(hits_u) == len(hits_ref)
        counts_u = [sum(1 for h in hits_u if h.i == idx) for idx in range(5)]
        assert counts_u == counts_ref
        dists_u = np.array(sorted(h.distance for h in hits_u))
        np.testing.assert_allclose(dists_u, dists_ref, rtol=0.0, atol=1e-12)


# ============================================================================
# Domain 2 — Brillouin Zone Volume & Unimodular Invariance (A03)
# ============================================================================


@pytest.mark.parametrize(
    ("label", "cell_or_atoms", "expected_v_prim", "expected_facets"),
    [
        (
            "SC",
            np.diag([3.85, 3.85, 3.85]),
            3.85**3,
            6,
        ),
        (
            "FCC_prim",
            0.5
            * 5.4309
            * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.float64),
            (5.4309**3) / 4.0,
            14,
        ),
        (
            "FCC_conv_Si",
            xm.load("si_diamond.cif"),
            (5.4309**3) / 4.0,
            14,
        ),
        (
            "BCC_conv_Fe",
            structure.AtomsData.from_fractional(
                ["Fe", "Fe"],
                np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
                np.diag([2.866, 2.866, 2.866]),
            ),
            (2.866**3) / 2.0,
            12,
        ),
        (
            "HEX",
            geometry.cell_vectors_from_parameters(
                3.16, 3.16, 5.15, 90.0, 90.0, 120.0
            ),
            0.5 * math.sqrt(3.0) * (3.16**2) * 5.15,
            8,
        ),
        (
            "Tetragonal",
            geometry.cell_vectors_from_parameters(
                3.90, 3.90, 6.40, 90.0, 90.0, 90.0
            ),
            (3.90**2) * 6.40,
            6,
        ),
        (
            "Orthorhombic",
            geometry.cell_vectors_from_parameters(
                3.40, 4.70, 6.10, 90.0, 90.0, 90.0
            ),
            3.40 * 4.70 * 6.10,
            6,
        ),
        (
            "Monoclinic",
            geometry.cell_vectors_from_parameters(
                4.60, 5.30, 6.80, 90.0, 114.0, 90.0
            ),
            4.60 * 5.30 * 6.80 * math.sin(math.radians(114.0)),
            None,
        ),
        (
            "Triclinic_obtuse_160",
            geometry.cell_vectors_from_parameters(
                4.20, 5.10, 6.30, 82.0, 85.0, 160.0
            ),
            None,
            None,
        ),
        (
            "Triclinic_acute_20",
            geometry.cell_vectors_from_parameters(
                4.50, 4.80, 5.20, 20.0, 25.0, 20.0
            ),
            None,
            None,
        ),
    ],
)
def test_domain2_brillouin_zone_exact_volume_all_crystal_systems(
    label: str,
    cell_or_atoms: structure.AtomsData | np.ndarray,
    expected_v_prim: float | None,
    expected_facets: int | None,
) -> None:
    """Verifies V_BZ == (2*pi)^3 / V_prim across all 7 crystal systems + skew.

    Governing equation:
      V_BZ = (2*pi)^3 / |det(A_prim)|, volume_error_rel < 1e-10.
    """
    bz = brillouin.compute_brillouin_zone(cell_or_atoms)
    if expected_v_prim is None:
        assert isinstance(cell_or_atoms, np.ndarray)
        v_prim = abs(float(np.linalg.det(cell_or_atoms)))
    else:
        v_prim = expected_v_prim

    analytical_v_bz = ((2.0 * math.pi) ** 3) / v_prim
    assert bz.expected_volume == pytest.approx(
        analytical_v_bz, rel=1e-11
    ), label
    assert bz.volume_error_rel < 1e-10, label
    assert bz.volume == pytest.approx(analytical_v_bz, rel=1e-10), label

    # Independent divergence-theorem polyhedron volume: V = (1/3) sum_f A_f d_f.
    div_volume = sum(f.area * f.distance for f in bz.facets) / 3.0
    assert div_volume == pytest.approx(analytical_v_bz, rel=1e-10), label
    if expected_facets is not None:
        assert len(bz.facets) == expected_facets, label


def test_domain2_brillouin_zone_unimodular_basis_invariance() -> None:
    """Verifies BZ volume and facet geometry are invariant under det(U) = +-1.

    Governing equation:
      BZ(U @ A_prim) is isometric/identical to BZ(A_prim) for U in GL(3, Z).
    Tolerance:
      rel = 1e-10 on volume; exact facet count; rel = 1e-9 on facet areas.
    """
    prim_cells = [
        0.5
        * 5.43
        * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.float64),
        geometry.cell_vectors_from_parameters(
            4.4, 5.2, 6.1, 90.0, 112.0, 90.0
        ),
        geometry.cell_vectors_from_parameters(
            4.1, 4.9, 5.7, 79.0, 84.0, 152.0
        ),
    ]
    unimodular_transforms = [
        np.array([[1, 3, 5], [0, 1, 2], [0, 0, 1]], dtype=np.float64),
        np.array([[1, 2, -1], [0, -1, 2], [0, 0, 1]], dtype=np.float64),
        np.array([[0, 1, 0], [0, 0, 1], [-1, -2, -3]], dtype=np.float64),
    ]
    for cell0 in prim_cells:
        bz0 = brillouin.compute_brillouin_zone(cell0)
        areas0 = np.array([f.area for f in bz0.facets], dtype=np.float64)
        dists0 = np.array(
            sorted(f.distance for f in bz0.facets), dtype=np.float64
        )
        for u_mat in unimodular_transforms:
            assert abs(abs(float(np.linalg.det(u_mat))) - 1.0) < 1e-12
            bzu = brillouin.compute_brillouin_zone(u_mat @ cell0)
            assert bzu.volume_error_rel < 1e-10
            assert bzu.volume == pytest.approx(bz0.volume, rel=1e-10)
            assert len(bzu.facets) == len(bz0.facets)
            areas_u = np.array([f.area for f in bzu.facets], dtype=np.float64)
            dists_u = np.array(
                sorted(f.distance for f in bzu.facets), dtype=np.float64
            )
            np.testing.assert_allclose(areas_u, areas0, rtol=1e-9, atol=1e-10)
            np.testing.assert_allclose(dists_u, dists0, rtol=1e-9, atol=1e-10)


# ============================================================================
# Domain 3 — Powder XRD Analytical Validation (§4.2)
# ============================================================================


def _cubic_bragg_two_theta(
    hkl: tuple[int, int, int],
    a_lat: float,
    wavelength: float,
) -> float:
    """Evaluates exact cubic Bragg angle 2*arcsin(lambda * |hkl| / 2a)."""
    h2_k2_l2 = float(hkl[0] ** 2 + hkl[1] ** 2 + hkl[2] ** 2)
    sin_theta = wavelength * math.sqrt(h2_k2_l2) / (2.0 * a_lat)
    return 2.0 * math.degrees(math.asin(sin_theta))


def test_domain3_diamond_si_bragg_angles_and_systematic_absences() -> None:
    """Verifies Si (Fd-3m) analytical Bragg angles (<0.02 deg) and absences.

    Governing equations:
      2*theta = 2 * arcsin(lambda * sqrt(h^2 + k^2 + l^2) / (2 * a)),
      Fd-3m absences: mixed-parity (hkl) and all-even h + k + l = 4n + 2.
    """
    wavelength = 1.54059313
    assert elements.CODATA2022.cu_kalpha1_angstrom == pytest.approx(
        wavelength, rel=1e-12
    )
    si = xm.load("si_diamond.cif")
    a_cif = float(si.cell[0, 0])
    assert a_cif == pytest.approx(5.431, abs=1e-3)

    spec = xm.compute_xrd(
        si, wavelength=wavelength, two_theta_range=(10.0, 90.0)
    )
    by_label = {pk.label: pk for pk in spec.peaks}

    allowed_hkls = ((1, 1, 1), (2, 2, 0), (3, 1, 1), (4, 0, 0))
    for hkl in allowed_hkls:
        lbl = f"({hkl[0]}{hkl[1]}{hkl[2]})"
        assert lbl in by_label, f"Missing allowed reflection {lbl}"
        tt_exact_cif = _cubic_bragg_two_theta(hkl, a_cif, wavelength)
        tt_nominal = _cubic_bragg_two_theta(hkl, 5.431, wavelength)
        assert abs(by_label[lbl].position - tt_exact_cif) < 1e-4
        assert abs(by_label[lbl].position - tt_nominal) < 0.02

    forbidden_hkls = ((1, 0, 0), (1, 1, 0), (2, 0, 0), (2, 2, 2))
    for hkl in forbidden_hkls:
        lbl = f"({hkl[0]}{hkl[1]}{hkl[2]})"
        assert lbl not in by_label
        tt_forb = _cubic_bragg_two_theta(hkl, a_cif, wavelength)
        assert all(abs(pk.position - tt_forb) > 0.25 for pk in spec.peaks)


def test_domain3_rocksalt_nacl_allowed_reflections_and_absences() -> None:
    """Verifies NaCl (Fm-3m) parity selection rules and Cromer-Mann F(hkl).

    Governing equations:
      F(hkl) = 4*(f_Na + f_Cl) for all-even, 4*(f_Na - f_Cl) for all-odd,
      F(hkl) = 0 for mixed-parity reflections.
    """
    wavelength = elements.CODATA2022.cu_kalpha1_angstrom
    nacl = xm.load("nacl.cif")
    a_lat = float(nacl.cell[0, 0])
    spec = xm.compute_xrd(
        nacl, wavelength=wavelength, two_theta_range=(10.0, 90.0)
    )
    by_label = {pk.label: pk for pk in spec.peaks}

    for hkl in ((1, 1, 1), (2, 0, 0), (2, 2, 0), (2, 2, 2)):
        lbl = f"({hkl[0]}{hkl[1]}{hkl[2]})"
        assert lbl in by_label
        tt_ref = _cubic_bragg_two_theta(hkl, a_lat, wavelength)
        assert abs(by_label[lbl].position - tt_ref) < 0.01

    for hkl in ((1, 0, 0), (1, 1, 0), (2, 1, 0)):
        lbl = f"({hkl[0]}{hkl[1]}{hkl[2]})"
        assert lbl not in by_label
        tt_forb = _cubic_bragg_two_theta(hkl, a_lat, wavelength)
        assert all(abs(pk.position - tt_forb) > 0.25 for pk in spec.peaks)

    # Analytical structure-factor ratio I(111) / I(200) with multiplicity & LP.
    s_111 = np.array([math.sqrt(3.0) / (2.0 * a_lat)])
    s_200 = np.array([math.sqrt(4.0) / (2.0 * a_lat)])
    f_na_111 = float(scattering.atomic_form_factor("Na", s_111)[0])
    f_cl_111 = float(scattering.atomic_form_factor("Cl", s_111)[0])
    f_na_200 = float(scattering.atomic_form_factor("Na", s_200)[0])
    f_cl_200 = float(scattering.atomic_form_factor("Cl", s_200)[0])
    th_111 = 0.5 * math.radians(by_label["(111)"].position)
    th_200 = 0.5 * math.radians(by_label["(200)"].position)
    lp_111 = (1.0 + math.cos(2.0 * th_111) ** 2) / (
        (math.sin(th_111) ** 2) * math.cos(th_111)
    )
    lp_200 = (1.0 + math.cos(2.0 * th_200) ** 2) / (
        (math.sin(th_200) ** 2) * math.cos(th_200)
    )
    expected_ratio = (
        8.0 * lp_111 * ((f_na_111 - f_cl_111) ** 2)
    ) / (6.0 * lp_200 * ((f_na_200 + f_cl_200) ** 2))
    got_ratio = by_label["(111)"].intensity / by_label["(200)"].intensity
    assert got_ratio == pytest.approx(expected_ratio, rel=1e-5)


def _group_reflections_with_tol(
    two_thetas: np.ndarray,
    intensities: np.ndarray,
    merge_tol: float,
    min_intensity: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Clusters reflections by angular merge_tol and normalizes max to 100."""
    entries = sorted(
        (
            (float(two_thetas[i]), float(intensities[i]))
            for i in range(len(two_thetas))
            if float(intensities[i]) > 1e-10
        ),
        key=lambda item: item[0],
    )
    groups: list[tuple[list[float], float]] = []
    for tt, val in entries:
        if groups and abs(tt - float(np.mean(groups[-1][0]))) <= merge_tol:
            groups[-1][0].append(tt)
            groups[-1] = (groups[-1][0], groups[-1][1] + val)
        else:
            groups.append(([tt], val))
    max_val = max((g[1] for g in groups), default=1.0)
    kept = [
        (float(np.mean(tts)), 100.0 * val / max_val)
        for tts, val in groups
        if val / max_val >= min_intensity
    ]
    return (
        np.array([k[0] for k in kept], dtype=np.float64),
        np.array([k[1] for k in kept], dtype=np.float64),
    )


def test_domain3_partial_occupancy_scaling_and_merge_tol_convergence() -> None:
    """Verifies I(occ=0.5)/I(occ=1.0) == 0.25 (<1e-6) and merge_tol stability.

    Governing equation:
      F_sub(hkl; c) = c * F_sub(hkl; 1) => I_sub(0.5) / I_sub(1.0) = 0.25.
    """
    wavelength = elements.CODATA2022.cu_kalpha1_angstrom
    # 1. Raw kinematical intensity scaling on NaCl (all reflections).
    nacl_full = xm.load("nacl.cif")
    nacl_half = nacl_full.copy()
    nacl_half.occupancies = np.full(len(nacl_half), 0.5, dtype=np.float64)
    _, _, i_raw_10 = spectroscopy._xrd_reflections(
        nacl_full, wavelength, 10.0, 90.0, 0.0
    )
    _, _, i_raw_05 = spectroscopy._xrd_reflections(
        nacl_half, wavelength, 10.0, 90.0, 0.0
    )
    nz = i_raw_10 > 1e-8
    assert np.any(nz)
    np.testing.assert_allclose(
        i_raw_05[nz] / i_raw_10[nz], 0.25, rtol=0.0, atol=1e-12
    )

    # 2. Public compute_xrd normalized peak ratio via symmetry-decoupled sites:
    # Pb at (0,0,0) & (0.5,0.5,0) (occ=1) fixes I(001)=100 independent of Si,
    # while Si at (0,0.25,0.25) & (0,0.25,0.75) (occ=c) contributes 0 to (001)
    # and is the sole contributor to (100) (where Pb cancels by C-centering).
    cell_tet = np.diag([4.0, 4.0, 6.0])
    frac_tet = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.25, 0.25],
            [0.0, 0.25, 0.75],
        ],
        dtype=np.float64,
    )
    cryst_10 = structure.AtomsData.from_fractional(
        ["Pb", "Pb", "Si", "Si"],
        frac_tet,
        cell_tet,
        occupancies=np.array([1.0, 1.0, 1.0, 1.0]),
    )
    cryst_05 = structure.AtomsData.from_fractional(
        ["Pb", "Pb", "Si", "Si"],
        frac_tet,
        cell_tet,
        occupancies=np.array([1.0, 1.0, 0.5, 0.5]),
    )
    xrd_10 = xm.compute_xrd(cryst_10, min_intensity=1e-4)
    xrd_05 = xm.compute_xrd(cryst_05, min_intensity=1e-4)
    p10 = {p.label: p.intensity for p in xrd_10.peaks}
    p05 = {p.label: p.intensity for p in xrd_05.peaks}
    assert p10["(001)"] == pytest.approx(100.0, abs=1e-12)
    assert p05["(001)"] == pytest.approx(100.0, abs=1e-12)
    assert abs(p05["(100)"] / p10["(100)"] - 0.25) < 1e-6

    # 3. Angular grouping convergence across merge_tol & 2x2x2 supercell.
    si_unit = xm.load("si_diamond.cif")
    si_super = si_unit.repeat((2, 2, 2))
    spec_unit = xm.compute_xrd(si_unit, two_theta_range=(10.0, 90.0))
    spec_super = xm.compute_xrd(si_super, two_theta_range=(10.0, 90.0))
    ref_pos = np.array([p.position for p in spec_unit.peaks])
    ref_int = np.array([p.intensity for p in spec_unit.peaks])
    np.testing.assert_allclose(
        [p.position for p in spec_super.peaks], ref_pos, rtol=0.0, atol=1e-10
    )
    np.testing.assert_allclose(
        [p.intensity for p in spec_super.peaks], ref_int, rtol=0.0, atol=1e-10
    )

    _, tts_raw, ints_raw = spectroscopy._xrd_reflections(
        si_super, wavelength, 10.0, 90.0, 0.0
    )
    for merge_tol in (1e-5, 1e-4, 1e-3, 1e-2, 5e-2):
        g_pos, g_int = _group_reflections_with_tol(tts_raw, ints_raw, merge_tol)
        assert len(g_pos) == len(ref_pos)
        np.testing.assert_allclose(g_pos, ref_pos, rtol=0.0, atol=1e-6)
        np.testing.assert_allclose(g_int, ref_int, rtol=0.0, atol=1e-6)


# ============================================================================
# Domain 4 — Electronic Structure & Band-Gap Semantics (A04)
# ============================================================================


def test_domain4_spin_degenerate_vs_spin_polarized_exact_gaps() -> None:
    """Verifies exact direct and indirect band gaps under occ=2.0 vs occ=1.0."""
    k_d = np.array([0.0, 0.4, 0.8, 1.2], dtype=np.float64)
    # Indirect spin-degenerate insulator: VBM=-0.10 at k=0, CBM=1.04 at k=2.
    eigs_deg = np.array(
        [
            [-1.50, -0.10, 2.00, 3.10],
            [-1.70, -0.35, 1.45, 2.90],
            [-1.95, -0.60, 1.04, 2.60],
            [-1.80, -0.40, 1.30, 2.80],
        ],
        dtype=np.float64,
    )
    occ_deg = np.array([[2.0, 2.0, 0.0, 0.0]] * 4, dtype=np.float64)
    bs_deg = electronic.BandStructure(
        k_distances=k_d, eigenvalues=eigs_deg, occupations=occ_deg
    )
    res_deg = bs_deg.analyze_band_gap()
    assert res_deg.convention == "spin_degenerate"
    assert res_deg.occupation_threshold == pytest.approx(1.0)
    assert not res_deg.is_metallic
    assert not res_deg.is_indeterminate
    assert res_deg.vbm == pytest.approx(-0.10, abs=1e-12)
    assert res_deg.cbm == pytest.approx(1.04, abs=1e-12)
    assert res_deg.indirect_gap == pytest.approx(1.14, abs=1e-12)
    assert res_deg.fundamental_gap == pytest.approx(1.14, abs=1e-12)
    assert res_deg.direct_gap == pytest.approx(1.64, abs=1e-12)

    # Single-channel spin-polarized direct gap (max occ = 1.0 at Gamma k=0).
    eigs_sp1 = np.array(
        [[0.00, 1.42], [-0.30, 1.80], [-0.55, 2.05], [-0.25, 1.70]],
        dtype=np.float64,
    )
    occ_sp1 = np.array([[1.0, 0.0]] * 4, dtype=np.float64)
    bs_sp1 = electronic.BandStructure(
        k_distances=k_d, eigenvalues=eigs_sp1, occupations=occ_sp1
    )
    res_sp1 = bs_sp1.analyze_band_gap()
    assert res_sp1.convention == "spin_polarized"
    assert res_sp1.occupation_threshold == pytest.approx(0.5)
    assert res_sp1.direct_gap == pytest.approx(1.42, abs=1e-12)
    assert res_sp1.indirect_gap == pytest.approx(1.42, abs=1e-12)
    assert res_sp1.fundamental_gap == pytest.approx(1.42, abs=1e-12)

    # Two-channel spin-polarized indirect gap across spin channels:
    # spin-up VBM = +0.05 at k=0; spin-down CBM = +0.92 at k=2 -> E_g = 0.87 eV.
    eigs_sp2 = np.array(
        [
            [[0.05, 1.60], [-0.20, 1.50], [-0.45, 1.40], [-0.30, 1.55]],
            [[-0.25, 1.35], [-0.40, 1.15], [-0.65, 0.92], [-0.50, 1.20]],
        ],
        dtype=np.float64,
    )
    occ_sp2 = np.broadcast_to(
        np.array([[[1.0, 0.0]] * 4], dtype=np.float64), (2, 4, 2)
    ).copy()
    bs_sp2 = electronic.BandStructure(
        k_distances=k_d, eigenvalues=eigs_sp2, occupations=occ_sp2
    )
    res_sp2 = bs_sp2.analyze_band_gap()
    assert res_sp2.convention == "spin_polarized"
    assert res_sp2.vbm == pytest.approx(0.05, abs=1e-12)
    assert res_sp2.cbm == pytest.approx(0.92, abs=1e-12)
    assert res_sp2.fundamental_gap == pytest.approx(0.87, abs=1e-12)


def test_domain4_fermi_dirac_smeared_insulator_vs_true_metal() -> None:
    """Verifies smeared insulator (0.99/0.01) vs metal crossing E_F (occ=0.50).

    Governing equation:
      f_FD(E) = 1 / (1 + exp((E - mu) / (k_B * T))).
    """
    k_d = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    eigs_ins = np.array(
        [[0.00, 0.90], [-0.25, 1.15], [-0.40, 0.85]], dtype=np.float64
    )
    # Explicit 0.99 at VBM, 0.01 at CBM thermal tail (spin-polarized scale=1.0).
    occ_tail = np.array(
        [[0.99, 0.005], [1.00, 0.001], [0.998, 0.01]], dtype=np.float64
    )
    bs_ins = electronic.BandStructure(
        k_distances=k_d,
        eigenvalues=eigs_ins,
        occupations=occ_tail,
        fermi_energy=0.425,
    )
    res_ins = bs_ins.analyze_band_gap(smearing_tol=0.05)
    assert not res_ins.is_metallic
    assert not res_ins.is_indeterminate
    assert res_ins.fundamental_gap == pytest.approx(0.85, abs=1e-12)
    assert res_ins.direct_gap == pytest.approx(0.90, abs=1e-12)

    # Physical Fermi-Dirac occupations at T = 1050 K across a 0.85 eV gap.
    kt_ev = elements.CODATA2022.boltzmann_ev_k * 1050.0
    occ_fd_deg = 2.0 / (1.0 + np.exp((eigs_ins - 0.425) / kt_ev))
    bs_fd = electronic.BandStructure(
        k_distances=k_d, eigenvalues=eigs_ins, occupations=occ_fd_deg
    )
    res_fd = bs_fd.analyze_band_gap(smearing_tol=0.05)
    assert not res_fd.is_metallic
    assert res_fd.fundamental_gap == pytest.approx(0.85, abs=1e-12)

    # Genuine metal dispersing across E_F = 0.0 eV with occ = 0.50 at crossing.
    eigs_met = np.array(
        [[-0.80, -0.45], [-0.65, 0.00], [-0.50, 0.55]], dtype=np.float64
    )
    occ_met = np.array(
        [[1.00, 1.00], [1.00, 0.50], [1.00, 0.00]], dtype=np.float64
    )
    bs_met = electronic.BandStructure(
        k_distances=k_d,
        eigenvalues=eigs_met,
        occupations=occ_met,
        fermi_energy=0.0,
    )
    res_met = bs_met.analyze_band_gap()
    assert res_met.is_metallic
    assert not res_met.is_indeterminate
    assert res_met.fundamental_gap == 0.0
    assert res_met.direct_gap == 0.0
    assert res_met.indirect_gap == 0.0
    assert res_met.vbm is None
    assert res_met.cbm is None


def test_domain4_ambiguous_and_single_sided_band_windows() -> None:
    """Verifies single-sided or invalid band windows are indeterminate."""
    k_d = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    cases = [
        # All occupied (valence-only window)
        electronic.BandStructure(
            k_distances=k_d,
            eigenvalues=np.array([[-3.0, -1.2], [-2.8, -0.9], [-2.5, -1.1]]),
            occupations=np.full((3, 2), 2.0),
        ),
        # All empty (conduction-only window with occupations)
        electronic.BandStructure(
            k_distances=k_d,
            eigenvalues=np.array([[1.2, 2.5], [1.0, 2.3], [1.4, 2.8]]),
            occupations=np.zeros((3, 2)),
        ),
        # All empty relative to Fermi level (no occupations)
        electronic.BandStructure(
            k_distances=k_d,
            eigenvalues=np.array([[0.8, 2.1], [0.9, 2.4], [1.1, 2.6]]),
            fermi_energy=0.0,
        ),
        # Ambiguous out-of-convention occupation > 2.0
        electronic.BandStructure(
            k_distances=k_d,
            eigenvalues=np.array([[-1.0, 1.0], [-0.8, 1.2], [-0.9, 1.1]]),
            occupations=np.array([[2.8, 0.0], [2.8, 0.0], [2.8, 0.0]]),
        ),
    ]
    for bs_case in cases:
        res = bs_case.analyze_band_gap()
        assert res.is_indeterminate
        assert not res.is_metallic
        assert res.fundamental_gap is None
        assert res.direct_gap is None
        assert res.indirect_gap is None
        assert bs_case.band_gap() == 0.0


# ============================================================================
# Domain 5 — MD Unwrapping, Image Flags & Analytical RDF (§4.1 & A05)
# ============================================================================


def test_domain5_ballistic_pbc_crossing_msd_and_rigid_rmsd(
    tmp_path: pathlib.Path,
) -> None:
    """Verifies MSD(t)=(v*t)^2 (rtol=1e-10) & rigid RMSD=0 across >3 boxes."""
    box_l = 10.0
    vel = np.array([0.82, -0.54, 0.37], dtype=np.float64)  # A / ps
    v_sq = float(np.dot(vel, vel))
    times = np.linspace(0.0, 50.0, 11)  # Crosses >4 box widths along x!
    r0 = np.array([[1.2, 3.4, 7.8], [4.6, 8.1, 2.3]], dtype=np.float64)

    lines_ix: list[str] = []
    lines_xu: list[str] = []
    for step_idx, t_val in enumerate(times):
        u_pos = r0 + t_val * vel[None, :]
        img = np.floor(u_pos / box_l).astype(np.int64)
        w_pos = u_pos - img.astype(np.float64) * box_l
        header = (
            f"ITEM: TIMESTEP\n{step_idx * 100}\n"
            "ITEM: NUMBER OF ATOMS\n2\n"
            "ITEM: BOX BOUNDS pp pp pp\n"
            f"0.0 {box_l}\n0.0 {box_l}\n0.0 {box_l}\n"
        )
        lines_ix.append(header + "ITEM: ATOMS id type element x y z ix iy iz\n")
        lines_xu.append(header + "ITEM: ATOMS id type element xu yu zu\n")
        for a_i in range(2):
            lines_ix.append(
                f"{a_i + 1} 1 Ar {w_pos[a_i, 0]:.12f} {w_pos[a_i, 1]:.12f} "
                f"{w_pos[a_i, 2]:.12f} {img[a_i, 0]} {img[a_i, 1]} "
                f"{img[a_i, 2]}\n"
            )
            lines_xu.append(
                f"{a_i + 1} 1 Ar {u_pos[a_i, 0]:.12f} {u_pos[a_i, 1]:.12f} "
                f"{u_pos[a_i, 2]:.12f}\n"
            )

    p_ix = tmp_path / "ballistic_ix.lammpstrj"
    p_xu = tmp_path / "ballistic_xu.lammpstrj"
    p_ix.write_text("".join(lines_ix), encoding="utf-8")
    p_xu.write_text("".join(lines_xu), encoding="utf-8")

    traj_ix = lammps.read_lammpstrj(p_ix)
    traj_xu = lammps.read_lammpstrj(p_xu)
    expected_msd = v_sq * (times**2)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        msd_ix = traj_ix.msd()
        msd_xu = traj_xu.msd()
    assert len(caught) == 0
    np.testing.assert_allclose(msd_ix, expected_msd, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(msd_xu, expected_msd, rtol=1e-10, atol=1e-12)

    # Rigid chiral 4-atom body undergoing SO(3) rotation + 3.8-box translation.
    cell = np.diag([box_l, box_l, box_l])
    body_local = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.42, 0.0, 0.0],
            [0.35, 1.28, 0.0],
            [0.22, 0.41, 1.19],
        ],
        dtype=np.float64,
    )
    rigid_frames: list[structure.AtomsData] = []
    for k_idx, t_val in enumerate(np.linspace(0.0, 48.0, 9)):
        rot = geometry.rotation_matrix(
            azimuth=27.0 * k_idx, elevation=14.0 * k_idx, roll=19.0 * k_idx
        )
        # Ensure proper rotation det == +1
        if float(np.linalg.det(rot)) < 0.0:
            rot[2] *= -1.0
        shift_com = np.array([3.0, 3.0, 3.0]) + t_val * vel
        unwrapped = body_local @ rot.T + shift_com
        flags = np.floor(unwrapped / box_l).astype(np.int64)
        wrapped = unwrapped - flags.astype(np.float64) * box_l
        rigid_frames.append(
            structure.AtomsData(
                symbols=["C", "O", "N", "F"],
                positions=wrapped,
                cell=cell,
                pbc=(True, True, True),
                metadata={
                    "image_flags": flags,
                    "unwrapped_positions": unwrapped,
                },
            )
        )
    rigid_traj = trajectory.Trajectory(rigid_frames)
    np.testing.assert_allclose(rigid_traj.rmsd(), 0.0, rtol=0.0, atol=1e-12)


def test_domain5_ambiguous_large_wrapped_jump_emits_warning() -> None:
    """Verifies minimum-image step > 0.45 * r_ws triggers UserWarning."""
    cell = geometry.cell_vectors_from_parameters(
        9.0, 9.5, 10.0, 78.0, 84.0, 112.0
    )
    r_ws = geometry.max_min_image_radius(cell, (True, True, True))
    step_vec = np.array([0.48 * r_ws, 0.0, 0.0], dtype=np.float64)
    f0 = structure.AtomsData(
        ["Cu", "Cu"],
        np.array([[1.0, 1.0, 1.0], [3.5, 3.5, 3.5]]),
        cell=cell,
        pbc=(True, True, True),
    )
    f1 = structure.AtomsData(
        ["Cu", "Cu"],
        (f0.positions + step_vec[None, :]),
        cell=cell,
        pbc=(True, True, True),
    )
    ambig_traj = trajectory.Trajectory([f0, f1])
    with pytest.warns(UserWarning, match="45%"):
        _ = ambig_traj.msd()
    with pytest.warns(UserWarning, match="45%"):
        _ = ambig_traj.rmsd()


def test_domain5_analytical_fcc_rdf_peak_and_coordination_number() -> None:
    """Verifies FCC first-shell r_1 = a/sqrt(2) and integral N_1 == 12.0.

    Governing equations:
      r_1 = a / sqrt(2), r_2 = a,
      N_1 = 4 * pi * rho * integral_0^{r_min} r^2 * g(r) dr == 12.0.
    Tolerance:
      atol = 0.05 on coordination number N_1; bin-width precision on r_1, r_2.
    """
    a_fcc = 3.615
    fcc_conv = structure.AtomsData.from_fractional(
        ["Cu", "Cu", "Cu", "Cu"],
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.5],
                [0.5, 0.5, 0.0],
            ],
            dtype=np.float64,
        ),
        np.diag([a_fcc, a_fcc, a_fcc]),
    )
    fcc_super = fcc_conv.repeat((5, 5, 5))  # N = 500 atoms
    r_max = 6.0
    n_bins = 300
    dr = r_max / n_bins
    r_centers, g_r = trajectory.Trajectory([fcc_super]).compute_rdf(
        ("Cu", "Cu"),
        r_max=r_max,
        n_bins=n_bins,
        same_species_correction=True,
    )

    r1_exact = a_fcc / math.sqrt(2.0)
    mask_shell_1 = r_centers < 0.5 * (r1_exact + a_fcc)
    r1_peak = float(r_centers[mask_shell_1][np.argmax(g_r[mask_shell_1])])
    assert abs(r1_peak - r1_exact) <= 0.5 * dr + 1e-12

    mask_shell_2 = (r_centers > 3.1) & (r_centers < 4.0)
    r2_peak = float(r_centers[mask_shell_2][np.argmax(g_r[mask_shell_2])])
    assert abs(r2_peak - a_fcc) <= 0.5 * dr + 1e-12

    rho_macro = len(fcc_super) / fcc_super.volume()
    integrand = 4.0 * math.pi * rho_macro * (r_centers[mask_shell_1] ** 2) * (
        g_r[mask_shell_1]
    )
    n1_coord = _trapezoid_1d(integrand, r_centers[mask_shell_1])
    assert n1_coord == pytest.approx(12.0, abs=0.05)

    # Exact finite-size background density (N - 1) / V gives 12.0 to <1e-4.
    rho_finite = (len(fcc_super) - 1) / fcc_super.volume()
    n1_exact_finite = n1_coord * (rho_finite / rho_macro)
    assert n1_exact_finite == pytest.approx(12.0, abs=1e-4)


# ============================================================================
# Domain 6 — Thermodynamics (EOS & Convex Hull, §4.3)
# ============================================================================


def test_domain6_birch_murnaghan_exact_noisy_and_extrapolated() -> None:
    """Verifies noiseless, noisy (3-sigma), and extrapolated BM3 EOS fits.

    Governing equation:
      E(V) = E0 + (9*V0*B0/16) * ([(V0/V)^(2/3) - 1]^3 * B0'
                                 + [(V0/V)^(2/3) - 1]^2 * [6 - 4*(V0/V)^(2/3)]),
      P(V) = -dE/dV.
    """
    v0_true = 40.88
    e0_true = -10.842
    b0_gpa_true = 98.5
    b0_ev_true = b0_gpa_true / elements.CODATA2022.ev_per_a3_to_gpa
    bp_true = 4.25

    vols = np.linspace(0.86 * v0_true, 1.16 * v0_true, 13)
    es_exact = wulff.birch_murnaghan_energy(
        vols, e0_true, v0_true, b0_ev_true, bp_true
    )

    # Verify analytical pressure P(V) == -dE/dV via central finite differences.
    dv = 1e-4
    p_fd = -(
        wulff.birch_murnaghan_energy(
            vols + dv, e0_true, v0_true, b0_ev_true, bp_true
        )
        - wulff.birch_murnaghan_energy(
            vols - dv, e0_true, v0_true, b0_ev_true, bp_true
        )
    ) / (2.0 * dv)
    p_an = wulff.birch_murnaghan_pressure(
        vols, v0_true, b0_ev_true, bp_true, unit="eV/A^3"
    )
    np.testing.assert_allclose(p_an, p_fd, rtol=1e-9, atol=1e-10)

    fit_clean = wulff.fit_birch_murnaghan(vols, es_exact)
    assert fit_clean.fit_status == "ok"
    assert fit_clean.spans_minimum is True
    assert fit_clean.v0 == pytest.approx(v0_true, rel=1e-9)
    assert fit_clean.e0 == pytest.approx(e0_true, rel=1e-9)
    assert fit_clean.b0_gpa == pytest.approx(b0_gpa_true, rel=1e-8)
    assert fit_clean.b0_prime == pytest.approx(bp_true, rel=1e-8)
    assert fit_clean.rmse_ev < 1e-12
    assert fit_clean.r_squared == pytest.approx(1.0, abs=1e-12)

    # Noisy synthetic curve: parameters recovered within 3-sigma covariance!
    rng = np.random.default_rng(314159)
    es_noisy = es_exact + rng.normal(0.0, 2.5e-4, size=vols.shape)
    fit_noisy = wulff.fit_birch_murnaghan(vols, es_noisy, energy_basis="atom")
    assert fit_noisy.fit_status == "ok"
    assert fit_noisy.spans_minimum is True
    assert fit_noisy.r_squared > 0.999
    assert fit_noisy.v0_err is not None
    assert fit_noisy.v0_err > 0.0
    assert fit_noisy.e0_err is not None
    assert fit_noisy.e0_err > 0.0
    assert fit_noisy.b0_gpa_err is not None
    assert fit_noisy.b0_gpa_err > 0.0
    assert fit_noisy.bp_err is not None
    assert fit_noisy.bp_err > 0.0
    assert abs(fit_noisy.v0 - v0_true) <= 3.0 * fit_noisy.v0_err
    assert abs(fit_noisy.e0 - e0_true) <= 3.0 * fit_noisy.e0_err
    assert abs(fit_noisy.b0_gpa - b0_gpa_true) <= 3.0 * fit_noisy.b0_gpa_err
    assert abs(fit_noisy.b0_prime - bp_true) <= 3.0 * fit_noisy.bp_err

    # Extrapolated window strictly above V0.
    v_high = np.linspace(44.5, 52.5, 8)
    e_high = wulff.birch_murnaghan_energy(
        v_high, e0_true, v0_true, b0_ev_true, bp_true
    )
    fit_ext = wulff.fit_birch_murnaghan(
        v_high, e_high, allow_extrapolation=True
    )
    assert fit_ext.spans_minimum is False
    assert fit_ext.fit_status == "extrapolated"
    assert fit_ext.v0 == pytest.approx(v0_true, rel=1e-5)
    with pytest.raises(exceptions.StructureError):
        wulff.fit_birch_murnaghan(v_high, e_high, allow_extrapolation=False)


def test_domain6_binary_and_ternary_convex_hull_exact_ehull() -> None:
    """Verifies exact binary tie-line vertices and E_hull distances above hull.

    Governing equation:
      E_hull(x) = max_{lambda >= 0, sum lambda_k = 1, sum lambda_k x_k = x}
                  min sum_k lambda_k E_k,
      Delta E_hull(i) = E_i - E_hull(x_i) >= 0.
    """
    # Ground states: A (0.0, 0.0), A3B (0.25, -0.40), AB (0.50, -0.45), B (1, 0)
    compositions = np.array(
        [0.00, 0.25, 0.25, 0.40, 0.50, 0.75, 1.00], dtype=np.float64
    )
    energies = np.array(
        [0.00, -0.40, -0.28, 0.08, -0.45, -0.10, 0.00], dtype=np.float64
    )
    expected_stable = {0, 1, 4, 6}
    # At x=0.40 on (0.25,-0.40)->(0.50,-0.45): E_hull = -0.43 => dE = 0.51.
    # At x=0.75 on (0.50,-0.45)->(1.00, 0.00): E_hull = -0.225 => dE = 0.125.
    expected_ehull = np.array(
        [0.00, 0.00, 0.12, 0.51, 0.00, 0.125, 0.00], dtype=np.float64
    )

    stable_idx = thermo._lower_hull_indices(compositions, energies)
    assert stable_idx == expected_stable

    hull_sorted = sorted(stable_idx, key=lambda idx: float(compositions[idx]))
    hull_x = compositions[hull_sorted]
    hull_y = energies[hull_sorted]
    e_above = energies - np.interp(compositions, hull_x, hull_y)
    np.testing.assert_allclose(e_above, expected_ehull, rtol=0.0, atol=1e-12)

    # Verify plot_phase_diagram_2d encodes exact hull vertices & drop segments.
    ax = xm.plot_phase_diagram_2d(compositions, energies)
    try:
        hull_lines = [
            ln for ln in ax.lines if ln.get_label() == "Convex Hull"
        ]
        assert len(hull_lines) == 1
        np.testing.assert_allclose(
            hull_lines[0].get_xdata(), [0.0, 0.25, 0.50, 1.0], atol=1e-12
        )
        np.testing.assert_allclose(
            hull_lines[0].get_ydata(), [0.0, -0.40, -0.45, 0.0], atol=1e-12
        )
        drop_lines = [ln for ln in ax.lines if ln.get_linestyle() == ":"]
        assert len(drop_lines) == 3
        drop_heights = sorted(
            abs(float(ln.get_ydata()[1]) - float(ln.get_ydata()[0]))
            for ln in drop_lines
        )
        np.testing.assert_allclose(
            drop_heights, [0.12, 0.125, 0.51], rtol=0.0, atol=1e-12
        )
    finally:
        plt.close(ax.figure)

    # Verify ternary simplex lower hull ground-state vertices & tie-lines.
    tern_comp = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        ],
        dtype=np.float64,
    )
    tern_e = np.array([0.0, 0.0, 0.0, -0.60, -0.25], dtype=np.float64)
    xy_tern = materials._barycentric_to_xy(tern_comp)
    t_stable, t_edges = materials._ternary_lower_hull(xy_tern, tern_e)
    assert t_stable == {0, 1, 2, 3}
    assert t_edges == {(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)}


# ============================================================================
# Domain 7 — Volumetric Grids & Marching Tetrahedra Isosurfaces
# ============================================================================


def test_domain7_spherical_gaussian_isosurface_and_disjoint_lobes() -> None:
    """Verifies 3D Gaussian isosurface radius (<0.04 A), area (<5%), & 2 lobes.

    Governing equations:
      f(r) = exp(-r^2 / (2 * sigma^2)) = c  =>  R = sqrt(-2 * sigma^2 * ln(c)),
      A_sphere = 4 * pi * R^2.
    """
    sigma = 1.25
    c_level = 0.30
    r_analytical = math.sqrt(-2.0 * (sigma**2) * math.log(c_level))
    area_analytical = 4.0 * math.pi * (r_analytical**2)

    n_pts = 48
    axis_1d = np.linspace(-4.5, 4.5, n_pts)
    step = float(axis_1d[1] - axis_1d[0])
    gx, gy, gz = np.meshgrid(axis_1d, axis_1d, axis_1d, indexing="ij")
    field = np.exp(-(gx**2 + gy**2 + gz**2) / (2.0 * (sigma**2)))
    grid = electronic.VolumetricGrid(
        origin=np.array([-4.5, -4.5, -4.5]),
        voxel_vectors=np.diag([step, step, step]),
        data=field,
    )

    mesh = isosurface.extract_isosurface(grid, c_level)
    assert len(mesh.vertices) > 500
    assert len(mesh.faces) > 1000

    radii = np.linalg.norm(mesh.vertices, axis=1)
    mean_radial_err = float(np.mean(np.abs(radii - r_analytical)))
    max_radial_err = float(np.max(np.abs(radii - r_analytical)))
    assert mean_radial_err < 0.04
    assert max_radial_err < 0.04

    v0 = mesh.vertices[mesh.faces[:, 0]]
    v1 = mesh.vertices[mesh.faces[:, 1]]
    v2 = mesh.vertices[mesh.faces[:, 2]]
    tri_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    area_mesh = float(np.sum(tri_areas))
    assert abs(area_mesh - area_analytical) / area_analytical < 0.05

    # Verify outward face normals align with radial outward vectors.
    face_centers = (v0 + v1 + v2) / 3.0
    radial_dirs = face_centers / np.linalg.norm(
        face_centers, axis=1, keepdims=True
    )
    cos_align = np.einsum("ij,ij->i", mesh.normals, radial_dirs)
    assert float(np.mean(cos_align)) > 0.99

    # Two-center separated Gaussian lobes at (-3.0, 0, 0) and (+3.0, 0, 0).
    ax_x = np.linspace(-6.5, 6.5, 56)
    ax_yz = np.linspace(-3.5, 3.5, 32)
    dx = float(ax_x[1] - ax_x[0])
    dyz = float(ax_yz[1] - ax_yz[0])
    tx, ty, tz = np.meshgrid(ax_x, ax_yz, ax_yz, indexing="ij")
    sig2 = 0.85**2
    two_lobe = np.exp(-((tx + 3.0) ** 2 + ty**2 + tz**2) / (2.0 * sig2)) + (
        np.exp(-((tx - 3.0) ** 2 + ty**2 + tz**2) / (2.0 * sig2))
    )
    grid_2 = electronic.VolumetricGrid(
        origin=np.array([-6.5, -3.5, -3.5]),
        voxel_vectors=np.diag([dx, dyz, dyz]),
        data=two_lobe,
    )
    mesh_2 = isosurface.extract_isosurface(grid_2, 0.45)
    n_verts = len(mesh_2.vertices)
    edges_a = np.concatenate(
        [mesh_2.faces[:, 0], mesh_2.faces[:, 1], mesh_2.faces[:, 2]]
    )
    edges_b = np.concatenate(
        [mesh_2.faces[:, 1], mesh_2.faces[:, 2], mesh_2.faces[:, 0]]
    )
    adj = sparse.coo_matrix(
        (np.ones(len(edges_a), dtype=np.int8), (edges_a, edges_b)),
        shape=(n_verts, n_verts),
    )
    n_comp, comp_labels = sparse.csgraph.connected_components(
        adj, directed=False
    )
    assert n_comp == 2
    c0 = np.mean(mesh_2.vertices[comp_labels == 0], axis=0)
    c1 = np.mean(mesh_2.vertices[comp_labels == 1], axis=0)
    centers_sorted = sorted([c0, c1], key=lambda c: float(c[0]))
    np.testing.assert_allclose(centers_sorted[0], [-3.0, 0.0, 0.0], atol=0.02)
    np.testing.assert_allclose(centers_sorted[1], [3.0, 0.0, 0.0], atol=0.02)


def test_domain7_triclinic_volumetric_slice_mesh_isometry() -> None:
    """Verifies affine-skewed VolumetricGrid.slice_mesh() coordinate transform.

    Governing equations:
      r(i, j, k) = r_0 + i*v_0 + j*v_1 + k*v_2,
      basis @ basis.T = I_2, ||(u_p - u_q, v_p - v_q)||_2 == ||r_p - r_q||_2.
    """
    triclinic_cell = geometry.cell_vectors_from_parameters(
        6.4, 7.2, 8.1, 74.0, 81.0, 118.0
    )
    nx, ny, nz = 12, 14, 10
    voxel_vecs = triclinic_cell / np.array([[nx], [ny], [nz]], dtype=np.float64)
    origin = np.array([-1.25, 0.80, -0.45], dtype=np.float64)

    gi, gj, gk = np.meshgrid(
        np.arange(nx, dtype=np.float64),
        np.arange(ny, dtype=np.float64),
        np.arange(nz, dtype=np.float64),
        indexing="ij",
    )
    pts_3d = (
        origin[None, None, None, :]
        + gi[:, :, :, None] * voxel_vecs[0][None, None, None, :]
        + gj[:, :, :, None] * voxel_vecs[1][None, None, None, :]
        + gk[:, :, :, None] * voxel_vecs[2][None, None, None, :]
    )
    k_vec = np.array([0.7, -0.4, 1.1], dtype=np.float64)
    analytic_data = np.einsum("ijkl,l->ijk", pts_3d, k_vec) + 0.08 * np.sum(
        pts_3d**2, axis=3
    )
    grid = electronic.VolumetricGrid(
        origin=origin, voxel_vectors=voxel_vecs, data=analytic_data
    )

    for ax_name, ax_idx, sl_idx in (("x", 0, 5), ("y", 1, 9), ("z", 2, 4)):
        uu, vv, plane, basis = grid.slice_mesh(axis=ax_name, index=sl_idx)
        assert basis.shape == (2, 3)
        np.testing.assert_allclose(
            basis @ basis.T, np.eye(2), rtol=0.0, atol=1e-12
        )

        pts_slice = np.take(pts_3d, sl_idx, axis=ax_idx)
        np.testing.assert_allclose(
            uu, pts_slice @ basis[0], rtol=0.0, atol=1e-12
        )
        np.testing.assert_allclose(
            vv, pts_slice @ basis[1], rtol=0.0, atol=1e-12
        )
        np.testing.assert_allclose(
            plane, np.take(analytic_data, sl_idx, axis=ax_idx), atol=1e-12
        )

        # In-plane 2D Euclidean distances match exact 3D distances everywhere.
        uv_flat = np.column_stack([uu.ravel(), vv.ravel()])
        r3_flat = pts_slice.reshape(-1, 3)
        d_2d = np.linalg.norm(uv_flat - uv_flat[0:1], axis=1)
        d_3d = np.linalg.norm(r3_flat - r3_flat[0:1], axis=1)
        np.testing.assert_allclose(d_2d, d_3d, rtol=0.0, atol=1e-12)


# ============================================================================
# Domain 8 — Independent Multi-Metric Water-to-Ice MLIP Validation (§4.4)
# ============================================================================


def _independent_tetrahedral_q(atoms: structure.AtomsData) -> float:
    """Computes Errington-Debenedetti q_O via independent brute-force MIC."""
    assert atoms.cell is not None
    o_pos = np.array(
        [
            atoms.positions[i]
            for i, s in enumerate(atoms.symbols)
            if s == "O"
        ],
        dtype=np.float64,
    )
    q_list: list[float] = []
    for i_idx in range(len(o_pos)):
        raw_d = o_pos - o_pos[i_idx : i_idx + 1]
        vecs, dists = _brute_force_supercell_mic(
            raw_d, atoms.cell, half_shell=1
        )
        order = np.argsort(dists)[1:5]
        u_vecs = vecs[order] / dists[order, None]
        acc = 0.0
        for j_idx in range(3):
            for k_idx in range(j_idx + 1, 4):
                cos_jk = float(np.dot(u_vecs[j_idx], u_vecs[k_idx]))
                acc += (cos_jk + 1.0 / 3.0) ** 2
        q_list.append(1.0 - (3.0 / 8.0) * acc)
    return float(np.mean(q_list))


def test_domain8_water_to_ice_mlip_four_independent_metrics() -> None:
    """Validates bundled water_to_ice.xyz (28 frames) across 4 physical metrics.

    Metrics:
      1. Errington-Debenedetti q increases from <0.80 to >=0.95 (0 K).
      2. Geometric H-bonds saturate (>=60 in final 0 K state, > initial).
      3. Final O-O RDF peaks at r_1 in [2.60, 2.90] A and r_2 in [4.25, 4.75] A
         with tetrahedral ratio r_2 / r_1 == sqrt(8/3) = 1.633 within +-0.12.
      4. Exact 64/64 intact covalent O-H bonds (deg(O)=2, deg(H)=1) and
         exothermic crystallization energy drop dE < -5.0 eV.
    """
    traj = xm.load_trajectory("water_to_ice.xyz")
    assert len(traj) == 28
    first_frame = traj[0]
    final_frame = traj[-1]
    assert len(final_frame) == 96

    # Metric 1: Errington-Debenedetti tetrahedral order q.
    q_series = traj.tetrahedral_order(center_symbol="O")
    assert q_series[0] < 0.80
    assert q_series[-1] >= 0.95
    assert q_series[-1] == pytest.approx(
        _independent_tetrahedral_q(final_frame), abs=1e-12
    )

    # Metric 2: Geometric O-H...O hydrogen-bond network saturation.
    hb_init = xm.infer_hbonds(first_frame)
    hb_final = xm.infer_hbonds(final_frame)
    assert len(hb_final) >= 60
    assert len(hb_final) > len(hb_init)

    # Metric 3: Periodic O-O RDF Ice Ih first & second tetrahedral shells.
    r_oo, g_oo = trajectory.Trajectory([final_frame]).compute_rdf(
        ("O", "O"), r_max=6.0, n_bins=120, same_species_correction=True
    )
    mask_1 = (r_oo >= 2.30) & (r_oo <= 3.20)
    mask_2 = (r_oo >= 3.80) & (r_oo <= 5.10)
    r_1 = float(r_oo[mask_1][np.argmax(g_oo[mask_1])])
    r_2 = float(r_oo[mask_2][np.argmax(g_oo[mask_2])])
    assert 2.60 <= r_1 <= 2.90
    assert 4.25 <= r_2 <= 4.75
    ideal_tet_ratio = math.sqrt(8.0 / 3.0)
    assert abs((r_2 / r_1) - ideal_tet_ratio) <= 0.12

    # Metric 4: Exact 64/64 intact covalent O-H bonds and dE < -5.0 eV.
    bonds_final = xm.infer_bonds(final_frame)
    oh_bonds = [
        b
        for b in bonds_final
        if {final_frame.symbols[b.i], final_frame.symbols[b.j]} == {"O", "H"}
    ]
    assert len(bonds_final) == 64
    assert len(oh_bonds) == 64
    deg = [0] * len(final_frame)
    for b in oh_bonds:
        deg[b.i] += 1
        deg[b.j] += 1
    for idx, sym in enumerate(final_frame.symbols):
        assert deg[idx] == (2 if sym == "O" else 1)

    assert traj.energies is not None
    delta_e = float(traj.energies[-1] - traj.energies[0])
    assert delta_e < -5.0


# ============================================================================
# Domain 9 — Electrochemistry & Analytical Spectroscopy
# ============================================================================


def _simulate_nernstian_planar_cv(
    n_electrons: int = 1,
    e0_v: float = 0.25,
    e_start_v: float = -0.10,
    e_vertex_v: float = 0.45,
    scan_rate_v_s: float = 0.10,
    temp_k: float = 298.15,
    n_half_steps: int = 1800,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Solves reversible Nernst-Fick planar CV via Duhamel-Abel convolution.

    Returns ``(potential_v, current_ma_cm2, randles_sevcik_peak_ma_cm2)``.
    """
    r_gas = (
        elements.CODATA2022.boltzmann_ev_k
        * elements.CODATA2022.ev_to_joule
        * elements.CODATA2022.avogadro
    )
    faraday = elements.CODATA2022.ev_to_joule * elements.CODATA2022.avogadro
    t_half = abs(e_vertex_v - e_start_v) / scan_rate_v_s
    t_grid = np.linspace(0.0, 2.0 * t_half, 2 * n_half_steps + 1)
    dt = float(t_grid[1] - t_grid[0])
    e_t = np.where(
        t_grid <= t_half,
        e_start_v + scan_rate_v_s * t_grid,
        e_vertex_v - scan_rate_v_s * (t_grid - t_half),
    )
    theta = np.exp(n_electrons * faraday * (e_t - e0_v) / (r_gas * temp_k))
    c_surf = theta / (1.0 + theta)
    dc = np.diff(c_surf)
    k_idx = np.arange(len(dc), dtype=np.float64)
    kernel = (2.0 / math.sqrt(math.pi * dt)) * (
        np.sqrt(k_idx + 1.0) - np.sqrt(k_idx)
    )
    flux = np.concatenate(
        [[0.0], np.convolve(dc, kernel, mode="full")[: len(dc)]]
    )
    c_star = 1.0e-6  # 1 mM in mol/cm^3
    d_cm2_s = 1.0e-5
    j_ma_cm2 = (
        1000.0 * n_electrons * faraday * c_star * math.sqrt(d_cm2_s) * flux
    )
    j_rs = (
        1000.0
        * 0.44629
        * (n_electrons**1.5)
        * (faraday**1.5)
        * c_star
        * math.sqrt(d_cm2_s * scan_rate_v_s / (r_gas * temp_k))
    )
    return e_t, j_ma_cm2, j_rs


def test_domain9_nernst_randles_sevcik_cv_and_tafel_slope() -> None:
    """Verifies reversible Nernst CV Delta E_p (~59 mV/n) & exact Tafel slope.

    Governing equations:
      Delta E_p = E_pa - E_pc ~ 57-59 mV / n at 298.15 K,
      j_p = 0.4463 * n * F * C* * sqrt(n * F * v * D / (R * T)),
      eta = b * (log10(|j|) - log10(j_0)).
    """
    for n_el in (1, 2):
        e_cv, j_cv, j_rs = _simulate_nernstian_planar_cv(
            n_electrons=n_el, e0_v=0.25, temp_k=298.15
        )
        e_pa, j_pa, e_pc, _, e_half = electrochem._detect_cv_peaks(e_cv, j_cv)
        dep_mv = (e_pa - e_pc) * 1000.0
        assert abs(dep_mv - (58.5 / n_el)) <= 2.2 / n_el
        assert e_half == pytest.approx(0.25, abs=1.5e-3)
        assert j_pa == pytest.approx(j_rs, rel=3e-3)

    # Exact asymptotic Tafel law recovery (atol = 1e-6).
    r_gas = (
        elements.CODATA2022.boltzmann_ev_k
        * elements.CODATA2022.ev_to_joule
        * elements.CODATA2022.avogadro
    )
    faraday = elements.CODATA2022.ev_to_joule * elements.CODATA2022.avogadro
    alpha_a = 0.5
    b_exact_mv = 1000.0 * math.log(10.0) * r_gas * 298.15 / (alpha_a * faraday)
    j0_true = 1.0e-3
    eta_mv = np.linspace(180.0, 380.0, 41)
    j_tafel_exact = j0_true * (10.0 ** (eta_mv / b_exact_mv))
    _, _, _, fit_exact = electrochem._prepare_tafel_fit(
        eta_mv, j_tafel_exact, "mV", None
    )
    assert fit_exact[2] == pytest.approx(b_exact_mv, abs=1e-6)
    assert fit_exact[3] == pytest.approx(math.log10(j0_true), abs=1e-6)

    # Full Butler-Volmer kinetics in high-overpotential Tafel regime [220, 380].
    eta_v = eta_mv / 1000.0
    j_bv = j0_true * (
        np.exp(alpha_a * faraday * eta_v / (r_gas * 298.15))
        - np.exp(-(1.0 - alpha_a) * faraday * eta_v / (r_gas * 298.15))
    )
    _, _, _, fit_bv = electrochem._prepare_tafel_fit(
        eta_mv, j_bv, "mV", (240.0, 380.0)
    )
    assert fit_bv[2] == pytest.approx(b_exact_mv, abs=5e-3)


def test_domain9_nmr_binomial_multiplets_and_xps_shirley_background() -> None:
    """Verifies Pascal 1:2:1 / 1:3:3:1 NMR ratios & Shirley XPS fixed-point BG.

    Governing equations:
      I_m / I_tot = C(n, m) / 2^n, delta_m = delta_0 + (m - n/2) * (J / nu_0),
      B(E) = I_low + (I_high - I_low) * int_{E_low}^E max(I - B, 0) dE' / A_tot.
    """
    mhz = 500.0
    j_hz = 7.5
    step_ppm = j_hz / mhz

    pos_t, wt_t = spec_plots.expand_nmr_multiplet(
        1.18, integral=3.0, multiplicity="t", j_hz=j_hz, spectrometer_mhz=mhz
    )
    np.testing.assert_allclose(
        pos_t, [1.18 - step_ppm, 1.18, 1.18 + step_ppm], rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(wt_t / wt_t[0], [1.0, 2.0, 1.0], atol=1e-12)
    assert float(np.sum(wt_t)) == pytest.approx(3.0, abs=1e-12)

    pos_q, wt_q = spec_plots.expand_nmr_multiplet(
        3.65, integral=2.0, multiplicity="q", j_hz=j_hz, spectrometer_mhz=mhz
    )
    np.testing.assert_allclose(
        pos_q,
        [
            3.65 - 1.5 * step_ppm,
            3.65 - 0.5 * step_ppm,
            3.65 + 0.5 * step_ppm,
            3.65 + 1.5 * step_ppm,
        ],
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        wt_q / wt_q[0], [1.0, 3.0, 3.0, 1.0], atol=1e-12
    )
    assert float(np.sum(wt_q)) == pytest.approx(2.0, abs=1e-12)

    # Iterative Shirley XPS background invariants: endpoints, monotonicity,
    # reversal invariance, and self-consistent integral equation satisfaction.
    be_asc = np.linspace(282.0, 293.0, 400)
    step_bg = 150.0 + 320.0 * 0.5 * (1.0 + np.tanh((be_asc - 285.2) / 1.2))
    y_xps = (
        step_bg
        + 920.0 * np.exp(-0.5 * ((be_asc - 284.8) / 0.52) ** 2)
        + 360.0 * np.exp(-0.5 * ((be_asc - 286.5) / 0.58) ** 2)
        + 240.0 * np.exp(-0.5 * ((be_asc - 288.7) / 0.62) ** 2)
    )
    bg = spec_plots.compute_shirley_background(
        be_asc, y_xps, max_iter=120, tol=1e-10
    )
    assert bg[0] == pytest.approx(float(y_xps[0]), abs=1e-12)
    assert bg[-1] == pytest.approx(float(y_xps[-1]), abs=1e-12)
    assert bool(np.all(np.diff(bg) >= -1e-12))

    bg_rev = spec_plots.compute_shirley_background(
        be_asc[::-1], y_xps[::-1], max_iter=120, tol=1e-10
    )
    np.testing.assert_allclose(bg_rev, bg[::-1], rtol=0.0, atol=1e-10)

    # Verify Shirley integral fixed-point equation directly on converged bg.
    net = np.maximum(y_xps - bg, 0.0)
    dx = np.diff(be_asc)
    cum_area = np.concatenate(
        [[0.0], np.cumsum(0.5 * (net[:-1] + net[1:]) * dx)]
    )
    rhs_bg = y_xps[0] + (y_xps[-1] - y_xps[0]) * (cum_area / cum_area[-1])
    np.testing.assert_allclose(bg, rhs_bg, rtol=0.0, atol=1e-5)
