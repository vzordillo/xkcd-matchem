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
"""Tests for core structures, periodic bonding, polyhedra, XRD, and geometry."""

from __future__ import annotations

import math
import re

import numpy as np
import pytest

from xkcd_matchem.core import bonding
from xkcd_matchem.core import electronic
from xkcd_matchem.core import exceptions
from xkcd_matchem.core import geometry
from xkcd_matchem.core import isosurface
from xkcd_matchem.core import spectroscopy
from xkcd_matchem.core import structure
from xkcd_matchem.core import trajectory
from xkcd_matchem.core import wulff
from xkcd_matchem.data import elements
from xkcd_matchem.io import loader


def test_atoms_data_operations() -> None:
    si = loader.load("si_diamond.cif")
    assert isinstance(si, structure.AtomsData)
    assert len(si) == 8
    assert si.is_periodic
    assert si.formula() == "Si8"
    assert si.formula(reduce_formula=True) == "Si"
    assert si.volume() == pytest.approx(5.4309**3, rel=1e-4)
    supercell = si.repeat((2, 1, 1))
    assert len(supercell) == 16
    assert supercell.volume() == pytest.approx(2.0 * si.volume(), rel=1e-6)


def test_atoms_validation_centering_and_repeat_properties() -> None:
    cell = np.diag([5.0, 5.0, 12.0])
    atoms = structure.AtomsData(
        symbols=["Ti", "O"],
        positions=np.array([[0.1, 0.1, 0.2], [2.5, 2.5, 2.2]]),
        cell=cell,
        pbc=(True, True, True),
        bonds=[structure.Bond(i=0, j=1, order=1.0, shift=(1, 0, 0))],
        charges=np.array([1.2, -0.6]),
        velocities=np.array([[0.1, 0.0, 0.0], [0.0, -0.1, 0.0]]),
        forces=np.array([[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]]),
        energies={"total": -12.5},
        atom_labels=["Ti1", "O1"],
    )
    centered = atoms.center_in_cell()
    frac = centered.fractional_positions()
    assert np.all((frac >= -1e-9) & (frac < 1.0 + 1e-9))

    rep = atoms.repeat((2, 1, 1))
    assert len(rep) == 4
    assert len(rep.bonds) == 2
    assert rep.energies == {}
    assert np.allclose(rep.charges, [1.2, -0.6, 1.2, -0.6])
    assert np.allclose(rep.velocities, np.tile(atoms.velocities, (2, 1)))
    assert np.allclose(rep.forces, np.tile(atoms.forces, (2, 1)))
    assert rep.atom_labels == ["Ti1", "O1", "Ti1", "O1"]

    with pytest.raises(exceptions.StructureError):
        structure.AtomsData(
            symbols=["H"],
            positions=np.array([[0.0, 0.0, 0.0]]),
            cell=np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
            pbc=(True, True, True),
        )
    with pytest.raises(exceptions.StructureError):
        structure.AtomsData(
            symbols=["H"],
            positions=np.array([[0.0, 0.0, 0.0]]),
            charges=np.array([np.nan]),
        )


def test_skewed_minimum_image_and_camera_presets() -> None:
    # Exact audit counterexample for skewed nonorthogonal cells.
    skew_cell = np.array(
        [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 0.0, 5.0]], dtype=np.float64
    )
    vec = np.array([0.49, 0.49, 0.0]) @ skew_cell
    mic = geometry.minimum_image_vector(vec, skew_cell, (True, True, True))
    # Closest lattice image is vec - b = (0.031, -0.051, 0.0) (norm ~ 0.0597 A).
    assert np.allclose(mic, [0.031, -0.051, 0.0], atol=1e-6)
    assert np.linalg.norm(mic) < 0.06
    assert geometry.max_min_image_radius(np.diag([10.0, 8.0, 6.0])) == (
        pytest.approx(3.0)
    )

    for name in (
        "isometric",
        "perspective",
        "front",
        "side",
        "top",
        "a",
        "b",
        "c",
        "a*",
        "b*",
        "c*",
    ):
        cam = geometry.Camera.from_preset(name, cell=skew_cell)
        assert isinstance(cam, geometry.Camera)
    with pytest.raises(exceptions.StructureError):
        geometry.Camera.from_preset("unknown_cam")


def test_periodic_bonding_and_polyhedra() -> None:
    si = loader.load("si_diamond.cif")
    bonds = bonding.infer_bonds(si, scale=1.20)
    # 8 Si atoms * 4 tetrahedrally coordinated bonds / 2 = 16 unique bonds.
    assert len(bonds) == 16
    assert all(b.order == 1.0 for b in bonds)

    # Organic bond-order perception for XYZ, ORCA, and PDB structures.
    aspirin = loader.load("aspirin.xyz")
    asp_bonds = bonding.infer_bonds(aspirin)
    double_bonds = [b for b in asp_bonds if b.order == 2.0]
    co_doubles = [
        b
        for b in double_bonds
        if {aspirin.symbols[b.i], aspirin.symbols[b.j]} == {"C", "O"}
    ]
    cc_doubles = [
        b
        for b in double_bonds
        if {aspirin.symbols[b.i], aspirin.symbols[b.j]} == {"C"}
    ]
    assert len(co_doubles) == 2
    assert len(cc_doubles) == 3

    benz_orca = loader.load("benzene_orca.out")
    benz_doubles = [b for b in bonding.infer_bonds(benz_orca) if b.order == 2.0]
    assert len(benz_doubles) == 3

    pep = loader.load("alaninedipeptide.pdb")
    pep_co_doubles = [
        b
        for b in pep.bonds
        if b.order == 2.0 and {pep.symbols[b.i], pep.symbols[b.j]} == {"C", "O"}
    ]
    assert len(pep_co_doubles) == 2

    tio2 = loader.load("tio2_slab.poscar")
    polys = bonding.find_coordination_polyhedra(
        tio2, "Ti", "O", min_coordination=4
    )
    assert len(polys) >= 2
    assert all(len(p.faces) >= 4 for p in polys)


def test_powder_xrd_peaks_for_si_and_nacl() -> None:
    si = loader.load("si_diamond.cif")
    xrd_si = spectroscopy.compute_xrd(si, two_theta_range=(20.0, 60.0))
    assert xrd_si.kind == "xrd"
    top_peak = max(xrd_si.peaks, key=lambda p: p.intensity)
    # Diamond Si (111) Cu K-alpha1 reflection sits at ~28.44 degrees 2theta.
    assert 28.0 <= top_peak.position <= 28.9
    si_labels = [p.label for p in xrd_si.peaks]
    assert "(111)" in si_labels
    assert "(220)" in si_labels
    assert "(311)" in si_labels

    nacl = loader.load("nacl.cif")
    nacl_closed = nacl.with_boundary_atoms()
    assert len(nacl_closed) == 27
    assert nacl_closed.symbols.count("Na") == 14
    assert nacl_closed.symbols.count("Cl") == 13
    assert len(bonding.infer_bonds(nacl_closed, use_pbc=False)) == 54

    si = loader.load("si_diamond.cif")
    si_closed = si.with_boundary_atoms()
    assert len(si_closed) == 18
    assert len(bonding.infer_bonds(si_closed, use_pbc=False)) == 16

    xrd_nacl = spectroscopy.compute_xrd(nacl, two_theta_range=(20.0, 60.0))
    top_nacl = max(xrd_nacl.peaks, key=lambda p: p.intensity)
    # NaCl (200) strongest reflection sits at ~31.7 degrees 2theta.
    assert 31.2 <= top_nacl.position <= 32.2
    assert top_nacl.label == "(200)"
    assert "(220)" in [p.label for p in xrd_nacl.peaks]


def test_geometry_camera_and_miller_plane() -> None:
    cam = geometry.Camera(azimuth=40.0, elevation=25.0)
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    xy, depth = geometry.project_points(pts, cam)
    assert xy.shape == (3, 2)
    assert depth.shape == (3,)
    cell = np.diag([5.0, 5.0, 5.0])
    edges = geometry.unit_cell_edges(cell, cam)
    assert len(edges) == 12
    assert sum(1 for _, _, is_back in edges if is_back) == 3
    poly = geometry.miller_plane_polygon(cell, (1, 1, 1))
    assert poly.shape == (3, 3)
    # Exact intercepts on a=5, b=5, c=5 => x + y + z = 5.
    assert np.allclose(np.sum(poly, axis=1), 5.0)
    # Negative-index planes must shift origin cleanly inside [0, 1]^3.
    poly_neg = geometry.miller_plane_polygon(cell, (1, -1, 0))
    assert poly_neg.shape == (4, 3)
    assert np.allclose(poly_neg[:, 0] - poly_neg[:, 1], 0.0)
    poly_neg_tri = geometry.miller_plane_polygon(cell, (1, -1, 1))
    assert poly_neg_tri.shape == (3, 3)
    assert np.allclose(
        poly_neg_tri[:, 0] - poly_neg_tri[:, 1] + poly_neg_tri[:, 2], 0.0
    )


def test_trajectory_rdf_unwrapped_rmsd_and_electronic_fixes() -> None:
    box = np.diag([10.0, 10.0, 10.0])
    f0 = structure.AtomsData(
        symbols=["Cu", "Cu"],
        positions=np.array([[0.1, 5.0, 5.0], [0.5, 5.0, 5.0]]),
        cell=box,
        pbc=(True, True, True),
    )
    f1 = structure.AtomsData(
        symbols=["Cu", "Cu"],
        positions=np.array([[9.9, 5.0, 5.0], [0.5, 5.0, 5.0]]),
        cell=box,
        pbc=(True, True, True),
    )
    traj = trajectory.Trajectory(frames=[f0, f1])
    rmsd_unwrapped = traj.rmsd(ref_index=0, indices=[0, 1], unwrap=True)
    rmsd_raw = traj.rmsd(ref_index=0, indices=[0, 1], unwrap=False)
    assert rmsd_unwrapped[1] == pytest.approx(0.1, abs=1e-6)
    assert rmsd_raw[1] == pytest.approx(4.5, abs=1e-6)

    with pytest.raises(exceptions.StructureError):
        traj.compute_rdf(("Cu", "Cu"), r_max=6.5, n_bins=20, strict=True)
    r_vals, g_vals = traj.compute_rdf(
        ("Cu", "Cu"), r_max=6.5, n_bins=13, strict=False
    )
    assert np.any(g_vals > 0.0)
    assert r_vals.shape == (13,)

    # Metallic Fermi-crossing bands yield zero gap even when points skip zero.
    metal = electronic.BandStructure(
        k_distances=np.array([0.0, 1.0]),
        eigenvalues=np.array([[-1.0, 1.0], [0.8, 1.5]]),
        fermi_energy=0.0,
    )
    assert metal.band_gap() == 0.0
    occ_semi = electronic.BandStructure(
        k_distances=np.array([0.0, 1.0]),
        eigenvalues=np.array([[1.0, 2.5], [1.2, 2.3]]),
        occupations=np.array([[2.0, 0.0], [2.0, 0.0]]),
    )
    assert occ_semi.band_gap() == pytest.approx(1.1)

    # Skewed VolumetricGrid slice preserves affine non-orthogonal shear.
    voxels = np.array([[0.5, 0.0, 0.0], [0.25, 0.5, 0.0], [0.0, 0.0, 0.5]])
    grid = electronic.VolumetricGrid(
        origin=np.zeros(3),
        voxel_vectors=voxels,
        data=np.ones((4, 4, 4)),
    )
    uu, vv, plane, _ = grid.slice_mesh(axis="z")
    assert plane.shape == (4, 4)
    assert uu[0, 1] > 0.2
    assert vv[0, 1] > 0.4


def _parse_hkl_label(label: str) -> tuple[int, int, int]:
    """Parses '(111)' or '(10 1 0)' reflection labels into (h, k, l)."""
    body = label.strip().strip("()")
    if " " in body:
        parts = [int(p) for p in body.split()]
        return (parts[0], parts[1], parts[2])
    tokens = re.findall(r"-?\d", body)
    return (int(tokens[0]), int(tokens[1]), int(tokens[2]))


def test_braggs_law_peak_positions_and_extinction_rules() -> None:
    """Verifies n*lambda = 2*d_hkl*sin(theta) and space-group absences."""
    lam = elements.CODATA2022.cu_kalpha1_angstrom
    for sample_name in ("si_diamond.cif", "nacl.cif", "mos2_2h.poscar"):
        crys = loader.load(sample_name)
        assert crys.cell is not None
        recip = 2.0 * math.pi * np.linalg.inv(crys.cell).T
        xrd = spectroscopy.compute_xrd(
            crys, wavelength=lam, two_theta_range=(10.0, 85.0)
        )
        assert len(xrd.peaks) >= 3
        for pk in xrd.peaks:
            hkl = _parse_hkl_label(pk.label)
            g_vec = np.asarray(hkl, dtype=np.float64) @ recip
            d_hkl = (2.0 * math.pi) / float(np.linalg.norm(g_vec))
            # Bragg's law: lambda == 2 * d_hkl * sin(theta).
            theta_rad = math.radians(0.5 * pk.position)
            lhs = 2.0 * d_hkl * math.sin(theta_rad)
            assert lhs == pytest.approx(lam, rel=2e-4)

    # Systematic absence rules for BCC (h+k+l even) and FCC/Diamond.
    bcc_fe = structure.AtomsData(
        symbols=["Fe", "Fe"],
        positions=[[0.0, 0.0, 0.0], [1.435, 1.435, 1.435]],
        cell=2.87 * np.eye(3),
        pbc=(True, True, True),
    )
    xrd_bcc = spectroscopy.compute_xrd(bcc_fe, two_theta_range=(15.0, 90.0))
    for pk in xrd_bcc.peaks:
        h, k, l_idx = _parse_hkl_label(pk.label)
        assert (h + k + l_idx) % 2 == 0

    xrd_nacl = spectroscopy.compute_xrd(
        loader.load("nacl.cif"), two_theta_range=(15.0, 85.0)
    )
    for pk in xrd_nacl.peaks:
        h, k, l_idx = _parse_hkl_label(pk.label)
        parities = {h % 2, k % 2, l_idx % 2}
        assert len(parities) == 1

    # Debye-Waller factor attenuates high-angle reflections relative to low.
    si = loader.load("si_diamond.cif")
    xrd_0 = spectroscopy.compute_xrd(si, debye_waller_b=0.0)
    xrd_b = spectroscopy.compute_xrd(si, debye_waller_b=1.5)
    assert xrd_b.peaks[-1].intensity < xrd_0.peaks[-1].intensity

    # Broadening lineshapes and validation error branches.
    grid = np.linspace(0.0, 20.0, 401)
    for prof in ("lorentzian", "gaussian", "pseudo_voigt", "voigt"):
        y_area = spectroscopy.broaden_peaks(
            [10.0], [2.5], grid, fwhm=1.2, profile=prof, normalize="area"
        )
        assert float(np.trapezoid(y_area, grid)) == pytest.approx(2.5, rel=0.05)
        y_h = spectroscopy.broaden_peaks(
            [10.0], [2.5], grid, fwhm=1.2, profile=prof, normalize="height"
        )
        assert float(np.max(y_h)) == pytest.approx(2.5, rel=1e-3)
    assert np.all(spectroscopy.broaden_peaks([], [], grid) == 0.0)
    with pytest.raises(exceptions.StructureError):
        spectroscopy.Spectrum(x=np.array([1.0, 2.0]), y=np.array([1.0]))
    with pytest.raises(exceptions.StructureError):
        spectroscopy.broaden_peaks([1.0], [1.0], grid, fwhm=0.0)
    with pytest.raises(exceptions.StructureError):
        spectroscopy.broaden_peaks([1.0], [1.0], grid, normalize="bad")
    with pytest.raises(exceptions.StructureError):
        spectroscopy.broaden_peaks([1.0], [1.0], grid, profile="bad")
    with pytest.raises(exceptions.StructureError):
        spectroscopy.compute_xrd(loader.load("aspirin.xyz"))


def test_wulff_polyhedron_euler_characteristic_and_symmetry() -> None:
    """Verifies V - E + F == 2, Minkowski closure, and symmetry of Wulff."""
    cubic_cell = 4.0 * np.eye(3)
    hex_cell = geometry.cell_vectors_from_parameters(
        3.2, 3.2, 5.2, 90.0, 90.0, 120.0
    )
    cases = (
        (cubic_cell, {(1, 0, 0): 1.0}, "cubic", 8, 12, 6),
        (cubic_cell, {(1, 1, 1): 1.0}, "cubic", 6, 12, 8),
        (cubic_cell, {(1, 1, 0): 1.0}, "cubic", 14, 24, 12),
        (cubic_cell, {(1, 0, 0): 1.15, (1, 1, 1): 1.0}, "cubic", 24, 36, 14),
        (hex_cell, {(0, 0, 1): 1.0, (1, 0, 0): 1.1}, "hexagonal", 12, 18, 8),
    )
    for cell, energies, sym, exp_v, exp_e, exp_f in cases:
        ws = wulff.compute_wulff_shape(cell, energies, symops=sym)
        v_set = {
            tuple(np.round(pt, 5).tolist())
            for f in ws.facets
            for pt in f.vertices
        }
        edges: dict[tuple[tuple[float, ...], tuple[float, ...]], int] = {}
        normal_sum = np.zeros(3, dtype=np.float64)
        for facet in ws.facets:
            pts = [tuple(np.round(pt, 5).tolist()) for pt in facet.vertices]
            for idx in range(len(pts)):
                p1, p2 = pts[idx], pts[(idx + 1) % len(pts)]
                key = (p1, p2) if p1 <= p2 else (p2, p1)
                edges[key] = edges.get(key, 0) + 1
            normal_sum += facet.area * facet.normal

        assert len(v_set) == exp_v
        assert len(edges) == exp_e
        assert len(ws.facets) == exp_f
        assert all(c == 2 for c in edges.values())
        assert len(v_set) - len(edges) + len(ws.facets) == 2
        assert float(np.linalg.norm(normal_sum)) < 1e-8
        assert sum(f.area_fraction for f in ws.facets) == pytest.approx(1.0)
        for v_pt in ws.vertices:
            assert np.any(
                np.all(np.isclose(ws.vertices, -v_pt, atol=1e-5), axis=1)
            )

    # Space-group name, affine symop list, and error validation branches.
    ws_sg = wulff.compute_wulff_shape(
        cubic_cell, {(1, 0, 0): 1.0}, symops="Fm-3m"
    )
    assert len(ws_sg.facets) == 6
    rot_ops = [(r, np.zeros(3)) for r in wulff._cubic_rotations()]
    ws_ops = wulff.compute_wulff_shape(
        cubic_cell, {(1, 0, 0): 1.0}, symops=rot_ops
    )
    assert len(ws_ops.facets) == 6

    for bad_args in (
        (np.zeros((3, 3)), {(1, 0, 0): 1.0}, "cubic"),
        (cubic_cell, {}, "cubic"),
        (cubic_cell, {(1, 0, 0): 1.0}, "UnknownSG"),
        (cubic_cell, {(1, 0, 0): 1.0}, 999),
        (cubic_cell, {(0, 0, 0): 1.0}, "cubic"),
        (cubic_cell, {(1, 0, 0): 1.0}, None),
    ):
        with pytest.raises(exceptions.StructureError):
            wulff.compute_wulff_shape(
                bad_args[0],
                bad_args[1],
                symops=bad_args[2],  # type: ignore[arg-type]
            )


def test_uniform_density_rdf_limit_and_trajectory_invariants() -> None:
    """Verifies uniform-density RDF limit g(r)==1 and energy unit roundtrips."""
    # Construct an analytical multi-frame ensemble where each radial shell bin
    # [r_k, r_{k+1}] receives pair count proportional to (r_{k+1}^3 - r_k^3).
    box_l = 20.0
    box = np.diag([box_l, box_l, box_l])
    r_max = 6.0
    n_bins = 6
    edges = np.linspace(0.0, r_max, n_bins + 1)
    r_centers = 0.5 * (edges[:-1] + edges[1:])
    # For bins k=0..5 with dr=1.0, (k+1)^3 - k^3 = 1, 7, 19, 37, 61, 91.
    shell_counts = [
        int(round((k + 1) ** 3 - k**3)) for k in range(n_bins)
    ]
    b_positions: list[list[float]] = []
    for r_c, count in zip(r_centers, shell_counts, strict=True):
        for m in range(count):
            phi = 2.0 * math.pi * m / max(count, 1)
            b_positions.append(
                [r_c * math.cos(phi), r_c * math.sin(phi), 0.0]
            )
    frame = structure.AtomsData(
        symbols=["Au"] + ["Cu"] * len(b_positions),
        positions=np.vstack([[[0.0, 0.0, 0.0]], np.array(b_positions)]),
        cell=box,
        pbc=(True, True, True),
        energies={"total": -10.0},
    )
    traj = trajectory.Trajectory(frames=[frame])
    r_out, g_out = traj.compute_rdf(
        ("Au", "Cu"), r_max=r_max, n_bins=n_bins, strict=True
    )
    expected_g = len(b_positions) / (
        (len(b_positions) / (box_l**3)) * (4.0 * math.pi / 3.0) * (r_max**3)
    )
    # Every shell bin has the exact same normalized density => flat g(r)!
    assert np.allclose(r_out, r_centers)
    assert np.allclose(g_out, expected_g, rtol=1e-12)
    assert float(np.std(g_out)) == pytest.approx(0.0, abs=1e-12)

    # Energy unit round-trip and trajectory error branches.
    f2 = frame.copy()
    f2.energies = {"total": -8.5}
    t2 = trajectory.Trajectory(frames=[frame, f2])
    e_ev = t2.energy_profile(relative=True, unit="eV")
    e_kj = t2.energy_profile(relative=True, unit="kJ/mol")
    e_kcal = t2.energy_profile(relative=True, unit="kcal/mol")
    assert np.allclose(
        e_kj / elements.CODATA2022.ev_to_kj_mol, e_ev, rtol=1e-12
    )
    assert np.allclose(
        e_kcal / elements.CODATA2022.ev_to_kcal_mol, e_ev, rtol=1e-12
    )

    assert len(trajectory.Trajectory(frames=[]).rmsd()) == 0
    with pytest.raises(exceptions.StructureError):
        trajectory.Trajectory(frames=[frame], times=np.array([0.0, 1.0]))
    with pytest.raises(exceptions.StructureError):
        trajectory.Trajectory(frames=[frame], times=np.array([np.nan]))
    with pytest.raises(exceptions.StructureError):
        t2.rmsd(indices=[-1])
    with pytest.raises(exceptions.StructureError):
        t2.energy_profile(unit="erg")
    with pytest.raises(exceptions.StructureError):
        trajectory.Trajectory(frames=[loader.load("aspirin.xyz")]).energy_profile()
    with pytest.raises(exceptions.StructureError):
        t2.compute_rdf(("Au", "Cu"), r_max=-1.0)
    with pytest.raises(exceptions.StructureError):
        t2.compute_rdf(("Au", "Cu"), start=5, stop=6)
    with pytest.raises(exceptions.StructureError):
        t2.compute_rdf(("Pt", "Cu"))
    with pytest.raises(exceptions.StructureError):
        trajectory.Trajectory(frames=[loader.load("aspirin.xyz")]).compute_rdf(
            ("C", "O")
        )


def _assert_watertight_closed_mesh(
    mesh: isosurface.IsosurfaceMesh,
    expected_euler: int = 2,
) -> None:
    """Asserts every directed triangle edge has its exact opposite edge."""
    assert len(mesh.vertices) > 0
    assert len(mesh.faces) > 0
    directed_edges: set[tuple[int, int]] = set()
    undirected_counts: dict[tuple[int, int], int] = {}
    for i0, i1, i2 in mesh.faces:
        tri_edges = (
            (int(i0), int(i1)),
            (int(i1), int(i2)),
            (int(i2), int(i0)),
        )
        for u_idx, v_idx in tri_edges:
            assert (u_idx, v_idx) not in directed_edges
            directed_edges.add((u_idx, v_idx))
            key = (u_idx, v_idx) if u_idx <= v_idx else (v_idx, u_idx)
            undirected_counts[key] = undirected_counts.get(key, 0) + 1
    assert all(cnt == 2 for cnt in undirected_counts.values())
    for u_idx, v_idx in directed_edges:
        assert (v_idx, u_idx) in directed_edges
    used_verts = len(np.unique(mesh.faces))
    n_edges = len(undirected_counts)
    n_faces = len(mesh.faces)
    assert used_verts - n_edges + n_faces == expected_euler


def test_marching_tetrahedra_closed_mesh_on_3d_gaussian_and_2pz() -> None:
    """Verifies closed 2-manifold topology on 3D Gaussian and 2pz orbital."""
    # 1. Analytical 3D Gaussian rho(r) = exp(-r^2 / R0^2) at level exp(-1).
    nx = 18
    r0 = 1.25
    coords = np.linspace(-2.2, 2.2, nx)
    gx, gy, gz = np.meshgrid(coords, coords, coords, indexing="ij")
    r_sq = gx**2 + gy**2 + gz**2
    grid_gauss = electronic.VolumetricGrid(
        origin=np.array([-2.2, -2.2, -2.2]),
        voxel_vectors=np.eye(3) * (4.4 / (nx - 1)),
        data=np.exp(-r_sq / (r0**2)),
    )
    mesh_g = isosurface.extract_isosurface(grid_gauss, level=math.exp(-1.0))
    _assert_watertight_closed_mesh(mesh_g, expected_euler=2)
    radii = np.linalg.norm(mesh_g.vertices, axis=1)
    assert np.allclose(radii, r0, atol=0.08)
    # Outward normals align with radial position vectors.
    tri_centers = np.mean(mesh_g.vertices[mesh_g.faces], axis=1)
    dots = np.einsum("ij,ij->i", mesh_g.normals, tri_centers)
    assert np.all(dots > 0.0)

    # 2. Analytical 2pz orbital psi(r) = z * exp(-r^2): two closed lobes.
    psi_2pz = gz * np.exp(-r_sq)
    grid_2pz = electronic.VolumetricGrid(
        origin=np.array([-2.2, -2.2, -2.2]),
        voxel_vectors=np.eye(3) * (4.4 / (nx - 1)),
        data=psi_2pz,
    )
    lobe_pos = isosurface.extract_isosurface(grid_2pz, level=0.14)
    lobe_neg = isosurface.extract_isosurface(grid_2pz, level=-0.14)
    _assert_watertight_closed_mesh(lobe_pos, expected_euler=2)
    _assert_watertight_closed_mesh(lobe_neg, expected_euler=2)
    assert len(lobe_pos.faces) == len(lobe_neg.faces)
    assert float(np.mean(lobe_pos.vertices[:, 2])) == pytest.approx(
        -float(np.mean(lobe_neg.vertices[:, 2])), rel=1e-5
    )

    # Non-intersecting level returns empty mesh; non-finite level raises.
    empty_m = isosurface.extract_isosurface(grid_gauss, level=5.0)
    assert len(empty_m.faces) == 0
    with pytest.raises(exceptions.StructureError):
        isosurface.extract_isosurface(grid_gauss, level=float("nan"))


def test_electronic_containers_validation_and_phonon_unit_invariants() -> None:
    """Covers BandStructure, DOS, MO, VolumetricGrid, Phonon, & CohpData."""
    k_d = np.array([0.0, 0.5, 1.0])
    eigs_2d = np.array([[-1.2, 0.8], [-0.9, 0.6], [-0.5, 1.1]])
    bs = electronic.BandStructure(
        k_distances=k_d,
        eigenvalues=eigs_2d,
        fermi_energy=0.0,
        projections={"s": np.ones((3, 2))},
    )
    assert not bs.spin_polarized
    assert bs.band_gap() == pytest.approx(1.1)

    bs_spin = electronic.BandStructure(
        k_distances=k_d,
        eigenvalues=np.stack([eigs_2d, eigs_2d], axis=0),
    )
    assert bs_spin.spin_polarized

    # Occupations-based fractional metal and all-occupied zero gap.
    bs_frac = electronic.BandStructure(
        k_distances=np.array([0.0, 1.0]),
        eigenvalues=np.array([[-1.0, 1.0], [-0.5, 1.2]]),
        occupations=np.array([[1.0, 0.5], [1.0, 0.0]]),
    )
    assert bs_frac.band_gap() == 0.0
    bs_all_occ = electronic.BandStructure(
        k_distances=np.array([0.0, 1.0]),
        eigenvalues=np.array([[-2.0, -1.0], [-1.5, -0.5]]),
        occupations=np.ones((2, 2)),
    )
    assert bs_all_occ.band_gap() == 0.0
    bs_unocc_only = electronic.BandStructure(
        k_distances=np.array([0.0, 1.0]),
        eigenvalues=np.array([[1.0, 2.0], [1.5, 2.5]]),
        fermi_energy=0.0,
    )
    assert bs_unocc_only.band_gap() == 0.0

    with pytest.raises(exceptions.StructureError):
        electronic.BandStructure(k_distances=k_d, eigenvalues=np.ones(3))
    with pytest.raises(exceptions.StructureError):
        electronic.BandStructure(k_distances=k_d, eigenvalues=np.ones((2, 2)))
    with pytest.raises(exceptions.StructureError):
        electronic.BandStructure(
            k_distances=k_d,
            eigenvalues=eigs_2d,
            occupations=np.ones((2, 2)),
        )
    with pytest.raises(exceptions.StructureError):
        electronic.BandStructure(
            k_distances=k_d,
            eigenvalues=eigs_2d,
            projections={"s": np.ones((2, 2))},
        )
    with pytest.raises(exceptions.StructureError):
        electronic.DensityOfStates(
            energies=np.array([0.0, 1.0]), total_dos=np.array([1.0])
        )

    # MolecularOrbitals edge cases.
    with pytest.raises(exceptions.StructureError):
        electronic.MolecularOrbitals(
            energies=np.array([0.0, 1.0]), occupations=np.array([2.0])
        )
    mo_empty_occ = electronic.MolecularOrbitals(
        energies=np.array([1.0, 2.0]), occupations=np.array([0.0, 0.0])
    )
    assert mo_empty_occ.homo_index is None
    assert mo_empty_occ.homo_lumo_gap() is None
    mo_full_occ = electronic.MolecularOrbitals(
        energies=np.array([-2.0, -1.0]), occupations=np.array([2.0, 2.0])
    )
    assert mo_full_occ.lumo_index is None
    assert mo_full_occ.homo_lumo_gap() is None

    # VolumetricGrid shape and slice error branches.
    with pytest.raises(exceptions.StructureError):
        electronic.VolumetricGrid(
            origin=np.zeros(2),
            voxel_vectors=np.eye(3),
            data=np.zeros((2, 2, 2)),
        )
    with pytest.raises(exceptions.StructureError):
        electronic.VolumetricGrid(
            origin=np.zeros(3),
            voxel_vectors=np.zeros((2, 3)),
            data=np.zeros((2, 2, 2)),
        )
    with pytest.raises(exceptions.StructureError):
        electronic.VolumetricGrid(
            origin=np.zeros(3), voxel_vectors=np.eye(3), data=np.zeros((2, 2))
        )
    vg = electronic.VolumetricGrid(
        origin=np.zeros(3), voxel_vectors=np.eye(3), data=np.zeros((3, 3, 3))
    )
    with pytest.raises(exceptions.StructureError):
        vg.slice_plane(axis="w")
    with pytest.raises(exceptions.StructureError):
        vg.slice_plane(axis="z", index=9)
    with pytest.raises(exceptions.StructureError):
        vg.slice_mesh(axis="w")
    with pytest.raises(exceptions.StructureError):
        vg.slice_mesh(axis="z", index=-1)

    # PhononBandStructure unit conversions and validation.
    ph = electronic.PhononBandStructure(
        q_distances=np.array([0.0, 1.0]),
        frequencies=np.array([[-0.1, 5.0], [0.2, 6.0]]),
    )
    assert ph.has_imaginary_modes
    f_thz = ph.frequencies_in("THz")
    f_cm = ph.frequencies_in("cm^-1")
    f_mev = ph.frequencies_in("meV")
    assert np.allclose(
        f_cm / f_mev,
        elements.CODATA2022.ev_to_cm_inv / 1000.0,
        rtol=1e-7,
    )
    assert np.allclose(f_thz, ph.frequencies)
    with pytest.raises(exceptions.StructureError):
        electronic.PhononBandStructure(
            q_distances=np.array([0.0, 1.0]), frequencies=np.ones(2)
        )
    with pytest.raises(exceptions.StructureError):
        ph.frequencies_in("GHz")

    # CohpData validation guards.
    with pytest.raises(exceptions.StructureError):
        electronic.CohpData(
            energies=np.array([]), cohp={"A": np.array([])}
        )
    with pytest.raises(exceptions.StructureError):
        electronic.CohpData(energies=np.array([0.0, 1.0]), cohp={})
    with pytest.raises(exceptions.StructureError):
        electronic.CohpData(
            energies=np.array([0.0, 1.0]), cohp={"A": np.array([1.0])}
        )

