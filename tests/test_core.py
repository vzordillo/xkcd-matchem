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

import numpy as np
import pytest

from xkcd_matchem.core import bonding
from xkcd_matchem.core import electronic
from xkcd_matchem.core import exceptions
from xkcd_matchem.core import geometry
from xkcd_matchem.core import spectroscopy
from xkcd_matchem.core import structure
from xkcd_matchem.core import trajectory
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

    nacl = loader.load("nacl.cif")
    xrd_nacl = spectroscopy.compute_xrd(nacl, two_theta_range=(20.0, 60.0))
    top_nacl = max(xrd_nacl.peaks, key=lambda p: p.intensity)
    # NaCl (200) strongest reflection sits at ~31.7 degrees 2theta.
    assert 31.2 <= top_nacl.position <= 32.2


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
