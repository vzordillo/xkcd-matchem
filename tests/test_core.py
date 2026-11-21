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
from xkcd_matchem.core import geometry
from xkcd_matchem.core import spectroscopy
from xkcd_matchem.core import structure
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


def test_periodic_bonding_and_polyhedra() -> None:
    si = loader.load("si_diamond.cif")
    bonds = bonding.infer_bonds(si, scale=1.20)
    # 8 Si atoms * 4 tetrahedrally coordinated bonds / 2 = 16 unique bonds.
    assert len(bonds) == 16

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
