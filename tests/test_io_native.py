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
"""Tests for all zero-dependency native file parsers and writers."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from xkcd_matchem.core import electronic
from xkcd_matchem.core import exceptions
from xkcd_matchem.core import spectroscopy
from xkcd_matchem.core import structure
from xkcd_matchem.core import trajectory
from xkcd_matchem.io import loader
from xkcd_matchem.io.native import cif
from xkcd_matchem.io.native import cp2k
from xkcd_matchem.io.native import gaussian
from xkcd_matchem.io.native import gromacs
from xkcd_matchem.io.native import lammps
from xkcd_matchem.io.native import orca
from xkcd_matchem.io.native import qe
from xkcd_matchem.io.native import vasp


def test_load_bundled_structure_samples(tmp_path: pathlib.Path) -> None:
    aspirin = loader.load("aspirin.xyz")
    assert isinstance(aspirin, structure.AtomsData)
    assert aspirin.formula() == "C9H8O4"

    # Round-trip XYZ writer + reader
    out_xyz = tmp_path / "roundtrip.xyz"
    loader.save(aspirin, out_xyz)
    reread = loader.load(out_xyz)
    assert np.allclose(reread.positions, aspirin.positions, atol=1e-6)

    benzene = loader.load("benzene.sdf")
    assert benzene.formula() == "C6H6"
    assert len(benzene.bonds) == 12

    mos2 = loader.load("mos2_2h.poscar")
    assert mos2.formula() == "Mo2S4"
    assert mos2.is_periodic

    nacl = loader.load("nacl.cif")
    assert len(nacl) == 8
    assert nacl.formula(reduce_formula=True) == "ClNa"

    pep = loader.load("alaninedipeptide.pdb")
    assert len(pep) == 12
    assert len(pep.bonds) >= 11


def test_load_electronic_spectroscopy_and_trajectories() -> None:
    cube = loader.load("water_homo.cube")
    assert isinstance(cube, electronic.VolumetricGrid)
    assert cube.data.shape == (12, 12, 12)
    assert cube.atoms is not None
    assert len(cube.atoms) == 3

    g_atoms = loader.load("h2o_gaussian.log")
    assert g_atoms.formula() == "H2O"
    assert isinstance(
        g_atoms.metadata["orbitals"], electronic.MolecularOrbitals
    )
    assert isinstance(g_atoms.metadata["ir_spectrum"], spectroscopy.Spectrum)

    o_atoms = loader.load("benzene_orca.out")
    assert o_atoms.formula() == "C6H6"
    assert "uvvis_spectrum" in o_atoms.metadata

    bands = loader.load("si_qe_bands.dat")
    assert isinstance(bands, electronic.BandStructure)
    assert bands.eigenvalues.shape == (1, 41, 8)

    dos = loader.load("si_qe_dos.dat")
    assert isinstance(dos, electronic.DensityOfStates)
    assert dos.fermi_energy == pytest.approx(6.245)

    neb = loader.load_trajectory("sn2_neb.xyz")
    assert isinstance(neb, trajectory.Trajectory)
    assert len(neb) == 9
    assert neb.energies is not None

    md = loader.load_trajectory("cu_md.lammpstrj")
    assert len(md) == 10
    assert md[0].formula() == "Cu8"

    xvg = loader.load("water_rdf.xvg")
    assert isinstance(xvg, spectroscopy.Spectrum)
    assert xvg.kind == "rdf"


def test_additional_native_formats(tmp_path: pathlib.Path) -> None:
    # LAMMPS data
    lmp_data = tmp_path / "sys.data"
    lmp_data.write_text(
        "Header\n\n2 atoms\n1 bonds\n1 atom types\n"
        "0.0 10.0 xlo xhi\n0.0 10.0 ylo yhi\n0.0 10.0 zlo zhi\n\n"
        "Masses\n\n1 12.011\n\nAtoms\n\n"
        "1 1 1 0.0 1.0 1.0 1.0\n2 1 1 0.0 2.5 1.0 1.0\n\n"
        "Bonds\n\n1 1 1 2\n",
        encoding="utf-8",
    )
    atoms_lmp = lammps.read_lammps_data(lmp_data)
    assert atoms_lmp.formula() == "C2"
    assert len(atoms_lmp.bonds) == 1

    # GROMACS GRO
    gro = tmp_path / "water.gro"
    gro.write_text(
        "Water\n    3\n"
        "    1SOL     OW    1   0.100   0.100   0.100\n"
        "    1SOL    HW1    2   0.195   0.100   0.100\n"
        "    1SOL    HW2    3   0.070   0.190   0.100\n"
        "   1.00000   1.00000   1.00000\n",
        encoding="utf-8",
    )
    atoms_gro = gromacs.read_gro(gro)
    assert atoms_gro.formula() == "H2O"
    assert atoms_gro.positions[0, 0] == pytest.approx(1.0)

    # CP2K .ener + QE input + ORCA .inp + Gaussian .gjf
    ener = tmp_path / "md.ener"
    ener.write_text(
        "# Step Time fs\n0 0.0 0.1 300.0 -10.0\n1 0.5 0.1 301.0 -10.01\n"
    )
    spec_cp2k = cp2k.read_cp2k_ener(ener)
    assert len(spec_cp2k.x) == 2

    pwi = tmp_path / "si.pwi"
    pwi.write_text(
        "&CONTROL\n/\nCELL_PARAMETERS angstrom\n"
        "2.7 2.7 0.0\n2.7 0.0 2.7\n0.0 2.7 2.7\n"
        "ATOMIC_POSITIONS crystal\nSi 0.0 0.0 0.0\nSi 0.25 0.25 0.25\n"
    )
    assert qe.read_qe_in(pwi).formula() == "Si2"

    inp = tmp_path / "h2.inp"
    inp.write_text("! B3LYP def2-SVP\n* xyz 0 1\nH 0 0 0\nH 0 0 0.74\n*\n")
    assert orca.read_orca_inp(inp).formula() == "H2"

    gjf = tmp_path / "h2.gjf"
    gjf.write_text("# B3LYP/6-31G\n\nH2\n\n0 1\nH 0 0 0\nH 0 0 0.74\n\n")
    assert gaussian.read_gaussian_gjf(gjf).formula() == "H2"

    # VASP POSCAR roundtrip
    mos2 = loader.load("mos2_2h.poscar")
    out_pos = tmp_path / "POSCAR"
    vasp.write_poscar(mos2, out_pos)
    assert vasp.read_poscar(out_pos).formula() == "Mo2S4"

    unknown_file = tmp_path / "dummy.unknown_fmt_xyz"
    unknown_file.write_text("unrecognized content\n", encoding="utf-8")
    with pytest.raises(exceptions.UnsupportedFormatError):
        loader.load(unknown_file)


def test_p0_vasp_lammps_qe_and_cif_fixes(tmp_path: pathlib.Path) -> None:
    # 1. VASP POSCAR Cartesian scaling & negative-volume scaling
    poscar_cart = tmp_path / "cart.poscar"
    poscar_cart.write_text(
        "Scaled cart\n2.0\n1 0 0\n0 1 0\n0 0 1\nH\n1\nCartesian\n0.5 0 0\n",
        encoding="utf-8",
    )
    atoms_cart = vasp.read_poscar(poscar_cart)
    assert np.allclose(atoms_cart.positions[0], [1.0, 0.0, 0.0])

    poscar_vol = tmp_path / "vol.poscar"
    poscar_vol.write_text(
        "Neg vol\n-125.0\n1 0 0\n0 1 0\n0 0 1\nSi\n1\nDirect\n0 0 0\n",
        encoding="utf-8",
    )
    assert vasp.read_poscar(poscar_vol).volume() == pytest.approx(125.0)

    poscar_zero = tmp_path / "zero.poscar"
    poscar_zero.write_text(
        "Bad\n0.0\n1 0 0\n0 1 0\n0 0 1\nH\n1\nDirect\n0 0 0\n",
        encoding="utf-8",
    )
    with pytest.raises(exceptions.ParseError):
        vasp.read_poscar(poscar_zero)

    # 2. VASP EIGENVAL reciprocal cell-metric weighting
    eig = tmp_path / "EIGENVAL"
    eig.write_text(
        "2 2 1 1\n0 0 0\n1.0\nCAR\nSi\n2 2 2\n\n"
        "0.0 0.0 0.0 0.5\n1 1.0\n2 2.0\n\n"
        "0.5 0.0 0.0 0.5\n1 1.2\n2 2.5\n",
        encoding="utf-8",
    )
    bs_hex = vasp.read_eigenval(eig, cell=np.diag([3.0, 3.0, 12.0]))
    assert bs_hex.k_distances[-1] == pytest.approx(np.pi / 3.0, rel=1e-5)

    # 3. LAMMPS data styles, ambiguous 6-col error, and triclinic lammpstrj
    lmp_bond = tmp_path / "bond.data"
    lmp_bond.write_text(
        "Bond style\n1 atoms\n1 atom types\n"
        "0 10 xlo xhi\n0 10 ylo yhi\n0 10 zlo zhi\n\n"
        "Masses\n\n1 28.085\n\nAtoms # bond\n\n1 1 1 2.0 3.0 4.0\n",
        encoding="utf-8",
    )
    assert lammps.read_lammps_data(lmp_bond).formula() == "Si"

    lmp_ambig = tmp_path / "ambig.data"
    lmp_ambig.write_text(
        "Ambiguous\n1 atoms\n1 atom types\n"
        "0 10 xlo xhi\n0 10 ylo yhi\n0 10 zlo zhi\n\n"
        "Masses\n\n1 12.011\n\nAtoms\n\n1 1 0.0 2.0 3.0 4.0\n",
        encoding="utf-8",
    )
    with pytest.raises(exceptions.ParseError):
        lammps.read_lammps_data(lmp_ambig)
    assert (
        lammps.read_lammps_data(
            lmp_ambig, atom_style="charge", type_map={1: "Fe"}
        ).formula()
        == "Fe"
    )

    trj_tri = tmp_path / "tri.lammpstrj"
    trj_tri.write_text(
        "ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n1\n"
        "ITEM: BOX BOUNDS xy xz yz pp pp pp\n"
        "0.0 12.0 2.0\n0.0 10.0 0.0\n0.0 10.0 0.0\n"
        "ITEM: ATOMS id type x y z\n1 1 1.0 1.0 1.0\n",
        encoding="utf-8",
    )
    with pytest.raises(exceptions.ParseError):
        lammps.read_lammpstrj(trj_tri)
    traj_tri = lammps.read_lammpstrj(trj_tri, type_map={1: "Fe"})
    assert traj_tri[0].cell[1, 0] == pytest.approx(2.0)
    assert traj_tri[0].cell[0, 0] == pytest.approx(10.0)

    # 4. QE nonzero ibrav error + A=alat support
    qe_bad = tmp_path / "bad.pwi"
    qe_bad.write_text(
        "&SYSTEM\n ibrav=2, celldm(1)=10.2 /\nATOMIC_POSITIONS alat\nSi 0 0 0\n"
    )
    with pytest.raises(exceptions.ParseError):
        qe.read_qe_in(qe_bad)

    qe_alat = tmp_path / "alat.pwi"
    qe_alat.write_text(
        "&SYSTEM\n A = 4.0 /\nCELL_PARAMETERS alat\n1 0 0\n0 1 0\n0 0 1\n"
        "ATOMIC_POSITIONS alat\nAl 0.0 0.0 0.0\n"
    )
    assert qe.read_qe_in(qe_alat).volume() == pytest.approx(64.0)

    # 5. CIF mixed occupancy preservation and _atom_site_occupancy
    cif_mix = tmp_path / "mix.cif"
    cif_mix.write_text(
        "data_mix\n"
        "_cell_length_a 4.0\n_cell_length_b 4.0\n_cell_length_c 4.0\n"
        "loop_\n_space_group_symop_operation_xyz\n'x, y, z'\n"
        "loop_\n_atom_site_label\n_atom_site_type_symbol\n"
        "_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n"
        "_atom_site_occupancy\n"
        "Fe1 Fe 0.0 0.0 0.0 0.6\n"
        "Mn1 Mn 0.0 0.0 0.0 0.4\n",
        encoding="utf-8",
    )
    mix_atoms = cif.read_cif(cif_mix)
    assert len(mix_atoms) == 2
    assert mix_atoms.formula() == "FeMn"
    assert mix_atoms.metadata["occupancies"] == pytest.approx([0.6, 0.4])
