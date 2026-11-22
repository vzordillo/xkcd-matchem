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
"""Bundled example structure, calculation, and trajectory data registry."""

from __future__ import annotations

import pathlib


_SAMPLES_DIR = pathlib.Path(__file__).resolve().parent / "samples"

_SAMPLE_DESCRIPTIONS: dict[str, str] = {
    "aspirin.xyz": "Aspirin (acetylsalicylic acid, 21 atoms, XYZ)",
    "benzene.sdf": "Benzene with explicit bond orders (V2000 SDF)",
    "mos2_2h.poscar": "2H-MoS2 layered hexagonal unit cell (VASP POSCAR)",
    "nacl.cif": "Rock-salt NaCl crystal with Fm-3m symmetry ops (CIF)",
    "si_diamond.cif": "Diamond cubic Si crystal with Fd-3m symmetry (CIF)",
    "alaninedipeptide.pdb": "Alanine dipeptide with CONECT records (PDB)",
    "water_homo.cube": "Synthetic water HOMO-like 3D orbital grid (CUBE)",
    "h2o_gaussian.log": "Water SCF, orbitals & IR frequencies (Gaussian)",
    "benzene_orca.out": "Benzene SCF, MOs, IR & UV-Vis spectra (ORCA)",
    "si_qe_bands.dat": "Silicon electronic band structure path (QE bands.dat)",
    "si_qe_dos.dat": "Silicon density of states curve (QE dos.dat)",
    "si_phonon_bands.yaml": "Silicon phonon dispersion branches (Phonopy YAML)",
    "si_cohp.lobster": "Silicon covalent Si-Si bonding pCOHP & ICOHP (LOBSTER)",
    "sn2_neb.xyz": "Cl- + CH3Cl SN2 9-image reaction path (extended XYZ)",
    "cu_md.lammpstrj": "Cu cluster 10-frame MD trajectory (LAMMPS lammpstrj)",
    "water_rdf.xvg": "Liquid water O-O radial distribution (GROMACS XVG)",
    "tio2_slab.poscar": "Rutile TiO2 (110) surface slab (VASP POSCAR)",
}


def list_samples() -> tuple[str, ...]:
    """Returns the sorted tuple of bundled sample file names."""
    return tuple(sorted(_SAMPLE_DESCRIPTIONS))


def describe_sample(name: str) -> str:
    """Returns the human-readable description for a bundled sample file.

    Args:
      name: Filename of the bundled sample.

    Returns:
      Description string for ``name``.

    Raises:
      FileNotFoundError: If ``name`` is not a registered sample file.
    """
    if name not in _SAMPLE_DESCRIPTIONS:
        available = ", ".join(list_samples())
        raise FileNotFoundError(
            f"Unknown bundled sample {name!r}. Available: {available}"
        )
    return _SAMPLE_DESCRIPTIONS[name]


def get_sample_path(name: str) -> pathlib.Path:
    """Resolves the filesystem path to a bundled example data file.

    Args:
      name: Filename of the bundled sample (e.g., ``"aspirin.xyz"``).

    Returns:
      Absolute path pointing to the bundled sample file.

    Raises:
      FileNotFoundError: If ``name`` does not exist inside ``data/samples``.
    """
    candidate = _SAMPLES_DIR / name
    if not candidate.is_file():
        available = ", ".join(list_samples())
        raise FileNotFoundError(
            f"Sample file {name!r} not found in {_SAMPLES_DIR}. "
            f"Available: {available}"
        )
    return candidate
