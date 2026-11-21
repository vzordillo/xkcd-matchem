# Supported Formats & Adapters

## Built-in Zero-Dependency Formats

| Format Key | Extensions / Filenames | Parsed Type | Loader Function |
| --- | --- | --- | --- |
| `xyz` | `.xyz`, `.extxyz` | `AtomsData` / `Trajectory` | `load`, `load_trajectory`, `save` |
| `vasp` | `.poscar`, `.contcar`, `.vasp`, `POSCAR`, `CONTCAR` | `AtomsData` | `load`, `save` |
| `xdatcar` | `.xdatcar`, `XDATCAR` | `Trajectory` | `load_trajectory` |
| `doscar` | `.doscar`, `DOSCAR` | `DensityOfStates` | `load` |
| `eigenval` | `.eigenval`, `EIGENVAL` | `BandStructure` | `load` |
| `chgcar` | `.chgcar`, `.locpot`, `CHGCAR`, `LOCPOT` | `VolumetricGrid` | `load` |
| `cif` | `.cif` (affine symop expansion) | `AtomsData` | `load` |
| `pdb` | `.pdb`, `.ent` (multi-MODEL + CONECT) | `AtomsData` / `Trajectory` | `load`, `load_trajectory` |
| `sdf` | `.sdf`, `.mol` (V2000 bond orders) | `AtomsData` | `load` |
| `mol2` | `.mol2` (Tripos ATOM/BOND) | `AtomsData` | `load` |
| `cube` | `.cube`, `.cub` (Bohr -> A) | `VolumetricGrid` | `load` |
| `gaussian_gjf` | `.gjf`, `.com` | `AtomsData` | `load` |
| `gaussian_log` | `.log` (structure, SCF, MOs, IR) | `AtomsData` + metadata | `load` |
| `gaussian_fchk` | `.fchk` | `AtomsData` | `load` |
| `orca_inp` | `.inp` | `AtomsData` | `load` |
| `orca_out` | `.orca`, ORCA `.out` | `AtomsData` + MOs/IR/UV-Vis | `load` |
| `qe_in` | `.pwi`, `.in` | `AtomsData` | `load` |
| `qe_out` | `.pwo`, QE `.out` | `AtomsData` | `load` |
| `qe_bands` | `bands.dat` / `&plot` | `BandStructure` | `load` |
| `qe_dos` | `dos.dat` / `EFermi` | `DensityOfStates` | `load` |
| `lammps_data` | `.lmp`, `.data`, `data.lammps` | `AtomsData` | `load` |
| `lammpstrj` | `.lammpstrj` | `Trajectory` | `load_trajectory` |
| `gro` | `.gro` (nm -> A) | `AtomsData` | `load` |
| `xvg` | `.xvg` | `Spectrum` | `load` |
| `cp2k_ener` | `.ener` (Hartree -> eV) | `Spectrum` | `load` |
| `xtb_out` | `xtb.out` | `AtomsData` | `load` |

## Optional Framework Adapters

Install via `pip install "xkcd-matchem[ase,pymatgen,qm,rdkit,md]"`:

- `from_ase(ase_atoms)`, `to_ase(atoms)`, `read_via_ase(path)`
- `from_pymatgen(struct_or_mol)`, `to_pymatgen(atoms)`
- `from_cclib(ccdata)`, `read_via_cclib(path)`
- `from_rdkit(mol)`, `from_smiles(smiles)`
- `from_mdanalysis(universe)`, `from_mdtraj(traj)`
