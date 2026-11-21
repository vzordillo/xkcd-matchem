# Supported Formats & Validation Contracts

A pretty figure built from silently misread coordinates is worse than a loud error. Whenever a file uses syntax outside a built-in parser's supported subset, `xkcd-matchem` raises a descriptive `ParseError` or `StructureError` and suggests using an optional adapter (**ASE**, **pymatgen**, or **cclib**) instead of guessing.

All **26 native formats** below require only **NumPy, SciPy, and Matplotlib**.

## Molecules & Biomolecules

| Format Key | Extensions / Filenames | Parsed Type | Supported Scope & Validation |
| --- | --- | --- | --- |
| `xyz` | `.xyz`, `.extxyz` | `AtomsData` / `Trajectory` | Reads extended comment metadata (`Lattice="..."`, `pbc="..."`, `energy=`) |
| `pdb` | `.pdb`, `.ent` | `AtomsData` / `Trajectory` | Multi-`MODEL` frames and explicit `CONECT` bond topology |
| `sdf` | `.sdf`, `.mol` | `AtomsData` | V2000 atom and bond tables with formal bond orders |
| `mol2` | `.mol2` | `AtomsData` | Tripos `ATOM` and `BOND` blocks |

## Crystals & Periodic DFT

| Format Key | Extensions / Filenames | Parsed Type | Supported Scope & Validation |
| --- | --- | --- | --- |
| `vasp` | `.poscar`, `.contcar`, `.vasp`, `POSCAR`, `CONTCAR` | `AtomsData` | Applies scalar or negative-volume factors to both lattice vectors and Cartesian coordinates; validates VASP5 element headers and rejects singular cells |
| `xdatcar` | `.xdatcar`, `XDATCAR` | `Trajectory` | Multi-step direct/Cartesian trajectories |
| `doscar` | `.doscar`, `DOSCAR` | `DensityOfStates` | Extracts Fermi level $E_F$ and spin-polarized or unpolarized DOS |
| `eigenval` | `.eigenval`, `EIGENVAL` | `BandStructure` | Accepts optional `cell=` so reciprocal $k$-distances are weighted properly via $B = 2\pi(A^{-1})^\top$ |
| `chgcar` | `.chgcar`, `.locpot`, `CHGCAR`, `LOCPOT` | `VolumetricGrid` | Applies VASP scaling in Cartesian atom mode and reconstructs the 3D FFT grid |
| `cif` | `.cif` | `AtomsData` | Expands affine symmetry operations while preserving distinct elements on mixed-occupancy sites; records `_atom_site_occupancy` in `metadata["occupancies"]` |
| `qe_in` | `.pwi`, `.in` | `AtomsData` | Requires explicit `CELL_PARAMETERS` when `ibrav != 0`, and explicit `celldm(1)` or `A` when `alat` units are requested |
| `qe_out` | `.pwo`, QE `.out` | `AtomsData` | Extracts final geometry, cell, total energy, and Fermi level |
| `qe_bands` | `bands.dat` / `&plot` | `BandStructure` | Parses `bands.x` $k$-path coordinates and Kohn-Sham eigenvalues |
| `qe_dos` | `dos.dat` / `EFermi` | `DensityOfStates` | Parses total and spin-projected density of states |
| `cp2k_ener` | `.ener` | `Spectrum` | Converts CP2K Hartree energies to eV |
| `xtb_out` | `xtb.out` | `AtomsData` | Extracts converged geometry and total energy summary |

## Molecular Dynamics & Trajectories

| Format Key | Extensions / Filenames | Parsed Type | Supported Scope & Validation |
| --- | --- | --- | --- |
| `lammps_data` | `.lmp`, `.data`, `data.lammps` | `AtomsData` | Reads `Atoms # <style>` tags or explicit `atom_style=` (`atomic`, `charge`, `bond`, `angle`, `molecular`, `full`) plus optional `type_map=`; rejects ambiguous 6- or 8+-column tables |
| `lammpstrj` | `.lammpstrj` | `Trajectory` | Handles both orthogonal and triclinic `ITEM: BOX BOUNDS xy xz yz` bounds; requires an `element` column or explicit `type_map=` |
| `gro` | `.gro` | `AtomsData` | Converts GROMACS nanometers to Angstroms |
| `xvg` | `.xvg` | `Spectrum` | Parses Grace / GROMACS two-column curves |

## Quantum Chemistry & Volumetric Grids

| Format Key | Extensions / Filenames | Parsed Type | Supported Scope & Validation |
| --- | --- | --- | --- |
| `cube` | `.cube`, `.cub` | `VolumetricGrid` | Converts Bohr origin and voxel vectors to Angstroms; slices non-orthogonal grids cleanly via `slice_mesh()` |
| `gaussian_gjf` | `.gjf`, `.com` | `AtomsData` | Cartesian geometry block |
| `gaussian_log` | `.log` | `AtomsData` + metadata | Converged geometry, SCF energy, molecular orbitals, and harmonic IR frequencies |
| `gaussian_fchk` | `.fchk` | `AtomsData` | Cartesian coordinates and available orbital records |
| `orca_inp` | `.inp` | `AtomsData` | `* xyz` geometry blocks |
| `orca_out` | `.orca`, ORCA `.out` | `AtomsData` + metadata | Geometry, molecular orbitals, IR frequencies, and TDDFT UV-Vis transitions |

## Scientific Geometry & Analysis Safeguards

Behind the playful visuals, core geometry routines avoid common shortcuts that break on real materials:

- **Skewed & Triclinic Minimum-Image Vectors:** Instead of naive fractional rounding (which fails on skewed cells), `geometry.minimum_image_vector()` applies Gauss/LLL lattice reduction followed by a Gram-bounded integer closest-vector search across periodic axes.
- **Property-Preserving Supercells:** `AtomsData.repeat()` tiles `charges`, `velocities`, `forces`, and `atom_labels`, remaps periodic bond image shifts, and clears single-cell total energies so stale unscaled values never linger.
- **Periodic RDF & Unwrapped RMSD:** `Trajectory.compute_rdf()` integrates exact spherical-shell volumes $\frac{4\pi}{3}(r_{\text{out}}^3 - r_{\text{in}}^3)$ across all periodic image shells enclosing $r_{\max}$ (with an optional `strict=True` Wigner-Seitz inradius guard). `Trajectory.rmsd()` unwraps periodic boundary jumps before Kabsch alignment and supports atom subsets via `indices=`.

## Optional Framework Adapters

When you need specialized force-field topologies or proprietary file formats outside the built-in parsers, install the optional bridges via `pip install "xkcd-matchem[ase,pymatgen,qm,rdkit,md]"`:

- **ASE:** `from_ase(ase_atoms)`, `to_ase(atoms)`, `read_via_ase(path)`
- **pymatgen:** `from_pymatgen(struct_or_mol)`, `to_pymatgen(atoms)`
- **cclib:** `from_cclib(ccdata)`, `read_via_cclib(path)`
- **RDKit:** `from_rdkit(mol)`, `from_smiles(smiles)`
- **MDAnalysis / MDTraj:** `from_mdanalysis(universe)`, `from_mdtraj(traj)`
