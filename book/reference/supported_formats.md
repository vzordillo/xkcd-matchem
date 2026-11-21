# Supported Formats & Validation Contracts

`xkcd-matchem` prioritizes **explicit validation over silent misinterpretation**: when an input file relies on conventions outside a native parser's supported subset, the reader raises a descriptive `ParseError` or `StructureError` recommending ASE, pymatgen, or cclib rather than fabricating coordinates.

## Built-in Zero-Dependency Formats

| Format Key | Extensions / Filenames | Parsed Type | Explicit Scope & Contracts |
| --- | --- | --- | --- |
| `xyz` | `.xyz`, `.extxyz` | `AtomsData` / `Trajectory` | Reads comments (`Lattice="..."`, `pbc="..."`, `energy=`) |
| `vasp` | `.poscar`, `.contcar`, `.vasp`, `POSCAR`, `CONTCAR` | `AtomsData` | Applies scalar/negative-volume factor to both lattice and Cartesian coords; validates VASP5 symbols & singular cells |
| `xdatcar` | `.xdatcar`, `XDATCAR` | `Trajectory` | Multi-step direct/Cartesian fractional frames |
| `doscar` | `.doscar`, `DOSCAR` | `DensityOfStates` | Extracts $E_F$, spin-resolved or unpolarized DOS |
| `eigenval` | `.eigenval`, `EIGENVAL` | `BandStructure` | Accepts optional `cell=` to weight reciprocal k-distances accurately using $B = 2\pi(A^{-1})^\top$ |
| `chgcar` | `.chgcar`, `.locpot`, `CHGCAR`, `LOCPOT` | `VolumetricGrid` | Applies lattice scaling to Cartesian atom modes; reconstructs 3D FFT grid |
| `cif` | `.cif` | `AtomsData` | Affine symmetry-operation expansion preserving distinct species on mixed-occupancy sites; stores `_atom_site_occupancy` in `metadata["occupancies"]` |
| `pdb` | `.pdb`, `.ent` | `AtomsData` / `Trajectory` | Multi-`MODEL` records and `CONECT` bond topology |
| `sdf` | `.sdf`, `.mol` | `AtomsData` | V2000 atom & bond blocks with formal bond orders |
| `mol2` | `.mol2` | `AtomsData` | Tripos `ATOM` and `BOND` sections |
| `cube` | `.cube`, `.cub` | `VolumetricGrid` | Converts Bohr origin & voxel vectors to Angstrom; renders affine-accurate slices via `slice_mesh()` |
| `gaussian_gjf` | `.gjf`, `.com` | `AtomsData` | Cartesian geometry extraction |
| `gaussian_log` | `.log` | `AtomsData` + metadata | Final geometry, SCF energy, MOs, and harmonic IR frequencies |
| `gaussian_fchk` | `.fchk` | `AtomsData` | Coordinates & orbital properties when present |
| `orca_inp` | `.inp` | `AtomsData` | `* xyz` geometry blocks |
| `orca_out` | `.orca`, ORCA `.out` | `AtomsData` + metadata | Geometry, MOs, IR, and TDDFT UV-Vis transitions |
| `qe_in` | `.pwi`, `.in` | `AtomsData` | Requires explicit `CELL_PARAMETERS` when `ibrav != 0`, and `celldm(1)` or `A` when `alat` units are used |
| `qe_out` | `.pwo`, QE `.out` | `AtomsData` | Extracts converged geometry, cell, total energy, and Fermi level |
| `qe_bands` | `bands.dat` / `&plot` | `BandStructure` | `bands.x` k-path & Kohn-Sham eigenvalues |
| `qe_dos` | `dos.dat` / `EFermi` | `DensityOfStates` | Total & spin-projected DOS |
| `lammps_data` | `.lmp`, `.data`, `data.lammps` | `AtomsData` | Supports `Atoms # <style>` comments or `atom_style=` (`atomic`, `charge`, `bond`, `angle`, `molecular`, `full`) and optional `type_map=`; rejects ambiguous 6/8+ column tables |
| `lammpstrj` | `.lammpstrj` | `Trajectory` | Orthogonal and triclinic `ITEM: BOX BOUNDS xy xz yz` unshifting; requires `'element'` column or `type_map=` |
| `gro` | `.gro` | `AtomsData` | Converts GROMACS nm -> Angstrom cleanly |
| `xvg` | `.xvg` | `Spectrum` | Parses Grace curves |
| `cp2k_ener` | `.ener` | `Spectrum` | Converts Hartree energies to eV |
| `xtb_out` | `xtb.out` | `AtomsData` | Final geometry & energy summary |

## Scientific Geometry & Analysis Safeguards

- **Triclinic Minimum-Image Vectors**: `geometry.minimum_image_vector()` applies Gauss/LLL lattice reduction and a Gram-bounded integer closest-vector search across active periodic subspace axes.
- **Supercell Preservation**: `AtomsData.repeat()` tiles `charges`, `velocities`, `forces`, and `atom_labels`, remaps periodic bond shifts, and clears unscaled single-cell energies.
- **RDF & MD RMSD**: `Trajectory.compute_rdf()` integrates exact spherical-shell volumes $\frac{4\pi}{3}(r_{\text{out}}^3 - r_{\text{in}}^3)$ over all periodic image shells enclosing $r_{\max}$ (with optional `strict=True` Wigner-Seitz check); `Trajectory.rmsd()` unwraps periodic jumps and supports Kabsch alignment on atom subsets (`indices=`).

## Optional Framework Adapters

Install extras via `pip install "xkcd-matchem[ase,pymatgen,qm,rdkit,md]"`:

- `from_ase(ase_atoms)`, `to_ase(atoms)`, `read_via_ase(path)`
- `from_pymatgen(struct_or_mol)`, `to_pymatgen(atoms)`
- `from_cclib(ccdata)`, `read_via_cclib(path)`
- `from_rdkit(mol)`, `from_smiles(smiles)`
- `from_mdanalysis(universe)`, `from_mdtraj(traj)`
