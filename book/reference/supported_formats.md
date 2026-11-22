# Supported Formats & Validation Contracts

A pretty figure built from silently misread coordinates is worse than a loud error. Whenever a file uses syntax outside a built-in parser's supported subset, `xkcd-matchem` raises a descriptive `ParseError` or `StructureError` and suggests using an optional adapter (**ASE**, **pymatgen**, or **cclib**) instead of guessing.

All **29 native readers** below (plus native writers for `xyz` and `vasp` POSCAR via `xm.save()`) require only **NumPy, SciPy, and Matplotlib**.

## Molecules & Biomolecules

| Format Key | Extensions / Filenames | Parsed Type | Supported Scope & Validation |
| --- | --- | --- | --- |
| `xyz` | `.xyz`, `.extxyz` | `AtomsData` / `Trajectory` | Reads extended comment metadata (`Lattice="..."`, `pbc="..."`, `energy=`, `time=`); `read_xyz_trajectory(..., time_unit=...)` converts `time` to `fs` when `time_unit` is given on call or in frame headers (`time_unit=ps`/`fs`) and records `metadata["time_unit"] = "unspecified"` otherwise; native writer supported (`xm.save`) |
| `pdb` | `.pdb`, `.ent` | `AtomsData` / `Trajectory` | Multi-`MODEL` frames, disambiguated metal elements (`FE  ` vs ` CA `), and explicit `CONECT` bond topology with inferred bond orders |
| `sdf` | `.sdf`, `.mol` | `AtomsData` | MDL V2000 atom and bond tables with formal/aromatic bond orders and stereobond flags (`wedge`/`dash`/`wavy`) |
| `mol2` | `.mol2` | `AtomsData` | Tripos `ATOM` and `BOND` blocks including formal/aromatic bond orders and partial atomic charges |

## Crystals & Periodic DFT

| Format Key | Extensions / Filenames | Parsed Type | Supported Scope & Validation |
| --- | --- | --- | --- |
| `vasp` | `.poscar`, `.contcar`, `.vasp`, `POSCAR`, `CONTCAR` | `AtomsData` | Applies scalar, 3-component `sx sy sz`, or negative-volume factors to lattice vectors and coordinates; validates VASP 4/5 element headers; native Direct POSCAR writer supported (`xm.save`) |
| `xdatcar` | `.xdatcar`, `XDATCAR` | `Trajectory` | Multi-step Direct, Cartesian (`Cartesian`/`Kartesian`), and variable-cell NPT trajectories |
| `doscar` | `.doscar`, `DOSCAR` | `DensityOfStates` | Extracts Fermi level $E_F$ and spin-polarized or unpolarized total DOS (handles fixed-width touching floats) |
| `eigenval` | `.eigenval`, `EIGENVAL` | `BandStructure` | Accepts optional `cell=` via `xm.io.native.vasp.read_eigenval(..., cell=...)` so reciprocal $k$-distances are weighted properly via $B = 2\pi(A^{-1})^\top$ |
| `procar` | `.procar`, `PROCAR` | `BandStructure` | Parses projected band weights (`s`, `p`, `d`, `tot`) for orbital fatband dispersion plots (`xm.read_procar`) |
| `lobster_cohp` | `.lobster`, `.cohp`, `.coop`, `COHPCAR.lobster`, `COOPCAR.lobster`, `COHPCAR`, `COOPCAR` | `CohpData` | Parses LOBSTER projected `-pCOHP`/`-pCOOP` and integrated `-ICOHP` chemical-bonding curves (spin-unpolarized and spin-polarized; also detected by content header) |
| `phonopy_bands` | `band.yaml`, `phonopy_bands.yaml`, `.band.yaml` (or YAML with `nqpoint:` + `phonon:` signature) | `PhononBandStructure` | Zero-dependency Phonopy YAML reader extracting `distance`, high-symmetry `q`-point labels, and phonon branch frequencies in THz |
| `chgcar` | `.chgcar`, `.locpot`, `CHGCAR`, `LOCPOT`, `AECCAR0`, `AECCAR2` | `VolumetricGrid` | Applies VASP scaling in Cartesian atom mode, reconstructs the 3D FFT grid, and normalizes `CHGCAR`/`AECCAR*` by cell volume (`e/A^3`) while keeping `LOCPOT` unscaled (`eV`) |
| `cif` | `.cif` | `AtomsData` | Multi-block (`xm.io.native.cif.read_cif(..., block=...)`, `xm.read_cif_all`) and explicit or implicit 61-group ITA symmetry expansion with Origin Choice 1/2 handling for `Fd-3m` (#227) and `_atom_site_occupancy` |
| `qe_in` | `.pwi`, `.in` | `AtomsData` | Requires explicit `CELL_PARAMETERS` when `ibrav != 0`, and explicit `celldm(1)` or `A` when `alat` units are requested |
| `qe_out` | `.pwo`, QE `.out` (content-detected) | `AtomsData` | Extracts final geometry, cell, total energy, and Fermi level |
| `qe_bands` | `bands.dat` / `&plot` + `nbnd` | `BandStructure` | Parses `bands.x` $k$-path coordinates and Kohn-Sham eigenvalues |
| `qe_dos` | `dos.dat` / `EFermi` or `dos(E)` | `DensityOfStates` | Parses total and spin-projected density of states |
| `cp2k_ener` | `.ener` | `Spectrum` | Converts CP2K Hartree potential energies to eV vs. simulation time (fs) |
| `xtb_out` | `xtb.out` (or content-detected `xTB` + `TOTAL ENERGY`) | `AtomsData` | Extracts converged geometry, total energy, and HOMO–LUMO gap summary |

## Molecular Dynamics & Trajectories

| Format Key | Extensions / Filenames | Parsed Type | Supported Scope & Validation |
| --- | --- | --- | --- |
| `lammps_data` | `.lmp`, `.data`, `data.lammps` | `AtomsData` | Reads `Atoms # <style>` tags or explicit `atom_style=` via `xm.io.native.lammps.read_lammps_data()` (`atomic`, `charge`, `bond`, `angle`, `molecular`, `full`) plus optional `type_map=`; rejects ambiguous 6- or 8+-column tables unless styled explicitly |
| `lammpstrj` | `.lammpstrj` (or content-detected `ITEM: TIMESTEP`) | `Trajectory` | Streams frames line-by-line (`read_lammpstrj_first` stops after frame 1); handles orthogonal and triclinic `ITEM: BOX BOUNDS xy xz yz` boxes, wrapped (`x/y/z`, `xs/ys/zs`) plus `ix/iy/iz` periodic image flags (`metadata["unwrapped_positions"]`), and direct unwrapped `xu/yu/zu` coordinates (`metadata["coords_unwrapped"] = True`); stores integer step counters in `metadata["timesteps"]` (`times=None` by default, or converted to `fs` when `timestep_size=` and `time_unit=` are supplied); requires an `element` column or explicit `type_map=` via `xm.io.native.lammps.read_lammpstrj()` |
| `gro` | `.gro` | `AtomsData` | Converts GROMACS nanometers to Angstroms (orthogonal and triclinic boxes) |
| `xvg` | `.xvg` | `Spectrum` | Parses Grace / GROMACS two-column curves along with title and axis-label metadata |

## Quantum Chemistry & Volumetric Grids

| Format Key | Extensions / Filenames | Parsed Type | Supported Scope & Validation |
| --- | --- | --- | --- |
| `cube` | `.cube`, `.cub` | `VolumetricGrid` | Converts Bohr origin, atomic coordinates, and Bohr-flagged voxel vectors to Angstroms (preserving Angstrom-flagged negative-count grids); slices non-orthogonal grids cleanly via `slice_mesh()` |
| `gaussian_gjf` | `.gjf`, `.com` | `AtomsData` | Cartesian geometry block |
| `gaussian_log` | `.log` | `AtomsData` + metadata | Converged geometry, SCF energy trajectory, molecular orbitals, and harmonic IR frequencies |
| `gaussian_fchk` | `.fchk` | `AtomsData` | Atomic numbers, Cartesian coordinates (Bohr → Å), total energy (Hartree → eV), and `Alpha`/`Beta Orbital Energies` in `metadata["orbitals"]` when present |
| `orca_inp` | `.inp` | `AtomsData` | Inline `* xyz` geometry blocks and external `* xyzfile` references |
| `orca_out` | `.orca`, ORCA `.out` (content-detected) | `AtomsData` + metadata | Geometry, molecular orbitals, IR frequencies, and TDDFT UV-Vis transitions |

## Parser Safety Resource Ceilings & Format Subset Boundaries

To protect interactive sessions and automated pipelines from malformed or adversarial inputs without out-of-memory crashes, native readers enforce configurable safety limits before large allocations and raise `ParseError` immediately when a ceiling is exceeded. Override any ceiling via environment variables when working with legitimately oversized datasets:

| Environment Variable | Default Ceiling | Enforced On |
| :--- | :--- | :--- |
| `XM_MAX_ATOMS` | `200,000` atoms | Atom-table allocations in `.xyz`, VASP `POSCAR`/`CONTCAR`, `.cif`, `.cube`, and LAMMPS `.data` / `.lammpstrj` |
| `XM_MAX_GRID_VOXELS` | `64,000,000` voxels (`nx × ny × nz`) | 3D volumetric grids in Gaussian `.cube` and VASP `CHGCAR`/`LOCPOT`/`AECCAR*` |
| `XM_MAX_FRAMES` | `50,000` frames | Multi-frame trajectories (`.xyz`, `XDATCAR`, streaming `.lammpstrj`) |
| `XM_MAX_FILE_BYTES` | `209,715,200` bytes (`200 MiB`) | Input file size check prior to reading (including LAMMPS `.data` and streaming `.lammpstrj`) |

### Explicit Format Subset Boundaries

- **LAMMPS `.lammpstrj` & `.data`:** `.lammpstrj` streams line-by-line under `XM_MAX_FILE_BYTES`, `XM_MAX_ATOMS`, and `XM_MAX_FRAMES`, supporting orthogonal and triclinic `BOX BOUNDS xy xz yz`, wrapped coordinates (`x/y/z`, `xs/ys/zs`) with optional `ix/iy/iz` periodic image flags (reconstructed losslessly into `metadata["unwrapped_positions"]`), and direct unwrapped `xu/yu/zu` coordinates (`metadata["coords_unwrapped"] = True`). Raw integer step counters are stored in `metadata["timesteps"]` with `times=None` unless `timestep_size=` (and optional `time_unit=`, default `"fs"`) is passed to `read_lammpstrj()` to convert step counts into physical `fs`. `.data` enforces `XM_MAX_FILE_BYTES` and `XM_MAX_ATOMS` across header and `Atoms` tables and supports `atomic`, `charge`, `bond`, `angle`, `molecular`, and `full` styles; ambiguous unstyled tables raise `ParseError`.
- **CIF:** Supports 61 built-in ITA space groups (including Origin Choice 1/2 for `Fd-3m` #227) and explicit `_symmetry_equiv_pos_as_xyz` loops; for modulated, magnetic, or non-standard CIFs, use `xm.io.read_via_ase()` or `pymatgen`.
- **Quantum ESPRESSO `pw.in`:** Requires explicit `CELL_PARAMETERS` when `ibrav != 0`, and explicit `celldm(1)` or `A` whenever `alat` units are used.
- **Offline Interactive 3D Studio HTML:** `export_interactive_html()` escapes `<`, `>`, `&`, `\u2028`, and `\u2029` inside embedded JSON payloads and HTML-escapes document titles so untrusted filenames or titles can never break out of `<script>` tags.

## Scientific Geometry & Analysis Safeguards

Behind the playful visuals, core geometry routines avoid common shortcuts that break on real periodic materials (see the [Capability & Validation Matrix](capabilities.md) for the full internal units contract and physical assumptions):

- **Skewed & Triclinic Minimum-Image Vectors:** In a periodic crystal, two atoms near opposite box walls can actually be right next to each other across the boundary. Instead of naive fractional rounding (which picks the wrong neighbor in skewed cells), `xm.core.geometry.minimum_image_vector()` applies Gauss/LLL lattice reduction followed by a Gram-bounded integer closest-vector search across periodic axes.
- **Property-Preserving Supercells:** `AtomsData.repeat()` tiles `charges`, `velocities`, `forces`, `occupancies`, `magnetic_moments`, and `atom_labels`, remaps periodic bond image shifts, and clears single-cell total energies so stale unscaled values never linger.
- **Periodic RDF & Unwrapped RMSD/MSD:** `Trajectory.compute_rdf()` integrates exact spherical-shell volumes $\frac{4\pi}{3}(r_{\text{out}}^3 - r_{\text{in}}^3)$ across all periodic image shells enclosing $r_{\max}$ with exact $N_\alpha(N_\alpha - 1)$ same-species normalization by default (`same_species_correction=True`) and an optional `strict=True` guard enforcing $r_{\max} \le r_{\mathrm{WS}}$. `Trajectory.rmsd()` and `Trajectory.msd()` prefer lossless unwrapped positions (`xu/yu/zu` or `ix/iy/iz`) when available and warn if minimum-image frame jumps exceed $0.45\,r_{\mathrm{WS}}$.

## Optional Framework Adapters

When you need specialized force-field topologies or proprietary file formats outside the built-in parsers, install the optional bridges via `pip install "xkcd-matchem[ase,pymatgen,qm,rdkit,md]"` (or `pip install -e ".[ase,pymatgen,qm,rdkit,md]"` from a local clone):

- **ASE:** `xm.from_ase(ase_atoms)`, `xm.to_ase(atoms)`, `xm.io.read_via_ase(path)`
- **pymatgen:** `xm.from_pymatgen(struct_or_mol)`, `xm.to_pymatgen(atoms)`
- **cclib:** `xm.from_cclib(ccdata)`, `xm.io.read_via_cclib(path)`
- **RDKit:** `xm.from_rdkit(mol)`, `xm.from_smiles(smiles)`
- **MDAnalysis / MDTraj:** `xm.from_mdanalysis(universe)`, `xm.from_mdtraj(traj)`
