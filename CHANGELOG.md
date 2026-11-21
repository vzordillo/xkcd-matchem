# Changelog

All notable changes to `xkcd-matchem` will be documented in this file.

## [0.1.0] - 2026-11-21

### Added
- Standalone package architecture with Apache-2.0 code license, CC BY 4.0 Jupyter Book documentation/tutorials, and SIL OFL 1.1 bundled handwritten fonts (`Patrick Hand` and `Comic Neue`).
- Verified scientific constants and periodic tables (`CODATA 2022`, CIAAW/IUPAC standard atomic weights, Blue Obelisk Data Repository covalent radii/metadata, Alvarez & Mantina van der Waals radii, Cromer-Mann LA-3816 neutral atomic X-ray form factors).
- Theme manager (`xkcd_style`) with five presets (`classic`, `notebook`, `poster`, `minimal`, `dark_chalk`), Okabe-Ito colorblind-safe palettes, Jmol element colors, and opt-in `fetch_xkcd_script()` helper.
- Zero-dependency native parsers for XYZ/extXYZ, VASP (`POSCAR`/`CONTCAR`/`XDATCAR`/`DOSCAR`/`EIGENVAL`/`CHGCAR`), CIF (with affine symmetry expansion), PDB, SDF/MOL/MOL2, Gaussian `.cube`/`.gjf`/`.log`/`.fchk`, ORCA `.inp`/`.out`, Quantum ESPRESSO `pw.in`/`pw.out`/`bands.dat`/`dos.dat`, LAMMPS `.data`/`.lammpstrj`, GROMACS `.gro`/`.xvg`, and CP2K/xTB outputs, plus lazy optional adapters for ASE, pymatgen, cclib, RDKit, MDAnalysis, and mdtraj.
- Hand-drawn 3D depth-sorted structure renderer (`ball_and_stick`, `spacefill`, `stick`, `polyhedra`, unit cells, Miller planes), 2D skeletal renderer, electronic structure plots (bands, DOS, MO diagram, volumetric CUBE slices), spectroscopy & pure-NumPy powder XRD, reaction/NEB profiles, MD trajectories/RDF/RMSD/GIF animation, 2D binary phase hull, Birch-Murnaghan EOS, Pourbaix diagram, parity/pipeline charts, physical-millimeter `Board`/`Card`/`Badge` layout engine, CLI, and 10 interactive Jupyter Book tutorials.
