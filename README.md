# xkcd-matchem

**Hand-drawn xkcd-style visualization for molecules, crystals, electronic structure, spectroscopy, and materials chemistry.**

[![CI](https://github.com/vzordillo/xkcd-matchem/actions/workflows/ci.yml/badge.svg)](https://github.com/vzordillo/xkcd-matchem/actions/workflows/ci.yml)
[![Docs](https://github.com/vzordillo/xkcd-matchem/actions/workflows/docs.yml/badge.svg)](https://vzordillo.github.io/xkcd-matchem/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Docs License: CC BY 4.0](https://img.shields.io/badge/Docs-CC%20BY%204.0-lightgrey.svg)](book/LICENSE-CC-BY-4.0.txt)

![xkcd-matchem Showcase](book/_static/gallery/hero_showcase.png)

## Why `xkcd-matchem`?

`xkcd-matchem` turns computational chemistry and materials science files directly into expressive, publication-clean, hand-drawn figures without forcing you to install a heavy stack first:

- **Lightweight core**: requires only `numpy`, `matplotlib`, and `scipy`.
- **12 native zero-dependency parsers**: `XYZ`/`extXYZ`, VASP (`POSCAR`, `CONTCAR`, `XDATCAR`, `DOSCAR`, `EIGENVAL`, `CHGCAR`, `LOCPOT`), `CIF` (with symmetry expansion), `PDB`, `SDF`/`MOL`/`MOL2`, Gaussian `.cube`/`.gjf`/`.log`/`.fchk`, ORCA `.inp`/`.out`, Quantum ESPRESSO `pw.in`/`pw.out`/`bands.dat`/`dos.dat`, LAMMPS `.data`/`.lammpstrj`, GROMACS `.gro`/`.xvg`, and CP2K/xTB summaries.
- **Optional framework adapters**: lazy adapters for `ase`, `pymatgen`, `cclib`, `rdkit`, `MDAnalysis`, and `mdtraj`.
- **Rich materials chemistry renderers & plots**: depth-sorted 3D spheres & two-tone bonds, periodic bond stubs, `ConvexHull` coordination polyhedra, Miller planes, 2D PCA skeletal diagrams, bands + DOS, MO energy ladders, volumetric CUBE slices, pure-NumPy powder XRD using Cromer-Mann LA-3816 form factors, IR/UV-Vis spectra, NEB & reaction pathways, MD RMSD/RDF/GIF animation, 2D binary convex hulls, Birch-Murnaghan EOS fits, Pourbaix stability maps, ML parity charts, and physical-millimeter `Board`/`Card`/`Badge` figure composition.
- **Legible open fonts bundled**: includes **Patrick Hand** and **Comic Neue** under **SIL OFL 1.1**, automatically pairing with `DejaVu Sans` so Unicode units (`A`, degrees, Greek letters, minus signs) never break.

## Installation

```bash
pip install xkcd-matchem
```

Optional extras install ecosystem bridges, GIF export, or documentation tools:

```bash
pip install "xkcd-matchem[ase,pymatgen,qm,rdkit,md,animation]"
```

## Quickstart

```python
import xkcd_matchem as xm

# One-line crystal, polyhedra, and powder XRD
ax = xm.plot_crystal("si_diamond.cif", miller_planes=[(1, 1, 1)])
xm.save_figure(ax.figure, "si_crystal.png")

# Compose multi-panel poster boards in physical millimeters
board = xm.Board(240, 140, title="Materials Overview")
left = board.add_card(8, 26, 108, 104, title="TiO2 Polyhedra")
right = board.add_card(124, 26, 108, 104, title="NaCl Powder XRD")
xm.plot_coordination("tio2_slab.poscar", "Ti", "O", ax=left.axes)
xm.plot_xrd("nacl.cif", ax=right.axes)
board.save("overview.png")
```

## Documentation & Interactive Tutorials

Browse the full **Jupyter Book** with 10 executable tutorials, gallery, Colab/Binder launch buttons, and verified scientific attributions at:

**<https://vzordillo.github.io/xkcd-matchem/>**

## Licensing & Attributions

- **Source code (`src/`, `tests/`, `scripts/`)**: Apache License 2.0 ([LICENSE](LICENSE)).
- **Documentation, tutorials & gallery (`book/`)**: CC BY 4.0 ([book/LICENSE-CC-BY-4.0.txt](book/LICENSE-CC-BY-4.0.txt)).
- **Bundled fonts**: SIL Open Font License 1.1 (`PatrickHand-OFL.txt`, `ComicNeue-OFL.txt`).
- **Scientific constants & element tables**: Verified against CODATA 2022, IUPAC/CIAAW atomic weights, Blue Obelisk Data Repository (CC0 1.0), Alvarez (2013) & Mantina et al. (2009) van der Waals radii, Cromer-Mann LA-3816 neutral X-ray scattering factors, Okabe-Ito CUD, and Jmol Colors. See [NOTICE](NOTICE) and [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
