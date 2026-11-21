# xkcd-matchem: An xkcd-Style Visualization Suite for Chemistry and Materials Science Communication

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/downloads/)
[![Code License: Apache-2.0](https://img.shields.io/badge/code-Apache--2.0-green.svg)](LICENSE)
[![Docs License: CC BY 4.0](https://img.shields.io/badge/docs-CC%20BY%204.0-lightgrey.svg)](LICENSE-DOCS)
[![Fonts: SIL OFL 1.1](https://img.shields.io/badge/fonts-SIL%20OFL%201.1-orange.svg)](THIRD_PARTY_LICENSES.md)

**A more fun — yet crisp and readable — xkcd-style way to view your molecules, crystals, and materials chemistry.**

> **How this started:**
> I have always had fun reading xkcd comics and loved how comic-style fonts and playful sketches make technical ideas feel instantly approachable. While putting together a research poster recently, I wanted my molecules and plots to look lively without becoming messy. What started as a quick helper for that poster grew into `xkcd-matchem` — an open Python toolkit for fellow students, researchers, and scientists to bring a bit of personality to their **science communication (scicomm)**, posters, group talks, and lectures.

---

## See It in Motion

| 3D Coordination Polyhedra Orbit | Climbing-Image NEB Pathway | Live Theme & Palette Switching |
| :---: | :---: | :---: |
| ![TiO2 coordination polyhedra orbit](book/_static/gallery/crystal_orbit.gif) | ![SN2 NEB reaction animation](book/_static/gallery/neb_reaction.gif) | ![Theme and color scheme switcher](book/_static/gallery/theme_switcher.gif) |
| *Orbiting $\text{TiO}_6$ polyhedra with depth-sorted bonds and faces* | *Animating a 9-image $\text{S}_\text{N}2$ reaction path into a looping GIF* | *Switching light, warm paper, and dark chalkboard presets on the fly* |

> **A quick note on scope:** `xkcd-matchem` is built for playful exploration, posters, talks, and scicomm. Because intentional xkcd line wiggle can soften fine peak shoulders or subtle curve features, standard straight-line Matplotlib plots remain the better choice for archival journal figures and quantitative peak fitting.

![xkcd-matchem hero showcase](book/_static/gallery/hero_showcase.png)

---

## What Makes It Fun (and Practical)

- **One-line defaults that look clean right away:** Load a molecule, crystal, band structure, or trajectory and plot it immediately on an open white canvas with unified **Patrick Hand** typography (SIL OFL 1.1), split half-bonds, white specular highlights, and zero clutter.
- **Customize every visual detail:** Swap individual atom colors with a plain Python dict (`{"Ti": "#E07A5F", "O": "#3D405B"}`), create reusable themes (`register_preset`) and color schemes (`register_palette`, `register_element_palette`), pick crystallographic camera angles (`Camera.from_preset`), or restyle your own `plt.subplots()` with `style_existing_axes()`.
- **Spin structures offline in 3D:** Launch the browser-based **Interactive 3D Studio** (`xm.view_structure(...)` or `xkcd-matchem view ...`) directly in Jupyter or as a standalone offline HTML file — rotate, build supercells, tweak colors live, export PNG/SVG, and copy the matching Python code.
- **Playful styling backed by real geometry:** Native parsers require only **NumPy, SciPy, and Matplotlib**, and reject ambiguous inputs loudly rather than quietly plotting wrong coordinates.

---

## Visual Tour

| Molecules (3D & 2D) | Crystals, Slabs & Powder XRD |
| :---: | :---: |
| ![Molecular renderers](book/_static/gallery/molecules_3d_2d.png) | ![Crystal and XRD](book/_static/gallery/crystal_polyhedra_xrd.png) |

| Electronic Bands & DOS | Orbitals & Volumetric Cubes |
| :---: | :---: |
| ![Bands and DOS](book/_static/gallery/bands_dos.png) | ![MOs and CUBE](book/_static/gallery/mo_and_cube.png) |

| IR & UV-Vis Spectra | Reaction Profiles & NEB |
| :---: | :---: |
| ![Spectroscopy](book/_static/gallery/spectroscopy.png) | ![Reaction pathways](book/_static/gallery/reaction_neb.png) |

| MD RMSD & Radial Distribution | Convex Hull, EOS & ML Parity |
| :---: | :---: |
| ![MD and RDF](book/_static/gallery/md_trajectory_rdf.png) | ![Thermodynamics and parity](book/_static/gallery/thermo_hull_eos.png) |

Putting together a research poster? Opt into `xm.Board`, `Container`, `Card`, and `Badge` for millimeter-accurate multi-panel layouts:

![Opt-in poster board](book/_static/gallery/poster_board.png)

---

## Installation

Install the core package (requires only Python 3.10+, NumPy, SciPy, and Matplotlib — **15 bundled sample files** and SIL OFL 1.1 fonts are included):

```bash
pip install xkcd-matchem
```

Add only the optional ecosystem extras you need (`ase`, `pymatgen`, `rdkit`, `qm` for `cclib`, `md` for `MDAnalysis` + `mdtraj`, and `animation` for Pillow GIF export):

```bash
pip install "xkcd-matchem[ase,pymatgen,rdkit,animation]"
```

For local development, tests, and building the Jupyter Book documentation:

```bash
git clone https://github.com/vzordillo/xkcd-matchem.git
cd xkcd-matchem
pip install -e ".[animation,dev,docs]"
```

---

## Quickstart

### Start in One Line (Default Presets)

All filenames below work out of the box using bundled examples — swap in any path to your own simulation files:

```python
import xkcd_matchem as xm

# Molecules, crystals, slabs, and coordination polyhedra
xm.plot_molecule("aspirin.xyz", title="Aspirin 3D")
xm.plot_crystal("si_diamond.cif", miller_planes=[(1, 1, 1)], title="Si (111)")
xm.plot_slab("tio2_slab.poscar", supercell=(3, 2, 1), view="side")
xm.plot_coordination("tio2_slab.poscar", "Ti", "O", title="TiO2 Coordination Polyhedra")

# Electronic structure, powder XRD, reaction barriers, and animated GIFs
bands = xm.load("si_qe_bands.dat")
dos = xm.load("si_qe_dos.dat")
xm.plot_bands_dos(bands, dos, ylim=(-8.0, 10.0), title="Silicon Bands & DOS")
xm.plot_xrd("nacl.cif", label_peaks=4)
xm.plot_neb(xm.load_trajectory("sn2_neb.xyz"))
xm.animate_trajectory(xm.load_trajectory("sn2_neb.xyz"), "sn2_path.gif", fps=6)

# Open the offline Interactive 3D Studio in Jupyter or a browser
xm.view_structure("tio2_slab.poscar", style="polyhedra", polyhedra=("Ti", "O"))
```

### Make It Yours (Custom Themes, Colors & Cameras)

```python
import matplotlib.pyplot as plt
import xkcd_matchem as xm

# 1. Derive reusable custom themes and element color schemes
xm.register_preset(
    "studio_chalk",
    base="dark_chalk",
    sketch_scale=1.1,
    linewidth=2.0,
)
xm.register_element_palette(
    "terracotta",
    {"Ti": "#E07A5F", "O": "#3D405B", "Si": "#F2CC8F"},
    base="pastel",
)

# 2. Override colors inline and pick a crystallographic camera angle
cam = xm.Camera.from_preset("perspective", zoom=1.15)
ax = xm.plot_crystal(
    "mos2_2h.poscar",
    supercell=(2, 2, 1),
    camera=cam,
    palette={"Mo": "#2A9D8F", "S": "#E9C46A"},
    show_cell="wireframe",
)

# 3. Apply your theme cleanly across custom multi-panel Matplotlib figures
with xm.xkcd_style("studio_chalk", palette=["#F4A261", "#81B29A"]):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    xm.style_existing_axes(axes[0], preset="studio_chalk")
    xm.plot_xrd("si_diamond.cif", ax=axes[0], preset="studio_chalk")
    xm.plot_molecule(
        "alaninedipeptide.pdb",
        ax=axes[1],
        preset="studio_chalk",
        palette="terracotta",
    )
```

---

## Command-Line Interface

Prefer quick terminal renders or instant browser previews? Use the `xkcd-matchem` CLI:

```bash
# Explore bundled examples, supported formats, and structure metadata
xkcd-matchem samples
xkcd-matchem formats
xkcd-matchem info tio2_slab.poscar

# Render static figures with supercells, camera presets, and custom atom colors
xkcd-matchem render tio2_slab.poscar -o tio2.png --supercell 2 1 1 --view perspective --colors Ti=#E07A5F O=#264653
xkcd-matchem xrd si_diamond.cif -o si_xrd.png

# Open a standalone offline Interactive 3D Studio in your browser
xkcd-matchem view mos2_2h.poscar -o mos2_studio.html --supercell 2 2 1 --backend browser
```

---

## Supported Formats & Scientific Safeguards

All **26 built-in readers** work with zero extra dependencies beyond NumPy, SciPy, and Matplotlib:

| Domain | Built-in Formats |
| --- | --- |
| **Molecules & Biomolecules** | XYZ / extXYZ, PDB (`MODEL` + `CONECT`), SDF / MOL (V2000), MOL2 |
| **Crystals & Periodic DFT** | VASP (`POSCAR`, `CONTCAR`, `XDATCAR`, `DOSCAR`, `EIGENVAL`, `CHGCAR`, `LOCPOT`), CIF (affine symmetry expansion + occupancy metadata), Quantum ESPRESSO (`pw.in`, `pw.out`, `bands.dat`, `dos.dat`), CP2K `.ener`, xTB `.out` |
| **Molecular Dynamics** | LAMMPS data (explicit style / `type_map`) & `.lammpstrj` (orthogonal + triclinic), GROMACS `.gro` & `.xvg` |
| **Quantum Chem & Grids** | Gaussian `.cube`, `.gjf`, `.log`, `.fchk`, ORCA `.inp`, `.out` |

**Built-in scientific care:**
- **Strict validation over silent guesses:** Unsupported QE cell blocks, ambiguous LAMMPS column tables, or singular lattice matrices raise clear errors pointing to optional **ASE**, **pymatgen**, **RDKit**, **cclib**, **MDAnalysis**, or **MDTraj** adapters.
- **Accurate periodic geometry:** Triclinic minimum-image vectors use LLL lattice reduction + Gram-bounded closest-vector search; `AtomsData.repeat()` preserves atomic properties across supercells; `Trajectory.compute_rdf()` integrates exact spherical-shell volumes; `Trajectory.rmsd()` unwraps periodic jumps before Kabsch alignment; and `VolumetricGrid.slice_mesh()` projects skewed cells onto an orthonormal 2D plane.

See [Supported Formats & Contracts](book/reference/supported_formats.md) for the complete specification.

---

## How to Cite

If `xkcd-matchem` helps you create a poster, lecture slide, tutorial, or outreach figure, please cite it using [`CITATION.cff`](CITATION.cff) (or GitHub's *"Cite this repository"* button; a Zenodo archive DOI can be added upon public release):

> Ordillo, V. (2026). *xkcd-matchem: An xkcd-Style Visualization Suite for Chemistry and Materials Science Communication* (Version 0.1.0) [Computer software]. https://github.com/vzordillo/xkcd-matchem

```bibtex
@software{ordillo2026xkcdmatchem,
  author  = {Ordillo, Viejay},
  title   = {{xkcd-matchem}: An xkcd-Style Visualization Suite for Chemistry and Materials Science Communication},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/vzordillo/xkcd-matchem}
}
```

---

## Acknowledgements

This repository was developed semi-autonomously with **Google DeepMind's Gemini** frontier models inside **[Google Antigravity](https://antigravity.google)**, supported through Google's **Antigravity for Science** academic researcher access program (providing six months of complimentary top-tier Google Antigravity and frontier Gemini access to scientists and academic researchers for non-commercial research). Many thanks to the **Google Antigravity**, **Google Research**, and **Google DeepMind** teams for supporting open scientific exploration and science communication tooling.

---

## Licenses & Scientific Attributions

- **Code**: [Apache License, Version 2.0](LICENSE)
- **Documentation & Tutorials**: [CC BY 4.0](LICENSE-DOCS)
- **Bundled Fonts**: `Patrick Hand` and `Comic Neue` under the **SIL Open Font License 1.1** ([`NOTICE`](NOTICE), [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md))
- **Scientific Reference Data**: CODATA 2022 (`Rev. Mod. Phys.` 97, 025002), IUPAC 2021 / CIAAW atomic weights, Blue Obelisk BODR covalent radii, Alvarez (2013) & Mantina et al. (2009) vdW radii, Cromer-Mann (LA-3816) + exact H 1s form factors, Okabe-Ito CUD palette, and Jmol element colors. Full citations live in [Citation & Attributions](book/reference/attributions.md).
