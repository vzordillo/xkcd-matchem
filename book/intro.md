# xkcd-matchem

**Hand-drawn xkcd-style visualization for molecules, crystals, electronic structure, spectroscopy, and materials chemistry.**

```{figure} _static/gallery/hero_showcase.png
:name: hero-showcase
:width: 100%

Six-panel xkcd-matchem showcase rendered purely with NumPy, SciPy, and Matplotlib using bundled SIL OFL 1.1 handwritten fonts.
```

`xkcd-matchem` turns computational chemistry and materials science files directly into expressive, hand-drawn publication figures with **zero mandatory domain dependencies** beyond `numpy`, `matplotlib`, and `scipy`.

::::{grid} 2
:gutter: 3

:::{grid-item-card} Zero-Dependency Native I/O
Parse XYZ/extXYZ, VASP (`POSCAR`/`XDATCAR`/`DOSCAR`/`EIGENVAL`/`CHGCAR`), CIF (with symmetry ops), PDB, SDF/MOL/MOL2, Gaussian `.cube`/`.log`/`.gjf`/`.fchk`, ORCA `.inp`/`.out`, Quantum ESPRESSO, LAMMPS, GROMACS, and CP2K/xTB directly out of the box.
:::

:::{grid-item-card} Crystallography & Quantum Plots
Painter's-algorithm 3D spheres & two-tone bonds, periodic bond stubs, `ConvexHull` coordination polyhedra, Miller planes, bands + DOS, MO ladders, volumetric CUBE slices, pure-NumPy powder XRD, NEB barriers, RDFs, binary hulls, and Birch-Murnaghan EOS fits.
:::

:::{grid-item-card} Colorblind-Safe & Legible Fonts
Okabe-Ito CUD series palettes, Jmol & hand-drawn element schemes, and bundled **Patrick Hand** and **Comic Neue** (SIL OFL 1.1) paired with DejaVu Sans so units, Greek letters, and minus signs render without missing glyphs.
:::

:::{grid-item-card} Physical-mm Poster Boards
Compose multi-panel graphical abstracts and slide decks in millimeters using `Board`, `Container`, `Card`, and `Badge` with vector SVG and embedded-font PDF export.
:::
::::
