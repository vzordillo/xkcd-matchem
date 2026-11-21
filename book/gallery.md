# Visual Gallery

Every figure and animated GIF below is generated automatically by `scripts/generate_gallery.py` using real bundled sample files, unified `xkcd Script` typography, and depth-sorted 3D scenes. Click any tutorial link to see the exact code behind each plot.

## Animated Motion & Theme Explorer

Looping GIFs are rendered with `xm.animate_trajectory()` (requires `pip install "xkcd-matchem[animation]"`) or custom camera/preset loops:

| 3D Coordination Polyhedra Orbit | Climbing-Image NEB Pathway | Live Theme & Palette Switching |
| :---: | :---: | :---: |
| ![TiO2 coordination polyhedra orbit](_static/gallery/crystal_orbit.gif) | ![SN2 NEB reaction animation](_static/gallery/neb_reaction.gif) | ![Theme and color scheme switcher](_static/gallery/theme_switcher.gif) |

## Full Multi-Domain Showcase

A quick overview spanning molecules, periodic crystals, electronic structure, spectroscopy, and thermodynamics in one view:

![Hero showcase](_static/gallery/hero_showcase.png)

## Molecules: 3D Ball-and-Stick, Spacefill & 2D Skeletal

Compare `plot_molecule(..., style="ball_and_stick")`, `style="spacefill"`, and `mode="2d"` with seamless two-tone bonds, automatic organic bond-order perception (ring and carbonyl double bonds), and inline element legends.
See [Tutorial 02: Molecules in 3D, 2D & Interactive Studio](tutorials/02_molecules_3d_and_2d.ipynb).

![Molecular renderers](_static/gallery/molecules_3d_2d.png)

## Crystals, Slabs, Miller Planes & Powder XRD

Combine `plot_crystal()` (with `(111)` Miller slices and supercells), `plot_slab(..., view="side")` (automatic shaded slab/vacuum band detection for $\text{TiO}_2(110)$), `plot_coordination()`, and pure-NumPy $\text{Cu K}\alpha_1$ `plot_xrd()`.
See [Tutorial 03: Crystals, Slabs & Polyhedra](tutorials/03_crystals_slabs_and_polyhedra.ipynb).

![Crystals slabs polyhedra and XRD](_static/gallery/crystal_polyhedra_xrd.png)

## Electronic Band Structure & Density of States

Fermi-aligned silicon bands and DOS parsed natively from Quantum ESPRESSO `bands.dat` and `dos.dat` via `plot_bands_dos()`.
See [Tutorial 04: Electronic Band Structures & DOS](tutorials/04_electronic_structure_bands_dos.ipynb).

![Silicon bands and DOS](_static/gallery/bands_dos.png)

## Molecular Orbitals & Volumetric Charge Density

Pair ORCA molecular orbital energy ladders (`plot_mo_diagram()`) with affine-accurate 2D slices of Gaussian `.cube` volumetric grids (`plot_volumetric_slice()`).
See [Tutorial 05: Molecular Orbitals & Volumetric Cubes](tutorials/05_orbitals_and_volumetric_cubes.ipynb).

![MOs and CUBE](_static/gallery/mo_and_cube.png)

## IR & UV-Vis Spectra (Exploratory Sketches)

Visualize Gaussian vibrational IR transitions and ORCA TDDFT UV-Vis peaks with `plot_spectrum()` and customizable line broadening.
See [Tutorial 06: Spectroscopy & Powder XRD](tutorials/06_spectroscopy_and_powder_xrd.ipynb).

![IR and UV-Vis spectroscopy](_static/gallery/spectroscopy.png)

## Reaction Energy Profiles & Climbing-Image NEB

Sketch multi-step catalytic profiles (`plot_reaction_profile()`) alongside spline-interpolated minimum-energy pathways (`plot_neb()`) in eV or kcal/mol.
See [Tutorial 07: Reaction Pathways & NEB](tutorials/07_reaction_pathways_and_neb.ipynb).

![Reaction profile and NEB](_static/gallery/reaction_neb.png)

## Molecular Dynamics RMSD & Radial Distribution

Plot unwrapped periodic MD RMSD traces (`plot_md_timeseries()`) and radial distribution functions (`plot_rdf()`) from LAMMPS and GROMACS outputs.
See [Tutorial 08: MD Trajectories & RDF](tutorials/08_md_trajectories_and_rdf.ipynb).

![MD trajectory and RDF](_static/gallery/md_trajectory_rdf.png)

## Convex Hull, Equations of State & ML Parity

Bring together binary stability hulls (`plot_phase_diagram_2d()`), third-order Birch-Murnaghan EOS fits (`plot_eos()`), and ML interatomic potential benchmarks (`plot_parity()`).
See [Tutorial 09: Thermodynamics & ML Parity](tutorials/09_thermodynamics_and_ml_parity.ipynb).

![Thermodynamics hull EOS and parity](_static/gallery/thermo_hull_eos.png)

## Opt-In Poster Composition (`Board`, `Container`, `Card`, `Badge`)

Individual plots always stay borderless on an open canvas. When assembling a full conference poster or group-meeting handout in millimeter coordinates, opt into `xm.Board` — you can group and **stack multiple cards vertically or side by side inside container boxes**. See [Tutorial 10: Opt-In Poster Boards & Interoperability](tutorials/10_posters_boards_and_interoperability.ipynb):

![Poster board example](_static/gallery/poster_board.png)

## Periodic Table of Elements & Property Heatmaps

Render all 118 elements by chemical family, map custom datasets or built-in atomic properties (`electronegativity`, `covalent_radius`, `vdw_radius`, `atomic_mass`) with hatched missing tiles, or spotlight formulas like `"LiFePO4"` using `plot_periodic_table()`.
See [Tutorial 11: Periodic Table, Scrolls, Grids, Icons & Graphs](tutorials/11_periodic_table_scrolls_and_informatics.ipynb).

![xkcd Periodic Table](_static/gallery/periodic_table.png)

## Materials Informatics (Ternary Hulls, Space Group Sunbursts & Histograms)

Inspired by `pymatviz`, explore ternary Gibbs convex hulls (`plot_ternary_phase_diagram()`), two-ring crystal-system/space-group sunbursts (`plot_spacegroup_distribution()`), and hatched property histograms with KDE overlays (`plot_histogram()`).
See [Tutorial 11: Periodic Table, Scrolls, Grids, Icons & Graphs](tutorials/11_periodic_table_scrolls_and_informatics.ipynb).

![Materials informatics gallery](_static/gallery/materials_informatics.png)

## Background Grids, Hatched Shading & Vintage Scroll Cards

Layer notebook/graph/dotgrid/isometric/blueprint paper backgrounds (`background_grid=`, `apply_background_grid()`, `clear_background_grid()`), hatched or stippled fills (`shade="cycle" | "crosshatch" | "dots"`), and portrait or landscape vintage parchment scroll cards (`plot_scroll_card()`, `board.add_scroll_card()`) with notebook-style line/grid text alignment (`align_to_grid=True`, `card.write_line()`) or clean grid-free parchment (`background_grid=None`).
See [Tutorial 11: Periodic Table, Scrolls, Grids, Icons & Graphs](tutorials/11_periodic_table_scrolls_and_informatics.ipynb).

![Scrolls, grids, and hatched shading](_static/gallery/scrolls_grids_shading.png)

## xkcd MatChem Icons, Element Badges & GNN Structure Graphs

Drop playful vector icons (`flask`, `crystal`, `battery`, `graphene`, `catalyst`, `dft`, `neural_net`) and element badges (`draw_element_badge()`) onto any figure, or turn molecules and periodic crystals into GNN graphs with dashed periodic boundary arcs (`plot_structure_graph()`, `plot_graph_network()`).
See [Tutorial 11: Periodic Table, Scrolls, Grids, Icons & Graphs](tutorials/11_periodic_table_scrolls_and_informatics.ipynb).

![Icons and GNN structure graphs](_static/gallery/icons_and_graphs.png)

## Bond-Free Crystals & Planar Flat 2D Crystal GNN Graphs

Need pure atomic lattice sites inside a unit cell without any bond network? Pass `show_bonds=False` to `plot_crystal()`, `plot_slab()`, or `plot_molecule()` (or toggle **Show bonds** in the Interactive 3D Studio). For flat 2D topological and crystallographic quotient multigraphs with fanned periodic translation edges (`000`, `100`, `010`, `001`), use `plot_structure_graph(..., layout="planar")` and `multi_edges=True, show_shift_labels=True`.
See [Tutorial 11: Periodic Table, Scrolls, Grids, Icons & Graphs](tutorials/11_periodic_table_scrolls_and_informatics.ipynb).

![Bond-free crystals and planar 2D GNN graphs](_static/gallery/crystal_graphs_2d.png)
