# Visual Gallery

Explore hand-drawn visualizations generated directly from the bundled chemistry and materials examples in `src/xkcd_matchem/data/samples/`.

## 1. Molecules in 3D, Spacefill & 2D Skeletal Form

```{figure} _static/gallery/molecules_3d_2d.png
:width: 95%

Aspirin (`aspirin.xyz`) rendered in depth-sorted ball-and-stick, van der Waals spacefill, and PCA-aligned 2D skeletal styles.
```

## 2. Crystals, Miller Planes, Polyhedra & Powder XRD

```{figure} _static/gallery/crystal_polyhedra_xrd.png
:width: 95%

Diamond Si with a `(111)` Miller slice, 2x2x1 2H-MoS2 supercell with dashed back unit-cell edges, and Cu K-alpha1 powder XRD computed from Cromer-Mann LA-3816 form factors.
```

## 3. Electronic Band Structure, DOS, MOs & Volumetric CUBE

```{figure} _static/gallery/bands_dos.png
:width: 85%

Synchronized Silicon band dispersion and vertical density of states from Quantum ESPRESSO outputs.
```

```{figure} _static/gallery/mo_and_cube.png
:width: 88%

ORCA benzene molecular orbital energy ladder paired with a 2D contour slice through a Gaussian `.cube` grid.
```

## 4. Spectroscopy & Reaction / NEB Pathways

```{figure} _static/gallery/spectroscopy.png
:width: 90%

Gaussian H2O IR spectrum (inverted wavenumber axis) and ORCA benzene TD-DFT UV-Vis absorption curve.
```

```{figure} _static/gallery/reaction_neb.png
:width: 90%

Discrete stationary-state catalytic reaction diagram alongside an SN2 Cl- + CH3Cl NEB minimum-energy path (`sn2_neb.xyz`).
```

## 5. MD Trajectories, Thermodynamics & Screening Posters

```{figure} _static/gallery/md_trajectory_rdf.png
:width: 90%

Cu LAMMPS trajectory RMSD with running mean alongside a GROMACS `.xvg` liquid water O-O radial distribution function.
```

```{figure} _static/gallery/thermo_hull_eos.png
:width: 95%

Binary convex hull stability diagram, third-order Birch-Murnaghan EOS fit, and ML parity scatter with MAE/RMSE/R^2 metrics.
```

```{figure} _static/gallery/poster_board.png
:width: 95%

Full 4-card graphical poster composed in millimeters using `Board`, `Container`, `Card`, and `Badge` with the `poster` preset.
```
