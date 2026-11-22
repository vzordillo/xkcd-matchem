# Capability & Validation Matrix

`xkcd-matchem` is built for approachable chemistry and materials science communication — posters, talks, lectures, and group-meeting stories — backed by honest geometry, validated core algorithms, and clear boundaries. This page maps every feature family to its scientific scope, documents our canonical internal units contract, and states the physical assumptions behind our numerical helpers so you always know what a function guarantees and when to reach for a specialized archival solver.

## Capability Categories

Every feature family in `xkcd-matchem` falls into one of five explicit categories:

- **`Visualization`** — Renders user-supplied structures, trajectories, grids, curves, and metadata with clean hand-drawn styling, depth sorting, and honest coordinate frames without modifying the underlying physical meaning.
- **`Numerical Analysis`** — Quantitative scientific algorithms validated against analytical invariants and literature tolerances in `tests/test_scientific_validation.py` (alongside unit and geometry invariant suites in `tests/test_core.py`, `tests/test_v010_wave.py`, and `tests/test_plots_spec_rxn_thermo.py`).
- **`Approximate Model`** — Useful physical approximations with explicitly documented simplifications. Ideal for exploratory sketches, teaching, and qualitative comparison; use dedicated domain codes for publication-grade quantitative refinement.
- **`Educational Illustration`** — Schematic diagrams, vector icons, stick-scientist storytelling, and layout templates designed for science communication rather than numerical calculation.
- **`Experimental`** — Newer analysis, provenance, or multi-frame rendering helpers whose optional parameters or return fields may evolve prior to `1.0`.

## Feature Capability & Validation Matrix

| Feature Family | Key Entry Points | Category | Scientific Scope, Validation & Boundaries |
| :--- | :--- | :--- | :--- |
| **3D & 2D Molecular / Crystal Rendering** | `plot_molecule`, `plot_crystal`, `plot_slab`, `plot_coordination`, `render_structure_3d`, `render_structure_2d`, `view_structure`, `export_interactive_html` | `Visualization` | Depth-sorted 3D scenes, Miller planes, coordination cages, and unified-Å 2D skeletal structures; covalent-radii bond perception and organic bond-order heuristics apply only when explicit topology (`CONECT`/SDF/MOL2) is absent. |
| **Electronic Bands, DOS, Fatbands, COHP, Phonons & MO Ladders** | `plot_bands`, `plot_dos`, `plot_bands_dos`, `plot_cohp`, `plot_phonon_bands`, `plot_mo_diagram`, `plot_volumetric_slice` | `Visualization` | Displays DFT/quantum-chemistry outputs faithfully with straight coordinate dividers (`sketch=(0, 0, 0)`). `VolumetricGrid.slice_mesh()` uses exact orthonormal projection for non-orthogonal cells (`Numerical Analysis`). |
| **Electronic Band-Gap Classification** | `BandStructure.analyze_band_gap`, `BandGapResult`, `BandStructure.band_gap` | `Numerical Analysis` | Separates spin-polarized, spin-degenerate, and Fermi-level conventions; computes direct and indirect gaps, tolerates finite smearing, and marks ambiguous/unfilled cases `is_indeterminate=True`. |
| **Periodic Geometry, Minimum Image & Neighbor Search** | `minimum_image_vector`, `max_min_image_radius`, `find_periodic_neighbors`, `miller_plane_polygon`, `AtomsData.repeat` | `Numerical Analysis` | Exact Gauss/LLL lattice reduction + Gram-bounded closest-vector search across skewed/triclinic periodic cells; property-preserving supercell tiling. |
| **3D Volumetric Isosurfaces** | `extract_isosurface`, `plot_isosurface`, `upsample_coarse_grid` | `Numerical Analysis` / `Visualization` | Pure-NumPy/SciPy Marching Tetrahedra mesh extraction (`Numerical Analysis`) paired with 8 depth-occluded pen-and-ink 3D shading modes (`Visualization`). |
| **MD Trajectory Unwrapping, RMSD, MSD, RDF & Order** | `Trajectory.rmsd`, `Trajectory.msd`, `Trajectory.compute_rdf`, `compute_tetrahedral_order`, `infer_hbonds` | `Numerical Analysis` | Kabsch SVD RMSD, unwrapped MSD (preferring LAMMPS `xu/yu/zu` or `ix/iy/iz`), exact spherical-shell periodic RDF with same-species finite-size normalization, Errington–Debenedetti tetrahedral `q`, and geometric `O–H···O` H-bonds. |
| **Equilibrium Wulff Shapes** | `compute_wulff_shape`, `plot_wulff_shape` | `Numerical Analysis` | Exact 3D convex half-space intersection of Wulff planes `n · x <= γ(hkl)` verified against analytical cubic/octahedral/truncated-octahedral volumes and facet areas. |
| **Birch–Murnaghan Equation of State** | `fit_birch_murnaghan`, `birch_murnaghan_energy`, `birch_murnaghan_pressure`, `plot_eos` | `Numerical Analysis` | 3rd-order isothermal Birch–Murnaghan NLS fit recovering `(E₀, V₀, B₀, B₀′)` with covariance `1σ` errors, RMSE, `R²`, minimum-spanning check, and explicit `energy_basis`. |
| **Convex Hulls, Defect Envelopes, Pourbaix & Volcanos** | `plot_phase_diagram_2d`, `plot_ternary_phase_diagram`, `plot_defect_formation`, `plot_pourbaix`, `plot_volcano` | `Numerical Analysis` / `Visualization` | Exact lower convex hull (2D/3D) and defect lower-envelope charge-transition kinks (`Numerical Analysis`); Pourbaix stability maps and two-leg Sabatier volcano plots (`Visualization`). |
| **First Brillouin Zones & High-Symmetry k-Paths** | `compute_brillouin_zone`, `plot_brillouin_zone` | `Numerical Analysis` / `Approximate Model` | Exact Wigner–Seitz Voronoi polyhedron via LLL-reduced reciprocal half-spaces validated against `(2π)³ / abs(det A_p)` and `V − E + F = 2` (`Numerical Analysis`); built-in metric-tolerance Bravais k-path classification is `Approximate Model` (pass `backend="seekpath"` or `"auto"` for full space-group standardization via SeeK-path). |
| **Kinematical Powder XRD** | `compute_xrd`, `plot_xrd` | `Approximate Model` | Kinematical Cu Kα₁ (`1.54059313 Å`) Bragg intensities with Cromer–Mann LA-3816 neutral form factors (exact H 1s), Lorentz–polarization factor, and pseudo-Voigt broadening. Omits anomalous dispersion (`f′, f″`), absorption, extinction, preferred orientation, and instrumental asymmetry — not a Rietveld engine. |
| **Broadened Spectra, ¹H NMR & XPS Core-Level Fits** | `broaden_peaks`, `plot_spectrum`, `expand_nmr_multiplet`, `plot_nmr`, `compute_shirley_background`, `plot_xps` | `Approximate Model` | Exact height/area-normalized Gaussian, Lorentzian, pseudo-Voigt, and Voigt lineshapes; first-order weak-coupling ¹H NMR multiplet trees (no second-order AB roofing); iterative Shirley inelastic XPS background (no Tougaard/cross-section physics). |
| **Electrochemistry & Battery Cycling** | `plot_cyclic_voltammetry`, `plot_tafel`, `plot_battery_profile`, `plot_dqdv` | `Visualization` | IUPAC/US cyclic voltammograms with `E_pa`, `E_pc`, `E_1/2`, `ΔE_p` callouts, log-linear Tafel fits, galvanostatic plateaus, and smoothed `dQ/dV` differential capacity from supplied data. |
| **Reactions, NEB & 2D Reaction Schemes** | `plot_reaction_profile`, `plot_neb`, `plot_reaction_scheme` | `Visualization` | Multi-step energy landscapes, NEB barrier paths in CODATA units (`eV`, `kJ/mol`, `kcal/mol`), and unified-Å 2D skeletal reaction schemes. |
| **Periodic Table, Informatics & Crystal Graphs** | `plot_periodic_table`, `plot_spacegroup_distribution`, `plot_histogram`, `plot_parity`, `plot_bar_comparison`, `plot_structure_graph`, `plot_graph_network` | `Visualization` | 118-element IUPAC/CIAAW 2024 + BODR + Alvarez/Mantina heatmaps, space-group sunbursts, KDE histograms, parity benchmarks, and periodic GNN quotient multigraphs. |
| **Crystal-Field Splitting Diagrams** | `crystal_field_levels`, `fill_crystal_field_orbitals`, `plot_crystal_field` | `Educational Illustration` | Schematic barycenter-conserving point-charge `d`-orbital levels (`Oh`, `Td`, `D4h`, `D3h`) and idealized weak-/strong-field Hund/Aufbau filling; does not compute ab-initio Racah/ligand-field splittings. |
| **Icons, Stick Scientists, Comics & SciComm Layouts** | `draw_icon`, `draw_element_badge`, `draw_stick_scientist`, `draw_speech_bubble`, `plot_comic_strip`, `Board`, `ScrollCard`, `create_graphical_abstract`, `create_talk_slide`, `create_conference_poster`, `create_mechanism_comic`, `plot_pipeline_diagram` | `Educational Illustration` | Stylized chemistry/materials icons (flasks, Bohr rings, crystals, neural nets), aspect-invariant stick scientists, speech callouts, parchment scrolls, posters, and turnkey talk/TOC templates. |
| **Animated GIFs & Stage A Provenance / Analysis APIs** | `animate_trajectory`, `animate_orbit`, `animate_isosurface`, `ScientificProvenance`, `analyze_band_gap`, `compute_brillouin_zone(..., backend="seekpath")` | `Experimental` | Multi-frame Pillow GIF renderers and newly standardized provenance/band-gap/SeeK-path hooks whose optional keyword arguments or metadata fields may expand before `1.0`. |

## Canonical Internal Units Contract (`xm.core.units`)

All native parsers convert incoming file values into one uniform internal unit system exposed as `xm.core.units.INTERNAL_UNITS` (see [Python API Reference](api.md) and [Supported Formats & Validation Contracts](supported_formats.md)):

| Quantity Key | Canonical Unit | Symbol | Notes & Conversion Helpers |
| :--- | :--- | :--- | :--- |
| `position` | Angstrom | Å | Cartesian atomic coordinates; Bohr/nm inputs converted via CODATA 2022 (`xm.CODATA2022`). |
| `cell` | Angstrom | Å | Row-vector lattice matrix `A` (`3 × 3`); fractional coordinates are dimensionless. |
| `energy` | Electronvolt | eV | Total energies, eigenvalues, barriers, and formation energies; convert via `convert_energy()`. |
| `force` | eV per Angstrom | eV/Å | Cartesian atomic force vectors stored on `AtomsData.forces`. |
| `pressure` | Gigapascal | GPa | Bulk modulus `B₀` and pressure curves (`birch_murnaghan_pressure`); convert via `convert_pressure()`. |
| `time` | Femtosecond | fs | Canonical MD time unit; convert `fs`/`ps`/`ns` via `convert_time()`. LAMMPS `.lammpstrj` stores raw integer step counters in `metadata["timesteps"]` (`times=None` until converted to `fs` via `read_lammpstrj(..., timestep_size=..., time_unit=...)`); extXYZ `read_xyz_trajectory(..., time_unit=...)` converts `time=` to `fs` when `time_unit` is given on call or in frame headers and marks `metadata["time_unit"] = "unspecified"` otherwise; `plot_md_timeseries()` labels the x-axis `"Time (<unit>)"`, `"Time"`, `"Timestep"`, or `"Frame Index"` accordingly. |
| `reciprocal_length` | Inverse Angstrom | Å⁻¹ | Crystallographic `2π` physics convention `B = 2π (A⁻¹)ᵀ` used in Brillouin zones and `k`-paths. |
| `temperature` | Kelvin | K | Instantaneous thermostat temperatures in MD trajectories. |
| `potential` | Volt | V | Electrochemical and battery potentials (`V vs. SHE`, `V vs. Li/Li⁺`, etc.). |

Analytical routines and exporters can attach deeply immutable structured provenance records (`MappingProxyType` mappings, `tuple` sequences, and read-only NumPy arrays) using the frozen dataclass `xm.core.units.ScientificProvenance(algorithm, units, source, parser, parameters, assumptions, package_version)` and serialize them cleanly (including NumPy scalars/arrays, `Path` objects, and nested containers) with `.to_dict()`.

## Explicit Physical Assumptions & Method Boundaries

### 1. MD Periodic Unwrapping, Nyquist Sampling & LAMMPS Image Flags

- **Lossless unwrapped coordinates preferred:** When a LAMMPS `.lammpstrj` trajectory contains explicit unwrapped columns (`xu, yu, zu`, setting `metadata["coords_unwrapped"] = True`) or periodic image flags (`ix, iy, iz`, reconstructed as $\mathbf{r}_{\text{unwrapped}} = \mathbf{r}_{\text{wrapped}} + i_x \mathbf{a} + i_y \mathbf{b} + i_z \mathbf{c}$ and stored in `metadata["unwrapped_positions"]`), `Trajectory.rmsd()` and `Trajectory.msd()` use those unwrapped coordinates directly without heuristic reconstruction.
- **Minimum-image Nyquist assumption:** When trajectories provide only wrapped coordinates without image flags, cumulative frame-to-frame unwrapping applies `minimum_image_vector()` to consecutive displacements. This is exact only while every atom moves less than half the shortest periodic image distance — bounded by the Wigner–Seitz inradius $r_{\mathrm{WS}}$ — between saved frames. If any inferred single-step displacement exceeds $0.45\,r_{\mathrm{WS}}$, `Trajectory` emits a `UserWarning` alerting you to possible periodic wrapping aliasing.
- **Variable-cell trajectories:** Both `Trajectory.rmsd()` and `Trajectory.msd()` expose `variable_cell_mode="cartesian"` (default) and `"fractional"` so NPT/anisotropic-box unwrapping behavior is explicit.

### 2. Periodic RDF Same-Species Estimator

`Trajectory.compute_rdf()` integrates exact spherical-shell volumes $V_{\text{shell}}(r_k) = \frac{4\pi}{3}(r_{\text{out},k}^3 - r_{\text{in},k}^3)$ across all periodic image shells enclosing `r_max` (pass `strict=True` to enforce $r_{\max} \le r_{\mathrm{WS}}$ and guarantee zero periodic self-image correlation).

For distinct species ($\alpha \neq \beta$), the ideal-gas pair density is $N_\alpha N_\beta / V$. For identical species ($\alpha = \beta$ with $N_\alpha > 1$), `same_species_correction=True` (the default) uses the exact self-excluding finite-size pair count $N_\alpha(N_\alpha - 1)$:

$$g_{\alpha\alpha}(r_k) = \left\langle \frac{n_{\text{pairs}}(r_{\text{in},k} \le r < r_{\text{out},k})}{\frac{N_\alpha (N_\alpha - 1)}{V} \cdot \frac{4\pi}{3}\left(r_{\text{out},k}^3 - r_{\text{in},k}^3\right)} \right\rangle_{\text{frames}}$$

### 3. Birch–Murnaghan EOS Uncertainty & Minimum-Spanning Check

`fit_birch_murnaghan()` initializes from an Eulerian finite-strain ordinary least-squares fit and refines with bounded nonlinear least squares (`scipy.optimize.curve_fit`). The returned `EOSFitResult` reports:

- Parameter $1\sigma$ covariance uncertainties (`v0_err`, `e0_err`, `b0_gpa_err`, `bp_err`) whenever the Jacobian covariance matrix is finite and well-conditioned,
- Goodness-of-fit diagnostics (`rmse_ev`, `r_squared`, and backwards-compatible `rms_residual`),
- Minimum-bracketing validation: `spans_minimum` is `True` only when $V_{\min} < V_0 < V_{\max}$. If the fitted minimum lies outside the sampled volume bracket, `fit_status` is set to `"extrapolated"` (or `StructureError` is raised when `allow_extrapolation=False`),
- Explicit `energy_basis` (`"cell"`, `"formula_unit"`, or `"atom"`), mirrored automatically in `plot_eos()` axis labels and summary callouts.

### 4. Electronic Band Gap: Occupation-Based vs. Fermi-Level Semantics

`BandStructure.analyze_band_gap()` returns a `BandGapResult` that makes band-gap semantics explicit instead of silently conflating occupations with $E_F$:

- **`"spin_polarized"`** (`max(occ) ~ 1.0`) and **`"spin_degenerate"`** (`max(occ) ~ 2.0`): determined from stored Kohn–Sham `occupations` with automatic or user-specified `occupation_scale` and `occ_threshold`. Finite-temperature Fermi–Dirac/Gaussian smearing is handled cleanly via `smearing_tol` so lightly smeared insulators are not misclassified as metals.
- **`"fermi_level"`**: used when `occupations is None` or `occupation_scale="fermi_level"`, partitioning bands relative to `fermi_energy` using `fermi_tolerance_ev` (default `1e-4` eV: bands dispersing across $[E_F - \Delta E_{\mathrm{tol}},\, E_F + \Delta E_{\mathrm{tol}}]$ are metallic, and states with $E \le E_F + \Delta E_{\mathrm{tol}}$ are occupied). Passing `occ_threshold` in Fermi-level mode raises `StructureError` so occupation cutoffs and energy tolerances are never conflated.
- **Direct, indirect & indeterminate states:** Reports `fundamental_gap`, `direct_gap` ($\min_k(E_{\mathrm{CBM},k} - E_{\mathrm{VBM},k})$), `indirect_gap` ($\min E_{\mathrm{CBM}} - \max E_{\mathrm{VBM}}$), `vbm`, `cbm`, `is_metallic`, and `is_indeterminate` (returning `fundamental_gap=None` when all bands are filled, all are empty, or occupation data are ambiguous). Backwards-compatible `BandStructure.band_gap()` delegates to `analyze_band_gap()` and returns `0.0` for metallic or indeterminate inputs.

### 5. Bundled `water_to_ice.xyz` MLIP Sample Qualification

The bundled `water_to_ice.xyz` trajectory (`32` H₂O / `96` atoms, `28` frames) demonstrates **direct-coexistence seeded hexagonal Ice I<sub>h</sub> growth** driven by the MACE-MPA-0 foundation potential (`scripts/simulate_water_to_ice_mlip.py`) — a fixed 16-H₂O crystalline Ice I<sub>h</sub> seed templates 16 melted water molecules during unconstrained NVT Langevin growth, 16-stage annealing, cryo quench, and 0 K BFGS minimization. It is **not** spontaneous homogeneous nucleation from bulk supercooled liquid water. Full software, hardware, seed, thermostat, and validation metrics ($q(t)$, H-bond count, and O–O RDF) are documented in [Citation & Scientific Attributions](attributions.md).

### 6. SciComm Stroke Wiggle vs. Archival Quantitative Figures

:::{note}
Intentional xkcd-style path wiggle makes lectures, posters, and science communication engaging while straight reference guides (`sketch=(0, 0, 0)`) preserve honest coordinate alignment. Because subtle hand-drawn path jitter can soften sharp spectral shoulders or fine numerical curvature, pass `sketch_scale=0.0` or use standard straight-line Matplotlib for archival quantitative peak fitting.
:::
