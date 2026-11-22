# Citation & Scientific Attributions

## How to Cite `xkcd-matchem`

If `xkcd-matchem` helps you build a poster, talk, tutorial, or science communication figure, please cite it using the metadata in `CITATION.cff`:

> Ordillo, V. (2026). *xkcd-matchem: An xkcd-Style Visualization Suite for Chemistry and Materials Science Communication* (Version 0.1.0) [Computer software]. Nara Institute of Science and Technology. https://github.com/vzordillo/xkcd-matchem

```bibtex
@software{ordillo2026xkcdmatchem,
  author      = {Ordillo, Viejay},
  institution = {Nara Institute of Science and Technology},
  title       = {{xkcd-matchem}: An xkcd-Style Visualization Suite for Chemistry and Materials Science Communication},
  year        = {2026},
  version     = {0.1.0},
  url         = {https://github.com/vzordillo/xkcd-matchem}
}
```

## Acknowledgements

This repository was developed semi-autonomously with **Google DeepMind's Gemini** frontier models inside **[Google Antigravity](https://antigravity.google)**, supported through Google's **Antigravity for Science** academic researcher access program (providing six months of complimentary top-tier Google Antigravity and frontier Gemini access to scientists and academic researchers for non-commercial research). Many thanks to the **Google Antigravity**, **Google Research**, and **Google DeepMind** teams for supporting open scientific exploration and science communication tooling.

## Scientific Data Sources & Attributions

Every physical constant, atomic weight, atomic radius, X-ray scattering form factor, and color reference table shipped in `xkcd-matchem` is cross-checked against its primary publication or standards database:

### 1. Fundamental Physical Constants (CODATA 2022)

- **Citation**: Mohr, P. J.; Newell, D. B.; Taylor, B. N.; Tiesinga, E. (2025). "CODATA recommended values of the fundamental physical constants: 2022". *Reviews of Modern Physics*, 97(2), 025002. DOI: [10.1103/RevModPhys.97.025002](https://doi.org/10.1103/RevModPhys.97.025002). Primary online table: NIST Standard Reference Database 121 ([physics.nist.gov/constants](https://physics.nist.gov/constants)).
- **Usage (`xm.CODATA2022`)**: Bohr-to-Angstrom (`0.529177210544` Å), Hartree-to-eV (`27.211386245981` eV), Rydberg-to-eV (`13.6056931229905` eV), exact SI defining constants ($e = 1.602176634 \times 10^{-19}$ C, $N_A = 6.02214076 \times 10^{23}\text{ mol}^{-1}$) yielding eV-to-kJ/mol (`96.48533212331002`), thermochemical eV-to-kcal/mol (`23.06054783061903`), and eV/Å³-to-GPa (`160.2176634`), spectroscopic conversions (`8065.543937349212` $\text{cm}^{-1}/\text{eV}$, `219474.63136314` $\text{cm}^{-1}/E_h$, `33.3564095198152` $\text{cm}^{-1}/\text{THz}$, `4.135667696923859` $\text{meV}/\text{THz}$), Boltzmann constant (`8.617333262145179e-5` eV/K), and the CODATA 2022 Table XXXIV Cu Kα₁ X-ray wavelength (`1.54059313` Å, from $1537.400\text{ xu} \times 1.00207697 \times 10^{-3}\text{ Å}$).

### 2. Standard Atomic Weights

- **Citation**: Prohaska, T. et al. (2022). "Standard atomic weights of the elements 2021 (IUPAC Technical Report)". *Pure and Applied Chemistry*, 94(5), 573-600. DOI: [10.1515/pac-2019-0603](https://doi.org/10.1515/pac-2019-0603).
- **CIAAW 2024 Updates**: Revised standard atomic weights for Gd (`157.249`), Lu (`174.96669`), and Zr (`91.222`) incorporated from [ciaaw.org/atomic-weights.htm](https://ciaaw.org/atomic-weights.htm), alongside Table 1 abridged weights for the 14 interval elements and Table 2 / PubChem longest-lived nuclide masses for radioactive elements (used for center-of-mass calculations, LAMMPS mass-to-element inference, and periodic-table mass heatmaps).

### 3. Covalent Radii & Element Classifications (BODR)

- **Citation**: Guha, R.; Howard, M. T.; Hutchison, G. R.; Murray-Rust, P.; Rzepa, H.; Steinbeck, C.; Wegner, J.; Willighagen, E. L. (2006). "The Blue Obelisk—Interoperability in Chemical Informatics". *Journal of Chemical Information and Modeling*, 46(3), 991-998. DOI: [10.1021/ci050400b](https://doi.org/10.1021/ci050400b).
- **License & Usage**: CC0 1.0 Universal (`elements.xml`); provides covalent radii (used for automatic bond perception and 3D ball-and-stick sphere scaling) and element family classifications.

### 4. PubChem Periodic Table Cross-Checks

- **Citation**: Kim, S.; Gindulyte, A.; Zhang, J.; Thiessen, P. A.; Bolton, E. E. (2021). "PubChem Periodic Table and Element pages: improving access to information on chemical elements from authoritative sources". *Chemistry Teacher International*, 3(1), 57-65. DOI: [10.1515/cti-2020-0006](https://doi.org/10.1515/cti-2020-0006).

### 5. van der Waals Radii

Used for 3D spacefill sphere radii and periodic-table size heatmaps:

- **Alvarez (2013)**: Alvarez, S. (2013). "A cartography of the van der Waals territories". *Dalton Transactions*, 42(24), 8617-8636. DOI: [10.1039/c3dt50599e](https://doi.org/10.1039/c3dt50599e) (Table 1, Supplementary Table S1, and Section 6 heuristic $r_{\mathrm{vdW}} = r_{\mathrm{cov}} + 0.90\text{ Å}$ for unmeasured heavy and superheavy elements).
- **Mantina et al. (2009)**: Mantina, M.; Chamberlin, A. C.; Valero, R.; Cramer, C. J.; Truhlar, D. G. (2009). "Consistent van der Waals Radii for the Whole Main Group". *The Journal of Physical Chemistry A*, 113(19), 5806-5812. DOI: [10.1021/jp8111556](https://doi.org/10.1021/jp8111556) (Table 12 main-group values for Po, At, Rn, Fr, and Ra).

### 6. Atomic X-Ray Scattering Form Factors

Used by `xm.compute_xrd()` and `xm.plot_xrd()` to weight Bragg peak intensities accurately by atomic number and scattering angle:

- **Citation**: Cromer, D. T.; Mann, J. B. (1967, distributed March 11, 1968). "X-Ray Scattering Factors Computed from Numerical Hartree-Fock Wave Functions". Los Alamos Scientific Laboratory Report LA-3816. DOI: [10.2172/4555098](https://doi.org/10.2172/4555098).
- **Scope (`xm.CROMER_MANN_TABLE`)**: 4-Gaussian + $c$ analytical fits from Table II (report pp. 66–69) for all 102 neutral atoms from He ($Z=2$) through Lr/Lw ($Z=103$), parameterized in $s = \sin\theta/\lambda$ (Å⁻¹). Neutral H ($Z=1$) uses the exact analytical spherical $1s$ scattering factor $f_{\mathrm{H}}(s) = (1 + (2\pi a_0 s)^2)^{-2}$.

### 7. Color Palettes, Bundled Fonts & License Separation

`xkcd-matchem` separates software, font, and documentation licenses cleanly so academic and commercial teams can adopt the library with zero ambiguity:

| Component | Path / Activation | License | Commercial Use |
| :--- | :--- | :--- | :--- |
| **Source code, tests & scripts** | `src/`, `tests/`, `scripts/` | Apache-2.0 (`LICENSE`, `NOTICE`) | Permitted |
| **Default hand-drawn fonts** | `Patrick Hand` (default), `Comic Neue` (`src/xkcd_matchem/data/fonts/`) | SIL OFL 1.1 (`PatrickHand-OFL.txt`, `ComicNeue-OFL.txt`) | Permitted |
| **Optional non-commercial font** | `xkcd Script` (`xm.use_xkcd_script()` opt-in; blocked when `XM_COMMERCIAL_SAFE=1`) | CC BY-NC 3.0 (`xkcd-script-LICENSE.txt`) | Non-commercial only |
| **Documentation & tutorials** | `book/` prose and tutorial notebooks | CC BY 4.0 (`book/LICENSE-CC-BY-4.0.txt`) | Permitted with attribution (note: any figures explicitly rendered with `xkcd Script` carry CC BY-NC 3.0 font terms) |

- **Okabe & Ito CUD Palette**: Okabe, M.; Ito, K. (2002; mod. 2008). *Color Universal Design (CUD)* ([https://jfly.uni-koeln.de/color/](https://jfly.uni-koeln.de/color/)) — colorblind-accessible 8-color default series palette (`xm.OKABE_ITO`).
- **Jmol Element Colors**: Jmol Development Team; Herráez, A.; Vervelle, N. *Jmol Colors* ([https://jmol.sourceforge.net/jscolors/](https://jmol.sourceforge.net/jscolors/), fallback `#FF1493` for $Z=110\text{–}118$).
- **Patrick Hand (default)**: Patrick Wagesreiter, SIL Open Font License 1.1 (`PatrickHand-OFL.txt`; commercial-safe default font across all built-in presets out of the box).
- **Comic Neue**: Craig Rozynski & Hrant Papazian, SIL Open Font License 1.1 (`ComicNeue-OFL.txt`; bundled commercial-safe alternate via `xm.set_default_font("Comic Neue")`).
- **xkcd Script (optional non-commercial opt-in)**: Derived from Randall Munroe's handwriting via [`ipython/xkcd-font`](https://github.com/ipython/xkcd-font), Creative Commons Attribution-NonCommercial 3.0 Unported (CC BY-NC 3.0; non-commercial/academic use only; see `src/xkcd_matchem/data/fonts/xkcd-script-LICENSE.txt`). Activate explicitly with `xm.use_xkcd_script()` (emits a one-time `UserWarning`, or raises `StyleError` when `XM_COMMERCIAL_SAFE=1`).

### 8. Bundled MLIP MD Sample Reproducibility Provenance

- **MACE-MPA-0 Foundation Potential**: Batatia, I. et al. (2023/2024). "A foundation model for atomistic materials chemistry" ([arXiv:2401.00096](https://arxiv.org/abs/2401.00096)); `medium-mpa-0` weights (`macempa0mediummodel`, trained on MPtrj + sAlexandria PBE).
- **Tetrahedral Order Parameter (`xm.compute_tetrahedral_order`)**: Errington, J. R.; Debenedetti, P. G. (2001). "Relationship between structural order and the anomalies of liquid water". *Nature*, 409, 318-321. DOI: [10.1038/35053024](https://doi.org/10.1038/35053024).

#### `water_to_ice.xyz` Provenance & Physical Qualification

The bundled `water_to_ice.xyz` trajectory was generated locally via `scripts/simulate_water_to_ice_mlip.py` and represents **direct-coexistence seeded hexagonal Ice I<sub>h</sub> growth** — a fixed 16-H₂O crystalline Ice I<sub>h</sub> seed templating 16 melted liquid H₂O molecules under unconstrained NVT Langevin dynamics — **not** spontaneous homogeneous nucleation from bulk supercooled liquid water.

| Parameter | Value / Artifact |
| :--- | :--- |
| **Model & Precision** | MACE-MPA-0 (`medium-mpa-0`, checkpoint `macempa0mediummodel`, `float32`) |
| **Software Stack** | `ase 3.29.0`, `mace-torch 0.3.16`, `PyTorch 2.12.1+cu130` |
| **Hardware** | NVIDIA RTX PRO 6000 Blackwell GPU |
| **Periodic Cell & System** | Orthorhombic `9.036 × 7.825 × 14.712 Å` (`2 × 1 × 2` Ice I<sub>h</sub> cell, `a_hex = 4.518 Å`, `c_hex = 7.356 Å`), `32` H₂O / `96` atoms, `pbc=(True, True, True)` |
| **RNG Seeds** | `seed = 42` (Bernal–Fowler proton ordering); `seed + 1 = 43` (mobile-layer scrambling & Langevin thermostat) |
| **Seed & Melt Preparation** | Initial BFGS relaxation (`fmax = 0.03 eV/Å`, 50 steps); fix bottom 16 H₂O seed; scramble & melt top 16 H₂O at `550 K` for `1,500` NVT Langevin steps (`dt = 0.5 fs`, friction `0.04 fs⁻¹`) |
| **Unconstrained Growth MD** | All constraints released; NVT Langevin `dt = 0.75 fs`, `330 K` for `12,000` steps (`9.0 ps`, friction `0.015 fs⁻¹`) |
| **Annealing & Cryo Quench** | 16-stage linear ramp `330 → 5 K` over `8,000` steps (`6.0 ps`, friction `0.025 fs⁻¹`), followed by `1 K` cryo quench for `1,500` steps (`1.125 ps`, friction `0.05 fs⁻¹`) |
| **Final 0 K Minimization** | 0 K BFGS relaxation to `fmax = 0.02 eV/Å` (max 200 steps); `28` evenly sampled frames saved spanning `0.000 → 16.425 ps` |
| **Validated Structural Metrics** | Tetrahedral order `q(t)`: `0.774 → 0.978`; geometric `O–H···O` H-bonds (`infer_hbonds`): `52 → 63` (ideal Ice I<sub>h</sub>: `64`); exothermic `ΔE = −7.87 eV`; first O–O RDF peak sharpens from `r ≈ 2.77 Å` (`g ≈ 9.7`) to `r ≈ 2.72 Å` (`g ≈ 16.5`) |
