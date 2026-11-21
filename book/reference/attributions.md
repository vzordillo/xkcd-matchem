# Scientific Sources & Attributions

Every physical constant, atomic property, X-ray form factor, and color scheme in `xkcd-matchem` was cross-referenced directly against its verified primary publication or official dataset.

## 1. Fundamental Physical Constants (CODATA 2022)

- **Citation**: Mohr, P. J.; Newell, D. B.; Taylor, B. N.; Tiesinga, E. (2025). "CODATA recommended values of the fundamental physical constants: 2022". *Reviews of Modern Physics*, 97(2), 025002. DOI: [10.1103/RevModPhys.97.025002](https://doi.org/10.1103/RevModPhys.97.025002).
- **Usage**: Bohr-to-Angstrom (`0.529177210544`), Hartree-to-eV (`27.211386245981`), exact SI defining constants ($e$, $N_A$), inverse-cm conversions, Boltzmann constant, and CODATA Table XXXIV $\text{Cu K}\alpha_1$ wavelength derivation.

## 2. Standard Atomic Weights

- **Citation**: Prohaska, T. et al. (2022). "Standard atomic weights of the elements 2021 (IUPAC Technical Report)". *Pure and Applied Chemistry*, 94(5), 573-600. DOI: [10.1515/pac-2019-0603](https://doi.org/10.1515/pac-2019-0603).
- **CIAAW 2024 Updates**: Gd (`157.249`), Lu (`174.96669`), and Zr (`91.222`) incorporated from [ciaaw.org/atomic-weights.htm](https://ciaaw.org/atomic-weights.htm), with Table 1 abridged standard atomic weights for the 14 interval elements and Table 2 / PubChem longest-lived nuclide masses for radioactive elements.

## 3. Covalent Radii & Element Classifications (BODR)

- **Citation**: Guha, R.; Howard, M. T.; Hutchison, G. R.; Murray-Rust, P.; Rzepa, H.; Steinbeck, C.; Wegner, J.; Willighagen, E. L. (2006). "The Blue Obelisk—Interoperability in Chemical Informatics". *Journal of Chemical Information and Modeling*, 46(3), 991-998. DOI: [10.1021/ci050400b](https://doi.org/10.1021/ci050400b).
- **License**: CC0 1.0 Universal (`elements.xml`).

## 4. PubChem Periodic Table Cross-Checks

- **Citation**: Kim, S.; Gindulyte, A.; Zhang, J.; Thiessen, P. A.; Bolton, E. E. (2021). "PubChem Periodic Table and Element pages: improving access to information on chemical elements from authoritative sources". *Chemistry Teacher International*, 3(1), 57-65. DOI: [10.1515/cti-2020-0006](https://doi.org/10.1515/cti-2020-0006).

## 5. van der Waals Radii

- **Alvarez (2013)**: Alvarez, S. (2013). "A cartography of the van der Waals territories". *Dalton Transactions*, 42(24), 8617-8636. DOI: [10.1039/c3dt50599e](https://doi.org/10.1039/c3dt50599e) (Table 1, Supplementary Table S1, and Section 6 heuristic $r_{\mathrm{vdW}} = r_{\mathrm{cov}} + 0.90\text{ \AA}$ for unmeasured heavy/superheavy elements).
- **Mantina et al. (2009)**: Mantina, M.; Chamberlin, A. C.; Valero, R.; Cramer, C. J.; Truhlar, D. G. (2009). "Consistent van der Waals Radii for the Whole Main Group". *The Journal of Physical Chemistry A*, 113(19), 5806-5812. DOI: [10.1021/jp8111556](https://doi.org/10.1021/jp8111556) (Table 12 values for Po, At, Rn, Fr, and Ra).

## 6. Atomic X-Ray Scattering Form Factors

- **Citation**: Cromer, D. T.; Mann, J. B. (1967, distributed March 11, 1968). "X-Ray Scattering Factors Computed from Numerical Hartree-Fock Wave Functions". Los Alamos Scientific Laboratory Report LA-3816. DOI: [10.2172/4555098](https://doi.org/10.2172/4555098).
- **Scope**: 4-Gaussian + $c$ analytical fits from Table II (report pp. 66-69) for neutral atoms He ($Z=2$) through Lr/Lw ($Z=103$). Neutral H ($Z=1$) uses the exact analytical spherical 1s scattering factor $f_{\mathrm{H}}(s) = (1 + (2\pi a_0 s)^2)^{-2}$.

## 7. Color Palettes & Bundled Fonts

- **Okabe & Ito CUD Palette**: Okabe, M.; Ito, K. (2002; mod. 2008). *Color Universal Design (CUD)* ([https://jfly.uni-koeln.de/color/](https://jfly.uni-koeln.de/color/)).
- **Jmol Element Colors**: Jmol Development Team; Herráez, A.; Vervelle, N. *Jmol Colors* ([https://jmol.sourceforge.net/jscolors/](https://jmol.sourceforge.net/jscolors/), fallback `#FF1493` for $Z=110\text{–}118$).
- **Patrick Hand**: Patrick Wagesreiter, SIL Open Font License 1.1.
- **Comic Neue**: Craig Rozynski & Hrant Papazian, SIL Open Font License 1.1.
