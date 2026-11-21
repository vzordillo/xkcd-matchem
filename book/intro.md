# xkcd-matchem: An xkcd-Style Visualization Suite for Chemistry and Materials Science Communication

**`xkcd-matchem`** gives you a much more fun — yet crisp and clean-looking — **xkcd-style** way to look at your molecules, crystals, surfaces, electronic bands, diffraction patterns, reaction barriers, and MD trajectories.

## From a Poster Idea to Everyday SciComm

Like many researchers, I have always loved reading xkcd comics and admired how comic-style fonts and expressive sketch graphics can make science click. While preparing a poster recently, I wanted my molecules and materials plots to feel lively and approachable without looking cluttered. What started as a small helper for that poster grew into `xkcd-matchem`: an open visualization suite for fellow students, researchers, and scientists to explore as part of their **science communication (scicomm)**, group talks, lectures, and posters!

![xkcd-matchem showcase](_static/gallery/hero_showcase.png)

| 3D Coordination Polyhedra Orbit | Climbing-Image NEB Pathway | Live Style & Palette Switching |
| :---: | :---: | :---: |
| ![TiO2 coordination polyhedra orbit](_static/gallery/crystal_orbit.gif) | ![SN2 NEB reaction animation](_static/gallery/neb_reaction.gif) | ![Theme and color scheme switcher](_static/gallery/theme_switcher.gif) |

:::{note}
**Design Philosophy: Clean Presets Out of the Box, Everything Customizable.**
Every plotting function in `xkcd-matchem` works cleanly in one line using balanced white-canvas defaults (`classic` preset, unified `Patrick Hand` typography, borderless structure scenes). Whenever you want a signature look for your poster, class, or talk, every color, element swatch, camera angle, background tint, wiggle strength, and theme preset is open for customization.
:::

:::{warning}
**Playful Exploration & SciComm vs. Journal Archival Plots**
xkcd-style wiggly lines make intuition-building, slides, tutorials, and posters far more engaging. However, intentional stroke jitter can soften tiny peak shoulders or numerical grid details — standard straight-line Matplotlib plots remain the right tool for archival journal figures and quantitative peak fitting.
:::

## How This Book Is Organized

1. **[Getting Started](installation.md)** — Install the package (plus optional ASE/pymatgen/RDKit/cclib/Pillow adapters), browse the visual gallery and animated GIFs, or launch the **offline Interactive 3D Studio**.
2. **Basic Usage (Tutorials 01, 02, 03, 04, 06)** — Step-by-step hands-on notebooks showing clean default presets for molecules, crystals, slabs, polyhedra, band structures, DOS, spectra, and powder XRD.
3. **Advanced Usage (Tutorials 05, 07, 08, 09, 10)** — Customize themes and element palettes, combine Volumetric Cubes with MOs, chart NEB reaction pathways, analyze triclinic MD trajectories & RDFs, fit Birch-Murnaghan EOS curves, compose millimeter posters with opt-in `Board`, and bridge to ASE/pymatgen/RDKit/cclib.
4. **[Reference](reference/supported_formats.md)** — Exact native file format contracts, scientific validations, complete Python API, citation info, and CODATA/Cromer-Mann/SIL OFL attributions.
