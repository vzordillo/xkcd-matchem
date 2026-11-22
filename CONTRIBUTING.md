# Contributing to `xkcd-matchem`

Thank you for your interest in contributing to **`xkcd-matchem`** — *An
xkcd-Style Visualization Suite for Chemistry and Materials Science
Communication*! Whether you are fixing a parser edge case, adding a new
spectroscopy or materials plotter, improving a Jupyter Book tutorial, or
reporting a visual bug, your contributions are warmly welcome.

---

## Quick Start: Local Development Setup

`xkcd-matchem` targets **Python 3.10+** and keeps its core runtime light
(`numpy`, `scipy`, and `matplotlib` only). To set up a full contributor
environment with linting, tests, animation export, and Jupyter Book docs:

```bash
git clone https://github.com/vzordillo/xkcd-matchem.git
cd xkcd-matchem
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[animation,dev,docs]"
```

Verify your environment before making changes:

```bash
.venv/bin/pytest tests/
.venv/bin/xkcd-matchem --version
```

---

## Repository Map

| Directory | Purpose |
| :--- | :--- |
| `src/xkcd_matchem/core/` | Scientific data models, geometry, neighbors, bonding, XRD, isosurfaces, Brillouin zones, Wulff shapes (zero Matplotlib imports) |
| `src/xkcd_matchem/data/` | Verified scientific tables (CODATA 2022, IUPAC/CIAAW, BODR, Alvarez/Mantina, Cromer–Mann), 62 ITA space groups, 17 bundled samples, fonts, and `viewer.html` |
| `src/xkcd_matchem/io/` | Format auto-detection, 29 zero-dependency native parsers, and optional lazy ecosystem adapters (ASE, pymatgen, RDKit, cclib, MDAnalysis, MDTraj) |
| `src/xkcd_matchem/style/` | Presets, 4-tier stroke hierarchy, colorblind-safe Okabe–Ito & Jmol palettes, bundled font handling, background grids, and hatched fills |
| `src/xkcd_matchem/render/` | Low-level 2D/3D drawing primitives, pen-and-ink 3D shading, icons, callouts, and offline Interactive 3D Studio HTML exporter |
| `src/xkcd_matchem/plots/` | High-level one-line domain plotters (structures, bands/DOS/COHP/phonons, crystal field, electrochemistry, spectroscopy/NMR/XPS, reactions, MD, thermodynamics, informatics, graphs) |
| `src/xkcd_matchem/layout/` | Opt-in millimeter poster boards (`Board`/`Card`/`ScrollCard`), narrative `ComicStrip` tools, and turnkey SciComm templates |
| `book/` | Executable Jupyter Book documentation, gallery, playground, reference pages, and step-by-step tutorials |
| `scripts/generate_gallery.py` | Reproducible generator for every gallery PNG/GIF and the embedded 3D Studio HTML |
| `tests/` | Unit, numerical-invariant, visual smoke, and Google-style AST compliance tests |

---

## Workflow: Audit & Plan First

Before implementing non-trivial visual, plotting, or architectural changes:

1. **Audit first:** Inspect the relevant modules and existing figures in
   `book/_static/gallery/` so new work integrates seamlessly with current APIs
   and visual conventions.
2. **Design intentionally:** For plotting or rendering updates, follow the six
   `xkcd-matchem` SciViz Design Pillars below.
3. **Visually inspect:** After any visual change, regenerate affected gallery
   assets with `.venv/bin/python scripts/generate_gallery.py` and visually
   inspect the rendered images before opening a pull request.

---

## The Six `xkcd-matchem` SciViz Design Pillars

Every plotting and rendering contribution should uphold our core visual design
standards (documented in `AGENTS.md`):

1. **4-Tier Stroke Hierarchy (`xm.stroke_weights(preset)`):**
   - `primary` / `frame` (`1.00x–1.06x`): Host polyhedra (Brillouin zones,
     Wulff shapes, Gibbs triangles, unit cells), primary curves, 2D skeletal
     bonds.
   - `overlay` (`0.70x`): Traces *inside* a host object (BZ **k**-paths,
     ternary tie-lines, volcano legs, EOS/Tafel/KDE fits, Miller borders,
     XRD/IR/NMR sticks, secondary curves). Internal traces must always stay
     thinner than the host frame.
   - `guide` (`0.52x`, muted alpha): Straight reference lines ($E_F = 0$,
     $\omega = 0$, $g(r) = 1$, barycenters, $y = x$, band-edge bounds,
     high-symmetry tick dividers, occluded dashed back-edges).
   - `leader` (`0.60x`): Callout pointers, `a, b, c` / `b1, b2, b3` axis
     triads, open comic speech leaders, and dimension arrows.
2. **Ink-First Typography (Color Role Separation):**
   - Use **data color** for series identity and **theme ink**
     (`preset.ink_color`) for text labels and annotations so light series
     colors (like amber `#E69F00` or sky blue) never wash out labels.
   - Exceptions: contrast-clamped 2D skeletal heteroatom symbols (`O`, `N`, `S`),
     compact crystallographic axis triads, and contrast-selected labels placed
     *inside* filled spheres, tiles, or facets.
3. **Outward-Normal Clearance & Un-Sketched Halos:**
   - Offset point and kink labels along outward normals / normal bisectors away
     from curves, vertices, and panel borders, and back dense labels with
     un-sketched paper halos (`theme.place_halo_text` /
     `set_sketch_params(0, 0, 0)`).
4. **Honest Coordinate Frames & Unified Physical Scale:**
   - Keep strict reference guides (`axvline` at **k**/**q** ticks, zero lines,
     grids) straight (`sketch=(0, 0, 0)`), while data curves use subtle
     long-wavelength hand drift (`~100 px`, avoiding rapid crinkles). Render
     multi-molecule reaction schemes in a single unified Angstrom coordinate
     system (`aspect="equal"`).
5. **True 3D Depth & Occlusion:**
   - Foreground atoms and orbital lobes occlude background rims and shading
     strokes; back edges stay thin and dashed; spectral sticks render behind
     broadened envelopes.
6. **Authentic `xkcd.com` Stick Scientists:**
   - Compensate horizontal geometry for axes aspect ratio so character heads
     remain circular on any subplot shape, paired with borderless speech and
     thin ink leaders (`kind="line"`).

---

## Code Style & Scientific Rigor Standards

- **Zero hallucinated constants or citations:** Every physical constant,
  unit conversion, analytical formula, and bibliographic citation must be
  verified against its primary peer-reviewed publication or standards database
  (e.g., CODATA 2022, IUPAC/CIAAW, BODR, Alvarez 2013, Cromer–Mann LA-3816) and
  documented in `book/reference/attributions.md` when adding new reference data.
- **Strict source conventions** (automatically checked by
  `tests/test_style_compliance.py`):
  - Include the standard Apache-2.0 copyright header within the first 400
    characters of every `.py` file under `src/xkcd_matchem/`.
  - Keep all lines `<= 80` characters in both `.py` files and `.ipynb` code
    cells.
  - Follow Google Python Style Guide Section 2.2 module-level imports (no
    relative imports; only whitelisted `from ... import ...` helpers).
  - Do not use bare `assert` or `@staticmethod` in `src/xkcd_matchem/`.
  - Keep functions concise (`<= 28` locals, `<= 18` branches, `<= 70`
    statements) and modules `< 2000` lines so `pylint` stays at `10.00/10`.
  - Keep `src/xkcd_matchem/core/` free of Matplotlib imports so numerical logic
    remains fast, testable, and headless.
- **Versioning:** Do not modify `src/xkcd_matchem/_version.py` or create Git
  tags in feature pull requests; release tagging is handled separately by
  maintainers.

---

## Required Quality Checks Before Opening a PR

Run the full local quality gate (matching `.github/workflows/ci.yml` and
`.github/workflows/docs.yml`) before submitting your pull request:

```bash
.venv/bin/ruff check . --no-cache
.venv/bin/pylint src/xkcd_matchem
.venv/bin/mypy src/xkcd_matchem
.venv/bin/pytest --cov=xkcd_matchem --cov-fail-under=90
.venv/bin/jupyter-book build book/ -W
```

All five checks must pass cleanly:

- **`ruff`**: zero lint, docstring, import-order, or 80-column violations.
- **`pylint`**: `10.00/10` across `src/xkcd_matchem`.
- **`mypy`**: zero type errors targeting Python 3.10+.
- **`pytest`**: 100% passing tests with `>= 90%` package line coverage (close
  every Matplotlib figure in tests with `plt.close(fig)`).
- **`jupyter-book build book/ -W`**: zero Sphinx or notebook execution
  warnings.

---

## Pull Request Process

1. Fork the repository and create a descriptive feature branch from `main`
   (e.g., `feat/raman-depolarization` or `fix/cif-origin-choice`).
2. Add focused unit tests in `tests/` validating mathematical invariants,
   edge cases, and visual smoke rendering.
3. Update relevant documentation (`book/reference/api.md`,
   `book/reference/supported_formats.md`, `book/gallery.md`, or tutorial
   notebooks) when adding public APIs or formats.
4. Fill out `.github/PULL_REQUEST_TEMPLATE.md` and attach screenshots for any
   visual or Interactive 3D Studio changes.

---

## Licensing & Attribution

By contributing to `xkcd-matchem`, you agree that your contributions will be
licensed under the repository's dual-component open licensing model:

- **Code, tests, and scripts (`src/`, `tests/`, `scripts/`):**
  [Apache License, Version 2.0](LICENSE) (see also [`NOTICE`](NOTICE)).
- **Documentation, tutorials, and prose (`book/`):**
  [Creative Commons Attribution 4.0 International (CC BY 4.0)](book/LICENSE-CC-BY-4.0.txt).
- **Fonts:** Do not commit proprietary or unlicensed fonts. Bundled fonts are
  documented in [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md)
  (`xkcd Script` under CC BY-NC 3.0 as the default academic/non-commercial font,
  plus commercial-safe `Patrick Hand` and `Comic Neue` under SIL OFL 1.1).

Thank you for helping make chemistry and materials science communication more
accessible, rigorous, and fun!
