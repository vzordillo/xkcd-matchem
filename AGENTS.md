# Repository Engineering & SciViz Design Guidelines

## 1. Workflow Rule: Audit & Plan First
- Before implementing non-trivial visual, plotting, or architectural changes,
  **audit** the relevant code and rendered figures in `book/_static/gallery/`
  and present a concrete **plan** for review.
- After every visual change, regenerate the affected figures with
  `.venv/bin/python scripts/generate_gallery.py` and **visually inspect** them
  before building docs or concluding.

## 2. The Six `xkcd-matchem` SciViz Design Pillars
1. **4-Tier Stroke Hierarchy (`theme.stroke_weights(preset)`):**
   - **Primary / Frame (`primary`, `frame`, `1.00x–1.06x`)**: Host polyhedra
     (BZ, Wulff, Gibbs triangle, unit cell), primary curves, 2D skeletal bonds.
   - **Secondary Traces & Overlays (`overlay`, `0.70x`)**: Paths *inside* an
     object (BZ **k**-path, ternary tie-lines, volcano scaling legs, EOS/KDE
     fits, Miller plane borders, XRD/IR sticks, secondary COHP curves). Traces
     inside an object must **always** be thinner than the host object edge.
   - **Reference & Hidden Guides (`guide`, `0.52x`, muted alpha)**: Zero lines
     ($E_F=0$, $\omega=0$, $g(r)=1$, $y=x$), VBM/CBM bounds, high-symmetry tick
     dividers, occluded back-edges (dashed).
   - **Annotation Leaders (`leader`, `0.60x`)**: Callout pointers, `a,b,c` /
     `b1,b2,b3` axis triads, comic speech leaders, dimension arrows.
2. **Ink-First Typography (Color Role Separation):**
   - **Data color encodes series identity**; **theme ink (`preset.ink_color`)
     encodes text and annotations** (defect `(q1/q2)` levels, volcano metals,
     peak tags, `HOMO`/`LUMO`/`Gap`, `TS`/`Ea`, summary boxes, edge tags). Never
     dye point labels with light series colors like amber `#E69F00` or sky blue.
   - *Exceptions:* (a) 2D skeletal heteroatom symbols (`O`, `N`, `S`, contrast-
     clamped), (b) compact `a,b,c` / `b1,b2,b3` axis triads, and (c) contrast-
     selected white/ink labels *inside* filled spheres, tiles, or facets.
3. **Outward-Normal Clearance & Un-Sketched Halos:**
   - Offset point/kink labels along **outward normals / normal bisectors** away
     from slopes, vertices, and outer borders; gate inside-facet text (Wulff)
     by projected inscribed width; back dense labels with un-sketched paper
     halos (`theme.place_halo_text` / `set_sketch_params(0, 0, 0)`).
4. **Honest Coordinate Frames & Unified Physical Scale:**
   - Keep strict coordinate dividers (`axvline` at k/q ticks, zero axes, grid
     lines) straight (`sketch=(0, 0, 0)`), while data curves and sketches use
     subtle long-wavelength hand drift (`~100 px`, never rapid sine crinkles).
   - Render multi-molecule reaction schemes (`plot_reaction_scheme`) in one
     unified Angstrom coordinate system (`aspect="equal"`, no per-molecule
     auto-zoomed `inset_axes`).
5. **True 3D Depth & Occlusion:**
   - Foreground lobes/atoms occlude background rims and shading strokes; back
     edges stay thin and dashed; spectral sticks render behind curves.
6. **Authentic `xkcd.com` Stick Scientists:**
   - Compensate horizontal coordinates for axes aspect ratio so heads stay
     circular on any subplot shape, and pair characters (`cueball`, `ponytail`,
     `megan`, `beret`, `goggles`) with open borderless speech + thin ink
     leaders (`kind="line"`).

## 3. Code Quality & Compliance
- Keep package version `0.1.0`; **never** push Git tags or trigger releases.
- Pass `.venv/bin/ruff check . --no-cache` (`<= 80` cols in `.py` & `.ipynb`),
  `.venv/bin/pylint src/xkcd_matchem` (`10.00/10`), `.venv/bin/mypy src/xkcd_matchem`,
  `.venv/bin/pytest --cov=xkcd_matchem --cov-fail-under=90`, and
  `.venv/bin/jupyter-book build book/ -W`.
