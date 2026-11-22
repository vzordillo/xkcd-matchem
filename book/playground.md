# Interactive 3D xkcd Studio

Explore molecules, crystals, slabs, coordination polyhedra, multi-frame trajectories, and 3D orbital isosurfaces directly in your browser — **completely offline, with no Python server or Binder wait time**.

The studio opens right away on a cross-hatched **Water HOMO 3D Orbital** (`water_homo.cube`) and includes a **9-Scene Quick Scene Switcher** along the top bar plus a 1-click **8-Mode Ink Shading** pill bar whenever an orbital isosurface is loaded.

## Try This in 30 Seconds

1. **Switch curated scenes in one click:** Use the **Quick Scene Switcher** pills across the top (synced with the scene dropdown) to jump immediately between 3D orbitals, TiO₂ coordination polyhedra, an animated 9-frame S<sub>N</sub>2 NEB reaction path, aspirin, layered 2H-MoS₂, silicon, NaCl, benzene, and alanine dipeptide.
2. **Orbit, zoom & snap to crystal axes:** Drag on the canvas to rotate (`Shift + Drag` to roll, scroll wheel to zoom), or click **Iso**, **Persp**, **Front**, **Side**, **Top**, or crystallographic **`a`, `b`, `c`, `a*`, `b*`, `c*`** — press **`R`** to reset the camera and **`Space`** to play/pause trajectories.
3. **Explore pen-and-ink shading & styles:** On orbital scenes, click any of the 8 **Ink Shading** pills (`solid`, `hatch`, `crosshatch`, `contour`, `cross_contour`, `stipple`, `scumble`, `outline`). Switch rendering styles (`ball_and_stick`, `spacefill`, `stick`, `polyhedra`), toggle **Bonds**, **Boundary**, **Center**, **Legend**, **Vectors**, or **Isosurface**, tile up to `3x3x3` supercells (`na`, `nb`, `nc`), recolor any element with the swatch pickers, swap **White** / **Paper** / **Chalk** backgrounds, or tune the **xkcd Wiggle** slider.
4. **Responsive layout & large-system LOD:** On narrow embeds or mobile screens (`<= 780px`), the controls panel collapses cleanly into a toggleable drawer (`Controls`) so the 3D canvas stays unobstructed. For larger supercells (`500–5,000+` atoms), automatic interactive drag Level-of-Detail (LOD) keeps orbiting smooth and snaps immediately to full hand-drawn pen-and-ink quality on pointer release (or toggle **Always full detail**).
5. **Load your own structure:** Click **Open Structure File** to preview a local `.xyz`, `POSCAR`/`.vasp`, `.cif`, `.sdf`/`.mol`, or `.pdb` file right in the browser (files stay completely local on your machine).
6. **1:1 Python snippet parity & export:** Click **Copy Python** to get runnable `xkcd_matchem` code matching your active camera, theme preset, scales, H-bonds, labels, frame index, and 3D isosurface settings (`xm.generate_python_snippet()`), or click **Download PNG** / **Download SVG** immediately.

## Curated 9-Scene Quick Switcher Lineup

Built automatically via `xm.render.interactive.build_playground_scenes()` (and embedded whenever `export_interactive_html(..., include_bundled_samples=True)` is called):

| # | Scene | Category | Highlights to Try |
| :---: | :--- | :--- | :--- |
| 1 | **Water HOMO 3D Orbital** (`water_homo.cube`) | 3D Orbitals | Default scene; switch across all 8 pen-and-ink **Ink Shading** modes |
| 2 | **TiO2 (110) Octahedra** (`tio2_slab.poscar`) | Crystals & Slabs | Depth-sorted TiO₆ coordination polyhedra (`polyhedra` style) |
| 3 | **SN2 Reaction Path** (`sn2_neb.xyz`) | Reactions & MD | Press **`Space`** or **Play** to scrub all 9 climbing-image NEB frames |
| 4 | **Aspirin Molecule** (`aspirin.xyz`) | Molecules | Try `spacefill` van der Waals spheres or custom atom swatch colors |
| 5 | **2H-MoS2 Monolayer** (`mos2_2h.poscar`) | Crystals & Slabs | Tile a `2x2x1` supercell and snap to **`c`** (top) vs. **`a`** (side) |
| 6 | **Silicon Diamond Cell** (`si_diamond.cif`) | Crystals & Slabs | Toggle **Boundary** and crystallographic **`a`, `b`, `c`** axes |
| 7 | **Rock-Salt NaCl** (`nacl.cif`) | Crystals & Slabs | Toggle **Bonds** off for a pure bond-free ionic lattice view |
| 8 | **Benzene Ring** (`benzene.sdf`) | Molecules | Inspect explicit alternating aromatic bond orders |
| 9 | **Alanine Dipeptide** (`alaninedipeptide.pdb`) | Biomolecules | Orbit a blocked peptide backbone parsed from PDB `CONECT` records |

<div style="margin: 1rem 0;">
  <p>
    <a href="_static/interactive_studio.html" target="_blank" rel="noopener"
       style="display:inline-block;padding:8px 14px;border-radius:8px;background:#1d4ed8;color:#fff;text-decoration:none;font-weight:700;">
      Open Full-Screen Interactive Studio
    </a>
  </p>
  <iframe
    src="_static/interactive_studio.html"
    title="xkcd-matchem Interactive 3D Studio"
    style="width:100%;height:760px;border:1px solid #d5dae4;border-radius:12px;background:#ffffff;"
    loading="lazy">
  </iframe>
</div>

## Launching from Python or the Terminal

You can open the exact same offline studio inside a Jupyter notebook, save it as a shareable single-file `.html`, or launch it in your default browser — including multi-frame trajectories (with **Play** scrubbing at `4 / 8 / 16 fps`) and Gaussian `.cube` 3D orbital isosurfaces:

```python
import xkcd_matchem as xm

# Render inline in Jupyter via _repr_html_()
view = xm.view_structure(
    "tio2_slab.poscar",
    style="polyhedra",
    polyhedra=("Ti", "O"),
    supercell=(2, 2, 1),
    palette={"Ti": "#4E79A7", "O": "#E15759"},
)
view

# Explore a 3D orbital CUBE or scrub a multi-frame NEB trajectory
xm.view_structure("water_homo.cube", iso_shade="crosshatch")
xm.view_structure(xm.load_trajectory("sn2_neb.xyz"))

# Or open directly in your system web browser
xm.view_structure("aspirin.xyz", backend="browser")
```

From the command line:

```bash
xkcd-matchem view tio2_slab.poscar --supercell 2 2 1 --view perspective --colors Ti=#E07A5F O=#3D405B --backend browser
```

:::{note}
Browser-side `.cif` drag-and-drop reads the explicit unit cell and fractional sites directly for fast previews. For full symmetry-operation expansion (`_symmetry_equiv_pos_as_xyz` or implicit ITA space-group lookup), load your `.cif` in Python via `xm.view_structure("crystal.cif")` or `xkcd-matchem view crystal.cif`.
:::
