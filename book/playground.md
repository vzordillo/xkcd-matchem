# Interactive 3D xkcd Studio

Explore molecules, crystals, slabs, coordination polyhedra, multi-frame trajectories, and 3D orbital isosurfaces directly in your browser — **completely offline, with no Python server or Binder wait time**.

## Try This in 30 Seconds

1. **Orbit & zoom:** Drag on the canvas to rotate (`Shift + Drag` to roll, scroll wheel to zoom).
2. **Snap to crystal axes:** Click **Iso**, **Persp**, **Front**, **Side**, **Top**, or crystallographic **`a`, `b`, `c`, `a*`, `b*`, `c*`** along the top bar.
3. **Make it yours:** Switch rendering styles (`ball_and_stick`, `spacefill`, `stick`, `polyhedra`), toggle **Bonds**, **Boundary**, **Center**, **Legend**, **Vectors**, or **Isosurface** (with all 8 **Iso Shading** pen-and-ink styles), tile up to `3x3x3` supercells (`na`, `nb`, `nc`), recolor any element with the swatch pickers, swap **White** / **Paper** / **Chalk** backgrounds, or adjust the **xkcd Wiggle** slider.
4. **Load your own structure:** Click **Open Structure File** to preview a local `.xyz`, `POSCAR`/`.vasp`, `.cif`, `.sdf`/`.mol`, or `.pdb` file right in the browser (files stay completely local on your machine).
5. **Take it back to Python:** Click **Copy Python** for the exact `xkcd_matchem` script matching your current camera and colors, or click **Download PNG** / **Download SVG** immediately.

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
