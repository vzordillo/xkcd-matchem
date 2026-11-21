# Interactive 3D xkcd Studio

Explore crystals, slabs, polyhedra, and molecules interactively right inside your browser — **no Python backend or MyBinder spin-up required**.

- **Drag** to orbit azimuth/elevation (`Shift + Drag` to roll, **Scroll** to zoom).
- Switch between **Isometric**, **Perspective**, **Front**, **Side**, **Top**, and crystallographic **`a`, `b`, `c`, `a*`, `b*`, `c*`** viewpoints.
- Build up to `3x3x3` supercells live, pick custom hex colors for **every chemical element** and the canvas background, tune the xkcd wiggle strength, or load your own `.xyz`, `POSCAR`, `.sdf`, or `.pdb` file.
- Click **Copy Python** to grab the exact `xkcd_matchem` script matching your interactive view, or download **PNG** / **SVG**.

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

## Launching from Python or CLI

In a Jupyter Notebook or script:

```python
import xkcd_matchem as xm

# Embed inline inside Jupyter or export a standalone offline HTML viewer:
view = xm.view_structure(
    "tio2_slab.poscar",
    style="polyhedra",
    polyhedra=("Ti", "O"),
    supercell=(2, 2, 1),
    palette={"Ti": "#4E79A7", "O": "#E15759"},
)
view  # renders inline via _repr_html_()

# Open directly in your system browser:
xm.view_structure("aspirin.xyz", backend="browser")
```

From your terminal:

```bash
xkcd-matchem view tio2_slab.poscar --supercell 2 2 1 --view perspective --colors Ti=#E07A5F O=#3D405B --backend browser
```
