# Python API Reference

All high-level plotting functions, core data classes, style helpers, and ecosystem adapters are re-exported directly at the top level via `import xkcd_matchem as xm`.

## Quick Function Finder

Looking for the right function? Start here:

| I want to... | Reach for |
| --- | --- |
| Load a structure, spectrum, band structure, or cube | `xm.load(path)`, `xm.load_trajectory(path)` |
| Plot a molecule in 3D or 2D skeletal view | `xm.plot_molecule(...)`, `xm.render_structure_3d(...)`, `xm.render_structure_2d(...)` |
| Plot a crystal, Miller plane, slab, or polyhedra | `xm.plot_crystal(...)`, `xm.plot_slab(...)`, `xm.plot_coordination(...)` |
| Spin a structure in the offline 3D Studio | `xm.view_structure(...)`, `xm.export_interactive_html(...)` |
| Plot band structures, DOS, MOs, or cube slices | `xm.plot_bands(...)`, `xm.plot_dos(...)`, `xm.plot_bands_dos(...)`, `xm.plot_mo_diagram(...)`, `xm.plot_volumetric_slice(...)` |
| Plot IR/UV-Vis spectra or simulated powder XRD | `xm.plot_spectrum(...)`, `xm.plot_xrd(...)`, `xm.compute_xrd(...)` |
| Sketch reaction pathways or NEB barriers | `xm.plot_reaction_profile(...)`, `xm.plot_neb(...)` |
| Analyze or animate MD trajectories & RDFs | `xm.plot_md_timeseries(...)`, `xm.plot_rdf(...)`, `xm.plot_rmsd(...)`, `xm.animate_trajectory(...)` |
| Plot hulls, EOS fits, Pourbaix maps, or parity benchmarks | `xm.plot_phase_diagram_2d(...)`, `xm.plot_eos(...)`, `xm.plot_pourbaix(...)`, `xm.plot_parity(...)`, `xm.plot_bar_comparison(...)`, `xm.plot_pipeline_diagram(...)` |
| Customize themes, element colors, or existing Matplotlib axes | `xm.xkcd_style(...)`, `xm.register_preset(...)`, `xm.register_palette(...)`, `xm.register_element_palette(...)`, `xm.style_existing_axes(...)` |
| Compose a millimeter-accurate conference poster | `xm.Board`, `xm.Container`, `xm.Card`, `xm.Badge` |

## Top-Level Package (`xkcd_matchem`)

```{eval-rst}
.. automodule:: xkcd_matchem
   :no-members:
```

## Themes, Palettes & Fonts (`xkcd_matchem.style`)

- **Presets & themes:** `PRESETS`, `ThemePreset`, `list_presets()`, `get_preset()`, `register_preset()`, `xkcd_style()`, `build_rc_params()`, `apply_xkcd_spines()`, `style_existing_axes()`
- **Palettes & elements:** `get_palette()`, `register_palette()`, `list_palettes()`, `get_element_color()`, `register_element_palette()`, `list_element_palettes()`
- **Fonts:** `list_bundled_fonts()`, `get_bundled_font_path()`, `ensure_fonts_registered()`, `fetch_xkcd_script()`

```{eval-rst}
.. automodule:: xkcd_matchem.style.theme
   :members:
```

```{eval-rst}
.. automodule:: xkcd_matchem.style.palettes
   :members:
```

## Core Data Models & Geometry (`xkcd_matchem.core`)

```{eval-rst}
.. automodule:: xkcd_matchem.core.structure
   :members:
```

```{eval-rst}
.. automodule:: xkcd_matchem.core.geometry
   :members:
```

```{eval-rst}
.. automodule:: xkcd_matchem.core.electronic
   :members:
```

```{eval-rst}
.. automodule:: xkcd_matchem.core.trajectory
   :members:
```

## High-Level Plotters (`xkcd_matchem.plots`)

```{eval-rst}
.. automodule:: xkcd_matchem.plots.structure
   :members:
```

```{eval-rst}
.. automodule:: xkcd_matchem.plots.electronic
   :members:
```

```{eval-rst}
.. automodule:: xkcd_matchem.plots.spectroscopy
   :members:
```

```{eval-rst}
.. automodule:: xkcd_matchem.plots.reactions
   :members:
```

```{eval-rst}
.. automodule:: xkcd_matchem.plots.trajectory
   :members:
```

```{eval-rst}
.. automodule:: xkcd_matchem.plots.thermo
   :members:
```

```{eval-rst}
.. automodule:: xkcd_matchem.plots.charts
   :members:
```

## Interactive 3D Studio & Renderers (`xkcd_matchem.render`)

```{eval-rst}
.. automodule:: xkcd_matchem.render.interactive
   :members:
```

```{eval-rst}
.. automodule:: xkcd_matchem.render.structure3d
   :members:
```

```{eval-rst}
.. automodule:: xkcd_matchem.render.structure2d
   :members:
```

## Opt-In Poster Layout (`xkcd_matchem.layout`)

```{eval-rst}
.. automodule:: xkcd_matchem.layout.canvas
   :members:
```
