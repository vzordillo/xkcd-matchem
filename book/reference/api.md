# Python API Reference

All high-level plotting functions, core data classes, primary style/layout/render helpers, and ecosystem conversion bridges are re-exported directly at the top level via `import xkcd_matchem as xm`. Lower-level helpers and format-specific readers live inside their dedicated subpackages (`xm.style`, `xm.io`, `xm.core`, `xm.render`, `xm.layout`, `xm.data`).

## Quick Function Finder

Looking for the right function? Start here:

| I want to... | Reach for |
| --- | --- |
| Load or save a structure, trajectory, spectrum, bands, or cube | `xm.load(path)`, `xm.load_trajectory(path)`, `xm.save(atoms, path)` |
| Plot a molecule in 3D or 2D skeletal view | `xm.plot_molecule(...)`, `xm.render_structure_3d(...)`, `xm.render_structure_2d(...)` |
| Plot a crystal, Miller plane, Brillouin zone, Wulff shape, slab, or polyhedra | `xm.plot_crystal(...)`, `xm.plot_brillouin_zone(...)`, `xm.plot_wulff_shape(...)`, `xm.plot_slab(...)`, `xm.plot_coordination(...)` |
| Spin a structure, trajectory, or CUBE orbital in the offline 3D Studio | `xm.view_structure(...)`, `xm.export_interactive_html(...)` |
| Plot band structures, COHP, phonons, DOS, MOs, or 3D/2D cubes | `xm.plot_bands(...)`, `xm.plot_cohp(...)`, `xm.plot_phonon_bands(...)`, `xm.plot_dos(...)`, `xm.plot_bands_dos(...)`, `xm.plot_mo_diagram(...)`, `xm.plot_volumetric_slice(...)`, `xm.plot_isosurface(...)` |
| Plot IR/UV-Vis spectra or simulated powder XRD | `xm.plot_spectrum(...)`, `xm.plot_xrd(...)`, `xm.compute_xrd(...)` |
| Sketch reaction pathways, schemes, or NEB barriers | `xm.plot_reaction_profile(...)`, `xm.plot_reaction_scheme(...)`, `xm.plot_neb(...)` |
| Analyze or animate MD trajectories & RDFs | `xm.plot_md_timeseries(...)`, `xm.plot_rdf(...)`, `xm.plot_rmsd(...)`, `xm.animate_trajectory(...)` |
| Plot hulls, EOS fits, Pourbaix, volcanos, defects, or parity | `xm.plot_phase_diagram_2d(...)`, `xm.plot_eos(...)`, `xm.plot_pourbaix(...)`, `xm.plot_volcano(...)`, `xm.plot_defect_formation(...)`, `xm.plot_parity(...)`, `xm.plot_bar_comparison(...)`, `xm.plot_pipeline_diagram(...)` |
| Plot an xkcd periodic table, heatmap, or formula spotlight | `xm.plot_periodic_table(...)` |
| Plot ternary hulls, space-group sunbursts, or histograms | `xm.plot_ternary_phase_diagram(...)`, `xm.plot_spacegroup_distribution(...)`, `xm.plot_histogram(...)` |
| Plot GNN crystal/molecular graphs or informatics networks | `xm.plot_structure_graph(...)`, `xm.plot_graph_network(...)` |
| Draw xkcd chemistry/materials icons & element tiles | `xm.draw_icon(...)`, `xm.draw_element_badge(...)`, `xm.plot_icon_sheet(...)` |
| Add notebook/graph/dotgrid/blueprint backgrounds or hatched fills | `xm.apply_background_grid(...)`, `xm.clear_background_grid(...)`, `xm.shade_between(...)`, `shade=...`, `background_grid=...` |
| Customize themes, stroke weights, element colors, or existing Matplotlib axes | `xm.xkcd_style(...)`, `xm.stroke_weights(...)`, `xm.register_preset(...)`, `xm.register_palette(...)`, `xm.register_element_palette(...)`, `xm.style_existing_axes(...)` |
| Compose a poster, stick-scientist comic strip, or vintage parchment scroll | `xm.Board`, `xm.Container`, `xm.Card`, `xm.ComicStrip`, `xm.plot_comic_strip(...)`, `xm.draw_stick_scientist(...)`, `xm.draw_speech_bubble(...)`, `xm.ScrollCard`, `xm.plot_scroll_card(...)`, `xm.Badge`, `xm.save_figure(...)` |

## Top-Level Package (`xkcd_matchem`)

```{eval-rst}
.. automodule:: xkcd_matchem
   :no-members:
```

## Themes, Palettes & Fonts (`xkcd_matchem.style`)

- **Presets & themes:** Top-level `xm.PRESETS`, `xm.ThemePreset`, `xm.StrokeHierarchy`, `xm.list_presets()`, `xm.register_preset()`, `xm.xkcd_style()`, `xm.stroke_weights()`, `xm.style_existing_axes()`; submodule helpers `xm.style.get_preset()`, `xm.style.build_rc_params()`, `xm.style.apply_xkcd_spines()`.
- **Palettes & elements:** Top-level `xm.OKABE_ITO`, `xm.get_palette()`, `xm.register_palette()`, `xm.get_element_color()`, `xm.register_element_palette()`, `xm.list_element_palettes()`; submodule helper `xm.style.list_palettes()`.
- **Background grids & shading:** Top-level `xm.BACKGROUND_STYLES`, `xm.SHADE_STYLES`, `xm.list_background_styles()`, `xm.list_shade_styles()`, `xm.apply_background_grid()`, `xm.clear_background_grid()`, `xm.compute_background_grid_coords()`, `xm.resolve_hatch()`, `xm.shade_between()`.
- **Fonts:** Top-level `xm.register_font()`, `xm.set_default_font()`, `xm.get_default_font()`, `xm.use_xkcd_script()`, `xm.fetch_xkcd_script()`; submodule helpers `xm.style.list_bundled_fonts()`, `xm.style.get_bundled_font_path()`, `xm.style.ensure_fonts_registered()`.

```{eval-rst}
.. automodule:: xkcd_matchem.style.theme
   :members:
```

```{eval-rst}
.. automodule:: xkcd_matchem.style.palettes
   :members:
```

```{eval-rst}
.. automodule:: xkcd_matchem.style.patterns
   :members:
```

```{eval-rst}
.. automodule:: xkcd_matchem.style.fonts
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
.. automodule:: xkcd_matchem.core.bonding
   :members:
```

```{eval-rst}
.. automodule:: xkcd_matchem.core.electronic
   :members:
```

```{eval-rst}
.. automodule:: xkcd_matchem.core.spectroscopy
   :members:
```

```{eval-rst}
.. automodule:: xkcd_matchem.core.isosurface
   :members:
```

```{eval-rst}
.. automodule:: xkcd_matchem.core.neighbors
   :members:
```

```{eval-rst}
.. automodule:: xkcd_matchem.core.brillouin
   :members:
```

```{eval-rst}
.. automodule:: xkcd_matchem.core.wulff
   :members:
```

```{eval-rst}
.. automodule:: xkcd_matchem.core.trajectory
   :members:
```

```{eval-rst}
.. automodule:: xkcd_matchem.core.exceptions
   :members:
```

## Unified File I/O (`xkcd_matchem.io`)

```{eval-rst}
.. automodule:: xkcd_matchem.io.loader
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

```{eval-rst}
.. automodule:: xkcd_matchem.plots.ptable
   :members:
```

```{eval-rst}
.. automodule:: xkcd_matchem.plots.materials
   :members:
```

```{eval-rst}
.. automodule:: xkcd_matchem.plots.graphs
   :members:
```

## Interactive 3D Studio, Icons & Renderers (`xkcd_matchem.render`)

```{eval-rst}
.. automodule:: xkcd_matchem.render.interactive
   :members:
```

```{eval-rst}
.. automodule:: xkcd_matchem.render.icons
   :members:
```

```{eval-rst}
.. automodule:: xkcd_matchem.render.structure3d
   :members:
```

```{eval-rst}
.. automodule:: xkcd_matchem.render.artistic3d
   :members:
```

```{eval-rst}
.. automodule:: xkcd_matchem.render.structure2d
   :members:
```

```{eval-rst}
.. automodule:: xkcd_matchem.render.primitives
   :members:
```

```{eval-rst}
.. automodule:: xkcd_matchem.render.callouts
   :members:
```

## Opt-In Poster Layout, Comics & Export (`xkcd_matchem.layout`)

```{eval-rst}
.. automodule:: xkcd_matchem.layout.canvas
   :members:
```

```{eval-rst}
.. automodule:: xkcd_matchem.layout.scroll
   :members:
```

```{eval-rst}
.. automodule:: xkcd_matchem.layout.comic
   :members:
```

```{eval-rst}
.. automodule:: xkcd_matchem.layout.export
   :members:
```
