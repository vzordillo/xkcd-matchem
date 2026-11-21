# Python API Reference

All primary classes and plotting functions are exported directly from the top-level package `import xkcd_matchem as xm`.

## Top-Level Package (`xkcd_matchem`)

```{eval-rst}
.. automodule:: xkcd_matchem
   :no-members:
```

## Themes, Palettes & Fonts (`xkcd_matchem.style`)

- Presets & themes: `PRESETS`, `ThemePreset`, `list_presets()`, `get_preset()`, `register_preset()`, `xkcd_style()`, `build_rc_params()`, `apply_xkcd_spines()`, `style_existing_axes()`
- Palettes & elements: `get_palette()`, `register_palette()`, `list_palettes()`, `get_element_color()`, `register_element_palette()`, `list_element_palettes()`
- Fonts: `list_bundled_fonts()`, `get_bundled_font_path()`, `ensure_fonts_registered()`, `fetch_xkcd_script()`

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
