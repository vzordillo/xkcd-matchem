# Copyright 2026 xkcd-matchem contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Styling, font management, background grids, shading, and color palettes."""

from __future__ import annotations

from xkcd_matchem.style import fonts
from xkcd_matchem.style import palettes
from xkcd_matchem.style import patterns
from xkcd_matchem.style import theme


__all__ = [
    "BACKGROUND_STYLES",
    "BLUEPRINT_BG",
    "BLUEPRINT_INK",
    "DARK_BOARD",
    "DARK_CHALK",
    "INK",
    "OKABE_ITO",
    "PAPER",
    "PRESETS",
    "SHADE_STYLES",
    "ThemePreset",
    "WARM_PAPER",
    "apply_background_grid",
    "apply_xkcd_spines",
    "build_rc_params",
    "clear_background_grid",
    "compute_background_grid_coords",
    "ensure_fonts_registered",
    "fetch_xkcd_script",
    "get_bundled_font_path",
    "get_default_font",
    "get_element_color",
    "get_palette",
    "get_preset",
    "list_background_styles",
    "list_bundled_fonts",
    "list_element_palettes",
    "list_palettes",
    "list_presets",
    "list_shade_styles",
    "register_element_palette",
    "register_font",
    "register_palette",
    "register_preset",
    "resolve_font_chain",
    "resolve_hatch",
    "set_default_font",
    "shade_between",
    "style_existing_axes",
    "use_xkcd_script",
    "xkcd_style",
]

BACKGROUND_STYLES = patterns.BACKGROUND_STYLES
BLUEPRINT_BG = patterns.BLUEPRINT_BG
BLUEPRINT_INK = patterns.BLUEPRINT_INK
DARK_BOARD = palettes.DARK_BOARD
DARK_CHALK = palettes.DARK_CHALK
INK = palettes.INK
OKABE_ITO = palettes.OKABE_ITO
PAPER = palettes.PAPER
PRESETS = theme.PRESETS
SHADE_STYLES = patterns.SHADE_STYLES
ThemePreset = theme.ThemePreset
WARM_PAPER = palettes.WARM_PAPER
apply_background_grid = patterns.apply_background_grid
apply_xkcd_spines = theme.apply_xkcd_spines
build_rc_params = theme.build_rc_params
clear_background_grid = patterns.clear_background_grid
compute_background_grid_coords = patterns.compute_background_grid_coords
ensure_fonts_registered = fonts.ensure_fonts_registered
fetch_xkcd_script = fonts.fetch_xkcd_script
get_bundled_font_path = fonts.get_bundled_font_path
get_default_font = fonts.get_default_font
get_element_color = palettes.get_element_color
get_palette = palettes.get_palette
get_preset = theme.get_preset
list_background_styles = patterns.list_background_styles
list_bundled_fonts = fonts.list_bundled_fonts
list_element_palettes = palettes.list_element_palettes
list_palettes = palettes.list_palettes
list_presets = theme.list_presets
list_shade_styles = patterns.list_shade_styles
register_element_palette = palettes.register_element_palette
register_font = fonts.register_font
register_palette = palettes.register_palette
register_preset = theme.register_preset
resolve_font_chain = fonts.resolve_font_chain
resolve_hatch = patterns.resolve_hatch
set_default_font = fonts.set_default_font
shade_between = patterns.shade_between
style_existing_axes = theme.style_existing_axes
use_xkcd_script = fonts.use_xkcd_script
xkcd_style = theme.xkcd_style
