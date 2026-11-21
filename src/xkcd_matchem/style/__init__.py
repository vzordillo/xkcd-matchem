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
"""Styling, font management, and color palette utilities."""

from __future__ import annotations

from xkcd_matchem.style import fonts
from xkcd_matchem.style import palettes
from xkcd_matchem.style import theme


__all__ = [
    "DARK_BOARD",
    "DARK_CHALK",
    "INK",
    "OKABE_ITO",
    "PAPER",
    "PRESETS",
    "ThemePreset",
    "apply_xkcd_spines",
    "build_rc_params",
    "ensure_fonts_registered",
    "fetch_xkcd_script",
    "get_bundled_font_path",
    "get_element_color",
    "get_palette",
    "get_preset",
    "list_bundled_fonts",
    "list_palettes",
    "list_presets",
    "resolve_font_chain",
    "xkcd_style",
]

DARK_BOARD = palettes.DARK_BOARD
DARK_CHALK = palettes.DARK_CHALK
INK = palettes.INK
OKABE_ITO = palettes.OKABE_ITO
PAPER = palettes.PAPER
PRESETS = theme.PRESETS
ThemePreset = theme.ThemePreset
apply_xkcd_spines = theme.apply_xkcd_spines
build_rc_params = theme.build_rc_params
ensure_fonts_registered = fonts.ensure_fonts_registered
fetch_xkcd_script = fonts.fetch_xkcd_script
get_bundled_font_path = fonts.get_bundled_font_path
get_element_color = palettes.get_element_color
get_palette = palettes.get_palette
get_preset = theme.get_preset
list_bundled_fonts = fonts.list_bundled_fonts
list_palettes = palettes.list_palettes
list_presets = theme.list_presets
resolve_font_chain = fonts.resolve_font_chain
xkcd_style = theme.xkcd_style
