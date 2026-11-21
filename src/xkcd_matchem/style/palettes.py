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
"""Perceptual and chemical color palettes for xkcd-matchem."""

from __future__ import annotations

from xkcd_matchem.core import exceptions
from xkcd_matchem.data import elements


INK = "#1E1E1E"
PAPER = "#FFFDF7"
DARK_BOARD = "#181C24"
DARK_CHALK = "#F2EFE9"
MUTED_GRAY = "#6B7280"
LIGHT_GRID = "#E5E7EB"
PASTEL_BG = "#F5F2EB"

# Okabe & Ito (2002; mod. 2008) Color Universal Design (CUD) 8-color palette:
# black, orange, sky blue, bluish green, yellow, blue, vermilion, purple.
OKABE_ITO: tuple[str, ...] = (
    "#000000",
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
)

OKABE_ITO_NAMED: dict[str, str] = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermilion": "#D55E00",
    "reddish_purple": "#CC79A7",
}

# Warm hand-drawn palette tuned for contrast against #FFFDF7 paper and dark ink.
_XKCD_ELEMENT_OVERRIDES: dict[str, str] = {
    "H": "#F7F5EF",
    "C": "#6E737D",
    "N": "#4B7BEC",
    "O": "#E85A4F",
    "F": "#68D391",
    "S": "#F6C945",
    "Cl": "#38B2AC",
    "Br": "#B84A39",
    "I": "#8854D0",
    "P": "#F59E0B",
    "Si": "#D9A86C",
    "Fe": "#D97736",
    "Cu": "#E08E55",
    "Ti": "#9BA3AE",
    "Mo": "#48A9A6",
    "Au": "#F2C94C",
    "Li": "#A78BFA",
    "Na": "#7C6CE0",
    "K": "#9B59B6",
    "Mg": "#6EE7B7",
    "Ca": "#48BB78",
    "Al": "#B8C2CC",
}

_COLORBLIND_FAMILY_COLORS: dict[str, str] = {
    "Non-Metal": "#56B4E9",
    "Noblegas": "#CC79A7",
    "Alkali_Earth": "#E69F00",
    "Alkaline_Earth": "#009E73",
    "Metalloids": "#F0E442",
    "Halogen": "#009E73",
    "Transition": "#0072B2",
    "Other_Metal": "#88A0B8",
    "Rare_Earth": "#CC79A7",
}

_COLORBLIND_ELEMENT_OVERRIDES: dict[str, str] = {
    "H": "#F5F5F0",
    "C": "#4A4E57",
    "N": "#0072B2",
    "O": "#D55E00",
    "S": "#F0E442",
    "P": "#E69F00",
    "F": "#56B4E9",
    "Cl": "#009E73",
    "Br": "#9C3D00",
    "I": "#CC79A7",
}

_NAMED_PALETTES: dict[str, tuple[str, ...]] = {
    "okabe_ito": OKABE_ITO,
    "pastel": (
        "#4E79A7",
        "#F28E2B",
        "#59A14F",
        "#E15759",
        "#76B7B2",
        "#B07AA1",
        "#EDC948",
        "#9C755F",
    ),
    "vibrant": (
        "#0072B2",
        "#D55E00",
        "#009E73",
        "#CC79A7",
        "#E69F00",
        "#56B4E9",
        "#332288",
        "#882255",
    ),
}


def list_palettes() -> tuple[str, ...]:
    """Returns available discrete series palette names."""
    return tuple(sorted(_NAMED_PALETTES))


def get_palette(name: str = "okabe_ito") -> tuple[str, ...]:
    """Returns a named tuple of hex colors for multi-series plots.

    Args:
      name: Palette identifier (``"okabe_ito"``, ``"pastel"``, ``"vibrant"``).

    Returns:
      Tuple of ``#RRGGBB`` hex strings.

    Raises:
      StyleError: If ``name`` is not a recognized palette.
    """
    key = name.lower()
    if key not in _NAMED_PALETTES:
        supported = ", ".join(list_palettes())
        raise exceptions.StyleError(
            f"Unknown palette {name!r}. Available: {supported}"
        )
    return _NAMED_PALETTES[key]


def get_element_color(
    symbol: str,
    scheme: str = "xkcd",
) -> str:
    """Resolves the display fill hex color for a chemical element.

    Args:
      symbol: Chemical element symbol (case-insensitive, e.g. ``"Mo"``).
      scheme: Element coloring scheme: ``"xkcd"``, ``"jmol"``, or
        ``"colorblind"``.

    Returns:
      Hex color string ``#RRGGBB``.

    Raises:
      StyleError: If ``scheme`` is unrecognized or the symbol is invalid.
    """
    record = elements.get_element(symbol)
    key = scheme.lower()
    if key == "jmol":
        return record.jmol_color
    if key == "xkcd":
        return _XKCD_ELEMENT_OVERRIDES.get(record.symbol, record.jmol_color)
    if key == "colorblind":
        if record.symbol in _COLORBLIND_ELEMENT_OVERRIDES:
            return _COLORBLIND_ELEMENT_OVERRIDES[record.symbol]
        return _COLORBLIND_FAMILY_COLORS.get(record.family, "#56B4E9")
    raise exceptions.StyleError(
        f"Unknown element color scheme {scheme!r}. "
        "Expected 'xkcd', 'jmol', or 'colorblind'."
    )
