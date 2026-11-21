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
"""Default and user-customizable perceptual and chemical color palettes."""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence

from xkcd_matchem.core import exceptions
from xkcd_matchem.data import elements


INK = "#111111"
PAPER = "#FFFFFF"
WARM_PAPER = "#FFFDF7"
DARK_BOARD = "#181C24"
DARK_CHALK = "#F2EFE9"
MUTED_GRAY = "#5A6270"
LIGHT_GRID = "#E5E7EB"
PASTEL_BG = "#F5F2EB"
VACUUM_TINT = "#D3E6EE"
SLAB_TINT = "#B9D5C9"
GAP_TINT = "#F8DDC0"

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

# Crisp high-contrast hand-drawn defaults tuned for pure #FFFFFF white and ink.
_XKCD_ELEMENT_OVERRIDES: dict[str, str] = {
    "H": "#F5F5F5",
    "C": "#5A5F69",
    "N": "#3B6FD8",
    "O": "#E74C3C",
    "F": "#68D391",
    "S": "#E5C13B",
    "Cl": "#2ECC71",
    "Br": "#A04000",
    "I": "#8854D0",
    "P": "#F39C12",
    "Si": "#D9A05B",
    "Fe": "#D97736",
    "Cu": "#C0392B",
    "Ti": "#7F8C9A",
    "Mo": "#3E6E8E",
    "Au": "#F2C94C",
    "Li": "#A78BFA",
    "Na": "#7C6CE0",
    "K": "#9B59B6",
    "Mg": "#27AE60",
    "Ca": "#48BB78",
    "Al": "#B8C2CC",
    "Cr": "#8E63CE",
    "Ta": "#8E63CE",
    "Zr": "#1ABC9C",
    "Pt": "#5D6D7E",
    "Ge": "#16A085",
    "Se": "#F39C12",
}

_PASTEL_ELEMENT_OVERRIDES: dict[str, str] = {
    "H": "#FAFAFA",
    "C": "#8D99AE",
    "N": "#8ECAE6",
    "O": "#F4978E",
    "F": "#B7E4C7",
    "S": "#FFE082",
    "Cl": "#95D5B2",
    "Br": "#D4A373",
    "I": "#CDB4DB",
    "P": "#F9C784",
    "Si": "#E9C46A",
    "Fe": "#E7A977",
    "Cu": "#E5989B",
    "Ti": "#B8C0CC",
    "Mo": "#9DB3C6",
    "Zr": "#80CBC4",
    "Cr": "#B8A9DC",
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

_CUSTOM_PALETTES: dict[str, tuple[str, ...]] = {}
_CUSTOM_ELEMENT_PALETTES: dict[str, tuple[dict[str, str], str]] = {}
_BUILTIN_ELEMENT_SCHEMES: tuple[str, ...] = (
    "colorblind",
    "jmol",
    "pastel",
    "xkcd",
)


def register_palette(name: str, colors: Sequence[str]) -> None:
    """Registers a custom discrete color series palette by name.

    Args:
      name: Case-insensitive palette key to register or update.
      colors: Non-empty sequence of Matplotlib-compatible color strings.

    Raises:
      StyleError: If ``name`` is blank or ``colors`` is empty.
    """
    key = name.strip().lower()
    if not key:
        raise exceptions.StyleError("Palette name cannot be empty.")
    color_tuple = tuple(str(col) for col in colors)
    if not color_tuple:
        raise exceptions.StyleError(
            f"Palette {name!r} must contain at least one color."
        )
    _CUSTOM_PALETTES[key] = color_tuple


def register_element_palette(
    name: str,
    colors: Mapping[str, str],
    base: str = "xkcd",
) -> None:
    """Registers a named custom chemical-element color scheme.

    Args:
      name: Scheme identifier passed later as ``palette=...`` or ``scheme=...``.
      colors: Mapping from element symbol (case-insensitive) to color string.
      base: Fallback built-in scheme for elements omitted from ``colors``.

    Raises:
      StyleError: If ``name`` is empty or ``base`` is invalid.
    """
    key = name.strip().lower()
    if not key:
        raise exceptions.StyleError("Element palette name cannot be empty.")
    base_key = base.strip().lower()
    if (
        base_key not in _BUILTIN_ELEMENT_SCHEMES
        and base_key not in _CUSTOM_ELEMENT_PALETTES
    ):
        supported = ", ".join(list_element_palettes())
        raise exceptions.StyleError(
            f"Unknown base element palette {base!r}. Available: {supported}"
        )
    normalized: dict[str, str] = {}
    for sym, col in colors.items():
        rec = elements.get_element(sym)
        normalized[rec.symbol] = str(col)
    _CUSTOM_ELEMENT_PALETTES[key] = (normalized, base_key)


def list_palettes() -> tuple[str, ...]:
    """Returns built-in and custom discrete series palette names."""
    return tuple(sorted(set(_NAMED_PALETTES) | set(_CUSTOM_PALETTES)))


def list_element_palettes() -> tuple[str, ...]:
    """Returns built-in and custom chemical-element color scheme names."""
    return tuple(
        sorted(set(_BUILTIN_ELEMENT_SCHEMES) | set(_CUSTOM_ELEMENT_PALETTES))
    )


def get_palette(name: str | Sequence[str] = "okabe_ito") -> tuple[str, ...]:
    """Resolves a named or inline sequence discrete color palette.

    Args:
      name: Registered palette identifier or an inline sequence of hex/named
        colors.

    Returns:
      Tuple of color strings for multi-series plots.

    Raises:
      StyleError: If ``name`` is unknown or an empty sequence.
    """
    if isinstance(name, Sequence) and not isinstance(name, str):
        custom = tuple(str(col) for col in name)
        if not custom:
            raise exceptions.StyleError("Custom palette sequence is empty.")
        return custom
    key = str(name).lower()
    if key in _CUSTOM_PALETTES:
        return _CUSTOM_PALETTES[key]
    if key not in _NAMED_PALETTES:
        supported = ", ".join(list_palettes())
        raise exceptions.StyleError(
            f"Unknown palette {name!r}. Available: {supported}"
        )
    return _NAMED_PALETTES[key]


def get_element_color(
    symbol: str,
    scheme: str | Mapping[str, str] = "xkcd",
    fallback: str = "xkcd",
) -> str:
    """Resolves the display fill color for a chemical element.

    Args:
      symbol: Chemical element symbol (case-insensitive, e.g. ``"Mo"``).
      scheme: Scheme name (``"xkcd"``, ``"jmol"``, ``"colorblind"``,
        ``"pastel"``, or registered custom name) or inline ``{symbol: color}``
        mapping overriding defaults.
      fallback: Fallback scheme name when an inline mapping omits ``symbol``.

    Returns:
      Color string such as ``"#RRGGBB"``.

    Raises:
      StyleError: If ``scheme`` is unrecognized or the symbol is invalid.
    """
    record = elements.get_element(symbol)
    if isinstance(scheme, Mapping):
        for probe in (record.symbol, record.symbol.lower(), symbol):
            if probe in scheme:
                return str(scheme[probe])
        return get_element_color(record.symbol, scheme=fallback)

    key = scheme.lower()
    if key in _CUSTOM_ELEMENT_PALETTES:
        overrides, base_scheme = _CUSTOM_ELEMENT_PALETTES[key]
        if record.symbol in overrides:
            return overrides[record.symbol]
        return get_element_color(record.symbol, scheme=base_scheme)

    result: str | None = None
    if key == "jmol":
        result = record.jmol_color
    elif key == "xkcd":
        result = _XKCD_ELEMENT_OVERRIDES.get(record.symbol, record.jmol_color)
    elif key == "pastel":
        result = _PASTEL_ELEMENT_OVERRIDES.get(
            record.symbol,
            _XKCD_ELEMENT_OVERRIDES.get(record.symbol, record.jmol_color),
        )
    elif key == "colorblind":
        result = _COLORBLIND_ELEMENT_OVERRIDES.get(
            record.symbol,
            _COLORBLIND_FAMILY_COLORS.get(record.family, "#56B4E9"),
        )
    if result is not None:
        return result
    supported = ", ".join(list_element_palettes())
    raise exceptions.StyleError(
        f"Unknown element color scheme {scheme!r}. Available: {supported}"
    )
