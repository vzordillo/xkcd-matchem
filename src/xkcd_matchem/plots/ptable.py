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
"""xkcd-style Periodic Table of Elements, heatmaps, and formula spotlights."""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
import re

import matplotlib
from matplotlib import axes as maxes
from matplotlib import colors as mcolors
from matplotlib import patches
from matplotlib import pyplot as plt
import numpy as np

from xkcd_matchem.core import exceptions
from xkcd_matchem.data import elements
from xkcd_matchem.render import primitives
from xkcd_matchem.style import patterns
from xkcd_matchem.style import theme


_FAMILY_DISPLAY_COLORS: dict[str, tuple[str, str]] = {
    "Alkali Metals": ("Alkali Metals", "#F4A261"),
    "Alkaline Earth": ("Alkaline Earth", "#E9C46A"),
    "Transition Metals": ("Transition Metals", "#8ECAE6"),
    "Post-Transition": ("Post-Transition", "#95D5B2"),
    "Metalloids": ("Metalloids", "#B7E4C7"),
    "Non-Metals": ("Non-Metals", "#A8DADC"),
    "Halogens": ("Halogens", "#CDB4DB"),
    "Noble Gases": ("Noble Gases", "#FFAFCC"),
    "Lanthanides": ("Lanthanides", "#F7CAD0"),
    "Actinides": ("Actinides", "#FBC4AB"),
}

_BODR_TO_DISPLAY: dict[str, str] = {
    "Alkali_Earth": "Alkali Metals",
    "Alkaline_Earth": "Alkaline Earth",
    "Transition": "Transition Metals",
    "Other_Metal": "Post-Transition",
    "Metalloids": "Metalloids",
    "Non-Metal": "Non-Metals",
    "Halogen": "Halogens",
    "Noblegas": "Noble Gases",
    "Rare_Earth": "Lanthanides",
}

_FORMULA_TOKEN_RE = re.compile(r"([A-Z][a-z]?)")


def _display_family(rec: elements.ElementRecord) -> str:
    """Returns clean IUPAC display family name for an ``ElementRecord``."""
    if 57 <= rec.atomic_number <= 71:
        return "Lanthanides"
    if 89 <= rec.atomic_number <= 103:
        return "Actinides"
    return _BODR_TO_DISPLAY.get(rec.family, "Transition Metals")


def _tile_coords(rec: elements.ElementRecord) -> tuple[float, float]:
    """Returns ``(col, row)`` periodic-table coordinates (1-indexed cols)."""
    z = rec.atomic_number
    if 57 <= z <= 71:
        return float(3 + (z - 57)), 8.45
    if 89 <= z <= 103:
        return float(3 + (z - 89)), 9.55
    return float(rec.group), float(rec.period)


def _extract_property_values(
    prop_name: str,
) -> tuple[dict[str, float], str]:
    """Extracts a built-in numeric element property across all 118 elements."""
    key = prop_name.strip().lower()
    attr_map = {
        "atomic_mass": ("atomic_mass", "Atomic Mass (u)"),
        "covalent_radius": ("covalent_radius", "Covalent Radius (A)"),
        "vdw_radius": ("vdw_radius", "van der Waals Radius (A)"),
        "electronegativity": (
            "electronegativity",
            "Pauling Electronegativity",
        ),
        "atomic_number": ("atomic_number", "Atomic Number (Z)"),
    }
    if key not in attr_map:
        valid = ", ".join(sorted(attr_map))
        raise exceptions.StructureError(
            f"Unknown element_property {prop_name!r}. Valid: {valid}"
        )
    attr, label = attr_map[key]
    vals: dict[str, float] = {}
    for rec in elements.ELEMENTS_BY_Z.values():
        val = getattr(rec, attr)
        if val is not None:
            vals[rec.symbol] = float(val)
    return vals, label


def _parse_highlight_spec(
    highlight: str | Sequence[str] | Mapping[str, str],
) -> dict[str, str]:
    """Coerces a formula string, symbol list, or dict into ``{sym: color}``."""
    if isinstance(highlight, Mapping):
        return {
            elements.get_element(k).symbol: str(v) for k, v in highlight.items()
        }
    if isinstance(highlight, str):
        syms = [
            elements.get_element(tok).symbol
            for tok in _FORMULA_TOKEN_RE.findall(highlight)
            if tok in elements.ELEMENTS_BY_SYMBOL
        ]
    else:
        syms = [elements.get_element(s).symbol for s in highlight]
    palette = ("#E76F51", "#2A9D8F", "#E9C46A", "#457B9D", "#9B5DE5", "#F4A261")
    return {
        sym: palette[idx % len(palette)]
        for idx, sym in enumerate(dict.fromkeys(syms))
    }


def _draw_ptable_colorbar(
    ax: maxes.Axes,
    cmap_obj: mcolors.Colormap,
    norm: mcolors.Normalize,
    label: str,
    ink_col: str,
) -> None:
    """Draws an inline horizontal colorbar inside the open Groups 3-12 bay."""
    x0, x1 = 4.2, 11.8
    y0, y1 = 1.18, 1.72
    grad = np.linspace(0.0, 1.0, 256)[None, :]
    ax.imshow(
        grad,
        extent=(x0, x1, y1, y0),
        cmap=cmap_obj,
        aspect="auto",
        zorder=2.5,
    )
    bar_box = patches.Rectangle(
        (x0, y0),
        x1 - x0,
        y1 - y0,
        facecolor="none",
        edgecolor=ink_col,
        linewidth=1.3,
        zorder=3.0,
    )
    ax.add_patch(bar_box)
    ax.text(
        0.5 * (x0 + x1),
        y0 - 0.18,
        label,
        ha="center",
        va="bottom",
        fontsize=10.5,
        color=ink_col,
        zorder=4.0,
    )
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        tx = x0 + frac * (x1 - x0)
        val = float(norm.inverse(frac))
        ax.text(
            tx,
            y1 + 0.14,
            f"{val:.2g}",
            ha="center",
            va="top",
            fontsize=8.2,
            color=ink_col,
            zorder=4.0,
        )


def _draw_family_legend(
    ax: maxes.Axes,
    family_colors: dict[str, str],
    ink_col: str,
) -> None:
    """Draws a clean 2-column family swatch key inside the Groups 3-12 bay."""
    names = list(family_colors.keys())
    for idx, fam in enumerate(names):
        col_idx = idx // 5
        row_idx = idx % 5
        lx = 3.6 + col_idx * 4.4
        ly = 0.85 + row_idx * 0.44
        sw = patches.FancyBboxPatch(
            (lx, ly),
            0.46,
            0.30,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            facecolor=family_colors[fam],
            edgecolor=ink_col,
            linewidth=1.0,
            zorder=3.0,
        )
        ax.add_patch(sw)
        ax.text(
            lx + 0.62,
            ly + 0.15,
            fam,
            ha="left",
            va="center",
            fontsize=8.3,
            color=ink_col,
            zorder=4.0,
        )


def _draw_single_tile(
    ax: maxes.Axes,
    rec: elements.ElementRecord,
    fill_color: str,
    hatch: str | None,
    sub_text: str,
    ink_col: str,
    *,
    show_numbers: bool,
    linewidth: float = 1.25,
) -> None:
    """Draws one rounded xkcd-style periodic table element tile."""
    col, row = _tile_coords(rec)
    x0 = col - 0.44
    y0 = row - 0.44
    tile = patches.FancyBboxPatch(
        (x0, y0),
        0.88,
        0.88,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        facecolor=fill_color,
        edgecolor="#94A3B8" if hatch else ink_col,
        linewidth=linewidth,
        hatch=hatch,
        zorder=2.5,
    )
    if hatch:
        tile.set_sketch_params(0.0, 0.0, 0.0)
    ax.add_patch(tile)
    if hatch:
        ax.add_patch(
            patches.FancyBboxPatch(
                (x0, y0),
                0.88,
                0.88,
                boxstyle="round,pad=0.02,rounding_size=0.10",
                facecolor="none",
                edgecolor=ink_col,
                linewidth=linewidth,
                zorder=2.8,
            )
        )
    txt_col = (
        primitives.ensure_readable_text_color(ink_col, fill_color)
        if hatch
        else primitives.contrast_text_color(mcolors.to_hex(fill_color))
    )
    if show_numbers:
        ax.text(
            x0 + 0.07,
            y0 + 0.14,
            str(rec.atomic_number),
            ha="left",
            va="top",
            fontsize=6.6,
            color=txt_col,
            zorder=4.0,
        )
    sym_bbox = (
        {
            "boxstyle": "round,pad=0.08",
            "facecolor": fill_color,
            "edgecolor": "none",
            "alpha": 0.88,
        }
        if hatch
        else None
    )
    ax.text(
        col,
        row - 0.02,
        rec.symbol,
        ha="center",
        va="center",
        fontsize=11.5,
        color=txt_col,
        bbox=sym_bbox,
        zorder=4.0,
    )
    if sub_text:
        ax.text(
            col,
            row + 0.30,
            sub_text,
            ha="center",
            va="center",
            fontsize=6.2,
            color=txt_col,
            zorder=4.0,
        )


def _resolve_tile_appearance(
    rec: elements.ElementRecord,
    data_map: dict[str, float] | None,
    highlight_map: dict[str, str] | None,
    family_colors: dict[str, str],
    cmap_and_norm: tuple[mcolors.Colormap, mcolors.Normalize] | None,
    missing_cfg: tuple[str, str | None],
    fmt_and_mass: tuple[str, bool],
) -> tuple[str, str | None, str, float]:
    """Computes ``(fill_hex, hatch, sub_text, linewidth)`` for one element."""
    missing_color, missing_hatch = missing_cfg
    val_fmt, show_mass = fmt_and_mass
    if data_map is not None and cmap_and_norm is not None:
        cmap_obj, norm = cmap_and_norm
        if rec.symbol in data_map:
            val = data_map[rec.symbol]
            return (
                mcolors.to_hex(cmap_obj(norm(val))),
                None,
                format(val, val_fmt),
                1.25,
            )
        return missing_color, missing_hatch, "-", 1.05
    if highlight_map is not None:
        if rec.symbol in highlight_map:
            return (
                highlight_map[rec.symbol],
                None,
                f"{rec.atomic_mass:.1f}",
                2.0,
            )
        return missing_color, missing_hatch, "", 0.95
    fam = _display_family(rec)
    sub = f"{rec.atomic_mass:.1f}" if show_mass else ""
    return family_colors.get(fam, "#CBD5E1"), None, sub, 1.25


def _draw_ptable_chrome(ax: maxes.Axes, ink_col: str) -> None:
    """Draws f-block Group-3 placeholders and period/group/series labels."""
    for row_idx, span_lbl in ((6.0, "57-71"), (7.0, "89-103")):
        ax.add_patch(
            patches.FancyBboxPatch(
                (2.56, row_idx - 0.44),
                0.88,
                0.88,
                boxstyle="round,pad=0.02,rounding_size=0.10",
                facecolor="none",
                edgecolor=ink_col,
                linestyle="--",
                linewidth=1.1,
                zorder=2.2,
            )
        )
        ax.text(
            3.0,
            row_idx,
            span_lbl,
            ha="center",
            va="center",
            fontsize=7.2,
            color=ink_col,
            zorder=3.0,
        )
    for g in range(1, 19):
        ax.text(
            float(g),
            0.35,
            str(g),
            ha="center",
            va="center",
            fontsize=8.5,
            color=ink_col,
            alpha=0.75,
        )
    for p in range(1, 8):
        ax.text(
            0.28,
            float(p),
            str(p),
            ha="center",
            va="center",
            fontsize=8.5,
            color=ink_col,
            alpha=0.75,
        )
    for y_pos, tag in ((8.45, "La-Lu"), (9.55, "Ac-Lr")):
        ax.text(
            2.35,
            y_pos,
            tag,
            ha="right",
            va="center",
            fontsize=8.5,
            color=ink_col,
        )


def _prepare_ptable_data(
    values: Mapping[str, float] | None,
    element_property: str | None,
    cmap: str,
    log_scale: bool,
    cbar_label: str | None,
) -> tuple[
    dict[str, float] | None,
    tuple[mcolors.Colormap, mcolors.Normalize] | None,
    str,
]:
    """Resolves heatmap values, colormap normalization, and colorbar label."""
    data_map: dict[str, float] | None = None
    label = cbar_label or "Value"
    if values is not None:
        data_map = {
            elements.get_element(k).symbol: float(v) for k, v in values.items()
        }
    elif element_property is not None:
        data_map, auto_lbl = _extract_property_values(element_property)
        label = cbar_label or auto_lbl
    if not data_map:
        return data_map, None, label
    arr = np.array(list(data_map.values()), dtype=np.float64)
    vmin = float(np.min(arr))
    vmax = float(np.max(arr))
    if abs(vmax - vmin) < 1e-12:
        vmax = vmin + 1.0
    norm: mcolors.Normalize = (
        mcolors.LogNorm(vmin=max(vmin, 1e-12), vmax=max(vmax, 1e-11))
        if log_scale
        else mcolors.Normalize(vmin=vmin, vmax=vmax)
    )
    return data_map, (matplotlib.colormaps[cmap], norm), label


def plot_periodic_table(
    values: Mapping[str, float] | None = None,
    *,
    element_property: str | None = None,
    highlight: str | Sequence[str] | Mapping[str, str] | None = None,
    ax: maxes.Axes | None = None,
    cmap: str = "YlOrRd",
    log_scale: bool = False,
    cbar_label: str | None = None,
    missing_shade: str | None = "hatch",
    family_colors: Mapping[str, str] | None = None,
    background_grid: str | None = None,
    preset: str | theme.ThemePreset = "classic",
    title: str | None = None,
) -> maxes.Axes:
    """Plots an xkcd-style Periodic Table of Elements, heatmap, or spotlight.

    Supports three distinct workflows:

    - **Reference periodic table** (default): all 118 elements color-coded by
      chemical family with an inline legend in the Groups 3-12 cavity.
    - **Property / dataset heatmap** (``values={...}`` or ``element_property``):
      maps element data through ``cmap`` with a horizontal colorbar and optional
      hatched shading on missing elements (``missing_shade="hatch"``).
    - **Formula / element spotlight** (``highlight="LiFePO4"`` or list/dict):
      highlights selected elements vividly while muting/hatching the rest.

    Args:
      values: Optional mapping ``{element_symbol: numeric_value}``.
      element_property: Optional built-in property key (``"electronegativity"``,
        ``"covalent_radius"``, ``"vdw_radius"``, ``"atomic_mass"``,
        ``"atomic_number"``).
      highlight: Optional formula string, element list, or ``{symbol: color}``.
      ax: Optional target Matplotlib ``Axes``.
      cmap: Matplotlib colormap name for heatmap mode.
      log_scale: Whether to apply logarithmic color normalization in heatmap
        mode.
      cbar_label: Colorbar heading for heatmap mode.
      missing_shade: Shading style for unmeasured/unhighlighted tiles (e.g.
        ``"hatch"``, ``"dots"``, ``"crosshatch"``, ``"solid"``).
      family_colors: Optional mapping overriding default chemical-family colors.
      background_grid: Optional background grid (``"dots"``, ``"grid"``, etc.).
      preset: Visual theme preset name or ``ThemePreset``.
      title: Optional heading above the periodic table.

    Returns:
      Matplotlib ``Axes`` containing the periodic table.
    """
    data_map, cmap_and_norm, resolved_cbar = _prepare_ptable_data(
        values, element_property, cmap, log_scale, cbar_label
    )
    highlight_map = (
        _parse_highlight_spec(highlight) if highlight is not None else None
    )
    fam_palette = {k: v[1] for k, v in _FAMILY_DISPLAY_COLORS.items()}
    if family_colors:
        fam_palette.update({str(k): str(v) for k, v in family_colors.items()})

    preset_obj = theme.get_preset(preset)
    ink_col = preset_obj.ink_color
    muted_bg = primitives.muted_surface_color(preset_obj.axes_facecolor)
    missing_hatch = patterns.resolve_hatch(missing_shade)

    with theme.xkcd_style(preset=preset):
        if ax is None:
            _, ax = plt.subplots(figsize=(14.5, 7.6))
        else:
            theme.style_existing_axes(ax, preset=preset)
        if background_grid:
            patterns.apply_background_grid(ax, style=background_grid)

        for rec in elements.ELEMENTS_BY_Z.values():
            fill_c, hatch_c, sub_txt, lw = _resolve_tile_appearance(
                rec,
                data_map,
                highlight_map,
                fam_palette,
                cmap_and_norm,
                (muted_bg, missing_hatch),
                (".2g", True),
            )
            _draw_single_tile(
                ax,
                rec,
                fill_c,
                hatch_c,
                sub_txt,
                ink_col,
                show_numbers=True,
                linewidth=lw,
            )

        _draw_ptable_chrome(ax, ink_col)
        if cmap_and_norm is not None:
            _draw_ptable_colorbar(
                ax,
                cmap_and_norm[0],
                cmap_and_norm[1],
                resolved_cbar,
                ink_col,
            )
        elif highlight_map is None:
            _draw_family_legend(ax, fam_palette, ink_col)

        ax.set_xlim(0.0, 18.7)
        ax.set_ylim(10.25, 0.05)
        ax.set_aspect("equal")
        ax.set_axis_off()
        if title:
            ax.set_title(title, pad=8.0, color=ink_col)
    return ax
