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
"""Theme context manager and visual presets for xkcd-matchem."""

from __future__ import annotations

from collections.abc import Iterator
from collections.abc import Mapping
import contextlib
import dataclasses
from typing import Any

import matplotlib
from matplotlib import axes as maxes
from matplotlib import pyplot as plt

from xkcd_matchem.core import exceptions
from xkcd_matchem.style import fonts
from xkcd_matchem.style import palettes


@dataclasses.dataclass(frozen=True)
class ThemePreset:
    """Immutable styling parameters defining an xkcd-matchem visual theme.

    Attributes:
      name: Preset identifier.
      font: Primary handwritten or sans-serif font family name.
      sketch_scale: Matplotlib ``path.sketch`` wiggle amplitude.
      sketch_length: Matplotlib ``path.sketch`` wiggle wavelength.
      sketch_randomness: Matplotlib ``path.sketch`` randomness factor.
      figure_facecolor: Background color of the outer Figure canvas.
      axes_facecolor: Background color inside plot Axes.
      ink_color: Primary stroke, spine, and label color.
      grid: Whether gridlines are enabled by default.
      grid_alpha: Gridline opacity when enabled.
      linewidth: Default line width in points.
      font_size: Base font size in points.
      title_size: Axes title font size in points.
      dpi: Default figure resolution.
    """

    name: str
    font: str
    sketch_scale: float
    sketch_length: float
    sketch_randomness: float
    figure_facecolor: str
    axes_facecolor: str
    ink_color: str
    grid: bool
    grid_alpha: float
    linewidth: float
    font_size: float
    title_size: float
    dpi: int


PRESETS: dict[str, ThemePreset] = {
    "classic": ThemePreset(
        name="classic",
        font="Patrick Hand",
        sketch_scale=1.0,
        sketch_length=100.0,
        sketch_randomness=2.0,
        figure_facecolor=palettes.PAPER,
        axes_facecolor=palettes.PAPER,
        ink_color=palettes.INK,
        grid=False,
        grid_alpha=0.25,
        linewidth=1.8,
        font_size=12.0,
        title_size=14.0,
        dpi=160,
    ),
    "notebook": ThemePreset(
        name="notebook",
        font="Patrick Hand",
        sketch_scale=0.85,
        sketch_length=110.0,
        sketch_randomness=1.8,
        figure_facecolor=palettes.PAPER,
        axes_facecolor="#FFFFFF",
        ink_color=palettes.INK,
        grid=True,
        grid_alpha=0.35,
        linewidth=1.7,
        font_size=11.0,
        title_size=13.0,
        dpi=140,
    ),
    "poster": ThemePreset(
        name="poster",
        font="Comic Neue",
        sketch_scale=0.75,
        sketch_length=120.0,
        sketch_randomness=1.6,
        figure_facecolor="#FFFFFF",
        axes_facecolor="#FFFFFF",
        ink_color=palettes.INK,
        grid=False,
        grid_alpha=0.2,
        linewidth=2.2,
        font_size=13.0,
        title_size=16.0,
        dpi=200,
    ),
    "minimal": ThemePreset(
        name="minimal",
        font="Comic Neue",
        sketch_scale=0.35,
        sketch_length=140.0,
        sketch_randomness=1.2,
        figure_facecolor="#FFFFFF",
        axes_facecolor="#FFFFFF",
        ink_color=palettes.INK,
        grid=False,
        grid_alpha=0.2,
        linewidth=1.5,
        font_size=11.0,
        title_size=13.0,
        dpi=160,
    ),
    "dark_chalk": ThemePreset(
        name="dark_chalk",
        font="Patrick Hand",
        sketch_scale=1.05,
        sketch_length=95.0,
        sketch_randomness=2.1,
        figure_facecolor=palettes.DARK_BOARD,
        axes_facecolor=palettes.DARK_BOARD,
        ink_color=palettes.DARK_CHALK,
        grid=True,
        grid_alpha=0.18,
        linewidth=1.9,
        font_size=12.0,
        title_size=14.0,
        dpi=160,
    ),
}


def list_presets() -> tuple[str, ...]:
    """Returns the sorted names of available visual theme presets."""
    return tuple(sorted(PRESETS))


def get_preset(name: str = "classic") -> ThemePreset:
    """Retrieves a ``ThemePreset`` by name.

    Args:
      name: Preset key (``"classic"``, ``"notebook"``, ``"poster"``,
        ``"minimal"``, ``"dark_chalk"``).

    Returns:
      Matching ``ThemePreset`` instance.

    Raises:
      StyleError: If ``name`` is not a registered preset.
    """
    key = name.lower()
    if key not in PRESETS:
        available = ", ".join(list_presets())
        raise exceptions.StyleError(
            f"Unknown theme preset {name!r}. Available: {available}"
        )
    return PRESETS[key]


def build_rc_params(
    preset: str = "classic",
    font: str | None = None,
    scale: float | None = None,
    length: float | None = None,
    randomness: float | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Constructs a Matplotlib ``rcParams`` dictionary without mutating globals.

    Sets empty ``path.effects`` to prevent Matplotlib white outline artifacts,
    embeds TrueType fonts for PDF (``pdf.fonttype=42``), converts SVG text to
    paths (``svg.fonttype="path"``), and enables ``DejaVu Sans`` plus STIX
    sans fallbacks for Unicode and mathtext chemical symbols.

    Args:
      preset: Name of the base theme preset.
      font: Optional font family override.
      scale: Optional ``path.sketch`` wiggle amplitude override.
      length: Optional ``path.sketch`` wiggle length override.
      randomness: Optional ``path.sketch`` randomness override.
      overrides: Additional raw Matplotlib rcParams overrides.

    Returns:
      Dictionary suitable for ``matplotlib.rc_context``.
    """
    theme = get_preset(preset)
    chosen_font = font if font is not None else theme.font
    sketch = (
        theme.sketch_scale if scale is None else scale,
        theme.sketch_length if length is None else length,
        theme.sketch_randomness if randomness is None else randomness,
    )
    font_chain = fonts.resolve_font_chain(chosen_font)
    rc: dict[str, Any] = {
        "font.family": font_chain,
        "font.size": theme.font_size,
        "axes.titlesize": theme.title_size,
        "axes.labelsize": theme.font_size,
        "axes.titleweight": "normal",
        "axes.titlepad": 10.0,
        "axes.edgecolor": theme.ink_color,
        "axes.facecolor": theme.axes_facecolor,
        "axes.labelcolor": theme.ink_color,
        "axes.linewidth": theme.linewidth,
        "axes.grid": theme.grid,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.unicode_minus": False,
        "axes.prop_cycle": matplotlib.cycler(color=list(palettes.OKABE_ITO)),
        "grid.color": palettes.MUTED_GRAY,
        "grid.alpha": theme.grid_alpha,
        "grid.linestyle": "--",
        "xtick.color": theme.ink_color,
        "ytick.color": theme.ink_color,
        "xtick.major.width": max(1.0, theme.linewidth * 0.8),
        "ytick.major.width": max(1.0, theme.linewidth * 0.8),
        "text.color": theme.ink_color,
        "figure.facecolor": theme.figure_facecolor,
        "figure.dpi": theme.dpi,
        "savefig.facecolor": theme.figure_facecolor,
        "savefig.dpi": theme.dpi,
        "lines.linewidth": theme.linewidth,
        "patch.edgecolor": theme.ink_color,
        "path.sketch": sketch,
        "path.effects": [],
        "mathtext.fontset": "dejavusans",
        "mathtext.fallback": "stixsans",
        "pdf.fonttype": 42,
        "svg.fonttype": "path",
    }
    if overrides is not None:
        rc.update(overrides)
    return rc


@contextlib.contextmanager
def xkcd_style(
    preset: str = "classic",
    font: str | None = None,
    scale: float | None = None,
    length: float | None = None,
    randomness: float | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> Iterator[None]:
    """Context manager applying non-leaking hand-drawn xkcd-matchem styling.

    Args:
      preset: Base preset name (``"classic"``, ``"notebook"``, ``"poster"``,
        ``"minimal"``, ``"dark_chalk"``).
      font: Optional font family override (default comes from preset).
      scale: Optional sketch wiggle amplitude.
      length: Optional sketch wiggle wavelength.
      randomness: Optional sketch wiggle randomness.
      overrides: Extra Matplotlib rcParams entries applied within context.

    Yields:
      None while the styled ``matplotlib.rc_context`` is active.
    """
    rc = build_rc_params(
        preset=preset,
        font=font,
        scale=scale,
        length=length,
        randomness=randomness,
        overrides=overrides,
    )
    with plt.rc_context(rc):
        yield


def apply_xkcd_spines(
    ax: maxes.Axes,
    ink_color: str | None = None,
    show_top: bool = False,
    show_right: bool = False,
    linewidth: float = 1.6,
) -> maxes.Axes:
    """Configures crisp hand-drawn axis spines on an existing Matplotlib Axes.

    Args:
      ax: Target Matplotlib ``Axes``.
      ink_color: Optional stroke color; keeps existing spine color if ``None``.
      show_top: Whether to keep the top spine visible.
      show_right: Whether to keep the right spine visible.
      linewidth: Spine stroke width in points.

    Returns:
      The modified ``Axes`` instance.
    """
    ax.spines["top"].set_visible(show_top)
    ax.spines["right"].set_visible(show_right)
    for side in ("bottom", "left", "top", "right"):
        spine = ax.spines[side]
        spine.set_linewidth(linewidth)
        if ink_color is not None:
            spine.set_edgecolor(ink_color)
    return ax
