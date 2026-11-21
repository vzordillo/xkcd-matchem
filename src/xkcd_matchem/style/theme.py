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
"""Theme context manager, default presets, and full style customization."""

from __future__ import annotations

from collections.abc import Iterator
from collections.abc import Mapping
from collections.abc import Sequence
import contextlib
import dataclasses
from typing import Any

import matplotlib
from matplotlib import axes as maxes
from matplotlib import pyplot as plt

from xkcd_matchem.core import exceptions
from xkcd_matchem.style import fonts
from xkcd_matchem.style import palettes
from xkcd_matchem.style import patterns


@dataclasses.dataclass(frozen=True)
class ThemePreset:
    """Immutable styling parameters defining an xkcd-matchem visual theme.

    Attributes:
      name: Preset identifier.
      font: Primary comic-style or sans-serif font family name.
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
        font="xkcd Script",
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
        font="xkcd Script",
        sketch_scale=0.85,
        sketch_length=110.0,
        sketch_randomness=1.8,
        figure_facecolor=palettes.PAPER,
        axes_facecolor=palettes.PAPER,
        ink_color=palettes.INK,
        grid=True,
        grid_alpha=0.30,
        linewidth=1.7,
        font_size=11.0,
        title_size=13.0,
        dpi=140,
    ),
    "poster": ThemePreset(
        name="poster",
        font="xkcd Script",
        sketch_scale=0.8,
        sketch_length=115.0,
        sketch_randomness=1.7,
        figure_facecolor=palettes.PAPER,
        axes_facecolor=palettes.PAPER,
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
        font="xkcd Script",
        sketch_scale=0.45,
        sketch_length=135.0,
        sketch_randomness=1.3,
        figure_facecolor=palettes.PAPER,
        axes_facecolor=palettes.PAPER,
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
        font="xkcd Script",
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
    "blueprint": ThemePreset(
        name="blueprint",
        font="xkcd Script",
        sketch_scale=0.85,
        sketch_length=110.0,
        sketch_randomness=1.7,
        figure_facecolor=patterns.BLUEPRINT_BG,
        axes_facecolor=patterns.BLUEPRINT_BG,
        ink_color=patterns.BLUEPRINT_INK,
        grid=True,
        grid_alpha=0.28,
        linewidth=1.8,
        font_size=12.0,
        title_size=14.0,
        dpi=160,
    ),
}

_CUSTOM_PRESETS: dict[str, ThemePreset] = {}
_ACTIVE_FONT_STACK: list[str] = []


def _resolve_active_font(
    preset_obj: ThemePreset,
    explicit_font: str | None = None,
) -> str:
    """Resolves font: explicit > active context > session default > preset."""
    if explicit_font is not None:
        return explicit_font
    if _ACTIVE_FONT_STACK:
        return _ACTIVE_FONT_STACK[-1]
    session_font = fonts.get_default_font()
    if session_font is not None:
        return session_font
    return preset_obj.font


def register_preset(
    name: str,
    preset: ThemePreset | None = None,
    base: str = "classic",
    **overrides: Any,
) -> ThemePreset:
    """Registers a custom theme preset derived from a base preset or object.

    Args:
      name: Preset key to register or update.
      preset: Optional pre-built ``ThemePreset`` instance.
      base: Base preset name when ``preset`` is ``None``.
      **overrides: ``ThemePreset`` fields to override (e.g., ``font``,
        ``sketch_scale``, ``figure_facecolor``, ``ink_color``).

    Returns:
      The registered ``ThemePreset`` instance.

    Raises:
      StyleError: If ``name`` is blank or an unknown field is given.
    """
    key = name.strip().lower()
    if not key:
        raise exceptions.StyleError("Theme preset name cannot be empty.")
    template = get_preset(base) if preset is None else preset
    try:
        built = dataclasses.replace(template, name=key, **overrides)
    except TypeError as exc:
        raise exceptions.StyleError(
            f"Invalid ThemePreset field override for {key!r}: {exc}"
        ) from exc
    _CUSTOM_PRESETS[key] = built
    return built


def list_presets() -> tuple[str, ...]:
    """Returns built-in and custom visual theme preset names."""
    return tuple(sorted(set(PRESETS) | set(_CUSTOM_PRESETS)))


def get_preset(name: str | ThemePreset = "classic") -> ThemePreset:
    """Resolves a ``ThemePreset`` from a preset name or existing instance.

    Args:
      name: Preset key (``"classic"``, ``"notebook"``, ``"poster"``,
        ``"minimal"``, ``"dark_chalk"``, or custom registered name) or an
        explicit ``ThemePreset``.

    Returns:
      Matching ``ThemePreset`` instance.

    Raises:
      StyleError: If ``name`` is not a registered preset.
    """
    if isinstance(name, ThemePreset):
        return name
    key = name.lower()
    if key in _CUSTOM_PRESETS:
        return _CUSTOM_PRESETS[key]
    if key not in PRESETS:
        available = ", ".join(list_presets())
        raise exceptions.StyleError(
            f"Unknown theme preset {name!r}. Available: {available}"
        )
    return PRESETS[key]


def build_rc_params(
    preset: str | ThemePreset = "classic",
    font: str | None = None,
    scale: float | None = None,
    length: float | None = None,
    randomness: float | None = None,
    palette: str | Sequence[str] | None = None,
    background: str | None = None,
    ink: str | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Constructs a Matplotlib ``rcParams`` dictionary without mutating globals.

    Configures ``path.effects=[]`` to avoid Matplotlib's white outline artifact,
    embeds TrueType fonts for PDF, converts SVG text to paths, aligns mathtext
    with the selected comic-style typeface, and keeps ``DejaVu Sans`` plus
    ``stixsans`` as scientific symbol fallbacks.

    Args:
      preset: Base theme preset name or ``ThemePreset`` instance.
      font: Optional font family override.
      scale: Optional ``path.sketch`` wiggle amplitude override.
      length: Optional ``path.sketch`` wiggle length override.
      randomness: Optional ``path.sketch`` randomness override.
      palette: Optional discrete series palette name or sequence of colors.
      background: Optional figure/axes facecolor override.
      ink: Optional ink/spine/label color override.
      overrides: Additional raw Matplotlib rcParams entries.

    Returns:
      Dictionary suitable for ``matplotlib.rc_context``.
    """
    theme = get_preset(preset)
    chosen_font = _resolve_active_font(theme, font)
    bg_fig = theme.figure_facecolor if background is None else background
    bg_ax = theme.axes_facecolor if background is None else background
    ink_col = theme.ink_color if ink is None else ink
    sketch = (
        theme.sketch_scale if scale is None else scale,
        theme.sketch_length if length is None else length,
        theme.sketch_randomness if randomness is None else randomness,
    )
    font_chain = fonts.resolve_font_chain(chosen_font)
    series_colors = list(
        palettes.get_palette("okabe_ito" if palette is None else palette)
    )
    rc: dict[str, Any] = {
        "font.family": font_chain,
        "font.size": theme.font_size,
        "axes.titlesize": theme.title_size,
        "axes.labelsize": theme.font_size,
        "axes.titleweight": "normal",
        "axes.titlepad": 10.0,
        "axes.edgecolor": ink_col,
        "axes.facecolor": bg_ax,
        "axes.labelcolor": ink_col,
        "axes.linewidth": theme.linewidth,
        "axes.grid": theme.grid,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.unicode_minus": False,
        "axes.prop_cycle": matplotlib.cycler(color=series_colors),
        "grid.color": palettes.MUTED_GRAY,
        "grid.alpha": theme.grid_alpha,
        "grid.linestyle": "--",
        "xtick.color": ink_col,
        "ytick.color": ink_col,
        "xtick.labelsize": theme.font_size,
        "ytick.labelsize": theme.font_size,
        "xtick.major.width": max(1.0, theme.linewidth * 0.8),
        "ytick.major.width": max(1.0, theme.linewidth * 0.8),
        "text.color": ink_col,
        "figure.facecolor": bg_fig,
        "figure.edgecolor": bg_fig,
        "figure.dpi": theme.dpi,
        "savefig.facecolor": bg_fig,
        "savefig.edgecolor": bg_fig,
        "savefig.dpi": theme.dpi,
        "lines.linewidth": theme.linewidth,
        "patch.edgecolor": ink_col,
        "path.sketch": sketch,
        "path.effects": [],
        "mathtext.fontset": "custom",
        "mathtext.rm": chosen_font,
        "mathtext.it": chosen_font,
        "mathtext.bf": chosen_font,
        "mathtext.sf": chosen_font,
        "mathtext.default": "regular",
        "mathtext.fallback": "stixsans",
        "pdf.fonttype": 42,
        "svg.fonttype": "path",
    }
    if overrides is not None:
        rc.update(overrides)
    return rc


@contextlib.contextmanager
def xkcd_style(
    preset: str | ThemePreset = "classic",
    font: str | None = None,
    scale: float | None = None,
    length: float | None = None,
    randomness: float | None = None,
    palette: str | Sequence[str] | None = None,
    background: str | None = None,
    ink: str | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> Iterator[None]:
    """Context manager applying non-leaking xkcd-matchem styling.

    Args:
      preset: Base preset name or custom ``ThemePreset``.
      font: Optional font family override.
      scale: Optional sketch wiggle amplitude.
      length: Optional sketch wiggle wavelength.
      randomness: Optional sketch wiggle randomness.
      palette: Optional series palette name or color sequence.
      background: Optional background facecolor override.
      ink: Optional ink stroke/label color override.
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
        palette=palette,
        background=background,
        ink=ink,
        overrides=overrides,
    )
    if font is not None:
        _ACTIVE_FONT_STACK.append(font)
    try:
        with plt.rc_context(rc):
            yield
    finally:
        if font is not None:
            _ACTIVE_FONT_STACK.pop()


def style_existing_axes(
    ax: maxes.Axes,
    preset: str | ThemePreset = "classic",
) -> maxes.Axes:
    """Applies preset fonts, ink colors, and sketch parameters onto an Axes.

    Useful when callers pass pre-created ``fig, ax = plt.subplots(...)`` from
    outside an ``xkcd_style()`` context so ticks, labels, and spines match
    uniformly in the same comic-style typeface.

    Args:
      ax: Target Matplotlib ``Axes`` (mutated in place).
      preset: Preset name or ``ThemePreset`` to apply.

    Returns:
      The updated ``Axes`` instance.
    """
    resolved = get_preset(preset)
    font_chain = fonts.resolve_font_chain(_resolve_active_font(resolved))
    ax.set_facecolor(resolved.axes_facecolor)
    ax.patch.set_sketch_params(0.0, 0.0, 0.0)
    if ax.figure is not None:
        ax.figure.patch.set_facecolor(resolved.figure_facecolor)
        ax.figure.patch.set_edgecolor(resolved.figure_facecolor)
        ax.figure.patch.set_sketch_params(0.0, 0.0, 0.0)
    for spine in ax.spines.values():
        spine.set_linewidth(resolved.linewidth)
        spine.set_edgecolor(resolved.ink_color)
        spine.set_sketch_params(
            resolved.sketch_scale,
            resolved.sketch_length,
            resolved.sketch_randomness,
        )
    ax.tick_params(
        colors=resolved.ink_color,
        width=max(1.0, resolved.linewidth * 0.8),
        labelsize=resolved.font_size,
    )
    for label_artist in (
        list(ax.get_xticklabels())
        + list(ax.get_yticklabels())
        + [ax.xaxis.label, ax.yaxis.label, ax.title]
    ):
        label_artist.set_fontfamily(font_chain)
        label_artist.set_color(resolved.ink_color)
    return ax


def apply_xkcd_spines(
    ax: maxes.Axes,
    ink_color: str | None = None,
    show_top: bool = False,
    show_right: bool = False,
    linewidth: float = 1.6,
    preset: str | ThemePreset | None = "classic",
) -> maxes.Axes:
    """Configures crisp xkcd-style axis spines and consistent ticks on an Axes.

    Args:
      ax: Target Matplotlib ``Axes``.
      ink_color: Optional stroke color; keeps existing spine color if ``None``.
      show_top: Whether to keep the top spine visible.
      show_right: Whether to keep the right spine visible.
      linewidth: Spine stroke width in points.
      preset: Optional preset used to align existing ticks and fonts.

    Returns:
      The modified ``Axes`` instance.
    """
    if preset is not None:
        style_existing_axes(ax, preset=preset)
    ax.spines["top"].set_visible(show_top)
    ax.spines["right"].set_visible(show_right)
    for side in ("bottom", "left", "top", "right"):
        spine = ax.spines[side]
        spine.set_linewidth(linewidth)
        if ink_color is not None:
            spine.set_edgecolor(ink_color)
    return ax
