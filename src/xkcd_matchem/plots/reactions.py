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
"""Reaction coordinate energy diagrams and NEB barrier visualization."""

from __future__ import annotations

from collections.abc import Sequence
import pathlib

from matplotlib import axes as maxes
from matplotlib import pyplot as plt
import numpy as np
from scipy import interpolate

from xkcd_matchem.core import exceptions
from xkcd_matchem.core import geometry
from xkcd_matchem.core import structure
from xkcd_matchem.core import trajectory
from xkcd_matchem.io import loader
from xkcd_matchem.render import structure2d
from xkcd_matchem.style import palettes
from xkcd_matchem.style import theme


def _is_local_peak(y_vals: np.ndarray, idx: int) -> bool:
    """Returns whether state ``idx`` is a local energy maximum."""
    if len(y_vals) < 2:
        return False
    if idx == 0:
        return bool(y_vals[0] > y_vals[1])
    if idx == len(y_vals) - 1:
        return bool(y_vals[-1] > y_vals[-2])
    return bool(
        y_vals[idx] >= y_vals[idx - 1] and y_vals[idx] >= y_vals[idx + 1]
    )


def _draw_barrier_span(
    ax: maxes.Axes,
    xs: np.ndarray,
    y_vals: np.ndarray,
    ref_idx: int,
    peak_idx: int,
    unit: str,
    barrier_col: str,
    bg_color: str,
) -> None:
    """Draws one activation-energy arrow and badge between minimum and peak."""
    e_ref = float(y_vals[ref_idx])
    e_act = float(y_vals[peak_idx] - e_ref)
    ax.annotate(
        "",
        xy=(xs[peak_idx], y_vals[peak_idx]),
        xytext=(xs[peak_idx], e_ref),
        arrowprops={"arrowstyle": "<->", "color": barrier_col, "lw": 1.6},
    )
    ax.plot(
        [xs[ref_idx] + 0.22, xs[peak_idx] + 0.10],
        [e_ref, e_ref],
        color=barrier_col,
        linestyle=":",
        lw=1.1,
    )
    ax.text(
        xs[peak_idx],
        0.5 * (e_ref + y_vals[peak_idx]),
        f"Ea = {e_act:.2f} {unit}",
        color=barrier_col,
        ha="center",
        va="center",
        fontsize=9.0,
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": bg_color,
            "edgecolor": "none",
            "alpha": 0.9,
        },
        zorder=6,
    )


def _annotate_reaction_barrier(
    ax: maxes.Axes,
    xs: np.ndarray,
    y_vals: np.ndarray,
    unit: str,
    barrier_col: str,
    bg_color: str,
) -> None:
    """Draws global activation-energy arrow from preceding minimum to max."""
    peak_idx = int(np.argmax(y_vals))
    if not 0 < peak_idx < len(y_vals) - 1:
        return
    ref_idx = int(np.argmin(y_vals[:peak_idx]))
    _draw_barrier_span(
        ax, xs, y_vals, ref_idx, peak_idx, unit, barrier_col, bg_color
    )


def _annotate_elementary_barriers(
    ax: maxes.Axes,
    xs: np.ndarray,
    y_vals: np.ndarray,
    unit: str,
    barrier_col: str,
    bg_color: str,
) -> None:
    """Draws an Ea arrow for each interior barrier relative to prior min."""
    last_min = 0
    for k in range(1, len(y_vals) - 1):
        if y_vals[k] <= y_vals[k - 1] and y_vals[k] <= y_vals[k + 1]:
            last_min = k
        elif y_vals[k] > y_vals[k - 1] and y_vals[k] >= y_vals[k + 1]:
            _draw_barrier_span(
                ax, xs, y_vals, last_min, k, unit, barrier_col, bg_color
            )
            last_min = k


def plot_reaction_profile(
    energies: Sequence[float] | np.ndarray,
    labels: Sequence[str] | None = None,
    ax: maxes.Axes | None = None,
    unit: str = "eV",
    barriers: bool = True,
    smooth_connect: bool = True,
    color: str | None = None,
    palette: str | Sequence[str] = "okabe_ito",
    preset: str | theme.ThemePreset = "classic",
    title: str | None = None,
    *,
    barrier_mode: str = "global",
) -> maxes.Axes:
    """Plots a stationary-state reaction energy diagram with barriers."""
    mode_key = "none" if not barriers else barrier_mode.strip().lower()
    if mode_key not in ("global", "elementary", "none"):
        raise exceptions.StructureError(
            f"Unsupported barrier_mode {barrier_mode!r}. "
            "Use 'global', 'elementary', or 'none'."
        )
    y_vals = np.asarray(energies, dtype=np.float64)
    n_pts = len(y_vals)
    span = max(float(np.ptp(y_vals)), 0.5)
    series = palettes.get_palette(palette)
    line_col = series[5 % len(series)] if color is None else color
    barrier_col = series[6 % len(series)]
    preset_obj = theme.get_preset(preset)
    with theme.xkcd_style(preset=preset):
        if ax is None:
            _, ax = plt.subplots(figsize=(6.4, 4.2))
        else:
            theme.style_existing_axes(ax, preset=preset)
        xs = np.arange(n_pts, dtype=np.float64)
        if smooth_connect and n_pts >= 2:
            for k in range(n_pts - 1):
                t = np.linspace(0.0, 1.0, 40)
                sx = (xs[k] + 0.22) + (0.56) * t
                sy = y_vals[k] + (y_vals[k + 1] - y_vals[k]) * (
                    3.0 * t * t - 2.0 * t * t * t
                )
                ax.plot(
                    sx, sy, color=line_col, linestyle="--", lw=1.4, alpha=0.8
                )

        for k in range(n_pts):
            ax.plot(
                [xs[k] - 0.22, xs[k] + 0.22],
                [y_vals[k], y_vals[k]],
                color=line_col,
                lw=3.2,
                solid_capstyle="round",
            )
            lbl_txt = labels[k] if (labels and k < len(labels)) else f"S{k}"
            above = _is_local_peak(y_vals, k)
            y_text = (
                y_vals[k] + 0.05 * span if above else y_vals[k] - 0.05 * span
            )
            ax.text(
                xs[k],
                y_text,
                f"{lbl_txt}\n({y_vals[k]:+.2f})",
                ha="center",
                va="bottom" if above else "top",
                fontsize=9.2,
                color=preset_obj.ink_color,
            )

        if n_pts >= 3 and mode_key == "global":
            _annotate_reaction_barrier(
                ax, xs, y_vals, unit, barrier_col, preset_obj.axes_facecolor
            )
        elif n_pts >= 3 and mode_key == "elementary":
            _annotate_elementary_barriers(
                ax, xs, y_vals, unit, barrier_col, preset_obj.axes_facecolor
            )

        ax.set_ylim(
            float(np.min(y_vals)) - 0.30 * span,
            float(np.max(y_vals)) + 0.32 * span,
        )
        ax.set_xlim(-0.55, n_pts - 0.35)
        ax.set_xticks([])
        ax.set_xlabel("Reaction Coordinate")
        ax.set_ylabel(f"Relative Energy ({unit})")
        theme.apply_xkcd_spines(ax, preset=preset)
        if title:
            ax.set_title(title)
    return ax


def _neb_coordinates(
    source: trajectory.Trajectory | Sequence[float] | np.ndarray,
    n_img: int,
) -> np.ndarray:
    """Computes normalized 3N path-length coordinates in ``[0, 1]``."""
    if not isinstance(source, trajectory.Trajectory) or n_img < 2:
        return np.linspace(0.0, 1.0, n_img)
    frames = source.frames
    if len(frames) != n_img:
        return np.linspace(0.0, 1.0, n_img)
    n_atoms = len(frames[0])
    if any(len(frame) != n_atoms for frame in frames):
        return np.linspace(0.0, 1.0, n_img)
    cum = np.zeros(n_img, dtype=np.float64)
    for idx in range(1, n_img):
        prev_f, curr_f = frames[idx - 1], frames[idx]
        disp = curr_f.positions - prev_f.positions
        if curr_f.cell is not None and any(curr_f.pbc):
            disp = geometry.minimum_image_vector(disp, curr_f.cell, curr_f.pbc)
        cum[idx] = cum[idx - 1] + float(np.linalg.norm(disp))
    if cum[-1] <= 1e-12:
        return np.linspace(0.0, 1.0, n_img)
    return np.asarray(cum / cum[-1], dtype=np.float64)


def plot_neb(
    source: trajectory.Trajectory | Sequence[float] | np.ndarray,
    ax: maxes.Axes | None = None,
    unit: str = "eV",
    annotate_ts: bool = True,
    color: str | None = None,
    palette: str | Sequence[str] = "okabe_ito",
    preset: str | theme.ThemePreset = "classic",
    title: str | None = None,
) -> maxes.Axes:
    """Plots a shape-preserving spline-interpolated NEB barrier curve."""
    if isinstance(source, trajectory.Trajectory):
        energies = source.energy_profile(relative=True, unit=unit)
    else:
        raw = np.asarray(source, dtype=np.float64)
        energies = raw - raw[0]

    series = palettes.get_palette(palette)
    line_col = series[6 % len(series)] if color is None else color
    ts_col = series[5 % len(series)]
    preset_obj = theme.get_preset(preset)
    n_img = len(energies)
    xi = _neb_coordinates(source, n_img)
    with theme.xkcd_style(preset=preset):
        if ax is None:
            _, ax = plt.subplots(figsize=(6.4, 4.2))
        else:
            theme.style_existing_axes(ax, preset=preset)
        if n_img >= 3 and bool(np.all(np.diff(xi) > 1e-12)):
            spline = interpolate.PchipInterpolator(xi, energies)
            x_fine = np.linspace(0.0, 1.0, 120)
            y_fine = spline(x_fine)
            ax.plot(x_fine, y_fine, color=line_col, lw=2.2, label="NEB MEP")
        else:
            ax.plot(xi, energies, color=line_col, lw=2.2)
        ax.scatter(
            xi,
            energies,
            s=65,
            facecolor=preset_obj.axes_facecolor,
            edgecolor=line_col,
            linewidth=2.0,
            zorder=5,
            label="NEB Images",
        )
        if annotate_ts and n_img >= 3:
            ts_idx = int(np.argmax(energies))
            ax.annotate(
                f"TS: {energies[ts_idx]:+.2f} {unit}",
                xy=(xi[ts_idx], energies[ts_idx]),
                xytext=(
                    xi[ts_idx],
                    energies[ts_idx] + 0.14 * max(float(np.ptp(energies)), 0.2),
                ),
                ha="center",
                color=ts_col,
                arrowprops={"arrowstyle": "->", "color": ts_col, "lw": 1.4},
            )
        ax.axhline(0.0, color=palettes.MUTED_GRAY, linestyle="--", lw=1.1)
        span = max(float(np.ptp(energies)), 0.4)
        ax.set_ylim(
            float(np.min(energies)) - 0.15 * span,
            float(np.max(energies)) + 0.28 * span,
        )
        ax.set_xlabel("Normalized Reaction Coordinate")
        ax.set_ylabel(f"Relative Energy ({unit})")
        theme.apply_xkcd_spines(ax, preset=preset)
        if title:
            ax.set_title(title)
    return ax


def _coerce_scheme_mols(
    items: (
        structure.AtomsData
        | str
        | pathlib.Path
        | Sequence[structure.AtomsData | str | pathlib.Path]
    ),
) -> list[structure.AtomsData]:
    """Coerces single or sequence reaction participants to ``AtomsData``."""
    seq = (
        [items]
        if isinstance(items, (structure.AtomsData, str, pathlib.Path))
        else list(items)
    )
    if not seq:
        raise exceptions.StructureError(
            "Reaction scheme requires at least one reactant and product."
        )
    out: list[structure.AtomsData] = []
    for item in seq:
        if isinstance(item, structure.AtomsData):
            out.append(item)
        else:
            loaded = loader.load(item)
            if not isinstance(loaded, structure.AtomsData):
                raise exceptions.StructureError(
                    f"Expected AtomsData, got {type(loaded).__name__}."
                )
            out.append(loaded)
    return out


def _draw_scheme_side(
    ax: maxes.Axes,
    mols: Sequence[structure.AtomsData],
    x_start: float,
    mol_w: float,
    plus_w: float,
    palette: str,
    ink_col: str,
) -> float:
    """Renders reactants or products horizontally with ``+`` separators."""
    x_cur = x_start
    for idx, mol in enumerate(mols):
        if idx > 0:
            ax.text(
                x_cur + 0.5 * plus_w,
                0.50,
                "+",
                ha="center",
                va="center",
                fontsize=16.0,
                color=ink_col,
            )
            x_cur += plus_w
        sub_ax = ax.inset_axes((x_cur, 0.12, mol_w, 0.76))
        structure2d.render_structure_2d(mol, ax=sub_ax, palette=palette)
        sub_ax.set_facecolor("none")
        sub_ax.set_axis_off()
        x_cur += mol_w
    return x_cur


def plot_reaction_scheme(
    reactants: (
        structure.AtomsData
        | str
        | pathlib.Path
        | Sequence[structure.AtomsData | str | pathlib.Path]
    ),
    products: (
        structure.AtomsData
        | str
        | pathlib.Path
        | Sequence[structure.AtomsData | str | pathlib.Path]
    ),
    *,
    conditions_above: str | None = None,
    conditions_below: str | None = None,
    arrow: str = "->",
    ax: maxes.Axes | None = None,
    palette: str = "xkcd",
    preset: str | theme.ThemePreset = "classic",
    title: str | None = None,
) -> maxes.Axes:
    """Draws an xkcd-style 2D skeletal chemical reaction equation scheme.

    Args:
      reactants: Reactant molecule or sequence of reactant structures/paths.
      products: Product molecule or sequence of product structures/paths.
      conditions_above: Optional reagents/catalyst text above reaction arrow.
      conditions_below: Optional temperature/solvent text below arrow.
      arrow: ``"->"`` for forward or ``"<=>"`` for equilibrium double arrow.
      ax: Optional host Matplotlib ``Axes``.
      palette: Element coloring scheme for 2D heteroatom labels.
      preset: Visual theme preset name or ``ThemePreset``.
      title: Optional scheme heading.

    Returns:
      Matplotlib ``Axes`` containing the horizontal reaction equation.
    """
    r_mols = _coerce_scheme_mols(reactants)
    p_mols = _coerce_scheme_mols(products)
    arrow_key = arrow.strip()
    if arrow_key not in ("->", "<=>"):
        raise exceptions.StructureError(
            f"Unsupported reaction arrow {arrow!r}. Use '->' or '<=>'."
        )
    ink_col = theme.get_preset(preset).ink_color
    n_mols = len(r_mols) + len(p_mols)
    n_plus = max(len(r_mols) - 1, 0) + max(len(p_mols) - 1, 0)
    plus_w = 0.055
    arrow_w = 0.16
    mol_w = (0.94 - n_plus * plus_w - arrow_w) / float(n_mols)

    with theme.xkcd_style(preset=preset):
        if ax is None:
            fig_w = max(7.2, 2.2 * (n_mols + 1))
            _, ax = plt.subplots(figsize=(fig_w, 3.6))
        else:
            theme.style_existing_axes(ax, preset=preset)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_axis_off()

        x_cur = _draw_scheme_side(
            ax, r_mols, 0.03, mol_w, plus_w, palette, ink_col
        )
        a_x0 = x_cur + 0.02
        a_x1 = x_cur + arrow_w - 0.02
        arrow_style = "<->" if arrow_key == "<=>" else "->"
        ax.annotate(
            "",
            xy=(a_x1, 0.50),
            xytext=(a_x0, 0.50),
            arrowprops={
                "arrowstyle": arrow_style,
                "color": ink_col,
                "lw": 2.2,
            },
        )
        mid_a = 0.5 * (a_x0 + a_x1)
        if conditions_above:
            ax.text(
                mid_a,
                0.56,
                conditions_above,
                ha="center",
                va="bottom",
                fontsize=9.5,
                color=ink_col,
            )
        if conditions_below:
            ax.text(
                mid_a,
                0.44,
                conditions_below,
                ha="center",
                va="top",
                fontsize=9.5,
                color=ink_col,
            )
        _draw_scheme_side(
            ax, p_mols, x_cur + arrow_w, mol_w, plus_w, palette, ink_col
        )
        if title:
            ax.set_title(title)
    return ax
