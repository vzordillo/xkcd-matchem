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
from typing import Any

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
    ink_col = str(plt.rcParams.get("axes.edgecolor", palettes.INK))
    ax.annotate(
        "",
        xy=(xs[peak_idx], y_vals[peak_idx]),
        xytext=(xs[peak_idx], e_ref),
        arrowprops={"arrowstyle": "<->", "color": ink_col, "lw": 1.10},
    )
    ref_line = ax.plot(
        [xs[ref_idx] + 0.22, xs[peak_idx] + 0.10],
        [e_ref, e_ref],
        color=barrier_col,
        linestyle=":",
        lw=0.95,
    )[0]
    ref_line.set_sketch_params(0.0, 0.0, 0.0)
    txt = ax.text(
        xs[peak_idx],
        0.5 * (e_ref + y_vals[peak_idx]),
        f"Ea = {e_act:.2f} {unit}",
        color=ink_col,
        ha="center",
        va="center",
        fontsize=9.0,
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": bg_color,
            "edgecolor": "none",
            "alpha": 0.90,
        },
        zorder=6,
    )
    patch = txt.get_bbox_patch()
    if patch is not None:
        patch.set_sketch_params(0.0, 0.0, 0.0)


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
    sw = theme.stroke_weights(preset)
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
                    sx,
                    sy,
                    color=line_col,
                    linestyle="--",
                    lw=sw.overlay,
                    alpha=0.78,
                )

        for k in range(n_pts):
            ax.plot(
                [xs[k] - 0.22, xs[k] + 0.22],
                [y_vals[k], y_vals[k]],
                color=line_col,
                lw=2.8,
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
    preset_obj = theme.get_preset(preset)
    ink_col = preset_obj.ink_color
    sw = theme.stroke_weights(preset)
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
            ax.plot(
                x_fine, y_fine, color=line_col, lw=sw.primary, label="NEB MEP"
            )
        else:
            ax.plot(xi, energies, color=line_col, lw=sw.primary)
        ax.scatter(
            xi,
            energies,
            s=58,
            facecolor=preset_obj.axes_facecolor,
            edgecolor=line_col,
            linewidth=sw.primary,
            zorder=5,
            label="NEB Images",
        )
        if annotate_ts and n_img >= 3:
            ts_idx = int(np.argmax(energies))
            ann = ax.annotate(
                f"TS: {energies[ts_idx]:+.2f} {unit}",
                xy=(xi[ts_idx], energies[ts_idx]),
                xytext=(
                    xi[ts_idx],
                    energies[ts_idx] + 0.14 * max(float(np.ptp(energies)), 0.2),
                ),
                ha="center",
                color=ink_col,
                bbox=theme.label_halo_bbox(
                    preset_obj.axes_facecolor, pad=0.10, alpha=0.86
                ),
                arrowprops={
                    "arrowstyle": "->",
                    "color": ink_col,
                    "lw": sw.leader,
                },
            )
            patch = ann.get_bbox_patch()
            if patch is not None:
                patch.set_sketch_params(0.0, 0.0, 0.0)
        z_line = ax.axhline(
            0.0, color=palettes.MUTED_GRAY, linestyle="--", lw=sw.guide
        )
        z_line.set_sketch_params(0.0, 0.0, 0.0)
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


def _estimate_condition_len(
    conditions_above: str | None,
    conditions_below: str | None,
) -> int:
    """Estimates plain character length of condition text for arrow width."""
    lengths = [0]
    for raw in (conditions_above, conditions_below):
        if not raw:
            continue
        cleaned = (
            raw.replace("$", "")
            .replace("\\mathrm{", "")
            .replace("\\circ", "o")
            .replace("{", "")
            .replace("}", "")
            .replace("\\,", " ")
            .replace("\\ ", " ")
            .replace("_", "")
            .replace("^", "")
        )
        lengths.append(len(cleaned))
    return max(lengths)


def _draw_xkcd_reaction_arrow(
    ax: maxes.Axes,
    x0: float,
    x1: float,
    arrow_key: str,
    ink_col: str,
    lw: float,
) -> None:
    """Draws hand-inked forward or equilibrium arrow matching skeletal lw."""
    head_l = min(0.38, 0.14 * max(x1 - x0, 1.0))
    head_h = 0.56 * head_l
    if arrow_key == "<=>":
        y_top, y_bot = 0.14, -0.14
        ax.plot(
            [x0, x1, x1 - head_l],
            [y_top, y_top, y_top + head_h],
            color=ink_col,
            lw=lw,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=3,
        )
        ax.plot(
            [x1, x0, x0 + head_l],
            [y_bot, y_bot, y_bot - head_h],
            color=ink_col,
            lw=lw,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=3,
        )
        return
    ax.plot(
        [x0, x1],
        [0.0, 0.0],
        color=ink_col,
        lw=lw,
        solid_capstyle="round",
        zorder=3,
    )
    ax.plot(
        [x1 - head_l, x1, x1 - head_l],
        [head_h, 0.0, -head_h],
        color=ink_col,
        lw=lw,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=3,
    )


def _layout_scheme_participants(
    r_proj: list[Any],
    p_proj: list[Any],
    arrow_len: float,
    plus_gap: float = 1.65,
    arrow_pad: float = 0.90,
) -> tuple[list[float], list[float], list[float], float, float, float]:
    """Computes horizontal Angstrom anchors for reactants, ``+``, and arrow."""
    x_cur = 0.0
    r_offs: list[float] = []
    plus_xs: list[float] = []
    for idx, proj in enumerate(r_proj):
        if idx > 0:
            plus_xs.append(x_cur + 0.5 * plus_gap)
            x_cur += plus_gap
        r_offs.append(x_cur)
        x_cur += max(float(proj[5][0]), 0.8)

    a_x0 = x_cur + arrow_pad
    a_x1 = a_x0 + arrow_len
    x_cur = a_x1 + arrow_pad

    p_offs: list[float] = []
    for idx, proj in enumerate(p_proj):
        if idx > 0:
            plus_xs.append(x_cur + 0.5 * plus_gap)
            x_cur += plus_gap
        p_offs.append(x_cur)
        x_cur += max(float(proj[5][0]), 0.8)
    return r_offs, p_offs, plus_xs, a_x0, a_x1, max(x_cur, 4.0)


def _draw_scheme_group(
    ax: maxes.Axes,
    mols: Sequence[structure.AtomsData],
    projs: list[Any],
    offsets: list[float],
    metrics: tuple[float, float, float, float],
    lw_scale: float,
    palette: str,
) -> tuple[list[float], list[float]]:
    """Draws one side of molecules and returns accumulated ``(xs, ys)``."""
    x_vals: list[float] = []
    y_vals: list[float] = []
    for mol, proj, x_off in zip(mols, projs, offsets, strict=True):
        bx0, bx1, by0, by1 = structure2d.draw_projected_molecule_2d(
            ax,
            mol,
            proj,
            offset=(x_off, 0.0),
            metrics=metrics,
            lw_scale=lw_scale,
            palette=palette,
        )
        x_vals.extend((bx0, bx1))
        y_vals.extend((by0, by1))
    return x_vals, y_vals


def _render_scheme_arrows_and_labels(
    ax: maxes.Axes,
    plus_xs: Sequence[float],
    arrow_span: tuple[float, float, str],
    conditions: tuple[str | None, str | None],
    ink_col: str,
    fs: float,
    lw_scale: float,
) -> None:
    """Draws ``+`` separators, xkcd reaction arrow, and above/below text."""
    for x_plus in plus_xs:
        ax.text(
            x_plus,
            0.0,
            "+",
            ha="center",
            va="center",
            fontsize=fs * 1.28,
            color=ink_col,
            zorder=4,
        )
    a_x0, a_x1, arrow_key = arrow_span
    _draw_xkcd_reaction_arrow(
        ax, a_x0, a_x1, arrow_key, ink_col, 1.85 * lw_scale
    )
    mid_a = 0.5 * (a_x0 + a_x1)
    cond_fs = max(8.2, 0.96 * fs)
    cond_above, cond_below = conditions
    if cond_above:
        ax.text(
            mid_a,
            0.34,
            cond_above,
            ha="center",
            va="bottom",
            fontsize=cond_fs,
            color=ink_col,
        )
    if cond_below:
        ax.text(
            mid_a,
            -0.34,
            cond_below,
            ha="center",
            va="top",
            fontsize=cond_fs,
            color=ink_col,
        )


def _draw_scheme_canvas(
    ax: maxes.Axes,
    groups: tuple[
        list[structure.AtomsData],
        list[structure.AtomsData],
        list[Any],
        list[Any],
    ],
    layout: tuple[list[float], list[float], list[float], float, float, float],
    max_h: float,
    style_cfg: tuple[str, str | None, str | None, str, str],
) -> None:
    """Renders molecules, hand-drawn arrow, and bounds onto unified axes."""
    r_mols, p_mols, r_proj, p_proj = groups
    r_offs, p_offs, plus_xs, a_x0, a_x1, total_w = layout
    arrow_key, cond_above, cond_below, ink_col, palette = style_cfg
    raw_m = structure2d.resolve_span_label_metrics(
        ax, total_w, max_h, pad=0.75
    )
    metrics = raw_m[:4]
    lw_scale = float(np.clip(raw_m[4] / 24.0, 0.74, 1.0))
    r_bounds = _draw_scheme_group(
        ax, r_mols, r_proj, r_offs, metrics, lw_scale, palette
    )
    p_bounds = _draw_scheme_group(
        ax, p_mols, p_proj, p_offs, metrics, lw_scale, palette
    )
    _render_scheme_arrows_and_labels(
        ax,
        plus_xs,
        (a_x0, a_x1, arrow_key),
        (cond_above, cond_below),
        ink_col,
        raw_m[0],
        lw_scale,
    )
    x_all = [0.0, total_w, *r_bounds[0], *p_bounds[0]]
    y_all = [-1.1, 1.1, *r_bounds[1], *p_bounds[1]]
    ax.set_xlim(min(x_all) - 0.75, max(x_all) + 0.75)
    ax.set_ylim(min(y_all) - 0.85, max(y_all) + 0.85)
    ax.set_aspect("equal")
    ax.set_axis_off()


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

    All participants render inside a single unified Angstrom coordinate space
    so ring sizes, bond lengths, and stroke weights stay strictly consistent
    across reactants and products.

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
    r_proj = [structure2d.project_molecule_2d(m) for m in r_mols]
    p_proj = [structure2d.project_molecule_2d(m) for m in p_mols]
    cond_len = _estimate_condition_len(conditions_above, conditions_below)
    layout = _layout_scheme_participants(
        r_proj, p_proj, max(3.2, 0.28 * float(cond_len) + 1.1)
    )
    max_h = max([float(p[5][1]) for p in r_proj + p_proj] + [2.4])

    with theme.xkcd_style(preset=preset):
        if ax is None:
            n_parts = len(r_mols) + len(p_mols)
            fig_w = max(7.4, min(14.0, 0.45 * layout[5] + 1.5 * n_parts))
            _, ax = plt.subplots(figsize=(fig_w, 3.6))
        else:
            theme.style_existing_axes(ax, preset=preset)
        _draw_scheme_canvas(
            ax,
            (r_mols, p_mols, r_proj, p_proj),
            layout,
            max_h,
            (
                arrow_key,
                conditions_above,
                conditions_below,
                theme.get_preset(preset).ink_color,
                palette,
            ),
        )
        if title:
            ax.set_title(title)
    return ax
