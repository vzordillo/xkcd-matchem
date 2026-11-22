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
"""Crystal-field d-orbital splitting and Hund/pairing occupation diagrams."""

from __future__ import annotations

from collections.abc import Sequence

from matplotlib import axes as maxes
from matplotlib import pyplot as plt

from xkcd_matchem.core import exceptions
from xkcd_matchem.style import layout
from xkcd_matchem.style import palettes
from xkcd_matchem.style import theme


_GEOMETRY_ALIASES: dict[str, str] = {
    "octahedral": "octahedral",
    "oh": "octahedral",
    "tetrahedral": "tetrahedral",
    "td": "tetrahedral",
    "square_planar": "square_planar",
    "square-planar": "square_planar",
    "d4h": "square_planar",
    "trigonal_bipyramidal": "trigonal_bipyramidal",
    "trigonal-bipyramidal": "trigonal_bipyramidal",
    "tbp": "trigonal_bipyramidal",
    "d3h": "trigonal_bipyramidal",
}


def _canonical_geometry(geometry: str) -> str:
    """Resolves a crystal-field geometry key or point-group alias."""
    key = geometry.strip().lower()
    if key not in _GEOMETRY_ALIASES:
        valid = ", ".join(sorted(set(_GEOMETRY_ALIASES.values())))
        raise exceptions.StructureError(
            f"Unsupported crystal-field geometry {geometry!r}. "
            f"Expected one of: {valid}."
        )
    return _GEOMETRY_ALIASES[key]


def crystal_field_levels(
    geometry: str = "octahedral",
    delta: float = 2.0,
    levels: Sequence[tuple[str, float, int]] | None = None,
) -> list[tuple[str, float, int]]:
    """Returns barycenter-conserving ``(label, energy, degeneracy)`` levels.

    Built-in geometries strictly satisfy ``sum(degeneracy * energy) == 0``:

    - ``"octahedral"``: ``t2g`` at ``-0.4 * delta`` (3), ``eg`` at
      ``+0.6 * delta`` (2).
    - ``"tetrahedral"``: ``e`` at ``-0.6 * delta`` (2), ``t2`` at
      ``+0.4 * delta`` (3).
    - ``"square_planar"``: ``dxz, dyz`` at ``-0.514 * delta`` (2), ``dz2`` at
      ``-0.428 * delta`` (1), ``dxy`` at ``+0.228 * delta`` (1), ``dx2-y2`` at
      ``+1.228 * delta`` (1).
    - ``"trigonal_bipyramidal"``: ``dxz, dyz`` at ``-0.272 * delta`` (2),
      ``dxy, dx2-y2`` at ``+0.082 * delta`` (2), ``dz2`` at ``+0.380 * delta``
      (1).

    Args:
      geometry: Coordination geometry name or point-group alias.
      delta: Crystal-field splitting parameter in ``unit`` (must be ``> 0``).
      levels: Optional custom ``(label, energy, degeneracy)`` level sequence.

    Returns:
      Ordered list of ``(label, energy, degeneracy)`` tuples.

    Raises:
      StructureError: If ``delta <= 0``, ``geometry`` is unknown, or custom
        ``levels`` are empty or have non-positive degeneracy.
    """
    if levels is not None:
        if not levels:
            raise exceptions.StructureError(
                "Custom crystal-field levels sequence cannot be empty."
            )
        custom: list[tuple[str, float, int]] = []
        for lbl, eng, deg in levels:
            deg_i = int(deg)
            if deg_i < 1:
                raise exceptions.StructureError(
                    f"Level {lbl!r} degeneracy must be >= 1, got {deg_i}."
                )
            custom.append((str(lbl), float(eng), deg_i))
        return custom

    d_val = float(delta)
    if d_val <= 0.0:
        raise exceptions.StructureError(
            f"Crystal-field splitting delta must be > 0, got {delta}."
        )
    geom = _canonical_geometry(geometry)
    if geom == "octahedral":
        return [
            (r"$t_{2g}$", -0.4 * d_val, 3),
            (r"$e_g$", 0.6 * d_val, 2),
        ]
    if geom == "tetrahedral":
        return [
            (r"$e$", -0.6 * d_val, 2),
            (r"$t_2$", 0.4 * d_val, 3),
        ]
    if geom == "square_planar":
        return [
            (r"$d_{xz}, d_{yz}$", -0.514 * d_val, 2),
            (r"$d_{z^2}$", -0.428 * d_val, 1),
            (r"$d_{xy}$", 0.228 * d_val, 1),
            (r"$d_{x^2-y^2}$", 1.228 * d_val, 1),
        ]
    return [
        (r"$d_{xz}, d_{yz}$", -0.272 * d_val, 2),
        (r"$d_{xy}, d_{x^2-y^2}$", 0.082 * d_val, 2),
        (r"$d_{z^2}$", 0.380 * d_val, 1),
    ]


def fill_crystal_field_orbitals(
    levels: Sequence[tuple[str, float, int]],
    d_electrons: int,
    spin_state: str = "high",
) -> list[list[tuple[int, int]]]:
    """Populates crystal-field orbitals under Hund's rule and spin state.

    Args:
      levels: Sequence of ``(label, energy, degeneracy)`` tuples.
      d_electrons: Electron count in ``0 .. 2 * sum(degeneracy)``.
      spin_state: ``"high"`` (weak-field Hund filling across all levels first)
        or ``"low"`` (strong-field Aufbau pairing within each level before
        populating higher-energy levels).

    Returns:
      Nested list ``occupations[level_idx][orb_idx] == (up, down)`` where each
      spin channel is ``0`` or ``1``.

    Raises:
      StructureError: If ``d_electrons`` or ``spin_state`` is invalid.
    """
    spin_key = spin_state.strip().lower()
    if spin_key not in ("high", "low"):
        raise exceptions.StructureError(
            f"Unsupported spin_state {spin_state!r}; expected 'high' or 'low'."
        )
    total_orbs = sum(int(deg) for _, _, deg in levels)
    n_elec = int(d_electrons)
    if n_elec < 0 or n_elec > 2 * total_orbs:
        raise exceptions.StructureError(
            f"d_electrons must be in [0, {2 * total_orbs}], got {d_electrons}."
        )
    occ: list[list[tuple[int, int]]] = [
        [(0, 0) for _ in range(int(deg))] for _, _, deg in levels
    ]
    order = sorted(range(len(levels)), key=lambda i: float(levels[i][1]))
    rem = n_elec
    if spin_key == "high":
        for idx in order:
            for orb in range(len(occ[idx])):
                if rem <= 0:
                    break
                occ[idx][orb] = (1, 0)
                rem -= 1
        for idx in order:
            for orb in range(len(occ[idx])):
                if rem <= 0:
                    break
                occ[idx][orb] = (occ[idx][orb][0], 1)
                rem -= 1
        return occ

    for idx in order:
        n_deg = len(occ[idx])
        for orb in range(n_deg):
            if rem <= 0:
                break
            occ[idx][orb] = (1, 0)
            rem -= 1
        for orb in range(n_deg):
            if rem <= 0:
                break
            occ[idx][orb] = (occ[idx][orb][0], 1)
            rem -= 1
    return occ


def _draw_spin_harpoons(
    ax: maxes.Axes,
    cx: float,
    energy: float,
    dims: tuple[float, float, float],
    spins: tuple[int, int],
    ink_col: str,
    lw: float,
) -> None:
    """Draws half-headed spin-up and spin-down harpoon arrows on one rung."""
    up, down = spins
    h_half, dx_pair, barb_w = dims
    barb_h = 0.34 * h_half
    if up > 0:
        x_up = cx - dx_pair if down > 0 else cx
        ax.plot(
            [x_up, x_up, x_up - barb_w],
            [energy - h_half, energy + h_half, energy + h_half - barb_h],
            color=ink_col,
            lw=lw,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=5.2,
        )
    if down > 0:
        x_dn = cx + dx_pair
        ax.plot(
            [x_dn, x_dn, x_dn + barb_w],
            [energy + h_half, energy - h_half, energy - h_half + barb_h],
            color=ink_col,
            lw=lw,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=5.2,
        )


def _compute_label_y_positions(
    energies: Sequence[float],
    min_sep: float,
) -> list[float]:
    """Computes anti-colliding y coordinates for adjacent orbital labels."""
    n_lvls = len(energies)
    if n_lvls <= 1:
        return [float(e) for e in energies]
    order = sorted(range(n_lvls), key=lambda i: float(energies[i]))
    y_sorted = [float(energies[i]) for i in order]
    for _ in range(18):
        moved = False
        for k in range(1, n_lvls):
            gap = y_sorted[k] - y_sorted[k - 1]
            if gap < min_sep - 1e-9:
                shift = 0.52 * (min_sep - gap)
                y_sorted[k - 1] -= shift
                y_sorted[k] += shift
                moved = True
        if not moved:
            break
    out = [0.0] * n_lvls
    for pos, orig_idx in enumerate(order):
        out[orig_idx] = y_sorted[pos]
    return out


def _draw_one_level_row(
    ax: maxes.Axes,
    level_info: tuple[str, float, int, float],
    spins_row: Sequence[tuple[int, int]],
    geom_x: tuple[float, float],
    h_half: float,
    colors: tuple[str, str, str],
    sw: theme.StrokeHierarchy,
) -> None:
    """Draws rungs, spin harpoons, and haloed label for one split level."""
    lbl, eng, deg, y_txt = level_info
    x_center, x_lbl = geom_x
    rung_col, ink_col, bg_col = colors
    rung_w, rung_gap = 0.20, 0.06
    tot_w = deg * rung_w + max(0, deg - 1) * rung_gap
    x_start = x_center - 0.5 * tot_w
    for orb in range(deg):
        x0 = x_start + orb * (rung_w + rung_gap)
        x1 = x0 + rung_w
        ax.plot(
            [x0, x1],
            [eng, eng],
            color=rung_col,
            lw=sw.primary,
            solid_capstyle="round",
            zorder=4.0,
        )
        _draw_spin_harpoons(
            ax,
            0.5 * (x0 + x1),
            eng,
            (h_half, 0.028, 0.026),
            spins_row[orb],
            ink_col,
            sw.overlay,
        )
    if abs(y_txt - eng) > 1e-6:
        ldr = ax.plot(
            [x_start + tot_w + 0.02, x_lbl - 0.02],
            [eng, y_txt],
            color=ink_col,
            lw=sw.leader * 0.85,
            alpha=0.65,
            zorder=3.5,
        )[0]
        ldr.set_sketch_params(0.0, 0.0, 0.0)
    theme.place_halo_text(
        ax,
        x_lbl,
        y_txt,
        lbl,
        color=ink_col,
        bg_color=bg_col,
        fontsize=9.5,
        ha="left",
        va="center",
        pad=0.08,
        zorder=6.5,
    )


def _draw_split_levels(
    ax: maxes.Axes,
    lvls: Sequence[tuple[str, float, int]],
    occ: Sequence[Sequence[tuple[int, int]]],
    x_center: float,
    y_span: float,
    colors: tuple[str, str, str, str],
    sw: theme.StrokeHierarchy,
) -> tuple[float, float]:
    """Draws split crystal-field orbital rungs, spins, and haloed labels."""
    stab_col, destab_col, ink_col, bg_col = colors
    uniq_e = sorted({round(float(e), 9) for _, e, _ in lvls})
    min_gap = (
        min(uniq_e[k + 1] - uniq_e[k] for k in range(len(uniq_e) - 1))
        if len(uniq_e) >= 2
        else y_span
    )
    h_half = max(0.022 * y_span, min(0.052 * y_span, 0.36 * min_gap))
    lbl_ys = _compute_label_y_positions(
        [float(e) for _, e, _ in lvls], 0.095 * y_span
    )
    max_deg = max(int(d) for _, _, d in lvls)
    max_half_w = 0.5 * (max_deg * 0.20 + max(0, max_deg - 1) * 0.06)
    x_lbl = x_center + max_half_w + 0.09
    for idx, (lbl, eng, deg) in enumerate(lvls):
        col = stab_col if eng <= 0.0 else destab_col
        _draw_one_level_row(
            ax,
            (lbl, float(eng), int(deg), lbl_ys[idx]),
            occ[idx],
            (x_center, x_lbl),
            h_half,
            (col, ink_col, bg_col),
            sw,
        )
    return x_center - max_half_w, x_center + max_half_w


def _draw_barycenter_group(
    ax: maxes.Axes,
    lvls: Sequence[tuple[str, float, int]],
    x_split_left: float,
    y_span: float,
    ink_col: str,
    bg_col: str,
    sw: theme.StrokeHierarchy,
) -> None:
    """Draws the unsplit barycenter rungs at E = 0 and dashed tie-lines."""
    n_orbs = sum(int(deg) for _, _, deg in lvls)
    rung_w = min(0.11, 0.52 / max(n_orbs, 1))
    gap = 0.03
    tot_w = n_orbs * rung_w + max(0, n_orbs - 1) * gap
    x0_base = 0.36 - 0.5 * tot_w
    for k in range(n_orbs):
        rx0 = x0_base + k * (rung_w + gap)
        ax.plot(
            [rx0, rx0 + rung_w],
            [0.0, 0.0],
            color=ink_col,
            lw=round(0.92 * sw.primary, 2),
            solid_capstyle="round",
            zorder=4.0,
        )
    theme.place_halo_text(
        ax,
        0.36,
        -0.085 * y_span,
        "Barycenter",
        color=ink_col,
        bg_color=bg_col,
        fontsize=8.6,
        ha="center",
        va="top",
        pad=0.08,
        zorder=6.0,
    )
    x_tie_start = x0_base + tot_w + 0.04
    x_tie_end = x_split_left - 0.05
    for _, eng, _ in lvls:
        ax.plot(
            [x_tie_start, x_tie_end],
            [0.0, float(eng)],
            color=palettes.MUTED_GRAY,
            linestyle="--",
            lw=sw.guide,
            alpha=0.72,
            zorder=2.0,
        )


def _resolve_delta_span(
    geom_or_custom: str,
    lvls: Sequence[tuple[str, float, int]],
    delta_label: str | None,
) -> tuple[float, float, str]:
    """Selects the lower/upper energy rungs and symbol for the Delta arrow."""
    if len(lvls) < 2:
        e0 = float(lvls[0][1])
        return e0, e0, delta_label or r"$\Delta$"
    if geom_or_custom == "square_planar" and len(lvls) == 4:
        lbl = r"$\Delta_o$" if delta_label is None else delta_label
        return float(lvls[2][1]), float(lvls[3][1]), lbl
    energies = sorted(float(e) for _, e, _ in lvls)
    if delta_label is not None:
        lbl = delta_label
    elif geom_or_custom == "octahedral":
        lbl = r"$\Delta_o$"
    elif geom_or_custom == "tetrahedral":
        lbl = r"$\Delta_t$"
    else:
        lbl = r"$\Delta$"
    return energies[0], energies[-1], lbl


def _draw_delta_callout(
    ax: maxes.Axes,
    e_bounds: tuple[float, float],
    x_arrow: float,
    x_rung_edge: float,
    label_txt: str,
    colors: tuple[str, str],
    sw: theme.StrokeHierarchy,
) -> None:
    """Draws the double-headed crystal-field splitting callout arrow."""
    e_lo, e_hi = e_bounds
    if abs(e_hi - e_lo) < 1e-9:
        return
    ink_col, bg_col = colors
    for y_val in (e_lo, e_hi):
        ext = ax.plot(
            [min(x_arrow, x_rung_edge), max(x_arrow, x_rung_edge)],
            [y_val, y_val],
            color=palettes.MUTED_GRAY,
            linestyle=":",
            lw=sw.guide,
            alpha=0.65,
            zorder=2.2,
        )[0]
        ext.set_sketch_params(0.0, 0.0, 0.0)
    ax.annotate(
        "",
        xy=(x_arrow, e_hi),
        xytext=(x_arrow, e_lo),
        arrowprops={
            "arrowstyle": "<->",
            "color": ink_col,
            "lw": sw.leader,
        },
        zorder=5.5,
    )
    theme.place_halo_text(
        ax,
        x_arrow,
        0.5 * (e_lo + e_hi),
        label_txt,
        color=ink_col,
        bg_color=bg_col,
        fontsize=9.0,
        ha="center",
        va="center",
        pad=0.12,
        alpha=0.92,
        zorder=6.8,
    )


def _draw_spin_summary_badge(
    ax: maxes.Axes,
    d_electrons: int,
    spin_state: str,
    occ: Sequence[Sequence[tuple[int, int]]],
    ink_col: str,
    bg_col: str,
) -> None:
    """Adds a haloed ink d-electron count and spin-multiplicity badge."""
    if d_electrons <= 0:
        return
    unpaired = sum(u - d for grp in occ for u, d in grp)
    spin_tag = spin_state.strip().lower()
    badge = (
        f"$d^{{{int(d_electrons)}}}$ ({spin_tag}-spin, {unpaired} unpaired)"
    )
    txt = ax.text(
        0.03,
        0.96,
        badge,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.8,
        color=ink_col,
        bbox=theme.label_halo_bbox(bg_col, pad=0.10, alpha=0.88),
        zorder=6.5,
    )
    patch = txt.get_bbox_patch()
    if patch is not None:
        patch.set_sketch_params(0.0, 0.0, 0.0)


def plot_crystal_field(
    geometry: str = "octahedral",
    d_electrons: int = 0,
    spin_state: str = "high",
    delta: float = 2.0,
    levels: Sequence[tuple[str, float, int]] | None = None,
    delta_label: str | None = None,
    unit: str = "eV",
    show_barycenter: bool = True,
    ax: maxes.Axes | None = None,
    palette: str | Sequence[str] = "okabe_ito",
    preset: str | theme.ThemePreset = "classic",
    title: str | None = None,
) -> maxes.Axes:
    r"""Plots a crystal-field d-orbital splitting and electron-filling diagram.

    Args:
      geometry: Coordination geometry (``"octahedral"``, ``"tetrahedral"``,
        ``"square_planar"``, or ``"trigonal_bipyramidal"``).
      d_electrons: Number of d-electrons (``0..10`` for 5-orbital manifolds).
      spin_state: ``"high"`` (weak-field) or ``"low"`` (strong-field).
      delta: Splitting scale ``Delta`` in ``unit``.
      levels: Optional custom ``(label, energy, degeneracy)`` sequence.
      delta_label: Optional splitting symbol override (auto ``$\Delta_o$`` /
        ``$\Delta_t$``).
      unit: Energy unit label (default ``"eV"``).
      show_barycenter: Draw unsplit barycenter rungs at ``0`` with tie-lines.
      ax: Optional target Matplotlib ``Axes``.
      palette: Series palette name or hex color sequence.
      preset: Visual theme preset name or ``ThemePreset``.
      title: Optional figure/axes title.

    Returns:
      Styled Matplotlib ``Axes`` containing the crystal-field diagram.
    """
    lvls = crystal_field_levels(geometry=geometry, delta=delta, levels=levels)
    occ = fill_crystal_field_orbitals(
        lvls, d_electrons=d_electrons, spin_state=spin_state
    )
    series = palettes.get_palette(palette)
    colors = (
        theme.get_preset(preset).ink_color,
        theme.get_preset(preset).axes_facecolor,
    )
    sw = theme.stroke_weights(preset)
    all_e = [float(e) for _, e, _ in lvls] + ([0.0] if show_barycenter else [])
    y_span = max(max(all_e) - min(all_e), 0.8)

    with theme.xkcd_style(preset=preset):
        if ax is None:
            _, ax = plt.subplots(figsize=(6.0, 4.6))
        else:
            theme.style_existing_axes(ax, preset=preset)

        z_line = ax.axhline(
            0.0,
            color=palettes.MUTED_GRAY,
            linestyle="--",
            lw=sw.guide,
            alpha=0.60,
            zorder=1.5,
        )
        z_line.set_sketch_params(0.0, 0.0, 0.0)

        x_left, x_right = _draw_split_levels(
            ax,
            lvls,
            occ,
            1.52 if show_barycenter else 0.0,
            y_span,
            (
                series[5 % len(series)],
                series[6 % len(series)],
                colors[0],
                colors[1],
            ),
            sw,
        )
        if show_barycenter:
            _draw_barycenter_group(
                ax, lvls, x_left, y_span, colors[0], colors[1], sw
            )

        geom_tag = (
            "custom" if levels is not None else _canonical_geometry(geometry)
        )
        e_lo, e_hi, d_sym = _resolve_delta_span(geom_tag, lvls, delta_label)
        _draw_delta_callout(
            ax,
            (e_lo, e_hi),
            (x_left - 0.19) if show_barycenter else (x_left - 0.24),
            x_left - 0.03,
            f"{d_sym} = {e_hi - e_lo:.2f} {unit}",
            colors,
            sw,
        )
        _draw_spin_summary_badge(
            ax, d_electrons, spin_state, occ, colors[0], colors[1]
        )

        max_lbl_chars = max(
            (len(lbl.replace("$", "").replace("\\", "")) for lbl, _, _ in lvls),
            default=4,
        )
        right_pad = max(0.68, 0.26 + 0.042 * float(max_lbl_chars))
        ax.set_xlim(
            -0.06 if show_barycenter else x_left - 0.56,
            x_right + right_pad,
        )
        ax.set_ylim(min(all_e) - 0.26 * y_span, max(all_e) + 0.30 * y_span)
        ax.set_xticks([])
        ax.set_ylabel(f"Relative Energy ({unit})")
        theme.apply_xkcd_spines(ax, preset=preset)
        ax.spines["bottom"].set_visible(False)
        if title:
            ax.set_title(title)
        layout.avoid_text_overlaps(ax, clamp_to_axes=True)
    return ax
