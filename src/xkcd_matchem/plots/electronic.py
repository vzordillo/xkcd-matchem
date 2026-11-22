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
"""Electronic structure visualizations (bands, DOS, MOs, volumetric slices)."""

from __future__ import annotations

from collections.abc import Sequence
import pathlib
from typing import Any
from typing import cast

from matplotlib import axes as maxes
from matplotlib import figure
from matplotlib import pyplot as plt
import numpy as np

from xkcd_matchem.core import electronic
from xkcd_matchem.core import exceptions
from xkcd_matchem.core import geometry
from xkcd_matchem.core import isosurface
from xkcd_matchem.io import loader
from xkcd_matchem.render import artistic3d
from xkcd_matchem.render import structure3d
from xkcd_matchem.style import palettes
from xkcd_matchem.style import patterns
from xkcd_matchem.style import theme


def _split_kpath_breaks(
    k_dist: np.ndarray,
    y_band: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Inserts NaNs at zero-length k-path jumps with differing eigenvalues."""
    if len(k_dist) < 2:
        return k_dist, y_band
    breaks = np.where(
        (np.abs(np.diff(k_dist)) < 1e-9) & (np.abs(np.diff(y_band)) > 1e-6)
    )[0]
    if breaks.size == 0:
        return k_dist, y_band
    pos = breaks + 1
    return np.insert(k_dist, pos, k_dist[breaks]), np.insert(
        y_band, pos, np.nan
    )


def _format_k_labels(labels: Sequence[str]) -> list[str]:
    """Formats standard Brillouin-zone tick names with mathtext symbols."""
    out: list[str] = []
    for raw in labels:
        token = str(raw).strip()
        if token.upper() in ("G", "GAMMA"):
            out.append("$\\Gamma$")
        else:
            out.append(str(raw))
    return out


def _resolve_fatband_weights(
    bands: electronic.BandStructure,
    fatband: str | Sequence[str] | None,
) -> np.ndarray | None:
    """Aggregates requested orbital projection channels onto weights."""
    if fatband is None:
        return None
    if not bands.projections:
        raise exceptions.StructureError(
            "BandStructure has no stored orbital projections for fatbands."
        )
    keys = [fatband] if isinstance(fatband, str) else list(fatband)
    if not keys:
        raise exceptions.StructureError("fatband channel list cannot be empty.")
    total: np.ndarray | None = None
    for key in keys:
        if key not in bands.projections:
            avail = ", ".join(sorted(bands.projections))
            raise exceptions.StructureError(
                f"Projection channel {key!r} missing. Available: {avail}"
            )
        arr = np.asarray(bands.projections[key], dtype=np.float64)
        total = arr.copy() if total is None else (total + arr)
    assert_non_none = total if total is not None else np.zeros((1, 1, 1))
    peak = float(np.max(np.abs(assert_non_none)))
    return np.clip(assert_non_none, 0.0, None) / max(peak, 1e-12)


def _draw_band_curves(
    ax: maxes.Axes,
    bands: electronic.BandStructure,
    eig: np.ndarray,
    fb_weights: np.ndarray | None,
    fatband_scale: float,
    series: Sequence[str],
    sw: theme.StrokeHierarchy,
) -> None:
    """Plots spin-resolved dispersion curves and optional orbital fatbands."""
    n_spin, _, n_bands = eig.shape
    for spin_idx in range(n_spin):
        col = (
            series[5 % len(series)]
            if spin_idx == 0
            else series[6 % len(series)]
        )
        ls = "-" if spin_idx == 0 else "--"
        for ib in range(n_bands):
            lbl = (
                ("Spin Up" if spin_idx == 0 else "Spin Down")
                if (n_spin == 2 and ib == 0)
                else None
            )
            kx, ky = _split_kpath_breaks(
                bands.k_distances, eig[spin_idx, :, ib]
            )
            ax.plot(
                kx, ky, color=col, linestyle=ls, linewidth=sw.primary, label=lbl
            )
            if fb_weights is not None:
                _, kw = _split_kpath_breaks(
                    bands.k_distances, fb_weights[spin_idx, :, ib]
                )
                half_w = fatband_scale * kw
                ax.fill_between(
                    kx,
                    ky - half_w,
                    ky + half_w,
                    color=col,
                    alpha=0.28,
                    linewidth=0.0,
                    zorder=1.8,
                )


def plot_bands(
    bands: electronic.BandStructure,
    ax: maxes.Axes | None = None,
    ylim: tuple[float, float] = (-6.0, 6.0),
    shift_fermi: bool = True,
    palette: str | Sequence[str] = "okabe_ito",
    preset: str | theme.ThemePreset = "classic",
    title: str | None = None,
    *,
    fatband: str | Sequence[str] | None = None,
    fatband_scale: float = 0.28,
) -> maxes.Axes:
    """Plots electronic band dispersion along a reciprocal-space k-path."""
    fb_weights = _resolve_fatband_weights(bands, fatband)
    eig = bands.shifted_eigenvalues() if shift_fermi else bands.eigenvalues
    e_ref = 0.0 if shift_fermi else bands.fermi_energy
    series = palettes.get_palette(palette)
    ink_col = theme.get_preset(preset).ink_color
    sw = theme.stroke_weights(preset)

    with theme.xkcd_style(preset=preset):
        if ax is None:
            _, ax = plt.subplots(figsize=(6.0, 4.4))
        else:
            theme.style_existing_axes(ax, preset=preset)
        _draw_band_curves(
            ax, bands, eig, fb_weights, fatband_scale, series, sw
        )
        ef_line = ax.axhline(
            e_ref,
            color=palettes.MUTED_GRAY,
            linestyle="--",
            linewidth=sw.guide,
            label="$E_F$",
        )
        ef_line.set_sketch_params(0.0, 0.0, 0.0)
        if bands.k_ticks and len(bands.k_ticks) == len(bands.k_labels):
            for tick_x in bands.k_ticks:
                t_line = ax.axvline(
                    tick_x, color=ink_col, linewidth=sw.guide, alpha=0.40
                )
                t_line.set_sketch_params(0.0, 0.0, 0.0)
            ax.set_xticks(bands.k_ticks)
            ax.set_xticklabels(_format_k_labels(bands.k_labels))
        if bands.k_distances.size >= 2:
            ax.set_xlim(
                float(bands.k_distances[0]), float(bands.k_distances[-1])
            )
        ax.set_ylim(ylim)
        ax.set_ylabel("$E - E_F$ (eV)" if shift_fermi else "Energy (eV)")
        ax.set_xlabel("Wavevector Path")
        if eig.shape[0] == 2:
            ax.legend(loc="upper right", frameon=True)
        theme.apply_xkcd_spines(ax, preset=preset)
        if title:
            ax.set_title(title)
    return ax


def _dos_window_limits(
    e_grid: np.ndarray,
    dos: electronic.DensityOfStates,
    xlim: tuple[float, float] | None,
) -> tuple[float, float]:
    """Computes DOS axis ``(lo, hi)`` scaled to the visible energy window."""
    mask: np.ndarray
    if xlim is not None and e_grid.size > 0:
        e_lo = min(float(xlim[0]), float(xlim[1]))
        e_hi = max(float(xlim[0]), float(xlim[1]))
        mask = (e_grid >= e_lo) & (e_grid <= e_hi)
    else:
        mask = np.ones(e_grid.shape, dtype=bool)
    if not np.any(mask):
        mask = np.ones(e_grid.shape, dtype=bool)

    d_hi = float(np.max(dos.total_dos[mask])) if e_grid.size else 1.0
    d_lo = min(0.0, float(np.min(dos.total_dos[mask]))) if e_grid.size else 0.0
    for p_vals in dos.projected_dos.values():
        d_hi = max(d_hi, float(np.max(p_vals[mask])))
        d_lo = min(d_lo, float(np.min(p_vals[mask])))
    d_hi = max(d_hi, 1e-6) * 1.12
    d_lo = d_lo * 1.12 if d_lo < 0.0 else 0.0
    if dos.spin_down_dos is not None and e_grid.size:
        dn_min = float(np.min(-dos.spin_down_dos[mask]))
        d_lo = min(d_lo, min(dn_min, -1e-6) * 1.12)
    return d_lo, d_hi


def _draw_dos_curves(
    ax: maxes.Axes,
    dos: electronic.DensityOfStates,
    e_grid: np.ndarray,
    *,
    vert: bool,
    cols: tuple[str, str, str],
    series: Sequence[str],
    shade: str | None,
    sw: theme.StrokeHierarchy,
) -> None:
    """Plots total, optional spin-down, and projected DOS curves."""
    total_col, fill_col, dn_col = cols
    up_label = "Spin Up" if dos.spin_down_dos is not None else "Total DOS"

    def _xy(y_vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (y_vals, e_grid) if vert else (e_grid, y_vals)

    tx, ty = _xy(dos.total_dos)
    ax.plot(tx, ty, color=total_col, lw=sw.primary, label=up_label)
    patterns.shade_between(
        ax,
        e_grid,
        0.0,
        dos.total_dos,
        color=fill_col,
        shade=shade,
        alpha=0.35,
        vertical=vert,
    )
    if dos.spin_down_dos is not None:
        dx, dy = _xy(-dos.spin_down_dos)
        ax.plot(
            dx,
            dy,
            color=dn_col,
            lw=round(0.88 * sw.primary, 2),
            label="Spin Down",
        )
        patterns.shade_between(
            ax,
            e_grid,
            0.0,
            -dos.spin_down_dos,
            color=dn_col,
            shade=shade,
            alpha=0.35,
            vertical=vert,
        )
    for idx, (lbl, p_vals) in enumerate(dos.projected_dos.items()):
        px, py = _xy(p_vals)
        ax.plot(
            px,
            py,
            color=series[(idx + 1) % len(series)],
            lw=sw.overlay,
            label=lbl,
        )


def _configure_dos_axes(
    ax: maxes.Axes,
    *,
    vert: bool,
    e_ref: float,
    d_lims: tuple[float, float],
    xlim: tuple[float, float] | None,
    shift_fermi: bool,
    has_spin_down: bool,
    ink_col: str,
    sw: theme.StrokeHierarchy,
) -> None:
    """Configures straight Fermi/zero guide lines, labels, and DOS limits."""
    e_label = "$E - E_F$ (eV)" if shift_fermi else "Energy (eV)"
    if vert:
        if has_spin_down:
            z_ln = ax.axvline(0.0, color=ink_col, lw=sw.guide, alpha=0.55)
            z_ln.set_sketch_params(0.0, 0.0, 0.0)
        ef_ln = ax.axhline(
            e_ref, color=palettes.MUTED_GRAY, linestyle="--", lw=sw.guide
        )
        ef_ln.set_sketch_params(0.0, 0.0, 0.0)
        ax.set_xlabel("DOS (states/eV)")
        ax.set_ylabel(e_label)
        ax.set_xlim(d_lims[0], d_lims[1])
        if xlim is not None:
            ax.set_ylim(xlim)
    else:
        if has_spin_down:
            z_ln = ax.axhline(0.0, color=ink_col, lw=sw.guide, alpha=0.55)
            z_ln.set_sketch_params(0.0, 0.0, 0.0)
        ef_ln = ax.axvline(
            e_ref, color=palettes.MUTED_GRAY, linestyle="--", lw=sw.guide
        )
        ef_ln.set_sketch_params(0.0, 0.0, 0.0)
        ax.set_xlabel(e_label)
        ax.set_ylabel("DOS (states/eV)")
        ax.set_ylim(d_lims[0], d_lims[1])
        if xlim is not None:
            ax.set_xlim(xlim)


def plot_dos(
    dos: electronic.DensityOfStates,
    ax: maxes.Axes | None = None,
    orientation: str = "horizontal",
    xlim: tuple[float, float] | None = None,
    shift_fermi: bool = True,
    shade: str | None = None,
    background_grid: str | None = None,
    palette: str | Sequence[str] = "okabe_ito",
    preset: str | theme.ThemePreset = "classic",
    title: str | None = None,
) -> maxes.Axes:
    """Plots total and optional projected/spin-resolved density of states."""
    e_grid = dos.energies - dos.fermi_energy if shift_fermi else dos.energies
    e_ref = 0.0 if shift_fermi else dos.fermi_energy
    series = palettes.get_palette(palette)
    ink_col = theme.get_preset(preset).ink_color
    sw = theme.stroke_weights(preset)
    total_col = ink_col if series[0] == "#000000" else series[0]
    fill_col = series[2 % len(series)]
    dn_col = series[6 % len(series)]
    d_lims = _dos_window_limits(e_grid, dos, xlim)
    vert = orientation.lower() == "vertical"

    with theme.xkcd_style(preset=preset):
        if ax is None:
            _, ax = plt.subplots(figsize=(5.8, 4.0))
        else:
            theme.style_existing_axes(ax, preset=preset)
        if background_grid:
            patterns.apply_background_grid(ax, style=background_grid)
        _draw_dos_curves(
            ax,
            dos,
            e_grid,
            vert=vert,
            cols=(total_col, fill_col, dn_col),
            series=series,
            shade=shade,
            sw=sw,
        )
        _configure_dos_axes(
            ax,
            vert=vert,
            e_ref=e_ref,
            d_lims=d_lims,
            xlim=xlim,
            shift_fermi=shift_fermi,
            has_spin_down=dos.spin_down_dos is not None,
            ink_col=ink_col,
            sw=sw,
        )
        ax.legend(loc="best", frameon=True)
        theme.apply_xkcd_spines(ax, preset=preset)
        if title:
            ax.set_title(title)
    return ax


def plot_bands_dos(
    bands: electronic.BandStructure,
    dos: electronic.DensityOfStates,
    ylim: tuple[float, float] = (-6.0, 6.0),
    shift_fermi: bool = True,
    palette: str | Sequence[str] = "okabe_ito",
    preset: str | theme.ThemePreset = "classic",
    title: str | None = None,
) -> tuple[figure.Figure, tuple[maxes.Axes, maxes.Axes]]:
    """Creates a synchronized two-panel band structure + vertical DOS figure."""
    with theme.xkcd_style(preset=preset):
        fig, (ax_b, ax_d) = plt.subplots(
            1,
            2,
            figsize=(8.4, 4.6),
            gridspec_kw={"width_ratios": [2.2, 1.0], "wspace": 0.08},
            sharey=True,
        )
        plot_bands(
            bands,
            ax=ax_b,
            ylim=ylim,
            shift_fermi=shift_fermi,
            palette=palette,
            preset=preset,
        )
        plot_dos(
            dos,
            ax=ax_d,
            orientation="vertical",
            xlim=ylim,
            shift_fermi=shift_fermi,
            palette=palette,
            preset=preset,
        )
        ax_d.set_ylabel("")
        ax_d.tick_params(labelleft=False)
        if title:
            fig.suptitle(title, y=0.98)
    return fig, (ax_b, ax_d)


def _draw_mo_group(
    ax: maxes.Axes,
    mo: electronic.MolecularOrbitals,
    grp: list[int],
    homo_idx: int | None,
    lumo_idx: int | None,
    colors: tuple[str, str, str],
    arrow_half: float = 0.45,
) -> None:
    """Draws one degenerate energy multiplet on the MO ladder."""
    occ_col, vir_col, ink_col = colors
    n_deg = len(grp)
    span = 0.55 if n_deg == 1 else 0.78
    xs = np.linspace(-span * 0.5, span * 0.5, n_deg + 1)
    for pos_k, idx in enumerate(grp):
        energy = float(mo.energies[idx])
        occ = float(mo.occupations[idx])
        x0 = float(xs[pos_k]) + 0.04
        x1 = float(xs[pos_k + 1]) - 0.04
        col = occ_col if occ > 0.5 else vir_col
        ax.plot([x0, x1], [energy, energy], color=col, lw=2.5)
        cx = 0.5 * (x0 + x1)
        if occ > 0.5:
            ax.annotate(
                "",
                xy=(cx - 0.03, energy + arrow_half),
                xytext=(cx - 0.03, energy - arrow_half),
                arrowprops={"arrowstyle": "->", "color": col, "lw": 1.15},
            )
        if occ > 1.5:
            ax.annotate(
                "",
                xy=(cx + 0.03, energy - arrow_half),
                xytext=(cx + 0.03, energy + arrow_half),
                arrowprops={"arrowstyle": "->", "color": col, "lw": 1.15},
            )
    if homo_idx is not None and homo_idx in grp:
        e_h = float(mo.energies[homo_idx])
        ax.text(
            0.46,
            e_h,
            f"HOMO ({e_h:.2f} eV)",
            va="center",
            fontsize=9.2,
            color=ink_col,
        )
    elif lumo_idx is not None and lumo_idx in grp:
        e_l = float(mo.energies[lumo_idx])
        ax.text(
            0.46,
            e_l,
            f"LUMO ({e_l:.2f} eV)",
            va="center",
            fontsize=9.2,
            color=ink_col,
        )


def plot_mo_diagram(
    mo: electronic.MolecularOrbitals,
    ax: maxes.Axes | None = None,
    window: tuple[float, float] = (-20.0, 6.0),
    degeneracy_tol: float = 0.08,
    palette: str | Sequence[str] = "okabe_ito",
    preset: str | theme.ThemePreset = "classic",
    title: str | None = None,
) -> maxes.Axes:
    """Draws a molecular orbital energy level ladder highlighting HOMO/LUMO."""
    series = palettes.get_palette(palette)
    occ_col = series[5 % len(series)]
    vir_col = series[6 % len(series)]
    ink_col = theme.get_preset(preset).ink_color
    sw = theme.stroke_weights(preset)
    arrow_half = 0.036 * max(float(window[1] - window[0]), 1.0)
    with theme.xkcd_style(preset=preset):
        if ax is None:
            _, ax = plt.subplots(figsize=(4.4, 5.2))
        else:
            theme.style_existing_axes(ax, preset=preset)
        mask = (mo.energies >= window[0]) & (mo.energies <= window[1])
        indices = list(np.where(mask)[0])
        groups: list[list[int]] = []
        for idx in sorted(indices, key=lambda k: float(mo.energies[k])):
            if (
                groups
                and abs(float(mo.energies[idx] - mo.energies[groups[-1][0]]))
                <= degeneracy_tol
            ):
                groups[-1].append(int(idx))
            else:
                groups.append([int(idx)])

        homo_idx = mo.homo_index
        lumo_idx = mo.lumo_index
        for grp in groups:
            _draw_mo_group(
                ax,
                mo,
                grp,
                homo_idx,
                lumo_idx,
                (occ_col, vir_col, ink_col),
                arrow_half,
            )

        gap = mo.homo_lumo_gap()
        if gap is not None and homo_idx is not None and lumo_idx is not None:
            e_h = float(mo.energies[homo_idx])
            e_l = float(mo.energies[lumo_idx])
            ax.annotate(
                "",
                xy=(-0.48, e_l),
                xytext=(-0.48, e_h),
                arrowprops={
                    "arrowstyle": "<->",
                    "color": ink_col,
                    "lw": sw.leader,
                },
            )
            ax.text(
                -0.52,
                0.5 * (e_h + e_l),
                f"Gap\n{gap:.2f} eV",
                ha="right",
                va="center",
                fontsize=9.2,
                color=ink_col,
            )

        z_line = ax.axhline(
            0.0, color=palettes.MUTED_GRAY, linestyle=":", lw=sw.guide
        )
        z_line.set_sketch_params(0.0, 0.0, 0.0)
        ax.set_xlim(-0.92, 1.32)
        ax.set_ylim(window)
        ax.set_xticks([])
        ax.set_ylabel("Orbital Energy (eV)")
        theme.apply_xkcd_spines(ax, preset=preset)
        if title:
            ax.set_title(title)
    return ax


def _draw_slice_atoms(
    ax: maxes.Axes,
    grid: electronic.VolumetricGrid,
    axis: str,
    slice_index: int | None,
    basis: np.ndarray,
    preset_obj: theme.ThemePreset,
) -> None:
    """Projects atoms onto a volumetric slice with element colors & depth."""
    if grid.atoms is None or len(grid.atoms) == 0:
        return
    ax_idx = {"x": 0, "y": 1, "z": 2}[axis.lower()]
    n_along = grid.data.shape[ax_idx]
    sl_idx = n_along // 2 if slice_index is None else int(slice_index)
    slice_origin = grid.origin + sl_idx * grid.voxel_vectors[ax_idx]
    normal = np.cross(basis[0], basis[1])
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    step_perp = max(
        abs(float(np.dot(grid.voxel_vectors[ax_idx], normal))), 0.25
    )
    d_perp = np.abs((grid.atoms.positions - slice_origin) @ normal)
    atom_uv = grid.atoms.positions @ basis.T
    colors = [
        palettes.get_element_color(sym, scheme="xkcd")
        for sym in grid.atoms.symbols
    ]
    in_plane = d_perp <= 1.5 * step_perp
    for mask, size, alpha in ((~in_plane, 45.0, 0.38), (in_plane, 85.0, 0.95)):
        if np.any(mask):
            cols = [c for c, flag in zip(colors, mask, strict=True) if flag]
            ax.scatter(
                atom_uv[mask, 0],
                atom_uv[mask, 1],
                s=size,
                facecolor=cols,
                edgecolor=preset_obj.ink_color,
                linewidth=1.5 if size > 60.0 else 1.1,
                alpha=alpha,
                zorder=6 if size > 60.0 else 5.5,
            )


def plot_volumetric_slice(
    grid: electronic.VolumetricGrid,
    ax: maxes.Axes | None = None,
    axis: str = "z",
    slice_index: int | None = None,
    levels: int = 9,
    cmap: str = "coolwarm",
    show_atoms: bool = True,
    preset: str | theme.ThemePreset = "classic",
    title: str | None = None,
) -> maxes.Axes:
    """Plots a 2D filled-plus-inked slice across a ``VolumetricGrid``."""
    u, v, plane, basis = grid.slice_mesh(axis=axis, index=slice_index)
    p_min, p_max = float(np.min(plane)), float(np.max(plane))
    vmax = max(float(np.max(np.abs(plane))), 1e-6)
    lo_lvl, hi_lvl = (-vmax, vmax) if p_min < -1e-9 * vmax else (0.0, vmax)
    is_flat = abs(p_max - p_min) < 1e-12
    lvl_arr = (
        np.array([p_min - 1e-6, p_max + 1e-6], dtype=np.float64)
        if is_flat
        else np.linspace(lo_lvl, hi_lvl, max(int(levels), 2))
    )
    preset_obj = theme.get_preset(preset)
    sw = theme.stroke_weights(preset)
    with theme.xkcd_style(preset=preset):
        if ax is None:
            _, ax = plt.subplots(figsize=(5.4, 4.6))
        else:
            theme.style_existing_axes(ax, preset=preset)
        ax.contourf(
            u,
            v,
            plane,
            levels=lvl_arr,
            cmap=cmap,
            vmin=lo_lvl,
            vmax=hi_lvl,
            alpha=0.82,
        )
        if not is_flat:
            c_lvls = lvl_arr[np.abs(lvl_arr) > 0.08 * vmax]
            if c_lvls.size >= 1:
                ax.contour(
                    u,
                    v,
                    plane,
                    levels=c_lvls,
                    colors=preset_obj.ink_color,
                    linewidths=round(0.85 * sw.guide, 2),
                    alpha=0.58,
                )
        if show_atoms:
            _draw_slice_atoms(ax, grid, axis, slice_index, basis, preset_obj)
        ax.set_aspect("equal")
        axis_labels = {
            "x": ("y (A)", "z (A)"),
            "y": ("x (A)", "z (A)"),
            "z": ("x (A)", "y (A)"),
        }
        xl, yl = (
            axis_labels[axis.lower()]
            if (
                float(np.max(np.abs(basis[0]))) > 0.999
                and float(np.max(np.abs(basis[1]))) > 0.999
            )
            else ("u (A)", "v (A)")
        )
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        theme.apply_xkcd_spines(ax, preset=preset)
        if title:
            ax.set_title(title)
    return ax


def _coerce_volumetric_grid(
    source: electronic.VolumetricGrid | str | pathlib.Path,
) -> electronic.VolumetricGrid:
    """Coerces path, sample name, or ``VolumetricGrid`` into grid object."""
    if isinstance(source, electronic.VolumetricGrid):
        return source
    loaded = loader.load(source)
    if not isinstance(loaded, electronic.VolumetricGrid):
        raise exceptions.StructureError(
            f"Expected VolumetricGrid, got {type(loaded).__name__}."
        )
    return loaded


def _expand_isosurface_limits(
    ax: maxes.Axes,
    all_xy: np.ndarray,
    has_atoms: bool,
) -> None:
    """Expands viewport limits cleanly around both atoms and isosurface."""
    mx0, my0 = np.min(all_xy, axis=0)
    mx1, my1 = np.max(all_xy, axis=0)
    if has_atoms:
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        ax.set_xlim(min(x0, float(mx0) - 0.5), max(x1, float(mx1) + 0.5))
        ax.set_ylim(min(y0, float(my0) - 0.5), max(y1, float(my1) + 0.5))
        return
    pad = 0.15 * max(float(mx1 - mx0), float(my1 - my0), 1.0)
    ax.set_xlim(float(mx0) - pad, float(mx1) + pad)
    ax.set_ylim(float(my0) - pad, float(my1) + pad)


def _extract_isosurface_meshes(
    source: electronic.VolumetricGrid | str | pathlib.Path,
    level: float | None,
    symmetric: bool | None,
    colors: tuple[str, str],
) -> tuple[
    electronic.VolumetricGrid,
    list[tuple[isosurface.IsosurfaceMesh, str]],
]:
    """Coerces grid data and extracts positive and optional negative lobes."""
    raw_grid = _coerce_volumetric_grid(source)
    grid = isosurface.upsample_coarse_grid(raw_grid, min_dim=36)
    max_abs = max(float(np.max(np.abs(grid.data))), 1e-12)
    iso_level = 0.35 * max_abs if level is None else float(level)
    if abs(iso_level) < 1e-15:
        raise exceptions.StructureError("Isosurface level must be non-zero.")
    sym_flag = (
        bool(float(np.min(grid.data)) < -0.05 * max_abs)
        if symmetric is None
        else bool(symmetric)
    )
    meshes: list[tuple[isosurface.IsosurfaceMesh, str]] = [
        (isosurface.extract_isosurface(grid, iso_level), colors[0])
    ]
    if sym_flag:
        meshes.append(
            (isosurface.extract_isosurface(grid, -abs(iso_level)), colors[1])
        )
    return grid, meshes


def _draw_iso_art_batches(
    ax: maxes.Axes,
    art: artistic3d.ShadeArtData,
    sketch_scale: float,
) -> None:
    """Draws front wash, stipple dots, comic strokes, and silhouettes."""
    if art.front_rgba is not None and art.wash_extent is not None:
        ax.imshow(
            art.front_rgba,
            extent=art.wash_extent,
            origin="lower",
            interpolation="bicubic",
            zorder=40.0,
        )
    for pts, sz, rgba_s, z_s in art.dot_batches:
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=sz,
            c=cast(Any, rgba_s),
            edgecolors="none",
            zorder=z_s,
        )
    for segs, col_l, lw_l, al_l, z_l, sk in art.line_batches:
        for poly in segs:
            ln = ax.plot(
                poly[:, 0],
                poly[:, 1],
                color=col_l,
                linewidth=lw_l,
                alpha=al_l,
                solid_capstyle="round",
                zorder=z_l,
            )[0]
            if sk is not None and sketch_scale > 0.0:
                ln.set_sketch_params(sk[0] * sketch_scale, sk[1], sk[2])


def plot_isosurface(
    source: electronic.VolumetricGrid | str | pathlib.Path,
    level: float | None = None,
    *,
    symmetric: bool | None = None,
    camera: geometry.Camera | str | None = None,
    ax: maxes.Axes | None = None,
    colors: tuple[str, str] = ("#0072B2", "#D55E00"),
    alpha: float = 0.28,
    shade: str = "solid",
    show_atoms: bool = True,
    palette: str = "xkcd",
    preset: str | theme.ThemePreset = "classic",
    title: str | None = None,
) -> maxes.Axes:
    """Plots a 3D shaded orbital or electron-density isosurface.

    Args:
      source: ``VolumetricGrid`` or path/sample (``.cube``, ``CHGCAR``).
      level: Isovalue threshold. Defaults to ``0.35 * max(|data|)``.
      symmetric: If ``True``, extracts both ``+level`` and ``-level`` lobes.
      camera: ``Camera`` object or preset string.
      ax: Optional target Matplotlib ``Axes``.
      colors: ``(positive_hex, negative_hex)`` lobe colors.
      alpha: Surface translucency opacity.
      shade: Shading style: ``"solid"`` (smooth blending), ``"hatch"``
        (parallel pen hatching), ``"crosshatch"`` (comic cross-hatching),
        ``"contour"`` / ``"cross_contour"`` (3D curved form lines),
        ``"stipple"`` (felt-tip stippling), ``"scumble"`` (xkcd scribble),
        or ``"outline"``.
      show_atoms: Whether to draw embedded atomic ball-and-stick structure.
      palette: Atom element color palette when ``show_atoms=True``.
      preset: Visual theme preset name or ``ThemePreset``.
      title: Optional heading above the panel.

    Returns:
      Matplotlib ``Axes`` containing the rendered 3D isosurface.
    """
    grid, meshes = _extract_isosurface_meshes(source, level, symmetric, colors)
    if camera is None or (
        isinstance(camera, str) and camera.strip().lower() == "auto"
    ):
        cam = artistic3d.auto_orbital_camera(meshes, grid)
    else:
        cam = geometry.resolve_camera(
            camera, cell=grid.atoms.cell if grid.atoms is not None else None
        )
    art = artistic3d.build_isosurface_art(
        meshes, grid, cam, alpha, shade=shade
    )
    preset_obj = theme.get_preset(preset)
    has_atoms = bool(
        show_atoms and grid.atoms is not None and len(grid.atoms) > 0
    )

    with theme.xkcd_style(preset=preset):
        if ax is None:
            _, ax = plt.subplots(figsize=(5.6, 4.8))
        else:
            theme.style_existing_axes(ax, preset=preset)
        if art.back_rgba is not None and art.wash_extent is not None:
            ax.imshow(
                art.back_rgba,
                extent=art.wash_extent,
                origin="lower",
                interpolation="bicubic",
                zorder=1.4,
            )
        if has_atoms and grid.atoms is not None:
            structure3d.render_structure_3d(
                grid.atoms,
                ax=ax,
                camera=cam,
                style="ball_and_stick",
                palette=palette,
                show_cell=False,
                crystal_axes=False,
                show_bonds=True,
                radius_scale=1.08,
            )
        _draw_iso_art_batches(ax, art, preset_obj.sketch_scale)
        if art.all_xy is not None:
            _expand_isosurface_limits(ax, art.all_xy, has_atoms)
        ax.set_aspect("equal")
        ax.set_axis_off()
        if title:
            ax.set_title(title, color=preset_obj.ink_color)
    return ax


def plot_phonon_bands(
    source: electronic.PhononBandStructure | str | pathlib.Path,
    *,
    ax: maxes.Axes | None = None,
    unit: str = "THz",
    ylim: tuple[float, float] | None = None,
    highlight_imaginary: bool = True,
    palette: str | Sequence[str] = "okabe_ito",
    preset: str | theme.ThemePreset = "classic",
    title: str | None = None,
) -> maxes.Axes:
    """Plots phonon dispersion branches along a reciprocal-space q-path.

    Args:
      source: ``PhononBandStructure`` or Phonopy ``band.yaml`` path/sample.
      ax: Optional target Matplotlib ``Axes``.
      unit: Display unit (``"THz"``, ``"cm-1"``, or ``"meV"``).
      ylim: Optional vertical axis limits in ``unit``.
      highlight_imaginary: Highlight soft/imaginary modes (< 0) in vermilion.
      palette: Series palette name or hex color sequence.
      preset: Visual theme preset name or ``ThemePreset``.
      title: Optional plot heading.

    Returns:
      Matplotlib ``Axes`` containing the phonon dispersion diagram.
    """
    if isinstance(source, electronic.PhononBandStructure):
        pbs = source
    else:
        loaded = loader.load(source)
        if not isinstance(loaded, electronic.PhononBandStructure):
            raise exceptions.StructureError(
                f"Expected PhononBandStructure, got {type(loaded).__name__}."
            )
        pbs = loaded

    freqs = pbs.frequencies_in(unit)
    series = palettes.get_palette(palette)
    ink_col = theme.get_preset(preset).ink_color
    sw = theme.stroke_weights(preset)
    mode_col = series[5 % len(series)]
    imag_col = "#D55E00"
    imag_labeled = False

    with theme.xkcd_style(preset=preset):
        if ax is None:
            _, ax = plt.subplots(figsize=(6.2, 4.4))
        else:
            theme.style_existing_axes(ax, preset=preset)
        n_modes = freqs.shape[1]
        for mode_idx in range(n_modes):
            qx, qy = _split_kpath_breaks(pbs.q_distances, freqs[:, mode_idx])
            ax.plot(
                qx,
                qy,
                color=mode_col,
                linewidth=sw.primary,
                label="Acoustic / Optical" if mode_idx == 0 else None,
            )
            if highlight_imaginary and bool(np.any(qy < -1e-2)):
                imag_mask = qy < -1e-2
                dilated = imag_mask.copy()
                dilated[1:] |= imag_mask[:-1]
                dilated[:-1] |= imag_mask[1:]
                imag_y = np.where(dilated, qy, np.nan)
                ax.plot(
                    qx,
                    imag_y,
                    color=imag_col,
                    linewidth=sw.frame,
                    zorder=4.0,
                    label="Imaginary Modes" if not imag_labeled else None,
                )
                imag_labeled = True

        z_line = ax.axhline(
            0.0, color=palettes.MUTED_GRAY, linestyle="--", lw=sw.guide
        )
        z_line.set_sketch_params(0.0, 0.0, 0.0)
        if pbs.q_ticks and len(pbs.q_ticks) == len(pbs.q_labels):
            for tick_q in pbs.q_ticks:
                t_line = ax.axvline(
                    tick_q, color=ink_col, linewidth=sw.guide, alpha=0.40
                )
                t_line.set_sketch_params(0.0, 0.0, 0.0)
            ax.set_xticks(pbs.q_ticks)
            ax.set_xticklabels(_format_k_labels(pbs.q_labels))
        ax.set_xlim(float(pbs.q_distances[0]), float(pbs.q_distances[-1]))
        if ylim is not None:
            ax.set_ylim(ylim)
        disp_unit = (
            "$\\mathrm{cm}^{-1}$" if unit.strip().lower() == "cm-1" else unit
        )
        ax.set_xlabel("Wavevector Path")
        ax.set_ylabel(f"Frequency ({disp_unit})")
        if imag_labeled:
            ax.legend(loc="best", frameon=True)
        theme.apply_xkcd_spines(ax, preset=preset)
        if title:
            ax.set_title(title)
    return ax


def _draw_cohp_series(
    ax: maxes.Axes,
    e_grid: np.ndarray,
    vals: np.ndarray,
    *,
    col: str,
    label: str,
    vertical: bool,
    shade_lobes: bool,
    spin_dn: np.ndarray | None,
    series_idx: int = 0,
    sw: theme.StrokeHierarchy | None = None,
) -> None:
    """Plots one COHP/ICOHP interaction curve and optional lobe shading."""
    weights = sw if sw is not None else theme.stroke_weights("classic")
    lw_curve = (
        weights.primary if series_idx == 0 else weights.overlay + 0.10
    )
    ls_curve = "-" if series_idx == 0 else "--"
    z_curve = 4.0 + 0.25 * min(series_idx, 3)
    if vertical:
        ax.plot(
            vals,
            e_grid,
            color=col,
            lw=lw_curve,
            linestyle=ls_curve,
            label=label,
            zorder=z_curve,
        )
        if shade_lobes and series_idx == 0:
            pos = np.where(vals >= 0.0, vals, 0.0)
            neg = np.where(vals < 0.0, vals, 0.0)
            patterns.shade_between(
                ax, e_grid, 0.0, pos, color="#0072B2", shade="solid",
                alpha=0.20, vertical=True,
            )
            patterns.shade_between(
                ax, e_grid, 0.0, neg, color="#D55E00", shade="solid",
                alpha=0.20, vertical=True,
            )
        if spin_dn is not None:
            ax.plot(
                -spin_dn, e_grid, color=col, lw=weights.overlay, linestyle=":"
            )
    else:
        ax.plot(
            e_grid,
            vals,
            color=col,
            lw=lw_curve,
            linestyle=ls_curve,
            label=label,
            zorder=z_curve,
        )
        if shade_lobes and series_idx == 0:
            pos = np.where(vals >= 0.0, vals, 0.0)
            neg = np.where(vals < 0.0, vals, 0.0)
            patterns.shade_between(
                ax, e_grid, 0.0, pos, color="#0072B2", shade="solid",
                alpha=0.20, vertical=False,
            )
            patterns.shade_between(
                ax, e_grid, 0.0, neg, color="#D55E00", shade="solid",
                alpha=0.20, vertical=False,
            )
        if spin_dn is not None:
            ax.plot(
                e_grid, -spin_dn, color=col, lw=weights.overlay, linestyle=":"
            )


def _configure_cohp_axes(
    ax: maxes.Axes,
    *,
    vertical: bool,
    pop_label: str,
    ink_col: str,
    xlim: tuple[float, float] | None,
    ylim: tuple[float, float] | None,
    sw: theme.StrokeHierarchy | None = None,
) -> None:
    """Applies Fermi reference lines, limits, and labels to a COHP panel."""
    weights = sw if sw is not None else theme.stroke_weights("classic")
    if vertical:
        ef_ln = ax.axhline(
            0.0, color=palettes.MUTED_GRAY, linestyle="--", lw=weights.guide
        )
        z_ln = ax.axvline(0.0, color=ink_col, lw=weights.guide, alpha=0.50)
        ef_ln.set_sketch_params(0.0, 0.0, 0.0)
        z_ln.set_sketch_params(0.0, 0.0, 0.0)
        ax.set_xlabel(pop_label)
        ax.set_ylabel("$E - E_F$ (eV)")
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
    else:
        ef_ln = ax.axvline(
            0.0, color=palettes.MUTED_GRAY, linestyle="--", lw=weights.guide
        )
        z_ln = ax.axhline(0.0, color=ink_col, lw=weights.guide, alpha=0.50)
        ef_ln.set_sketch_params(0.0, 0.0, 0.0)
        z_ln.set_sketch_params(0.0, 0.0, 0.0)
        ax.set_xlabel("$E - E_F$ (eV)")
        ax.set_ylabel(pop_label)
        if ylim is not None:
            ax.set_xlim(ylim)
        if xlim is not None:
            ax.set_ylim(xlim)


def _resolve_cohp_inputs(
    source: electronic.CohpData | str | pathlib.Path,
    interactions: Sequence[str] | None,
    integrated: bool,
) -> tuple[
    electronic.CohpData,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    list[str],
]:
    """Loads COHP data and resolves spin tables and selected interactions."""
    if isinstance(source, electronic.CohpData):
        data = source
    else:
        loaded = loader.load(source)
        if not isinstance(loaded, electronic.CohpData):
            raise exceptions.StructureError(
                f"Expected CohpData, got {type(loaded).__name__}."
            )
        data = loaded
    table = data.icohp if integrated else data.cohp
    dn_table = data.spin_down_icohp if integrated else data.spin_down_cohp
    if not table:
        raise exceptions.StructureError("COHP dataset contains no data table.")
    selected = (
        list(table.keys()) if interactions is None else list(interactions)
    )
    missing = [lbl for lbl in selected if lbl not in table]
    if missing:
        raise exceptions.StructureError(
            f"Unknown COHP interaction labels: {missing!r}."
        )
    return data, table, dn_table, selected


def plot_cohp(
    source: electronic.CohpData | str | pathlib.Path,
    *,
    interactions: Sequence[str] | None = None,
    minus_cohp: bool = True,
    integrated: bool = False,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = (-8.0, 6.0),
    orientation: str = "vertical",
    shade_lobes: bool = True,
    palette: str | Sequence[str] = "okabe_ito",
    preset: str | theme.ThemePreset = "classic",
    ax: maxes.Axes | None = None,
    title: str | None = None,
) -> maxes.Axes:
    """Plots LOBSTER ``-pCOHP`` or ``-ICOHP`` bonding curves in xkcd style.

    By chemical convention (``minus_cohp=True``), positive ``-pCOHP`` values
    denote bonding states (shaded cerulean) while negative values denote
    antibonding states (shaded vermilion).

    Args:
      source: ``CohpData`` instance or path/sample name to ``COHPCAR.lobster``.
      interactions: Optional subset of interaction labels to plot.
      minus_cohp: Negate ``pCOHP``/``ICOHP`` so bonding appears positive.
      integrated: Plot cumulative ``ICOHP`` instead of differential ``pCOHP``.
      xlim: Optional population-axis bounds.
      ylim: Optional energy window around ``E_F`` in eV.
      orientation: ``"vertical"`` (E - E_F on y-axis) or ``"horizontal"``.
      shade_lobes: Shade bonding vs. antibonding lobes when ``not integrated``.
      palette: Series palette name or hex color list.
      preset: Visual theme preset name or ``ThemePreset``.
      ax: Optional target Matplotlib ``Axes``.
      title: Optional plot heading.

    Returns:
      Matplotlib ``Axes`` containing the COHP/ICOHP plot.

    Raises:
      StructureError: If input cannot be loaded or requested labels are absent.
    """
    data, table, dn_table, selected = _resolve_cohp_inputs(
        source, interactions, integrated
    )
    sign = -1.0 if (minus_cohp and not data.is_coop) else 1.0
    prefix = "ICOHP" if integrated else "COOP" if data.is_coop else "pCOHP"
    e_grid = data.energies - data.fermi_energy
    vertical = orientation.strip().lower() != "horizontal"
    series = palettes.get_palette(palette)
    sw = theme.stroke_weights(preset)

    with theme.xkcd_style(preset=preset):
        if ax is None:
            _, ax = plt.subplots(
                figsize=(5.2, 5.4) if vertical else (6.8, 4.2)
            )
        else:
            theme.style_existing_axes(ax, preset=preset)
        for idx, lbl in enumerate(selected):
            dn_vals = sign * dn_table[lbl] if lbl in dn_table else None
            _draw_cohp_series(
                ax,
                e_grid,
                sign * table[lbl],
                col=series[(idx + 1) % len(series)],
                label=lbl,
                vertical=vertical,
                shade_lobes=bool(shade_lobes and not integrated),
                spin_dn=dn_vals,
                series_idx=idx,
                sw=sw,
            )
        _configure_cohp_axes(
            ax,
            vertical=vertical,
            pop_label=f"-{prefix} (eV)" if sign < 0.0 else f"{prefix} (eV)",
            ink_col=theme.get_preset(preset).ink_color,
            xlim=xlim,
            ylim=ylim,
            sw=sw,
        )
        ax.legend(loc="best", frameon=True, fontsize=8.5)
        theme.apply_xkcd_spines(ax, preset=preset)
        if title:
            ax.set_title(title)
    return ax
