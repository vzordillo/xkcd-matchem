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

from matplotlib import axes as maxes
from matplotlib import figure
from matplotlib import pyplot as plt
import numpy as np

from xkcd_matchem.core import electronic
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


def plot_bands(
    bands: electronic.BandStructure,
    ax: maxes.Axes | None = None,
    ylim: tuple[float, float] = (-6.0, 6.0),
    shift_fermi: bool = True,
    palette: str | Sequence[str] = "okabe_ito",
    preset: str | theme.ThemePreset = "classic",
    title: str | None = None,
) -> maxes.Axes:
    """Plots electronic band dispersion along a reciprocal-space k-path."""
    eig = bands.shifted_eigenvalues() if shift_fermi else bands.eigenvalues
    e_ref = 0.0 if shift_fermi else bands.fermi_energy
    series = palettes.get_palette(palette)
    ink_col = theme.get_preset(preset).ink_color

    with theme.xkcd_style(preset=preset):
        if ax is None:
            _, ax = plt.subplots(figsize=(6.0, 4.4))
        else:
            theme.style_existing_axes(ax, preset=preset)
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
                    kx,
                    ky,
                    color=col,
                    linestyle=ls,
                    linewidth=1.8,
                    label=lbl,
                )
        ax.axhline(
            e_ref,
            color=palettes.MUTED_GRAY,
            linestyle="--",
            linewidth=1.3,
            label="E_F",
        )
        if bands.k_ticks and len(bands.k_ticks) == len(bands.k_labels):
            for tick_x in bands.k_ticks:
                ax.axvline(tick_x, color=ink_col, linewidth=0.9, alpha=0.45)
            ax.set_xticks(bands.k_ticks)
            ax.set_xticklabels(bands.k_labels)
        ax.set_xlim(float(bands.k_distances[0]), float(bands.k_distances[-1]))
        ax.set_ylim(ylim)
        ax.set_ylabel("E - E_F (eV)" if shift_fermi else "Energy (eV)")
        ax.set_xlabel("Wavevector Path")
        if n_spin == 2:
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
    total_col = ink_col if series[0] == "#000000" else series[0]
    fill_col = series[2 % len(series)]
    dn_col = series[6 % len(series)]
    up_label = "Spin Up" if dos.spin_down_dos is not None else "Total DOS"
    d_lo, d_hi = _dos_window_limits(e_grid, dos, xlim)

    with theme.xkcd_style(preset=preset):
        if ax is None:
            _, ax = plt.subplots(figsize=(5.8, 4.0))
        else:
            theme.style_existing_axes(ax, preset=preset)
        if background_grid:
            patterns.apply_background_grid(ax, style=background_grid)
        vert = orientation.lower() == "vertical"
        if vert:
            ax.plot(
                dos.total_dos,
                e_grid,
                color=total_col,
                lw=1.9,
                label=up_label,
            )
            patterns.shade_between(
                ax,
                e_grid,
                0.0,
                dos.total_dos,
                color=fill_col,
                shade=shade,
                alpha=0.35,
                vertical=True,
            )
            if dos.spin_down_dos is not None:
                ax.plot(
                    -dos.spin_down_dos,
                    e_grid,
                    color=dn_col,
                    lw=1.6,
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
                    vertical=True,
                )
                ax.axvline(0.0, color=ink_col, lw=0.9, alpha=0.55)
            for idx, (lbl, p_vals) in enumerate(dos.projected_dos.items()):
                ax.plot(
                    p_vals,
                    e_grid,
                    color=series[(idx + 1) % len(series)],
                    lw=1.5,
                    label=lbl,
                )
            ax.axhline(e_ref, color=palettes.MUTED_GRAY, linestyle="--", lw=1.2)
            ax.set_xlabel("DOS (states/eV)")
            ax.set_ylabel("E - E_F (eV)" if shift_fermi else "Energy (eV)")
            ax.set_xlim(d_lo, d_hi)
            if xlim is not None:
                ax.set_ylim(xlim)
        else:
            ax.plot(
                e_grid,
                dos.total_dos,
                color=total_col,
                lw=1.9,
                label=up_label,
            )
            patterns.shade_between(
                ax,
                e_grid,
                0.0,
                dos.total_dos,
                color=fill_col,
                shade=shade,
                alpha=0.35,
                vertical=False,
            )
            if dos.spin_down_dos is not None:
                ax.plot(
                    e_grid,
                    -dos.spin_down_dos,
                    color=dn_col,
                    lw=1.6,
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
                    vertical=False,
                )
                ax.axhline(0.0, color=ink_col, lw=0.9, alpha=0.55)
            for idx, (lbl, p_vals) in enumerate(dos.projected_dos.items()):
                ax.plot(
                    e_grid,
                    p_vals,
                    color=series[(idx + 1) % len(series)],
                    lw=1.5,
                    label=lbl,
                )
            ax.axvline(e_ref, color=palettes.MUTED_GRAY, linestyle="--", lw=1.2)
            ax.set_xlabel("E - E_F (eV)" if shift_fermi else "Energy (eV)")
            ax.set_ylabel("DOS (states/eV)")
            ax.set_ylim(d_lo, d_hi)
            if xlim is not None:
                ax.set_xlim(xlim)
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
    colors: tuple[str, str],
    arrow_half: float = 0.45,
) -> None:
    """Draws one degenerate energy multiplet on the MO ladder."""
    occ_col, vir_col = colors
    n_deg = len(grp)
    span = 0.55 if n_deg == 1 else 0.78
    xs = np.linspace(-span * 0.5, span * 0.5, n_deg + 1)
    for pos_k, idx in enumerate(grp):
        energy = float(mo.energies[idx])
        occ = float(mo.occupations[idx])
        x0 = float(xs[pos_k]) + 0.04
        x1 = float(xs[pos_k + 1]) - 0.04
        col = occ_col if occ > 0.5 else vir_col
        ax.plot([x0, x1], [energy, energy], color=col, lw=2.6)
        cx = 0.5 * (x0 + x1)
        if occ > 0.5:
            ax.annotate(
                "",
                xy=(cx - 0.03, energy + arrow_half),
                xytext=(cx - 0.03, energy - arrow_half),
                arrowprops={"arrowstyle": "->", "color": col, "lw": 1.3},
            )
        if occ > 1.5:
            ax.annotate(
                "",
                xy=(cx + 0.03, energy - arrow_half),
                xytext=(cx + 0.03, energy + arrow_half),
                arrowprops={"arrowstyle": "->", "color": col, "lw": 1.3},
            )
    if homo_idx is not None and homo_idx in grp:
        e_h = float(mo.energies[homo_idx])
        ax.text(
            0.46,
            e_h,
            f"HOMO ({e_h:.2f} eV)",
            va="center",
            fontsize=9.5,
            color=occ_col,
        )
    elif lumo_idx is not None and lumo_idx in grp:
        e_l = float(mo.energies[lumo_idx])
        ax.text(
            0.46,
            e_l,
            f"LUMO ({e_l:.2f} eV)",
            va="center",
            fontsize=9.5,
            color=vir_col,
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
    gap_col = series[3 % len(series)]
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
                groups[-1].append(idx)
            else:
                groups.append([idx])

        homo_idx = mo.homo_index
        lumo_idx = mo.lumo_index
        for grp in groups:
            _draw_mo_group(
                ax,
                mo,
                grp,
                homo_idx,
                lumo_idx,
                (occ_col, vir_col),
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
                arrowprops={"arrowstyle": "<->", "color": gap_col, "lw": 1.6},
            )
            ax.text(
                -0.52,
                0.5 * (e_h + e_l),
                f"Gap\n{gap:.2f} eV",
                ha="right",
                va="center",
                fontsize=9.5,
                color=gap_col,
            )

        ax.axhline(0.0, color=palettes.MUTED_GRAY, linestyle=":", lw=1.2)
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
    uu, vv = u, v
    p_min, p_max = float(np.min(plane)), float(np.max(plane))
    vmax = max(float(np.max(np.abs(plane))), 1e-6)
    is_signed = p_min < -1e-9 * vmax
    lo_lvl, hi_lvl = (-vmax, vmax) if is_signed else (0.0, vmax)
    is_flat = abs(p_max - p_min) < 1e-12
    lvl_arr = (
        np.array([p_min - 1e-6, p_max + 1e-6], dtype=np.float64)
        if is_flat
        else np.linspace(lo_lvl, hi_lvl, max(int(levels), 2))
    )
    preset_obj = theme.get_preset(preset)
    with theme.xkcd_style(preset=preset):
        if ax is None:
            _, ax = plt.subplots(figsize=(5.4, 4.6))
        else:
            theme.style_existing_axes(ax, preset=preset)
        ax.contourf(
            uu,
            vv,
            plane,
            levels=lvl_arr,
            cmap=cmap,
            vmin=lo_lvl,
            vmax=hi_lvl,
            alpha=0.82,
        )
        if not is_flat:
            ax.contour(
                uu,
                vv,
                plane,
                levels=lvl_arr,
                colors=preset_obj.ink_color,
                linewidths=0.9,
                alpha=0.7,
            )
        if show_atoms:
            _draw_slice_atoms(ax, grid, axis, slice_index, basis, preset_obj)
        ax.set_aspect("equal")
        is_ortho = (
            float(np.max(np.abs(basis[0]))) > 0.999
            and float(np.max(np.abs(basis[1]))) > 0.999
        )
        axis_labels = {
            "x": ("y (A)", "z (A)"),
            "y": ("x (A)", "z (A)"),
            "z": ("x (A)", "y (A)"),
        }
        xl, yl = axis_labels[axis.lower()] if is_ortho else ("u (A)", "v (A)")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        theme.apply_xkcd_spines(ax, preset=preset)
        if title:
            ax.set_title(title)
    return ax
