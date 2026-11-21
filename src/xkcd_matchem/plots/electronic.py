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
from xkcd_matchem.style import theme


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
                ax.plot(
                    bands.k_distances,
                    eig[spin_idx, :, ib],
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


def plot_dos(
    dos: electronic.DensityOfStates,
    ax: maxes.Axes | None = None,
    orientation: str = "horizontal",
    xlim: tuple[float, float] | None = None,
    shift_fermi: bool = True,
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

    with theme.xkcd_style(preset=preset):
        if ax is None:
            _, ax = plt.subplots(figsize=(5.8, 4.0))
        else:
            theme.style_existing_axes(ax, preset=preset)
        vert = orientation.lower() == "vertical"
        if vert:
            ax.plot(
                dos.total_dos,
                e_grid,
                color=total_col,
                lw=1.9,
                label="Total DOS",
            )
            ax.fill_betweenx(
                e_grid, 0.0, dos.total_dos, color=fill_col, alpha=0.35
            )
            if dos.spin_down_dos is not None:
                ax.plot(
                    -dos.spin_down_dos,
                    e_grid,
                    color=dn_col,
                    lw=1.6,
                    label="Spin Down",
                )
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
            if xlim is not None:
                ax.set_ylim(xlim)
        else:
            ax.plot(
                e_grid,
                dos.total_dos,
                color=total_col,
                lw=1.9,
                label="Total DOS",
            )
            ax.fill_between(
                e_grid, 0.0, dos.total_dos, color=fill_col, alpha=0.35
            )
            if dos.spin_down_dos is not None:
                ax.plot(
                    e_grid,
                    -dos.spin_down_dos,
                    color=dn_col,
                    lw=1.6,
                    label="Spin Down",
                )
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
                xy=(cx - 0.03, energy + 0.45),
                xytext=(cx - 0.03, energy - 0.45),
                arrowprops={"arrowstyle": "->", "color": col, "lw": 1.3},
            )
        if occ > 1.5:
            ax.annotate(
                "",
                xy=(cx + 0.03, energy - 0.45),
                xytext=(cx + 0.03, energy + 0.45),
                arrowprops={"arrowstyle": "->", "color": col, "lw": 1.3},
            )
        if idx == homo_idx and pos_k == n_deg - 1:
            ax.text(
                0.46,
                energy,
                f"HOMO ({energy:.2f} eV)",
                va="center",
                fontsize=9.5,
                color=occ_col,
            )
        elif idx == lumo_idx and pos_k == n_deg - 1:
            ax.text(
                0.46,
                energy,
                f"LUMO ({energy:.2f} eV)",
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
            _draw_mo_group(ax, mo, grp, homo_idx, lumo_idx, (occ_col, vir_col))

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
        ax.set_xlim(-0.85, 1.05)
        ax.set_ylim(window)
        ax.set_xticks([])
        ax.set_ylabel("Orbital Energy (eV)")
        theme.apply_xkcd_spines(ax, preset=preset)
        if title:
            ax.set_title(title)
    return ax


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
    vmax = max(float(np.max(np.abs(plane))), 1e-6)
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
            levels=levels,
            cmap=cmap,
            vmin=-vmax,
            vmax=vmax,
            alpha=0.82,
        )
        ax.contour(
            uu,
            vv,
            plane,
            levels=levels,
            colors=preset_obj.ink_color,
            linewidths=0.9,
            alpha=0.7,
        )
        if show_atoms and grid.atoms is not None and len(grid.atoms) > 0:
            atom_uv = grid.atoms.positions @ basis.T
            ax.scatter(
                atom_uv[:, 0],
                atom_uv[:, 1],
                s=75,
                facecolor=preset_obj.axes_facecolor,
                edgecolor=preset_obj.ink_color,
                linewidth=1.5,
                zorder=6,
            )
        ax.set_aspect("equal")
        axis_labels = {
            "x": ("y (A)", "z (A)"),
            "y": ("x (A)", "z (A)"),
            "z": ("x (A)", "y (A)"),
        }
        xl, yl = axis_labels[axis.lower()]
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        theme.apply_xkcd_spines(ax, preset=preset)
        if title:
            ax.set_title(title)
    return ax
