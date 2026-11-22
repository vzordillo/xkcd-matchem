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
from matplotlib import collections
from matplotlib import colors as mcolors
from matplotlib import figure
from matplotlib import pyplot as plt
import numpy as np

from xkcd_matchem.core import electronic
from xkcd_matchem.core import exceptions
from xkcd_matchem.core import geometry
from xkcd_matchem.core import isosurface
from xkcd_matchem.io import loader
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
        if fb_weights is not None:
            for spin_idx in range(n_spin):
                col = (
                    series[5 % len(series)]
                    if spin_idx == 0
                    else series[6 % len(series)]
                )
                for ib in range(n_bands):
                    kx, ky = _split_kpath_breaks(
                        bands.k_distances, eig[spin_idx, :, ib]
                    )
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
        ax.axhline(
            e_ref,
            color=palettes.MUTED_GRAY,
            linestyle="--",
            linewidth=1.3,
            label="$E_F$",
        )
        if bands.k_ticks and len(bands.k_ticks) == len(bands.k_labels):
            for tick_x in bands.k_ticks:
                ax.axvline(tick_x, color=ink_col, linewidth=0.9, alpha=0.45)
            ax.set_xticks(bands.k_ticks)
            ax.set_xticklabels(_format_k_labels(bands.k_labels))
        ax.set_xlim(float(bands.k_distances[0]), float(bands.k_distances[-1]))
        ax.set_ylim(ylim)
        ax.set_ylabel("$E - E_F$ (eV)" if shift_fermi else "Energy (eV)")
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
            ax.set_ylabel("$E - E_F$ (eV)" if shift_fermi else "Energy (eV)")
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
            ax.set_xlabel("$E - E_F$ (eV)" if shift_fermi else "Energy (eV)")
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


def _chain_silhouette_edges(
    edges: Sequence[tuple[int, int]],
    xy_v: np.ndarray,
) -> list[np.ndarray]:
    """Chains unordered 2D silhouette edges into continuous stroke paths."""
    if not edges:
        return []
    adj: dict[int, list[int]] = {}
    for u, v in edges:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
    visited: set[tuple[int, int]] = set()
    paths: list[np.ndarray] = []
    for start, nbrs in adj.items():
        for nxt in nbrs:
            e_key = (start, nxt) if start < nxt else (nxt, start)
            if e_key in visited:
                continue
            chain = [start]
            cur = nxt
            visited.add(e_key)
            while True:
                chain.append(cur)
                if cur == start:
                    break
                moved = False
                for cand in adj.get(cur, ()):
                    c_key = (cur, cand) if cur < cand else (cand, cur)
                    if c_key not in visited:
                        visited.add(c_key)
                        cur = cand
                        moved = True
                        break
                if not moved:
                    break
            if len(chain) >= 3:
                paths.append(xy_v[np.asarray(chain, dtype=np.int64)])
    return paths


def _extract_silhouette_polylines(
    mesh: isosurface.IsosurfaceMesh,
    xy_v: np.ndarray,
    front_mask: np.ndarray,
) -> list[np.ndarray]:
    """Finds front/back horizon edges separating visible and hidden faces."""
    edge_flags: dict[tuple[int, int], list[bool]] = {}
    for t_idx, tri in enumerate(mesh.faces):
        is_f = bool(front_mask[t_idx])
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        for u, v in ((i0, i1), (i1, i2), (i2, i0)):
            key = (u, v) if u < v else (v, u)
            edge_flags.setdefault(key, []).append(is_f)
    sil_edges = [
        key
        for key, flags in edge_flags.items()
        if (len(flags) == 2 and flags[0] != flags[1])
        or (len(flags) == 1 and flags[0])
    ]
    return _chain_silhouette_edges(sil_edges, xy_v)


def _collect_mesh_triangles(
    mesh: isosurface.IsosurfaceMesh,
    cam: geometry.Camera,
    pivot: np.ndarray,
    span: float,
    base_hex: str,
    basis: np.ndarray,
    light_dir: np.ndarray,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]
]:
    """Projects front-facing ``IsosurfaceMesh`` triangles & rim paths."""
    if len(mesh.faces) == 0:
        return (
            np.zeros((0, 3, 2), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 2), dtype=np.float64),
            [],
        )
    xy_v, depth_v = geometry.project_points(
        mesh.vertices, cam, center=pivot, span=span
    )
    n_rot = mesh.normals @ basis.T
    front_mask = n_rot[:, 2] >= -0.02
    sils = _extract_silhouette_polylines(mesh, xy_v, front_mask)
    if not np.any(front_mask):
        return (
            np.zeros((0, 3, 2), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
            np.zeros((0, 3), dtype=np.float64),
            xy_v,
            sils,
        )
    vis_faces = mesh.faces[front_mask]
    tris = xy_v[vis_faces]
    tri_depth = np.mean(depth_v[vis_faces], axis=1)
    lambert = 0.68 + 0.32 * np.clip(n_rot[front_mask] @ light_dir, 0.0, 1.0)
    rgb_base = np.asarray(mcolors.to_rgb(base_hex), dtype=np.float64)
    face_rgb = np.clip(
        0.26 + 0.74 * rgb_base[None, :] * lambert[:, None], 0.0, 1.0
    )
    return tris, tri_depth, face_rgb, xy_v, sils


def _iso_scene_frame(
    meshes: Sequence[tuple[isosurface.IsosurfaceMesh, str]],
    grid: electronic.VolumetricGrid,
    cam: geometry.Camera,
) -> tuple[np.ndarray, float, np.ndarray | None, np.ndarray | None]:
    """Resolves common pivot, projection span, and projected atom depths."""
    if grid.atoms is not None and len(grid.atoms) > 0:
        pivot = (
            0.5 * (grid.atoms.cell[0] + grid.atoms.cell[1] + grid.atoms.cell[2])
            if grid.atoms.cell is not None
            else np.mean(grid.atoms.positions, axis=0)
        )
        span = geometry.scene_projection_span(
            grid.atoms.positions, cam, center=pivot, cell=grid.atoms.cell
        )
        atom_xy, atom_depth = geometry.project_points(
            grid.atoms.positions, cam, center=pivot, span=span
        )
        return pivot, span, atom_xy, atom_depth

    ext = np.asarray(grid.data.shape, dtype=np.float64)[:, None]
    pivot = grid.origin + 0.5 * np.sum(ext * grid.voxel_vectors, axis=0)
    pts = [mesh.vertices for mesh, _ in meshes if len(mesh.vertices)]
    box_pts = np.vstack(pts) if pts else np.zeros((0, 3), dtype=np.float64)
    span = (
        geometry.scene_projection_span(box_pts, cam, center=pivot)
        if len(box_pts)
        else 5.0
    )
    return pivot, span, None, None


def _make_iso_poly_collection(
    tris: np.ndarray,
    cols: np.ndarray,
    depths: np.ndarray,
    alpha: float,
    zorder: float,
) -> collections.PolyCollection | None:
    """Constructs a depth-sorted, un-sketched PolyCollection for smooth fill."""
    if len(tris) == 0:
        return None
    order = np.argsort(depths)
    coll = collections.PolyCollection(
        cast(Any, tris[order]),
        facecolors=cols[order],
        edgecolors="none",
        linewidths=0.0,
        antialiaseds=False,
        alpha=alpha,
        zorder=zorder,
    )
    coll.set_sketch_params(0.0, 0.0, 0.0)
    return coll


def _split_depth_layer_collections(
    tri_chunks: Sequence[np.ndarray],
    dep_chunks: Sequence[np.ndarray],
    rgb_chunks: Sequence[np.ndarray],
    atom_xy: np.ndarray | None,
    atom_depth: np.ndarray | None,
    alpha: float,
) -> tuple[
    collections.PolyCollection | None,
    collections.PolyCollection | None,
]:
    """Splits shaded triangles into behind-atom and front-of-atom layers."""
    all_tris = np.concatenate(tri_chunks, axis=0)
    all_deps = np.concatenate(dep_chunks, axis=0)
    all_rgbs = np.concatenate(rgb_chunks, axis=0)
    if atom_xy is not None and atom_depth is not None:
        tri_xy = np.mean(all_tris, axis=1)
        nearest = np.argmin(
            np.sum((tri_xy[:, None, :] - atom_xy[None, :, :]) ** 2, axis=2),
            axis=1,
        )
        back_mask = all_deps < (atom_depth[nearest] - 0.03)
    else:
        back_mask = np.zeros(len(all_tris), dtype=bool)
    back_coll = _make_iso_poly_collection(
        all_tris[back_mask],
        all_rgbs[back_mask],
        all_deps[back_mask],
        alpha * 0.95,
        zorder=1.4,
    )
    front_coll = _make_iso_poly_collection(
        all_tris[~back_mask],
        all_rgbs[~back_mask],
        all_deps[~back_mask],
        alpha,
        zorder=40.0,
    )
    return back_coll, front_coll


def _build_isosurface_layers(
    meshes: Sequence[tuple[isosurface.IsosurfaceMesh, str]],
    grid: electronic.VolumetricGrid,
    cam: geometry.Camera,
    alpha: float,
) -> tuple[
    collections.PolyCollection | None,
    collections.PolyCollection | None,
    list[tuple[str, float, list[np.ndarray]]],
    np.ndarray | None,
]:
    """Splits shaded isosurface triangles behind/in front of atoms + rims."""
    pivot, span, atom_xy, atom_depth = _iso_scene_frame(meshes, grid, cam)
    basis = geometry.rotation_matrix(cam.azimuth, cam.elevation, cam.roll)
    light_dir = np.array([-0.45, 0.55, 0.70], dtype=np.float64)
    light_dir /= float(np.linalg.norm(light_dir))
    tri_chunks: list[np.ndarray] = []
    dep_chunks: list[np.ndarray] = []
    rgb_chunks: list[np.ndarray] = []
    xy_chunks: list[np.ndarray] = []
    sil_draws: list[tuple[str, float, list[np.ndarray]]] = []
    for mesh, hex_col in meshes:
        tris, deps, rgbs, xy_m, sils = _collect_mesh_triangles(
            mesh, cam, pivot, span, hex_col, basis, light_dir
        )
        if len(tris) > 0:
            tri_chunks.append(tris)
            dep_chunks.append(deps)
            rgb_chunks.append(rgbs)
            xy_chunks.append(xy_m)
            z_sil = 41.0
            if atom_xy is not None and atom_depth is not None:
                c_xy = np.mean(tris.reshape(-1, 2), axis=0)
                near = int(np.argmin(np.sum((atom_xy - c_xy) ** 2, axis=1)))
                if float(np.mean(deps)) < float(atom_depth[near]) - 0.03:
                    z_sil = 1.5
            sil_draws.append((hex_col, z_sil, sils))
    if not tri_chunks:
        return None, None, [], None

    back_coll, front_coll = _split_depth_layer_collections(
        tri_chunks, dep_chunks, rgb_chunks, atom_xy, atom_depth, alpha
    )
    return back_coll, front_coll, sil_draws, np.vstack(xy_chunks)


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


def plot_isosurface(
    source: electronic.VolumetricGrid | str | pathlib.Path,
    level: float | None = None,
    *,
    symmetric: bool | None = None,
    camera: geometry.Camera | str | None = "isometric",
    ax: maxes.Axes | None = None,
    colors: tuple[str, str] = ("#0072B2", "#D55E00"),
    alpha: float = 0.28,
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
      show_atoms: Whether to draw embedded atomic ball-and-stick structure.
      palette: Atom element color palette when ``show_atoms=True``.
      preset: Visual theme preset name or ``ThemePreset``.
      title: Optional heading above the panel.

    Returns:
      Matplotlib ``Axes`` containing the rendered 3D isosurface.
    """
    grid = _coerce_volumetric_grid(source)
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

    cam = geometry.resolve_camera(
        camera, cell=grid.atoms.cell if grid.atoms is not None else None
    )
    back_coll, front_coll, sil_draws, all_xy = _build_isosurface_layers(
        meshes, grid, cam, alpha
    )
    ink_col = theme.get_preset(preset).ink_color
    has_atoms = bool(
        show_atoms and grid.atoms is not None and len(grid.atoms) > 0
    )

    with theme.xkcd_style(preset=preset):
        if ax is None:
            _, ax = plt.subplots(figsize=(5.6, 4.8))
        else:
            theme.style_existing_axes(ax, preset=preset)
        if back_coll is not None:
            ax.add_collection(back_coll)
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
        if front_coll is not None:
            ax.add_collection(front_coll)
        for hex_col, z_sil, sil_paths in sil_draws:
            for poly in sil_paths:
                ax.plot(
                    poly[:, 0],
                    poly[:, 1],
                    color=hex_col,
                    linewidth=1.6,
                    alpha=min(1.0, alpha + 0.44),
                    zorder=z_sil,
                )
        if all_xy is not None:
            _expand_isosurface_limits(ax, all_xy, has_atoms)
        ax.set_aspect("equal")
        ax.set_axis_off()
        if title:
            ax.set_title(title, color=ink_col)
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
                linewidth=1.7,
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
                    linewidth=2.4,
                    zorder=4.0,
                    label="Imaginary Modes" if not imag_labeled else None,
                )
                imag_labeled = True

        ax.axhline(0.0, color=palettes.MUTED_GRAY, linestyle="--", lw=1.2)
        if pbs.q_ticks and len(pbs.q_ticks) == len(pbs.q_labels):
            for tick_q in pbs.q_ticks:
                ax.axvline(tick_q, color=ink_col, linewidth=0.9, alpha=0.45)
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
) -> None:
    """Plots one COHP/ICOHP interaction curve and optional lobe shading."""
    if vertical:
        ax.plot(vals, e_grid, color=col, lw=2.0, label=label, zorder=4.0)
        if shade_lobes:
            pos = np.where(vals >= 0.0, vals, 0.0)
            neg = np.where(vals < 0.0, vals, 0.0)
            patterns.shade_between(
                ax, e_grid, 0.0, pos, color="#0072B2", shade="solid",
                alpha=0.22, vertical=True,
            )
            patterns.shade_between(
                ax, e_grid, 0.0, neg, color="#D55E00", shade="solid",
                alpha=0.22, vertical=True,
            )
        if spin_dn is not None:
            ax.plot(-spin_dn, e_grid, color=col, lw=1.6, linestyle="--")
    else:
        ax.plot(e_grid, vals, color=col, lw=2.0, label=label, zorder=4.0)
        if shade_lobes:
            pos = np.where(vals >= 0.0, vals, 0.0)
            neg = np.where(vals < 0.0, vals, 0.0)
            patterns.shade_between(
                ax, e_grid, 0.0, pos, color="#0072B2", shade="solid",
                alpha=0.22, vertical=False,
            )
            patterns.shade_between(
                ax, e_grid, 0.0, neg, color="#D55E00", shade="solid",
                alpha=0.22, vertical=False,
            )
        if spin_dn is not None:
            ax.plot(e_grid, -spin_dn, color=col, lw=1.6, linestyle="--")


def _configure_cohp_axes(
    ax: maxes.Axes,
    *,
    vertical: bool,
    pop_label: str,
    ink_col: str,
    xlim: tuple[float, float] | None,
    ylim: tuple[float, float] | None,
) -> None:
    """Applies Fermi reference lines, limits, and labels to a COHP panel."""
    if vertical:
        ax.axhline(0.0, color=palettes.MUTED_GRAY, linestyle="--", lw=1.2)
        ax.axvline(0.0, color=ink_col, lw=1.0, alpha=0.55)
        ax.set_xlabel(pop_label)
        ax.set_ylabel("$E - E_F$ (eV)")
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
    else:
        ax.axvline(0.0, color=palettes.MUTED_GRAY, linestyle="--", lw=1.2)
        ax.axhline(0.0, color=ink_col, lw=1.0, alpha=0.55)
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
            )
        _configure_cohp_axes(
            ax,
            vertical=vertical,
            pop_label=f"-{prefix} (eV)" if sign < 0.0 else f"{prefix} (eV)",
            ink_col=theme.get_preset(preset).ink_color,
            xlim=xlim,
            ylim=ylim,
        )
        ax.legend(loc="best", frameon=True, fontsize=8.5)
        theme.apply_xkcd_spines(ax, preset=preset)
        if title:
            ax.set_title(title)
    return ax
