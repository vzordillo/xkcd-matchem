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
"""Electronic band structures, fatbands, and density-of-states plots."""

from __future__ import annotations

from collections.abc import Sequence

from matplotlib import axes as maxes
from matplotlib import figure
from matplotlib import pyplot as plt
import numpy as np

from xkcd_matchem.core import electronic
from xkcd_matchem.core import exceptions
from xkcd_matchem.plots import _materials_ternary as mat_tern
from xkcd_matchem.style import palettes
from xkcd_matchem.style import patterns
from xkcd_matchem.style import theme


def split_kpath_breaks(
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


def format_k_labels(labels: Sequence[str]) -> list[str]:
    """Formats standard Brillouin-zone tick names with mathtext symbols."""
    out: list[str] = []
    for raw in labels:
        token = str(raw).strip()
        if token.upper() in ("G", "GAMMA"):
            out.append("$\\Gamma$")
        else:
            out.append(str(raw))
    return out


def resolve_fatband_weights(
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


def draw_band_curves(
    ax: maxes.Axes,
    bands: electronic.BandStructure,
    eig: np.ndarray,
    fb_weights: np.ndarray | None,
    fatband_scale: float,
    series: Sequence[str],
    sw: theme.StrokeHierarchy,
    *,
    bg_color: str = palettes.PAPER,
    ink_col: str = palettes.INK,
) -> None:
    """Plots spin-resolved dispersion curves and optional orbital fatbands."""
    n_spin, _, n_bands = eig.shape
    for spin_idx in range(n_spin):
        raw_col = (
            series[5 % len(series)]
            if spin_idx == 0
            else series[6 % len(series)]
        )
        col = mat_tern.theme_series_color(raw_col, bg_color, ink_col)
        ls = "-" if spin_idx == 0 else "--"
        for ib in range(n_bands):
            lbl = (
                ("Spin Up" if spin_idx == 0 else "Spin Down")
                if (n_spin == 2 and ib == 0)
                else None
            )
            kx, ky = split_kpath_breaks(
                bands.k_distances, eig[spin_idx, :, ib]
            )
            ax.plot(
                kx, ky, color=col, linestyle=ls, linewidth=sw.primary, label=lbl
            )
            if fb_weights is not None:
                _, kw = split_kpath_breaks(
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
    fb_weights = resolve_fatband_weights(bands, fatband)
    eig = bands.shifted_eigenvalues() if shift_fermi else bands.eigenvalues
    e_ref = 0.0 if shift_fermi else bands.fermi_energy
    series = palettes.get_palette(palette)
    preset_obj = theme.get_preset(preset)
    ink_col = preset_obj.ink_color
    bg_col = preset_obj.axes_facecolor
    guide_col = mat_tern.theme_muted_guide(bg_col)
    sw = theme.stroke_weights(preset)

    with theme.xkcd_style(preset=preset):
        if ax is None:
            _, ax = plt.subplots(figsize=(6.0, 4.4))
        else:
            theme.style_existing_axes(ax, preset=preset)
        draw_band_curves(
            ax,
            bands,
            eig,
            fb_weights,
            fatband_scale,
            series,
            sw,
            bg_color=bg_col,
            ink_col=ink_col,
        )
        ef_line = ax.axhline(
            e_ref,
            color=guide_col,
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
            ax.set_xticklabels(format_k_labels(bands.k_labels))
        if bands.k_distances.size >= 2:
            ax.set_xlim(
                float(bands.k_distances[0]), float(bands.k_distances[-1])
            )
        ax.set_ylim(ylim)
        ax.set_ylabel("$E - E_F$ (eV)" if shift_fermi else "Energy (eV)")
        ax.set_xlabel("Wavevector Path")
        if eig.shape[0] == 2:
            mat_tern.style_legend(
                ax, ink_col, bg_col, loc="upper right", frameon=True
            )
        theme.apply_xkcd_spines(ax, preset=preset)
        if title:
            ax.set_title(title, color=ink_col)
    return ax


def dos_window_limits(
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


def draw_dos_curves(
    ax: maxes.Axes,
    dos: electronic.DensityOfStates,
    e_grid: np.ndarray,
    *,
    vert: bool,
    cols: tuple[str, str, str],
    series: Sequence[str],
    shade: str | None,
    sw: theme.StrokeHierarchy,
    bg_color: str = palettes.PAPER,
    ink_col: str = palettes.INK,
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
        p_col = mat_tern.theme_series_color(
            series[(idx + 1) % len(series)], bg_color, ink_col
        )
        ax.plot(px, py, color=p_col, lw=sw.overlay, label=lbl)


def configure_dos_axes(
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
    bg_color: str = palettes.PAPER,
) -> None:
    """Configures straight Fermi/zero guide lines, labels, and DOS limits."""
    e_label = "$E - E_F$ (eV)" if shift_fermi else "Energy (eV)"
    guide_col = mat_tern.theme_muted_guide(bg_color)
    if vert:
        if has_spin_down:
            z_ln = ax.axvline(0.0, color=ink_col, lw=sw.guide, alpha=0.55)
            z_ln.set_sketch_params(0.0, 0.0, 0.0)
        ef_ln = ax.axhline(
            e_ref, color=guide_col, linestyle="--", lw=sw.guide
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
            e_ref, color=guide_col, linestyle="--", lw=sw.guide
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
    preset_obj = theme.get_preset(preset)
    ink_col = preset_obj.ink_color
    bg_col = preset_obj.axes_facecolor
    sw = theme.stroke_weights(preset)
    total_col = (
        ink_col
        if series[0] == "#000000"
        else mat_tern.theme_series_color(series[0], bg_col, ink_col)
    )
    fill_col = mat_tern.theme_series_color(
        series[2 % len(series)], bg_col, ink_col
    )
    dn_col = mat_tern.theme_series_color(
        series[6 % len(series)], bg_col, ink_col
    )
    d_lims = dos_window_limits(e_grid, dos, xlim)
    vert = orientation.lower() == "vertical"

    with theme.xkcd_style(preset=preset):
        if ax is None:
            _, ax = plt.subplots(figsize=(5.8, 4.0))
        else:
            theme.style_existing_axes(ax, preset=preset)
        if background_grid:
            patterns.apply_background_grid(ax, style=background_grid)
        draw_dos_curves(
            ax,
            dos,
            e_grid,
            vert=vert,
            cols=(total_col, fill_col, dn_col),
            series=series,
            shade=shade,
            sw=sw,
            bg_color=bg_col,
            ink_col=ink_col,
        )
        configure_dos_axes(
            ax,
            vert=vert,
            e_ref=e_ref,
            d_lims=d_lims,
            xlim=xlim,
            shift_fermi=shift_fermi,
            has_spin_down=dos.spin_down_dos is not None,
            ink_col=ink_col,
            sw=sw,
            bg_color=bg_col,
        )
        mat_tern.style_legend(ax, ink_col, bg_col, loc="best", frameon=True)
        theme.apply_xkcd_spines(ax, preset=preset)
        if title:
            ax.set_title(title, color=ink_col)
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
    ink_col = theme.get_preset(preset).ink_color
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
            fig.suptitle(title, y=0.98, color=ink_col)
    return fig, (ax_b, ax_d)
