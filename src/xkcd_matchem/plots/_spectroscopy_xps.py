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
"""XPS core-level deconvolution with Shirley background and Voigt fits."""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any

from matplotlib import axes as maxes
from matplotlib import pyplot as plt
import numpy as np

from xkcd_matchem.core import exceptions
from xkcd_matchem.core import spectroscopy
from xkcd_matchem.plots import _materials_ternary as mat_tern
from xkcd_matchem.plots import _spectroscopy_nmr_core as nmr_core
from xkcd_matchem.style import palettes
from xkcd_matchem.style import patterns
from xkcd_matchem.style import theme


def compute_shirley_background(
    binding_energy: Sequence[float] | np.ndarray,
    intensity: Sequence[float] | np.ndarray,
    max_iter: int = 60,
    tol: float = 1e-7,
) -> np.ndarray:
    """Computes an iterative Shirley inelastic XPS background curve.

    Evaluates ``B(E)`` in ascending binding energy so ``B(E_min) = I_low``,
    ``B(E_max) = I_high``, and ``B(E_i) - I_low`` is proportional to the
    integrated net intensity ``max(I - B, 0)`` at lower binding energies
    ``[E_min, E_i]``.

    Args:
      binding_energy: 1D binding energy grid in eV (ascending or descending).
      intensity: 1D raw photoemission intensity array matching
        ``binding_energy``.
      max_iter: Maximum fixed-point iterations.
      tol: Relative convergence tolerance.

    Returns:
      1D float64 background array matching the order of ``binding_energy``.
    """
    be = np.asarray(binding_energy, dtype=np.float64)
    inten = np.asarray(intensity, dtype=np.float64)
    valid = (
        be.ndim == 1
        and inten.ndim == 1
        and be.shape == inten.shape
        and be.size >= 2
    )
    if not valid:
        raise exceptions.StructureError(
            "Shirley background requires matching 1D arrays with >= 2 points."
        )
    order = np.argsort(be)
    e_asc = be[order]
    y_asc = inten[order]
    i_low = float(y_asc[0])
    i_high = float(y_asc[-1])
    span_e = max(float(e_asc[-1] - e_asc[0]), 1e-12)
    b_cur: np.ndarray = i_low + (i_high - i_low) * (
        (e_asc - e_asc[0]) / span_e
    )
    conv_thresh = float(tol) * max(float(np.ptp(y_asc)), 1.0)

    for _ in range(max(1, int(max_iter))):
        net = np.maximum(y_asc - b_cur, 0.0)
        cum_area = nmr_core.cumulative_trapezoid(e_asc, net)
        tot_area = float(cum_area[-1])
        if tot_area <= 1e-12:
            break
        b_target = i_low + (i_high - i_low) * (cum_area / tot_area)
        b_next = 0.65 * b_target + 0.35 * b_cur
        b_next[0] = i_low
        b_next[-1] = i_high
        if float(np.max(np.abs(b_next - b_cur))) <= conv_thresh:
            b_cur = b_next
            break
        b_cur = b_next

    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(be.size)
    return np.asarray(b_cur[inv_order], dtype=np.float64)


def compute_xps_background(
    be_asc: np.ndarray,
    y_asc: np.ndarray,
    mode: str | None,
) -> np.ndarray:
    """Computes Shirley, linear, or zero XPS baseline on ascending BE."""
    if mode is None:
        return np.zeros_like(y_asc, dtype=np.float64)
    key = mode.strip().lower()
    if key in ("none", "zero", "off"):
        return np.zeros_like(y_asc, dtype=np.float64)
    if key == "linear":
        span = max(float(be_asc[-1] - be_asc[0]), 1e-12)
        return np.asarray(
            y_asc[0] + (y_asc[-1] - y_asc[0]) * ((be_asc - be_asc[0]) / span),
            dtype=np.float64,
        )
    if key == "shirley":
        return compute_shirley_background(be_asc, y_asc)
    raise exceptions.StructureError(
        f"Unsupported XPS background {mode!r}; expected 'shirley', 'linear', "
        "or None."
    )


def resolve_comp_dict_center_amp(
    comp: Mapping[str, Any],
) -> tuple[float, float, str]:
    """Extracts center BE, amplitude/area, and normalization mode from dict."""
    c_raw = comp.get("center", comp.get("position", comp.get("be")))
    c_be = float(comp.get("binding_energy", 0.0) if c_raw is None else c_raw)
    if "area" in comp and "amplitude" not in comp and "height" not in comp:
        return c_be, float(comp["area"]), "area"
    a_raw = comp.get(
        "amplitude", comp.get("height", comp.get("intensity", 1.0))
    )
    return c_be, float(a_raw), "height"


def evaluate_one_xps_component(
    be_asc: np.ndarray,
    comp: Any,
    default_fwhm: float,
    order: np.ndarray,
) -> list[tuple[str, float, np.ndarray]]:
    """Evaluates a single XPS component dict or tuple into ``(lbl, be0, y)``."""
    if isinstance(comp, Mapping):
        c_be, amp, norm_m = resolve_comp_dict_center_amp(comp)
        w_fwhm = float(comp.get("fwhm", default_fwhm))
        prof = str(comp.get("profile", "pseudo_voigt"))
        lbl = str(comp.get("label", f"{c_be:.1f} eV"))
        y_c = spectroscopy.broaden_peaks(
            [c_be], [amp], be_asc, fwhm=w_fwhm, profile=prof, normalize=norm_m
        )
        out = [(lbl, c_be, y_c)]
        if "splitting" in comp or "delta_be" in comp:
            split = float(comp.get("splitting", comp.get("delta_be", 0.0)))
            ratio = float(comp.get("branching_ratio", 0.5))
            lbl2 = str(comp.get("doublet_label", f"{lbl} (SO)"))
            y_c2 = spectroscopy.broaden_peaks(
                [c_be + split],
                [amp * ratio],
                be_asc,
                fwhm=w_fwhm,
                profile=prof,
                normalize=norm_m,
            )
            out.append((lbl2, c_be + split, y_c2))
        return out

    if isinstance(comp, Sequence) and not isinstance(comp, str):
        items = list(comp)
        if len(items) == 2 and isinstance(items[0], str):
            arr = np.asarray(items[1], dtype=np.float64)[order]
            pk_be = float(be_asc[int(np.argmax(arr))])
            return [(items[0], pk_be, arr)]
        if isinstance(items[0], str) and len(items) >= 3:
            lbl, c_be, amp = items[0], float(items[1]), float(items[2])
            w_fwhm = float(items[3]) if len(items) >= 4 else default_fwhm
        elif len(items) >= 2:
            c_be, amp = float(items[0]), float(items[1])
            if len(items) == 3 and isinstance(items[2], str):
                w_fwhm, lbl = default_fwhm, items[2]
            else:
                w_fwhm = float(items[2]) if len(items) >= 3 else default_fwhm
                lbl = str(items[3]) if len(items) >= 4 else f"{c_be:.1f} eV"
        else:
            raise exceptions.StructureError(f"Invalid XPS component {comp!r}.")
        y_c = spectroscopy.broaden_peaks(
            [c_be], [amp], be_asc, fwhm=w_fwhm, profile="pseudo_voigt"
        )
        return [(lbl, c_be, y_c)]

    raise exceptions.StructureError(f"Unsupported XPS component {comp!r}.")


def annotate_xps_components(
    ax: maxes.Axes,
    be_asc: np.ndarray,
    envelope: np.ndarray,
    comp_curves: Sequence[tuple[str, float, np.ndarray]],
    colors: tuple[str, str],
) -> float:
    """Places non-overlapping haloed ink labels above XPS component peaks."""
    ink_col, bg_col = colors
    y_span = max(float(np.ptp(envelope)), 1.0)
    be_span = max(float(be_asc[-1] - be_asc[0]), 1.0)
    placed: list[tuple[float, float]] = []
    top_y = float(np.max(envelope))

    for lbl, c_be, _ in sorted(
        comp_curves, key=lambda item: float(np.max(item[2])), reverse=True
    ):
        if not lbl:
            continue
        y_lbl = float(np.interp(c_be, be_asc, envelope)) + 0.055 * y_span
        for px, py in placed:
            near_x = abs(c_be - px) < 0.14 * be_span
            near_y = abs(y_lbl - py) < 0.09 * y_span
            if near_x and near_y:
                y_lbl = max(y_lbl, py + 0.095 * y_span)
        placed.append((c_be, y_lbl))
        top_y = max(top_y, y_lbl + 0.07 * y_span)
        theme.place_halo_text(
            ax,
            c_be,
            y_lbl,
            lbl,
            color=ink_col,
            bg_color=bg_col,
            fontsize=8.4,
            va="bottom",
            pad=0.07,
            alpha=0.88,
            zorder=6.5,
        )
    return top_y


def prepare_xps_arrays(
    binding_energy: Sequence[float] | np.ndarray | spectroscopy.Spectrum,
    intensity: Sequence[float] | np.ndarray | None,
    components: Sequence[Any] | None,
    fwhm: float,
    background: str | None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[tuple[str, float, np.ndarray]],
    bool,
]:
    """Coerces XPS energy/intensity grids, baseline, and component curves."""
    be_raw: np.ndarray
    y_raw: np.ndarray | None
    if isinstance(binding_energy, spectroscopy.Spectrum) and intensity is None:
        be_raw = np.asarray(binding_energy.x, dtype=np.float64)
        y_raw = np.asarray(binding_energy.y, dtype=np.float64)
    else:
        be_raw = np.asarray(binding_energy, dtype=np.float64)
        y_raw = (
            None
            if intensity is None
            else np.asarray(intensity, dtype=np.float64)
        )
    if be_raw.ndim != 1 or be_raw.size < 2:
        raise exceptions.StructureError(
            "XPS binding_energy must be a 1D array with >= 2 points."
        )
    order = np.argsort(be_raw)
    be_asc = be_raw[order]
    comp_curves: list[tuple[str, float, np.ndarray]] = []
    if components is not None:
        for comp in components:
            comp_curves.extend(
                evaluate_one_xps_component(be_asc, comp, float(fwhm), order)
            )

    has_raw = y_raw is not None
    y_asc: np.ndarray
    if y_raw is not None:
        if y_raw.ndim != 1 or y_raw.shape != be_raw.shape:
            raise exceptions.StructureError(
                "XPS intensity must be a 1D array matching binding_energy."
            )
        y_asc = y_raw[order]
    elif comp_curves:
        y_asc = np.asarray(
            np.sum([c[2] for c in comp_curves], axis=0), dtype=np.float64
        )
    else:
        raise exceptions.StructureError(
            "XPS plot requires intensity data or non-empty components."
        )

    bg_asc = compute_xps_background(be_asc, y_asc, background)
    env_asc: np.ndarray = (
        bg_asc + np.sum([c[2] for c in comp_curves], axis=0)
        if comp_curves
        else y_asc.copy()
    )
    return be_asc, y_asc, bg_asc, env_asc, comp_curves, has_raw


def draw_xps_traces(
    ax: maxes.Axes,
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    comp_curves: Sequence[tuple[str, float, np.ndarray]],
    background: str | None,
    has_raw: bool,
    series: Sequence[str],
    colors: tuple[str, str],
    sw: theme.StrokeHierarchy,
) -> None:
    """Renders XPS baseline, shaded components, raw scatter, and envelope."""
    be_asc, y_asc, bg_asc, env_asc = arrays
    ink_col, bg_col = colors
    guide_col = mat_tern.theme_muted_guide(bg_col)
    env_col = mat_tern.theme_series_color(
        series[5 % len(series)], bg_col, ink_col
    )
    if background is not None and background.strip().lower() not in (
        "none",
        "zero",
        "off",
    ):
        ax.plot(
            be_asc,
            bg_asc,
            color=guide_col,
            linestyle="--",
            lw=sw.guide + 0.15,
            alpha=0.85,
            label=f"{background.strip().title()} BG",
            zorder=2.0,
        )

    for idx, (_, _, y_c) in enumerate(comp_curves):
        c_col = series[(idx + 1) % len(series)]
        if c_col == "#F0E442":
            c_col = series[(idx + 2) % len(series)]
        c_col = mat_tern.theme_series_color(c_col, bg_col, ink_col)
        y_top_c = bg_asc + y_c
        ax.plot(be_asc, y_top_c, color=c_col, lw=sw.overlay, zorder=3.0)
        patterns.shade_between(
            ax,
            be_asc,
            bg_asc,
            y_top_c,
            color=c_col,
            shade=None,
            alpha=0.22,
            zorder=2.3,
        )

    if comp_curves and has_raw:
        stride = max(1, be_asc.size // 90)
        ax.scatter(
            be_asc[::stride],
            y_asc[::stride],
            s=22.0,
            facecolor=bg_col,
            edgecolor=ink_col,
            linewidth=1.0,
            alpha=0.82,
            label="Raw Data",
            zorder=4.2,
        )
        ax.plot(
            be_asc,
            env_asc,
            color=env_col,
            lw=sw.primary,
            label="Total Envelope",
            zorder=4.0,
        )
    else:
        ax.plot(
            be_asc,
            y_asc,
            color=env_col,
            lw=sw.primary,
            label="XPS Spectrum",
            zorder=4.0,
        )


def _draw_xps_residuals(
    ax: maxes.Axes,
    be_asc: np.ndarray,
    resid: np.ndarray,
    top_y: float,
    y_span: float,
    colors: tuple[str, str],
    sw: theme.StrokeHierarchy,
) -> float:
    """Draws the top XPS fit residual strip and returns updated upper y."""
    guide_col, resid_col = colors
    r_base = top_y + 0.14 * y_span
    r_ref = ax.axhline(
        r_base,
        color=guide_col,
        linestyle=":",
        lw=sw.guide,
        zorder=2.0,
    )
    r_ref.set_sketch_params(0.0, 0.0, 0.0)
    ax.plot(
        be_asc,
        r_base + resid,
        color=resid_col,
        lw=sw.guide + 0.15,
        label="Residual",
        zorder=3.8,
    )
    r_top = r_base + float(np.max(np.abs(resid)))
    return max(r_top, top_y + 0.22 * y_span) + 0.28 * y_span


def plot_xps(
    binding_energy: Sequence[float] | np.ndarray | spectroscopy.Spectrum,
    intensity: Sequence[float] | np.ndarray | None = None,
    components: Sequence[Any] | None = None,
    background: str | None = "shirley",
    invert_x: bool = True,
    show_residuals: bool = False,
    fwhm: float = 1.2,
    ax: maxes.Axes | None = None,
    palette: str | Sequence[str] = "okabe_ito",
    preset: str | theme.ThemePreset = "classic",
    title: str | None = None,
) -> maxes.Axes:
    """Plots an XPS core-level spectrum with Shirley background & Voigt fits.

    Args:
      binding_energy: 1D binding energy array in eV (or ``Spectrum`` when
        ``intensity`` is ``None``).
      intensity: 1D photoemission intensity array (optional if ``components``
        are supplied to synthesize the envelope).
      components: Optional sequence of Voigt/GL component dicts or tuples.
      background: ``"shirley"`` (iterative inelastic background), ``"linear"``,
        or ``None``.
      invert_x: Invert horizontal binding-energy axis (standard XPS convention).
      show_residuals: Overlay ``I_raw - I_envelope`` residual strip at top.
      fwhm: Default pseudo-Voigt component FWHM in eV.
      ax: Optional target Matplotlib ``Axes``.
      palette: Series palette name or hex sequence.
      preset: Visual theme preset name or ``ThemePreset``.
      title: Optional plot heading.

    Returns:
      Styled Matplotlib ``Axes`` containing the XPS core-level deconvolution.
    """
    be_asc, y_asc, bg_asc, env_asc, comp_curves, has_raw = prepare_xps_arrays(
        binding_energy, intensity, components, fwhm, background
    )
    series = palettes.get_palette(palette)
    preset_obj = theme.get_preset(preset)
    colors = (preset_obj.ink_color, preset_obj.axes_facecolor)
    sw = theme.stroke_weights(preset)

    with theme.xkcd_style(preset=preset):
        if ax is None:
            _, ax = plt.subplots(figsize=(6.4, 4.4))
        else:
            theme.style_existing_axes(ax, preset=preset)

        draw_xps_traces(
            ax,
            (be_asc, y_asc, bg_asc, env_asc),
            comp_curves,
            background,
            has_raw,
            series,
            colors,
            sw,
        )
        top_y = max(float(np.max(y_asc)), float(np.max(env_asc)))
        y_lo = min(float(np.min(y_asc)), float(np.min(bg_asc)))
        if comp_curves:
            top_y = max(
                top_y,
                annotate_xps_components(
                    ax, be_asc, env_asc, comp_curves, colors
                ),
            )

        y_span = max(top_y - y_lo, 1.0)
        if show_residuals and comp_curves and has_raw:
            resid_colors = (
                mat_tern.theme_muted_guide(colors[1]),
                mat_tern.theme_series_color(
                    series[6 % len(series)], colors[1], colors[0]
                ),
            )
            top_y = _draw_xps_residuals(
                ax, be_asc, y_asc - env_asc, top_y, y_span, resid_colors, sw
            )

        ax.set_xlim(float(be_asc[0]), float(be_asc[-1]))
        if invert_x:
            ax.invert_xaxis()
        ax.set_ylim(y_lo - 0.06 * y_span, top_y + 0.10 * y_span)
        ax.set_xlabel("Binding Energy (eV)")
        ax.set_ylabel("Intensity (counts / arb. units)")
        if comp_curves or background is not None:
            mat_tern.style_legend(
                ax,
                colors[0],
                colors[1],
                loc="upper left",
                frameon=True,
                fontsize=8.3,
            )
        theme.apply_xkcd_spines(ax, preset=preset)
        if title:
            ax.set_title(title, color=colors[0])
    return ax
