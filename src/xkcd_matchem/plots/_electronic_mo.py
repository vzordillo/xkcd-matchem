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
"""Molecular orbital energy-level ladder renderer."""

from __future__ import annotations

from collections.abc import Sequence

from matplotlib import axes as maxes
from matplotlib import pyplot as plt
import numpy as np

from xkcd_matchem.core import electronic
from xkcd_matchem.plots import _materials_ternary as mat_tern
from xkcd_matchem.style import palettes
from xkcd_matchem.style import theme


def draw_mo_group(
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
    preset_obj = theme.get_preset(preset)
    ink_col = preset_obj.ink_color
    bg_col = preset_obj.axes_facecolor
    occ_col = mat_tern.theme_series_color(
        series[5 % len(series)], bg_col, ink_col
    )
    vir_col = mat_tern.theme_series_color(
        series[6 % len(series)], bg_col, ink_col
    )
    guide_col = mat_tern.theme_muted_guide(bg_col)
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
            draw_mo_group(
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
            0.0, color=guide_col, linestyle=":", lw=sw.guide
        )
        z_line.set_sketch_params(0.0, 0.0, 0.0)
        ax.set_xlim(-0.92, 1.32)
        ax.set_ylim(window)
        ax.set_xticks([])
        ax.set_ylabel("Orbital Energy (eV)")
        theme.apply_xkcd_spines(ax, preset=preset)
        if title:
            ax.set_title(title, color=ink_col)
    return ax
