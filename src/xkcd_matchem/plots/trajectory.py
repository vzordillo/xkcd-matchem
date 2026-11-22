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
"""Molecular dynamics time-series, RDF, RMSD, and GIF trajectory animation."""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
from importlib import util
import pathlib

from matplotlib import animation
from matplotlib import axes as maxes
from matplotlib import pyplot as plt
import numpy as np

from xkcd_matchem.core import exceptions
from xkcd_matchem.core import geometry
from xkcd_matchem.core import structure
from xkcd_matchem.core import trajectory
from xkcd_matchem.render import structure3d
from xkcd_matchem.style import palettes
from xkcd_matchem.style import theme


def plot_md_timeseries(
    traj: trajectory.Trajectory,
    ax: maxes.Axes | None = None,
    quantity: str = "rmsd",
    running_window: int = 0,
    color: str = "#0072B2",
    preset: str | theme.ThemePreset = "classic",
    title: str | None = None,
) -> maxes.Axes:
    """Plots energy, temperature, pressure, or RMSD along an MD trajectory."""
    xs = (
        traj.times
        if traj.times is not None
        else np.arange(len(traj), dtype=np.float64)
    )
    q_key = quantity.lower()
    if q_key == "rmsd":
        ys = traj.rmsd(ref_index=0)
        y_label = "RMSD (A)"
    elif q_key == "energy":
        ys = traj.energy_profile(relative=True, unit="eV")
        y_label = "Delta Energy (eV)"
    elif q_key == "temperature":
        if traj.temperatures is None:
            raise exceptions.StructureError(
                "Trajectory has no stored temperatures."
            )
        ys = traj.temperatures
        y_label = "Temperature (K)"
    elif q_key == "pressure":
        if traj.pressures is None:
            raise exceptions.StructureError(
                "Trajectory has no stored pressures."
            )
        ys = traj.pressures
        y_label = "Pressure"
    else:
        raise exceptions.StructureError(
            f"Unsupported MD quantity {quantity!r}."
        )

    with theme.xkcd_style(preset=preset):
        if ax is None:
            _, ax = plt.subplots(figsize=(6.4, 3.8))
        else:
            theme.style_existing_axes(ax, preset=preset)
        ax.plot(xs, ys, color=color, lw=1.9, alpha=0.88, label=quantity.upper())
        if 1 < running_window <= len(ys):
            kernel = np.ones(running_window, dtype=np.float64) / running_window
            smooth = np.convolve(ys, kernel, mode="valid")
            xs_smooth = np.convolve(xs, kernel, mode="valid")
            ax.plot(
                xs_smooth,
                smooth,
                color="#D55E00",
                lw=2.2,
                label=f"{running_window}-pt mean",
            )
            ax.legend(loc="best", frameon=True)
        ax.set_xlabel("Time / Step")
        ax.set_ylabel(y_label)
        theme.apply_xkcd_spines(ax, preset=preset)
        if title:
            ax.set_title(title)
    return ax


def plot_rdf(
    r_vals: np.ndarray,
    g_vals: np.ndarray,
    ax: maxes.Axes | None = None,
    unit: str = "A",
    color: str = "#009E73",
    preset: str | theme.ThemePreset = "classic",
    title: str | None = None,
) -> maxes.Axes:
    """Plots a radial distribution function ``g(r)`` with ``g(r)=1`` guide."""
    with theme.xkcd_style(preset=preset):
        if ax is None:
            _, ax = plt.subplots(figsize=(6.0, 4.0))
        else:
            theme.style_existing_axes(ax, preset=preset)
        ax.plot(r_vals, g_vals, color=color, lw=2.2)
        ax.fill_between(r_vals, 0.0, g_vals, color=color, alpha=0.18)
        ax.axhline(1.0, color=palettes.MUTED_GRAY, linestyle="--", lw=1.3)
        if len(g_vals) > 0 and len(r_vals) >= 2:
            peak_idx = int(np.argmax(g_vals))
            r_span = max(float(r_vals[-1] - r_vals[0]), 1e-6)
            near_right = float(r_vals[peak_idx]) > float(r_vals[0]) + (
                0.68 * r_span
            )
            text_x = float(r_vals[peak_idx]) + (
                -0.14 * r_span if near_right else 0.12 * r_span
            )
            ax.annotate(
                f"r1 = {r_vals[peak_idx]:.2f} {unit}",
                xy=(float(r_vals[peak_idx]), float(g_vals[peak_idx])),
                xytext=(text_x, float(g_vals[peak_idx]) * 0.9),
                ha="right" if near_right else "left",
                color="#D55E00",
                arrowprops={"arrowstyle": "->", "color": "#D55E00", "lw": 1.3},
            )
        if len(r_vals) >= 2:
            ax.set_xlim(float(r_vals[0]), float(r_vals[-1]))
        ax.set_ylim(bottom=0.0)
        ax.set_xlabel(f"Pair Distance r ({unit})")
        ax.set_ylabel("g(r)")
        theme.apply_xkcd_spines(ax, preset=preset)
        if title:
            ax.set_title(title)
    return ax


def plot_rmsd(
    times: np.ndarray,
    rmsd_vals: np.ndarray,
    ax: maxes.Axes | None = None,
    time_unit: str = "ps",
    color: str = "#0072B2",
    preset: str | theme.ThemePreset = "classic",
    title: str | None = None,
) -> maxes.Axes:
    """Plots a precomputed RMSD curve vs simulation time."""
    with theme.xkcd_style(preset=preset):
        if ax is None:
            _, ax = plt.subplots(figsize=(6.0, 3.8))
        else:
            theme.style_existing_axes(ax, preset=preset)
        ax.plot(times, rmsd_vals, color=color, lw=2.0)
        ax.set_xlabel(f"Time ({time_unit})")
        ax.set_ylabel("RMSD (A)")
        theme.apply_xkcd_spines(ax, preset=preset)
        if title:
            ax.set_title(title)
    return ax


def _trajectory_viewport(
    selected: Sequence[structure.AtomsData],
    cam: geometry.Camera,
) -> tuple[float, float, float, float]:
    """Computes a shared projected 2D viewport across trajectory frames."""
    chunks: list[np.ndarray] = []
    for atoms in selected:
        if len(atoms) == 0:
            continue
        pivot = (
            0.5 * (atoms.cell[0] + atoms.cell[1] + atoms.cell[2])
            if atoms.cell is not None
            else np.mean(atoms.positions, axis=0)
        )
        proj_span = geometry.scene_projection_span(
            atoms.positions, cam, center=pivot, cell=atoms.cell
        )
        xy, _ = geometry.project_points(
            atoms.positions, cam, center=pivot, span=proj_span
        )
        chunks.append(xy)
        if atoms.cell is not None:
            c2d, _, _ = geometry.cell_corners_projected(
                atoms.cell, cam, center=pivot, span=proj_span
            )
            chunks.append(c2d)
    all_pts = np.vstack(chunks)
    margin = 1.45
    return (
        float(np.min(all_pts[:, 0])) - margin,
        float(np.max(all_pts[:, 0])) + margin,
        float(np.min(all_pts[:, 1])) - margin,
        float(np.max(all_pts[:, 1])) + margin,
    )


def animate_trajectory(
    traj: trajectory.Trajectory,
    out_path: str | pathlib.Path,
    fps: int = 8,
    stride: int = 1,
    camera: geometry.Camera | str | None = None,
    palette: str | Mapping[str, str] = "xkcd",
    preset: str | theme.ThemePreset = "classic",
) -> pathlib.Path:
    """Renders an xkcd-styled animated GIF across ``Trajectory`` frames."""
    if util.find_spec("PIL") is None:
        raise exceptions.OptionalDependencyError(
            "GIF animation requires 'pillow'. Install via: "
            "pip install 'xkcd-matchem[animation]'"
        )

    selected = traj.frames[:: max(1, stride)]
    if not selected or all(len(frame) == 0 for frame in selected):
        raise exceptions.StructureError("Trajectory has no frames to animate.")
    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ref_cell = selected[0].cell
    cam = geometry.resolve_camera(camera, cell=ref_cell)
    x_lo, x_hi, y_lo, y_hi = _trajectory_viewport(selected, cam)
    with theme.xkcd_style(preset=preset):
        fig, ax = plt.subplots(figsize=(4.8, 4.2))

        def _update(frame_idx: int) -> None:
            ax.clear()
            atoms = selected[frame_idx]
            structure3d.render_structure_3d(
                atoms, ax=ax, camera=cam, palette=palette
            )
            ax.set_xlim(x_lo, x_hi)
            ax.set_ylim(y_lo, y_hi)
            ax.set_title(f"Frame {frame_idx + 1}/{len(selected)}")

        anim = animation.FuncAnimation(
            fig, _update, frames=len(selected), interval=int(1000 / max(1, fps))
        )
        writer = animation.PillowWriter(fps=fps)
        anim.save(str(out), writer=writer)
        plt.close(fig)
    return out
