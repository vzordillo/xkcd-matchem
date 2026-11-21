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

from importlib import util
import pathlib

from matplotlib import animation
from matplotlib import axes as maxes
from matplotlib import pyplot as plt
import numpy as np

from xkcd_matchem.core import exceptions
from xkcd_matchem.core import geometry
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
    preset: str = "classic",
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
    else:
        raise exceptions.StructureError(
            f"Unsupported MD quantity {quantity!r}."
        )

    with theme.xkcd_style(preset=preset):
        if ax is None:
            _, ax = plt.subplots(figsize=(6.4, 3.8))
        ax.plot(xs, ys, color=color, lw=1.9, alpha=0.88, label=quantity.upper())
        if 1 < running_window <= len(ys):
            kernel = np.ones(running_window, dtype=np.float64) / running_window
            smooth = np.convolve(ys, kernel, mode="valid")
            ax.plot(
                xs[running_window - 1 :],
                smooth,
                color="#D55E00",
                lw=2.2,
                label=f"{running_window}-pt mean",
            )
            ax.legend(loc="best", frameon=True)
        ax.set_xlabel("Time / Step")
        ax.set_ylabel(y_label)
        theme.apply_xkcd_spines(ax)
        if title:
            ax.set_title(title)
    return ax


def plot_rdf(
    r_vals: np.ndarray,
    g_vals: np.ndarray,
    ax: maxes.Axes | None = None,
    unit: str = "A",
    color: str = "#009E73",
    preset: str = "classic",
    title: str | None = None,
) -> maxes.Axes:
    """Plots a radial distribution function ``g(r)`` with ``g(r)=1`` guide."""
    with theme.xkcd_style(preset=preset):
        if ax is None:
            _, ax = plt.subplots(figsize=(6.0, 4.0))
        ax.plot(r_vals, g_vals, color=color, lw=2.2)
        ax.fill_between(r_vals, 0.0, g_vals, color=color, alpha=0.18)
        ax.axhline(1.0, color=palettes.MUTED_GRAY, linestyle="--", lw=1.3)
        if len(g_vals) > 0:
            peak_idx = int(np.argmax(g_vals))
            ax.annotate(
                f"r1 = {r_vals[peak_idx]:.2f} {unit}",
                xy=(r_vals[peak_idx], g_vals[peak_idx]),
                xytext=(
                    r_vals[peak_idx] + 0.12 * float(r_vals[-1] - r_vals[0]),
                    g_vals[peak_idx] * 0.9,
                ),
                color="#D55E00",
                arrowprops={"arrowstyle": "->", "color": "#D55E00", "lw": 1.3},
            )
        ax.set_xlabel(f"Pair Distance r ({unit})")
        ax.set_ylabel("g(r)")
        theme.apply_xkcd_spines(ax)
        if title:
            ax.set_title(title)
    return ax


def plot_rmsd(
    times: np.ndarray,
    rmsd_vals: np.ndarray,
    ax: maxes.Axes | None = None,
    time_unit: str = "ps",
    color: str = "#0072B2",
    preset: str = "classic",
    title: str | None = None,
) -> maxes.Axes:
    """Plots a precomputed RMSD curve vs simulation time."""
    with theme.xkcd_style(preset=preset):
        if ax is None:
            _, ax = plt.subplots(figsize=(6.0, 3.8))
        ax.plot(times, rmsd_vals, color=color, lw=2.0)
        ax.set_xlabel(f"Time ({time_unit})")
        ax.set_ylabel("RMSD (A)")
        theme.apply_xkcd_spines(ax)
        if title:
            ax.set_title(title)
    return ax


def animate_trajectory(
    traj: trajectory.Trajectory,
    out_path: str | pathlib.Path,
    fps: int = 8,
    stride: int = 1,
    camera: geometry.Camera | None = None,
    palette: str = "xkcd",
    preset: str = "classic",
) -> pathlib.Path:
    """Renders an xkcd-styled animated GIF across ``Trajectory`` frames."""
    if util.find_spec("PIL") is None:
        raise exceptions.OptionalDependencyError(
            "GIF animation requires 'pillow'. Install via: "
            "pip install 'xkcd-matchem[animation]'"
        )

    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    selected = traj.frames[:: max(1, stride)]
    cam = geometry.Camera() if camera is None else camera
    with theme.xkcd_style(preset=preset):
        fig, ax = plt.subplots(figsize=(4.8, 4.2))

        def _update(frame_idx: int) -> None:
            ax.clear()
            atoms = selected[frame_idx]
            structure3d.render_structure_3d(
                atoms, ax=ax, camera=cam, palette=palette
            )
            ax.set_title(f"Frame {frame_idx + 1}/{len(selected)}")

        anim = animation.FuncAnimation(
            fig, _update, frames=len(selected), interval=int(1000 / max(1, fps))
        )
        writer = animation.PillowWriter(fps=fps)
        anim.save(str(out), writer=writer)
        plt.close(fig)
    return out
