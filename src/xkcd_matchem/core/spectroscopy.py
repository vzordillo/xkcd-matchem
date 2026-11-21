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
"""Spectroscopy containers, peak broadening, and pure-NumPy powder XRD."""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np

from xkcd_matchem.core import exceptions
from xkcd_matchem.core import structure
from xkcd_matchem.data import elements
from xkcd_matchem.data import scattering


@dataclasses.dataclass(frozen=True)
class Peak:
    """Discrete spectral line or Bragg reflection peak.

    Attributes:
      position: Center coordinate along the spectral axis.
      intensity: Relative or integrated intensity.
      label: Optional peak assignment or ``(hkl)`` label.
    """

    position: float
    intensity: float
    label: str = ""


@dataclasses.dataclass
class Spectrum:
    """1D spectroscopy or diffraction profile with optional discrete peaks.

    Attributes:
      x: Independent axis grid of shape ``(M,)``.
      y: Intensity values of shape ``(M,)``.
      kind: Spectral modality (``"ir"``, ``"raman"``, ``"uvvis"``, ``"xrd"``,
        ``"rdf"``, ``"xvg"``, or ``"generic"``).
      x_label: Axis label for ``x``.
      y_label: Axis label for ``y``.
      peaks: Optional underlying discrete peaks.
      metadata: Free-form parser or simulation metadata.
    """

    x: np.ndarray
    y: np.ndarray
    kind: str = "generic"
    x_label: str = "x"
    y_label: str = "Intensity (arb. units)"
    peaks: list[Peak] = dataclasses.field(default_factory=list)
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        """Coerces ``x`` and ``y`` to 1D float64 arrays."""
        self.x = np.asarray(self.x, dtype=np.float64)
        self.y = np.asarray(self.y, dtype=np.float64)
        if self.x.shape != self.y.shape:
            raise exceptions.StructureError(
                "Spectrum x and y arrays must have identical shapes."
            )


def broaden_peaks(
    positions: np.ndarray | list[float],
    intensities: np.ndarray | list[float],
    x_grid: np.ndarray,
    fwhm: float = 10.0,
    profile: str = "lorentzian",
) -> np.ndarray:
    """Broadens discrete spectral lines onto a continuous grid.

    Args:
      positions: Stick peak centers.
      intensities: Stick peak weights.
      x_grid: 1D evaluation grid.
      fwhm: Full width at half maximum in units of ``x_grid``.
      profile: Lineshape (``"lorentzian"``, ``"gaussian"``, or
        ``"pseudo_voigt"``).

    Returns:
      Broadened 1D intensity array matching ``x_grid``.

    Raises:
      StructureError: If ``profile`` is unknown or ``fwhm <= 0``.
    """
    pos = np.asarray(positions, dtype=np.float64)
    amp = np.asarray(intensities, dtype=np.float64)
    grid = np.asarray(x_grid, dtype=np.float64)
    if fwhm <= 0.0:
        raise exceptions.StructureError("FWHM must be strictly positive.")
    y_out = np.zeros_like(grid, dtype=np.float64)
    if pos.size == 0:
        return y_out
    dx = grid[:, None] - pos[None, :]
    gamma = fwhm * 0.5
    sigma = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    kind = profile.lower()
    if kind == "lorentzian":
        shape = (gamma * gamma) / (dx * dx + gamma * gamma)
    elif kind == "gaussian":
        shape = np.exp(-0.5 * (dx / sigma) ** 2)
    elif kind in ("pseudo_voigt", "voigt"):
        lor = (gamma * gamma) / (dx * dx + gamma * gamma)
        gau = np.exp(-0.5 * (dx / sigma) ** 2)
        shape = 0.5 * lor + 0.5 * gau
    else:
        raise exceptions.StructureError(
            f"Unknown lineshape profile {profile!r}."
        )
    return shape @ amp


def _xrd_reflections(
    atoms: structure.AtomsData,
    wavelength: float,
    tt_min: float,
    tt_max: float,
    debye_waller_b: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Enumerates reciprocal-space reflections and raw intensities."""
    cell = atoms.cell
    if cell is None:
        return np.zeros((0, 3)), np.zeros((0,)), np.zeros((0,))
    recip = 2.0 * math.pi * np.linalg.inv(cell).T
    s_max = min(0.9999, math.sin(math.radians(tt_max * 0.5))) / wavelength
    bounds = [
        max(1, int(math.ceil(2.0 * s_max * float(np.linalg.norm(cell[k]))))) + 1
        for k in range(3)
    ]
    axes = [np.arange(-b, b + 1) for b in bounds]
    gh, gk, gl = np.meshgrid(*axes, indexing="ij")
    hkl_all = np.column_stack([gh.ravel(), gk.ravel(), gl.ravel()])
    hkl_all = hkl_all[np.any(hkl_all != 0, axis=1)]

    s_vals = np.linalg.norm(hkl_all @ recip, axis=1) / (4.0 * math.pi)
    sin_th = s_vals * wavelength
    sin_min = math.sin(math.radians(max(0.01, tt_min) * 0.5))
    valid = (sin_th >= sin_min) & (sin_th <= min(0.9999, s_max * wavelength))
    hkl_v, s_v, sin_v = hkl_all[valid], s_vals[valid], sin_th[valid]
    two_thetas = 2.0 * np.degrees(np.arcsin(sin_v))

    phases = 2.0 * math.pi * (hkl_v @ atoms.fractional_positions().T)
    form_map = {
        sym: scattering.atomic_form_factor(sym, s_v)
        for sym in sorted(set(atoms.symbols))
    }
    f_mat = np.column_stack([form_map[sym] for sym in atoms.symbols])
    if debye_waller_b > 0.0:
        f_mat = f_mat * np.exp(-debye_waller_b * np.square(s_v))[:, None]
    f_hkl = np.sum(f_mat * (np.cos(phases) + 1j * np.sin(phases)), axis=1)
    thetas = np.radians(two_thetas * 0.5)
    lp = (1.0 + np.cos(2.0 * thetas) ** 2) / np.maximum(
        (np.sin(thetas) ** 2) * np.cos(thetas), 1e-12
    )
    return hkl_v, two_thetas, np.real(f_hkl * np.conj(f_hkl)) * lp


def compute_xrd(
    atoms: structure.AtomsData,
    wavelength: float = elements.CODATA2022.cu_kalpha1_angstrom,
    two_theta_range: tuple[float, float] = (10.0, 90.0),
    fwhm: float = 0.35,
    min_intensity: float = 1e-3,
    step: float = 0.05,
    debye_waller_b: float = 0.0,
) -> Spectrum:
    """Computes a kinematical powder XRD pattern for a periodic crystal.

    Enumerates reciprocal-lattice reflections ``(hkl)`` within the Ewald limit,
    evaluates structure factors using Cromer-Mann LA-3816 neutral atomic form
    factors, multiplies by the Lorentz-polarization factor, clusters symmetry-
    equivalent reflections by ``2*theta``, and applies pseudo-Voigt broadening.

    Args:
      atoms: Periodic ``AtomsData`` crystal structure.
      wavelength: X-ray wavelength in Angstrom (defaults to CODATA Cu K-alpha1).
      two_theta_range: ``(min_2theta, max_2theta)`` range in degrees.
      fwhm: Instrumental peak FWHM in degrees.
      min_intensity: Relative threshold (fraction of max = 100) to retain peaks.
      step: Angular grid step size in degrees.
      debye_waller_b: Optional isotropic thermal parameter ``B`` in A^2.

    Returns:
      ``Spectrum`` with ``kind="xrd"`` normalized so the strongest peak is 100.

    Raises:
      StructureError: If ``atoms`` lacks a 3D unit cell.
    """
    if atoms.cell is None or atoms.volume() < 1e-8:
        raise exceptions.StructureError(
            "Powder XRD requires a periodic structure with a nonzero unit cell."
        )
    tt_min, tt_max = float(two_theta_range[0]), float(two_theta_range[1])
    hkl_valid, two_thetas, intensities = _xrd_reflections(
        atoms, wavelength, tt_min, tt_max, debye_waller_b
    )
    grouped: dict[int, tuple[list[float], float, tuple[int, int, int]]] = {}
    for idx, tt in enumerate(two_thetas):
        inten = float(intensities[idx])
        if inten <= 1e-10:
            continue
        bin_key = int(round(tt * 20.0))
        hkl_t = tuple(int(v) for v in hkl_valid[idx])
        if bin_key not in grouped:
            grouped[bin_key] = (
                [float(tt)],
                inten,
                (hkl_t[0], hkl_t[1], hkl_t[2]),
            )
        else:
            tts, acc, rep_hkl = grouped[bin_key]
            tts.append(float(tt))
            best = (
                (hkl_t[0], hkl_t[1], hkl_t[2])
                if sum(x >= 0 for x in hkl_t) > sum(x >= 0 for x in rep_hkl)
                else rep_hkl
            )
            grouped[bin_key] = (tts, acc + inten, best)

    raw_peaks = sorted(
        (
            (float(np.mean(tts)), tot, f"({rep[0]}{rep[1]}{rep[2]})")
            for tts, tot, rep in grouped.values()
        ),
        key=lambda item: item[0],
    )
    max_raw = max((item[1] for item in raw_peaks), default=1.0)
    peaks = [
        Peak(position=pos, intensity=100.0 * val / max_raw, label=lbl)
        for pos, val, lbl in raw_peaks
        if val / max_raw >= min_intensity
    ]
    x_grid = np.arange(tt_min, tt_max + 0.5 * step, step, dtype=np.float64)
    y_vals = broaden_peaks(
        [p.position for p in peaks],
        [p.intensity for p in peaks],
        x_grid,
        fwhm=fwhm,
        profile="pseudo_voigt",
    )
    max_curve = float(np.max(y_vals)) if y_vals.size else 1.0
    if max_curve > 0.0:
        y_vals = 100.0 * y_vals / max_curve
    return Spectrum(
        x=x_grid,
        y=y_vals,
        kind="xrd",
        x_label=r"2$\theta$ (deg)",
        y_label="Relative Intensity (%)",
        peaks=peaks,
        metadata={"wavelength": wavelength, "formula": atoms.formula(True)},
    )
