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
from scipy import special

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
    *,
    normalize: str = "height",
) -> np.ndarray:
    """Broadens discrete spectral lines onto a continuous grid.

    Args:
      positions: Stick peak centers.
      intensities: Stick peak weights.
      x_grid: 1D evaluation grid.
      fwhm: Full width at half maximum in units of ``x_grid``.
      profile: Lineshape (``"lorentzian"``, ``"gaussian"``, ``"pseudo_voigt"``,
        or ``"voigt"``).
      normalize: ``"height"`` (unit peak amplitude) or ``"area"`` (unit
        integrated area per stick peak).

    Returns:
      Broadened 1D intensity array matching ``x_grid``.

    Raises:
      StructureError: If ``profile``/``normalize`` is unknown or ``fwhm <= 0``.
    """
    pos = np.asarray(positions, dtype=np.float64).ravel()
    amp = np.asarray(intensities, dtype=np.float64).ravel()
    grid = np.asarray(x_grid, dtype=np.float64)
    if pos.shape != amp.shape:
        raise exceptions.StructureError(
            "Peak positions and intensities must have identical lengths."
        )
    if fwhm <= 0.0:
        raise exceptions.StructureError("FWHM must be strictly positive.")
    norm_mode = normalize.strip().lower()
    if norm_mode not in ("height", "area"):
        raise exceptions.StructureError(
            f"Unknown normalization mode {normalize!r}; expected 'height' "
            "or 'area'."
        )
    y_out = np.zeros_like(grid, dtype=np.float64)
    if pos.size == 0:
        return y_out
    dx = grid[:, None] - pos[None, :]
    gamma = fwhm * 0.5
    sigma = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    kind = profile.lower()
    if kind == "lorentzian":
        if norm_mode == "area":
            shape = (gamma / math.pi) / (dx * dx + gamma * gamma)
        else:
            shape = (gamma * gamma) / (dx * dx + gamma * gamma)
    elif kind == "gaussian":
        shape = np.exp(-0.5 * (dx / sigma) ** 2)
        if norm_mode == "area":
            shape = shape / (sigma * math.sqrt(2.0 * math.pi))
    elif kind == "pseudo_voigt":
        if norm_mode == "area":
            lor = (gamma / math.pi) / (dx * dx + gamma * gamma)
            gau = np.exp(-0.5 * (dx / sigma) ** 2) / (
                sigma * math.sqrt(2.0 * math.pi)
            )
        else:
            lor = (gamma * gamma) / (dx * dx + gamma * gamma)
            gau = np.exp(-0.5 * (dx / sigma) ** 2)
        shape = 0.5 * lor + 0.5 * gau
    elif kind == "voigt":
        shape = special.voigt_profile(dx, sigma, gamma)
        if norm_mode == "height":
            peak0 = max(float(special.voigt_profile(0.0, sigma, gamma)), 1e-12)
            shape = shape / peak0
    else:
        raise exceptions.StructureError(
            f"Unknown lineshape profile {profile!r}."
        )
    return shape @ amp


def _structure_factors(
    atoms: structure.AtomsData,
    hkl_v: np.ndarray,
    s_v: np.ndarray,
    debye_waller_b: float,
) -> np.ndarray:
    """Evaluates Cromer-Mann kinematical structure factors F(hkl)."""
    frac = atoms.fractional_positions()
    occ = (
        atoms.occupancies
        if atoms.occupancies is not None
        else np.ones(len(atoms), dtype=np.float64)
    )
    f_hkl = np.zeros(len(hkl_v), dtype=np.complex128)
    for sym in sorted(set(atoms.symbols)):
        mask = np.array([s == sym for s in atoms.symbols], dtype=bool)
        phases = 2.0 * math.pi * (hkl_v @ frac[mask].T)
        geom_sum = np.exp(1j * phases) @ occ[mask]
        f_hkl += scattering.atomic_form_factor(sym, s_v) * geom_sum
    if debye_waller_b > 0.0:
        f_hkl = f_hkl * np.exp(-debye_waller_b * np.square(s_v))
    return f_hkl


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
    if len(hkl_v) == 0:
        return (
            np.zeros((0, 3), dtype=np.int64),
            np.zeros((0,), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
        )
    two_thetas = 2.0 * np.degrees(np.arcsin(sin_v))
    f_hkl = _structure_factors(atoms, hkl_v, s_v, debye_waller_b)
    thetas = np.radians(two_thetas * 0.5)
    lp = (1.0 + np.cos(2.0 * thetas) ** 2) / np.maximum(
        (np.sin(thetas) ** 2) * np.cos(thetas), 1e-12
    )
    return hkl_v, two_thetas, np.real(f_hkl * np.conj(f_hkl)) * lp


def _hkl_rank(hkl: tuple[int, int, int]) -> tuple[int, int, int, int]:
    """Ranks symmetry-equivalent reflections to prefer canonical (hkl)."""
    return (sum(val >= 0 for val in hkl), hkl[0], hkl[1], hkl[2])


def _format_hkl(hkl: tuple[int, int, int]) -> str:
    """Formats Miller index triples compactly or space-separated if needed."""
    if all(0 <= val <= 9 for val in hkl):
        return f"({hkl[0]}{hkl[1]}{hkl[2]})"
    return f"({hkl[0]} {hkl[1]} {hkl[2]})"


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
    if len(atoms) == 0 or atoms.cell is None or atoms.volume() < 1e-8:
        raise exceptions.StructureError(
            "Powder XRD requires a non-empty periodic structure with a "
            "nonzero unit cell."
        )
    tt_min, tt_max = float(two_theta_range[0]), float(two_theta_range[1])
    if wavelength <= 0.0 or step <= 0.0 or tt_min < 0.0 or tt_max <= tt_min:
        raise exceptions.StructureError(
            "Invalid XRD wavelength, step, or two_theta_range bounds."
        )
    hkl_valid, two_thetas, intensities = _xrd_reflections(
        atoms, wavelength, tt_min, tt_max, debye_waller_b
    )
    entries = sorted(
        (
            (
                float(two_thetas[idx]),
                float(intensities[idx]),
                (
                    int(hkl_valid[idx][0]),
                    int(hkl_valid[idx][1]),
                    int(hkl_valid[idx][2]),
                ),
            )
            for idx in range(len(two_thetas))
            if float(intensities[idx]) > 1e-10
        ),
        key=lambda item: item[0],
    )
    grouped: list[tuple[list[float], float, tuple[int, int, int]]] = []
    for tt, inten, hkl_t in entries:
        if grouped and abs(tt - float(np.mean(grouped[-1][0]))) <= 0.01:
            tts, acc, rep_hkl = grouped[-1]
            tts.append(tt)
            best = hkl_t if _hkl_rank(hkl_t) > _hkl_rank(rep_hkl) else rep_hkl
            grouped[-1] = (tts, acc + inten, best)
        else:
            grouped.append(([tt], inten, hkl_t))

    raw_peaks = [
        (float(np.mean(tts)), tot, _format_hkl(rep))
        for tts, tot, rep in grouped
    ]
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
