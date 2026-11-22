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
"""First-order NMR multiplet algebra, resonance coercion, and integration."""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
import math
from typing import Any

import numpy as np

from xkcd_matchem.core import exceptions
from xkcd_matchem.core import spectroscopy


NMR_WORD_MAP: dict[str, str] = {
    "singlet": "s",
    "doublet": "d",
    "triplet": "t",
    "quartet": "q",
    "quintet": "quint",
    "pentet": "quint",
    "sextet": "sext",
    "septet": "sept",
    "heptet": "sept",
    "multiplet": "m",
}

NMR_CHAR_NEIGHBORS: dict[str, int] = {
    "d": 1,
    "t": 2,
    "q": 3,
    "p": 4,
    "h": 5,
}


def format_nucleus_symbol(nucleus: str) -> str:
    """Formats leading mass numbers as mathtext superscripts (e.g. ``1H``)."""
    tok = nucleus.strip()
    if "$" in tok or not tok:
        return tok
    idx = 0
    while idx < len(tok) and tok[idx].isdigit():
        idx += 1
    if 0 < idx < len(tok) and tok[idx:].isalpha():
        return f"$^{{{tok[:idx]}}}{tok[idx:]}$"
    return tok


def parse_multiplicity_stages(
    multiplicity: str,
    j_hz: float | Sequence[float] | None,
) -> list[tuple[int, float]]:
    """Converts an NMR multiplicity token and J values into splitting stages."""
    raw_code = multiplicity.strip().lower()
    code = NMR_WORD_MAP.get(raw_code, raw_code)
    if j_hz is None:
        j_list: list[float] = []
    elif isinstance(j_hz, (int, float, np.integer, np.floating)):
        j_list = [float(j_hz)]
    else:
        j_list = [float(val) for val in j_hz]

    if code in ("", "s", "br", "brs"):
        return []
    named_n = {"quint": 4, "pent": 4, "sext": 5, "sept": 6, "hept": 6}
    if code in named_n:
        j0 = j_list[0] if j_list else 7.0
        return [(named_n[code], j0)] if j0 > 0.0 else []
    if code == "m":
        if len(j_list) >= 2:
            return [(1, jk) for jk in j_list if jk > 0.0]
        j0 = j_list[0] if (j_list and j_list[0] > 0.0) else 6.5
        return [(4, j0)]
    if all(ch in NMR_CHAR_NEIGHBORS for ch in code):
        stages: list[tuple[int, float]] = []
        for idx, ch in enumerate(code):
            if idx < len(j_list):
                j_val = j_list[idx]
            elif j_list:
                j_val = j_list[-1] * (0.5 ** (idx - len(j_list) + 1))
            else:
                j_val = 7.0
            if j_val > 0.0:
                stages.append((NMR_CHAR_NEIGHBORS[ch], j_val))
        return stages
    raise exceptions.StructureError(
        f"Unsupported NMR multiplicity {multiplicity!r}."
    )


def expand_nmr_multiplet(
    shift_ppm: float,
    integral: float = 1.0,
    multiplicity: str = "s",
    j_hz: float | Sequence[float] | None = None,
    spectrometer_mhz: float = 400.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthesizes exact first-order Pascal/doublet NMR stick lines.

    Args:
      shift_ppm: Resonance chemical shift center in ppm.
      integral: Total integrated proton/spin count for the multiplet.
      multiplicity: Splitting code (``"s"``, ``"d"``, ``"t"``, ``"q"``,
        ``"quint"``, ``"sext"``, ``"sept"``, ``"dd"``, ``"dt"``, ``"td"``,
        ``"ddd"``, ``"m"``).
      j_hz: Scalar coupling constant(s) ``J`` in Hz.
      spectrometer_mhz: Larmor frequency ``nu_0`` in MHz (``> 0``).

    Returns:
      Tuple ``(positions_ppm, stick_areas)`` of sorted 1D float64 arrays whose
      weights sum to ``integral``.
    """
    mhz = float(spectrometer_mhz)
    if mhz <= 0.0:
        raise exceptions.StructureError("spectrometer_mhz must be > 0.")
    area = float(integral)
    if area < 0.0:
        raise exceptions.StructureError("NMR integral must be non-negative.")
    stages = parse_multiplicity_stages(multiplicity, j_hz)
    sticks: list[tuple[float, float]] = [(float(shift_ppm), area)]
    for n_eq, j_val in stages:
        delta_ppm = float(j_val) / mhz
        denom = float(2**n_eq)
        nxt: list[tuple[float, float]] = []
        for pos0, w0 in sticks:
            for m_idx in range(n_eq + 1):
                offset = (float(m_idx) - 0.5 * float(n_eq)) * delta_ppm
                wt = w0 * float(math.comb(n_eq, m_idx)) / denom
                nxt.append((pos0 + offset, wt))
        sticks = nxt

    sticks.sort(key=lambda item: item[0])
    merged_pos: list[float] = []
    merged_wt: list[float] = []
    for pos, wt in sticks:
        if merged_pos and abs(pos - merged_pos[-1]) <= 1e-10:
            merged_wt[-1] += wt
        else:
            merged_pos.append(pos)
            merged_wt.append(wt)
    return (
        np.asarray(merged_pos, dtype=np.float64),
        np.asarray(merged_wt, dtype=np.float64),
    )


def parse_resonance_entry(
    entry: Any,
) -> tuple[float, float, str, float | tuple[float, ...], str]:
    """Parses one dict or tuple NMR resonance specification."""
    if isinstance(entry, Mapping):
        shift = float(
            entry.get("shift", entry.get("ppm", entry.get("delta", 0.0)))
        )
        integ = float(
            entry.get("integral", entry.get("n_h", entry.get("protons", 1.0)))
        )
        mult = str(entry.get("multiplicity", entry.get("mult", "s")))
        raw_j = entry.get("j_hz", entry.get("couplings_hz", entry.get("j")))
        lbl = str(entry.get("label", ""))
    elif isinstance(entry, Sequence) and not isinstance(entry, str):
        items = list(entry)
        if len(items) < 2:
            raise exceptions.StructureError(
                "NMR resonance tuple requires at least (shift, integral)."
            )
        shift = float(items[0])
        if isinstance(items[1], str):
            integ, mult = 1.0, items[1]
            raw_j = items[2] if len(items) >= 3 else 0.0
            lbl = str(items[3]) if len(items) >= 4 else ""
        else:
            integ = float(items[1])
            mult = str(items[2]) if len(items) >= 3 else "s"
            raw_j = items[3] if len(items) >= 4 else 0.0
            lbl = str(items[4]) if len(items) >= 5 else ""
    else:
        raise exceptions.StructureError(
            f"Invalid NMR resonance item {entry!r}."
        )

    if raw_j is None:
        j_norm: float | tuple[float, ...] = 0.0
    elif isinstance(raw_j, (int, float, np.integer, np.floating)):
        j_norm = float(raw_j)
    else:
        j_norm = tuple(float(v) for v in raw_j)
    return shift, integ, mult, j_norm, lbl


def coerce_nmr_resonances(
    shifts_ppm: Sequence[Any] | np.ndarray,
    couplings_hz: Sequence[Any] | float | None,
    multiplicities: Sequence[str] | None,
    integrals: Sequence[float] | np.ndarray | None,
    labels: Sequence[str | None] | None,
) -> list[tuple[float, float, str, float | tuple[float, ...], str]]:
    """Normalizes structured or parallel-array NMR resonance inputs."""
    seq = list(shifts_ppm)
    if not seq:
        raise exceptions.StructureError("shifts_ppm cannot be empty.")
    first = seq[0]
    if isinstance(first, Mapping) or (
        isinstance(first, Sequence) and not isinstance(first, str)
    ):
        return [parse_resonance_entry(item) for item in seq]

    n_res = len(seq)
    mult_list = (
        [str(m) for m in multiplicities]
        if multiplicities is not None
        else ["s"] * n_res
    )
    int_list = (
        [float(v) for v in integrals]
        if integrals is not None
        else [1.0] * n_res
    )
    lbl_list = (
        ["" if v is None else str(v) for v in labels]
        if labels is not None
        else [""] * n_res
    )
    if couplings_hz is None:
        j_seq: list[Any] = [0.0] * n_res
    elif isinstance(couplings_hz, (int, float, np.integer, np.floating)):
        j_seq = [float(couplings_hz)] * n_res
    else:
        j_seq = list(couplings_hz)
    same_len = (
        len(mult_list) == n_res
        and len(int_list) == n_res
        and len(lbl_list) == n_res
        and len(j_seq) == n_res
    )
    if not same_len:
        raise exceptions.StructureError(
            "NMR shifts, couplings, multiplicities, and integrals must match."
        )
    out: list[tuple[float, float, str, float | tuple[float, ...], str]] = []
    for k in range(n_res):
        rj = j_seq[k]
        j_val: float | tuple[float, ...] = (
            0.0
            if rj is None
            else (
                float(rj)
                if isinstance(rj, (int, float, np.integer, np.floating))
                else tuple(float(v) for v in rj)
            )
        )
        out.append(
            (float(seq[k]), int_list[k], mult_list[k], j_val, lbl_list[k])
        )
    return out


def cumulative_trapezoid(x_arr: np.ndarray, y_arr: np.ndarray) -> np.ndarray:
    """Computes exact 1D cumulative trapezoidal integral starting at 0."""
    if x_arr.size < 2:
        return np.zeros_like(y_arr, dtype=np.float64)
    dx = np.abs(np.diff(x_arr))
    seg = 0.5 * (y_arr[:-1] + y_arr[1:]) * dx
    return np.concatenate([np.zeros(1, dtype=np.float64), np.cumsum(seg)])


def spectrum_to_nmr_bundle(
    spec: spectroscopy.Spectrum,
    ppm_range: tuple[float, float] | None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[tuple[float, float, str, float | tuple[float, ...], str]],
    tuple[float, float],
]:
    """Extracts sorted NMR arrays and ppm bounds from a ``Spectrum``."""
    order = np.argsort(spec.x)
    ppm_grid = np.asarray(spec.x[order], dtype=np.float64)
    y_curve = np.asarray(spec.y[order], dtype=np.float64)
    stick_pos = np.asarray([p.position for p in spec.peaks], dtype=np.float64)
    stick_wt = np.asarray([p.intensity for p in spec.peaks], dtype=np.float64)
    lo_p = float(ppm_grid[0]) if ppm_grid.size else -0.2
    hi_p = float(ppm_grid[-1]) if ppm_grid.size else 10.0
    if ppm_range is not None:
        lo_p = min(float(ppm_range[0]), float(ppm_range[1]))
        hi_p = max(float(ppm_range[0]), float(ppm_range[1]))
    return ppm_grid, y_curve, stick_pos, stick_wt, [], (lo_p, hi_p)


def prepare_nmr_data(
    shifts_ppm: spectroscopy.Spectrum | Sequence[Any] | np.ndarray,
    couplings_hz: Sequence[Any] | float | None,
    multiplicities: Sequence[str] | None,
    integrals: Sequence[float] | np.ndarray | None,
    labels: Sequence[str | None] | None,
    cfg: tuple[float, tuple[float, float] | None, float, bool],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[tuple[float, float, str, float | tuple[float, ...], str]],
    tuple[float, float],
]:
    """Builds NMR grid, Lorentzian trace, stick lines, and ppm bounds."""
    mhz, ppm_range, fwhm_ppm, show_tms = cfg
    if isinstance(shifts_ppm, spectroscopy.Spectrum):
        return spectrum_to_nmr_bundle(shifts_ppm, ppm_range)

    resonances = coerce_nmr_resonances(
        shifts_ppm, couplings_hz, multiplicities, integrals, labels
    )
    pairs = [
        expand_nmr_multiplet(sh, integ, mult, jv, mhz)
        for sh, integ, mult, jv, _ in resonances
    ]
    stick_pos = np.concatenate([p[0] for p in pairs])
    stick_wt = np.concatenate([p[1] for p in pairs])
    if ppm_range is not None:
        lo_p = min(float(ppm_range[0]), float(ppm_range[1]))
        hi_p = max(float(ppm_range[0]), float(ppm_range[1]))
    else:
        min_s, max_s = float(np.min(stick_pos)), float(np.max(stick_pos))
        pad = max(0.35, 0.12 * (max_s - min_s))
        lo_p = min(-0.25, min_s - pad) if show_tms else (min_s - pad)
        hi_p = max_s + pad
    raw_n = int(round((hi_p - lo_p) / max(fwhm_ppm * 0.22, 1e-4)))
    ppm_grid = np.linspace(lo_p, hi_p, min(max(1600, raw_n), 8000))
    y_curve = spectroscopy.broaden_peaks(
        stick_pos,
        stick_wt,
        ppm_grid,
        fwhm=fwhm_ppm,
        profile="lorentzian",
        normalize="area",
    )
    return ppm_grid, y_curve, stick_pos, stick_wt, resonances, (lo_p, hi_p)
