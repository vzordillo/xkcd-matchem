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
"""Canonical internal unit contract, zero-dep converters, and provenance."""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
from typing import Any
from typing import overload

import numpy as np

from xkcd_matchem import _version
from xkcd_matchem.core import exceptions
from xkcd_matchem.data import elements


__all__ = [
    "INTERNAL_UNITS",
    "ScientificProvenance",
    "convert_energy",
    "convert_pressure",
    "convert_time",
]


INTERNAL_UNITS: dict[str, str] = {
    "position": "A",
    "cell": "A",
    "energy": "eV",
    "force": "eV/A",
    "pressure": "GPa",
    "time": "fs",
    "reciprocal_length": "A^-1",
    "temperature": "K",
    "potential": "V",
}

_ATOMIC_TIME_TO_FS = 0.024188843265864
_HA_BOHR3_TO_GPA = (
    elements.CODATA2022.hartree_to_ev
    / (elements.CODATA2022.bohr_to_angstrom**3)
) * elements.CODATA2022.ev_per_a3_to_gpa

_ENERGY_TO_EV: dict[str, float] = {
    "ev": 1.0,
    "mev": 1e-3,
    "ha": elements.CODATA2022.hartree_to_ev,
    "hartree": elements.CODATA2022.hartree_to_ev,
    "eh": elements.CODATA2022.hartree_to_ev,
    "ry": elements.CODATA2022.rydberg_to_ev,
    "rydberg": elements.CODATA2022.rydberg_to_ev,
    "kj/mol": 1.0 / elements.CODATA2022.ev_to_kj_mol,
    "kcal/mol": 1.0 / elements.CODATA2022.ev_to_kcal_mol,
    "j": 1.0 / elements.CODATA2022.ev_to_joule,
    "joule": 1.0 / elements.CODATA2022.ev_to_joule,
    "cm-1": 1.0 / elements.CODATA2022.ev_to_cm_inv,
    "cm^-1": 1.0 / elements.CODATA2022.ev_to_cm_inv,
}

_TIME_TO_FS: dict[str, float] = {
    "fs": 1.0,
    "femtosecond": 1.0,
    "femtoseconds": 1.0,
    "ps": 1e3,
    "picosecond": 1e3,
    "picoseconds": 1e3,
    "ns": 1e6,
    "nanosecond": 1e6,
    "nanoseconds": 1e6,
    "as": 1e-3,
    "attosecond": 1e-3,
    "attoseconds": 1e-3,
    "s": 1e15,
    "sec": 1e15,
    "second": 1e15,
    "seconds": 1e15,
    "au": _ATOMIC_TIME_TO_FS,
    "a.u.": _ATOMIC_TIME_TO_FS,
    "atomic_time": _ATOMIC_TIME_TO_FS,
}

_PRESSURE_TO_GPA: dict[str, float] = {
    "gpa": 1.0,
    "mpa": 1e-3,
    "kpa": 1e-6,
    "pa": 1e-9,
    "bar": 1e-4,
    "kbar": 0.1,
    "atm": 1.01325e-4,
    "ev/a^3": elements.CODATA2022.ev_per_a3_to_gpa,
    "ev/a3": elements.CODATA2022.ev_per_a3_to_gpa,
    "ha/bohr^3": _HA_BOHR3_TO_GPA,
    "ha/bohr3": _HA_BOHR3_TO_GPA,
}


def _normalize_unit_key(unit: str) -> str:
    """Normalizes unit strings for case- and symbol-insensitive lookup."""
    key = unit.strip().lower().replace(" ", "")
    key = key.replace("\u00e5", "a").replace("\u212b", "a")
    return key.replace("angstrom", "a")


def _lookup_scale(
    table: Mapping[str, float],
    from_unit: str,
    to_unit: str,
    quantity: str,
) -> float:
    """Computes multiplicative conversion factor between two units."""
    src = _normalize_unit_key(from_unit)
    dst = _normalize_unit_key(to_unit)
    if src not in table:
        raise exceptions.StructureError(
            f"Unsupported {quantity} unit {from_unit!r}."
        )
    if dst not in table:
        raise exceptions.StructureError(
            f"Unsupported {quantity} unit {to_unit!r}."
        )
    if src == dst:
        return 1.0
    return table[src] / table[dst]


@overload
def convert_energy(
    values: float,
    from_unit: str,
    to_unit: str = "eV",
) -> float: ...


@overload
def convert_energy(
    values: np.ndarray | Sequence[float],
    from_unit: str,
    to_unit: str = "eV",
) -> np.ndarray: ...


def convert_energy(
    values: float | np.ndarray | Sequence[float],
    from_unit: str,
    to_unit: str = "eV",
) -> float | np.ndarray:
    """Converts scalar or array energies using CODATA 2022 factors."""
    scale = _lookup_scale(_ENERGY_TO_EV, from_unit, to_unit, "energy")
    if isinstance(values, (int, float, np.number)) and not isinstance(
        values, bool
    ):
        return float(values) * scale
    arr = np.asarray(values, dtype=np.float64)
    return np.asarray(arr * scale, dtype=np.float64)


@overload
def convert_time(
    values: float,
    from_unit: str,
    to_unit: str = "fs",
) -> float: ...


@overload
def convert_time(
    values: np.ndarray | Sequence[float],
    from_unit: str,
    to_unit: str = "fs",
) -> np.ndarray: ...


def convert_time(
    values: float | np.ndarray | Sequence[float],
    from_unit: str,
    to_unit: str = "fs",
) -> float | np.ndarray:
    """Converts scalar or array times across fs, ps, ns, as, s, and a.u."""
    scale = _lookup_scale(_TIME_TO_FS, from_unit, to_unit, "time")
    if isinstance(values, (int, float, np.number)) and not isinstance(
        values, bool
    ):
        return float(values) * scale
    arr = np.asarray(values, dtype=np.float64)
    return np.asarray(arr * scale, dtype=np.float64)


@overload
def convert_pressure(
    values: float,
    from_unit: str,
    to_unit: str = "GPa",
) -> float: ...


@overload
def convert_pressure(
    values: np.ndarray | Sequence[float],
    from_unit: str,
    to_unit: str = "GPa",
) -> np.ndarray: ...


def convert_pressure(
    values: float | np.ndarray | Sequence[float],
    from_unit: str,
    to_unit: str = "GPa",
) -> float | np.ndarray:
    """Converts scalar or array pressures across GPa, bar, atm, Pa, & eV/A^3."""
    scale = _lookup_scale(_PRESSURE_TO_GPA, from_unit, to_unit, "pressure")
    if isinstance(values, (int, float, np.number)) and not isinstance(
        values, bool
    ):
        return float(values) * scale
    arr = np.asarray(values, dtype=np.float64)
    return np.asarray(arr * scale, dtype=np.float64)


@dataclasses.dataclass(frozen=True)
class ScientificProvenance:
    """Immutable metadata record for numerical analysis & parser provenance.

    Attributes:
      algorithm: Identifier or name of the numerical model/estimator.
      units: Mapping of physical quantities to unit symbols.
      source: Optional input file path or origin identifier.
      parser: Optional parser or adapter name.
      parameters: Immutable ``(key, value)`` parameter pairs (or mapping input).
      assumptions: Explicit physical/numerical modeling assumptions.
      package_version: ``xkcd-matchem`` version string.
    """

    algorithm: str
    units: Mapping[str, str] = dataclasses.field(
        default_factory=lambda: dict(INTERNAL_UNITS)
    )
    source: str | None = None
    parser: str | None = None
    parameters: tuple[tuple[str, Any], ...] | Mapping[str, Any] = ()
    assumptions: Sequence[str] = ()
    package_version: str = _version.__version__

    def __post_init__(self) -> None:
        """Freezes mutable input mappings and sequences into immutable forms."""
        if not self.algorithm or not str(self.algorithm).strip():
            raise exceptions.StructureError(
                "ScientificProvenance algorithm must be a non-empty string."
            )
        norm_units = {str(k): str(v) for k, v in dict(self.units).items()}
        if isinstance(self.parameters, Mapping):
            norm_params = tuple(
                (str(k), v) for k, v in self.parameters.items()
            )
        else:
            norm_params = tuple((str(k), v) for k, v in self.parameters)
        norm_assumptions = tuple(str(item) for item in self.assumptions)
        object.__setattr__(self, "units", norm_units)
        object.__setattr__(self, "parameters", norm_params)
        object.__setattr__(self, "assumptions", norm_assumptions)

    def to_dict(self) -> dict[str, Any]:
        """Serializes provenance metadata into a plain JSON-ready dictionary."""
        param_pairs = (
            tuple(self.parameters.items())
            if isinstance(self.parameters, Mapping)
            else self.parameters
        )
        return {
            "algorithm": self.algorithm,
            "units": dict(self.units),
            "source": self.source,
            "parser": self.parser,
            "parameters": dict(param_pairs),
            "assumptions": list(self.assumptions),
            "package_version": self.package_version,
        }
