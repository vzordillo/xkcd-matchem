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
"""File format detection and parser registry."""

from __future__ import annotations

from collections.abc import Callable
import dataclasses
import pathlib
from typing import Any

from xkcd_matchem.core import exceptions


@dataclasses.dataclass(frozen=True)
class FormatSpec:
    """Metadata and entrypoints for a registered chemistry/materials format.

    Attributes:
      name: Canonical lowercase format key (e.g., ``"xyz"``, ``"vasp"``).
      extensions: Supported file suffixes including leading dots.
      filenames: Exact basename matches (e.g., ``"POSCAR"``, ``"CHGCAR"``).
      reader: Callable reading a ``pathlib.Path`` into an xkcd-matchem object.
      writer: Optional callable ``(obj, pathlib.Path) -> None``.
      description: Concise summary of supported content.
    """

    name: str
    extensions: tuple[str, ...]
    filenames: tuple[str, ...]
    reader: Callable[[pathlib.Path], Any]
    writer: Callable[[Any, pathlib.Path], None] | None = None
    description: str = ""


# Global parser registry populated at module import time for built-in formats.
_FORMAT_REGISTRY: dict[str, FormatSpec] = {}


def register_format(spec: FormatSpec) -> None:
    """Registers a ``FormatSpec`` in the global format lookup table."""
    _FORMAT_REGISTRY[spec.name.lower()] = spec


def list_formats() -> tuple[FormatSpec, ...]:
    """Returns all registered formats sorted alphabetically by name."""
    return tuple(_FORMAT_REGISTRY[key] for key in sorted(_FORMAT_REGISTRY))


def get_format(name: str) -> FormatSpec:
    """Retrieves a registered ``FormatSpec`` by canonical name.

    Args:
      name: Format key (case-insensitive).

    Returns:
      Matching ``FormatSpec``.

    Raises:
      UnsupportedFormatError: If ``name`` is not registered.
    """
    key = name.lower().strip()
    if key not in _FORMAT_REGISTRY:
        available = ", ".join(sorted(_FORMAT_REGISTRY))
        raise exceptions.UnsupportedFormatError(
            f"Unsupported format {name!r}. Registered: {available}"
        )
    return _FORMAT_REGISTRY[key]


_CONTENT_SIGNATURES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("Gaussian", "orientation:"), "gaussian_log"),
    (("* O   R   C   A *",), "orca_out"),
    (("FINAL SINGLE POINT ENERGY",), "orca_out"),
    (("&plot", "nbnd"), "qe_bands"),
    (("dos(E)",), "qe_dos"),
    (("E (eV)", "EFermi"), "qe_dos"),
    (("pwscf",), "qe_out"),
    (("Quantum ESPRESSO",), "qe_out"),
    (("ITEM: TIMESTEP",), "lammpstrj"),
    (("xTB", "TOTAL ENERGY"), "xtb_out"),
)


def _match_by_name_or_suffix(p: pathlib.Path) -> FormatSpec | None:
    """Matches exact/substring filename or suffix against registered formats."""
    upper_name = p.name.upper()
    for spec in _FORMAT_REGISTRY.values():
        if any(
            upper_name == fname.upper() or fname.upper() in upper_name
            for fname in spec.filenames
        ):
            return spec
    suffix = p.suffix.lower()
    if suffix:
        for spec in _FORMAT_REGISTRY.values():
            if suffix in spec.extensions:
                return spec
    return None


def detect_format(path: str | pathlib.Path) -> FormatSpec:
    """Detects the format of a file path from filename, extension, or content.

    Args:
      path: Input file path.

    Returns:
      Matching ``FormatSpec``.

    Raises:
      UnsupportedFormatError: If no registered format matches ``path``.
    """
    p = pathlib.Path(path)
    matched = _match_by_name_or_suffix(p)
    if matched is not None:
        return matched
    if p.is_file():
        head = p.read_text(encoding="utf-8", errors="replace")[:4096]
        low_head = head.lower()
        for tokens, fmt_name in _CONTENT_SIGNATURES:
            if all(
                (tok in low_head if tok == "pwscf" else tok in head)
                for tok in tokens
            ):
                return get_format(fmt_name)
    raise exceptions.UnsupportedFormatError(
        f"Cannot detect format for file: {p}"
    )
