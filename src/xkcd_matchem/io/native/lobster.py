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
"""Native LOBSTER parser for COHPCAR.lobster and COOPCAR.lobster files."""

from __future__ import annotations

import pathlib

import numpy as np

from xkcd_matchem.core import electronic
from xkcd_matchem.core import exceptions
from xkcd_matchem.io import registry


def _try_parse_floats(line: str) -> list[float] | None:
    """Attempts to split a whitespace line exclusively into floats."""
    toks = line.split()
    if not toks:
        return None
    try:
        return [float(tok) for tok in toks]
    except ValueError:
        return None


def _read_lobster_header(
    lines: list[str],
    p: pathlib.Path,
) -> tuple[bool, int, int, list[str], int]:
    """Parses header metadata and interaction labels from a COHPCAR stream."""
    if len(lines) < 4:
        raise exceptions.ParseError(f"Truncated LOBSTER file: {p}")
    is_coop = "coop" in lines[0].lower() or "overlap" in lines[0].lower()
    meta_toks = lines[1].split()
    if len(meta_toks) < 2:
        raise exceptions.ParseError(
            f"Malformed LOBSTER count header in {p}: {lines[1]!r}"
        )
    try:
        n_pairs = int(meta_toks[0])
        n_spins = int(meta_toks[1])
    except ValueError as exc:
        raise exceptions.ParseError(
            f"Invalid integer counts in LOBSTER header: {lines[1]!r}"
        ) from exc
    if n_pairs < 1 or n_spins not in (1, 2):
        raise exceptions.ParseError(
            f"Unsupported LOBSTER pair/spin counts ({n_pairs}, {n_spins})."
        )

    labels: list[str] = []
    cursor = 2
    while cursor < len(lines) and len(labels) < n_pairs:
        stripped = lines[cursor].strip()
        if not stripped:
            cursor += 1
            continue
        if _try_parse_floats(stripped) is not None and labels:
            break
        labels.append(stripped)
        cursor += 1
    while len(labels) < n_pairs:
        labels.append(
            "Average" if not labels else f"Interaction_{len(labels)}"
        )
    return is_coop, n_pairs, n_spins, labels, cursor


def read_cohpcar(path: str | pathlib.Path) -> electronic.CohpData:
    """Reads a LOBSTER ``COHPCAR.lobster`` or ``COOPCAR.lobster`` file.

    Args:
      path: Path to a LOBSTER pCOHP or pCOOP bonding table.

    Returns:
      Populated ``CohpData`` instance preserving all bond labels and ICOHP.

    Raises:
      ParseError: If the file cannot be decoded as a LOBSTER table.
    """
    p = pathlib.Path(path)
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    is_coop, n_pairs, n_spins, labels, cursor = _read_lobster_header(lines, p)

    expected_cols = 1 + 2 * n_pairs * n_spins
    rows: list[list[float]] = []
    for raw in lines[cursor:]:
        vals = _try_parse_floats(raw)
        if vals is None:
            continue
        if len(vals) < expected_cols:
            raise exceptions.ParseError(
                f"LOBSTER row has {len(vals)} columns, expected "
                f"{expected_cols} for {n_pairs} interactions and "
                f"{n_spins} spins."
            )
        rows.append(vals)
    if not rows:
        raise exceptions.ParseError(f"No numeric COHP rows found in {p}.")

    arr = np.array(rows, dtype=np.float64)
    energies = arr[:, 0]
    cohp: dict[str, np.ndarray] = {}
    icohp: dict[str, np.ndarray] = {}
    spin_down_cohp: dict[str, np.ndarray] = {}
    spin_down_icohp: dict[str, np.ndarray] = {}
    dn_offset = 1 + 2 * n_pairs
    for idx, lbl in enumerate(labels):
        cohp[lbl] = arr[:, 1 + 2 * idx]
        icohp[lbl] = arr[:, 2 + 2 * idx]
        if n_spins == 2:
            spin_down_cohp[lbl] = arr[:, dn_offset + 2 * idx]
            spin_down_icohp[lbl] = arr[:, dn_offset + 1 + 2 * idx]

    return electronic.CohpData(
        energies=energies,
        cohp=cohp,
        icohp=icohp,
        fermi_energy=0.0,
        spin_down_cohp=spin_down_cohp,
        spin_down_icohp=spin_down_icohp,
        is_coop=is_coop,
        metadata={"source": str(p), "title": lines[0].strip()},
    )


registry.register_format(
    registry.FormatSpec(
        name="lobster_cohp",
        extensions=(".lobster", ".cohp", ".coop"),
        filenames=("COHPCAR.lobster", "COOPCAR.lobster", "COHPCAR", "COOPCAR"),
        reader=read_cohpcar,
        description="LOBSTER pCOHP/ICOHP and COOP chemical bonding tables",
    )
)
