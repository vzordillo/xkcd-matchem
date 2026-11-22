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
"""Zero-dependency native parser for Phonopy band.yaml phonon dispersions."""

from __future__ import annotations

import pathlib
import re

import numpy as np

from xkcd_matchem.core import electronic
from xkcd_matchem.core import exceptions
from xkcd_matchem.io import registry


def _clean_q_label(raw: str) -> str:
    """Formats phonopy YAML q-point labels cleanly for Matplotlib."""
    txt = raw.strip().strip("'\"")
    return txt.replace("\\\\", "\\")


def _extract_segment_metadata(
    lines: list[str],
) -> tuple[list[int], list[tuple[str, str]]]:
    """Parses optional ``segment_nqpoint`` and ``labels`` header lists."""
    seg_counts: list[int] = []
    seg_pairs: list[tuple[str, str]] = []
    section = ""
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped == "phonon:":
            break
        if stripped.startswith("segment_nqpoint:"):
            section = "seg"
            continue
        if stripped.startswith("labels:"):
            section = "lbl"
            continue
        if not stripped.startswith("-"):
            section = ""
            continue
        if section == "seg":
            val = stripped.lstrip("-").strip()
            if val.isdigit():
                seg_counts.append(int(val))
        elif section == "lbl":
            m_pair = re.findall(r"""['"]([^'"]+)['"]""", stripped)
            if len(m_pair) >= 2:
                seg_pairs.append(
                    (_clean_q_label(m_pair[0]), _clean_q_label(m_pair[1]))
                )
    return seg_counts, seg_pairs


def _resolve_phonon_ticks(
    q_dists: np.ndarray,
    inline_labels: dict[int, str],
    seg_counts: list[int],
    seg_pairs: list[tuple[str, str]],
) -> tuple[list[float], list[str]]:
    """Computes high-symmetry q-tick coordinates and labels."""
    n_q = len(q_dists)
    if seg_counts and sum(seg_counts) == n_q:
        ticks: list[float] = []
        labels: list[str] = []
        cursor = 0
        for s_idx, cnt in enumerate(seg_counts):
            if cnt <= 0:
                continue
            i_start = cursor
            i_end = cursor + cnt - 1
            start_lbl = (
                seg_pairs[s_idx][0]
                if s_idx < len(seg_pairs)
                else inline_labels.get(i_start, "")
            )
            end_lbl = (
                seg_pairs[s_idx][1]
                if s_idx < len(seg_pairs)
                else inline_labels.get(i_end, "")
            )
            if ticks and abs(q_dists[i_start] - ticks[-1]) < 1e-7:
                if start_lbl and labels[-1] and start_lbl != labels[-1]:
                    labels[-1] = f"{labels[-1]}|{start_lbl}"
            else:
                ticks.append(float(q_dists[i_start]))
                labels.append(start_lbl)
            ticks.append(float(q_dists[i_end]))
            labels.append(end_lbl)
            cursor += cnt
        return ticks, labels

    ticks = [float(q_dists[idx]) for idx in sorted(inline_labels)]
    labels = [inline_labels[idx] for idx in sorted(inline_labels)]
    return ticks, labels


def read_phonopy_bands(
    path: str | pathlib.Path,
) -> electronic.PhononBandStructure:
    """Reads phonon dispersion branches from a Phonopy ``band.yaml`` file."""
    p = pathlib.Path(path)
    lines = p.read_text(encoding="utf-8").splitlines()
    seg_counts, seg_pairs = _extract_segment_metadata(lines)

    q_dists: list[float] = []
    branches: list[list[float]] = []
    inline_labels: dict[int, str] = {}
    cur_freqs: list[float] | None = None

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("- q-position:"):
            if cur_freqs is not None:
                branches.append(cur_freqs)
            cur_freqs = []
            continue
        if cur_freqs is None:
            continue
        try:
            if stripped.startswith("distance:"):
                q_dists.append(float(stripped.split(":", 1)[1].strip()))
            elif stripped.startswith("label:"):
                inline_labels[len(q_dists) - 1] = _clean_q_label(
                    stripped.split(":", 1)[1]
                )
            elif stripped.startswith("frequency:"):
                cur_freqs.append(float(stripped.split(":", 1)[1].strip()))
        except ValueError as exc:
            raise exceptions.ParseError(
                f"Malformed numeric value in Phonopy YAML {p}: {stripped!r}"
            ) from exc

    if cur_freqs is not None:
        branches.append(cur_freqs)

    if not q_dists or not branches or len(q_dists) != len(branches):
        raise exceptions.ParseError(
            f"Could not extract phonon q-path and branches from {p}."
        )
    freq_arr = np.asarray(branches, dtype=np.float64)
    dist_arr = np.asarray(q_dists, dtype=np.float64)
    ticks, labels = _resolve_phonon_ticks(
        dist_arr, inline_labels, seg_counts, seg_pairs
    )
    return electronic.PhononBandStructure(
        q_distances=dist_arr,
        frequencies=freq_arr,
        q_labels=labels,
        q_ticks=ticks,
        metadata={"source": str(p)},
    )


registry.register_format(
    registry.FormatSpec(
        name="phonopy_bands",
        extensions=(".band.yaml",),
        filenames=("band.yaml", "phonopy_bands.yaml"),
        reader=read_phonopy_bands,
        description="Phonopy band.yaml phonon dispersion branches (THz)",
    )
)
