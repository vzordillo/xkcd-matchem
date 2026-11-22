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
"""Configurable parser resource ceilings to prevent OOM on hostile inputs."""

from __future__ import annotations

import os
import pathlib

from xkcd_matchem.core import exceptions


DEFAULT_MAX_ATOMS = 200_000
DEFAULT_MAX_GRID_VOXELS = 64_000_000
DEFAULT_MAX_FRAMES = 50_000
DEFAULT_MAX_FILE_BYTES = 200 * 1024 * 1024


def _read_limit(env_var: str, default: int) -> int:
    """Reads a positive integer resource limit from ``env_var``."""
    raw = os.environ.get(env_var, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise exceptions.ParseError(
            f"Invalid {env_var}={raw!r}; expected a positive integer."
        ) from exc
    if value <= 0:
        raise exceptions.ParseError(
            f"Invalid {env_var}={raw!r}; limit must be a positive integer."
        )
    return value


def _source_suffix(source: str | pathlib.Path) -> str:
    """Formats an optional source path suffix for ParseError diagnostics."""
    text = str(source).strip()
    return f" in {text}" if text else ""


def max_atoms() -> int:
    """Returns the maximum permitted atoms per structure (``XM_MAX_ATOMS``)."""
    return _read_limit("XM_MAX_ATOMS", DEFAULT_MAX_ATOMS)


def max_grid_voxels() -> int:
    """Returns the maximum permitted grid voxels (``XM_MAX_GRID_VOXELS``)."""
    return _read_limit("XM_MAX_GRID_VOXELS", DEFAULT_MAX_GRID_VOXELS)


def max_frames() -> int:
    """Returns the maximum permitted trajectory frames (``XM_MAX_FRAMES``)."""
    return _read_limit("XM_MAX_FRAMES", DEFAULT_MAX_FRAMES)


def max_file_bytes() -> int:
    """Returns the maximum permitted input file size (``XM_MAX_FILE_BYTES``)."""
    return _read_limit("XM_MAX_FILE_BYTES", DEFAULT_MAX_FILE_BYTES)


def check_atom_count(
    n_atoms: int,
    source: str | pathlib.Path = "",
) -> None:
    """Raises ``ParseError`` if ``n_atoms`` is negative or exceeds ceiling."""
    suffix = _source_suffix(source)
    if n_atoms < 0:
        raise exceptions.ParseError(
            f"Negative atom count {n_atoms}{suffix}."
        )
    limit = max_atoms()
    if n_atoms > limit:
        raise exceptions.ParseError(
            f"Atom count {n_atoms} exceeds parser limit {limit}{suffix} "
            "(override with XM_MAX_ATOMS)."
        )


def check_grid_voxels(
    n_voxels: int,
    source: str | pathlib.Path = "",
) -> None:
    """Raises ``ParseError`` if ``n_voxels`` is negative or exceeds ceiling."""
    suffix = _source_suffix(source)
    if n_voxels < 0:
        raise exceptions.ParseError(
            f"Negative grid voxel count {n_voxels}{suffix}."
        )
    limit = max_grid_voxels()
    if n_voxels > limit:
        raise exceptions.ParseError(
            f"Grid voxel count {n_voxels} exceeds parser limit {limit}{suffix} "
            "(override with XM_MAX_GRID_VOXELS)."
        )


def check_frame_count(
    n_frames: int,
    source: str | pathlib.Path = "",
) -> None:
    """Raises ``ParseError`` if ``n_frames`` is negative or exceeds ceiling."""
    suffix = _source_suffix(source)
    if n_frames < 0:
        raise exceptions.ParseError(
            f"Negative frame count {n_frames}{suffix}."
        )
    limit = max_frames()
    if n_frames > limit:
        raise exceptions.ParseError(
            f"Frame count {n_frames} exceeds parser limit {limit}{suffix} "
            "(override with XM_MAX_FRAMES)."
        )


def check_file_size(path: str | pathlib.Path) -> None:
    """Raises ``ParseError`` if file at ``path`` exceeds ``max_file_bytes``."""
    p = pathlib.Path(path)
    if not p.is_file():
        return
    size = p.stat().st_size
    limit = max_file_bytes()
    if size > limit:
        raise exceptions.ParseError(
            f"File size {size} bytes exceeds parser limit {limit} bytes "
            f"in {p} (override with XM_MAX_FILE_BYTES)."
        )
