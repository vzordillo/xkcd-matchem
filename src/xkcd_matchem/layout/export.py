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
"""Figure export helpers supporting PNG, SVG, and PDF."""

from __future__ import annotations

import pathlib

from matplotlib import figure


def save_figure(
    fig: figure.Figure,
    path: str | pathlib.Path,
    dpi: int = 200,
    transparent: bool = False,
) -> pathlib.Path:
    """Saves a Matplotlib ``Figure`` creating missing parent directories.

    Args:
      fig: Target figure instance.
      path: Output path whose suffix determines format (``.png``, ``.svg``,
        ``.pdf``).
      dpi: Rasterization resolution for bitmap outputs.
      transparent: Whether to save with transparent canvas background.

    Returns:
      Resolved output ``pathlib.Path``.
    """
    out = pathlib.Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out,
        dpi=dpi,
        transparent=transparent,
        facecolor=fig.get_facecolor(),
        bbox_inches="tight",
        pad_inches=0.08,
    )
    return out
