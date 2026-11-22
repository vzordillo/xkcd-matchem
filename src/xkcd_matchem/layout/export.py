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

from collections.abc import Mapping
import pathlib
from typing import Any

import matplotlib
from matplotlib import figure


def _unsketch_canvas_patches(fig: figure.Figure) -> None:
    """Disables path sketch on figure, subfigure, and axes background frames."""
    fig.patch.set_sketch_params(0.0, 0.0, 0.0)
    for ax in fig.get_axes():
        ax.patch.set_sketch_params(0.0, 0.0, 0.0)
    for subfig in getattr(fig, "subfigs", ()):
        _unsketch_canvas_patches(subfig)


def save_figure(
    fig: figure.Figure,
    path: str | pathlib.Path,
    dpi: int = 200,
    transparent: bool = False,
    *,
    tight: bool = True,
    svg_fonttype: str = "path",
    pdf_fonttype: int = 42,
    metadata: Mapping[str, Any] | None = None,
) -> pathlib.Path:
    """Saves a Matplotlib ``Figure`` creating missing parent directories.

    Embeds TrueType outlines for PDF/PS exports (``pdf_fonttype=42``) and
    defaults SVG text to crisp vector paths (``svg_fonttype="path"``), while
    un-sketching all figure and axes background patches so dark themes never
    leak white border artifacts.

    Args:
      fig: Target figure instance.
      path: Output path whose suffix determines format (``.png``, ``.svg``,
        ``.pdf``).
      dpi: Rasterization resolution for bitmap outputs.
      transparent: Whether to save with transparent canvas background.
      tight: Use ``bbox_inches="tight"`` cropping when ``True``, or preserve
        exact figure canvas dimensions when ``False``.
      svg_fonttype: Matplotlib ``svg.fonttype`` setting (``"path"`` for exact
        vector outlines or ``"none"`` for editable text in vector editors).
      pdf_fonttype: Matplotlib ``pdf.fonttype`` / ``ps.fonttype`` setting
        (``42`` embeds TrueType outlines).
      metadata: Optional backend-specific document metadata dictionary.

    Returns:
      Resolved output ``pathlib.Path``.
    """
    out = pathlib.Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    _unsketch_canvas_patches(fig)
    if transparent:
        face_col: Any = "none"
        edge_col: Any = "none"
    else:
        face_col = fig.get_facecolor()
        edge_col = face_col
        fig.patch.set_edgecolor(edge_col)
    export_rc = {
        "pdf.fonttype": pdf_fonttype,
        "ps.fonttype": pdf_fonttype,
        "svg.fonttype": svg_fonttype,
    }
    meta_dict = dict(metadata) if metadata is not None else None
    with matplotlib.rc_context(export_rc):  # type: ignore[arg-type]
        fig.savefig(
            out,
            dpi=dpi,
            transparent=transparent,
            facecolor=face_col,
            edgecolor=edge_col,
            bbox_inches="tight" if tight else None,
            pad_inches=0.08 if tight else 0.0,
            metadata=meta_dict,
        )
    return out
