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
"""Deterministic generator for approved fixed-baseline visual regression PNGs.

Renders five canonical ``xkcd-matchem`` figures under the commercial-safe
default ``Patrick Hand`` typeface (``xm.set_default_font(None)``), fixed
``dpi=90``, compact fixed ``figsize`` values, and fixed cameras/data:

1. ``molecule_aspirin.png`` (3D aspirin with fixed perspective camera, classic)
2. ``crystal_si_111.png`` (diamond Si with ``(1, 1, 1)`` Miller plane, classic)
3. ``bands_si_qe.png`` (Silicon QE electronic band dispersion, classic)
4. ``xrd_nacl.png`` (NaCl kinematical powder XRD pattern, classic)
5. ``dark_chalk_volcano.png`` (HER catalysis volcano on ``dark_chalk``)

Usage::

    .venv/bin/python scripts/update_visual_baselines.py
    .venv/bin/python scripts/update_visual_baselines.py --names xrd_nacl
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from collections.abc import Iterator
from collections.abc import Sequence
import contextlib
import dataclasses
import pathlib

import matplotlib
from matplotlib import figure as mfigure
from matplotlib import pyplot as plt

import xkcd_matchem as xm


BASELINE_DPI: int = 90
DEFAULT_BASELINES_DIR: pathlib.Path = (
    pathlib.Path(__file__).resolve().parents[1] / "tests" / "baselines"
)


@dataclasses.dataclass(frozen=True)
class BaselineSpec:
    """Canonical deterministic metadata for one golden baseline render."""

    name: str
    preset: str
    figsize: tuple[float, float]
    min_contrast: float = 3.5


@contextlib.contextmanager
def _canonical_default_font() -> Iterator[None]:
    """Temporarily resets session font to default ``Patrick Hand``."""
    previous = xm.get_default_font()
    xm.set_default_font(None)
    try:
        yield
    finally:
        xm.set_default_font(previous)


def _build_molecule_aspirin() -> mfigure.Figure:
    """Renders canonical 3D aspirin molecule baseline."""
    with xm.xkcd_style("classic"):
        fig, ax = plt.subplots(figsize=(4.8, 3.8))
        xm.plot_molecule(
            "aspirin.xyz",
            ax=ax,
            camera=xm.Camera(
                azimuth=-125.0,
                elevation=42.0,
                perspective=0.28,
            ),
            title="Aspirin (3D)",
        )
        fig.tight_layout()
    return fig


def _build_crystal_si_111() -> mfigure.Figure:
    """Renders canonical Si (111) crystal baseline with corner triad."""
    with xm.xkcd_style("classic"):
        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        xm.plot_crystal(
            "si_diamond.cif",
            ax=ax,
            miller_planes=[(1, 1, 1)],
            crystal_axes="corner",
            title="Si (111)",
        )
        fig.tight_layout()
    return fig


def _build_bands_si_qe() -> mfigure.Figure:
    """Renders canonical Quantum ESPRESSO Silicon band structure baseline."""
    loaded = xm.load("si_qe_bands.dat")
    if not isinstance(loaded, xm.BandStructure):
        raise TypeError("Expected BandStructure from 'si_qe_bands.dat'.")
    with xm.xkcd_style("classic"):
        fig, ax = plt.subplots(figsize=(5.2, 3.8))
        xm.plot_bands(loaded, ax=ax, title="Si Bands (QE)")
        fig.tight_layout()
    return fig


def _build_xrd_nacl() -> mfigure.Figure:
    """Renders canonical NaCl powder XRD baseline."""
    with xm.xkcd_style("classic"):
        fig, ax = plt.subplots(figsize=(5.4, 3.6))
        xm.plot_xrd("nacl.cif", ax=ax, title="NaCl Powder XRD")
        fig.tight_layout()
    return fig


def _build_dark_chalk_volcano() -> mfigure.Figure:
    """Renders canonical HER catalysis volcano baseline on ``dark_chalk``."""
    with xm.xkcd_style("dark_chalk"):
        fig, ax = plt.subplots(figsize=(5.4, 3.8))
        xm.plot_volcano(
            [-0.75, -0.35, -0.05, 0.25, 0.65],
            [-2.9, -1.35, -0.35, -1.25, -3.1],
            labels=["Ru", "Rh", "Pt", "Pd", "Au"],
            preset="dark_chalk",
            ax=ax,
            title="HER Volcano",
        )
        fig.tight_layout()
    return fig


BASELINE_SPECS: dict[str, BaselineSpec] = {
    "molecule_aspirin": BaselineSpec(
        name="molecule_aspirin",
        preset="classic",
        figsize=(4.8, 3.8),
    ),
    # Corner a/b/c triad uses Okabe-Ito green #009E73 (3.42:1 on white PAPER),
    # which passes WCAG 2.1 AA graphical/large-text contrast (>= 3.0).
    "crystal_si_111": BaselineSpec(
        name="crystal_si_111",
        preset="classic",
        figsize=(4.8, 4.2),
        min_contrast=3.4,
    ),
    "bands_si_qe": BaselineSpec(
        name="bands_si_qe",
        preset="classic",
        figsize=(5.2, 3.8),
    ),
    "xrd_nacl": BaselineSpec(
        name="xrd_nacl",
        preset="classic",
        figsize=(5.4, 3.6),
    ),
    "dark_chalk_volcano": BaselineSpec(
        name="dark_chalk_volcano",
        preset="dark_chalk",
        figsize=(5.4, 3.8),
    ),
}

BASELINE_FACTORIES: dict[str, Callable[[], mfigure.Figure]] = {
    "molecule_aspirin": _build_molecule_aspirin,
    "crystal_si_111": _build_crystal_si_111,
    "bands_si_qe": _build_bands_si_qe,
    "xrd_nacl": _build_xrd_nacl,
    "dark_chalk_volcano": _build_dark_chalk_volcano,
}

BASELINE_NAMES: tuple[str, ...] = tuple(BASELINE_SPECS.keys())


def render_baseline_figure(name: str) -> mfigure.Figure:
    """Builds a fresh canonical ``Figure`` for golden baseline ``name``.

    Args:
      name: Canonical baseline identifier from ``BASELINE_NAMES``.

    Returns:
      Rendered Matplotlib ``Figure`` under default ``Patrick Hand`` font.

    Raises:
      KeyError: If ``name`` is not a registered baseline key.
    """
    if name not in BASELINE_FACTORIES:
        available = ", ".join(BASELINE_NAMES)
        raise KeyError(f"Unknown baseline {name!r}. Expected: {available}")
    with _canonical_default_font():
        return BASELINE_FACTORIES[name]()


def generate_baselines(
    output_dir: pathlib.Path = DEFAULT_BASELINES_DIR,
    names: Sequence[str] | None = None,
) -> dict[str, pathlib.Path]:
    """Generates and saves approved golden baseline PNGs into ``output_dir``.

    Args:
      output_dir: Destination directory for ``<name>.png`` files.
      names: Optional subset of baseline keys; defaults to all five baselines.

    Returns:
      Mapping from baseline name to saved PNG path.
    """
    selected = tuple(BASELINE_NAMES if names is None else names)
    out_root = pathlib.Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    written: dict[str, pathlib.Path] = {}
    for name in selected:
        fig = render_baseline_figure(name)
        try:
            spec = BASELINE_SPECS[name]
            report = xm.audit_figure_layout(
                fig,
                min_contrast=spec.min_contrast,
            )
            if not report.passed:
                raise RuntimeError(
                    f"Baseline {name!r} failed layout audit: {report!r}"
                )
            out_path = out_root / f"{name}.png"
            xm.save_figure(fig, out_path, dpi=BASELINE_DPI, tight=True)
            written[name] = out_path
        finally:
            plt.close(fig)
    return written


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parses CLI options for regenerating visual regression baselines."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate approved fixed-baseline visual regression PNGs "
            "under tests/baselines/."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=DEFAULT_BASELINES_DIR,
        help="Output directory for baseline PNG files.",
    )
    parser.add_argument(
        "--names",
        nargs="+",
        choices=BASELINE_NAMES,
        default=None,
        help="Optional subset of baseline names to regenerate.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for updating approved golden baseline PNGs."""
    matplotlib.use("Agg")
    args = _parse_args(argv)
    written = generate_baselines(
        output_dir=args.output_dir,
        names=args.names,
    )
    total_bytes = 0
    for name, path in written.items():
        size_b = path.stat().st_size
        total_bytes += size_b
        print(f"Wrote {path} ({size_b:,} B) [{name}]")
    print(
        f"Generated {len(written)} baselines: "
        f"{total_bytes:,} B ({total_bytes / 1024.0:.1f} KB)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
