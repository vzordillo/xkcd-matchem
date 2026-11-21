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
"""Command-line interface for xkcd-matchem."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from matplotlib import pyplot as plt

from xkcd_matchem import _version
from xkcd_matchem.core import structure
from xkcd_matchem.data import samples_registry
from xkcd_matchem.io import loader
from xkcd_matchem.io import registry
from xkcd_matchem.layout import export
from xkcd_matchem.plots import spectroscopy as spec_plots
from xkcd_matchem.plots import structure as struct_plots


def _build_parser() -> argparse.ArgumentParser:
    """Constructs the top-level CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="xkcd-matchem",
        description="Hand-drawn xkcd plots for molecules and materials.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"xkcd-matchem {_version.__version__}",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("samples", help="List bundled sample files.")
    sub.add_parser("formats", help="List registered native file formats.")

    info_p = sub.add_parser("info", help="Print parsed file summary.")
    info_p.add_argument("input", help="File path or bundled sample filename.")
    info_p.add_argument(
        "--format", dest="fmt", default=None, help="Format key."
    )

    render_p = sub.add_parser("render", help="Render structure to image/PDF.")
    render_p.add_argument("input", help="Input structure file or sample name.")
    render_p.add_argument("-o", "--output", default="structure.png")
    render_p.add_argument("--style", default="ball_and_stick")
    render_p.add_argument("--preset", default="classic")
    render_p.add_argument("--palette", default="xkcd")
    render_p.add_argument("--mode", default="3d", choices=("3d", "2d"))

    xrd_p = sub.add_parser("xrd", help="Compute and plot powder XRD.")
    xrd_p.add_argument("input", help="Input periodic crystal file or sample.")
    xrd_p.add_argument("-o", "--output", default="xrd.png")
    xrd_p.add_argument("--preset", default="classic")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Runs the ``xkcd-matchem`` CLI entrypoint."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "samples":
        for name in samples_registry.list_samples():
            print(f"{name:24s}  {samples_registry.describe_sample(name)}")
        return 0

    if args.command == "formats":
        for spec in registry.list_formats():
            exts = ", ".join(spec.extensions + spec.filenames)
            print(f"{spec.name:14s}  [{exts:28s}]  {spec.description}")
        return 0

    if args.command == "info":
        obj = loader.load(args.input, fmt=args.fmt)
        if isinstance(obj, structure.AtomsData):
            print(f"Formula: {obj.formula()} ({len(obj)} atoms)")
            print(f"Periodic: {obj.is_periodic} (V={obj.volume():.2f} A^3)")
        else:
            print(repr(obj))
        return 0

    if args.command == "render":
        atoms = loader.load(args.input)
        if (
            isinstance(atoms, structure.AtomsData)
            and atoms.is_periodic
            and args.mode == "3d"
        ):
            ax = struct_plots.plot_crystal(
                atoms,
                style=args.style,
                palette=args.palette,
                preset=args.preset,
                title=atoms.formula(),
            )
        elif isinstance(atoms, structure.AtomsData):
            ax = struct_plots.plot_molecule(
                atoms,
                mode=args.mode,
                style=args.style,
                palette=args.palette,
                preset=args.preset,
                title=atoms.formula(),
            )
        else:
            raise TypeError(
                f"Cannot render non-structure input {args.input!r}."
            )
        out_path = export.save_figure(ax.figure, args.output)
        plt.close(ax.figure)
        print(f"Saved {out_path}")
        return 0

    if args.command == "xrd":
        ax = spec_plots.plot_xrd(args.input, preset=args.preset)
        out_path = export.save_figure(ax.figure, args.output)
        plt.close(ax.figure)
        print(f"Saved {out_path}")
        return 0

    return 1
