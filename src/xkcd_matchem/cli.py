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
from xkcd_matchem.core import geometry
from xkcd_matchem.core import structure
from xkcd_matchem.data import samples_registry
from xkcd_matchem.io import loader
from xkcd_matchem.io import registry
from xkcd_matchem.layout import export
from xkcd_matchem.plots import spectroscopy as spec_plots
from xkcd_matchem.plots import structure as struct_plots
from xkcd_matchem.render import interactive


def _parse_color_overrides(pairs: Sequence[str] | None) -> str | dict[str, str]:
    """Parses repeated ``Element=Color`` CLI pairs into a palette dict."""
    if not pairs:
        return "xkcd"
    overrides: dict[str, str] = {}
    for item in pairs:
        sym, sep, col = item.partition("=")
        if not sep or not sym.strip() or not col.strip():
            raise ValueError(
                f"Invalid --colors entry {item!r}; expected Element=#hex."
            )
        overrides[sym.strip()] = col.strip()
    return overrides


def _resolve_cli_camera(args: argparse.Namespace) -> geometry.Camera | str:
    """Resolves camera preset or explicit Euler angles from CLI arguments."""
    if args.azimuth is not None or args.elevation is not None:
        return geometry.Camera(
            azimuth=-118.0 if args.azimuth is None else args.azimuth,
            elevation=24.0 if args.elevation is None else args.elevation,
            perspective=args.perspective,
        )
    if args.perspective > 0.0 and args.view == "isometric":
        return geometry.Camera(perspective=args.perspective)
    return args.view


def _add_common_view_args(subparser: argparse.ArgumentParser) -> None:
    """Attaches shared camera, supercell, and palette customization flags."""
    subparser.add_argument(
        "--style",
        default="ball_and_stick",
        help="3D style (ball_and_stick, spacefill, stick, polyhedra).",
    )
    subparser.add_argument(
        "--preset",
        default="classic",
        help="Visual theme preset (classic, notebook, poster, minimal, ...).",
    )
    subparser.add_argument(
        "--palette",
        default="xkcd",
        help="Element color scheme (xkcd, jmol, pastel, colorblind).",
    )
    subparser.add_argument(
        "--colors",
        nargs="+",
        action="extend",
        default=None,
        help="Element color overrides as Element=#HEX (e.g. Ti=#E07A5F).",
    )
    subparser.add_argument(
        "--view",
        default="isometric",
        help="Camera preset (isometric, perspective, front, side, top, a/b/c).",
    )
    subparser.add_argument(
        "--azimuth", type=float, default=None, help="Camera azimuth in degrees."
    )
    subparser.add_argument(
        "--elevation",
        type=float,
        default=None,
        help="Camera elevation in degrees.",
    )
    subparser.add_argument(
        "--perspective",
        type=float,
        default=0.0,
        help="Perspective strength (0.0 for orthographic).",
    )
    subparser.add_argument(
        "--supercell",
        type=int,
        nargs=3,
        default=[1, 1, 1],
        metavar=("NA", "NB", "NC"),
        help="Periodic supercell repetitions along a, b, c.",
    )


def _build_parser() -> argparse.ArgumentParser:
    """Constructs the top-level CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="xkcd-matchem",
        description=(
            "xkcd-style visualization suite for chemistry and materials."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"xkcd-matchem {_version.__version__}",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("samples", help="List bundled sample structure/data files.")
    sub.add_parser("formats", help="List zero-dependency native file formats.")

    info_p = sub.add_parser(
        "info", help="Inspect formula, atom count, and cell summary of a file."
    )
    info_p.add_argument("input", help="File path or bundled sample filename.")
    info_p.add_argument(
        "--format", dest="fmt", default=None, help="Explicit format key."
    )

    render_p = sub.add_parser(
        "render", help="Render a structure to PNG, SVG, or PDF."
    )
    render_p.add_argument("input", help="Input structure file or sample name.")
    render_p.add_argument(
        "-o", "--output", default="structure.png", help="Output image path."
    )
    render_p.add_argument(
        "--mode",
        default="3d",
        choices=("3d", "2d"),
        help="Render in 3D depth-sorted or 2D skeletal mode.",
    )
    _add_common_view_args(render_p)

    view_p = sub.add_parser(
        "view",
        help="Open a structure in the offline Interactive 3D Studio.",
    )
    view_p.add_argument("input", help="Input structure file or sample name.")
    view_p.add_argument(
        "-o", "--output", default=None, help="Optional HTML output path."
    )
    view_p.add_argument(
        "--backend",
        default="html",
        choices=("html", "browser", "matplotlib"),
        help="Viewer target: standalone html file, system browser, or mpl.",
    )
    _add_common_view_args(view_p)

    xrd_p = sub.add_parser(
        "xrd", help="Simulate and plot a Cu Ka1 powder XRD pattern."
    )
    xrd_p.add_argument("input", help="Input periodic crystal file or sample.")
    xrd_p.add_argument(
        "-o", "--output", default="xrd.png", help="Output image path."
    )
    xrd_p.add_argument(
        "--preset", default="classic", help="Visual theme preset name."
    )
    return parser


def _cmd_render(args: argparse.Namespace) -> int:
    """Executes the static ``render`` CLI subcommand."""
    atoms = loader.load(args.input)
    if not isinstance(atoms, structure.AtomsData):
        raise TypeError(f"Cannot render non-structure input {args.input!r}.")
    palette: str | dict[str, str] = (
        _parse_color_overrides(args.colors) if args.colors else args.palette
    )
    rep = tuple(int(v) for v in args.supercell)
    if atoms.is_periodic and rep != (1, 1, 1):
        atoms = atoms.repeat((rep[0], rep[1], rep[2]))
    cam = _resolve_cli_camera(args)
    if atoms.is_periodic and args.mode == "3d":
        ax = struct_plots.plot_crystal(
            atoms,
            style=args.style,
            palette=palette,
            preset=args.preset,
            camera=cam,
            title=atoms.formula(),
        )
    else:
        ax = struct_plots.plot_molecule(
            atoms,
            mode=args.mode,
            style=args.style,
            palette=palette,
            preset=args.preset,
            camera=cam,
            title=atoms.formula(),
        )
    out_path = export.save_figure(ax.figure, args.output)
    plt.close(ax.figure)
    print(f"Saved {out_path}")
    return 0


def _cmd_view(args: argparse.Namespace) -> int:
    """Executes the interactive ``view`` CLI subcommand."""
    palette: str | dict[str, str] = (
        _parse_color_overrides(args.colors) if args.colors else args.palette
    )
    rep = (
        int(args.supercell[0]),
        int(args.supercell[1]),
        int(args.supercell[2]),
    )
    cam = _resolve_cli_camera(args)
    out_target = args.output
    if out_target is None and args.backend == "html":
        out_target = "structure_view.html"
    res = interactive.view_structure(
        args.input,
        backend=args.backend,
        output=out_target,
        style=args.style,
        palette=palette,
        preset=args.preset,
        camera=cam,
        supercell=rep,
    )
    if args.backend == "matplotlib" and isinstance(res, plt.Axes):
        plt.show()
    elif out_target is not None:
        print(f"Saved interactive viewer to {out_target}")
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    """Executes the ``info`` CLI subcommand."""
    obj = loader.load(args.input, fmt=args.fmt)
    if isinstance(obj, structure.AtomsData):
        print(f"Formula: {obj.formula()} ({len(obj)} atoms)")
        print(f"Periodic: {obj.is_periodic} (V={obj.volume():.2f} A^3)")
    else:
        print(repr(obj))
    return 0


def _cmd_xrd(args: argparse.Namespace) -> int:
    """Executes the ``xrd`` CLI subcommand."""
    ax = spec_plots.plot_xrd(args.input, preset=args.preset)
    out_path = export.save_figure(ax.figure, args.output)
    plt.close(ax.figure)
    print(f"Saved {out_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Runs the ``xkcd-matchem`` CLI entrypoint."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    exit_code = 1

    if args.command == "samples":
        for name in samples_registry.list_samples():
            print(f"{name:24s}  {samples_registry.describe_sample(name)}")
        exit_code = 0
    elif args.command == "formats":
        for spec in registry.list_formats():
            exts = ", ".join(spec.extensions + spec.filenames)
            print(f"{spec.name:14s}  [{exts:28s}]  {spec.description}")
        exit_code = 0
    elif args.command == "info":
        exit_code = _cmd_info(args)
    elif args.command == "render":
        exit_code = _cmd_render(args)
    elif args.command == "view":
        exit_code = _cmd_view(args)
    elif args.command == "xrd":
        exit_code = _cmd_xrd(args)

    return exit_code
