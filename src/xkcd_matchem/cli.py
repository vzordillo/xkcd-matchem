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
from collections.abc import Callable
from collections.abc import Sequence
import pathlib
from typing import cast

from matplotlib import axes as maxes
from matplotlib import figure as mfigure
from matplotlib import pyplot as plt

from xkcd_matchem import _version
from xkcd_matchem.core import electronic
from xkcd_matchem.core import geometry
from xkcd_matchem.core import structure
from xkcd_matchem.data import elements
from xkcd_matchem.data import samples_registry
from xkcd_matchem.io import loader
from xkcd_matchem.io import registry
from xkcd_matchem.layout import export
from xkcd_matchem.plots import electronic as elec_plots
from xkcd_matchem.plots import ptable as ptable_plots
from xkcd_matchem.plots import reactions as rxn_plots
from xkcd_matchem.plots import spectroscopy as spec_plots
from xkcd_matchem.plots import structure as struct_plots
from xkcd_matchem.render import interactive
from xkcd_matchem.style import palettes


def _parse_color_overrides(
    pairs: Sequence[str] | None,
    base_palette: str = "xkcd",
) -> str | dict[str, str]:
    """Merges repeated ``Element=Color`` CLI overrides atop ``base_palette``."""
    if not pairs:
        return base_palette
    resolved: dict[str, str] = {
        sym: palettes.get_element_color(sym, scheme=base_palette)
        for sym in elements.ELEMENTS_BY_SYMBOL
    }
    for item in pairs:
        sym, sep, col = item.partition("=")
        if not sep or not sym.strip() or not col.strip():
            raise ValueError(
                f"Invalid --colors entry {item!r}; expected Element=#hex."
            )
        resolved[sym.strip()] = col.strip()
    return resolved


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
    return str(args.view)


def _save_and_close(
    ax: maxes.Axes, output: str | pathlib.Path
) -> pathlib.Path:
    """Saves an axes' parent figure and closes it cleanly."""
    fig = cast(mfigure.Figure, ax.figure)
    out_path = export.save_figure(fig, output)
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


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


def _add_analysis_subparsers(
    sub: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Attaches electronic, phonon, NEB, and periodic-table subcommands."""
    bands_p = sub.add_parser(
        "bands", help="Plot an electronic band structure (EIGENVAL, bands.dat)."
    )
    bands_p.add_argument("input", help="Band-structure file or sample name.")
    bands_p.add_argument(
        "-o", "--output", default="bands.png", help="Output image path."
    )
    bands_p.add_argument(
        "--preset", default="classic", help="Visual theme preset name."
    )
    bands_p.add_argument(
        "--ylim",
        type=float,
        nargs=2,
        default=None,
        metavar=("EMIN", "EMAX"),
        help="Optional vertical energy range in eV.",
    )

    dos_p = sub.add_parser(
        "dos", help="Plot electronic density of states (DOSCAR, dos.dat)."
    )
    dos_p.add_argument("input", help="DOS file or sample name.")
    dos_p.add_argument(
        "-o", "--output", default="dos.png", help="Output image path."
    )
    dos_p.add_argument(
        "--preset", default="classic", help="Visual theme preset name."
    )

    neb_p = sub.add_parser(
        "neb", help="Plot an interpolated NEB energy barrier curve."
    )
    neb_p.add_argument(
        "input", help="Trajectory file (XDATCAR, XYZ, lammpstrj)."
    )
    neb_p.add_argument(
        "-o", "--output", default="neb.png", help="Output image path."
    )
    neb_p.add_argument(
        "--preset", default="classic", help="Visual theme preset name."
    )
    neb_p.add_argument(
        "--unit", default="eV", help="Energy unit (eV, kcal/mol, kJ/mol)."
    )

    ptable_p = sub.add_parser(
        "ptable", help="Render an xkcd Periodic Table of Elements."
    )
    ptable_p.add_argument(
        "-o", "--output", default="ptable.png", help="Output image path."
    )
    ptable_p.add_argument(
        "--preset", default="classic", help="Visual theme preset name."
    )
    ptable_p.add_argument(
        "--property",
        default=None,
        help="Optional property heatmap (e.g. electronegativity, atomic_mass).",
    )
    ptable_p.add_argument(
        "--highlight",
        nargs="+",
        default=None,
        help="Optional element symbols or chemical formula to spotlight.",
    )
    ptable_p.add_argument("--title", default=None, help="Optional chart title.")

    phonon_p = sub.add_parser(
        "phonon", help="Plot phonon dispersion bands from Phonopy band.yaml."
    )
    phonon_p.add_argument(
        "input", help="Phonopy band.yaml file or sample name."
    )
    phonon_p.add_argument(
        "-o", "--output", default="phonon.png", help="Output image path."
    )
    phonon_p.add_argument(
        "--preset", default="classic", help="Visual theme preset name."
    )
    phonon_p.add_argument(
        "--unit", default="THz", help="Frequency unit (THz, cm-1, meV)."
    )

    cohp_p = sub.add_parser(
        "cohp", help="Plot LOBSTER -pCOHP chemical bonding curves."
    )
    cohp_p.add_argument("input", help="COHPCAR.lobster file or sample name.")
    cohp_p.add_argument(
        "-o", "--output", default="cohp.png", help="Output image path."
    )
    cohp_p.add_argument(
        "--preset", default="classic", help="Visual theme preset name."
    )
    cohp_p.add_argument(
        "--ylim",
        type=float,
        nargs=2,
        default=None,
        metavar=("EMIN", "EMAX"),
        help="Optional E - E_F window in eV.",
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
    render_p.add_argument(
        "--no-bonds",
        action="store_true",
        help="Show only atoms (no bond sticks) in 3D view.",
    )
    render_p.add_argument(
        "--crystal-axes",
        default=None,
        help="Crystal axes mode (corner, origin, edges, off).",
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
    _add_analysis_subparsers(sub)
    return parser


def _cmd_render(args: argparse.Namespace) -> int:
    """Executes the static ``render`` CLI subcommand."""
    atoms = loader.load(args.input)
    if not isinstance(atoms, structure.AtomsData):
        raise TypeError(f"Cannot render non-structure input {args.input!r}.")
    palette = _parse_color_overrides(args.colors, base_palette=args.palette)
    rep = tuple(int(v) for v in args.supercell)
    if any(v <= 0 for v in rep):
        raise ValueError(
            f"--supercell factors must be positive integers, got {rep}."
        )
    if atoms.is_periodic and rep != (1, 1, 1):
        atoms = atoms.repeat((rep[0], rep[1], rep[2]))
    cam = _resolve_cli_camera(args)
    show_bonds = False if args.no_bonds else None
    if atoms.is_periodic and args.mode == "3d":
        ax = struct_plots.plot_crystal(
            atoms,
            style=args.style,
            palette=palette,
            preset=args.preset,
            camera=cam,
            title=atoms.formula(),
            crystal_axes=args.crystal_axes,
            show_bonds=show_bonds,
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
            show_bonds=show_bonds,
        )
    _save_and_close(ax, args.output)
    return 0


def _cmd_view(args: argparse.Namespace) -> int:
    """Executes the interactive ``view`` CLI subcommand."""
    palette = _parse_color_overrides(args.colors, base_palette=args.palette)
    rep = (
        int(args.supercell[0]),
        int(args.supercell[1]),
        int(args.supercell[2]),
    )
    if any(v <= 0 for v in rep):
        raise ValueError(
            f"--supercell factors must be positive integers, got {rep}."
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
    if args.backend == "matplotlib" and isinstance(res, maxes.Axes):
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
    _save_and_close(ax, args.output)
    return 0


def _cmd_bands(args: argparse.Namespace) -> int:
    """Executes the ``bands`` CLI subcommand."""
    obj = loader.load(args.input)
    if not isinstance(obj, electronic.BandStructure):
        raise TypeError(f"Expected BandStructure from {args.input!r}.")
    ylim = (
        (float(args.ylim[0]), float(args.ylim[1]))
        if args.ylim is not None
        else (-6.0, 6.0)
    )
    ax = elec_plots.plot_bands(obj, ylim=ylim, preset=args.preset)
    _save_and_close(ax, args.output)
    return 0


def _cmd_dos(args: argparse.Namespace) -> int:
    """Executes the ``dos`` CLI subcommand."""
    obj = loader.load(args.input)
    if not isinstance(obj, electronic.DensityOfStates):
        raise TypeError(f"Expected DensityOfStates from {args.input!r}.")
    ax = elec_plots.plot_dos(obj, preset=args.preset)
    _save_and_close(ax, args.output)
    return 0


def _cmd_neb(args: argparse.Namespace) -> int:
    """Executes the ``neb`` CLI subcommand."""
    traj = loader.load_trajectory(args.input)
    ax = rxn_plots.plot_neb(traj, unit=args.unit, preset=args.preset)
    _save_and_close(ax, args.output)
    return 0


def _cmd_ptable(args: argparse.Namespace) -> int:
    """Executes the ``ptable`` CLI subcommand."""
    hl = (
        args.highlight[0]
        if args.highlight is not None and len(args.highlight) == 1
        else args.highlight
    )
    ax = ptable_plots.plot_periodic_table(
        element_property=args.property,
        highlight=hl,
        preset=args.preset,
        title=args.title,
    )
    _save_and_close(ax, args.output)
    return 0


def _cmd_phonon(args: argparse.Namespace) -> int:
    """Executes the ``phonon`` CLI subcommand."""
    ax = elec_plots.plot_phonon_bands(
        args.input, unit=args.unit, preset=args.preset
    )
    _save_and_close(ax, args.output)
    return 0


def _cmd_cohp(args: argparse.Namespace) -> int:
    """Executes the ``cohp`` CLI subcommand."""
    ylim = (
        (float(args.ylim[0]), float(args.ylim[1]))
        if args.ylim is not None
        else (-8.0, 6.0)
    )
    ax = elec_plots.plot_cohp(args.input, ylim=ylim, preset=args.preset)
    _save_and_close(ax, args.output)
    return 0


def _cmd_samples(_: argparse.Namespace) -> int:
    """Lists bundled chemistry/materials sample datasets."""
    for name in samples_registry.list_samples():
        print(f"{name:24s}  {samples_registry.describe_sample(name)}")
    return 0


def _cmd_formats(_: argparse.Namespace) -> int:
    """Lists zero-dependency native file formats."""
    for spec in registry.list_formats():
        exts = ", ".join(spec.extensions + spec.filenames)
        print(f"{spec.name:14s}  [{exts:28s}]  {spec.description}")
    return 0


_DISPATCH: dict[str, Callable[[argparse.Namespace], int]] = {
    "samples": _cmd_samples,
    "formats": _cmd_formats,
    "info": _cmd_info,
    "render": _cmd_render,
    "view": _cmd_view,
    "xrd": _cmd_xrd,
    "bands": _cmd_bands,
    "dos": _cmd_dos,
    "neb": _cmd_neb,
    "ptable": _cmd_ptable,
    "phonon": _cmd_phonon,
    "cohp": _cmd_cohp,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Runs the ``xkcd-matchem`` CLI entrypoint."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    handler = _DISPATCH.get(args.command)
    return handler(args) if handler is not None else 1
