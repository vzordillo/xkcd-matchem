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
"""Reproducible runtime and memory scalability benchmarks for xkcd-matchem.

Measures median wall-clock latency (ms) and peak Python heap allocation
(``tracemalloc``, MB) across five core computational workloads:

1. LLL-reduced periodic neighbor search (100, 1,000, and 5,000 atoms).
2. Periodic radial distribution function ``compute_rdf`` (20 frames, 64 / 256).
3. Kinematical powder XRD ``compute_xrd`` (``si_diamond.cif`` & 2x2x2 cell).
4. Kuhn Marching Tetrahedra ``extract_isosurface`` (32^3 and 64^3 grids).
5. Standalone interactive HTML Studio export (``water_homo.cube`` & TiO2 slab).

Usage::

    python scripts/benchmark_scalability.py
    python scripts/benchmark_scalability.py --quick
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import dataclasses
import os
import pathlib
import platform
import statistics
import tempfile
import time
import tracemalloc

import numpy as np
import scipy

import xkcd_matchem as xm


@dataclasses.dataclass(frozen=True)
class BenchmarkRow:
    """Single scalability benchmark measurement record."""

    domain: str
    workload: str
    wall_ms: float
    peak_mb: float
    detail: str


def _measure_workload(
    fn: Callable[[], str],
    repeats: int,
) -> tuple[float, float, str]:
    """Measures un-instrumented median wall time (ms) and tracemalloc MB."""
    detail = fn()
    times_ms: list[float] = []
    for _ in range(max(1, repeats)):
        t_start = time.perf_counter()
        detail = fn()
        times_ms.append((time.perf_counter() - t_start) * 1000.0)
    tracemalloc.start()
    detail = fn()
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return statistics.median(times_ms), peak_bytes / (1024.0 * 1024.0), detail


def _build_perovskite_supercell(rep: tuple[int, int, int]) -> xm.AtomsData:
    """Builds a 5-atom cubic SrTiO3 cell replicated to exact atom counts."""
    unit = xm.AtomsData.from_fractional(
        symbols=["Sr", "Ti", "O", "O", "O"],
        frac_positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.0],
                [0.5, 0.0, 0.5],
                [0.0, 0.5, 0.5],
            ],
            dtype=np.float64,
        ),
        cell=np.diag([3.905, 3.905, 3.905]),
    )
    return unit.repeat(rep)


def _build_thermal_trajectory(
    base_atoms: xm.AtomsData,
    n_frames: int,
    seed: int = 42,
) -> xm.Trajectory:
    """Generates a deterministic thermalized periodic MD trajectory."""
    rng = np.random.default_rng(seed)
    frames: list[xm.AtomsData] = []
    for _ in range(n_frames):
        snap = base_atoms.copy()
        snap.positions = snap.positions + rng.normal(
            0.0, 0.06, size=snap.positions.shape
        )
        frames.append(snap)
    return xm.Trajectory(frames)


def _build_orbital_grid(n_dim: int) -> xm.VolumetricGrid:
    """Builds a synthetic 3D two-lobe p_z orbital VolumetricGrid of n_dim^3."""
    axis = np.linspace(-4.0, 4.0, n_dim)
    step = float(axis[1] - axis[0])
    gx, gy, gz = np.meshgrid(axis, axis, axis, indexing="ij")
    r_sq = gx * gx + gy * gy + gz * gz
    data = gz * np.exp(-0.5 * r_sq / (1.15**2))
    return xm.VolumetricGrid(
        origin=np.array([-4.0, -4.0, -4.0]),
        voxel_vectors=np.diag([step, step, step]),
        data=data,
    )


def _bench_neighbors(repeats: int) -> list[BenchmarkRow]:
    """Benchmarks periodic neighbor search at 100, 1,000, and 5,000 atoms."""
    rows: list[BenchmarkRow] = []
    configs = (
        ("100 atoms (SrTiO3 2x2x5)", (2, 2, 5)),
        ("1,000 atoms (SrTiO3 5x5x8)", (5, 5, 8)),
        ("5,000 atoms (SrTiO3 10x10x10)", (10, 10, 10)),
    )
    for label, rep in configs:
        atoms = _build_perovskite_supercell(rep)

        def _run(at: xm.AtomsData = atoms) -> str:
            hits = xm.find_periodic_neighbors(at, max_cutoff=3.2)
            return f"{len(at)} atoms, {len(hits)} pairs"

        ms, mb, info = _measure_workload(_run, repeats)
        rows.append(BenchmarkRow("Periodic Neighbors", label, ms, mb, info))
    return rows


def _bench_rdf(repeats: int) -> list[BenchmarkRow]:
    """Benchmarks 20-frame periodic RDF at 64 and 256 atoms."""
    rows: list[BenchmarkRow] = []
    si_64 = xm.load("si_diamond.cif").repeat((2, 2, 2))
    fcc_4 = xm.AtomsData.from_fractional(
        ["Cu", "Cu", "Cu", "Cu"],
        np.array(
            [[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]
        ),
        np.diag([3.615, 3.615, 3.615]),
    )
    cu_256 = fcc_4.repeat((4, 4, 4))
    cases = (
        ("20 frames x 64 atoms (Si)", si_64, ("Si", "Si")),
        ("20 frames x 256 atoms (Cu)", cu_256, ("Cu", "Cu")),
    )
    for label, cryst, pair in cases:
        traj = _build_thermal_trajectory(cryst, n_frames=20)

        def _run(
            tr: xm.Trajectory = traj, pr: tuple[str, str] = pair
        ) -> str:
            r_c, g_r = tr.compute_rdf(pr, r_max=5.0, n_bins=100)
            return f"n_bins={len(r_c)}, g_max={float(np.max(g_r)):.2f}"

        ms, mb, info = _measure_workload(_run, repeats)
        rows.append(BenchmarkRow("Periodic RDF", label, ms, mb, info))
    return rows


def _bench_xrd(repeats: int) -> list[BenchmarkRow]:
    """Benchmarks kinematical powder XRD on unit and 2x2x2 Si cells."""
    rows: list[BenchmarkRow] = []
    si_unit = xm.load("si_diamond.cif")
    si_222 = si_unit.repeat((2, 2, 2))
    for label, atoms in (
        ("si_diamond.cif (8 atoms)", si_unit),
        ("si_diamond 2x2x2 (64 atoms)", si_222),
    ):

        def _run(at: xm.AtomsData = atoms) -> str:
            spec = xm.compute_xrd(at, two_theta_range=(10.0, 90.0))
            return f"{len(spec.peaks)} Bragg peaks"

        ms, mb, info = _measure_workload(_run, repeats)
        rows.append(BenchmarkRow("Powder XRD", label, ms, mb, info))
    return rows


def _bench_isosurface(repeats: int) -> list[BenchmarkRow]:
    """Benchmarks Marching Tetrahedra extraction on 32^3 and 64^3 grids."""
    rows: list[BenchmarkRow] = []
    for n_dim in (32, 64):
        grid = _build_orbital_grid(n_dim)
        level = 0.25 * float(np.max(np.abs(grid.data)))

        def _run(gr: xm.VolumetricGrid = grid, lv: float = level) -> str:
            mesh = xm.extract_isosurface(gr, lv)
            return f"{len(mesh.vertices)} verts, {len(mesh.faces)} tris"

        ms, mb, info = _measure_workload(_run, repeats)
        rows.append(
            BenchmarkRow(
                "Marching Tetrahedra", f"{n_dim}^3 grid", ms, mb, info
            )
        )
    return rows


def _bench_interactive_html(repeats: int) -> list[BenchmarkRow]:
    """Benchmarks standalone 3D HTML Studio export and payload sizes."""
    rows: list[BenchmarkRow] = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_root = pathlib.Path(tmp_dir)
        cases: tuple[
            tuple[str, str, str, tuple[str, str | None] | None], ...
        ] = (
            ("water_homo.cube", "water_homo.cube", "ball_and_stick", None),
            (
                "tio2_slab.poscar",
                "tio2_slab.poscar",
                "polyhedra",
                ("Ti", "O"),
            ),
        )
        for label, sample, style, poly in cases:
            out_file = tmp_root / f"{sample}.html"

            def _run(
                smp: str = sample,
                dst: pathlib.Path = out_file,
                st: str = style,
                pl: tuple[str, str | None] | None = poly,
            ) -> str:
                xm.export_interactive_html(smp, dst, style=st, polyhedra=pl)
                size_kb = dst.stat().st_size / 1024.0
                return f"HTML={size_kb:.1f} KB"

            ms, mb, info = _measure_workload(_run, repeats)
            rows.append(BenchmarkRow("Interactive HTML", label, ms, mb, info))
    return rows


def _format_hardware_header(repeats: int, quick: bool) -> str:
    """Formats hardware and dependency provenance summary."""
    cpu_cnt = os.cpu_count() or 1
    proc = platform.processor() or platform.machine()
    mode_str = "quick (1 timed pass)" if quick else f"standard ({repeats}x)"
    lines = [
        "### Scalability Benchmark Environment",
        f"- **Platform**: `{platform.platform()}` (`{proc}`, {cpu_cnt} CPUs)",
        (
            f"- **Runtime**: Python `{platform.python_version()}`, "
            f"NumPy `{np.__version__}`, SciPy `{scipy.__version__}`, "
            f"xkcd-matchem `{xm.__version__}`"
        ),
        f"- **Protocol**: {mode_str}, 1 warmup pass, `tracemalloc` peak heap",
        "",
    ]
    return "\n".join(lines)


def _format_markdown_table(rows: list[BenchmarkRow]) -> str:
    """Formats benchmark records as an aligned GitHub Markdown table."""
    lines = [
        "| Domain | Workload | Wall Time (ms) | Peak Memory (MB) | Output |",
        "| :--- | :--- | ---: | ---: | :--- |",
    ]
    for row in rows:
        lines.append(
            f"| {row.domain} | `{row.workload}` | {row.wall_ms:.2f} | "
            f"{row.peak_mb:.2f} | {row.detail} |"
        )
    return "\n".join(lines)


def run_benchmarks(quick: bool = False) -> list[BenchmarkRow]:
    """Executes all five scalability benchmark suites."""
    repeats = 1 if quick else 3
    rows: list[BenchmarkRow] = []
    rows.extend(_bench_neighbors(repeats))
    rows.extend(_bench_rdf(repeats))
    rows.extend(_bench_xrd(repeats))
    rows.extend(_bench_isosurface(repeats))
    rows.extend(_bench_interactive_html(repeats))
    return rows


def main() -> None:
    """CLI entry point for standalone scalability benchmarking."""
    parser = argparse.ArgumentParser(
        description="Run xkcd-matchem runtime and memory scalability suite."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run single-pass quick mode for fast CI/developer checks.",
    )
    args = parser.parse_args()
    repeats = 1 if args.quick else 3
    rows = run_benchmarks(quick=bool(args.quick))
    print(_format_hardware_header(repeats=repeats, quick=bool(args.quick)))
    print(_format_markdown_table(rows))


if __name__ == "__main__":
    main()
