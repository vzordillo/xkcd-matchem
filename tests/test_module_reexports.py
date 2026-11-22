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
"""Tests for modularized facade re-exports, dense peaks, and dark themes."""

from __future__ import annotations

import pathlib

import matplotlib
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from matplotlib import text as mtext
import numpy as np
import pytest

from xkcd_matchem.core import electronic as elec_core
from xkcd_matchem.core import spectroscopy as spec_core
from xkcd_matchem.core import structure as struct_core
from xkcd_matchem.io import loader
from xkcd_matchem.plots import electronic as elec_plots
from xkcd_matchem.plots import materials as mat_plots
from xkcd_matchem.plots import spectroscopy as spec_plots
from xkcd_matchem.render import structure3d
from xkcd_matchem.style import theme


matplotlib.use("Agg")

_SRC_ROOT = pathlib.Path(__file__).resolve().parents[1] / "src" / "xkcd_matchem"


def _relative_luminance(color: object) -> float:
    """Computes WCAG 2.1 relative luminance for a Matplotlib color spec."""
    rgb = mcolors.to_rgb(color)  # type: ignore[arg-type]
    lin = [
        c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
        for c in rgb
    ]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def _contrast_ratio(fg: object, bg: object) -> float:
    """Computes WCAG contrast ratio in ``[1, 21]``."""
    l1 = _relative_luminance(fg)
    l2 = _relative_luminance(bg)
    hi, lo = (l1, l2) if l1 >= l2 else (l2, l1)
    return (hi + 0.05) / (lo + 0.05)


def _assert_axes_dark_contrast(ax: plt.Axes, bg_hex: str) -> None:
    """Verifies axes background matches dark preset and text is readable."""
    assert mcolors.to_hex(ax.get_facecolor()).lower() == bg_hex.lower()
    for artist in ax.texts:
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        bbox_patch = artist.get_bbox_patch()
        eff_bg: object = (
            bbox_patch.get_facecolor() if bbox_patch is not None else bg_hex
        )
        # Skip interior labels drawn directly over colored filled wedges/facets
        # where the local patch fill is checked by contrast_text_color.
        if bbox_patch is None and artist.get_text().startswith(("#", "(")):
            continue
        ratio = _contrast_ratio(artist.get_color(), eff_bg)
        assert ratio >= 3.5, (
            f"Low contrast ({ratio:.2f}) for {artist.get_text()!r} "
            f"fg={artist.get_color()} on bg={eff_bg}"
        )


def test_facade_reexports_backwards_compatibility() -> None:
    """Every public and key internal symbol remains accessible on facades."""
    structure3d_syms = (
        "render_structure_3d",
        "CellConfig3D",
        "SceneGeometry3D",
        "RenderOptions3D",
        "RenderState3D",
        "BondPayload",
        "HalfBondPayload",
        "TrianglePayload",
        "MillerPayload",
        "CellEdgePayload",
        "VectorPayload",
        "HBondPayload",
        "PainterItem",
        "_VALID_STYLES",
        "_CORNER_AXES_LOCS",
        "_MAX_AUTOBOND_ATOMS",
        "_FRONT_CELL_EDGES",
        "_atom_visual_radius",
        "_resolve_axes_override",
        "_parse_cell_and_axes",
        "_polyhedra_skip_spec",
        "_should_skip_poly_bond",
        "_build_interior_bond_item",
        "_add_bond_queue_items",
        "_add_hbond_queue_items",
        "_add_polyhedra_items",
        "_resolve_vectors_3d",
        "_add_vector_items",
        "_format_atom_label",
        "_draw_queue_item",
        "_estimate_pt_per_angstrom",
        "_miller_plane_colors",
        "_add_miller_queue_items",
        "_draw_miller_labels",
        "_prepare_scene_geometry",
        "_build_3d_painter_queue",
        "_add_front_cell_edges",
        "_finalize_3d_viewport",
        "_render_prepared_3d",
    )
    for sym in structure3d_syms:
        assert hasattr(structure3d, sym), f"Missing structure3d.{sym}"

    materials_syms = (
        "spacegroup_to_crystal_system",
        "plot_ternary_phase_diagram",
        "plot_spacegroup_distribution",
        "plot_histogram",
        "plot_wulff_shape",
        "plot_brillouin_zone",
        "_CRYSTAL_SYSTEM_RANGES",
        "_SQRT3_HALF",
        "_barycentric_to_xy",
        "_collinear_lower_hull",
        "_ternary_lower_hull",
        "_draw_ternary_frame",
        "_annotate_ternary_labels",
        "_aggregate_spacegroups",
        "_draw_sunburst_rings",
        "_draw_hist_series",
        "_coerce_wulff_cell",
        "_format_hkl_label",
        "_lambert_shade_rgb",
        "_facet_clearance",
        "_collect_wulff_facet_visuals",
        "_project_wulff_facets",
        "_draw_wulff_labels",
        "_draw_bz_corner_axes",
        "_kpoint_label_offset",
        "_draw_bz_kpath_and_axes",
        "_project_bz_facets",
    )
    for sym in materials_syms:
        assert hasattr(mat_plots, sym), f"Missing materials.{sym}"

    spectroscopy_syms = (
        "plot_spectrum",
        "plot_xrd",
        "expand_nmr_multiplet",
        "plot_nmr",
        "compute_shirley_background",
        "plot_xps",
        "_resolve_stick_scale",
        "_peak_label_offset",
        "_annotate_spectrum_peaks",
        "_NMR_WORD_MAP",
        "_NMR_CHAR_NEIGHBORS",
        "_format_nucleus_symbol",
        "_parse_multiplicity_stages",
        "_parse_resonance_entry",
        "_coerce_nmr_resonances",
        "_cumulative_trapezoid",
        "_spectrum_to_nmr_bundle",
        "_prepare_nmr_data",
        "_draw_straight_seg",
        "_draw_nmr_splitting_tree",
        "_draw_nmr_integrals",
        "_render_nmr_axes",
        "_compute_xps_background",
        "_resolve_comp_dict_center_amp",
        "_evaluate_one_xps_component",
        "_annotate_xps_components",
        "_prepare_xps_arrays",
        "_draw_xps_traces",
    )
    for sym in spectroscopy_syms:
        assert hasattr(spec_plots, sym), f"Missing spectroscopy.{sym}"

    electronic_syms = (
        "plot_bands",
        "plot_dos",
        "plot_bands_dos",
        "plot_mo_diagram",
        "plot_volumetric_slice",
        "plot_isosurface",
        "plot_phonon_bands",
        "plot_cohp",
        "_split_kpath_breaks",
        "_format_k_labels",
        "_resolve_fatband_weights",
        "_draw_band_curves",
        "_dos_window_limits",
        "_draw_dos_curves",
        "_configure_dos_axes",
        "_draw_mo_group",
        "_draw_slice_atoms",
        "_coerce_volumetric_grid",
        "_expand_isosurface_limits",
        "_extract_isosurface_meshes",
        "_draw_iso_art_batches",
        "_draw_cohp_series",
        "_configure_cohp_axes",
        "_resolve_cohp_inputs",
    )
    for sym in electronic_syms:
        assert hasattr(elec_plots, sym), f"Missing electronic.{sym}"


def test_split_module_line_limits() -> None:
    """Every split facade and private sibling module stays under 600 lines."""
    groups = (
        ("render", "structure3d.py", "_structure3d_*.py"),
        ("plots", "materials.py", "_materials_*.py"),
        ("plots", "spectroscopy.py", "_spectroscopy_*.py"),
        ("plots", "electronic.py", "_electronic_*.py"),
    )
    for subdir, facade_name, pattern in groups:
        folder = _SRC_ROOT / subdir
        facade = folder / facade_name
        siblings = sorted(folder.glob(pattern))
        assert facade.is_file()
        assert len(siblings) >= 2, f"Expected split siblings for {facade_name}"
        for mod_path in [facade, *siblings]:
            line_count = len(mod_path.read_text(encoding="utf-8").splitlines())
            assert line_count <= 600, (
                f"{mod_path.name} has {line_count} lines (limit 600)"
            )


def _make_dense_spectrum(
    *,
    positions: np.ndarray,
    kind: str,
    x_bounds: tuple[float, float],
    fwhm: float,
) -> spec_core.Spectrum:
    """Builds a synthetic multi-peak spectrum with 15 closely spaced peaks."""
    x_grid = np.linspace(x_bounds[0], x_bounds[1], 1200)
    intensities = np.linspace(45.0, 100.0, len(positions))
    intensities[1::2] *= 0.82
    y_vals = spec_core.broaden_peaks(
        positions, intensities, x_grid, fwhm=fwhm, profile="lorentzian"
    )
    peaks = [
        spec_core.Peak(
            position=float(pos),
            intensity=float(amp),
            label=f"({idx + 1}10)",
        )
        for idx, (pos, amp) in enumerate(
            zip(positions, intensities, strict=True)
        )
    ]
    return spec_core.Spectrum(
        x=x_grid,
        y=y_vals,
        kind=kind,
        x_label="2theta (deg)" if kind == "xrd" else "Wavenumber (cm-1)",
        y_label="Intensity",
        peaks=peaks,
    )


def test_dense_peak_anti_collision_xrd_and_ir() -> None:
    """15 closely spaced peaks stagger cleanly with leader lines and no pile."""
    xrd_pos = np.linspace(28.0, 42.0, 15)
    dense_xrd = _make_dense_spectrum(
        positions=xrd_pos, kind="xrd", x_bounds=(15.0, 80.0), fwhm=0.35
    )
    ax_xrd = spec_plots.plot_xrd(
        dense_xrd,
        two_theta_range=(15.0, 80.0),
        label_peaks=15,
        preset="classic",
    )
    fig_xrd = ax_xrd.figure
    fig_xrd.canvas.draw()
    anns = [t for t in ax_xrd.texts if isinstance(t, mtext.Annotation)]
    assert len(anns) == 15

    # Vertical tier staggering and thin leader lines must be active.
    y_positions = [float(ann.get_position()[1]) for ann in anns]
    assert len({round(y, 1) for y in y_positions}) >= 3
    expected_lw = theme.stroke_weights("classic").leader
    leaders = [ann.arrow_patch for ann in anns if ann.arrow_patch is not None]
    assert len(leaders) == 15
    assert all(
        float(lp.get_linewidth()) == pytest.approx(expected_lw)
        for lp in leaders
    )

    # Monotonic horizontal order avoids crossing leaders; bboxes do not collide.
    x_positions = [float(ann.get_position()[0]) for ann in anns]
    assert all(
        x_positions[k] <= x_positions[k + 1] + 1e-6
        for k in range(len(x_positions) - 1)
    )
    renderer = fig_xrd.canvas.get_renderer()  # type: ignore[attr-defined]
    bboxes = [
        mtext.Text.get_window_extent(ann, renderer=renderer) for ann in anns
    ]
    overlaps = 0
    for i in range(len(bboxes)):
        for j in range(i + 1, len(bboxes)):
            if bboxes[i].overlaps(bboxes[j]):
                overlaps += 1
    assert overlaps == 0, f"Found {overlaps} overlapping dense XRD labels"
    # Auto-expanded y-limit keeps every staggered annotation inside bounds.
    _, y_hi = ax_xrd.get_ylim()
    assert y_hi > max(y_positions)
    plt.close(fig_xrd)

    # Inverted-x dense IR spectrum (high -> low wavenumber left-to-right).
    ir_pos = np.linspace(1200.0, 1620.0, 15)
    dense_ir = _make_dense_spectrum(
        positions=ir_pos, kind="ir", x_bounds=(500.0, 3500.0), fwhm=14.0
    )
    ax_ir = spec_plots.plot_spectrum(
        dense_ir, annotate_peaks=15, invert_x=True, preset="dark_chalk"
    )
    assert ax_ir.xaxis_inverted()
    ax_ir.figure.canvas.draw()
    ir_anns = [t for t in ax_ir.texts if isinstance(t, mtext.Annotation)]
    assert len(ir_anns) == 15
    ir_renderer = ax_ir.figure.canvas.get_renderer()  # type: ignore[attr-defined]
    ir_boxes = [
        mtext.Text.get_window_extent(ann, renderer=ir_renderer)
        for ann in ir_anns
    ]
    ir_overlaps = sum(
        1
        for i in range(len(ir_boxes))
        for j in range(i + 1, len(ir_boxes))
        if ir_boxes[i].overlaps(ir_boxes[j])
    )
    assert ir_overlaps == 0, f"Found {ir_overlaps} overlapping dense IR labels"
    dark_bg = theme.get_preset("dark_chalk").axes_facecolor
    _assert_axes_dark_contrast(ax_ir, dark_bg)
    plt.close(ax_ir.figure)


@pytest.mark.parametrize(
    "preset_name", ["dark_chalk", "talk_dark", "blueprint"]
)
def test_dark_theme_rendering_all_four_modules(preset_name: str) -> None:
    """All 4 modularized groups render cleanly with high contrast on dark."""
    preset_obj = theme.get_preset(preset_name)
    bg_hex = preset_obj.axes_facecolor

    # 1. render/structure3d
    si = loader.load("si_diamond.cif")
    assert isinstance(si, struct_core.AtomsData)
    spins = np.zeros((len(si), 3), dtype=np.float64)
    spins[:, 2] = [1.0 if k % 2 == 0 else -1.0 for k in range(len(si))]
    with theme.xkcd_style(preset=preset_name):
        ax3d = structure3d.render_structure_3d(
            si,
            miller_planes=[(1, 1, 1)],
            crystal_axes="corner",
            vectors=spins,
            vector_color="spin",
        )
        _assert_axes_dark_contrast(ax3d, bg_hex)
        plt.close(ax3d.figure)

    # 2. plots/materials
    comps = np.array(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.5, 0.5, 0], [0.33, 0.33, 0.34]]
    )
    ax_tern = mat_plots.plot_ternary_phase_diagram(
        comps,
        energies=[0.0, 0.0, 0.0, -0.4, -0.2],
        corners=("Li", "Fe", "O"),
        labels=["Li", "Fe", "O", "LiFe", "LiFeO"],
        preset=preset_name,
        title="Dark Ternary",
    )
    _assert_axes_dark_contrast(ax_tern, bg_hex)
    plt.close(ax_tern.figure)

    ax_sb = mat_plots.plot_spacegroup_distribution(
        [225, 225, 221, 194, 62, 14], preset=preset_name, title="Dark Sunburst"
    )
    _assert_axes_dark_contrast(ax_sb, bg_hex)
    plt.close(ax_sb.figure)

    ax_hist = mat_plots.plot_histogram(
        {"A": [1.0, 1.5, 2.0, 2.5, 3.0], "B": [2.0, 2.5, 3.0, 3.5, 4.0]},
        preset=preset_name,
        title="Dark Histogram",
    )
    _assert_axes_dark_contrast(ax_hist, bg_hex)
    plt.close(ax_hist.figure)

    ax_wf = mat_plots.plot_wulff_shape(
        np.eye(3) * 4.0,
        {(1, 0, 0): 1.0, (1, 1, 1): 0.85},
        preset=preset_name,
        title="Dark Wulff",
    )
    _assert_axes_dark_contrast(ax_wf, bg_hex)
    plt.close(ax_wf.figure)

    ax_bz = mat_plots.plot_brillouin_zone(
        si, preset=preset_name, title="Dark BZ"
    )
    _assert_axes_dark_contrast(ax_bz, bg_hex)
    plt.close(ax_bz.figure)

    # 3. plots/spectroscopy
    ax_xrd = spec_plots.plot_xrd(
        si, two_theta_range=(20.0, 80.0), preset=preset_name, title="Dark XRD"
    )
    _assert_axes_dark_contrast(ax_xrd, bg_hex)
    plt.close(ax_xrd.figure)

    ax_nmr = spec_plots.plot_nmr(
        [(4.12, 2.0, "q", 7.1, "OCH2"), (1.26, 3.0, "t", 7.1, "CH3")],
        preset=preset_name,
        title="Dark NMR",
    )
    _assert_axes_dark_contrast(ax_nmr, bg_hex)
    plt.close(ax_nmr.figure)

    be = np.linspace(282.0, 292.0, 180)
    y_xps = 100.0 + 500.0 * np.exp(-0.5 * ((be - 285.0) / 0.5) ** 2)
    ax_xps = spec_plots.plot_xps(
        be,
        y_xps,
        components=[{"center": 285.0, "amplitude": 480.0, "label": "C-C"}],
        show_residuals=True,
        preset=preset_name,
        title="Dark XPS",
    )
    _assert_axes_dark_contrast(ax_xps, bg_hex)
    plt.close(ax_xps.figure)

    # 4. plots/electronic
    bands = loader.load("si_qe_bands.dat")
    dos = loader.load("si_qe_dos.dat")
    assert isinstance(bands, elec_core.BandStructure)
    assert isinstance(dos, elec_core.DensityOfStates)
    fig_bd, (ax_b, ax_d) = elec_plots.plot_bands_dos(
        bands, dos, preset=preset_name, title="Dark Bands+DOS"
    )
    _assert_axes_dark_contrast(ax_b, bg_hex)
    _assert_axes_dark_contrast(ax_d, bg_hex)
    plt.close(fig_bd)

    orca = loader.load("benzene_orca.out")
    assert isinstance(orca, struct_core.AtomsData)
    ax_mo = elec_plots.plot_mo_diagram(
        orca.metadata["orbitals"], preset=preset_name, title="Dark MO"
    )
    _assert_axes_dark_contrast(ax_mo, bg_hex)
    plt.close(ax_mo.figure)

    cube = loader.load("water_homo.cube")
    assert isinstance(cube, elec_core.VolumetricGrid)
    ax_sl = elec_plots.plot_volumetric_slice(
        cube, preset=preset_name, title="Dark Slice"
    )
    _assert_axes_dark_contrast(ax_sl, bg_hex)
    plt.close(ax_sl.figure)

    ax_iso = elec_plots.plot_isosurface(
        cube, preset=preset_name, title="Dark Iso"
    )
    _assert_axes_dark_contrast(ax_iso, bg_hex)
    plt.close(ax_iso.figure)

    ax_ph = elec_plots.plot_phonon_bands(
        "si_phonon_bands.yaml", preset=preset_name, title="Dark Phonons"
    )
    _assert_axes_dark_contrast(ax_ph, bg_hex)
    plt.close(ax_ph.figure)

    ax_cohp = elec_plots.plot_cohp(
        "si_cohp.lobster", preset=preset_name, title="Dark COHP"
    )
    _assert_axes_dark_contrast(ax_cohp, bg_hex)
    plt.close(ax_cohp.figure)
