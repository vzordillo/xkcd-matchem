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
"""Hardening tests for Interactive 3D Studio responsiveness, LOD, & snippets."""

from __future__ import annotations

import ast
import inspect
import json
import pathlib
import time
from typing import Any

import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pytest

import xkcd_matchem as xm
from xkcd_matchem.render import interactive


def _extract_embedded_payload(html_text: str) -> dict[str, Any]:
    """Extracts and parses the embedded ``PAYLOAD`` JSON from Studio HTML."""
    marker = "const PAYLOAD = "
    start = html_text.index(marker) + len(marker)
    end = html_text.index(";\n  const ELEM_TABLE", start)
    raw = html_text[start:end]
    return json.loads(raw)  # type: ignore[no-any-return]


def _assert_valid_xm_snippet(code: str) -> list[str]:
    """Asserts PEP-8 width, AST validity, and real ``xm.*`` call signatures."""
    for line in code.splitlines():
        assert len(line) <= 80, f"Snippet line exceeds 80 cols: {line!r}"
    tree = ast.parse(code)
    called_funcs: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "xm"
        ):
            attr = node.func.attr
            assert hasattr(xm, attr), f"xm has no public attribute {attr!r}"
            target = getattr(xm, attr)
            sig = inspect.signature(target)
            pos_vals = [object() for _ in node.args]
            kw_vals: dict[str, object] = {}
            for kw in node.keywords:
                assert kw.arg is not None, "Unexpected **kwargs splat"
                kw_vals[kw.arg] = object()
            sig.bind(*pos_vals, **kw_vals)
            called_funcs.append(attr)
    assert called_funcs, "Snippet contained no xm.* API calls"
    return called_funcs


def test_responsive_sidebar_drawer_and_resize_observer_hooks() -> None:
    """Verifies narrow-iframe drawer markup, <=780px breakpoint, & observers."""
    html_doc = interactive.build_interactive_html("aspirin.xyz")

    assert 'id="xm-sidebar-toggle"' in html_doc
    assert 'aria-controls="xm-sidebar"' in html_doc
    assert 'aria-expanded="true"' in html_doc
    assert 'id="xm-sidebar"' in html_doc
    assert 'id="xm-drawer-backdrop"' in html_doc
    assert 'id="xm-sidebar-close"' in html_doc
    assert "@media (max-width: 780px)" in html_doc
    assert "NARROW_BREAKPOINT = 780" in html_doc
    assert "setSidebarOpen(window.innerWidth > NARROW_BREAKPOINT)" in html_doc
    assert "new ResizeObserver(" in html_doc
    assert 'window.addEventListener("resize", scheduleResizeRedraw)' in html_doc


@pytest.mark.parametrize(
    ("supercell", "expected_atoms"),
    [
        ((4, 4, 4), 512),
        ((7, 7, 7), 2744),
    ],
)
def test_large_system_export_speed_lod_hooks_and_xss_safety(
    tmp_path: pathlib.Path,
    supercell: tuple[int, int, int],
    expected_atoms: int,
) -> None:
    """Exports >=500 and >=2,500 atom systems in <3s with LOD & XSS intact."""
    si_unit = xm.load("si_diamond.cif")
    # Build unbonded supercell so spatial-grid bond inference is exercised.
    big_si = si_unit.repeat(supercell)
    big_si.bonds = []

    hostile_title = f"Si-{expected_atoms} </script><svg onload=alert(1)>&"
    out_file = tmp_path / f"large_{expected_atoms}.html"

    t0 = time.perf_counter()
    exported = interactive.export_interactive_html(
        big_si,
        out_file,
        name=hostile_title,
    )
    elapsed = time.perf_counter() - t0
    assert elapsed < 3.0, (
        f"Export of {expected_atoms} atoms took {elapsed:.2f}s"
    )

    html_text = exported.read_text(encoding="utf-8")
    assert html_text.startswith("<!--")
    assert "<!DOCTYPE html>" in html_text
    assert html_text.lower().count("</script>") == 1
    assert "<svg onload=alert(1)>" not in html_text
    assert "Patrick Hand" in html_text

    # Verify perf metadata in embedded JSON
    payload = _extract_embedded_payload(html_text)
    active = payload["active"]
    assert active["name"] == hostile_title
    perf = active["perf"]
    assert perf["atomCount"] == expected_atoms
    assert perf["bondCount"] == expected_atoms * 2
    assert perf["largeSystem"] is True
    assert perf["autoLod"] is True

    # Verify TypedArray hot-loop & interactive drag LOD hooks in viewer.html
    for js_hook in (
        "Float32Array",
        "Int32Array",
        "sortDepthQueue",
        "LOD_SETTLE_MS = 120",
        "isDragging",
        "drawAtomFast",
        'id="xm-lod-badge"',
        'id="xm-full-detail"',
    ):
        assert js_hook in html_text

    # Small structure (<400 atoms) flags largeSystem=False and autoLod=False
    small_payload = interactive.serialize_structure("aspirin.xyz")
    assert small_payload["perf"] == {
        "atomCount": 21,
        "bondCount": 21,
        "largeSystem": False,
        "autoLod": False,
    }


def test_python_snippet_parity_and_signature_validity_across_domains() -> None:
    """Checks snippet AST, inspect.signature parity, & execution on 4 types."""
    matplotlib.use("Agg")

    # 1. Molecule with camera, theme preset, hbonds, labels, & custom bond_scale
    mol_scene = interactive.serialize_structure("aspirin.xyz")
    mol_code = interactive.generate_python_snippet(
        mol_scene,
        camera={
            "azimuth": -132.5,
            "elevation": 31.0,
            "roll": 5.0,
            "perspective": 0.40,
            "zoom": 1.15,
        },
        preset="notebook",
        wiggle=1.25,
        background="#FFFFFF",
        style="ball_and_stick",
        palette={"C": "#333333", "O": "#D55E00", "H": "#EEEEEE"},
        bond_scale=1.25,
        show_bonds=True,
        show_hbonds=True,
        show_labels=True,
        show_legend=False,
    )
    mol_calls = _assert_valid_xm_snippet(mol_code)
    assert "infer_bonds" in mol_calls
    assert "Camera" in mol_calls
    assert "xkcd_style" in mol_calls
    assert "plot_molecule" in mol_calls
    assert "show_hbonds=True" in mol_code
    assert "show_labels=True" in mol_code
    assert 'preset="notebook"' in mol_code

    # 2. Molecule / Crystal with non-default atom_scale -> render_structure_3d
    scaled_code = interactive.generate_python_snippet(
        mol_scene,
        atom_scale=1.35,
        bond_scale=1.15,
        show_bonds=False,
    )
    scaled_calls = _assert_valid_xm_snippet(scaled_code)
    assert "render_structure_3d" in scaled_calls
    assert "radius_scale=1.35" in scaled_code
    assert "bond_scale=1.15" in scaled_code
    assert "show_bonds=False" in scaled_code

    # 3. Periodic Crystal with cell mode, crystal_axes, supercell, & centering
    cryst_scene = interactive.serialize_structure("si_diamond.cif")
    cryst_code = interactive.generate_python_snippet(
        cryst_scene,
        camera=xm.Camera(azimuth=-110.0, elevation=28.0, perspective=0.30),
        style="stick",
        palette="colorblind",
        preset="dark_chalk",
        show_cell="wireframe",
        crystal_axes="origin",
        supercell=(2, 1, 1),
        center_in_cell=True,
        boundary_atoms=False,
    )
    cryst_calls = _assert_valid_xm_snippet(cryst_code)
    assert "plot_crystal" in cryst_calls
    assert 'show_cell="wireframe"' in cryst_code
    assert 'crystal_axes="origin"' in cryst_code
    assert "supercell=(2, 1, 1)" in cryst_code
    assert "center_in_cell=True" in cryst_code
    assert "boundary_atoms=False" in cryst_code

    # Coordination polyhedra crystal snippet
    poly_scene = interactive.serialize_structure(
        "tio2_slab.poscar",
        style="polyhedra",
        polyhedra=("Ti", "O"),
    )
    poly_code = interactive.generate_python_snippet(
        poly_scene,
        crystal_axes="corner",
    )
    poly_calls = _assert_valid_xm_snippet(poly_code)
    assert "plot_coordination" in poly_calls
    assert '"Ti"' in poly_code
    assert '"O"' in poly_code

    # 4. Volumetric Cube with isovalue, iso_shade, opacity, & +/- lobe colors
    cube_scene = interactive.serialize_structure(
        "water_homo.cube",
        iso_shade="crosshatch",
    )
    assert "iso_grid" in cube_scene
    cube_code = interactive.generate_python_snippet(
        cube_scene,
        isovalue=0.14,
        iso_shade="contour",
        iso_alpha=0.42,
        iso_colors=("#1B9E77", "#D95F02"),
        preset="minimal",
        show_atoms=False,
    )
    cube_calls = _assert_valid_xm_snippet(cube_code)
    assert "plot_isosurface" in cube_calls
    assert "level=0.14" in cube_code
    assert 'shade="contour"' in cube_code
    assert "alpha=0.42" in cube_code
    assert 'colors=("#1B9E77", "#D95F02")' in cube_code
    assert "show_atoms=False" in cube_code

    # 5. Trajectory with explicit frame_index & animate_trajectory export
    neb_traj = xm.load_trajectory("sn2_neb.xyz")
    traj_scene = interactive.serialize_structure(neb_traj)
    traj_code = interactive.generate_python_snippet(
        traj_scene,
        source="sn2_neb.xyz",
        frame_index=4,
        animate=True,
        fps=16,
        show_hbonds=True,
        preset="poster",
    )
    traj_calls = _assert_valid_xm_snippet(traj_code)
    assert "load_trajectory" in traj_calls
    assert "traj[4]" in traj_code
    assert "plot_molecule" in traj_calls
    assert "animate_trajectory" in traj_calls
    assert "fps=16" in traj_code

    # Execute non-GIF snippets end-to-end to prove runtime validity
    for runnable in (mol_code, scaled_code, cryst_code, poly_code, cube_code):
        ns: dict[str, Any] = {}
        exec(compile(runnable, "<snippet>", "exec"), ns)  # noqa: S102
        assert "ax" in ns
        plt.close("all")

    with pytest.raises(TypeError, match="Unsupported"):
        interactive.generate_python_snippet(mol_scene, bogus_kwarg=123)


def test_viewer_html_ui_controls_match_python_snippet_contract() -> None:
    """Ensures every Studio UI control ID and JS snippet call stays synced."""
    neb = xm.load_trajectory("sn2_neb.xyz")
    html_doc = interactive.build_interactive_html(
        neb,
        isosurface="water_homo.cube",
        vectors=np.ones((len(neb[0]), 3)) * 0.1,
    )
    required_control_ids = (
        "xm-sidebar-toggle",
        "xm-traj-slider",
        "xm-traj-play",
        "xm-traj-fps",
        "xm-preset",
        "xm-style",
        "xm-palette",
        "xm-bg-color",
        "xm-cell-mode",
        "xm-axes-mode",
        "xm-show-bonds",
        "xm-show-hbonds",
        "xm-show-labels",
        "xm-boundary-atoms",
        "xm-center-cell",
        "xm-legend",
        "xm-show-vectors",
        "xm-show-iso",
        "xm-full-detail",
        "xm-iso-shade",
        "xm-iso-level",
        "xm-iso-alpha",
        "xm-iso-pos-color",
        "xm-iso-neg-color",
        "xm-iso-atoms",
        "xm-persp",
        "xm-zoom",
        "xm-scale",
        "xm-bond-scale",
        "xm-wiggle",
        "xm-na",
        "xm-nb",
        "xm-nc",
        "xm-py-preview",
    )
    for cid in required_control_ids:
        assert f'id="{cid}"' in html_doc, f"Missing UI control #{cid}"

    for api_token in (
        "xm.load_trajectory(",
        "xm.plot_isosurface",
        "xm.plot_crystal",
        "xm.plot_molecule",
        "xm.plot_coordination",
        "xm.render_structure_3d",
        "xm.animate_trajectory",
        "xm.infer_bonds(",
        "xm.Camera",
        "xm.xkcd_style",
    ):
        assert api_token in html_doc
