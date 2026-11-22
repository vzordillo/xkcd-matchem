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
"""Security, parser resource ceiling, HTML escaping, and font license tests."""

from __future__ import annotations

import json
import pathlib
import warnings

import pytest

from xkcd_matchem.core import exceptions
from xkcd_matchem.io.native import _limits
from xkcd_matchem.io.native import cif
from xkcd_matchem.io.native import cube
from xkcd_matchem.io.native import vasp
from xkcd_matchem.io.native import xyz
from xkcd_matchem.render import interactive
from xkcd_matchem.style import fonts
from xkcd_matchem.style import theme


def test_patrick_hand_sil_ofl_is_default_across_all_presets() -> None:
    """Every built-in ThemePreset defaults to SIL OFL 1.1 Patrick Hand."""
    assert fonts.set_default_font(None) == "Patrick Hand"
    assert fonts.resolve_font_chain()[0] == "Patrick Hand"
    for preset_name, preset in theme.PRESETS.items():
        assert preset.font == "Patrick Hand", preset_name
        rc = theme.build_rc_params(preset=preset_name)
        assert rc["font.family"][0] == "Patrick Hand"
        assert rc["mathtext.rm"] == "Patrick Hand"
    html_doc = interactive.build_interactive_html("aspirin.xyz")
    assert "Patrick Hand" in html_doc
    assert "xkcd Script" not in html_doc


def test_xkcd_script_opt_in_warning_and_commercial_safe_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Opt-in xkcd Script warns CC BY-NC and raises under XM_COMMERCIAL_SAFE."""
    monkeypatch.delenv("XM_COMMERCIAL_SAFE", raising=False)
    fonts.set_default_font(None)
    try:
        # First opt-in emits the CC BY-NC 3.0 UserWarning once.
        with pytest.warns(UserWarning, match="CC BY-NC 3.0") as rec:
            activated = fonts.use_xkcd_script()
        assert activated == "xkcd Script"
        assert fonts.get_default_font() == "xkcd Script"
        assert any("Patrick Hand" in str(w.message) for w in rec)
        assert any("Comic Neue" in str(w.message) for w in rec)

        # Subsequent opt-in calls stay silent (one-time warning per session).
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fonts.use_xkcd_script()
            fonts.set_default_font("xkcd Script")
            nc_warns = [
                w
                for w in caught
                if issubclass(w.category, UserWarning)
                and "CC BY-NC 3.0" in str(w.message)
            ]
            assert not nc_warns

        # Resetting to default and opting in via set_default_font warns once.
        fonts.set_default_font(None)
        with pytest.warns(UserWarning, match="CC BY-NC 3.0"):
            assert fonts.set_default_font("xkcd Script") == "xkcd Script"
    finally:
        fonts.set_default_font(None)

    for env_val in ("1", "true", "YES"):
        monkeypatch.setenv("XM_COMMERCIAL_SAFE", env_val)
        assert fonts.is_commercial_safe()
        assert fonts.get_default_font() == "Patrick Hand"
        with pytest.raises(exceptions.StyleError, match="CC BY-NC 3.0"):
            fonts.use_xkcd_script()
        with pytest.raises(exceptions.StyleError, match="CC BY-NC 3.0"):
            fonts.set_default_font("xkcd Script")
        bundled_xkcd = fonts.get_bundled_font_path("xkcd Script")
        with pytest.raises(exceptions.StyleError, match="CC BY-NC 3.0"):
            fonts.set_default_font(bundled_xkcd)
        with pytest.raises(exceptions.StyleError, match="CC BY-NC 3.0"):
            theme.build_rc_params(font="xkcd Script")


def test_limits_env_overrides_and_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    """Resource ceiling helpers honor defaults, overrides, and bad env vars."""
    for var in (
        "XM_MAX_ATOMS",
        "XM_MAX_GRID_VOXELS",
        "XM_MAX_FRAMES",
        "XM_MAX_FILE_BYTES",
    ):
        monkeypatch.delenv(var, raising=False)
    assert _limits.max_atoms() == 200_000
    assert _limits.max_grid_voxels() == 64_000_000
    assert _limits.max_frames() == 50_000
    assert _limits.max_file_bytes() == 200 * 1024 * 1024

    monkeypatch.setenv("XM_MAX_ATOMS", "0")
    with pytest.raises(exceptions.ParseError, match="positive integer"):
        _limits.max_atoms()

    monkeypatch.setenv("XM_MAX_ATOMS", "not_a_number")
    with pytest.raises(exceptions.ParseError, match="positive integer"):
        _limits.max_atoms()
    monkeypatch.delenv("XM_MAX_ATOMS")

    small_file = tmp_path / "payload.xyz"
    small_file.write_text("1\ncomment\nH 0 0 0\n", encoding="utf-8")
    monkeypatch.setenv("XM_MAX_FILE_BYTES", "5")
    with pytest.raises(exceptions.ParseError, match="XM_MAX_FILE_BYTES"):
        xyz.read_xyz(small_file)


def test_adversarial_parser_resource_ceilings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    """Adversarial huge counts in cube/xyz/POSCAR/CHGCAR/CIF raise fast."""
    # 1. CUBE: huge nx*ny*nz voxel grid (1000^3 = 1e9 > 64e6) and huge atoms
    huge_cube = tmp_path / "huge_grid.cube"
    huge_cube.write_text(
        "Adversarial cube\nHuge grid\n"
        "1 0.0 0.0 0.0\n"
        "1000 0.1 0.0 0.0\n"
        "1000 0.0 0.1 0.0\n"
        "1000 0.0 0.0 0.1\n"
        "1 1.0 0.0 0.0 0.0\n"
        "0.0\n",
        encoding="utf-8",
    )
    with pytest.raises(exceptions.ParseError, match="XM_MAX_GRID_VOXELS"):
        cube.read_cube(huge_cube)

    huge_cube_atoms = tmp_path / "huge_atoms.cube"
    huge_cube_atoms.write_text(
        "Adversarial cube\nHuge atoms\n"
        "999999999 0.0 0.0 0.0\n"
        "2 0.1 0.0 0.0\n"
        "2 0.0 0.1 0.0\n"
        "2 0.0 0.0 0.1\n"
        "1 1.0 0.0 0.0 0.0\n",
        encoding="utf-8",
    )
    with pytest.raises(exceptions.ParseError, match="XM_MAX_ATOMS"):
        cube.read_cube(huge_cube_atoms)

    # 2. XYZ: huge n_atoms header & trajectory frame ceiling
    huge_xyz = tmp_path / "huge.xyz"
    huge_xyz.write_text(
        "999999999\nAdversarial XYZ\nH 0.0 0.0 0.0\n",
        encoding="utf-8",
    )
    with pytest.raises(exceptions.ParseError, match="XM_MAX_ATOMS"):
        xyz.read_xyz(huge_xyz)

    multi_xyz = tmp_path / "frames.xyz"
    multi_xyz.write_text(
        "1\nf1\nH 0 0 0\n1\nf2\nH 0 0 1\n1\nf3\nH 0 0 2\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("XM_MAX_FRAMES", "2")
    with pytest.raises(exceptions.ParseError, match="XM_MAX_FRAMES"):
        xyz.read_xyz_trajectory(multi_xyz)
    monkeypatch.delenv("XM_MAX_FRAMES")

    # 3. VASP POSCAR: huge sum(counts) before list allocation & CHGCAR grid
    huge_poscar = tmp_path / "POSCAR"
    huge_poscar.write_text(
        "Adversarial POSCAR\n"
        "1.0\n"
        "5.0 0.0 0.0\n"
        "0.0 5.0 0.0\n"
        "0.0 0.0 5.0\n"
        "Si O\n"
        "500000000 500000000\n"
        "Direct\n"
        "0.0 0.0 0.0\n",
        encoding="utf-8",
    )
    with pytest.raises(exceptions.ParseError, match="XM_MAX_ATOMS"):
        vasp.read_poscar(huge_poscar)

    huge_chgcar = tmp_path / "CHGCAR"
    huge_chgcar.write_text(
        "Adversarial CHGCAR\n"
        "1.0\n"
        "5.0 0.0 0.0\n"
        "0.0 5.0 0.0\n"
        "0.0 0.0 5.0\n"
        "Si\n"
        "1\n"
        "Direct\n"
        "0.0 0.0 0.0\n\n"
        "1000 1000 1000\n"
        "0.0\n",
        encoding="utf-8",
    )
    with pytest.raises(exceptions.ParseError, match="XM_MAX_GRID_VOXELS"):
        vasp.read_chgcar(huge_chgcar)

    # 4. CIF: asymmetric site ceiling & symmetry-expansion ceiling
    cif_file = tmp_path / "expanded.cif"
    cif_file.write_text(
        "data_test\n"
        "_cell_length_a 5.64\n"
        "_cell_length_b 5.64\n"
        "_cell_length_c 5.64\n"
        "_cell_angle_alpha 90\n"
        "_cell_angle_beta 90\n"
        "_cell_angle_gamma 90\n"
        "_space_group_IT_number 225\n"
        "loop_\n"
        "_atom_site_label\n"
        "_atom_site_type_symbol\n"
        "_atom_site_fract_x\n"
        "_atom_site_fract_y\n"
        "_atom_site_fract_z\n"
        "Na1 Na 0.0 0.0 0.0\n"
        "Cl1 Cl 0.5 0.5 0.5\n",
        encoding="utf-8",
    )
    # 2 asymmetric sites expand to 8 atoms in Fm-3m; limit 4 triggers guard.
    monkeypatch.setenv("XM_MAX_ATOMS", "4")
    with pytest.raises(exceptions.ParseError, match="XM_MAX_ATOMS"):
        cif.read_cif(cif_file)
    # Limit 1 catches raw _atom_site loop before expansion.
    monkeypatch.setenv("XM_MAX_ATOMS", "1")
    with pytest.raises(exceptions.ParseError, match="XM_MAX_ATOMS"):
        cif.read_cif(cif_file)
    monkeypatch.delenv("XM_MAX_ATOMS")


def test_interactive_html_escapes_hostile_title_and_json_xss(
    tmp_path: pathlib.Path,
) -> None:
    """Hostile XSS payloads cannot break out of <script> or <title>."""
    hostile = "</script><img src=x onerror=alert(1)>&\u2028\u2029"
    out_html = tmp_path / "xss_check.html"
    exported = interactive.export_interactive_html(
        "aspirin.xyz",
        out_html,
        name=hostile,
    )
    html_text = exported.read_text(encoding="utf-8")

    # Only the template's single closing </script> tag may appear.
    assert html_text.lower().count("</script>") == 1
    assert "<img src=x onerror=alert(1)>" not in html_text
    assert "\u2028" not in html_text
    assert "\u2029" not in html_text

    # <title> contains HTML-escaped hostile string.
    title_start = html_text.index("<title>") + len("<title>")
    title_end = html_text.index("</title>", title_start)
    title_inner = html_text[title_start:title_end]
    assert "&lt;/script&gt;&lt;img src=x onerror=alert(1)&gt;&amp;" in (
        title_inner
    )
    assert "</script>" not in title_inner

    # Embedded PAYLOAD JSON uses \u003c / \u003e / \u0026 and still parses.
    marker = "const PAYLOAD = "
    p_start = html_text.index(marker) + len(marker)
    p_end = html_text.index(";\n  const ELEM_TABLE", p_start)
    raw_json = html_text[p_start:p_end]
    assert "<" not in raw_json
    assert ">" not in raw_json
    assert "&" not in raw_json
    assert "\\u003c/script\\u003e" in raw_json
    parsed = json.loads(raw_json)
    assert parsed["active"]["name"] == hostile

    # _repr_html_ also escapes hostile titles cleanly.
    view = interactive.InteractiveView(html_str=html_text, title=hostile)
    iframe = view._repr_html_()
    assert 'title="&lt;/script&gt;&lt;img src=x onerror=alert(1)&gt;&amp;' in (
        iframe
    )
