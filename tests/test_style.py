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
"""Tests for bundled fonts, non-leaking theme presets, and color palettes."""

from __future__ import annotations

import io
import pathlib
import warnings

import matplotlib
from matplotlib import pyplot as plt
import pytest

from xkcd_matchem.core import exceptions
from xkcd_matchem.style import fonts
from xkcd_matchem.style import palettes
from xkcd_matchem.style import patterns
from xkcd_matchem.style import theme


def test_bundled_fonts_and_no_rcparams_leak() -> None:
    registered = fonts.ensure_fonts_registered()
    assert "xkcd-script.ttf" in registered
    assert "PatrickHand-Regular.ttf" in registered
    assert "ComicNeue-Regular.ttf" in registered
    assert fonts.get_bundled_font_path("xkcd Script").is_file()
    assert fonts.get_bundled_font_path("Patrick Hand").is_file()
    before_bg = matplotlib.rcParams["figure.facecolor"]
    before_sketch = matplotlib.rcParams["path.sketch"]

    with theme.xkcd_style(preset="dark_chalk"):
        assert matplotlib.rcParams["figure.facecolor"] == palettes.DARK_BOARD
        assert matplotlib.rcParams["path.effects"] == []

    assert matplotlib.rcParams["figure.facecolor"] == before_bg
    assert matplotlib.rcParams["path.sketch"] == before_sketch


def test_all_theme_presets_and_scientific_glyphs() -> None:
    matplotlib.use("Agg")
    for preset_name in theme.list_presets():
        assert theme.get_preset(preset_name).font == "Patrick Hand"
        with theme.xkcd_style(preset=preset_name):
            fig, ax = plt.subplots()
            ax.plot([0.0, 1.0], [1.0, 2.0])
            ax.set_title(r"Si (111): d = 3.14 Å, ΔE = −1.12 eV ± 0.02 (λ)")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fig.canvas.draw()
            missing_warnings = [
                str(w.message)
                for w in caught
                if "missing from font" in str(w.message)
            ]
            assert not missing_warnings, missing_warnings
            plt.close(fig)


def test_palettes_and_errors() -> None:
    assert len(palettes.get_palette("okabe_ito")) == 8
    assert "pastel" in palettes.list_element_palettes()
    assert palettes.get_element_color("O", scheme="jmol") == "#FF0D0D"
    assert palettes.get_element_color("N", scheme="colorblind") == "#0072B2"
    assert palettes.get_element_color("Fe", scheme="colorblind") == "#0072B2"
    assert (
        palettes.get_element_color("Ti", scheme={"Ti": "#E07A5F"}) == "#E07A5F"
    )
    with pytest.raises(exceptions.StyleError):
        theme.get_preset("nonexistent_preset")
    with pytest.raises(exceptions.StyleError):
        palettes.get_element_color("C", scheme="unknown")


def test_custom_presets_palettes_and_existing_axes() -> None:
    matplotlib.use("Agg")
    theme.register_preset(
        "test_custom_theme", base="classic", figure_facecolor="#FAF6EE"
    )
    assert "test_custom_theme" in theme.list_presets()
    assert theme.get_preset("test_custom_theme").figure_facecolor == "#FAF6EE"

    palettes.register_palette("test_series", ["#111111", "#222222", "#333333"])
    assert palettes.get_palette("test_series") == (
        "#111111",
        "#222222",
        "#333333",
    )
    palettes.register_element_palette(
        "test_elem", {"Si": "#00AA88"}, base="pastel"
    )
    assert "test_elem" in palettes.list_element_palettes()
    assert palettes.get_element_color("Si", scheme="test_elem") == "#00AA88"

    fig, ax = plt.subplots()
    theme.style_existing_axes(ax, preset="dark_chalk")
    assert ax.get_facecolor()[:3] != (1.0, 1.0, 1.0)
    with theme.xkcd_style(
        "classic",
        palette=["#123456", "#654321"],
        background="#F0F4F8",
        ink="#222222",
    ):
        assert matplotlib.rcParams["figure.facecolor"] == "#F0F4F8"
        assert matplotlib.rcParams["axes.edgecolor"] == "#222222"
    plt.close(fig)

    # Nested xkcd_style preserves outer font override; set_default_font works.
    cn_path = fonts.get_bundled_font_path("Comic Neue")
    assert fonts.register_font(cn_path) == "Comic Neue"
    with (
        theme.xkcd_style("classic", font="Comic Neue"),
        theme.xkcd_style("classic"),
    ):
        assert matplotlib.rcParams["font.family"][0] == "Comic Neue"
    try:
        assert fonts.set_default_font(cn_path) == "Comic Neue"
        assert fonts.get_default_font() == "Comic Neue"
        with theme.xkcd_style("classic"):
            assert matplotlib.rcParams["font.family"][0] == "Comic Neue"
    finally:
        assert fonts.set_default_font(None) == "Patrick Hand"
        assert fonts.get_default_font() is None


def test_fonts_palettes_and_patterns_complete_coverage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    """Covers error guards and branches in fonts, palettes, and patterns."""
    matplotlib.use("Agg")
    # 1. Fonts error branches, fetch_xkcd_script download mocks, use_xkcd_script
    with pytest.raises(exceptions.StyleError):
        fonts.get_bundled_font_path("NonExistentFont")
    with pytest.raises(exceptions.StyleError):
        fonts.register_font(tmp_path / "missing_font.ttf")
    with pytest.raises(exceptions.StyleError):
        fonts.set_default_font("   ")

    try:
        with pytest.warns(UserWarning, match="CC BY-NC 3.0"):
            bundled = fonts.fetch_xkcd_script(cache_dir=None, activate=True)
        assert bundled.is_file()
        assert fonts.use_xkcd_script(font_path=bundled) == "xkcd Script"
    finally:
        fonts.set_default_font(None)

    def _fail_urlopen(*_args: object, **_kwargs: object) -> object:
        raise OSError("network unavailable")

    monkeypatch.setattr("urllib.request.urlopen", _fail_urlopen)
    with pytest.raises(exceptions.StyleError, match="Failed to download"):
        fonts.fetch_xkcd_script(cache_dir=tmp_path / "err_cache")

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_a, **_k: io.BytesIO(b""),
    )
    with pytest.raises(exceptions.StyleError, match="payload was empty"):
        fonts.fetch_xkcd_script(cache_dir=tmp_path / "empty_cache")

    ttf_bytes = fonts.get_bundled_font_path("xkcd Script").read_bytes()
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_a, **_k: io.BytesIO(ttf_bytes),
    )
    try:
        with pytest.warns(UserWarning, match="LICENSE NOTICE"):
            dl_path = fonts.fetch_xkcd_script(
                cache_dir=tmp_path / "ok_cache", activate=True
            )
        assert dl_path.is_file()
    finally:
        fonts.set_default_font(None)

    # 2. Palettes error branches and listings
    assert "okabe_ito" in palettes.list_palettes()
    with pytest.raises(exceptions.StyleError):
        palettes.register_palette("   ", ["#000000"])
    with pytest.raises(exceptions.StyleError):
        palettes.register_palette("empty_pal", [])
    with pytest.raises(exceptions.StyleError):
        palettes.register_element_palette("   ", {"C": "#000000"})
    with pytest.raises(exceptions.StyleError):
        palettes.register_element_palette(
            "bad_base", {"C": "#000000"}, base="unknown_base"
        )
    with pytest.raises(exceptions.StyleError):
        palettes.get_palette([])
    with pytest.raises(exceptions.StyleError):
        palettes.get_palette("unknown_series_palette")

    # 3. Patterns hatch sequences, dotgrid clear, coords, and shade_between
    assert patterns.resolve_hatch("//") == "//"
    assert patterns.resolve_hatch([]) is None
    assert patterns.resolve_hatch(["hatch", "dots"], index=1) == "..."
    xs0, ys0 = patterns.compute_background_grid_coords(None)
    assert xs0.size == 0
    assert ys0.size == 0
    xs_u, ys_u = patterns.compute_background_grid_coords("unknown_bg")
    assert xs_u.size == 0
    assert ys_u.size == 0

    fig, ax = plt.subplots()
    patterns.apply_background_grid(ax, style="dotgrid")
    assert len(ax.collections) == 1
    patterns.clear_background_grid(ax)
    assert len(ax.collections) == 0
    patterns.apply_background_grid(
        ax, style="grid", sketch=(1.1, 80.0, 2.0)
    )
    patterns.apply_background_grid(ax, style="blueprint", sketch=True)
    patterns.apply_background_grid(ax, style="none")
    patterns.shade_between(ax, [0.0, 1.0], [1.0, 2.0], 0.0, shade="hatch")
    patterns.shade_between(
        ax, [0.0, 1.0], [1.0, 2.0], 0.0, shade="solid", vertical=True
    )
    plt.close(fig)

