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
"""Font utilities for bundled SIL OFL 1.1 comic-style typefaces."""

from __future__ import annotations

import pathlib
import urllib.request

from matplotlib import font_manager

from xkcd_matchem.core import exceptions


_FONTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "fonts"

# Track registered font files within the process to keep registration
# idempotent across repeated xkcd_style context entries.
_REGISTERED_FONT_PATHS: set[pathlib.Path] = set()

_BUNDLED_FONT_FILES: dict[str, str] = {
    "Patrick Hand": "PatrickHand-Regular.ttf",
    "Comic Neue": "ComicNeue-Regular.ttf",
    "Comic Neue Bold": "ComicNeue-Bold.ttf",
}

_DEFAULT_FALLBACK_FONTS: tuple[str, ...] = ("DejaVu Sans",)

_XKCD_SCRIPT_URL = (
    "https://github.com/ipython/xkcd-font/raw/master/"
    "xkcd-script/font/xkcd-script.ttf"
)

_XKCD_SCRIPT_NOTICE = (
    "Downloaded xkcd-script.ttf to %s.\n"
    "LICENSE NOTICE: xkcd-script is licensed under Creative Commons "
    "Attribution-NonCommercial 3.0 (CC BY-NC 3.0), NOT Apache-2.0 or SIL OFL.\n"
    "Verify non-commercial compliance before redistributing derived outputs."
)


def list_bundled_fonts() -> tuple[str, ...]:
    """Returns the font family names bundled with xkcd-matchem."""
    return tuple(_BUNDLED_FONT_FILES.keys())


def get_bundled_font_path(family: str) -> pathlib.Path:
    """Resolves the filesystem path for a bundled SIL OFL 1.1 font file.

    Args:
      family: Font display key, e.g. ``"Patrick Hand"`` or ``"Comic Neue"``.

    Returns:
      Path to the `.ttf` file inside ``xkcd_matchem/data/fonts``.

    Raises:
      StyleError: If the requested bundled font key is unknown or missing.
    """
    if family not in _BUNDLED_FONT_FILES:
        supported = ", ".join(sorted(_BUNDLED_FONT_FILES))
        raise exceptions.StyleError(
            f"Unknown bundled font {family!r}. Available: {supported}"
        )
    font_path = _FONTS_DIR / _BUNDLED_FONT_FILES[family]
    if not font_path.is_file():
        raise exceptions.StyleError(f"Bundled font file not found: {font_path}")
    return font_path


def ensure_fonts_registered() -> tuple[str, ...]:
    """Registers bundled comic-style fonts with Matplotlib's font manager.

    Returns:
      Tuple of newly or previously registered font file names.
    """
    loaded: list[str] = []
    for filename in _BUNDLED_FONT_FILES.values():
        font_path = _FONTS_DIR / filename
        if not font_path.is_file():
            continue
        if font_path not in _REGISTERED_FONT_PATHS:
            font_manager.fontManager.addfont(str(font_path))
            _REGISTERED_FONT_PATHS.add(font_path)
        loaded.append(filename)
    return tuple(loaded)


def resolve_font_chain(preferred: str = "Patrick Hand") -> list[str]:
    """Builds an ordered Matplotlib ``font.family`` fallback list.

    Ensures bundled comic-style fonts are registered and appends DejaVu Sans
    so scientific glyphs (angstrom, arrows, Greek letters, minus) render cleanly

    Args:
      preferred: Preferred primary font family name.

    Returns:
      Ordered list of font family names starting with ``preferred``.
    """
    ensure_fonts_registered()
    chain = [preferred]
    for fallback in _DEFAULT_FALLBACK_FONTS:
        if fallback not in chain:
            chain.append(fallback)
    return chain


def fetch_xkcd_script(
    cache_dir: str | pathlib.Path | None = None,
) -> pathlib.Path:
    """Downloads the optional ``xkcd-script`` font into a user cache directory.

    ``xkcd-script`` is licensed under CC BY-NC 3.0 and is intentionally excluded
    from package data. Users opt in to downloading it explicitly and should
    review non-commercial licensing terms before commercial publication.

    Args:
      cache_dir: Optional directory for the downloaded `.ttf` file. Defaults to
        ``~/.cache/xkcd_matchem/fonts``.

    Returns:
      Path to the downloaded ``xkcd-script.ttf`` file registered in Matplotlib.

    Raises:
      StyleError: If the network download fails or produces an empty file.
    """
    target_dir = (
        pathlib.Path(cache_dir).expanduser()
        if cache_dir is not None
        else pathlib.Path.home() / ".cache" / "xkcd_matchem" / "fonts"
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / "xkcd-script.ttf"
    if not target_file.is_file() or target_file.stat().st_size == 0:
        try:
            with urllib.request.urlopen(
                _XKCD_SCRIPT_URL, timeout=30
            ) as response:
                payload = response.read()
        except OSError as exc:
            raise exceptions.StyleError(
                f"Failed to download optional xkcd-script font: {exc}"
            ) from exc
        if not payload:
            raise exceptions.StyleError(
                "Downloaded xkcd-script payload was empty."
            )
        target_file.write_bytes(payload)
    if target_file not in _REGISTERED_FONT_PATHS:
        font_manager.fontManager.addfont(str(target_file))
        _REGISTERED_FONT_PATHS.add(target_file)
    print(_XKCD_SCRIPT_NOTICE % target_file)
    return target_file
