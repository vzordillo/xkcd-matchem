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
"""Font utilities for bundled SIL OFL 1.1 and optional xkcd Script fonts."""

from __future__ import annotations

import os
import pathlib
import urllib.request
import warnings

from matplotlib import font_manager

from xkcd_matchem.core import exceptions


_FONTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "fonts"

# Track registered font files within the process to keep registration
# idempotent across repeated xkcd_style context entries.
_REGISTERED_FONT_PATHS: set[pathlib.Path] = set()
_DEFAULT_FONT_STATE: list[str | None] = [None]
_XKCD_SCRIPT_WARNED: list[bool] = [False]

_BUNDLED_FONT_FILES: dict[str, str] = {
    "Patrick Hand": "PatrickHand-Regular.ttf",
    "Comic Neue": "ComicNeue-Regular.ttf",
    "Comic Neue Bold": "ComicNeue-Bold.ttf",
    "xkcd Script": "xkcd-script.ttf",
}

_DEFAULT_BUNDLED_FONT = "Patrick Hand"
_XKCD_SCRIPT_FONT = "xkcd Script"
_DEFAULT_FALLBACK_FONTS: tuple[str, ...] = ("DejaVu Sans",)

_XKCD_SCRIPT_URL = (
    "https://github.com/ipython/xkcd-font/raw/master/"
    "xkcd-script/font/xkcd-script.ttf"
)

_XKCD_SCRIPT_NOTICE = (
    "LICENSE NOTICE: 'xkcd Script' (xkcd-script.ttf, from ipython/xkcd-font "
    "/ Randall Munroe) is licensed under Creative Commons "
    "Attribution-NonCommercial 3.0 (CC BY-NC 3.0) and is restricted to "
    "non-commercial use only. Commercial-safe SIL OFL 1.1 defaults are "
    "'Patrick Hand' (default) and 'Comic Neue' via "
    "set_default_font('Patrick Hand') or set_default_font('Comic Neue')."
)

_XKCD_SCRIPT_COMMERCIAL_ERROR = (
    "Cannot activate 'xkcd Script' while XM_COMMERCIAL_SAFE is enabled: "
    "xkcd-script.ttf is licensed under CC BY-NC 3.0 (non-commercial use "
    "only). Use the commercial-safe SIL OFL 1.1 defaults 'Patrick Hand' "
    "or 'Comic Neue' instead."
)

def is_commercial_safe() -> bool:
    """Returns True when ``XM_COMMERCIAL_SAFE`` forbids CC BY-NC fonts."""
    return os.environ.get("XM_COMMERCIAL_SAFE", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _is_xkcd_script_font(family: str) -> bool:
    """Returns True if ``family`` names the CC BY-NC ``xkcd Script`` font."""
    normalized = family.strip().lower().replace("_", " ").replace("-", " ")
    return normalized in ("xkcd script", "xkcdscript")


def _enforce_font_license(
    family: str,
    *,
    warn: bool = True,
    stacklevel: int = 3,
) -> None:
    """Blocks or warns once on activation of CC BY-NC ``xkcd Script``."""
    if not _is_xkcd_script_font(family):
        return
    if is_commercial_safe():
        raise exceptions.StyleError(_XKCD_SCRIPT_COMMERCIAL_ERROR)
    if warn and not _XKCD_SCRIPT_WARNED[0]:
        _XKCD_SCRIPT_WARNED[0] = True
        warnings.warn(_XKCD_SCRIPT_NOTICE, UserWarning, stacklevel=stacklevel)


def list_bundled_fonts() -> tuple[str, ...]:
    """Returns the font family names bundled with xkcd-matchem."""
    return tuple(_BUNDLED_FONT_FILES.keys())


def get_bundled_font_path(family: str) -> pathlib.Path:
    """Resolves the filesystem path for a bundled comic-style font file.

    Args:
      family: Font display key, e.g. ``"xkcd Script"``, ``"Patrick Hand"``,
        or ``"Comic Neue"``.

    Returns:
      Path to the ``.ttf`` file inside ``xkcd_matchem/data/fonts``.

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


def resolve_font_chain(preferred: str = _DEFAULT_BUNDLED_FONT) -> list[str]:
    """Builds an ordered Matplotlib ``font.family`` fallback list.

    Ensures bundled comic-style fonts are registered and appends DejaVu Sans
    so scientific glyphs (degree, superscripts, Greek letters, arrows) render
    cleanly without missing-glyph warnings.

    Args:
      preferred: Preferred primary font family name.

    Returns:
      Ordered list of font family names starting with ``preferred``.
    """
    _enforce_font_license(preferred, warn=False, stacklevel=3)
    ensure_fonts_registered()
    chain = [preferred]
    for fallback in _DEFAULT_FALLBACK_FONTS:
        if fallback not in chain:
            chain.append(fallback)
    return chain


def register_font(font_path: str | pathlib.Path) -> str:
    """Registers a local ``.ttf`` or ``.otf`` font file with Matplotlib.

    Args:
      font_path: Path to a local TrueType or OpenType font file.

    Returns:
      Matplotlib font family name read from the font metadata.

    Raises:
      StyleError: If ``font_path`` does not point to an existing file.
    """
    resolved = pathlib.Path(font_path).expanduser().resolve()
    if not resolved.is_file():
        raise exceptions.StyleError(f"Font file not found: {resolved}")
    if resolved not in _REGISTERED_FONT_PATHS:
        font_manager.fontManager.addfont(str(resolved))
        _REGISTERED_FONT_PATHS.add(resolved)
    return str(font_manager.FontProperties(fname=str(resolved)).get_name())


def get_default_font() -> str | None:
    """Returns active session font override or ``XM_COMMERCIAL_SAFE`` font."""
    if _DEFAULT_FONT_STATE[0] is not None:
        if is_commercial_safe() and _is_xkcd_script_font(
            _DEFAULT_FONT_STATE[0]
        ):
            return _DEFAULT_BUNDLED_FONT
        return _DEFAULT_FONT_STATE[0]
    if is_commercial_safe():
        return _DEFAULT_BUNDLED_FONT
    return None


def set_default_font(font: str | pathlib.Path | None) -> str:
    """Sets or resets the default font used across all ``xkcd-matchem`` plots.

    Args:
      font: Font family name (for example ``"Patrick Hand"``,
        ``"Comic Neue"``, or non-commercial opt-in ``"xkcd Script"``), path
        to a local ``.ttf``/``.otf`` file to register and activate, or ``None``
        to reset to the commercial-safe ``"Patrick Hand"`` default.

    Returns:
      The resolved active default font family name.

    Raises:
      StyleError: If ``font`` is blank, missing, or attempts to activate
        CC BY-NC ``"xkcd Script"`` while ``XM_COMMERCIAL_SAFE`` is enabled.
    """
    if font is None:
        _DEFAULT_FONT_STATE[0] = None
        _XKCD_SCRIPT_WARNED[0] = False
        return _DEFAULT_BUNDLED_FONT
    candidate = pathlib.Path(font).expanduser()
    if isinstance(font, pathlib.Path) or (
        candidate.suffix.lower() in (".ttf", ".otf", ".ttc")
        and candidate.is_file()
    ):
        if (
            is_commercial_safe()
            and candidate.name.lower() == "xkcd-script.ttf"
        ):
            raise exceptions.StyleError(_XKCD_SCRIPT_COMMERCIAL_ERROR)
        family = register_font(candidate)
    else:
        family = str(font).strip()
        if not family:
            raise exceptions.StyleError("Font name cannot be empty.")
    _enforce_font_license(family, warn=True, stacklevel=3)
    _DEFAULT_FONT_STATE[0] = family
    return family


def fetch_xkcd_script(
    cache_dir: str | pathlib.Path | None = None,
    *,
    activate: bool = True,
) -> pathlib.Path:
    """Returns the bundled ``xkcd-script.ttf`` (or fetches into ``cache_dir``).

    ``xkcd-script`` is licensed under CC BY-NC 3.0 (non-commercial use only)
    and is bundled strictly as an explicit opt-in typeface. The default
    typeface is commercial-safe SIL OFL 1.1 ``"Patrick Hand"``. Pass
    ``cache_dir`` to download a standalone copy into a custom directory.

    Args:
      cache_dir: Optional directory for a standalone downloaded ``.ttf`` copy.
        When ``None``, returns the bundled ``xkcd-script.ttf`` offline.
      activate: Whether to set ``"xkcd Script"`` as the active default font for
        all subsequent ``xkcd-matchem`` plots in this Python session.

    Returns:
      Path to ``xkcd-script.ttf`` registered in Matplotlib.

    Raises:
      StyleError: If ``XM_COMMERCIAL_SAFE`` is enabled while ``activate=True``,
        or if the network download fails or produces an empty file.
    """
    if activate and is_commercial_safe():
        raise exceptions.StyleError(_XKCD_SCRIPT_COMMERCIAL_ERROR)
    if cache_dir is None:
        bundled = get_bundled_font_path(_XKCD_SCRIPT_FONT)
        family = register_font(bundled)
        if activate:
            _enforce_font_license(family, warn=True, stacklevel=3)
            _DEFAULT_FONT_STATE[0] = family
        return bundled

    target_dir = pathlib.Path(cache_dir).expanduser()
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
                f"Failed to download xkcd-script font: {exc}"
            ) from exc
        if not payload:
            raise exceptions.StyleError(
                "Downloaded xkcd-script payload was empty."
            )
        target_file.write_bytes(payload)
    family = register_font(target_file)
    if activate:
        _enforce_font_license(family, warn=True, stacklevel=3)
        _DEFAULT_FONT_STATE[0] = family
    else:
        warnings.warn(_XKCD_SCRIPT_NOTICE, UserWarning, stacklevel=2)
    return target_file


def use_xkcd_script(
    font_path: str | pathlib.Path | None = None,
    cache_dir: str | pathlib.Path | None = None,
) -> str:
    """Explicitly opts into the non-commercial CC BY-NC ``xkcd Script`` font.

    Args:
      font_path: Optional path to a custom local ``xkcd-script.ttf`` file.
      cache_dir: Optional download cache directory when fetching separately.

    Returns:
      The registered font family name (``"xkcd Script"``).

    Raises:
      StyleError: If ``XM_COMMERCIAL_SAFE`` is enabled.
    """
    if is_commercial_safe():
        raise exceptions.StyleError(_XKCD_SCRIPT_COMMERCIAL_ERROR)
    if font_path is not None:
        return set_default_font(pathlib.Path(font_path))
    resolved = fetch_xkcd_script(cache_dir=cache_dir, activate=True)
    return register_font(resolved)
