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
"""Custom exception hierarchy for xkcd-matchem."""

from __future__ import annotations


class MatchemError(Exception):
    """Base exception for all xkcd-matchem errors."""


class ParseError(MatchemError):
    """Raised when reading or parsing an input file fails."""


class UnsupportedFormatError(MatchemError):
    """Raised when a file format cannot be identified or is unsupported."""


class StructureError(MatchemError):
    """Raised when atomic coordinates, cells, or topology are invalid."""


class OptionalDependencyError(MatchemError):
    """Raised when an optional third-party integration package is missing."""


class StyleError(MatchemError):
    """Raised when an invalid theme preset, palette, or font is requested."""
