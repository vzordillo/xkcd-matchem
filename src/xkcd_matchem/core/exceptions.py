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

from xkcd_matchem import _exceptions


__all__ = [
    "MatchemError",
    "OptionalDependencyError",
    "ParseError",
    "StructureError",
    "StyleError",
    "UnsupportedFormatError",
]

MatchemError = _exceptions.MatchemError
OptionalDependencyError = _exceptions.OptionalDependencyError
ParseError = _exceptions.ParseError
StructureError = _exceptions.StructureError
StyleError = _exceptions.StyleError
UnsupportedFormatError = _exceptions.UnsupportedFormatError
