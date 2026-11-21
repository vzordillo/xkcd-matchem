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
"""AST and license checks enforcing Google Python Style Guide rules in src/."""

from __future__ import annotations

import ast
import pathlib


_SRC_DIR = (
    pathlib.Path(__file__).resolve().parent.parent / "src" / "xkcd_matchem"
)
_EXEMPT_FROM_MODULES = frozenset(
    {"typing", "collections.abc", "typing_extensions"}
)


def _known_submodules() -> set[str]:
    """Enumerates all package and module paths inside xkcd_matchem."""
    mods: set[str] = set()
    for py_file in _SRC_DIR.rglob("*.py"):
        rel = py_file.relative_to(_SRC_DIR.parent)
        parts = list(rel.parts)
        if parts[-1] == "__init__.py":
            parts = parts[:-1]
        else:
            parts[-1] = parts[-1].removesuffix(".py")
        mods.add(".".join(parts))
    return mods


def test_apache_header_and_google_style_ast() -> None:
    internal_mods = _known_submodules()
    external_known_modules = {
        "matplotlib.axes",
        "matplotlib.colors",
        "matplotlib.font_manager",
        "matplotlib.lines",
        "matplotlib.patches",
        "matplotlib.pyplot",
        "matplotlib.animation",
        "matplotlib.figure",
        "importlib.util",
        "scipy.optimize",
        "scipy.spatial",
        "scipy.stats",
        "scipy.interpolate",
        "ase.io",
        "pymatgen.core.structure",
        "cclib.io",
        "rdkit.Chem",
        "rdkit.Chem.AllChem",
        "numpy",
    }

    for py_file in sorted(_SRC_DIR.rglob("*.py")):
        text = py_file.read_text(encoding="utf-8")
        assert "Apache License, Version 2.0" in text[:400], (
            f"Missing header: {py_file}"
        )
        tree = ast.parse(text, filename=str(py_file))
        for node in ast.walk(tree):
            assert not isinstance(node, ast.Assert), (
                f"Forbidden assert in {py_file}"
            )
            if isinstance(node, ast.FunctionDef):
                for dec in node.decorator_list:
                    if isinstance(dec, ast.Name) and dec.id == "staticmethod":
                        raise AssertionError(
                            f"Forbidden @staticmethod in {py_file}"
                        )
            if isinstance(node, ast.ImportFrom):
                assert node.level == 0, (
                    f"Forbidden relative import in {py_file}"
                )
                mod = node.module or ""
                if mod in _EXEMPT_FROM_MODULES or mod == "__future__":
                    continue
                for alias in node.names:
                    target = f"{mod}.{alias.name}" if mod else alias.name
                    assert (
                        target in internal_mods
                        or target in external_known_modules
                    ), (
                        f"Google style 2.2 module import failed in {py_file}: "
                        f"from {mod} import {alias.name}"
                    )
