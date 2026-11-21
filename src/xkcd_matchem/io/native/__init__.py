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
"""Built-in zero-dependency parsers registering into xkcd-matchem."""

from __future__ import annotations

from xkcd_matchem.io.native import cif
from xkcd_matchem.io.native import cp2k
from xkcd_matchem.io.native import cube
from xkcd_matchem.io.native import gaussian
from xkcd_matchem.io.native import gromacs
from xkcd_matchem.io.native import lammps
from xkcd_matchem.io.native import orca
from xkcd_matchem.io.native import pdb
from xkcd_matchem.io.native import qe
from xkcd_matchem.io.native import sdf
from xkcd_matchem.io.native import vasp
from xkcd_matchem.io.native import xyz


__all__ = [
    "cif",
    "cp2k",
    "cube",
    "gaussian",
    "gromacs",
    "lammps",
    "orca",
    "pdb",
    "qe",
    "sdf",
    "vasp",
    "xyz",
]
