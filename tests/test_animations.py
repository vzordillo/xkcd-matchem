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
"""Tests for animate_orbit, animate_isosurface, and animate_trajectory."""

from __future__ import annotations

from importlib import util
import pathlib

from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import pytest

import xkcd_matchem as xm


def test_animate_orbit_crystal_polyhedra_and_molecule(
    tmp_path: pathlib.Path,
) -> None:
    """Renders TiO2 polyhedra and aspirin molecule 360-deg orbit GIFs."""
    tio2_gif = xm.animate_orbit(
        "tio2_slab.poscar",
        tmp_path / "tio2_orbit.gif",
        n_frames=3,
        fps=8,
        supercell=(2, 1, 1),
        polyhedra=("Ti", "O"),
        crystal_axes="corner",
        dpi=72,
        title="TiO2 Orbit",
    )
    assert tio2_gif.is_file()
    with Image.open(tio2_gif) as img:
        assert img.format == "GIF"
        assert getattr(img, "n_frames", 1) == 3

    mol_gif = xm.animate_orbit(
        "aspirin.xyz",
        tmp_path / "aspirin_orbit.gif",
        n_frames=2,
        fps=6,
        show_legend=False,
        preset="dark_chalk",
        dpi=72,
    )
    assert mol_gif.is_file()
    with Image.open(mol_gif) as img:
        assert getattr(img, "n_frames", 1) == 2
    assert plt.get_fignums() == []


def test_animate_orbit_validation_and_missing_pillow(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validates error branches for animate_orbit."""
    with pytest.raises(xm.StructureError, match="n_frames"):
        xm.animate_orbit("aspirin.xyz", tmp_path / "bad.gif", n_frames=1)

    empty_atoms = xm.AtomsData(symbols=[], positions=np.zeros((0, 3)))
    with pytest.raises(xm.StructureError, match="empty structure"):
        xm.animate_orbit(empty_atoms, tmp_path / "empty.gif", n_frames=2)

    with pytest.raises(xm.StructureError, match="Expected AtomsData"):
        xm.animate_orbit("si_qe_dos.dat", tmp_path / "dos.gif", n_frames=2)

    monkeypatch.setattr(util, "find_spec", lambda _name: None)
    with pytest.raises(xm.OptionalDependencyError, match="pillow"):
        xm.animate_orbit("aspirin.xyz", tmp_path / "nopil.gif", n_frames=2)


def test_animate_isosurface_light_dark_and_no_atoms(
    tmp_path: pathlib.Path,
) -> None:
    """Renders 3D orbital isosurface orbit GIFs across themes and shades."""
    light_gif = xm.animate_isosurface(
        "water_homo.cube",
        tmp_path / "water_light.gif",
        n_frames=3,
        fps=6,
        shade="crosshatch",
        dpi=72,
        title="Water HOMO",
    )
    assert light_gif.is_file()
    with Image.open(light_gif) as img:
        assert img.format == "GIF"
        assert getattr(img, "n_frames", 1) == 3

    dark_gif = xm.animate_isosurface(
        "water_homo.cube",
        tmp_path / "water_dark.gif",
        n_frames=2,
        fps=6,
        shade="stipple",
        colors=("#56B4E9", "#F0A202"),
        alpha=0.38,
        show_atoms=False,
        preset="dark_chalk",
        dpi=72,
    )
    assert dark_gif.is_file()
    with Image.open(dark_gif) as img:
        assert getattr(img, "n_frames", 1) == 2
    assert plt.get_fignums() == []


def test_animate_isosurface_validation_and_missing_pillow(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validates error paths for animate_isosurface."""
    with pytest.raises(xm.StructureError, match="n_frames"):
        xm.animate_isosurface("water_homo.cube", tmp_path / "a.gif", n_frames=1)

    with pytest.raises(xm.StyleError, match="Unknown isosurface shade"):
        xm.animate_isosurface(
            "water_homo.cube", tmp_path / "b.gif", n_frames=2, shade="invalid"
        )

    with pytest.raises(xm.StructureError, match="non-zero"):
        xm.animate_isosurface(
            "water_homo.cube", tmp_path / "c.gif", n_frames=2, level=0.0
        )

    with pytest.raises(xm.StructureError, match="Expected VolumetricGrid"):
        xm.animate_isosurface("aspirin.xyz", tmp_path / "d.gif", n_frames=2)

    monkeypatch.setattr(util, "find_spec", lambda _name: None)
    with pytest.raises(xm.OptionalDependencyError, match="pillow"):
        xm.animate_isosurface("water_homo.cube", tmp_path / "e.gif", n_frames=2)


def test_animate_trajectory_path_and_ping_pong(tmp_path: pathlib.Path) -> None:
    """Supports filename coercion, ping-pong looping, and dark theme."""
    gif_path = xm.animate_trajectory(
        "sn2_neb.xyz",
        tmp_path / "sn2_pp.gif",
        fps=6,
        stride=2,
        ping_pong=True,
        show_cell=False,
        preset="dark_chalk",
        dpi=72,
        title="SN2",
    )
    assert gif_path.is_file()
    # 9 frames with stride=2 -> 5 unique frames -> ping-pong has 5 + 3 = 8
    with Image.open(gif_path) as img:
        assert img.format == "GIF"
        assert getattr(img, "n_frames", 1) == 8
    assert plt.get_fignums() == []
