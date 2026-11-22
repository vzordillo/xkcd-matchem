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


def test_water_to_ice_hbonds_and_tetrahedral_order(
    tmp_path: pathlib.Path,
) -> None:
    """Verifies MLIP water_to_ice.xyz metadata, H-bonds, q(t), and rendering."""
    traj = xm.load_trajectory("water_to_ice.xyz")
    assert len(traj) == 28
    assert traj.times is not None
    assert len(traj.times) == 28
    assert traj.temperatures is not None
    assert len(traj.temperatures) == 28
    assert float(traj.temperatures[-1]) == 0.0
    assert traj.energies is not None
    assert len(traj.energies) == 28

    q_vals = traj.tetrahedral_order("O")
    assert q_vals.shape == (28,)
    assert float(q_vals[-1]) > float(q_vals[0])
    assert 0.95 <= float(q_vals[-1]) <= 1.0

    hb_start = xm.infer_hbonds(traj[0])
    hb_end = xm.infer_hbonds(traj[-1])
    assert len(hb_end) > len(hb_start)
    assert len(hb_end) >= 55
    for hb in hb_end:
        assert hb.distance_ha <= 2.35
        assert hb.distance_da <= 3.35
        assert hb.angle_deg >= 135.0

    # Render static & animated water-to-ice with show_hbonds=True and custom hex
    ax1 = xm.plot_crystal(
        traj[-1], show_cell="clean", show_hbonds=True, boundary_atoms=False
    )
    assert ax1 is not None
    plt.close(ax1.figure)

    ax2 = xm.plot_molecule("alaninedipeptide.pdb", show_hbonds="#56B4E9")
    assert ax2 is not None
    plt.close(ax2.figure)

    gif_ice = xm.animate_trajectory(
        "water_to_ice.xyz",
        tmp_path / "water_ice_hb.gif",
        fps=6,
        stride=10,
        show_cell="clean",
        show_hbonds=True,
        dpi=72,
    )
    assert gif_ice.is_file()
    with Image.open(gif_ice) as img:
        assert getattr(img, "n_frames", 1) == 3

    # Edge-case guards for infer_hbonds & compute_tetrahedral_order
    tiny = xm.AtomsData(symbols=["O", "H"], positions=np.zeros((2, 3)))
    assert xm.infer_hbonds(tiny) == []
    with pytest.raises(xm.StructureError, match="at least 5"):
        xm.compute_tetrahedral_order(tiny, center_symbol="O")
    assert plt.get_fignums() == []
