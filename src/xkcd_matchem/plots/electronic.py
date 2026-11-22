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
"""Electronic structure visualizations (bands, DOS, MOs, volumetric slices)."""

from __future__ import annotations

from xkcd_matchem.plots import _electronic_bands_dos as elec_bd
from xkcd_matchem.plots import _electronic_cohp_phonons as elec_cp
from xkcd_matchem.plots import _electronic_iso as elec_iso
from xkcd_matchem.plots import _electronic_mo as elec_mo


__all__ = [
    "plot_bands",
    "plot_bands_dos",
    "plot_cohp",
    "plot_dos",
    "plot_isosurface",
    "plot_mo_diagram",
    "plot_phonon_bands",
    "plot_volumetric_slice",
]

# Public plotting API.
plot_bands = elec_bd.plot_bands
plot_bands_dos = elec_bd.plot_bands_dos
plot_cohp = elec_cp.plot_cohp
plot_dos = elec_bd.plot_dos
plot_isosurface = elec_iso.plot_isosurface
plot_mo_diagram = elec_mo.plot_mo_diagram
plot_phonon_bands = elec_cp.plot_phonon_bands
plot_volumetric_slice = elec_iso.plot_volumetric_slice

# Backwards-compatible internal symbol re-exports.
_split_kpath_breaks = elec_bd.split_kpath_breaks
_format_k_labels = elec_bd.format_k_labels
_resolve_fatband_weights = elec_bd.resolve_fatband_weights
_draw_band_curves = elec_bd.draw_band_curves
_dos_window_limits = elec_bd.dos_window_limits
_draw_dos_curves = elec_bd.draw_dos_curves
_configure_dos_axes = elec_bd.configure_dos_axes

_draw_mo_group = elec_mo.draw_mo_group
_draw_cohp_series = elec_cp.draw_cohp_series
_configure_cohp_axes = elec_cp.configure_cohp_axes
_resolve_cohp_inputs = elec_cp.resolve_cohp_inputs

_draw_slice_atoms = elec_iso.draw_slice_atoms
_coerce_volumetric_grid = elec_iso.coerce_volumetric_grid
_expand_isosurface_limits = elec_iso.expand_isosurface_limits
_extract_isosurface_meshes = elec_iso.extract_isosurface_meshes
_draw_iso_art_batches = elec_iso.draw_iso_art_batches
