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
"""Materials-informatics plots inspired by pymatviz in xkcd style."""

from __future__ import annotations

from xkcd_matchem.plots import _materials_bz as mat_bz
from xkcd_matchem.plots import _materials_stats as mat_stats
from xkcd_matchem.plots import _materials_ternary as mat_tern
from xkcd_matchem.plots import _materials_wulff as mat_wulff


__all__ = [
    "plot_brillouin_zone",
    "plot_histogram",
    "plot_spacegroup_distribution",
    "plot_ternary_phase_diagram",
    "plot_wulff_shape",
    "spacegroup_to_crystal_system",
]

# Public plotting and utility API.
plot_brillouin_zone = mat_bz.plot_brillouin_zone
plot_histogram = mat_stats.plot_histogram
plot_spacegroup_distribution = mat_stats.plot_spacegroup_distribution
plot_ternary_phase_diagram = mat_tern.plot_ternary_phase_diagram
plot_wulff_shape = mat_wulff.plot_wulff_shape
spacegroup_to_crystal_system = mat_stats.spacegroup_to_crystal_system

# Backwards-compatible internal symbol re-exports.
_CRYSTAL_SYSTEM_RANGES = mat_stats.CRYSTAL_SYSTEM_RANGES
_SQRT3_HALF = mat_tern.SQRT3_HALF

_barycentric_to_xy = mat_tern.barycentric_to_xy
_collinear_lower_hull = mat_tern.collinear_lower_hull
_ternary_lower_hull = mat_tern.ternary_lower_hull
_draw_ternary_frame = mat_tern.draw_ternary_frame
_annotate_ternary_labels = mat_tern.annotate_ternary_labels

_aggregate_spacegroups = mat_stats.aggregate_spacegroups
_draw_sunburst_rings = mat_stats.draw_sunburst_rings
_draw_hist_series = mat_stats.draw_hist_series

_coerce_wulff_cell = mat_wulff.coerce_wulff_cell
_format_hkl_label = mat_wulff.format_hkl_label
_lambert_shade_rgb = mat_wulff.lambert_shade_rgb
_facet_clearance = mat_wulff.facet_clearance
_collect_wulff_facet_visuals = mat_wulff.collect_wulff_facet_visuals
_project_wulff_facets = mat_wulff.project_wulff_facets
_draw_wulff_labels = mat_wulff.draw_wulff_labels

_draw_bz_corner_axes = mat_bz.draw_bz_corner_axes
_kpoint_label_offset = mat_bz.kpoint_label_offset
_draw_bz_kpath_and_axes = mat_bz.draw_bz_kpath_and_axes
_project_bz_facets = mat_bz.project_bz_facets
