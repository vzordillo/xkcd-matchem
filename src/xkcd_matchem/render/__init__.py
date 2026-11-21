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
"""2D and 3D molecular, crystallographic, icon, and annotation renderers."""

from __future__ import annotations

from xkcd_matchem.render import callouts
from xkcd_matchem.render import icons
from xkcd_matchem.render import interactive
from xkcd_matchem.render import primitives
from xkcd_matchem.render import structure2d
from xkcd_matchem.render import structure3d


__all__ = [
    "InteractiveView",
    "annotate_angle",
    "annotate_distance",
    "build_interactive_html",
    "draw_atom_sphere",
    "draw_callout",
    "draw_curved_arrow",
    "draw_element_badge",
    "draw_element_legend",
    "draw_icon",
    "draw_miller_plane",
    "draw_polyhedron_faces",
    "draw_two_tone_bond",
    "draw_unit_cell_box",
    "export_interactive_html",
    "list_icons",
    "plot_icon_sheet",
    "render_structure_2d",
    "render_structure_3d",
    "serialize_structure",
    "view_structure",
]

InteractiveView = interactive.InteractiveView
annotate_angle = callouts.annotate_angle
annotate_distance = callouts.annotate_distance
build_interactive_html = interactive.build_interactive_html
draw_atom_sphere = primitives.draw_atom_sphere
draw_callout = callouts.draw_callout
draw_curved_arrow = callouts.draw_curved_arrow
draw_element_badge = icons.draw_element_badge
draw_element_legend = callouts.draw_element_legend
draw_icon = icons.draw_icon
draw_miller_plane = primitives.draw_miller_plane
draw_polyhedron_faces = primitives.draw_polyhedron_faces
draw_two_tone_bond = primitives.draw_two_tone_bond
draw_unit_cell_box = primitives.draw_unit_cell_box
export_interactive_html = interactive.export_interactive_html
list_icons = icons.list_icons
plot_icon_sheet = icons.plot_icon_sheet
render_structure_2d = structure2d.render_structure_2d
render_structure_3d = structure3d.render_structure_3d
serialize_structure = interactive.serialize_structure
view_structure = interactive.view_structure
