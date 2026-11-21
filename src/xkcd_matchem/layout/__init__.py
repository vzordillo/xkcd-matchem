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
"""Multi-panel layout composition, parchment scrolls, and figure export."""

from __future__ import annotations

from xkcd_matchem.layout import canvas
from xkcd_matchem.layout import export
from xkcd_matchem.layout import scroll


__all__ = [
    "Badge",
    "Board",
    "Card",
    "Container",
    "ScrollCard",
    "mm_to_inches",
    "plot_scroll_card",
    "render_scroll_list",
    "save_figure",
]

Badge = canvas.Badge
Board = canvas.Board
Card = canvas.Card
Container = canvas.Container
ScrollCard = scroll.ScrollCard
mm_to_inches = canvas.mm_to_inches
plot_scroll_card = scroll.plot_scroll_card
render_scroll_list = scroll.render_scroll_list
save_figure = export.save_figure
