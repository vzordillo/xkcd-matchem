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
from xkcd_matchem.layout import comic
from xkcd_matchem.layout import export
from xkcd_matchem.layout import scroll
from xkcd_matchem.layout import templates


__all__ = [
    "Badge",
    "Board",
    "Card",
    "ComicPanel",
    "ComicStrip",
    "ConferencePosterLayout",
    "Container",
    "GraphicalAbstractLayout",
    "MechanismComicLayout",
    "ScrollCard",
    "TalkSlideLayout",
    "create_conference_poster",
    "create_graphical_abstract",
    "create_mechanism_comic",
    "create_talk_slide",
    "draw_speech_bubble",
    "draw_stick_scientist",
    "mm_to_inches",
    "plot_comic_strip",
    "plot_scroll_card",
    "render_scroll_list",
    "save_figure",
]

Badge = canvas.Badge
Board = canvas.Board
Card = canvas.Card
ComicPanel = comic.ComicPanel
ComicStrip = comic.ComicStrip
ConferencePosterLayout = templates.ConferencePosterLayout
Container = canvas.Container
GraphicalAbstractLayout = templates.GraphicalAbstractLayout
MechanismComicLayout = templates.MechanismComicLayout
ScrollCard = scroll.ScrollCard
TalkSlideLayout = templates.TalkSlideLayout
create_conference_poster = templates.create_conference_poster
create_graphical_abstract = templates.create_graphical_abstract
create_mechanism_comic = templates.create_mechanism_comic
create_talk_slide = templates.create_talk_slide
draw_speech_bubble = comic.draw_speech_bubble
draw_stick_scientist = comic.draw_stick_scientist
mm_to_inches = canvas.mm_to_inches
plot_comic_strip = comic.plot_comic_strip
plot_scroll_card = scroll.plot_scroll_card
render_scroll_list = scroll.render_scroll_list
save_figure = export.save_figure
