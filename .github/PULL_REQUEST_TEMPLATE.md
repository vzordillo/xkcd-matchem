## Summary

<!-- Briefly explain what this pull request changes and why. -->

Fixes #<!-- issue number (if applicable) -->

## Type of Change

- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (new plotter, parser, layout template, or 3D Studio capability)
- [ ] Visual polish / SciViz design improvement
- [ ] Documentation, tutorial notebook, or gallery update
- [ ] Refactoring, performance optimization, or test coverage improvement

## Visual & SciViz Pillar Review (If Applicable)

<!-- Attach before/after screenshots for any plotting, rendering, layout, or 3D Studio changes. -->

- [ ] Follows the 4-tier stroke hierarchy (`xm.stroke_weights(preset)`) and ink-first typography
- [ ] Dense labels use outward-normal clearance and un-sketched paper halos (`place_halo_text`)
- [ ] Reference guides / zero lines / tick dividers stay straight (`sketch=(0, 0, 0)`)
- [ ] Regenerated affected figures via `python scripts/generate_gallery.py` and visually inspected them

## Scientific Rigor & Attribution Check

- [ ] Physical constants, equations, units, and sign conventions are cross-validated against primary sources
- [ ] Any new external data tables or citations are accurately credited in `book/reference/attributions.md`
- [ ] No version bump in `src/xkcd_matchem/_version.py` and no Git tags added

## Local Quality Gate Checklist

- [ ] `ruff check . --no-cache` passes (`<= 80` cols across `.py` and `.ipynb`)
- [ ] `pylint src/xkcd_matchem` scores `10.00/10`
- [ ] `mypy src/xkcd_matchem` passes cleanly
- [ ] `pytest --cov=xkcd_matchem --cov-fail-under=90` passes with all Matplotlib figures closed
- [ ] `jupyter-book build book/ -W` builds with zero warnings
- [ ] Contributions comply with Apache-2.0 (code) and CC BY 4.0 (docs/tutorials) licensing
