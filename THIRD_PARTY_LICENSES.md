# Third-Party Licenses and Attributions

`xkcd-matchem` uses clean, compatible open licenses across its software,
documentation, bundled fonts, and scientific data tables:

| Component | Path | License | Notes |
| --- | --- | --- | --- |
| Python library, tests, scripts | `src/`, `tests/`, `scripts/` | Apache-2.0 | Permissive with explicit patent grant & attribution (`LICENSE`, `NOTICE`) |
| Tutorials & documentation text | `book/` | CC BY 4.0 | Open academic and commercial reuse with attribution (`book/LICENSE-CC-BY-4.0.txt`); showcase figures in `book/_static/gallery/` are rendered with opt-in `xkcd Script` (CC BY-NC 3.0) |
| Patrick Hand font (default) | `src/xkcd_matchem/data/fonts/PatrickHand-*` | SIL OFL 1.1 | Bundled commercial-safe default font across all presets by Patrick Wagesreiter |
| Comic Neue font | `src/xkcd_matchem/data/fonts/ComicNeue-*` | SIL OFL 1.1 | Bundled commercial-safe alternate font by Craig Rozynski & Hrant Papazian |
| `xkcd Script` font (optional opt-in) | `src/xkcd_matchem/data/fonts/xkcd-script*` | CC BY-NC 3.0 | Bundled non-commercial opt-in font from `ipython/xkcd-font` (Randall Munroe); activated only via `xm.use_xkcd_script()` and blocked when `XM_COMMERCIAL_SAFE=1` (`xkcd-script-LICENSE.txt`) |
| Blue Obelisk Data Repository (BODR) | Numeric values in `src/xkcd_matchem/data/elements.py` | CC0 1.0 | Factual element constants (Guha et al., 2006) |
| CODATA 2022, IUPAC/CIAAW, Cromer-Mann LA-3816, Alvarez, Mantina, PubChem | Numeric values in `src/xkcd_matchem/data/` | Factual data + academic citation | Compiled directly from primary publications & NIST/CIAAW tables; see `NOTICE` |
| Okabe-Ito CUD & Jmol Colors | Hex triplets in `src/xkcd_matchem/style/palettes.py` | Open reference palettes | See `NOTICE` |

## Runtime Dependencies

Core dependencies (`numpy`, `scipy`, `matplotlib`) and optional adapters (`ase`,
`pymatgen`, `cclib`, `rdkit`, `MDAnalysis`, `mdtraj`, `pillow`) are installed
separately by the user under their respective BSD/MIT/LGPL/MPL licenses and are
imported dynamically only when requested.
