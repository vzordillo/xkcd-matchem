# Third-Party Licenses and Attributions

`xkcd-matchem` uses clean, compatible open licenses across its software,
documentation, bundled fonts, and scientific data tables:

| Component | Path | License | Notes |
| --- | --- | --- | --- |
| Python library, tests, scripts | `src/`, `tests/`, `scripts/` | Apache-2.0 | Permissive with explicit patent grant & attribution (`LICENSE`, `NOTICE`) |
| Tutorials, docs, gallery figures | `book/` | CC BY 4.0 | Open academic reuse with attribution (`book/LICENSE-CC-BY-4.0.txt`) |
| Patrick Hand font | `src/xkcd_matchem/data/fonts/PatrickHand-*` | SIL OFL 1.1 | Bundled default hand-drawn font by Patrick Wagesreiter |
| Comic Neue font | `src/xkcd_matchem/data/fonts/ComicNeue-*` | SIL OFL 1.1 | Bundled alternate font by Craig Rozynski & Hrant Papazian |
| Blue Obelisk Data Repository (BODR) | Numeric values in `src/xkcd_matchem/data/elements.py` | CC0 1.0 | Factual element constants (Guha et al., 2006) |
| CODATA 2022, IUPAC/CIAAW, Cromer-Mann LA-3816, Alvarez, Mantina, PubChem | Numeric values in `src/xkcd_matchem/data/` | Factual data + academic citation | Compiled directly from primary publications & NIST/CIAAW tables; see `NOTICE` |
| Okabe-Ito CUD & Jmol Colors | Hex triplets in `src/xkcd_matchem/style/palettes.py` | Open reference palettes | See `NOTICE` |
| Optional `xkcd-script` font (NOT bundled) | Downloaded only via opt-in `fetch_xkcd_script()` | CC BY-NC 3.0 | Non-commercial font excluded from distribution; explicit notice printed on download |

## Runtime Dependencies

Core dependencies (`numpy`, `scipy`, `matplotlib`) and optional adapters (`ase`,
`pymatgen`, `cclib`, `rdkit`, `MDAnalysis`, `mdtraj`, `pillow`) are installed
separately by the user under their respective BSD/MIT/LGPL/MPL licenses and are
imported dynamically only when requested.
