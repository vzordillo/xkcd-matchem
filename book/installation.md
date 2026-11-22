# Installation & Optional Extras

## Core Package

`xkcd-matchem` has a light core footprint: **Python 3.10+**, **NumPy**, **SciPy**, and **Matplotlib**. All **17 bundled sample files** (molecules, crystals, slabs, DFT outputs, LOBSTER COHP tables, Phonopy bands, spectra, orbital cubes, and MD/NEB trajectories) and bundled comic-style fonts—`xkcd Script` (default, CC BY-NC 3.0 for non-commercial/academic use) plus commercial-safe SIL OFL 1.1 alternates (`Patrick Hand` and `Comic Neue`)—ship inside the wheel.

Install directly from GitHub:

```bash
pip install git+https://github.com/vzordillo/xkcd-matchem.git
```

:::{note}
**PyPI Release Note:** Standard PyPI installation (`pip install xkcd-matchem`) will be enabled with the first tagged PyPI release. Until then, install from GitHub using the `git+https://...` commands on this page or from a local editable checkout below.
:::

### Quick Verification

Once installed, check the CLI, inspect the bundled examples and built-in file parsers, and render a quick verification figure:

```bash
xkcd-matchem --version
xkcd-matchem samples
xkcd-matchem formats
xkcd-matchem render si_diamond.cif -o si_check.png
```

You should see `xkcd-matchem 0.1.0`, all **17 sample datasets** (alphabetically from `alaninedipeptide.pdb` to `water_rdf.xvg`), all **29 zero-dependency native formats**, and a rendered `si_check.png` crystal figure in your working directory.

## Optional Ecosystem Adapters

Already using ASE, pymatgen, RDKit, cclib, or MDAnalysis? Install only the bridges you need — none are required for the core library:

| Extra Key | Installs | Install this if you want to... |
| --- | --- | --- |
| `ase` | `ase>=3.22` | Convert ASE `Atoms` (`xm.from_ase`, `xm.to_ase`) or fall back to `xm.io.read_via_ase()` for extra formats |
| `pymatgen` | `pymatgen>=2024.1` | Convert `Structure` / `Molecule` objects via `xm.from_pymatgen` and `xm.to_pymatgen` |
| `rdkit` | `rdkit>=2023.9` | Generate 3D conformers directly from SMILES strings (`xm.from_smiles`, `xm.from_rdkit`) |
| `qm` | `cclib>=1.8` | Parse quantum chemistry logfiles beyond native Gaussian/ORCA subsets (`xm.from_cclib`, `xm.io.read_via_cclib`) |
| `md` | `MDAnalysis>=2.6`, `mdtraj>=1.9` | Import MDAnalysis `Universe` or MDTraj `Trajectory` objects (`xm.from_mdanalysis`, `xm.from_mdtraj`) |
| `animation` | `pillow>=10.0` | Export looping trajectory animations as GIFs via `xm.animate_trajectory()` |

For example, to add ASE, pymatgen, RDKit, and GIF animation support together from GitHub:

```bash
pip install "xkcd-matchem[ase,pymatgen,rdkit,animation] @ git+https://github.com/vzordillo/xkcd-matchem.git"
```

:::{tip}
Want to run the tutorials in your browser with zero local setup? Launch the repository directly on **[Binder](https://mybinder.org/v2/gh/vzordillo/xkcd-matchem/main?urlpath=lab%2Ftree%2Fbook%2Ftutorials%2F01_quickstart_and_themes.ipynb)**, or explore structures immediately in the zero-dependency **[Interactive 3D Studio](playground.md)** — no WebGL plugin, JavaScript build step, or remote server needed.
:::

## Contributor & Documentation Setup

If you want to run the full quality gate, tests, or build this Jupyter Book locally (see [`CONTRIBUTING.md`](https://github.com/vzordillo/xkcd-matchem/blob/main/CONTRIBUTING.md) for SciViz design guidelines and PR checklists):

```bash
git clone https://github.com/vzordillo/xkcd-matchem.git
cd xkcd-matchem
python -m venv .venv
source .venv/bin/activate
pip install -e ".[animation,dev,docs]"
ruff check . --no-cache
pylint src/xkcd_matchem
mypy src/xkcd_matchem
pytest --cov=xkcd_matchem --cov-fail-under=90
jupyter-book build book/ -W
```
