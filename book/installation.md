# Installation & Optional Extras

## Core Package

`xkcd-matchem` has a light core footprint: **Python 3.10+**, **NumPy**, **SciPy**, and **Matplotlib**. All **17 bundled sample files** (molecules, crystals, slabs, DFT outputs, LOBSTER COHP tables, Phonopy bands, spectra, orbital cubes, and MD/NEB trajectories) and bundled comic-style fonts—`xkcd Script` (default, CC BY-NC 3.0 for non-commercial/academic use) plus commercial-safe SIL OFL 1.1 alternates (`Patrick Hand` and `Comic Neue`)—ship inside the wheel.

```bash
pip install xkcd-matchem
```

### Quick Verification

Once installed, check the CLI and inspect the bundled examples and built-in file parsers:

```bash
xkcd-matchem --version
xkcd-matchem samples
xkcd-matchem formats
```

You should see all **17 sample datasets** (alphabetically from `alaninedipeptide.pdb` to `water_rdf.xvg`) and all **29 zero-dependency native formats** listed cleanly.

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

For example, to add ASE, pymatgen, RDKit, and GIF animation support together:

```bash
pip install "xkcd-matchem[ase,pymatgen,rdkit,animation]"
```

:::{tip}
The browser-based **Interactive 3D Studio** (`xm.view_structure(...)` and `xkcd-matchem view ...`) works with **zero extra dependencies** — no WebGL plugin, JavaScript build step, or remote server needed.
:::

## Contributor & Documentation Setup

If you want to run the test suite, lint checks, or build this Jupyter Book locally:

```bash
git clone https://github.com/vzordillo/xkcd-matchem.git
cd xkcd-matchem
python -m venv .venv
source .venv/bin/activate
pip install -e ".[animation,dev,docs]"
pytest tests/
jupyter-book build book/ -W
```
