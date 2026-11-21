# Installation & Optional Extras

## Core Package

`xkcd-matchem` has a light core footprint: **Python 3.10+**, **NumPy**, **SciPy**, and **Matplotlib**. All **15 bundled sample files** (molecules, crystals, slabs, DFT outputs, spectra, and trajectories) and bundled comic-style fonts—`xkcd Script` (default, CC BY-NC 3.0 for non-commercial/academic use) plus commercial-safe SIL OFL 1.1 alternates (`Patrick Hand` and `Comic Neue`)—ship inside the wheel.

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

You should see all 15 sample datasets (from `aspirin.xyz` to `tio2_slab.poscar`) and all 26 zero-dependency native formats listed cleanly.

## Optional Ecosystem Adapters

Already using ASE, pymatgen, RDKit, cclib, or MDAnalysis? Install only the bridges you need — none are required for the core library:

| Extra Key | Installs | Install this if you want to... |
| --- | --- | --- |
| `ase` | `ase>=3.22` | Convert ASE `Atoms` (`from_ase`, `to_ase`) or fall back to `read_via_ase()` for extra formats |
| `pymatgen` | `pymatgen>=2024.1` | Convert `Structure` / `Molecule` objects via `from_pymatgen` and `to_pymatgen` |
| `rdkit` | `rdkit>=2023.9` | Generate 3D conformers directly from SMILES strings (`from_smiles`, `from_rdkit`) |
| `qm` | `cclib>=1.8` | Parse quantum chemistry logfiles beyond native Gaussian/ORCA subsets (`from_cclib`, `read_via_cclib`) |
| `md` | `MDAnalysis`, `mdtraj` | Import MDAnalysis `Universe` or MDTraj `Trajectory` objects (`from_mdanalysis`, `from_mdtraj`) |
| `animation` | `pillow>=10.0` | Export looping trajectory animations as GIFs via `animate_trajectory()` |

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
jupyter-book build --all book/
```
