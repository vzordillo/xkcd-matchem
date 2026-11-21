# Installation & Optional Extras

## Core Package

`xkcd-matchem` requires only **Python 3.10+**, **NumPy**, **SciPy**, and **Matplotlib**. All 25+ bundled scientific sample files and SIL OFL 1.1 handwritten fonts (`Patrick Hand` and `Comic Neue`) ship inside the wheel.

```bash
pip install xkcd-matchem
```

Verify your installation and inspect bundled samples:

```bash
xkcd-matchem --version
xkcd-matchem samples
xkcd-matchem formats
```

## Optional Ecosystem Adapters

Install only the chemistry/materials frameworks you already use:

| Extra Key | Installs | Enables |
| --- | --- | --- |
| `ase` | `ase>=3.22` | `from_ase`, `to_ase`, `read_via_ase` |
| `pymatgen` | `pymatgen>=2024.1` | `from_pymatgen`, `to_pymatgen` |
| `rdkit` | `rdkit>=2023.9` | `from_rdkit`, `from_smiles` |
| `qm` | `cclib>=1.8` | `from_cclib`, `read_via_cclib` |
| `md` | `MDAnalysis`, `mdtraj` | `from_mdanalysis`, `from_mdtraj` |
| `animation` | `pillow>=10.0` | `animate_trajectory` GIF/MP4 export |

Example:

```bash
pip install "xkcd-matchem[ase,pymatgen,rdkit,animation]"
```

## Source / Documentation Development Setup

```bash
git clone https://github.com/vzordillo/xkcd-matchem.git
cd xkcd-matchem
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,docs]"
pytest tests/
jupyter-book build --all book/
```

:::{tip}
Offline HTML5/Canvas 3D viewing via `xm.view_structure(...)` and `xkcd-matchem view ...` works with **zero extra dependencies** — no WebGL plugin or remote server required!
:::
