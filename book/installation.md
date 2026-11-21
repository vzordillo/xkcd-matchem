# Installation & Setup

## Core Install

Install the lightweight core (`numpy`, `matplotlib`, `scipy` + bundled sample data and fonts):

```bash
pip install xkcd-matchem
```

## Optional Ecosystem Extras

`xkcd-matchem` lazily imports optional simulation frameworks only when you call an adapter (`from_ase`, `from_pymatgen`, `from_cclib`, `from_smiles`, `from_mdanalysis`, `from_mdtraj`, or `animate_trajectory`):

```bash
pip install "xkcd-matchem[ase,pymatgen,qm,rdkit,md,animation]"
```

| Extra | Underlying Package | Capabilities Added |
| --- | --- | --- |
| `[ase]` | `ase` | Read/write any ASE format (`to_ase`, `from_ase`, `read_via_ase`) |
| `[pymatgen]` | `pymatgen` | Interoperate with `Structure` / `Molecule` (`from_pymatgen`, `to_pymatgen`) |
| `[qm]` | `cclib` | Parse additional QM logfiles via `cclib` (`from_cclib`, `read_via_cclib`) |
| `[rdkit]` | `rdkit` | Generate 3D conformers directly from SMILES (`from_smiles`, `from_rdkit`) |
| `[md]` | `MDAnalysis`, `mdtraj` | Load GROMACS/Amber/CHARMM trajectories (`from_mdanalysis`, `from_mdtraj`) |
| `[animation]` | `pillow` | Export animated xkcd GIF trajectories (`animate_trajectory`) |
| `[all]` | All of the above + `docs` & `dev` | Complete developer and documentation toolkit |

## Optional `xkcd-script` Font

For licensing clarity (`Apache-2.0` code / `SIL OFL 1.1` bundled fonts), the non-commercial `xkcd-script` typeface (`CC BY-NC 3.0`) is **not bundled**. If your non-commercial work requires it, download it into your user cache explicitly:

```python
import xkcd_matchem as xm

xm.fetch_xkcd_script()
```
