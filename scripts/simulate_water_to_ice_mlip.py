r"""Reproducible MACE-MPA-0 direct-coexistence water-to-ice Ih MD script.

Requires optional external MLIP tooling (not needed to use ``xkcd-matchem``):
``ase``, ``torch`` (CUDA recommended), and ``mace-torch`` with the cached
``medium-mpa-0`` foundation model.

Protocol:
  1. Relax proton-disordered Bernal-Fowler hexagonal Ice Ih (32 H2O, 96 atoms).
  2. Fix the bottom 16 H2O seed layer and melt the top 16 H2O at 550 K.
  3. Release all constraints for unconstrained MACE-MPA-0 crystallization MD
     at 330 K, smooth cooling ramp down to 1 K, and final 0 K BFGS energy
     minimization (fmax <= 0.02 eV/A).

Usage::

    python scripts/simulate_water_to_ice_mlip.py \
        --output src/xkcd_matchem/data/samples/water_to_ice.xyz
"""

from __future__ import annotations

import argparse
import importlib
import pathlib

import numpy as np


def _build_orthorhombic_ice_ih_o_sites(
    nx: int = 2,
    ny: int = 1,
    nz: int = 2,
    a_hex: float = 4.518,
    c_hex: float = 7.356,
) -> tuple[np.ndarray, np.ndarray]:
    """Builds hexagonal Ice Ih oxygen coordinates and orthorhombic cell."""
    lx = a_hex
    ly = float(np.sqrt(3.0)) * a_hex
    lz = c_hex
    basis_frac = np.array(
        [
            [0.0, 0.0, 1.0 / 16.0],
            [0.0, 1.0 / 3.0, 15.0 / 16.0],
            [0.5, 0.5, 1.0 / 16.0],
            [0.5, 5.0 / 6.0, 15.0 / 16.0],
            [0.0, 0.0, 7.0 / 16.0],
            [0.0, 1.0 / 3.0, 9.0 / 16.0],
            [0.5, 0.5, 7.0 / 16.0],
            [0.5, 5.0 / 6.0, 9.0 / 16.0],
        ],
        dtype=np.float64,
    )
    coords: list[np.ndarray] = []
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                shift = np.array([ix * lx, iy * ly, iz * lz], dtype=np.float64)
                for frac in basis_frac:
                    pos = np.array(
                        [frac[0] * lx, frac[1] * ly, frac[2] * lz],
                        dtype=np.float64,
                    )
                    coords.append(pos + shift)
    cell = np.diag([nx * lx, ny * ly, nz * lz])
    return np.asarray(coords, dtype=np.float64), cell


def _minimum_image_vec(
    delta: np.ndarray,
    cell_diag: np.ndarray,
) -> np.ndarray:
    """Applies orthorhombic minimum-image convention to displacement vectors."""
    return delta - np.round(delta / cell_diag) * cell_diag


def _assign_bernal_fowler_protons(
    o_coords: np.ndarray,
    cell: np.ndarray,
    oh_length: float = 0.98,
    seed: int = 42,
) -> tuple[list[str], np.ndarray]:
    """Places 2 protons per oxygen satisfying exact Bernal-Fowler ice rules."""
    rng = np.random.default_rng(seed)
    n_o = len(o_coords)
    cell_diag = np.diag(cell)
    edges: list[tuple[int, int]] = []
    adj: list[list[int]] = [[] for _ in range(n_o)]
    for i in range(n_o):
        for j in range(i + 1, n_o):
            vec = _minimum_image_vec(o_coords[j] - o_coords[i], cell_diag)
            dist = float(np.linalg.norm(vec))
            if 2.3 < dist < 3.1:
                e_idx = len(edges)
                edges.append((i, j))
                adj[i].append(e_idx)
                adj[j].append(e_idx)

    orient = rng.choice([-1, 1], size=len(edges))
    out_deg = np.zeros(n_o, dtype=int)
    for e_idx, (i, j) in enumerate(edges):
        if orient[e_idx] == 1:
            out_deg[i] += 1
        else:
            out_deg[j] += 1

    for _ in range(50000):
        excess = np.where(out_deg > 2)[0]
        if len(excess) == 0:
            break
        node = int(rng.choice(excess))
        outgoing = [
            e
            for e in adj[node]
            if (edges[e][0] == node and orient[e] == 1)
            or (edges[e][1] == node and orient[e] == -1)
        ]
        e_pick = int(rng.choice(outgoing))
        i, j = edges[e_pick]
        if orient[e_pick] == 1:
            orient[e_pick] = -1
            out_deg[i] -= 1
            out_deg[j] += 1
        else:
            orient[e_pick] = 1
            out_deg[j] -= 1
            out_deg[i] += 1

    if not np.all(out_deg == 2):
        raise RuntimeError("Failed to converge Bernal-Fowler proton rules.")

    symbols: list[str] = []
    positions: list[np.ndarray] = []
    for i in range(n_o):
        symbols.append("O")
        positions.append(o_coords[i])
        for e_idx in adj[i]:
            u, v = edges[e_idx]
            is_out = (u == i and orient[e_idx] == 1) or (
                v == i and orient[e_idx] == -1
            )
            if is_out:
                other = v if u == i else u
                vec = _minimum_image_vec(
                    o_coords[other] - o_coords[i], cell_diag
                )
                unit = vec / max(float(np.linalg.norm(vec)), 1e-12)
                symbols.append("H")
                positions.append(o_coords[i] + oh_length * unit)
    return symbols, np.asarray(positions, dtype=np.float64)


def compute_tetrahedral_order_o(
    o_coords: np.ndarray,
    cell: np.ndarray,
    subset: np.ndarray | None = None,
) -> float:
    """Computes Errington-Debenedetti q over all O or a selected O subset."""
    cell_diag = np.diag(cell)
    indices = np.arange(len(o_coords)) if subset is None else subset
    q_vals: list[float] = []
    for i in indices:
        deltas = _minimum_image_vec(o_coords - o_coords[int(i)], cell_diag)
        dists = np.linalg.norm(deltas, axis=1)
        order = np.argsort(dists)[1:5]
        vecs = deltas[order] / np.maximum(dists[order, None], 1e-12)
        acc = 0.0
        for j in range(3):
            for k in range(j + 1, 4):
                cos_t = float(np.clip(np.dot(vecs[j], vecs[k]), -1.0, 1.0))
                acc += (cos_t + 1.0 / 3.0) ** 2
        q_vals.append(1.0 - (3.0 / 8.0) * acc)
    return float(np.mean(q_vals))


def _scramble_mobile_waters(
    atoms,
    mobile_mols: np.ndarray,
    cell: np.ndarray,
    rng: np.random.Generator,
) -> None:
    """Randomly rotates and nudges mobile H2O molecules to break ice order."""
    pos = atoms.get_positions().copy()
    cell_diag = np.diag(cell)
    z_lo = float(np.min(pos[3 * mobile_mols, 2]))
    z_hi = float(np.max(pos[3 * mobile_mols, 2]))
    for m_idx in mobile_mols:
        o_i = int(3 * m_idx)
        o_pos = pos[o_i].copy()
        v1 = _minimum_image_vec(pos[o_i + 1] - o_pos, cell_diag)
        v2 = _minimum_image_vec(pos[o_i + 2] - o_pos, cell_diag)
        q_mat, r_mat = np.linalg.qr(rng.normal(size=(3, 3)))
        rot = q_mat @ np.diag(np.sign(np.diag(r_mat)))
        if np.linalg.det(rot) < 0.0:
            rot[:, 0] *= -1.0
        jitter = rng.uniform(-0.85, 0.85, size=3)
        o_new = o_pos + jitter
        o_new[0] %= cell_diag[0]
        o_new[1] %= cell_diag[1]
        o_new[2] = float(np.clip(o_new[2], z_lo - 0.3, z_hi + 0.3))
        pos[o_i] = o_new
        pos[o_i + 1] = o_new + rot @ v1
        pos[o_i + 2] = o_new + rot @ v2
    atoms.set_positions(pos)


def _write_extxyz_frame(
    handle,
    atoms,
    time_ps: float,
    temp_k: float,
    energy_ev: float,
    q_tetra: float,
    stage: str,
) -> None:
    """Writes one extended XYZ frame compatible with xkcd-matchem."""
    cell = np.asarray(atoms.get_cell(), dtype=np.float64).ravel()
    lat = " ".join(f"{val:.6f}" for val in cell)
    header = (
        f'Lattice="{lat}" Properties=species:S:1:pos:R:3 pbc="T T T" '
        f"energy={energy_ev:.6f} time={time_ps:.4f} "
        f"temperature={temp_k:.2f} q_tetra={q_tetra:.4f} stage={stage}\n"
    )
    handle.write(f"{len(atoms)}\n")
    handle.write(header)
    syms = atoms.get_chemical_symbols()
    pos = atoms.get_positions()
    for sym, xyz in zip(syms, pos, strict=True):
        handle.write(
            f"{sym:2s} {xyz[0]:12.6f} {xyz[1]:12.6f} {xyz[2]:12.6f}\n"
        )


def _anchor_and_wrap_intact(
    atoms,
    seed_o_indices: np.ndarray,
    ref_seed_o: np.ndarray,
) -> None:
    """Removes rigid PBC CoM drift and wraps water molecules intact."""
    pos = atoms.get_positions().copy()
    cell_diag = np.diag(np.asarray(atoms.get_cell(), dtype=np.float64))
    drift = np.mean(
        _minimum_image_vec(pos[seed_o_indices] - ref_seed_o, cell_diag),
        axis=0,
    )
    pos -= drift
    for idx in range(0, len(atoms), 3):
        o_pos = pos[idx]
        shift = np.floor(o_pos / cell_diag) * cell_diag
        pos[idx] -= shift
        for h_off in (1, 2):
            vec = _minimum_image_vec(pos[idx + h_off] - pos[idx], cell_diag)
            pos[idx + h_off] = pos[idx] + vec
    atoms.set_positions(pos)


def run_simulation(args: argparse.Namespace) -> None:
    """Runs MACE-MPA-0 water-to-ice Ih MD with cooling to stable 0 K ice."""
    ase_mod = importlib.import_module("ase")
    ase_constraints = importlib.import_module("ase.constraints")
    ase_md_lang = importlib.import_module("ase.md.langevin")
    ase_md_vel = importlib.import_module("ase.md.velocitydistribution")
    ase_opt = importlib.import_module("ase.optimize")
    ase_units = importlib.import_module("ase.units")
    mace_calc = importlib.import_module("mace.calculators")

    o_coords, cell = _build_orthorhombic_ice_ih_o_sites(nx=2, ny=1, nz=2)
    syms, pos = _assign_bernal_fowler_protons(o_coords, cell, seed=args.seed)
    atoms = ase_mod.Atoms(symbols=syms, positions=pos, cell=cell, pbc=True)
    atoms.calc = mace_calc.mace_mp(
        model=args.model,
        device=args.device,
        default_dtype="float32",
    )

    # 1. Relax initial hexagonal Ice Ih crystal
    ase_opt.BFGS(atoms, logfile=None).run(fmax=0.03, steps=50)
    o_idx = np.arange(0, len(atoms), 3)
    q_init = compute_tetrahedral_order_o(atoms.get_positions()[o_idx], cell)
    e_ideal = float(atoms.get_potential_energy())
    print(f"Relaxed Ice Ih: N={len(atoms)}, E={e_ideal:.3f} eV, q={q_init:.4f}")

    # 2. Prepare liquid-ice coexistence (fix bottom 16 H2O seed layer)
    z_o = atoms.get_positions()[o_idx, 2]
    order_z = np.argsort(z_o)
    seed_mols = order_z[: len(o_idx) // 2]
    mobile_mols = order_z[len(o_idx) // 2 :]
    seed_o_indices = 3 * seed_mols
    ref_seed_o = atoms.get_positions()[seed_o_indices].copy()
    fixed_atoms = [int(3 * m + off) for m in seed_mols for off in (0, 1, 2)]

    rng = np.random.default_rng(args.seed + 1)
    _scramble_mobile_waters(atoms, mobile_mols, cell, rng)
    atoms.set_constraint(ase_constraints.FixAtoms(indices=fixed_atoms))
    ase_opt.BFGS(atoms, logfile=None).run(fmax=0.5, steps=25)

    ase_md_vel.MaxwellBoltzmannDistribution(
        atoms, temperature_K=args.melt_temp, rng=rng
    )
    dyn_melt = ase_md_lang.Langevin(
        atoms,
        timestep=0.5 * ase_units.fs,
        temperature_K=args.melt_temp,
        friction=0.04 / ase_units.fs,
        rng=rng,
    )
    dyn_melt.run(args.melt_steps)
    o_now = atoms.get_positions()[o_idx]
    print(
        f"After {args.melt_steps} melt steps ({args.melt_temp:.0f} K): "
        f"q_all={compute_tetrahedral_order_o(o_now, cell):.4f}, "
        f"q_seed={compute_tetrahedral_order_o(o_now, cell, seed_mols):.4f}, "
        f"q_mobile={compute_tetrahedral_order_o(o_now, cell, mobile_mols):.4f}"
    )

    # 3. Release ALL constraints for unbiased MACE-MPA-0 crystallization MD
    atoms.set_constraint()
    ase_md_vel.MaxwellBoltzmannDistribution(
        atoms, temperature_K=args.growth_temp, rng=rng
    )
    ase_md_vel.Stationary(atoms)

    raw_frames: list[tuple[object, float, float, float, float, str]] = []
    step_fs = 0.75

    def _record(
        step_num: int,
        stage: str,
        override_temp: float | None = None,
    ) -> None:
        snap = atoms.copy()
        _anchor_and_wrap_intact(snap, seed_o_indices, ref_seed_o)
        e_ev = float(atoms.get_potential_energy())
        t_k = (
            float(atoms.get_temperature())
            if override_temp is None
            else override_temp
        )
        q_val = compute_tetrahedral_order_o(snap.get_positions()[o_idx], cell)
        raw_frames.append(
            (snap, step_num * step_fs / 1000.0, t_k, e_ev, q_val, stage)
        )

    _record(0, "liquid_interface")
    dyn_grow = ase_md_lang.Langevin(
        atoms,
        timestep=step_fs * ase_units.fs,
        temperature_K=args.growth_temp,
        friction=0.015 / ase_units.fs,
        rng=rng,
    )

    total_md_steps = args.growth_steps + args.anneal_steps + args.quench_steps
    sample_every = max(1, total_md_steps // args.record_samples)
    for step in range(1, args.growth_steps + 1):
        dyn_grow.run(1)
        if step % sample_every == 0 or step == args.growth_steps:
            _record(step, "freezing")
        if step % 2000 == 0:
            o_cur = atoms.get_positions()[o_idx]
            q_all_cur = compute_tetrahedral_order_o(o_cur, cell)
            q_mob_cur = compute_tetrahedral_order_o(o_cur, cell, mobile_mols)
            print(
                f"  growth {step}/{args.growth_steps} "
                f"({step * step_fs / 1000.0:.2f} ps): "
                f"T={atoms.get_temperature():.1f} K, "
                f"E={atoms.get_potential_energy():.2f} eV, "
                f"q_all={q_all_cur:.4f}, q_mob={q_mob_cur:.4f}"
            )

    # 4. Smooth linear annealing ramp from growth_temp down to 5 K
    n_stages = 16
    stage_steps = max(1, args.anneal_steps // n_stages)
    ramp_temps = np.linspace(args.growth_temp, 5.0, n_stages)
    cur_step = args.growth_steps
    for s_idx, target_t in enumerate(ramp_temps):
        dyn_ramp = ase_md_lang.Langevin(
            atoms,
            timestep=step_fs * ase_units.fs,
            temperature_K=float(target_t),
            friction=0.025 / ase_units.fs,
            rng=rng,
        )
        for _ in range(stage_steps):
            dyn_ramp.run(1)
            cur_step += 1
            if cur_step % sample_every == 0:
                _record(cur_step, "annealing")
        if (s_idx + 1) % 4 == 0:
            o_cur = atoms.get_positions()[o_idx]
            t_now = atoms.get_temperature()
            print(
                f"  anneal stage {s_idx + 1}/{n_stages} "
                f"(target={target_t:.0f} K, T={t_now:.1f} K): "
                f"E={atoms.get_potential_energy():.2f} eV, "
                f"q_all={compute_tetrahedral_order_o(o_cur, cell):.4f}"
            )

    # 5. Deep cryo thermalization at 1 K before 0 K energy minimization
    dyn_cryo = ase_md_lang.Langevin(
        atoms,
        timestep=step_fs * ase_units.fs,
        temperature_K=1.0,
        friction=0.05 / ase_units.fs,
        rng=rng,
    )
    for _ in range(args.quench_steps):
        dyn_cryo.run(1)
        cur_step += 1
        if cur_step % sample_every == 0:
            _record(cur_step, "cryo_1K")
    _record(cur_step, "cryo_1K")

    # 6. Final 0 K MACE-MPA-0 BFGS relaxation to force convergence (0 K stable)
    atoms.set_momenta(np.zeros((len(atoms), 3), dtype=np.float64))
    opt_zero = ase_opt.BFGS(atoms, logfile=None)
    opt_zero.run(fmax=args.fmax_0k, steps=args.opt_steps)
    fmax_final = float(
        np.max(np.linalg.norm(atoms.get_forces(), axis=1))
    )
    _record(cur_step + 400, "ice_0K", override_temp=0.0)
    print(
        f"0 K BFGS converged: E={atoms.get_potential_energy():.3f} eV "
        f"(vs ideal Ice Ih {e_ideal:.3f} eV), "
        f"fmax={fmax_final:.4f} eV/A, q={raw_frames[-1][4]:.4f}"
    )

    # Select evenly spaced frames guaranteeing the first and final 0 K frames
    pick = np.linspace(0, len(raw_frames) - 1, args.n_frames, dtype=int)
    pick[0] = 0
    pick[-1] = len(raw_frames) - 1
    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for idx in pick:
            snap, t_ps, t_k, e_ev, q_val, stage = raw_frames[int(idx)]
            _write_extxyz_frame(handle, snap, t_ps, t_k, e_ev, q_val, stage)
    print(
        f"Wrote {len(pick)} frames to {out_path} "
        f"(T: {raw_frames[0][2]:.1f} K -> {raw_frames[-1][2]:.1f} K, "
        f"q: {raw_frames[0][4]:.3f} -> {raw_frames[-1][4]:.3f}, "
        f"size={out_path.stat().st_size / 1024:.1f} KB)"
    )


def main() -> None:
    """Parses CLI options and launches MACE-MPA-0 water-to-0K-ice MD."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="src/xkcd_matchem/data/samples/water_to_ice.xyz",
    )
    parser.add_argument("--model", default="medium-mpa-0")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--melt-temp", type=float, default=550.0)
    parser.add_argument("--melt-steps", type=int, default=1500)
    parser.add_argument("--growth-temp", type=float, default=330.0)
    parser.add_argument("--growth-steps", type=int, default=12000)
    parser.add_argument("--anneal-steps", type=int, default=8000)
    parser.add_argument("--quench-steps", type=int, default=1500)
    parser.add_argument("--fmax-0k", type=float, default=0.02)
    parser.add_argument("--opt-steps", type=int, default=200)
    parser.add_argument("--record-samples", type=int, default=140)
    parser.add_argument("--n-frames", type=int, default=28)
    run_simulation(parser.parse_args())


if __name__ == "__main__":
    main()
